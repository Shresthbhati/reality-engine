"""Pinhole camera model: intrinsics, extrinsics, projection, unprojection.

Neutral camera types shared between reconstruction and perception.
Lives in WorldIR (the contract layer) so that perception modules do not
import reconstruction internals, breaking the layer rule.

CameraExtrinsics.rotation is CAMERA-TO-WORLD (rotating a direction
expressed in the camera's local frame into world space);
position is the camera's position in world space.
World-to-camera uses the orientation's conjugate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from engine.core.rigid_transform import CalibrationTransformError, RigidTransform  # noqa: F401  (canonical home: core layer)
from engine.math import Quat, Vec3

#: Fixed-point undistortion iterations.
_UNDISTORT_ITERATIONS = 10

#: A point this close to (or behind) the camera plane has no well-defined
#: projection -- reported as None, never as a fabricated pixel coordinate.
_MIN_PROJECTABLE_DEPTH_M = 1e-6


class CameraIntrinsicsError(ValueError):
    pass


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics + Brown-Conrady distortion coefficients, all in
    pixel units except the (dimensionless) distortion coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def __post_init__(self):
        values = (self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2, self.k3)
        if any(not math.isfinite(v) for v in values):
            raise CameraIntrinsicsError(f"non-finite intrinsics/distortion value(s): {values}")
        if self.fx <= 0 or self.fy <= 0:
            raise CameraIntrinsicsError(f"focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise CameraIntrinsicsError(f"image dimensions must be positive, got {self.width}x{self.height}")

    def to_dict(self) -> dict:
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "k1": self.k1, "k2": self.k2, "p1": self.p1, "p2": self.p2, "k3": self.k3,
        }

    @staticmethod
    def from_dict(data: dict) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=data["fx"], fy=data["fy"], cx=data["cx"], cy=data["cy"],
            width=data["width"], height=data["height"],
            k1=data.get("k1", 0.0), k2=data.get("k2", 0.0),
            p1=data.get("p1", 0.0), p2=data.get("p2", 0.0), k3=data.get("k3", 0.0),
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    """Camera placement in world space. `rotation` is camera-to-world
    (see module docstring) -- matches
    `reconstruction.backend.interface.ReconstructedCameraPose`."""

    position: Vec3
    rotation: Quat

    def world_to_camera(self, point_world: Vec3) -> Vec3:
        relative = point_world - self.position
        return self.rotation.conjugate().rotate(relative)

    def camera_to_world(self, point_camera: Vec3) -> Vec3:
        return self.rotation.rotate(point_camera) + self.position

    def to_dict(self) -> dict:
        return {"position": self.position.to_dict(), "rotation": self.rotation.to_dict()}

    @staticmethod
    def from_dict(data: dict) -> "CameraExtrinsics":
        return CameraExtrinsics(
            position=Vec3.from_dict(data["position"]), rotation=Quat.from_dict(data["rotation"]),
        )


@dataclass(frozen=True)
class PinholeCamera:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics

    def project(self, point_world: Vec3) -> Optional[Tuple[float, float]]:
        """World point -> distorted pixel (u, v), or None when the point
        is behind (or on) the camera plane -- never a fabricated pixel
        for an unprojectable point."""
        p_cam = self.extrinsics.world_to_camera(point_world)
        if p_cam.z <= _MIN_PROJECTABLE_DEPTH_M:
            return None
        x = p_cam.x / p_cam.z
        y = p_cam.y / p_cam.z
        x_d, y_d = _distort(x, y, self.intrinsics)
        u = self.intrinsics.fx * x_d + self.intrinsics.cx
        v = self.intrinsics.fy * y_d + self.intrinsics.cy
        return (u, v)

    def unproject(self, u: float, v: float, depth: float) -> Vec3:
        """Distorted pixel (u, v) + metric depth -> world point."""
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        x_d = (u - self.intrinsics.cx) / self.intrinsics.fx
        y_d = (v - self.intrinsics.cy) / self.intrinsics.fy
        x, y = _undistort(x_d, y_d, self.intrinsics)
        p_cam = Vec3(x * depth, y * depth, depth)
        return self.extrinsics.camera_to_world(p_cam)

    def ray(self, u: float, v: float) -> Tuple[Vec3, Vec3]:
        """World-space (origin, normalized direction) for the ray through
        pixel (u, v)."""
        origin = self.extrinsics.position
        target = self.unproject(u, v, depth=1.0)
        return origin, (target - origin).normalized()

    def to_dict(self) -> dict:
        return {"intrinsics": self.intrinsics.to_dict(), "extrinsics": self.extrinsics.to_dict()}

    @staticmethod
    def from_dict(data: dict) -> "PinholeCamera":
        return PinholeCamera(
            intrinsics=CameraIntrinsics.from_dict(data["intrinsics"]),
            extrinsics=CameraExtrinsics.from_dict(data["extrinsics"]),
        )


def _distort(x: float, y: float, k: CameraIntrinsics) -> Tuple[float, float]:
    """Ideal (undistorted) normalized camera-plane coords -> distorted
    normalized coords (Brown-Conrady forward model)."""
    r2 = x * x + y * y
    radial = 1.0 + k.k1 * r2 + k.k2 * r2 ** 2 + k.k3 * r2 ** 3
    x_d = x * radial + 2.0 * k.p1 * x * y + k.p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + k.p1 * (r2 + 2.0 * y * y) + 2.0 * k.p2 * x * y
    return x_d, y_d


def _undistort(x_d: float, y_d: float, k: CameraIntrinsics) -> Tuple[float, float]:
    """Distorted normalized coords -> ideal normalized coords, by
    fixed-point iteration (no closed-form inverse for this model;
    see module docstring)."""
    x, y = x_d, y_d
    for _ in range(_UNDISTORT_ITERATIONS):
        r2 = x * x + y * y
        radial = 1.0 + k.k1 * r2 + k.k2 * r2 ** 2 + k.k3 * r2 ** 3
        tangential_x = 2.0 * k.p1 * x * y + k.p2 * (r2 + 2.0 * x * x)
        tangential_y = k.p1 * (r2 + 2.0 * y * y) + 2.0 * k.p2 * x * y
        x = (x_d - tangential_x) / radial
        y = (y_d - tangential_y) / radial
    return x, y


def quat_to_matrix(q: Quat) -> Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]:
    """Unit quaternion -> 3x3 row-major rotation matrix (camera-to-world,
    matching q.rotate for standard (w, x, y, z) ordering)."""
    w, x, y, z = q.w, q.x, q.y, q.z
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def matrix_to_quat(R) -> Quat:
    """Convert 3x3 rotation matrix to unit Quat(w, x, y, z).
    Works with tuples, lists, or numpy arrays."""
    r00, r01, r02 = float(R[0][0]), float(R[0][1]), float(R[0][2])
    r10, r11, r12 = float(R[1][0]), float(R[1][1]), float(R[1][2])
    r20, r21, r22 = float(R[2][0]), float(R[2][1]), float(R[2][2])

    trace = r00 + r11 + r22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r21 - r12) * s
        y = (r02 - r20) * s
        z = (r10 - r01) * s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + r00 - r11 - r22))
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + r11 - r00 - r22))
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + r22 - r00 - r11))
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s

    return Quat(w, x, y, z).normalized()


