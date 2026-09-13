"""Pinhole camera model: intrinsics, Brown-Conrady distortion, extrinsics,
projection, unprojection, and ray casting (reconstruction-hardening
campaign, Phase 2 "Camera Calibration").

Before this module, `reconstruction/calibration/` was a README
placeholder and `reconstruction.backend.interface.ReconstructedCameraPose`
carried a raw (position, quaternion) pair with no attached intrinsics and
nothing to project a 3D point into a pixel or unproject a pixel+depth
back into 3D. That operation is the prerequisite for every downstream
step the campaign asks for next (depth -> point cloud, depth/camera
consistency checks) -- none of them are meaningful without a real,
tested camera model underneath.

Convention (matches `reconstruction.backend.interface.ReconstructedCameraPose`,
so a COLMAP-parsed pose plugs in directly): `CameraExtrinsics.rotation`
is the CAMERA-TO-WORLD orientation (rotating a direction expressed in the
camera's local frame into world space), `position` is the camera's
position in world space. World-to-camera uses the orientation's
conjugate (a unit quaternion's inverse rotation, see
`engine.physics.math3.Quat.conjugate`) -- reused rather than
reimplemented, same rotation math the physics engine already relies on.

Distortion model: standard Brown-Conrady (radial k1/k2/k3 + tangential
p1/p2), the same 5-parameter model COLMAP and OpenCV both use. Forward
distortion is a closed-form polynomial; undistortion (needed by
`unproject`, since a real pixel coordinate is always in distorted space)
uses fixed-point iteration -- there is no closed-form inverse for this
model, and 10 iterations converges to sub-pixel precision for anything
but extreme fisheye-level distortion, which nothing in this repo's
COLMAP/pinhole path produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from engine.physics.math3 import Quat, Vec3

#: Fixed-point undistortion iterations. 10 converges to well under a
#: pixel of error for the mild-to-moderate distortion real camera
#: calibrations produce; not validated against extreme fisheye lenses.
_UNDISTORT_ITERATIONS = 10

#: A point this close to (or behind) the camera plane has no well-defined
#: projection (division by ~zero/negative z) -- reported as None, never
#: as a fabricated pixel coordinate.
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
        """Distorted pixel (u, v) + metric depth (the z-distance along the
        camera's optical axis, matching how a real depth map is defined)
        -> world point."""
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        x_d = (u - self.intrinsics.cx) / self.intrinsics.fx
        y_d = (v - self.intrinsics.cy) / self.intrinsics.fy
        x, y = _undistort(x_d, y_d, self.intrinsics)
        p_cam = Vec3(x * depth, y * depth, depth)
        return self.extrinsics.camera_to_world(p_cam)

    def ray(self, u: float, v: float) -> Tuple[Vec3, Vec3]:
        """World-space (origin, normalized direction) for the ray through
        pixel (u, v) -- the camera's position, and the direction toward
        the unprojected point at unit depth."""
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
