"""Minimal viewport types for Studio (core).

This is a minimal subset of the full render module, providing only the
types needed by the core StudioSession without depending on the full
physics/rendering engine in the child project.

The full render module with physics integration, materials, lighting,
etc. lives in the child project (reality-engine-child.physics.engine.render).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.math import Vec3


@dataclass(frozen=True)
class Camera:
    """Camera for Studio viewport: position + orientation + projection."""

    position: Vec3
    forward: Vec3
    up: Vec3 = Vec3(0.0, 0.0, 1.0)
    fov_deg: float = 60.0
    aspect: float = 16.0 / 9.0
    near: float = 0.01
    far: float = 1000.0

    def __post_init__(self):
        fwd = self.forward.normalized() if self.forward.length() > 1e-12 else Vec3(1.0, 0.0, 0.0)
        u = self.up.normalized() if self.up.length() > 1e-12 else Vec3(0.0, 0.0, 1.0)
        object.__setattr__(self, "forward", fwd)
        object.__setattr__(self, "up", u)

    @staticmethod
    def look_at(position: Vec3, target: Vec3, up: Vec3 = Vec3(0.0, 0.0, 1.0)) -> "Camera":
        fwd = target - position
        if fwd.length() < 1e-12:
            fwd = Vec3(1.0, 0.0, 0.0)
        else:
            fwd = fwd.normalized()
        return Camera(position=position, forward=fwd, up=up)

    def right(self) -> Vec3:
        r = self.forward.cross(self.up)
        if r.length() < 1e-12:
            return Vec3(1.0, 0.0, 0.0)
        return r.normalized()

    def true_up(self) -> Vec3:
        """Up vector orthogonalized against forward."""
        return self.right().cross(self.forward).normalized()

    def is_visible(self, point: Vec3, radius: float = 0.0) -> bool:
        """Frustum test: near/far depth plus horizontal/vertical FOV cone, radius-expanded."""
        import math

        to_point = point - self.position
        depth = to_point.dot(self.forward)

        if depth + radius < self.near or depth - radius > self.far:
            return False

        dist = to_point.length()
        if dist < 1e-9:
            return True

        half_fov = math.radians(self.fov_deg) / 2.0
        angle = math.acos(max(-1.0, min(1.0, depth / dist)))
        angular_pad = math.atan2(radius, max(depth, 1e-6))
        return angle - angular_pad <= half_fov

    def view_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Compute view matrix (4x4 row-major tuple)."""
        from engine.math import Mat3

        right = self.right()
        up = self.true_up()

        rot = Mat3((
            (right.x, right.y, right.z),
            (up.x, up.y, up.z),
            (-self.forward.x, -self.forward.y, -self.forward.z),
        ))

        rot_t = rot.transpose()
        pos = self.position
        tx = -(rot_t.rows[0][0] * pos.x + rot_t.rows[0][1] * pos.y + rot_t.rows[0][2] * pos.z)
        ty = -(rot_t.rows[1][0] * pos.x + rot_t.rows[1][1] * pos.y + rot_t.rows[1][2] * pos.z)
        tz = -(rot_t.rows[2][0] * pos.x + rot_t.rows[2][1] * pos.y + rot_t.rows[2][2] * pos.z)

        return (
            (rot_t.rows[0][0], rot_t.rows[0][1], rot_t.rows[0][2], tx),
            (rot_t.rows[1][0], rot_t.rows[1][1], rot_t.rows[1][2], ty),
            (rot_t.rows[2][0], rot_t.rows[2][1], rot_t.rows[2][2], tz),
            (0.0, 0.0, 0.0, 1.0),
        )


@dataclass
class Viewport:
    """Viewport for Studio."""

    camera: Camera
    width: int = 1920
    height: int = 1080

    def set_camera(self, camera: Camera) -> None:
        self.camera = camera

    def projection_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Compute perspective projection matrix (4x4 row-major tuple)."""
        import math

        f = 1.0 / math.tan(math.radians(self.camera.fov_deg) / 2.0)
        aspect = self.camera.aspect
        near = self.camera.near
        far = self.camera.far

        return (
            (f / aspect, 0.0, 0.0, 0.0),
            (0.0, f, 0.0, 0.0),
            (0.0, 0.0, (far + near) / (near - far), -1.0),
            (0.0, 0.0, (2.0 * far * near) / (near - far), 0.0),
        )

    def cull(self, candidates):
        """Filter and depth-sort entities visible from the camera.

        Args:
            candidates: Iterable of (entity_id, position: Vec3, radius: float)

        Returns:
            List of dicts with entity_id, position, and distance sorted nearest first.
        """
        visible = []
        for entity_id, pos, radius in candidates:
            if self.camera.is_visible(pos, radius):
                dist = (pos - self.camera.position).length()
                visible.append({
                    "entity_id": entity_id,
                    "position": pos.to_dict() if hasattr(pos, "to_dict") else {"x": pos.x, "y": pos.y, "z": pos.z},
                    "distance": dist,
                })
        visible.sort(key=lambda e: e["distance"])
        return visible