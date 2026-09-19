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
    """Minimal camera for Studio viewport."""

    position: Vec3
    forward: Vec3
    up: Vec3 = Vec3(0.0, 0.0, 1.0)
    fov_deg: float = 60.0
    aspect: float = 16.0 / 9.0
    near: float = 0.01
    far: float = 1000.0

    def view_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Compute view matrix (4x4 row-major tuple)."""
        from engine.math import Mat3, Quat

        # Right = forward x up
        right = self.forward.cross(self.up)
        if right.length() < 1e-12:
            right = Vec3(1.0, 0.0, 0.0)
        else:
            right = right.normalized()

        # Recompute up = right x forward
        up = right.cross(self.forward).normalized()

        # Camera-to-world rotation
        rot = Mat3((
            (right.x, right.y, right.z),
            (up.x, up.y, up.z),
            (-self.forward.x, -self.forward.y, -self.forward.z),
        ))

        # World-to-camera is transpose of camera-to-world
        rot_t = rot.transpose()

        # Translation: -R^T * position
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
    """Minimal viewport for Studio."""

    camera: Camera
    width: int = 1920
    height: int = 1080

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
        """Simple frustum culling: return entities within the view frustum.

        Args:
            candidates: List of (entity_id, position: Vec3, radius: float)

        Returns:
            List of dicts with entity_id and position for visible entities.
        """
        visible = []
        for entity_id, pos, radius in candidates:
            # Simple distance-based culling for now
            # In a full implementation, this would do proper frustum culling
            to_camera = pos - self.camera.position
            dist = to_camera.length()
            if dist <= self.camera.far + radius:
                visible.append({"entity_id": entity_id, "position": {"x": pos.x, "y": pos.y, "z": pos.z}})
        return visible