"""Viewport: camera model and frustum culling for the visualization layer.

Headless engine -- no GPU/graphics library. A "viewport" here is the data
model a renderer would consume: camera pose, projection parameters, and a
visibility query that turns world entities into a depth-sorted render list.

Spec §16: VIEWPORT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from engine.physics.math3 import Vec3


@dataclass(frozen=True)
class Camera:
    """Pinhole camera: position + orientation (forward/up) + projection."""

    position: Vec3
    forward: Vec3
    up: Vec3 = Vec3(0.0, 0.0, 1.0)
    fov_deg: float = 60.0
    aspect: float = 16.0 / 9.0
    near: float = 0.1
    far: float = 1000.0

    def __post_init__(self):
        object.__setattr__(self, "forward", self.forward.normalized())
        object.__setattr__(self, "up", self.up.normalized())

    @staticmethod
    def look_at(position: Vec3, target: Vec3, up: Vec3 = Vec3(0.0, 0.0, 1.0)) -> "Camera":
        return Camera(position=position, forward=(target - position).normalized(), up=up)

    def right(self) -> Vec3:
        return self.forward.cross(self.up).normalized()

    def true_up(self) -> Vec3:
        """Up vector orthogonalized against forward (forward/up need not be perpendicular as given)."""
        return self.right().cross(self.forward).normalized()

    def is_visible(self, point: Vec3, radius: float = 0.0) -> bool:
        """Frustum test: near/far depth plus horizontal/vertical FOV cone, radius-expanded."""
        to_point = point - self.position
        depth = to_point.dot(self.forward)

        if depth + radius < self.near or depth - radius > self.far:
            return False

        # Angle-based frustum check against a padded bound so a sphere
        # straddling the frustum edge still counts as visible.
        dist = to_point.length()
        if dist < 1e-9:
            return True

        half_fov = math.radians(self.fov_deg) / 2.0
        angle = math.acos(max(-1.0, min(1.0, depth / dist)))
        angular_pad = math.atan2(radius, max(depth, 1e-6))
        return angle - angular_pad <= half_fov


class Viewport:
    """Owns a camera and produces a depth-sorted visible-entity render list."""

    def __init__(self, camera: Camera):
        self.camera = camera

    def set_camera(self, camera: Camera) -> None:
        self.camera = camera

    def cull(
        self, entities: Iterable[Tuple[str, Vec3, float]]
    ) -> List[dict]:
        """Filter and depth-sort entities visible from the camera.

        Args:
            entities: iterable of (entity_id, position, radius)

        Returns:
            List of {"entity_id", "position", "distance"} dicts, nearest first.
        """
        visible = []
        for entity_id, position, radius in entities:
            if self.camera.is_visible(position, radius):
                distance = (position - self.camera.position).length()
                visible.append({
                    "entity_id": entity_id,
                    "position": position.to_dict(),
                    "distance": distance,
                })

        visible.sort(key=lambda e: e["distance"])
        return visible
