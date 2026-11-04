"""Occlusion Reasoning and Line-of-Sight Visibility Engine (Directive Rule 5 & 6).

Determines whether an entity or candidate fixture in 3D space is genuinely visible,
partially occluded, or fully obstructed from a given camera viewpoint.
Prevents false-positive detections on hidden or obstructed surfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from world_ir.schema_v1 import Vector3


@dataclass(frozen=True)
class VisibilityCheckResult:
    """Quantitative visibility assessment from a specific camera pose."""

    is_visible: bool
    occlusion_ratio: float       # 0.0 (completely clear) to 1.0 (completely occluded)
    view_angle_deg: float        # Angle between surface normal and camera optical axis
    distance_m: float            # Distance from camera optical center to target centroid
    obstruction_entity_id: Optional[str] = None
    reason: str = ""


class OcclusionEngine:
    """Ray-casting line-of-sight and surface normal angle visibility validator."""

    def __init__(self, max_view_angle_deg: float = 75.0, min_distance_m: float = 0.2, max_distance_m: float = 12.0) -> None:
        self.max_view_angle_deg = max_view_angle_deg
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m

    def check_visibility(
        self,
        camera_position: Tuple[float, float, float],
        target_center: Tuple[float, float, float],
        target_normal: Tuple[float, float, float],
        occluders: Optional[Sequence[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]]] = None,
    ) -> VisibilityCheckResult:
        """Check if target surface is visible from camera position against occluding bounding boxes."""
        cam_pos = np.asarray(camera_position, dtype=np.float64)
        tgt_pos = np.asarray(target_center, dtype=np.float64)
        norm = np.asarray(target_normal, dtype=np.float64)

        norm_len = np.linalg.norm(norm)
        if norm_len > 1e-12:
            norm = norm / norm_len

        # 1. Ray vector from target to camera
        ray = cam_pos - tgt_pos
        dist = float(np.linalg.norm(ray))
        if dist < 1e-6:
            return VisibilityCheckResult(is_visible=True, occlusion_ratio=0.0, view_angle_deg=0.0, distance_m=dist)

        ray_dir = ray / dist

        # 2. Distance range check
        if dist < self.min_distance_m:
            return VisibilityCheckResult(
                is_visible=False,
                occlusion_ratio=1.0,
                view_angle_deg=0.0,
                distance_m=dist,
                reason=f"Target too close to camera ({dist:.2f}m < {self.min_distance_m}m)",
            )
        if dist > self.max_distance_m:
            return VisibilityCheckResult(
                is_visible=False,
                occlusion_ratio=1.0,
                view_angle_deg=0.0,
                distance_m=dist,
                reason=f"Target exceeds maximum resolving distance ({dist:.2f}m > {self.max_distance_m}m)",
            )

        # 3. Surface normal check (back-face culling): camera must be in front of the surface
        cos_theta = float(np.dot(norm, ray_dir))
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))

        if cos_theta <= 0.0 or angle_deg > self.max_view_angle_deg:
            return VisibilityCheckResult(
                is_visible=False,
                occlusion_ratio=1.0,
                view_angle_deg=angle_deg,
                distance_m=dist,
                reason=f"Oblique or back-facing viewpoint (angle {angle_deg:.1f}° > {self.max_view_angle_deg}°)",
            )

        # 4. Ray-AABB intersection test against potential occluders
        if occluders:
            for occ_id, occ_min, occ_max in occluders:
                if self._ray_intersects_aabb(tgt_pos, ray_dir, dist - 0.05, occ_min, occ_max):
                    return VisibilityCheckResult(
                        is_visible=False,
                        occlusion_ratio=1.0,
                        view_angle_deg=angle_deg,
                        distance_m=dist,
                        obstruction_entity_id=occ_id,
                        reason=f"Obstructed by solid structure '{occ_id}'",
                    )

        # Grazing angle penalization
        occlusion_ratio = max(0.0, (angle_deg - 45.0) / 45.0) if angle_deg > 45.0 else 0.0
        return VisibilityCheckResult(
            is_visible=True,
            occlusion_ratio=occlusion_ratio,
            view_angle_deg=angle_deg,
            distance_m=dist,
            reason="Clear line-of-sight",
        )

    @staticmethod
    def _ray_intersects_aabb(
        origin: np.ndarray,
        direction: np.ndarray,
        max_dist: float,
        b_min: Tuple[float, float, float],
        b_max: Tuple[float, float, float],
    ) -> bool:
        """Fast slab method ray-AABB intersection test."""
        bmin = np.asarray(b_min, dtype=np.float64)
        bmax = np.asarray(b_max, dtype=np.float64)

        t_min = 0.05  # Slight offset to avoid self-intersection
        t_max = max_dist

        for i in range(3):
            d = direction[i]
            if abs(d) < 1e-12:
                if origin[i] < bmin[i] or origin[i] > bmax[i]:
                    return False
            else:
                inv_d = 1.0 / d
                t1 = (bmin[i] - origin[i]) * inv_d
                t2 = (bmax[i] - origin[i]) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return False

        return True
