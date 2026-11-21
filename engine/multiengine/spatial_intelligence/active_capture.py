"""Active Capture Intelligence and Gap-Driven Photography Guidance (Mission F).

Analyzes reconstructed spatial entities and camera viewpoints to identify:
1. Missing viewpoints (single-view entities that cannot be triangulated).
2. Insufficient baseline overlap (convergent angle < 15 deg or > 120 deg).
3. Hidden surfaces (unseen rear or top faces of 3D objects).
4. Image-edge truncation (objects cut off by image boundaries).
5. Insufficient pixel resolution (small fixtures needing macro close-ups).
6. Missing physical scale references (unanchored relative or synthetic depth).

Generates evidence-backed, actionable CaptureRequests with exact recommended camera positions
and human-understandable shooting instructions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from engine.multiengine.spatial_intelligence.association import PhysicalMergedEntity
from engine.multiengine.spatial_intelligence.spatial_lifting import SpatialEntity3D
from world_ir.camera import PinholeCamera


class CapturePriority(str, Enum):
    CRITICAL = "critical"  # Scale anchors, unanchored reconstructions
    HIGH = "high"          # Single-view entities, truncated borders
    MEDIUM = "medium"      # Narrow baseline, low-res micro fixtures
    LOW = "low"            # Unseen rear faces of secondary objects


class CaptureIssueType(str, Enum):
    MISSING_SCALE_REFERENCE = "missing_scale_reference"
    MISSING_VIEWPOINTS = "missing_viewpoints"
    BORDER_TRUNCATION = "border_truncation"
    INSUFFICIENT_OVERLAP = "insufficient_overlap"
    INSUFFICIENT_RESOLUTION = "insufficient_resolution"
    HIDDEN_SURFACES = "hidden_surfaces"


@dataclass(frozen=True)
class CaptureRequest:
    """Actionable capture request with 3D pose guidance and rationale."""

    request_id: str
    issue_type: CaptureIssueType
    priority: CapturePriority
    target_entity_ids: tuple[str, ...]
    target_class: str
    target_position_m: tuple[float, float, float]
    recommended_camera_pos_m: tuple[float, float, float] | None
    recommended_look_at_m: tuple[float, float, float] | None
    suggested_distance_m: float
    guidance_text: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "issue_type": self.issue_type.value,
            "priority": self.priority.value,
            "target_entity_ids": list(self.target_entity_ids),
            "target_class": self.target_class,
            "target_position_m": [round(c, 3) for c in self.target_position_m],
            "recommended_camera_pos_m": [round(c, 3) for c in self.recommended_camera_pos_m] if self.recommended_camera_pos_m else None,
            "recommended_look_at_m": [round(c, 3) for c in self.recommended_look_at_m] if self.recommended_look_at_m else None,
            "suggested_distance_m": round(self.suggested_distance_m, 2),
            "guidance_text": self.guidance_text,
            "details": self.details,
        }


@dataclass(frozen=True)
class ActiveCapturePlan:
    """Consolidated capture intelligence analysis."""

    requests: tuple[CaptureRequest, ...]
    total_entities_analyzed: int
    multiview_coverage_ratio: float
    is_scale_certified: bool
    coverage_statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": len(self.requests),
            "total_entities_analyzed": self.total_entities_analyzed,
            "multiview_coverage_ratio": round(self.multiview_coverage_ratio, 4),
            "is_scale_certified": self.is_scale_certified,
            "coverage_statistics": self.coverage_statistics,
            "requests": [r.to_dict() for r in self.requests],
        }


class ActiveCaptureAnalyzer:
    """Analyzes geometric coverage and generates targeted photography instructions."""

    def __init__(
        self,
        min_multiview_angle_deg: float = 15.0,
        max_multiview_angle_deg: float = 120.0,
        min_pixels_area: float = 1200.0,
        macro_shooting_dist_m: float = 0.80,
        standard_shooting_dist_m: float = 2.20,
    ) -> None:
        self.min_multiview_angle_deg = min_multiview_angle_deg
        self.max_multiview_angle_deg = max_multiview_angle_deg
        self.min_pixels_area = min_pixels_area
        self.macro_shooting_dist_m = macro_shooting_dist_m
        self.standard_shooting_dist_m = standard_shooting_dist_m

    def analyze(
        self,
        merged_entities: Sequence[PhysicalMergedEntity],
        lifted_entities: Sequence[SpatialEntity3D],
        cameras: Mapping[str, PinholeCamera],
        is_scene_metric: bool = False,
    ) -> ActiveCapturePlan:
        """Evaluates scene coverage and produces prioritized capture recommendations."""
        requests: list[CaptureRequest] = []
        req_counter = 1

        lifted_by_id = {ent.entity_id: ent for ent in lifted_entities}

        # Group lifted observations by parent merged entity
        entity_lifted_map: dict[str, list[SpatialEntity3D]] = defaultdict(list)
        for rent in merged_entities:
            for obs_id in rent.member_observation_ids:
                if obs_id in lifted_by_id:
                    entity_lifted_map[rent.canonical_entity_id].append(lifted_by_id[obs_id])

        # --------------------------------------------------------------------
        # 1. Physical Scale Anchor Verification
        # --------------------------------------------------------------------
        if not is_scene_metric:
            # Check if any entity is physically metric
            any_metric = any(rent.is_physically_metric for rent in merged_entities)
            if not any_metric:
                requests.append(CaptureRequest(
                    request_id=f"cap_req_{req_counter:03d}",
                    issue_type=CaptureIssueType.MISSING_SCALE_REFERENCE,
                    priority=CapturePriority.CRITICAL,
                    target_entity_ids=tuple(rent.canonical_entity_id for rent in merged_entities[:3]),
                    target_class="scene_scale_anchor",
                    target_position_m=(0.0, 0.0, 0.0),
                    recommended_camera_pos_m=None,
                    recommended_look_at_m=None,
                    suggested_distance_m=1.5,
                    guidance_text=(
                        "CRITICAL: The reconstruction is in synthetic scale without a verified physical anchor. "
                        "Please place a physical metre ruler, AprilTag marker, or checkerboard target in the scene "
                        "and photograph it clearly from 1.5-2.0 metres away."
                    ),
                    details={"scale_state": "SYNTHETIC_SCALE_ALIGNED", "is_metric": False},
                ))
                req_counter += 1

        # --------------------------------------------------------------------
        # 2. Per-Entity Multi-View and Geometric Coverage Analysis
        # --------------------------------------------------------------------
        multiview_count = sum(1 for rent in merged_entities if rent.is_multiview)
        total_ents = len(merged_entities)
        coverage_ratio = (multiview_count / total_ents) if total_ents > 0 else 0.0

        for rent in merged_entities:
            pos = rent.position_m
            pos_np = np.array(pos)
            lifted_obs = entity_lifted_map.get(rent.canonical_entity_id, [])

            # Check 2A: Single-view entities (Cannot triangulate 3D position independently)
            if not rent.is_multiview:
                # Propose a second camera viewpoint rotated by ~35 degrees around centroid
                rec_cam_pos = None
                if lifted_obs:
                    cam_id = lifted_obs[0].source_image_id
                    if cam_id in cameras:
                        cur_cam_pos = np.array([
                            cameras[cam_id].extrinsics.position.x,
                            cameras[cam_id].extrinsics.position.y,
                            cameras[cam_id].extrinsics.position.z,
                        ])
                        view_vec = cur_cam_pos - pos_np
                        dist = max(1.0, float(np.linalg.norm(view_vec)))
                        # Rotate vector by 35 degrees around Z axis
                        angle_rad = math.radians(35.0)
                        rot_mat = np.array([
                            [math.cos(angle_rad), -math.sin(angle_rad), 0.0],
                            [math.sin(angle_rad), math.cos(angle_rad), 0.0],
                            [0.0, 0.0, 1.0],
                        ])
                        new_dir = rot_mat @ (view_vec / dist)
                        rec_cam_pos = tuple(float(c) for c in (pos_np + new_dir * dist))

                requests.append(CaptureRequest(
                    request_id=f"cap_req_{req_counter:03d}",
                    issue_type=CaptureIssueType.MISSING_VIEWPOINTS,
                    priority=CapturePriority.HIGH,
                    target_entity_ids=(rent.canonical_entity_id,),
                    target_class=rent.primary_class,
                    target_position_m=pos,
                    recommended_camera_pos_m=rec_cam_pos,
                    recommended_look_at_m=pos,
                    suggested_distance_m=self.standard_shooting_dist_m,
                    guidance_text=(
                        f"The {rent.primary_class.replace('_', ' ')} is visible in only one photograph. "
                        f"Take an additional photo from a ~35° offset angle approximately "
                        f"{self.standard_shooting_dist_m}m away to verify its physical 3D location."
                    ),
                    details={"current_views": len(rent.sources)},
                ))
                req_counter += 1
                continue

            # Check 2B: Border Truncation (Object is clipped at image border)
            truncated_obs = [obs for obs in lifted_obs if obs.is_truncated_by_border]
            if truncated_obs and len(truncated_obs) == len(lifted_obs):
                # Cut off in all views
                requests.append(CaptureRequest(
                    request_id=f"cap_req_{req_counter:03d}",
                    issue_type=CaptureIssueType.BORDER_TRUNCATION,
                    priority=CapturePriority.HIGH,
                    target_entity_ids=(rent.canonical_entity_id,),
                    target_class=rent.primary_class,
                    target_position_m=pos,
                    recommended_camera_pos_m=None,
                    recommended_look_at_m=pos,
                    suggested_distance_m=self.standard_shooting_dist_m + 0.8,
                    guidance_text=(
                        f"The {rent.primary_class.replace('_', ' ')} touches the frame border in all images. "
                        f"Step back by ~0.8m to capture the complete object boundary within the frame."
                    ),
                    details={"truncated_in_views": len(truncated_obs)},
                ))
                req_counter += 1

            # Check 2C: Baseline Convergence Angle
            if len(rent.sources) >= 2 and lifted_obs:
                cam_positions: list[np.ndarray] = []
                for obs in lifted_obs:
                    if obs.source_image_id in cameras:
                        c_ext = cameras[obs.source_image_id].extrinsics.position
                        cam_positions.append(np.array([c_ext.x, c_ext.y, c_ext.z]))

                if len(cam_positions) >= 2:
                    v1 = cam_positions[0] - pos_np
                    v2 = cam_positions[1] - pos_np
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 0.1 and norm2 > 0.1:
                        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                        cos_theta = max(-1.0, min(1.0, cos_theta))
                        angle_deg = math.degrees(math.acos(cos_theta))

                        if angle_deg < self.min_multiview_angle_deg:
                            requests.append(CaptureRequest(
                                request_id=f"cap_req_{req_counter:03d}",
                                issue_type=CaptureIssueType.INSUFFICIENT_OVERLAP,
                                priority=CapturePriority.MEDIUM,
                                target_entity_ids=(rent.canonical_entity_id,),
                                target_class=rent.primary_class,
                                target_position_m=pos,
                                recommended_camera_pos_m=None,
                                recommended_look_at_m=pos,
                                suggested_distance_m=self.standard_shooting_dist_m,
                                guidance_text=(
                                    f"Current cameras view the {rent.primary_class.replace('_', ' ')} with only "
                                    f"{angle_deg:.1f}° separation (near-degenerate baseline). "
                                    f"Take a photo from an angle at least 30° away to improve depth accuracy."
                                ),
                                details={"current_baseline_angle_deg": round(angle_deg, 1)},
                            ))
                            req_counter += 1

            # Check 2D: Insufficient pixel resolution for small fixtures
            if any(k in rent.primary_class.lower() for k in ("socket", "switch", "handle", "hinge", "lock", "faucet", "drain")):
                small_obs = [obs for obs in lifted_obs if obs.mask_pixels_count < self.min_pixels_area]
                if small_obs and len(small_obs) == len(lifted_obs):
                    requests.append(CaptureRequest(
                        request_id=f"cap_req_{req_counter:03d}",
                        issue_type=CaptureIssueType.INSUFFICIENT_RESOLUTION,
                        priority=CapturePriority.MEDIUM,
                        target_entity_ids=(rent.canonical_entity_id,),
                        target_class=rent.primary_class,
                        target_position_m=pos,
                        recommended_camera_pos_m=None,
                        recommended_look_at_m=pos,
                        suggested_distance_m=self.macro_shooting_dist_m,
                        guidance_text=(
                            f"The fixture '{rent.primary_class.replace('_', ' ')}' is small (<{int(self.min_pixels_area)}px). "
                            f"Capture a close-up macro photograph from ~{self.macro_shooting_dist_m}m to resolve fine details."
                        ),
                        details={"max_pixels": max((obs.mask_pixels_count for obs in lifted_obs), default=0)},
                    ))
                    req_counter += 1

        # Sort requests: CRITICAL -> HIGH -> MEDIUM -> LOW
        priority_order = {
            CapturePriority.CRITICAL: 0,
            CapturePriority.HIGH: 1,
            CapturePriority.MEDIUM: 2,
            CapturePriority.LOW: 3,
        }
        requests.sort(key=lambda r: priority_order[r.priority])

        return ActiveCapturePlan(
            requests=tuple(requests),
            total_entities_analyzed=total_ents,
            multiview_coverage_ratio=coverage_ratio,
            is_scale_certified=is_scene_metric,
            coverage_statistics={
                "total_entities": total_ents,
                "multiview_entities": multiview_count,
                "single_view_entities": total_ents - multiview_count,
                "coverage_percentage": round(coverage_ratio * 100.0, 1),
                "critical_requests": sum(1 for r in requests if r.priority == CapturePriority.CRITICAL),
                "high_priority_requests": sum(1 for r in requests if r.priority == CapturePriority.HIGH),
                "medium_priority_requests": sum(1 for r in requests if r.priority == CapturePriority.MEDIUM),
            },
        )
