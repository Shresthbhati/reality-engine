"""Multi-View Intelligence and Cross-View Association Engine (Phase 3).

Implements:
1. Epipolar Ray-Ray Closest Point of Approach (CPA) and Midpoint Triangulation.
2. Cross-view reprojection consistency and 2D bounding box inclusion.
3. Multi-cue appearance and color histogram similarity.
4. Semantic taxonomy compatibility gating.
5. Constrained graph / union-find association preventing same-image collisions.
6. Comprehensive ledger of decisions, competing hypotheses, and explicit refusal reasons.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from engine.multiengine.spatial_intelligence.spatial_lifting import SpatialEntity3D
from engine.multiengine.spatial_intelligence.taxonomy import (
    are_semantically_compatible,
)
from world_ir.camera import PinholeCamera, Vec3


class AssociationOutcome(str, Enum):
    MERGED = "merged"
    REFUSED = "refused"
    SPLIT = "split"


@dataclass(frozen=True)
class CrossViewRayCandidate:
    """Camera ray passing through the 2D box centroid of an observation."""

    observation_id: str
    source_image_id: str
    class_name: str
    origin: np.ndarray  # Camera center C in world frame (3,)
    direction: np.ndarray  # Unit ray direction in world frame (3,)
    box_2d: tuple[float, float, float, float]
    camera: PinholeCamera
    lifted_entity: SpatialEntity3D | None = None
    appearance_features: tuple[float, ...] = ()
    scale_state: str = "relative_depth"
    scale_units: str = "relative"
    is_physically_metric: bool = False
    coordinate_frame: str = "world:canonical"
    physical_anchor_ref: str | None = None
    track_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def validate_ray_candidate(candidate: CrossViewRayCandidate) -> list[str]:
    """Validates candidate ray properties, finite origins/directions, normalized ray, and scale/units alignment."""
    errors: list[str] = []
    if not np.all(np.isfinite(candidate.origin)):
        errors.append("non_finite_origin")
    if not np.all(np.isfinite(candidate.direction)):
        errors.append("non_finite_direction")
    else:
        norm = float(np.linalg.norm(candidate.direction))
        if norm < 1e-8:
            errors.append("zero_direction_vector")
        elif abs(norm - 1.0) > 1e-3:
            errors.append(f"unnormalized_direction_norm_{norm:.4f}")

    s_state = candidate.scale_state.lower()
    s_units = candidate.scale_units.lower()
    if s_state == "metric_anchored":
        if s_units != "m":
            errors.append(f"metric_anchored_invalid_units_{s_units}")
        if not candidate.is_physically_metric:
            errors.append("metric_anchored_requires_is_physically_metric")
        if not candidate.physical_anchor_ref or not candidate.physical_anchor_ref.startswith(("anchor://", "artifact://")):
            errors.append("metric_anchored_missing_or_invalid_physical_anchor_ref")
    elif s_state == "synthetic_scale_aligned":
        if candidate.is_physically_metric:
            errors.append("synthetic_scale_aligned_cannot_be_physically_metric")
        if candidate.physical_anchor_ref is not None:
            errors.append("synthetic_scale_aligned_cannot_have_physical_anchor_ref")
    elif s_state == "sfm_gauge":
        if candidate.is_physically_metric:
            errors.append("sfm_gauge_cannot_be_physically_metric")
        if candidate.physical_anchor_ref is not None:
            errors.append("sfm_gauge_cannot_have_physical_anchor_ref")
    elif s_state == "relative_depth":
        if candidate.is_physically_metric:
            errors.append("relative_depth_cannot_be_physically_metric")
        if candidate.physical_anchor_ref is not None:
            errors.append("relative_depth_cannot_have_physical_anchor_ref")
    elif s_state == "unknown":
        if candidate.is_physically_metric:
            errors.append("unknown_scale_cannot_be_physically_metric")
        if candidate.physical_anchor_ref is not None:
            errors.append("unknown_scale_cannot_have_physical_anchor_ref")
    else:
        errors.append(f"unrecognized_scale_state_{s_state}")

    return errors


def are_scale_states_compatible(state_a: str, state_b: str) -> bool:
    """Evaluates whether two scale states can be legitimately associated and merged."""
    sa = state_a.lower()
    sb = state_b.lower()
    if "unknown" in (sa, sb):
        # UNKNOWN scale can NEVER match or bridge any scale state
        return False
    # Only identical scale states may be merged across views
    return sa == sb


@dataclass(frozen=True)
class PairScore:
    """Scored compatibility between two cross-view observations."""

    obs_a_id: str
    obs_b_id: str
    score: float
    outcome: AssociationOutcome
    reason: str
    triangulated_position: tuple[float, float, float] | None = None
    ray_distance: float | None = None
    ray_distance_m: float | None = None
    ray_distance_synthetic: float | None = None
    ray_distance_gauge: float | None = None
    reprojection_err_px: float | None = None
    appearance_sim: float | None = None
    basis: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalMergedEntity:
    """Reconciled multi-view physical entity formed by merging compatible observations."""

    canonical_entity_id: str
    primary_class: str
    member_observation_ids: tuple[str, ...]
    sources: tuple[str, ...]
    position: tuple[float, float, float] | None = None
    dimensions: tuple[float, float, float] | None = None
    orientation_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    surface_normal: tuple[float, float, float] | None = None
    confidence: float = 0.5
    reprojection_residual_px: float = 0.0
    triangulation_residual: float = 0.0
    is_multiview: bool = False
    scale_state: str = "relative_depth"
    scale_units: str = "relative"
    is_physically_metric: bool = False
    dimension_status: str = "UNKNOWN"  # DIRECT, ESTIMATED, PRIOR, UNKNOWN
    covariance_xyz: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None
    residual_units: str = "relative"
    covariance_units: str = "relative^2"
    coordinate_frame: str = "world:canonical"
    physical_anchor_ref: str | None = None
    competing_hypotheses: tuple[dict[str, Any], ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        canonical_entity_id: str,
        primary_class: str,
        member_observation_ids: tuple[str, ...] | Sequence[str],
        sources: tuple[str, ...] | Sequence[str],
        position: tuple[float, float, float] | None = None,
        dimensions: tuple[float, float, float] | None = None,
        orientation_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        surface_normal: tuple[float, float, float] | None = None,
        confidence: float = 0.5,
        reprojection_residual_px: float = 0.0,
        triangulation_residual: float = 0.0,
        is_multiview: bool = False,
        scale_state: str = "relative_depth",
        scale_units: str = "relative",
        is_physically_metric: bool = False,
        dimension_status: str = "UNKNOWN",
        covariance_xyz: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None,
        residual_units: str = "relative",
        covariance_units: str = "relative^2",
        coordinate_frame: str = "world:canonical",
        physical_anchor_ref: str | None = None,
        competing_hypotheses: tuple[dict[str, Any], ...] | Sequence[dict[str, Any]] = (),
        details: dict[str, Any] | None = None,
        position_m: tuple[float, float, float] | None = None,
        dimensions_m: tuple[float, float, float] | None = None,
        ray_triangulation_residual_m: float | None = None,
    ) -> None:
        eff_pos = position if position is not None else position_m
        eff_dims = dimensions if dimensions is not None else dimensions_m
        eff_res = triangulation_residual if triangulation_residual != 0.0 else (ray_triangulation_residual_m or 0.0)

        object.__setattr__(self, "canonical_entity_id", canonical_entity_id)
        object.__setattr__(self, "primary_class", primary_class)
        object.__setattr__(self, "member_observation_ids", tuple(member_observation_ids))
        object.__setattr__(self, "sources", tuple(sources))
        object.__setattr__(self, "position", eff_pos)
        object.__setattr__(self, "dimensions", eff_dims)
        object.__setattr__(self, "orientation_quat_wxyz", orientation_quat_wxyz)
        object.__setattr__(self, "surface_normal", surface_normal)
        object.__setattr__(self, "confidence", float(confidence))
        object.__setattr__(self, "reprojection_residual_px", float(reprojection_residual_px))
        object.__setattr__(self, "triangulation_residual", float(eff_res))
        object.__setattr__(self, "is_multiview", bool(is_multiview))
        object.__setattr__(self, "scale_state", scale_state)
        object.__setattr__(self, "scale_units", scale_units)
        object.__setattr__(self, "is_physically_metric", bool(is_physically_metric))
        object.__setattr__(self, "dimension_status", dimension_status)
        object.__setattr__(self, "covariance_xyz", covariance_xyz)
        object.__setattr__(self, "residual_units", residual_units)
        object.__setattr__(self, "covariance_units", covariance_units)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "physical_anchor_ref", physical_anchor_ref)
        object.__setattr__(self, "competing_hypotheses", tuple(competing_hypotheses))
        object.__setattr__(self, "details", details if details is not None else {})

    @property
    def is_physically_anchored(self) -> bool:
        """Physical metric anchor verification: requires metric_anchored scale, is_physically_metric, and valid anchor reference."""
        return (
            self.is_physically_metric
            and self.scale_state.lower() == "metric_anchored"
            and bool(
                self.physical_anchor_ref
                and self.physical_anchor_ref.startswith(("anchor://", "artifact://"))
            )
        )

    @property
    def position_m(self) -> tuple[float, float, float] | None:
        """Physical metric position in metres. Strictly None unless physically anchored."""
        return self.position if self.is_physically_anchored else None

    @property
    def dimensions_m(self) -> tuple[float, float, float] | None:
        """Physical metric dimensions in metres. Strictly None unless physically anchored."""
        return self.dimensions if self.is_physically_anchored else None

    @property
    def ray_triangulation_residual_m(self) -> float | None:
        """Physical metric residual in metres. Strictly None unless physically anchored."""
        return self.triangulation_residual if self.is_physically_anchored else None

    @property
    def position_synthetic(self) -> tuple[float, float, float] | None:
        """Synthetic-unit position. Strictly None unless synthetic_scale_aligned."""
        return self.position if self.scale_state.lower() == "synthetic_scale_aligned" else None

    @property
    def dimensions_synthetic(self) -> tuple[float, float, float] | None:
        """Synthetic-unit dimensions. Strictly None unless synthetic_scale_aligned."""
        return self.dimensions if self.scale_state.lower() == "synthetic_scale_aligned" else None

    @property
    def ray_triangulation_residual_synthetic(self) -> float | None:
        """Synthetic-unit residual. Strictly None unless synthetic_scale_aligned."""
        return self.triangulation_residual if self.scale_state.lower() == "synthetic_scale_aligned" else None

    @property
    def position_gauge(self) -> tuple[float, float, float] | None:
        """SfM-gauge position. Strictly None unless sfm_gauge."""
        return self.position if self.scale_state.lower() == "sfm_gauge" else None

    @property
    def dimensions_gauge(self) -> tuple[float, float, float] | None:
        """SfM-gauge dimensions. Strictly None unless sfm_gauge."""
        return self.dimensions if self.scale_state.lower() == "sfm_gauge" else None

    @property
    def ray_triangulation_residual_gauge(self) -> float | None:
        """SfM-gauge residual. Strictly None unless sfm_gauge."""
        return self.triangulation_residual if self.scale_state.lower() == "sfm_gauge" else None

    def to_dict(self) -> dict[str, Any]:
        is_anchored = self.is_physically_anchored
        is_synth = self.scale_state.lower() == "synthetic_scale_aligned"
        is_gauge = self.scale_state.lower() == "sfm_gauge"

        return {
            "canonical_entity_id": self.canonical_entity_id,
            "primary_class": self.primary_class,
            "member_observation_ids": list(self.member_observation_ids),
            "sources": list(self.sources),
            "position": [round(c, 4) for c in self.position] if self.position is not None else None,
            "position_m": [round(c, 4) for c in self.position] if (is_anchored and self.position is not None) else None,
            "position_synthetic": [round(c, 4) for c in self.position] if (is_synth and self.position is not None) else None,
            "position_gauge": [round(c, 4) for c in self.position] if (is_gauge and self.position is not None) else None,
            "dimensions": [round(d, 4) for d in self.dimensions] if self.dimensions is not None else None,
            "dimensions_m": [round(d, 4) for d in self.dimensions] if (is_anchored and self.dimensions is not None) else None,
            "dimensions_synthetic": [round(d, 4) for d in self.dimensions] if (is_synth and self.dimensions is not None) else None,
            "dimensions_gauge": [round(d, 4) for d in self.dimensions] if (is_gauge and self.dimensions is not None) else None,
            "dimension_status": self.dimension_status,
            "orientation_quat_wxyz": [round(q, 4) for q in self.orientation_quat_wxyz],
            "surface_normal": [round(n, 4) for n in self.surface_normal] if self.surface_normal else None,
            "confidence": round(self.confidence, 4),
            "reprojection_residual_px": round(self.reprojection_residual_px, 2),
            "triangulation_residual": round(self.triangulation_residual, 4),
            "ray_triangulation_residual_m": round(self.triangulation_residual, 4) if is_anchored else None,
            "ray_triangulation_residual_synthetic": round(self.triangulation_residual, 4) if is_synth else None,
            "ray_triangulation_residual_gauge": round(self.triangulation_residual, 4) if is_gauge else None,
            "residual_units": self.residual_units,
            "covariance_units": self.covariance_units,
            "is_multiview": self.is_multiview,
            "scale_state": self.scale_state,
            "scale_units": self.scale_units,
            "is_physically_metric": self.is_physically_metric,
            "coordinate_frame": self.coordinate_frame,
            "physical_anchor_ref": self.physical_anchor_ref,
            "covariance_xyz": [[round(v, 6) for v in row] for row in self.covariance_xyz] if self.covariance_xyz is not None else None,
            "competing_hypotheses": list(self.competing_hypotheses),
            "details": self.details,
        }


def compute_ray_cpa(
    origin_a: np.ndarray,
    dir_a: np.ndarray,
    origin_b: np.ndarray,
    dir_b: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Computes the Closest Point of Approach (CPA) between two 3D rays.

    Rays:
      L_a(s) = origin_a + s * dir_a, s > 0
      L_b(t) = origin_b + t * dir_b, t > 0

    Returns:
      (distance_between_rays, s_param, t_param, point_on_ray_a, point_on_ray_b)
    """
    w0 = origin_a - origin_b
    a = float(np.dot(dir_a, dir_a))
    b = float(np.dot(dir_a, dir_b))
    c = float(np.dot(dir_b, dir_b))
    d = float(np.dot(dir_a, w0))
    e = float(np.dot(dir_b, w0))

    denom = a * c - b * b
    if denom < 1e-6:
        # Near parallel rays
        s_c = 0.0
        t_c = e / c if c > 1e-6 else 0.0
    else:
        s_c = (b * e - c * d) / denom
        t_c = (a * e - b * d) / denom

    p_a = origin_a + s_c * dir_a
    p_b = origin_b + t_c * dir_b
    dist = float(np.linalg.norm(p_a - p_b))

    return dist, s_c, t_c, p_a, p_b


def build_camera_ray(
    observation_id: str,
    source_image_id: str,
    class_name: str,
    box_2d: Sequence[float],
    camera: PinholeCamera,
    lifted_entity: SpatialEntity3D | None = None,
    appearance_features: Sequence[float] | None = None,
    scale_state: str | None = None,
    scale_units: str | None = None,
    is_physically_metric: bool | None = None,
    coordinate_frame: str = "world:canonical",
    physical_anchor_ref: str | None = None,
    track_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> CrossViewRayCandidate:
    """Builds a CrossViewRayCandidate from a 2D bounding box and calibrated camera."""
    u_mid = 0.5 * (float(box_2d[0]) + float(box_2d[2]))
    v_mid = 0.5 * (float(box_2d[1]) + float(box_2d[3]))

    # Direction in camera coordinates (Z=1.0)
    fx, fy = camera.intrinsics.fx, camera.intrinsics.fy
    cx, cy = camera.intrinsics.cx, camera.intrinsics.cy
    dir_cam = np.array([(u_mid - cx) / fx, (v_mid - cy) / fy, 1.0], dtype=np.float64)
    norm_cam = float(np.linalg.norm(dir_cam))
    if norm_cam < 1e-8 or not np.isfinite(norm_cam):
        raise ValueError(f"Invalid camera ray direction norm: {norm_cam}")
    dir_cam /= norm_cam

    # Transform direction to world coordinates via R_wc
    q = camera.extrinsics.rotation
    R_wc = np.array([
        [1.0 - 2.0 * (q.y ** 2 + q.z ** 2), 2.0 * (q.x * q.y - q.z * q.w), 2.0 * (q.x * q.z + q.y * q.w)],
        [2.0 * (q.x * q.y + q.z * q.w), 1.0 - 2.0 * (q.x ** 2 + q.z ** 2), 2.0 * (q.y * q.z - q.x * q.w)],
        [2.0 * (q.x * q.z - q.y * q.w), 2.0 * (q.y * q.z + q.x * q.w), 1.0 - 2.0 * (q.x ** 2 + q.y ** 2)],
    ], dtype=np.float64)

    dir_world = R_wc @ dir_cam
    norm_w = float(np.linalg.norm(dir_world))
    if norm_w < 1e-8 or not np.isfinite(norm_w):
        raise ValueError(f"Invalid world ray direction norm: {norm_w}")
    dir_world /= norm_w

    cam_pos = np.array([camera.extrinsics.position.x, camera.extrinsics.position.y, camera.extrinsics.position.z], dtype=np.float64)
    if not np.all(np.isfinite(cam_pos)):
        raise ValueError(f"Non-finite camera origin: {cam_pos}")

    raw_scale = (
        scale_state if scale_state is not None
        else (lifted_entity.scale_state if lifted_entity is not None else "relative_depth")
    ).lower()

    if raw_scale == "metric_anchored":
        eff_scale_state = "metric_anchored"
        eff_scale_units = scale_units if scale_units is not None else "m"
        eff_is_metric = is_physically_metric if is_physically_metric is not None else True
        eff_anchor_ref = physical_anchor_ref
    elif raw_scale == "synthetic_scale_aligned":
        eff_scale_state = "synthetic_scale_aligned"
        eff_scale_units = scale_units if scale_units is not None else "synthetic_m"
        eff_is_metric = False  # Synthetic scale is NEVER physically metric
        eff_anchor_ref = None
    elif raw_scale == "sfm_gauge":
        eff_scale_state = "sfm_gauge"
        eff_scale_units = scale_units if scale_units is not None else "sfm_units"
        eff_is_metric = False
        eff_anchor_ref = None
    elif raw_scale == "unknown":
        eff_scale_state = "unknown"
        eff_scale_units = scale_units if scale_units is not None else "unknown"
        eff_is_metric = False
        eff_anchor_ref = None
    else:
        eff_scale_state = "relative_depth"
        eff_scale_units = scale_units if scale_units is not None else "relative"
        eff_is_metric = False
        eff_anchor_ref = None

    return CrossViewRayCandidate(
        observation_id=observation_id,
        source_image_id=source_image_id,
        class_name=class_name,
        origin=cam_pos,
        direction=dir_world,
        box_2d=(float(box_2d[0]), float(box_2d[1]), float(box_2d[2]), float(box_2d[3])),
        camera=camera,
        lifted_entity=lifted_entity,
        appearance_features=tuple(appearance_features or ()),
        scale_state=eff_scale_state,
        scale_units=eff_scale_units,
        is_physically_metric=eff_is_metric,
        coordinate_frame=coordinate_frame,
        physical_anchor_ref=eff_anchor_ref,
        track_id=track_id,
        details=dict(details or {}),
    )


def triangulate_multiview_rays(
    rays: Sequence[CrossViewRayCandidate],
    max_condition_number: float = 1000.0,
    min_triangulation_angle_deg: float = 2.0,
    min_depth_m: float = 0.05,
    max_depth_m: float = 40.0,
    huber_delta_m: float = 0.20,
) -> tuple[tuple[float, float, float] | None, float, float, np.ndarray | None, dict[str, Any]]:
    """Robust scale-aware N-view ray triangulation using SVD/least-squares with Huber re-weighting.

    Solves for 3D point p minimizing sum of squared perpendicular distances to rays:
        min_p sum_i ||(I - d_i d_i^T) (p - c_i)||^2
        => (sum_i P_i) p = sum_i P_i c_i, where P_i = I - d_i d_i^T.

    Enforces:
    - Finite origins, normalized unit directions.
    - Scale state consistency across all rays (fails on mixed or unknown scale).
    - Coordinate frame consistency across all rays.
    - Unit-aware depth, baseline, and Huber thresholds (never applies metre limits to SfM gauge).
    - Positive depth along all rays in the camera coordinate frame.
    - Covariance expressed in scene units and frame, without labeling gauge as metres.

    Returns:
        (triangulated_position_tuple, mean_ray_residual, mean_reprojection_px, covariance_3x3, metadata)
    """
    n = len(rays)
    if n < 2:
        return None, 0.0, 0.0, None, {"fail_reason": "fewer_than_2_rays"}

    # 1. Validate all ray candidates
    for r in rays:
        ray_errs = validate_ray_candidate(r)
        if ray_errs:
            return None, 0.0, 0.0, None, {
                "fail_reason": "invalid_ray_geometry",
                "observation_id": r.observation_id,
                "errors": ray_errs,
            }

    # 2. Validate scale state and coordinate frame consistency across all rays
    scales = set(r.scale_state.lower() for r in rays)
    if len(scales) > 1:
        return None, 0.0, 0.0, None, {
            "fail_reason": "mixed_scale_states",
            "scales": list(scales),
        }
    common_scale = next(iter(scales))
    if common_scale == "unknown":
        return None, 0.0, 0.0, None, {"fail_reason": "unknown_scale_refusal"}

    frames = set(r.coordinate_frame for r in rays)
    if len(frames) > 1:
        return None, 0.0, 0.0, None, {
            "fail_reason": "mixed_coordinate_frames",
            "frames": list(frames),
        }

    units_set = set(r.scale_units.lower() for r in rays)
    if len(units_set) > 1:
        return None, 0.0, 0.0, None, {
            "fail_reason": "mixed_scale_units",
            "units": list(units_set),
        }

    # 3. Check pairwise baselines and max angle
    max_angle_deg = 0.0
    max_baseline = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cos_a = float(np.dot(rays[i].direction, rays[j].direction))
            cos_a = max(-1.0, min(1.0, cos_a))
            deg = math.degrees(math.acos(cos_a))
            if deg > max_angle_deg:
                max_angle_deg = deg
            b = float(np.linalg.norm(rays[i].origin - rays[j].origin))
            if b > max_baseline:
                max_baseline = b

    if max_angle_deg < min_triangulation_angle_deg:
        return None, 0.0, 0.0, None, {
            "fail_reason": "near_parallel_rays",
            "max_angle_deg": round(max_angle_deg, 2),
        }

    # 4. Scale policy parameter setup
    is_gauge = common_scale in ("sfm_gauge", "relative_depth")
    is_synthetic = common_scale == "synthetic_scale_aligned"
    is_metric = common_scale == "metric_anchored"

    if is_gauge:
        if max_baseline < 1e-5:
            return None, 0.0, 0.0, None, {
                "fail_reason": "zero_camera_baseline",
                "max_baseline": round(max_baseline, 6),
            }
        min_depth = 1e-4 * max(1.0, max_baseline)
        max_depth = 1000.0 * max(1.0, max_baseline)
        huber_delta = max(1e-4, 0.05 * max_baseline)
        residual_units = "sfm_units" if common_scale == "sfm_gauge" else "relative"
        covariance_units = f"{residual_units}^2"
    elif is_synthetic:
        if max_baseline < 0.05:
            return None, 0.0, 0.0, None, {
                "fail_reason": "zero_camera_baseline",
                "max_baseline": round(max_baseline, 4),
            }
        min_depth = 0.10
        max_depth = 50.0
        huber_delta = 0.25
        residual_units = "synthetic_m"
        covariance_units = "synthetic_m^2"
    else:  # is_metric
        # Require valid anchor provenance across all rays
        anchors = set(r.physical_anchor_ref for r in rays)
        if len(anchors) > 1 or None in anchors:
            return None, 0.0, 0.0, None, {
                "fail_reason": "mixed_anchor_references",
                "anchors": list(anchors),
            }
        anchor_val = next(iter(anchors))
        if not anchor_val or not anchor_val.startswith(("anchor://", "artifact://")):
            return None, 0.0, 0.0, None, {
                "fail_reason": "invalid_anchor_reference",
                "anchor": anchor_val,
            }

        if max_baseline < 0.05:
            return None, 0.0, 0.0, None, {
                "fail_reason": "zero_camera_baseline",
                "max_baseline": round(max_baseline, 4),
            }
        min_depth = min_depth_m
        max_depth = max_depth_m
        huber_delta = huber_delta_m
        residual_units = "m"
        covariance_units = "m^2"

    # 5. Build linear system A p = b
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    P_matrices: list[np.ndarray] = []

    for r in rays:
        d = r.direction / np.linalg.norm(r.direction)
        c = r.origin
        P_i = np.eye(3, dtype=np.float64) - np.outer(d, d)
        P_matrices.append(P_i)
        A += P_i
        b += P_i @ c

    # 6. Check condition number of A
    try:
        _, S, _ = np.linalg.svd(A)
        cond_num = float(S[0] / max(1e-12, S[2]))
        if S[2] < 1e-4 or cond_num > max_condition_number:
            return None, 0.0, 0.0, None, {
                "fail_reason": "ill_conditioned_geometry",
                "condition_number": round(cond_num, 2),
            }
        p_init = np.linalg.solve(A, b)
    except Exception as e:
        return None, 0.0, 0.0, None, {"fail_reason": f"solver_error: {e}"}

    # 7. Check positive depth along all rays in the camera coordinate frame
    depths: list[float] = []
    for r in rays:
        depth = float(np.dot(p_init - r.origin, r.direction))
        depths.append(depth)
        if depth < min_depth:
            return None, 0.0, 0.0, None, {
                "fail_reason": "ray_intersection_behind_camera",
                "depths": [round(d, 3) for d in depths],
            }
        if depth > max_depth:
            return None, 0.0, 0.0, None, {
                "fail_reason": "excessive_scene_depth",
                "depths": [round(d, 3) for d in depths],
            }

    p_sol = p_init

    # 8. Huber re-weighting if n >= 3
    if n >= 3:
        A_w = np.zeros((3, 3), dtype=np.float64)
        b_w = np.zeros(3, dtype=np.float64)
        for i, r in enumerate(rays):
            c = r.origin
            P_i = P_matrices[i]
            res_vec = P_i @ (p_init - c)
            res_norm = float(np.linalg.norm(res_vec))
            w = huber_delta / res_norm if res_norm > huber_delta else 1.0
            A_w += w * P_i
            b_w += w * (P_i @ c)
        try:
            p_hub = np.linalg.solve(A_w, b_w)
            all_valid = True
            for r in rays:
                d = float(np.dot(p_hub - r.origin, r.direction))
                if d < min_depth or d > max_depth:
                    all_valid = False
                    break
            if all_valid:
                p_sol = p_hub
        except Exception:
            pass

    # 9. Reprojection error across all supporting views
    p_vec = Vec3(float(p_sol[0]), float(p_sol[1]), float(p_sol[2]))
    reproj_errors: list[float] = []
    for r in rays:
        uv = r.camera.project(p_vec)
        if uv is None:
            return None, 0.0, 0.0, None, {"fail_reason": "reprojection_behind_camera"}
        cx = 0.5 * (r.box_2d[0] + r.box_2d[2])
        cy = 0.5 * (r.box_2d[1] + r.box_2d[3])
        err = math.hypot(uv[0] - cx, uv[1] - cy)
        reproj_errors.append(err)

    mean_reproj = float(np.mean(reproj_errors)) if reproj_errors else 0.0

    # 10. Ray residuals
    ray_residuals: list[float] = []
    for i, r in enumerate(rays):
        P_i = P_matrices[i]
        res = float(np.linalg.norm(P_i @ (p_sol - r.origin)))
        ray_residuals.append(res)
    mean_ray_res = float(np.mean(ray_residuals)) if ray_residuals else 0.0

    # 11. Spatial Covariance Matrix
    sigma_ang = 0.005
    A_reg = A + 1e-4 * np.eye(3)
    inv_A = np.linalg.inv(A_reg)
    mean_depth = float(np.mean(depths))
    cov_3x3 = (sigma_ang * mean_depth) ** 2 * inv_A

    p_tuple = (float(p_sol[0]), float(p_sol[1]), float(p_sol[2]))
    meta = {
        "depths": [round(d, 3) for d in depths],
        "condition_number": round(cond_num, 2),
        "max_angle_deg": round(max_angle_deg, 2),
        "max_baseline": round(max_baseline, 4),
        "reproj_errors_px": [round(e, 2) for e in reproj_errors],
        "ray_residuals": [round(r, 4) for r in ray_residuals],
        "scale_state": common_scale,
        "residual_units": residual_units,
        "covariance_units": covariance_units,
    }
    if is_metric:
        meta["ray_residuals_m"] = [round(r, 4) for r in ray_residuals]
        meta["physical_anchor_ref"] = rays[0].physical_anchor_ref
    elif is_synthetic:
        meta["ray_residuals_synthetic"] = [round(r, 4) for r in ray_residuals]
    elif is_gauge:
        meta["ray_residuals_gauge"] = [round(r, 4) for r in ray_residuals]

    return p_tuple, mean_ray_res, mean_reproj, cov_3x3, meta


def evaluate_cross_view_pair(
    ray_a: CrossViewRayCandidate,
    ray_b: CrossViewRayCandidate,
    max_ray_distance_m: float = 0.40,
    max_reproj_err_px: float = 120.0,
    min_appearance_sim: float = 0.20,
) -> PairScore:
    """Rigorously evaluates whether two observations across different views represent the same physical entity."""
    # 1. Same-image refusal
    if ray_a.source_image_id == ray_b.source_image_id:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="same_image_conflict",
        )

    # 2. Candidate ray validation
    err_a = validate_ray_candidate(ray_a)
    err_b = validate_ray_candidate(ray_b)
    if err_a or err_b:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="invalid_ray_geometry",
            basis={"errors_a": err_a, "errors_b": err_b},
        )

    # 3. Scale state compatibility check
    sa = ray_a.scale_state.lower()
    sb = ray_b.scale_state.lower()
    if "unknown" in (sa, sb):
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="unknown_scale_refusal",
            basis={"scale_a": ray_a.scale_state, "scale_b": ray_b.scale_state},
        )
    if not are_scale_states_compatible(sa, sb):
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="incompatible_scale_states",
            basis={"scale_a": ray_a.scale_state, "scale_b": ray_b.scale_state},
        )

    # 4. Coordinate frame check
    if ray_a.coordinate_frame != ray_b.coordinate_frame:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="coordinate_frame_mismatch",
            basis={"frame_a": ray_a.coordinate_frame, "frame_b": ray_b.coordinate_frame},
        )

    # 4b. Physical anchor provenance verification for metric_anchored
    if sa == "metric_anchored":
        if not ray_a.physical_anchor_ref or not ray_b.physical_anchor_ref or ray_a.physical_anchor_ref != ray_b.physical_anchor_ref:
            return PairScore(
                obs_a_id=ray_a.observation_id,
                obs_b_id=ray_b.observation_id,
                score=0.0,
                outcome=AssociationOutcome.REFUSED,
                reason="mixed_anchor_references",
                basis={"anchor_a": ray_a.physical_anchor_ref, "anchor_b": ray_b.physical_anchor_ref},
            )

    # 5. Semantic compatibility
    if not are_semantically_compatible(ray_a.class_name, ray_b.class_name):
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="semantic_incompatible",
            basis={"class_a": ray_a.class_name, "class_b": ray_b.class_name},
        )

    # 6. Baseline check: cameras must not be co-located
    cam_baseline = float(np.linalg.norm(ray_a.origin - ray_b.origin))
    is_gauge = sa in ("sfm_gauge", "relative_depth")
    min_base = 1e-5 if is_gauge else 0.05
    if cam_baseline < min_base:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="zero_camera_baseline",
            basis={"baseline": round(cam_baseline, 6 if is_gauge else 4)},
        )

    # 7. Minimum triangulation angle check: rays must not be near-parallel
    cos_angle = float(np.dot(ray_a.direction, ray_b.direction))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_deg = math.degrees(math.acos(cos_angle))
    if angle_deg < 2.0:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="near_parallel_rays",
            basis={"angle_deg": round(angle_deg, 2)},
        )

    # 8. Ray-Ray CPA and Triangulation
    dist_ray, s_a, t_b, p_a, p_b = compute_ray_cpa(
        ray_a.origin, ray_a.direction, ray_b.origin, ray_b.direction
    )

    # Scale-aware depth limits and ray distance tolerances
    if is_gauge:
        min_depth = 1e-4 * max(1.0, cam_baseline)
        max_depth = 1000.0 * max(1.0, cam_baseline)
        mean_dist = 0.5 * (s_a + t_b)
        adaptive_tol = max(0.10 * cam_baseline, 0.08 * mean_dist)
        ray_dist_m_val = None  # Never label gauge residual as metres
        ray_dist_synth_val = None
        ray_dist_gauge_val = dist_ray
    elif sa == "synthetic_scale_aligned":
        min_depth = 0.10
        max_depth = 50.0
        mean_dist = 0.5 * (s_a + t_b)
        adaptive_tol = max(0.50, 0.08 * mean_dist)
        ray_dist_m_val = None  # Never label synthetic residual as metres
        ray_dist_synth_val = dist_ray
        ray_dist_gauge_val = None
    else:  # metric_anchored
        min_depth = 0.05
        max_depth = 30.0
        mean_dist = 0.5 * (s_a + t_b)
        adaptive_tol = max(max_ray_distance_m, 0.05 * mean_dist)
        ray_dist_m_val = dist_ray
        ray_dist_synth_val = None
        ray_dist_gauge_val = None

    # Rays must intersect in FRONT of both cameras
    if s_a <= min_depth or t_b <= min_depth:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="ray_intersection_behind_camera",
            basis={"s_depth_a": round(s_a, 3), "t_depth_b": round(t_b, 3)},
        )

    # Maximum distance bound
    if s_a > max_depth or t_b > max_depth:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="excessive_scene_depth",
            basis={"s_depth_a": round(s_a, 2), "t_depth_b": round(t_b, 2)},
        )

    # Adaptive ray distance tolerance
    if dist_ray > adaptive_tol:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="excessive_ray_distance",
            ray_distance=dist_ray,
            ray_distance_m=ray_dist_m_val,
            ray_distance_synthetic=ray_dist_synth_val,
            ray_distance_gauge=ray_dist_gauge_val,
            basis={"dist_ray": round(dist_ray, 4), "tolerance": round(adaptive_tol, 4)},
        )

    # Triangulated 3D position (midpoint)
    p_mid = 0.5 * (p_a + p_b)
    p_mid_tuple = (float(p_mid[0]), float(p_mid[1]), float(p_mid[2]))

    # 9. Cross-View Reprojection Verification
    p_vec = Vec3(p_mid_tuple[0], p_mid_tuple[1], p_mid_tuple[2])
    uv_a = ray_a.camera.project(p_vec)
    uv_b = ray_b.camera.project(p_vec)

    if uv_a is None or uv_b is None:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="projection_failed",
            triangulated_position=p_mid_tuple,
            ray_distance=dist_ray,
            ray_distance_m=ray_dist_m_val,
            ray_distance_synthetic=ray_dist_synth_val,
            ray_distance_gauge=ray_dist_gauge_val,
        )

    # Reprojection error to box center
    cx_a = 0.5 * (ray_a.box_2d[0] + ray_a.box_2d[2])
    cy_a = 0.5 * (ray_a.box_2d[1] + ray_a.box_2d[3])
    err_a = math.hypot(uv_a[0] - cx_a, uv_a[1] - cy_a)

    cx_b = 0.5 * (ray_b.box_2d[0] + ray_b.box_2d[2])
    cy_b = 0.5 * (ray_b.box_2d[1] + ray_b.box_2d[3])
    err_b = math.hypot(uv_b[0] - cx_b, uv_b[1] - cy_b)

    mean_reproj = 0.5 * (err_a + err_b)
    if mean_reproj > max_reproj_err_px:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="excessive_reprojection_error",
            triangulated_position=p_mid_tuple,
            ray_distance=dist_ray,
            ray_distance_m=ray_dist_m_val,
            ray_distance_synthetic=ray_dist_synth_val,
            ray_distance_gauge=ray_dist_gauge_val,
            reprojection_err_px=mean_reproj,
            basis={"err_a_px": round(err_a, 2), "err_b_px": round(err_b, 2), "mean_err_px": round(mean_reproj, 2)},
        )

    # 10. Appearance Similarity (if available)
    app_sim = None
    if ray_a.appearance_features and ray_b.appearance_features:
        fa = np.array(ray_a.appearance_features, dtype=np.float64)
        fb = np.array(ray_b.appearance_features, dtype=np.float64)
        na = np.linalg.norm(fa)
        nb = np.linalg.norm(fb)
        if na > 1e-6 and nb > 1e-6:
            app_sim = float(np.dot(fa, fb) / (na * nb))
            if app_sim < min_appearance_sim:
                return PairScore(
                    obs_a_id=ray_a.observation_id,
                    obs_b_id=ray_b.observation_id,
                    score=0.0,
                    outcome=AssociationOutcome.REFUSED,
                    reason="appearance_dissimilarity",
                    triangulated_position=p_mid_tuple,
                    ray_distance=dist_ray,
                    ray_distance_m=ray_dist_m_val,
                    ray_distance_synthetic=ray_dist_synth_val,
                    ray_distance_gauge=ray_dist_gauge_val,
                    appearance_sim=app_sim,
                    basis={"sim": round(app_sim, 4)},
                )

    # Composite Score Calculation
    score_ray = math.exp(-0.5 * (dist_ray / max(1e-4, adaptive_tol * 0.5)) ** 2)
    score_reproj = math.exp(-0.5 * (mean_reproj / (max_reproj_err_px * 0.5)) ** 2)

    composite = (score_ray ** 0.6) * (score_reproj ** 0.4)
    if app_sim is not None:
        composite *= (0.7 + 0.3 * max(0.0, app_sim))

    # Check temporal track relationship if available
    if ray_a.track_id and ray_b.track_id:
        if ray_a.track_id != ray_b.track_id:
            frames_a = set(ray_a.details.get("track_frames", []))
            frames_b = set(ray_b.details.get("track_frames", []))
            if frames_a and frames_b and (frames_a & frames_b):
                return PairScore(
                    obs_a_id=ray_a.observation_id,
                    obs_b_id=ray_b.observation_id,
                    score=0.0,
                    outcome=AssociationOutcome.REFUSED,
                    reason="distinct_cooccurring_temporal_tracks",
                    basis={"track_a": ray_a.track_id, "track_b": ray_b.track_id},
                )
        else:
            # Same temporal track: boost composite confidence
            composite = min(0.99, composite + 0.15)

    basis_dict = {
        "score_ray": round(score_ray, 4),
        "score_reproj": round(score_reproj, 4),
        "dist_ray": round(dist_ray, 4),
        "mean_reproj_px": round(mean_reproj, 2),
        "s_a": round(s_a, 3),
        "t_b": round(t_b, 3),
    }
    if ray_a.track_id and ray_b.track_id and ray_a.track_id == ray_b.track_id:
        basis_dict["temporal_track_continuity"] = True
        basis_dict["track_id"] = ray_a.track_id

    return PairScore(
        obs_a_id=ray_a.observation_id,
        obs_b_id=ray_b.observation_id,
        score=round(composite, 4),
        outcome=AssociationOutcome.MERGED,
        reason="epipolar_reprojection_consensus",
        triangulated_position=p_mid_tuple,
        ray_distance=dist_ray,
        ray_distance_m=ray_dist_m_val,
        ray_distance_synthetic=ray_dist_synth_val,
        ray_distance_gauge=ray_dist_gauge_val,
        reprojection_err_px=mean_reproj,
        appearance_sim=app_sim,
        basis=basis_dict,
    )


# ============================================================================
# TEMPORAL IDENTITY TRACKING FOR SEQUENTIAL VIDEO AND FRAMES
# ============================================================================

@dataclass
class TemporalTrack:
    """Continuous temporal track of an observation sequence across video frames.

    NOTE: Temporal track IDs represent video-frame continuity and tracking evidence,
    NOT proof of 3D physical identity. Multi-view physical identity requires 3D cross-view
    epipolar geometry and cluster consistency verification.
    """

    track_id: str
    class_name: str
    active: bool = True
    observation_ids: list[str] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    timestamps_sec: list[float] = field(default_factory=list)
    last_box_2d: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_appearance: tuple[float, ...] = ()
    velocity_2d: tuple[float, float] = (0.0, 0.0)  # (vx, vy) pixels per second
    age_frames: int = 0
    missed_frames: int = 0
    is_physical_identity: bool = False
    termination_reason: str = "active"
    provenance: str = "temporal_observation_tracking"

    @property
    def centroid(self) -> tuple[float, float]:
        return (
            0.5 * (self.last_box_2d[0] + self.last_box_2d[2]),
            0.5 * (self.last_box_2d[1] + self.last_box_2d[3]),
        )

    def predicted_centroid(self, dt_sec: float) -> tuple[float, float]:
        """Estimates centroid position forward in time using 2D velocity."""
        cx, cy = self.centroid
        vx, vy = self.velocity_2d
        # Bound velocity extrapolation to accommodate camera pan without wild leaps on drops
        vx_clamped = max(-2500.0, min(2500.0, vx))
        vy_clamped = max(-2500.0, min(2500.0, vy))
        dt_clamped = max(0.0, min(1.0, dt_sec))
        return (cx + vx_clamped * dt_clamped, cy + vy_clamped * dt_clamped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "active": self.active,
            "observation_count": len(self.observation_ids),
            "frame_indices": self.frame_indices,
            "age_frames": self.age_frames,
            "missed_frames": self.missed_frames,
            "termination_reason": self.termination_reason,
            "velocity_2d": [round(v, 2) for v in self.velocity_2d],
            "is_physical_identity": self.is_physical_identity,
            "provenance": self.provenance,
        }


class TemporalIdentityTracker:
    """Associates observations across sequential video frames with velocity prediction and appearance cues."""

    def __init__(
        self,
        max_missed_frames: int = 5,
        min_iou_match: float = 0.15,
        max_centroid_shift_px: float = 350.0,
    ) -> None:
        self.max_missed_frames = max_missed_frames
        self.min_iou_match = min_iou_match
        self.max_centroid_shift_px = max_centroid_shift_px
        self.active_tracks: dict[str, TemporalTrack] = {}
        self.closed_tracks: dict[str, TemporalTrack] = {}
        self._next_track_idx = 1

    def process_frame_observations(
        self,
        frame_index: int,
        timestamp_sec: float,
        observations: Sequence[Any],
    ) -> dict[str, str]:
        """Matches current frame observations with active tracks; returns {obs_id: track_id}."""
        from engine.multiengine.spatial_intelligence.detection_pipeline import compute_box_iou

        obs_to_track: dict[str, str] = {}
        matched_tracks: set[str] = set()

        # Score all candidate pairs (obs, track)
        candidate_matches: list[tuple[float, Any, TemporalTrack]] = []

        for obs in observations:
            obs_box = tuple(obs.box_2d)
            obs_cls = obs.class_name
            obs_cx = 0.5 * (obs_box[0] + obs_box[2])
            obs_cy = 0.5 * (obs_box[1] + obs_box[3])
            obs_app = tuple(getattr(obs, "appearance_features", ())) or tuple(getattr(obs, "attributes", {}).get("appearance_features", ()))

            for track in self.active_tracks.values():
                if not are_semantically_compatible(obs_cls, track.class_name):
                    continue

                # Compute motion-compensated centroid distance
                dt = max(0.001, timestamp_sec - (track.timestamps_sec[-1] if track.timestamps_sec else timestamp_sec))
                tcx, tcy = track.predicted_centroid(dt)
                c_dist = math.hypot(obs_cx - tcx, obs_cy - tcy)

                # 2D IoU with previous box
                iou = compute_box_iou(obs_box, track.last_box_2d)

                # Appearance cosine similarity if available
                app_score = 0.5
                if obs_app and track.last_appearance and len(obs_app) == len(track.last_appearance):
                    u = np.asarray(obs_app, dtype=float)
                    v = np.asarray(track.last_appearance, dtype=float)
                    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                    if nu > 1e-6 and nv > 1e-6:
                        app_score = float(np.clip(np.dot(u, v) / (nu * nv), 0.0, 1.0))

                # Allow match if either IoU meets floor or predicted centroid distance is within range
                if iou >= self.min_iou_match or c_dist <= self.max_centroid_shift_px:
                    dist_score = max(0.0, 1.0 - (c_dist / self.max_centroid_shift_px))
                    match_score = 0.45 * iou + 0.35 * dist_score + 0.20 * app_score
                    candidate_matches.append((match_score, obs, track))

        candidate_matches.sort(key=lambda x: x[0], reverse=True)
        matched_obs_ids: set[str] = set()

        for score, obs, track in candidate_matches:
            obs_id = getattr(obs, "observation_id", getattr(obs, "box_id", getattr(obs, "detection_id", getattr(obs, "entity_id", ""))))
            if obs_id in matched_obs_ids or track.track_id in matched_tracks:
                continue

            obs_to_track[obs_id] = track.track_id
            matched_obs_ids.add(obs_id)
            matched_tracks.add(track.track_id)

            # Update velocity
            dt = max(0.001, timestamp_sec - (track.timestamps_sec[-1] if track.timestamps_sec else timestamp_sec))
            new_cx = 0.5 * (obs.box_2d[0] + obs.box_2d[2])
            new_cy = 0.5 * (obs.box_2d[1] + obs.box_2d[3])
            old_cx, old_cy = track.centroid
            vx = (new_cx - old_cx) / dt
            vy = (new_cy - old_cy) / dt
            # Smooth velocity (EMA)
            alpha = 0.6
            smoothed_vx = alpha * vx + (1.0 - alpha) * track.velocity_2d[0]
            smoothed_vy = alpha * vy + (1.0 - alpha) * track.velocity_2d[1]

            track.observation_ids.append(obs_id)
            track.frame_indices.append(frame_index)
            track.timestamps_sec.append(timestamp_sec)
            track.last_box_2d = tuple(obs.box_2d)
            track.velocity_2d = (smoothed_vx, smoothed_vy)
            app = tuple(getattr(obs, "appearance_features", ())) or tuple(getattr(obs, "attributes", {}).get("appearance_features", ()))
            if app:
                track.last_appearance = app
            track.age_frames += 1
            track.missed_frames = 0

        # Unmatched observations spawn new tracks
        for obs in observations:
            obs_id = getattr(obs, "observation_id", getattr(obs, "box_id", getattr(obs, "detection_id", getattr(obs, "entity_id", ""))))
            if obs_id not in matched_obs_ids:
                new_tid = f"track_{self._next_track_idx:04d}"
                self._next_track_idx += 1
                app = tuple(getattr(obs, "appearance_features", ())) or tuple(getattr(obs, "attributes", {}).get("appearance_features", ()))
                new_track = TemporalTrack(
                    track_id=new_tid,
                    class_name=obs.class_name,
                    active=True,
                    observation_ids=[obs_id],
                    frame_indices=[frame_index],
                    timestamps_sec=[timestamp_sec],
                    last_box_2d=tuple(obs.box_2d),
                    last_appearance=app,
                    velocity_2d=(0.0, 0.0),
                    age_frames=1,
                    missed_frames=0,
                    is_physical_identity=False,
                    termination_reason="active",
                    provenance="temporal_observation_tracking",
                )
                self.active_tracks[new_tid] = new_track
                obs_to_track[obs_id] = new_tid

        # Update missed frames for unobserved tracks (coasting)
        to_close: list[str] = []
        for tid, track in self.active_tracks.items():
            if tid not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    track.active = False
                    track.termination_reason = "missed_frames_limit"
                    to_close.append(tid)

        for tid in to_close:
            self.closed_tracks[tid] = self.active_tracks.pop(tid)

        return obs_to_track


class MultiViewIntelligenceEngine:
    """Production Multi-View Intelligence and Association Engine."""

    def __init__(
        self,
        max_ray_distance_m: float = 0.35,
        max_reproj_err_px: float = 100.0,
        min_match_score: float = 0.25,
    ) -> None:
        self.max_ray_distance_m = max_ray_distance_m
        self.max_reproj_err_px = max_reproj_err_px
        self.min_match_score = min_match_score

    def associate_candidates(
        self,
        candidates: Sequence[CrossViewRayCandidate],
    ) -> tuple[tuple[PhysicalMergedEntity, ...], list[PairScore], dict[str, int]]:
        """Associates cross-view ray candidates into unique physical merged entities."""
        decisions: list[PairScore] = []
        refusal_counts: Counter[str] = Counter()
        n = len(candidates)
        obs_by_id = {c.observation_id: c for c in candidates}

        # Build pairwise compatibility matrix
        qualifying_edges: list[tuple[float, str, str, tuple[float, float, float], float, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                c_a, c_b = candidates[i], candidates[j]
                ps = evaluate_cross_view_pair(
                    c_a,
                    c_b,
                    max_ray_distance_m=self.max_ray_distance_m,
                    max_reproj_err_px=self.max_reproj_err_px,
                )
                decisions.append(ps)

                if ps.outcome == AssociationOutcome.MERGED and ps.score >= self.min_match_score:
                    pos = ps.triangulated_position or (0.0, 0.0, 0.0)
                    r_dist = ps.ray_distance or 0.0
                    rep_err = ps.reprojection_err_px or 0.0
                    qualifying_edges.append((ps.score, ps.obs_a_id, ps.obs_b_id, pos, r_dist, rep_err))
                else:
                    refusal_counts[ps.reason] += 1

        # Sort edges by descending score for greedy one-to-one clustering
        qualifying_edges.sort(key=lambda x: x[0], reverse=True)

        # Disjoint-set clustering with strict cluster-wide invariants
        parent = {c.observation_id: c.observation_id for c in candidates}
        cluster_members: dict[str, set[str]] = {c.observation_id: {c.observation_id} for c in candidates}
        cluster_sources: dict[str, set[str]] = {c.observation_id: {c.source_image_id} for c in candidates}
        cluster_scale_states: dict[str, set[str]] = {c.observation_id: {c.scale_state.lower()} for c in candidates}
        cluster_units: dict[str, set[str]] = {c.observation_id: {c.scale_units.lower()} for c in candidates}
        cluster_frames: dict[str, set[str]] = {c.observation_id: {c.coordinate_frame} for c in candidates}
        cluster_anchor_refs: dict[str, set[str]] = {
            c.observation_id: ({c.physical_anchor_ref} if c.physical_anchor_ref else set()) for c in candidates
        }
        cluster_positions: dict[str, list[tuple[float, float, float]]] = {
            c.observation_id: [c.lifted_entity.position_m] if (c.lifted_entity and c.lifted_entity.position_m is not None) else [] for c in candidates
        }
        cluster_ray_residuals: dict[str, list[float]] = {c.observation_id: [] for c in candidates}
        cluster_reproj_residuals: dict[str, list[float]] = {c.observation_id: [] for c in candidates}
        cluster_competing_hypotheses: dict[str, list[dict[str, Any]]] = defaultdict(list)

        def find(u: str) -> str:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        accepted_merges = 0
        for score, id_a, id_b, pos_mid, r_dist, rep_err in qualifying_edges:
            root_a = find(id_a)
            root_b = find(id_b)
            if root_a == root_b:
                continue

            # 1. Hard constraint: a physical cluster cannot have two observations from the same image
            if cluster_sources[root_a] & cluster_sources[root_b]:
                refusal_counts["transitive_same_image_conflict"] += 1
                hyp = {
                    "rejected_edge": (id_a, id_b),
                    "score": score,
                    "reason": "transitive_same_image_conflict",
                    "conflicting_sources": list(cluster_sources[root_a] & cluster_sources[root_b]),
                }
                cluster_competing_hypotheses[root_a].append(hyp)
                cluster_competing_hypotheses[root_b].append(hyp)
                continue

            # 2. Hard constraint: scale compatibility across the entire proposed cluster
            scales_a = cluster_scale_states[root_a]
            scales_b = cluster_scale_states[root_b]
            scale_conflict = False
            for sa in scales_a:
                for sb in scales_b:
                    if not are_scale_states_compatible(sa, sb):
                        scale_conflict = True
                        break
                if scale_conflict:
                    break

            if scale_conflict:
                refusal_counts["transitive_scale_incompatibility"] += 1
                hyp = {
                    "rejected_edge": (id_a, id_b),
                    "score": score,
                    "reason": "transitive_scale_incompatibility",
                    "scales_a": list(scales_a),
                    "scales_b": list(scales_b),
                }
                cluster_competing_hypotheses[root_a].append(hyp)
                cluster_competing_hypotheses[root_b].append(hyp)
                continue

            # 3. Hard constraint: coordinate frame compatibility across entire cluster
            if cluster_frames[root_a] != cluster_frames[root_b]:
                refusal_counts["transitive_coordinate_frame_mismatch"] += 1
                hyp = {
                    "rejected_edge": (id_a, id_b),
                    "score": score,
                    "reason": "transitive_coordinate_frame_mismatch",
                    "frames_a": list(cluster_frames[root_a]),
                    "frames_b": list(cluster_frames[root_b]),
                }
                cluster_competing_hypotheses[root_a].append(hyp)
                cluster_competing_hypotheses[root_b].append(hyp)
                continue

            # 4. Hard constraint: units compatibility across entire cluster
            if cluster_units[root_a] != cluster_units[root_b]:
                refusal_counts["transitive_units_mismatch"] += 1
                hyp = {
                    "rejected_edge": (id_a, id_b),
                    "score": score,
                    "reason": "transitive_units_mismatch",
                    "units_a": list(cluster_units[root_a]),
                    "units_b": list(cluster_units[root_b]),
                }
                cluster_competing_hypotheses[root_a].append(hyp)
                cluster_competing_hypotheses[root_b].append(hyp)
                continue

            # 5. Hard constraint: physical anchor consistency for metric_anchored
            if "metric_anchored" in scales_a and "metric_anchored" in scales_b:
                if cluster_anchor_refs[root_a] != cluster_anchor_refs[root_b]:
                    refusal_counts["transitive_mixed_anchor_references"] += 1
                    hyp = {
                        "rejected_edge": (id_a, id_b),
                        "score": score,
                        "reason": "transitive_mixed_anchor_references",
                        "anchors_a": list(cluster_anchor_refs[root_a]),
                        "anchors_b": list(cluster_anchor_refs[root_b]),
                    }
                    cluster_competing_hypotheses[root_a].append(hyp)
                    cluster_competing_hypotheses[root_b].append(hyp)
                    continue

            # Merge root_b into root_a
            parent[root_b] = root_a
            cluster_members[root_a].update(cluster_members[root_b])
            cluster_sources[root_a].update(cluster_sources[root_b])
            cluster_scale_states[root_a].update(cluster_scale_states[root_b])
            cluster_units[root_a].update(cluster_units[root_b])
            cluster_frames[root_a].update(cluster_frames[root_b])
            cluster_anchor_refs[root_a].update(cluster_anchor_refs[root_b])
            cluster_positions[root_a].append(pos_mid)
            cluster_positions[root_a].extend(cluster_positions[root_b])
            cluster_ray_residuals[root_a].append(r_dist)
            cluster_ray_residuals[root_a].extend(cluster_ray_residuals[root_b])
            cluster_reproj_residuals[root_a].append(rep_err)
            cluster_reproj_residuals[root_a].extend(cluster_reproj_residuals[root_b])
            cluster_competing_hypotheses[root_a].extend(cluster_competing_hypotheses[root_b])
            accepted_merges += 1

        # Post-pass: False-Split Reduction
        # If two distinct clusters have zero image collision, compatible class, and close centroids, merge them.
        unique_roots = list(set(find(c.observation_id) for c in candidates))
        for r_i in range(len(unique_roots)):
            for r_j in range(r_i + 1, len(unique_roots)):
                root_i = find(unique_roots[r_i])
                root_j = find(unique_roots[r_j])
                if root_i == root_j:
                    continue
                # Check for image collision
                if cluster_sources[root_i] & cluster_sources[root_j]:
                    continue

                # Check scale compatibility
                scales_i = cluster_scale_states[root_i]
                scales_j = cluster_scale_states[root_j]
                if any(not are_scale_states_compatible(si, sj) for si in scales_i for sj in scales_j):
                    continue
                if cluster_frames[root_i] != cluster_frames[root_j] or cluster_units[root_i] != cluster_units[root_j]:
                    continue

                pts_i = cluster_positions.get(root_i, [])
                pts_j = cluster_positions.get(root_j, [])
                if not pts_i or not pts_j:
                    continue

                valid_pts_i = [p for p in pts_i if p is not None]
                valid_pts_j = [p for p in pts_j if p is not None]
                if not valid_pts_i or not valid_pts_j:
                    continue

                pos_i = np.median(np.array(valid_pts_i), axis=0)
                pos_j = np.median(np.array(valid_pts_j), axis=0)
                dist_ij = float(np.linalg.norm(pos_i - pos_j))

                # Check semantic compatibility
                cand_i = obs_by_id[root_i]
                cand_j = obs_by_id[root_j]
                if not are_semantically_compatible(cand_i.class_name, cand_j.class_name):
                    continue

                scale_common = next(iter(scales_i))
                can_merge = False
                if scale_common == "metric_anchored":
                    if cluster_anchor_refs[root_i] == cluster_anchor_refs[root_j] and dist_ij <= 0.30:
                        can_merge = True
                elif scale_common == "synthetic_scale_aligned":
                    if dist_ij <= 0.30:
                        can_merge = True
                elif scale_common == "sfm_gauge":
                    joint_members = [obs_by_id[m_id] for m_id in (cluster_members[root_i] | cluster_members[root_j])]
                    p_test, _, rep_test, _, _ = triangulate_multiview_rays(joint_members)
                    if p_test is not None and rep_test <= self.max_reproj_err_px:
                        can_merge = True

                if can_merge:
                    # Merge root_j into root_i (false split repaired)
                    parent[root_j] = root_i
                    cluster_members[root_i].update(cluster_members[root_j])
                    cluster_sources[root_i].update(cluster_sources[root_j])
                    cluster_scale_states[root_i].update(cluster_scale_states[root_j])
                    cluster_units[root_i].update(cluster_units[root_j])
                    cluster_frames[root_i].update(cluster_frames[root_j])
                    cluster_anchor_refs[root_i].update(cluster_anchor_refs[root_j])
                    cluster_positions[root_i].extend(cluster_positions[root_j])
                    cluster_ray_residuals[root_i].extend(cluster_ray_residuals[root_j])
                    cluster_reproj_residuals[root_i].extend(cluster_reproj_residuals[root_j])
                    cluster_competing_hypotheses[root_i].extend(cluster_competing_hypotheses[root_j])
                    refusal_counts["false_split_repaired"] += 1

        # Form Reconciled Physical Entities
        grouped_members: dict[str, list[str]] = defaultdict(list)
        for c in candidates:
            grouped_members[find(c.observation_id)].append(c.observation_id)

        merged_entities: list[PhysicalMergedEntity] = []
        for root_id, member_ids in grouped_members.items():
            members = [obs_by_id[m_id] for m_id in member_ids]
            labels = [m.class_name for m in members]
            primary_cls = Counter(labels).most_common(1)[0][0]
            sources = tuple(sorted(set(m.source_image_id for m in members)))
            is_multi = len(sources) >= 2

            # Compute reconciled position using multi-view triangulation when available
            fused_pos: tuple[float, float, float] | None = None
            fused_cov: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None
            mean_ray_res = 0.0
            mean_rep_res = 0.0
            triangulation_meta: dict[str, Any] = {}
            residual_units = "relative"
            covariance_units = "relative^2"

            if is_multi:
                p_tri, r_res, rep_res, cov_mat, meta = triangulate_multiview_rays(members)
                if p_tri is not None:
                    fused_pos = p_tri
                    mean_ray_res = r_res
                    mean_rep_res = rep_res
                    triangulation_meta = meta
                    residual_units = meta.get("residual_units", "relative")
                    covariance_units = meta.get("covariance_units", "relative^2")
                    if cov_mat is not None:
                        fused_cov = (
                            (float(cov_mat[0, 0]), float(cov_mat[0, 1]), float(cov_mat[0, 2])),
                            (float(cov_mat[1, 0]), float(cov_mat[1, 1]), float(cov_mat[1, 2])),
                            (float(cov_mat[2, 0]), float(cov_mat[2, 1]), float(cov_mat[2, 2])),
                        )

            # If multi-ray triangulation was not possible, check lifted observations or pairwise pools
            if fused_pos is None:
                lifted_positions = [
                    m.lifted_entity.position_m for m in members
                    if m.lifted_entity and m.lifted_entity.position_m is not None
                ]
                if lifted_positions:
                    arr = np.array(lifted_positions, dtype=np.float64)
                    fused_pos = tuple(float(c) for c in np.median(arr, axis=0))
                    for m in members:
                        if m.lifted_entity and m.lifted_entity.covariance_xyz:
                            fused_cov = m.lifted_entity.covariance_xyz
                            residual_units = m.scale_units
                            covariance_units = f"{m.scale_units}^2"
                            break
                elif cluster_positions.get(root_id):
                    valid_c_pos = [p for p in cluster_positions[root_id] if p is not None]
                    if valid_c_pos:
                        arr = np.array(valid_c_pos, dtype=np.float64)
                        fused_pos = tuple(float(c) for c in np.median(arr, axis=0))
                    else:
                        fused_pos = None
                else:
                    # ZERO FABRICATION: Missing 3D position remains None
                    fused_pos = None

            # Dimensions: observed only, never taxonomy min as reconstructed measurement
            dims_list = [
                m.lifted_entity.dimensions_m for m in members
                if m.lifted_entity and m.lifted_entity.dimensions_m is not None
            ]
            if dims_list:
                d_arr = np.array(dims_list, dtype=np.float64)
                fused_dims: tuple[float, float, float] | None = tuple(float(d) for d in np.median(d_arr, axis=0))
                dim_status = "DIRECT" if any(m.lifted_entity.mask_pixels_count > 50 for m in members if m.lifted_entity) else "ESTIMATED"
            else:
                fused_dims = None
                dim_status = "UNKNOWN"

            # Orientation & Normal
            orient = (1.0, 0.0, 0.0, 0.0)
            norm = None
            for m in members:
                if m.lifted_entity and m.lifted_entity.surface_normal:
                    norm = m.lifted_entity.surface_normal
                    orient = m.lifted_entity.orientation_quat_wxyz
                    break

            # Scale state derivation: strict preservation across members, never default to synthetic
            member_scales = [m.scale_state.lower() for m in members]
            if any(s == "metric_anchored" for s in member_scales):
                scale_state = "metric_anchored"
                scale_units = "m"
                is_metric = True
                anchor_ref = next((m.physical_anchor_ref for m in members if m.physical_anchor_ref), None)
            elif any(s == "synthetic_scale_aligned" for s in member_scales):
                scale_state = "synthetic_scale_aligned"
                scale_units = "synthetic_m"
                is_metric = False
                anchor_ref = None
            elif any(s == "sfm_gauge" for s in member_scales):
                scale_state = "sfm_gauge"
                scale_units = "sfm_units"
                is_metric = False
                anchor_ref = None
            else:
                scale_state = "relative_depth"
                scale_units = "relative"
                is_metric = False
                anchor_ref = None

            if not mean_ray_res and cluster_ray_residuals[root_id]:
                mean_ray_res = float(np.mean(cluster_ray_residuals[root_id]))
            if not mean_rep_res and cluster_reproj_residuals[root_id]:
                mean_rep_res = float(np.mean(cluster_reproj_residuals[root_id]))

            # Confidence boosted by multi-view evidence
            base_conf = 0.6 if is_multi else 0.45
            calibrated_conf = min(0.95, base_conf + 0.15 * (len(sources) - 1))
            if mean_rep_res > 50.0:
                calibrated_conf = max(0.2, calibrated_conf - 0.2)

            merged_entities.append(PhysicalMergedEntity(
                canonical_entity_id=f"phys_{root_id}",
                primary_class=primary_cls,
                member_observation_ids=tuple(member_ids),
                sources=sources,
                position=fused_pos,
                dimensions=fused_dims,
                orientation_quat_wxyz=orient,
                surface_normal=norm,
                confidence=calibrated_conf,
                reprojection_residual_px=mean_rep_res,
                triangulation_residual=mean_ray_res,
                residual_units=residual_units,
                covariance_units=covariance_units,
                is_multiview=is_multi,
                scale_state=scale_state,
                scale_units=scale_units,
                is_physically_metric=is_metric,
                coordinate_frame=members[0].coordinate_frame,
                physical_anchor_ref=anchor_ref,
                dimension_status=dim_status,
                covariance_xyz=fused_cov,
                competing_hypotheses=tuple(cluster_competing_hypotheses[root_id]),
                details={
                    "view_count": len(sources),
                    "observation_count": len(member_ids),
                    "triangulation_meta": triangulation_meta,
                },
            ))

        return tuple(merged_entities), decisions, dict(refusal_counts)
