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
    resolve_taxonomy_class,
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


@dataclass(frozen=True)
class PairScore:
    """Scored compatibility between two cross-view observations."""

    obs_a_id: str
    obs_b_id: str
    score: float
    outcome: AssociationOutcome
    reason: str
    triangulated_position: tuple[float, float, float] | None = None
    ray_distance_m: float | None = None
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
    position_m: tuple[float, float, float]
    dimensions_m: tuple[float, float, float]
    orientation_quat_wxyz: tuple[float, float, float, float]
    surface_normal: tuple[float, float, float] | None
    confidence: float
    reprojection_residual_px: float
    ray_triangulation_residual_m: float
    is_multiview: bool
    scale_state: str = "SYNTHETIC_SCALE_ALIGNED"
    scale_units: str = "METRES"
    is_physically_metric: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_entity_id": self.canonical_entity_id,
            "primary_class": self.primary_class,
            "member_observation_ids": list(self.member_observation_ids),
            "sources": list(self.sources),
            "position_m": [round(c, 4) for c in self.position_m],
            "dimensions_m": [round(d, 4) for d in self.dimensions_m],
            "orientation_quat_wxyz": [round(q, 4) for q in self.orientation_quat_wxyz],
            "surface_normal": [round(n, 4) for n in self.surface_normal] if self.surface_normal else None,
            "confidence": round(self.confidence, 4),
            "reprojection_residual_px": round(self.reprojection_residual_px, 2),
            "ray_triangulation_residual_m": round(self.ray_triangulation_residual_m, 4),
            "is_multiview": self.is_multiview,
            "scale_state": self.scale_state,
            "scale_units": self.scale_units,
            "is_physically_metric": self.is_physically_metric,
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
) -> CrossViewRayCandidate:
    """Builds a CrossViewRayCandidate from a 2D bounding box and calibrated camera."""
    u_mid = 0.5 * (float(box_2d[0]) + float(box_2d[2]))
    v_mid = 0.5 * (float(box_2d[1]) + float(box_2d[3]))

    # Direction in camera coordinates (Z=1.0)
    fx, fy = camera.intrinsics.fx, camera.intrinsics.fy
    cx, cy = camera.intrinsics.cx, camera.intrinsics.cy
    dir_cam = np.array([(u_mid - cx) / fx, (v_mid - cy) / fy, 1.0], dtype=np.float64)
    dir_cam /= np.linalg.norm(dir_cam)

    # Transform direction to world coordinates via R_wc
    q = camera.extrinsics.rotation
    R_wc = np.array([
        [1.0 - 2.0 * (q.y ** 2 + q.z ** 2), 2.0 * (q.x * q.y - q.z * q.w), 2.0 * (q.x * q.z + q.y * q.w)],
        [2.0 * (q.x * q.y + q.z * q.w), 1.0 - 2.0 * (q.x ** 2 + q.z ** 2), 2.0 * (q.y * q.z - q.x * q.w)],
        [2.0 * (q.x * q.z - q.y * q.w), 2.0 * (q.y * q.z + q.x * q.w), 1.0 - 2.0 * (q.x ** 2 + q.y ** 2)],
    ], dtype=np.float64)

    dir_world = R_wc @ dir_cam
    dir_world /= np.linalg.norm(dir_world)

    cam_pos = np.array([camera.extrinsics.position.x, camera.extrinsics.position.y, camera.extrinsics.position.z], dtype=np.float64)

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
    )


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

    # 2. Semantic compatibility
    if not are_semantically_compatible(ray_a.class_name, ray_b.class_name):
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="semantic_incompatible",
            basis={"class_a": ray_a.class_name, "class_b": ray_b.class_name},
        )

    # 3. Baseline check: cameras must not be co-located
    cam_baseline = float(np.linalg.norm(ray_a.origin - ray_b.origin))
    if cam_baseline < 0.05:
        # Near-zero baseline, cannot triangulate rays
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="zero_camera_baseline",
            basis={"baseline_m": round(cam_baseline, 4)},
        )

    # 4. Ray-Ray CPA and Triangulation
    dist_ray, s_a, t_b, p_a, p_b = compute_ray_cpa(
        ray_a.origin, ray_a.direction, ray_b.origin, ray_b.direction
    )

    # Rays must intersect in FRONT of both cameras (positive depth parameter)
    if s_a <= 0.1 or t_b <= 0.1:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="ray_intersection_behind_camera",
            basis={"s_depth_a": round(s_a, 3), "t_depth_b": round(t_b, 3)},
        )

    # Maximum distance bound: object cannot be 50 meters away indoors
    if s_a > 30.0 or t_b > 30.0:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="excessive_scene_depth",
            basis={"s_depth_a": round(s_a, 2), "t_depth_b": round(t_b, 2)},
        )

    # Adaptive ray distance tolerance based on scene depth (angular tolerance ~ 2 degrees)
    mean_dist = 0.5 * (s_a + t_b)
    adaptive_tol = max(max_ray_distance_m, 0.05 * mean_dist)

    if dist_ray > adaptive_tol:
        return PairScore(
            obs_a_id=ray_a.observation_id,
            obs_b_id=ray_b.observation_id,
            score=0.0,
            outcome=AssociationOutcome.REFUSED,
            reason="excessive_ray_distance",
            ray_distance_m=dist_ray,
            basis={"dist_ray_m": round(dist_ray, 4), "tolerance_m": round(adaptive_tol, 4)},
        )

    # Triangulated 3D position (midpoint)
    p_mid = 0.5 * (p_a + p_b)
    p_mid_tuple = (float(p_mid[0]), float(p_mid[1]), float(p_mid[2]))

    # 5. Cross-View Reprojection Verification
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
            ray_distance_m=dist_ray,
            reprojection_err_px=mean_reproj,
            basis={"err_a_px": round(err_a, 2), "err_b_px": round(err_b, 2), "mean_err_px": round(mean_reproj, 2)},
        )

    # 6. Appearance Similarity (if available)
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
                    appearance_sim=app_sim,
                    basis={"sim": round(app_sim, 4)},
                )

    # Composite Score Calculation
    score_ray = math.exp(-0.5 * (dist_ray / max(0.05, adaptive_tol * 0.5)) ** 2)
    score_reproj = math.exp(-0.5 * (mean_reproj / (max_reproj_err_px * 0.5)) ** 2)

    composite = (score_ray ** 0.6) * (score_reproj ** 0.4)
    if app_sim is not None:
        composite *= (0.7 + 0.3 * max(0.0, app_sim))

    return PairScore(
        obs_a_id=ray_a.observation_id,
        obs_b_id=ray_b.observation_id,
        score=round(composite, 4),
        outcome=AssociationOutcome.MERGED,
        reason="epipolar_reprojection_consensus",
        triangulated_position=p_mid_tuple,
        ray_distance_m=dist_ray,
        reprojection_err_px=mean_reproj,
        appearance_sim=app_sim,
        basis={
            "score_ray": round(score_ray, 4),
            "score_reproj": round(score_reproj, 4),
            "dist_ray_m": round(dist_ray, 4),
            "mean_reproj_px": round(mean_reproj, 2),
            "s_a_m": round(s_a, 3),
            "t_b_m": round(t_b, 3),
        },
    )


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
                    r_dist = ps.ray_distance_m or 0.0
                    rep_err = ps.reprojection_err_px or 0.0
                    qualifying_edges.append((ps.score, ps.obs_a_id, ps.obs_b_id, pos, r_dist, rep_err))
                else:
                    refusal_counts[ps.reason] += 1

        # Sort edges by descending score for greedy one-to-one clustering
        qualifying_edges.sort(key=lambda x: x[0], reverse=True)

        # Disjoint-set clustering with strict same-image mutual exclusion
        parent = {c.observation_id: c.observation_id for c in candidates}
        cluster_sources = {c.observation_id: {c.source_image_id} for c in candidates}
        cluster_positions: dict[str, list[tuple[float, float, float]]] = {
            c.observation_id: [c.lifted_entity.position_m] if c.lifted_entity else [] for c in candidates
        }
        cluster_ray_residuals: dict[str, list[float]] = {c.observation_id: [] for c in candidates}
        cluster_reproj_residuals: dict[str, list[float]] = {c.observation_id: [] for c in candidates}

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

            # Hard constraint: a physical cluster cannot have two observations from the same image
            if cluster_sources[root_a] & cluster_sources[root_b]:
                refusal_counts["transitive_same_image_conflict"] += 1
                continue

            # Merge root_b into root_a
            parent[root_b] = root_a
            cluster_sources[root_a].update(cluster_sources[root_b])
            cluster_positions[root_a].append(pos_mid)
            cluster_positions[root_a].extend(cluster_positions[root_b])
            cluster_ray_residuals[root_a].append(r_dist)
            cluster_ray_residuals[root_a].extend(cluster_ray_residuals[root_b])
            cluster_reproj_residuals[root_a].append(rep_err)
            cluster_reproj_residuals[root_a].extend(cluster_reproj_residuals[root_b])
            accepted_merges += 1

        # Post-pass: False-Split Reduction
        # If two distinct clusters have zero image collision, compatible class, and close 3D centroids, merge them.
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

                pts_i = cluster_positions.get(root_i, [])
                pts_j = cluster_positions.get(root_j, [])
                if not pts_i or not pts_j:
                    continue

                pos_i = np.median(np.array(pts_i), axis=0)
                pos_j = np.median(np.array(pts_j), axis=0)
                dist_ij = float(np.linalg.norm(pos_i - pos_j))

                # Check semantic compatibility
                cand_i = obs_by_id[root_i]
                cand_j = obs_by_id[root_j]
                if dist_ij <= 0.30 and are_semantically_compatible(cand_i.class_name, cand_j.class_name):
                    # Merge root_j into root_i (false split repaired)
                    parent[root_j] = root_i
                    cluster_sources[root_i].update(cluster_sources[root_j])
                    cluster_positions[root_i].extend(cluster_positions[root_j])
                    cluster_ray_residuals[root_i].extend(cluster_ray_residuals[root_j])
                    cluster_reproj_residuals[root_i].extend(cluster_reproj_residuals[root_j])
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

            # Compute reconciled position
            pts_pool = cluster_positions.get(root_id, [])
            if pts_pool:
                arr = np.array(pts_pool, dtype=np.float64)
                fused_pos = tuple(float(c) for c in np.median(arr, axis=0))
            else:
                first_m = members[0]
                p_approx = first_m.origin + 2.0 * first_m.direction
                fused_pos = (float(p_approx[0]), float(p_approx[1]), float(p_approx[2]))

            # Dimensions
            dims_list = [m.lifted_entity.dimensions_m for m in members if m.lifted_entity]
            if dims_list:
                d_arr = np.array(dims_list, dtype=np.float64)
                fused_dims = tuple(float(d) for d in np.median(d_arr, axis=0))
            else:
                tax_node = resolve_taxonomy_class(primary_cls)
                fused_dims = tax_node.min_dimensions_m

            # Orientation & Normal & Scale contract
            orient = (1.0, 0.0, 0.0, 0.0)
            norm = None
            scale_state = "SYNTHETIC_SCALE_ALIGNED"
            is_metric = False
            for m in members:
                if m.lifted_entity:
                    scale_state = m.lifted_entity.scale_state
                    is_metric = m.lifted_entity.is_physically_metric
                    if m.lifted_entity.surface_normal:
                        norm = m.lifted_entity.surface_normal
                        orient = m.lifted_entity.orientation_quat_wxyz
                        break

            mean_ray_res = float(np.mean(cluster_ray_residuals[root_id])) if cluster_ray_residuals[root_id] else 0.0
            mean_rep_res = float(np.mean(cluster_reproj_residuals[root_id])) if cluster_reproj_residuals[root_id] else 0.0

            # Confidence boosted by multi-view evidence
            base_conf = 0.6 if is_multi else 0.45
            calibrated_conf = min(0.95, base_conf + 0.15 * (len(sources) - 1))

            merged_entities.append(PhysicalMergedEntity(
                canonical_entity_id=f"phys_{root_id}",
                primary_class=primary_cls,
                member_observation_ids=tuple(member_ids),
                sources=sources,
                position_m=fused_pos,
                dimensions_m=fused_dims,
                orientation_quat_wxyz=orient,
                surface_normal=norm,
                confidence=calibrated_conf,
                reprojection_residual_px=mean_rep_res,
                ray_triangulation_residual_m=mean_ray_res,
                is_multiview=is_multi,
                scale_state=scale_state,
                scale_units="METRES",
                is_physically_metric=is_metric,
                details={
                    "view_count": len(sources),
                    "observation_count": len(member_ids),
                },
            ))

        return tuple(merged_entities), decisions, dict(refusal_counts)
