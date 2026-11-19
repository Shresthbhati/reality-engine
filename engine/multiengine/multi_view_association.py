"""Multi-View Physical Identity Association and Geometric Reconciliation.

File: engine/multiengine/multi_view_association.py

Implements physical object identity association across multi-view photographs:
1. Constructs Candidate records from 3D lifted entity observations.
2. Computes cross-view geometric, epipolar reprojection, and semantic compatibility pair scores.
3. Enforces strict physical constraints via `engine.multiengine.association.associate`:
   - HARD REFUSAL: Two detections from the same photograph CAN NEVER merge into the same physical entity.
   - HARD REFUSAL: Incompatible semantic labels (e.g. window vs tree) CAN NEVER merge.
   - HARD REFUSAL: Whole-part and part-parent hierarchical consistency.
4. Generates an immutable Decision Ledger detailing every accepted merge and refusal reason.
5. Reconciles multi-view geometry (weighted 3D centroid, bounding box dimensions, and reprojection residuals).
6. Computes composite evidence-calibrated confidence reflecting view count and reprojection consistency
   without discarding raw detector confidence provenance.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.math import Vec3
from engine.multiengine.association import (
    MERGED,
    REFUSED,
    AssociatedEntity,
    AssociationResult,
    Candidate,
    Decision,
    PairScore,
    associate,
    ontology_label_compat,
)
from world_ir.camera import PinholeCamera


@dataclass(frozen=True)
class MultiViewEntityObservation:
    """An observation of a detected and 3D-lifted entity in a specific camera view."""

    observation_id: str  # e.g. "obs_P1180141.JPG_det_01"
    source_id: str  # e.g. "P1180141.JPG"
    class_name: str  # e.g. "window"
    position: tuple[float, float, float]  # (x, y, z) in canonical world frame
    dimensions: tuple[float, float, float]  # (w, d, h) in meters or SfM units
    orientation: tuple[float, float, float, float] | None = None  # (w, x, y, z)
    confidence: float = 0.5  # raw model detector confidence
    box_2d: Sequence[float] = ()  # [u_min, v_min, u_max, v_max]
    camera: PinholeCamera | None = None
    parent_observation_id: str | None = None
    is_part: bool = False
    custom_properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconciledPhysicalEntity:
    """A unique physical entity formed from associated multi-view observations."""

    entity_id: str  # Derived from AssociatedEntity.entity_key
    primary_class: str
    member_observation_ids: tuple[str, ...]
    sources: tuple[str, ...]  # Distinct photo views
    observation_count: int
    is_multiview: bool
    position: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    orientation: tuple[float, float, float, float] | None
    mean_detector_confidence: float
    calibrated_confidence: float
    reprojection_residual_px: float | None
    parent_entity_id: str | None = None
    is_part: bool = False
    custom_properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "primary_class": self.primary_class,
            "member_observation_ids": list(self.member_observation_ids),
            "sources": list(self.sources),
            "observation_count": self.observation_count,
            "is_multiview": self.is_multiview,
            "position": list(self.position),
            "dimensions": list(self.dimensions),
            "orientation": list(self.orientation) if self.orientation else None,
            "mean_detector_confidence": round(self.mean_detector_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "reprojection_residual_px": round(self.reprojection_residual_px, 2) if self.reprojection_residual_px is not None else None,
            "parent_entity_id": self.parent_entity_id,
            "is_part": self.is_part,
            "custom_properties": self.custom_properties,
        }


@dataclass(frozen=True)
class MultiViewAssociationSummary:
    """Telemetry and benchmark metrics for multi-view association."""

    total_input_observations: int
    unique_physical_entities_count: int
    single_view_entities_count: int
    multiview_merged_entities_count: int
    max_views_per_entity: int
    accepted_merges_count: int
    refused_pairs_count: int
    refusal_reasons_breakdown: dict[str, int]
    class_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_observations": self.total_input_observations,
            "unique_physical_entities_count": self.unique_physical_entities_count,
            "single_view_entities_count": self.single_view_entities_count,
            "multiview_merged_entities_count": self.multiview_merged_entities_count,
            "max_views_per_entity": self.max_views_per_entity,
            "accepted_merges_count": self.accepted_merges_count,
            "refused_pairs_count": self.refused_pairs_count,
            "refusal_reasons_breakdown": self.refusal_reasons_breakdown,
            "class_counts": self.class_counts,
        }


def compute_cross_view_pair_score(
    obs_a: MultiViewEntityObservation,
    obs_b: MultiViewEntityObservation,
    max_distance_3d: float = 3.5,
    max_reproj_err_px: float = 120.0,
    sigma_3d: float = 1.0,
    sigma_reproj: float = 40.0,
) -> PairScore | None:
    """Computes geometric and reprojection compatibility between two observations.

    Returns None or score <= 0 if:
    - Observations come from the same photograph (same_image_conflict)
    - 3D Euclidean distance exceeds max_distance_3d
    - Cross-view reprojection error exceeds max_reproj_err_px
    """
    if obs_a.source_id == obs_b.source_id:
        return PairScore(
            a=obs_a.observation_id,
            b=obs_b.observation_id,
            score=0.0,
            basis={"reason": "same_image_conflict"},
        )

    # 1. 3D Centroid Euclidean Distance
    p_a = np.array(obs_a.position, dtype=np.float64)
    p_b = np.array(obs_b.position, dtype=np.float64)
    dist_3d = float(np.linalg.norm(p_a - p_b))

    if dist_3d > max_distance_3d:
        return PairScore(
            a=obs_a.observation_id,
            b=obs_b.observation_id,
            score=0.0,
            basis={"reason": "excessive_3d_distance", "dist_3d": round(dist_3d, 3)},
        )

    score_3d = math.exp(-0.5 * (dist_3d / max(1e-3, sigma_3d)) ** 2)

    # 2. Cross-View Reprojection Error (if cameras available)
    reproj_errors: list[float] = []

    if obs_b.camera is not None and len(obs_b.box_2d) == 4:
        uv_a_in_b = obs_b.camera.project(Vec3(obs_a.position[0], obs_a.position[1], obs_a.position[2]))
        if uv_a_in_b is not None:
            # Box center in B
            cx_b = 0.5 * (obs_b.box_2d[0] + obs_b.box_2d[2])
            cy_b = 0.5 * (obs_b.box_2d[1] + obs_b.box_2d[3])
            err_b = math.hypot(uv_a_in_b[0] - cx_b, uv_a_in_b[1] - cy_b)
            reproj_errors.append(err_b)

    if obs_a.camera is not None and len(obs_a.box_2d) == 4:
        uv_b_in_a = obs_a.camera.project(Vec3(obs_b.position[0], obs_b.position[1], obs_b.position[2]))
        if uv_b_in_a is not None:
            cx_a = 0.5 * (obs_a.box_2d[0] + obs_a.box_2d[2])
            cy_a = 0.5 * (obs_a.box_2d[1] + obs_a.box_2d[3])
            err_a = math.hypot(uv_b_in_a[0] - cx_a, uv_b_in_a[1] - cy_a)
            reproj_errors.append(err_a)

    if reproj_errors:
        mean_reproj_err = float(np.mean(reproj_errors))
        if mean_reproj_err > max_reproj_err_px:
            return PairScore(
                a=obs_a.observation_id,
                b=obs_b.observation_id,
                score=0.0,
                basis={"reason": "excessive_reprojection_error", "mean_reproj_px": round(mean_reproj_err, 2)},
            )
        score_reproj = math.exp(-0.5 * (mean_reproj_err / max(1e-3, sigma_reproj)) ** 2)
    else:
        mean_reproj_err = None
        score_reproj = 0.5  # Neutral fallback when camera projection unavailable

    # Combined composite score
    total_score = (score_3d ** 0.6) * (score_reproj ** 0.4)

    return PairScore(
        a=obs_a.observation_id,
        b=obs_b.observation_id,
        score=round(total_score, 4),
        basis={
            "dist_3d": round(dist_3d, 3),
            "mean_reproj_px": round(mean_reproj_err, 2) if mean_reproj_err is not None else None,
            "score_3d": round(score_3d, 4),
            "score_reproj": round(score_reproj, 4),
        },
    )


def reconcile_associated_geometry(
    cluster: AssociatedEntity,
    obs_map: dict[str, MultiViewEntityObservation],
) -> ReconciledPhysicalEntity:
    """Fuses multi-view observations of one entity into a reconciled physical entity."""
    members = [obs_map[oid] for oid in cluster.members if oid in obs_map]
    assert members, f"Cluster {cluster.entity_key} has no valid member observations"

    # 1. Primary class from consensus/most frequent label
    labels = [m.class_name for m in members]
    primary_class = Counter(labels).most_common(1)[0][0]

    # 2. Evidence-weighted centroid position
    positions = np.array([m.position for m in members], dtype=np.float64)
    confidences = np.array([max(0.01, m.confidence) for m in members], dtype=np.float64)
    weights = confidences / np.sum(confidences)

    pos_fused = np.sum(positions * weights[:, None], axis=0)

    # 3. Robust median dimensions
    dims = np.array([m.dimensions for m in members], dtype=np.float64)
    dims_fused = np.median(dims, axis=0)

    # 4. Orientation from best-confidence member with known orientation
    best_orientation = None
    best_conf = -1.0
    for m in members:
        if m.orientation is not None and m.confidence > best_conf:
            best_conf = m.confidence
            best_orientation = m.orientation

    # 5. Multi-view 2D-3D reprojection residual across all observing cameras
    reproj_errors: list[float] = []
    for m in members:
        if m.camera is not None and len(m.box_2d) == 4:
            uv = m.camera.project(Vec3(float(pos_fused[0]), float(pos_fused[1]), float(pos_fused[2])))
            if uv is not None:
                cx = 0.5 * (m.box_2d[0] + m.box_2d[2])
                cy = 0.5 * (m.box_2d[1] + m.box_2d[3])
                reproj_errors.append(math.hypot(uv[0] - cx, uv[1] - cy))

    mean_reproj_px = float(np.mean(reproj_errors)) if reproj_errors else None

    # 6. Evidence-calibrated composite confidence
    # - observation count bonus: increases from 1 to multiple views
    # - reprojection error penalty: decreases if 3D centroid reprojects poorly into views
    mean_det_conf = float(np.mean(confidences))
    k_views = len(cluster.sources)
    view_factor = 1.0 - (1.0 / (k_views + 1.0))  # 0.5 for 1 view, 0.67 for 2, 0.75 for 3, 0.8 for 4

    reproj_factor = 1.0
    if mean_reproj_px is not None:
        reproj_factor = math.exp(-0.5 * (min(mean_reproj_px, 150.0) / 40.0) ** 2)

    calibrated_conf = min(0.99, max(0.05, mean_det_conf * (0.6 + 0.4 * view_factor) * (0.5 + 0.5 * reproj_factor)))

    # Parent entity / part determination
    parent_obs_ids = [m.parent_observation_id for m in members if m.parent_observation_id]
    is_part = any(m.is_part for m in members)

    return ReconciledPhysicalEntity(
        entity_id=cluster.entity_key,
        primary_class=primary_class,
        member_observation_ids=cluster.members,
        sources=cluster.sources,
        observation_count=len(members),
        is_multiview=k_views >= 2,
        position=(float(pos_fused[0]), float(pos_fused[1]), float(pos_fused[2])),
        dimensions=(float(dims_fused[0]), float(dims_fused[1]), float(dims_fused[2])),
        orientation=best_orientation,
        mean_detector_confidence=mean_det_conf,
        calibrated_confidence=calibrated_conf,
        reprojection_residual_px=mean_reproj_px,
        parent_entity_id=parent_obs_ids[0] if parent_obs_ids else None,
        is_part=is_part,
    )


class MultiViewAssociator:
    """End-to-end multi-view physical identity associator."""

    def __init__(
        self,
        max_distance_3d: float = 3.5,
        max_reproj_err_px: float = 120.0,
        min_match_score: float = 0.20,
    ) -> None:
        self.max_distance_3d = max_distance_3d
        self.max_reproj_err_px = max_reproj_err_px
        self.min_match_score = min_match_score

    def associate_observations(
        self,
        observations: Sequence[MultiViewEntityObservation],
    ) -> tuple[tuple[ReconciledPhysicalEntity, ...], MultiViewAssociationSummary, tuple[Decision, ...]]:
        """Associates observations across multiple camera views into unique physical entities."""
        if not observations:
            empty_summary = MultiViewAssociationSummary(
                total_input_observations=0,
                unique_physical_entities_count=0,
                single_view_entities_count=0,
                multiview_merged_entities_count=0,
                max_views_per_entity=0,
                accepted_merges_count=0,
                refused_pairs_count=0,
                refusal_reasons_breakdown={},
                class_counts={},
            )
            return (), empty_summary, ()

        obs_map = {o.observation_id: o for o in observations}

        # 1. Build Candidate records
        candidates = [
            Candidate(
                observation_id=o.observation_id,
                source_id=o.source_id,
                label=o.class_name,
                parent_observation_id=o.parent_observation_id,
            )
            for o in observations
        ]

        # 2. Build pairwise cross-view scores
        pair_scores: list[PairScore] = []
        n = len(observations)
        for i in range(n):
            for j in range(i + 1, n):
                o_a, o_b = observations[i], observations[j]
                ps = compute_cross_view_pair_score(
                    o_a,
                    o_b,
                    max_distance_3d=self.max_distance_3d,
                    max_reproj_err_px=self.max_reproj_err_px,
                )
                if ps is not None and ps.score >= self.min_match_score:
                    pair_scores.append(ps)

        # 3. Run rigorous association union-find with hard constraints
        # Use ontology_label_compat or relaxed match for architectural concepts
        label_compat = ontology_label_compat()
        result: AssociationResult = associate(
            candidates=candidates,
            scores=pair_scores,
            labels_compatible=label_compat,
        )

        # 4. Reconcile geometry for each associated entity cluster
        reconciled_entities: list[ReconciledPhysicalEntity] = []
        for cluster in result.entities:
            reconciled = reconcile_associated_geometry(cluster, obs_map)
            reconciled_entities.append(reconciled)

        # 5. Compute summary metrics and ledger breakdown
        refusal_counter: Counter[str] = Counter()
        accepted_count = 0
        for d in result.decisions:
            if d.outcome == MERGED:
                accepted_count += 1
            elif d.outcome == REFUSED:
                refusal_counter[d.reason] += 1

        multiview_count = sum(1 for e in reconciled_entities if e.is_multiview)
        single_count = sum(1 for e in reconciled_entities if not e.is_multiview)
        max_views = max((len(e.sources) for e in reconciled_entities), default=0)
        class_counts = dict(Counter(e.primary_class for e in reconciled_entities))

        summary = MultiViewAssociationSummary(
            total_input_observations=len(observations),
            unique_physical_entities_count=len(reconciled_entities),
            single_view_entities_count=single_count,
            multiview_merged_entities_count=multiview_count,
            max_views_per_entity=max_views,
            accepted_merges_count=accepted_count,
            refused_pairs_count=sum(refusal_counter.values()),
            refusal_reasons_breakdown=dict(refusal_counter),
            class_counts=class_counts,
        )

        return tuple(reconciled_entities), summary, result.decisions
