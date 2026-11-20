"""Multi-View Physical Identity Association and Geometric Reconciliation.

File: engine/multiengine/multi_view_association.py

Implements physical object identity association across multi-view photographs:
1. Constructs Candidate records from 3D lifted entity observations.
2. Derives unit-aware and uncertainty-aware 3D geometric compatibility pair scores.
3. Computes cross-view reprojection with strict camera visibility validation:
   - Forbids neutral fallbacks (e.g. 0.5) when camera calibration is missing; missing evidence remains missing.
   - Requires 3D point to be in front of camera plane (z_cam > 0.1).
4. Supports appearance descriptor consistency checks without making appearance alone sufficient for identity.
5. Enforces strict physical constraints via `engine.multiengine.association.associate`:
   - HARD REFUSAL: Two detections from the same photograph CAN NEVER merge into the same physical entity.
   - HARD REFUSAL: Incompatible semantic labels (e.g. window vs tree) CAN NEVER merge.
   - HARD REFUSAL: Whole-part and part-parent hierarchical consistency.
6. Generates an immutable Decision Ledger detailing every accepted merge and refusal reason.
7. Reconciles multi-view geometry (weighted 3D centroid, bounding box dimensions, and reprojection residuals).
8. Computes composite evidence-calibrated confidence reflecting view count, reprojection consistency,
   and metric scale anchor status without discarding raw detector confidence provenance.
9. Provides quantitative benchmark evaluation: Precision, Recall, F1, False Merge Rate, False Split Rate,
   and per-class metrics under occlusion and repeated structures.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
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
    confidence: float | None = None  # raw model detector confidence (None if unmeasured)
    scale_units: str = "METRES"  # "METRES", "COLMAP_SFM_UNITS", "RELATIVE_DISPARITY"
    is_physically_metric: bool = False
    position_uncertainty: tuple[float, float, float] | None = None  # (std_x, std_y, std_z)
    appearance_features: Sequence[float] | None = None  # Normalized color histogram or embedding
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
    mean_detector_confidence: float | None
    calibrated_confidence: float | None
    confidence_basis: str | None
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
            "mean_detector_confidence": round(self.mean_detector_confidence, 4) if self.mean_detector_confidence is not None else None,
            "calibrated_confidence": round(self.calibrated_confidence, 4) if self.calibrated_confidence is not None else None,
            "confidence_basis": self.confidence_basis,
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


@dataclass(frozen=True)
class AssociationBenchmarkReport:
    """Quantitative performance report for multi-view physical identity association."""

    total_pairs_evaluated: int
    true_positives: int
    false_positives: int  # False merges
    false_negatives: int  # False splits
    precision: float | None
    recall: float | None
    f1_score: float | None
    false_merge_rate: float | None
    false_split_rate: float | None
    per_class_metrics: dict[str, dict[str, float | None]]
    performance_under_occlusion: dict[str, float | None]
    performance_repeated_structures: dict[str, float | None]
    performance_missing_calibration: dict[str, float | None]
    evaluation_status: str = "COMPLETED"  # "COMPLETED", "NOT_EVALUATED", "EMPTY_GROUND_TRUTH"
    unmatched_observations_count: int = 0
    unlabelled_observations_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_status": self.evaluation_status,
            "total_pairs_evaluated": self.total_pairs_evaluated,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1_score": round(self.f1_score, 4) if self.f1_score is not None else None,
            "false_merge_rate": round(self.false_merge_rate, 4) if self.false_merge_rate is not None else None,
            "false_split_rate": round(self.false_split_rate, 4) if self.false_split_rate is not None else None,
            "unmatched_observations_count": self.unmatched_observations_count,
            "unlabelled_observations_count": self.unlabelled_observations_count,
            "per_class_metrics": self.per_class_metrics,
            "performance_under_occlusion": self.performance_under_occlusion,
            "performance_repeated_structures": self.performance_repeated_structures,
            "performance_missing_calibration": self.performance_missing_calibration,
        }


def compute_cross_view_pair_score(
    obs_a: MultiViewEntityObservation,
    obs_b: MultiViewEntityObservation,
    max_distance_3d: float | None = None,
    max_reproj_err_px: float = 120.0,
    min_appearance_sim: float = 0.35,
    sigma_reproj: float = 40.0,
) -> PairScore | None:
    """Computes geometric, reprojection, and appearance compatibility between two observations.

    Strict Rules:
    - Same-image detections are strictly refused (same_image_conflict).
    - Thresholds are derived from coordinate scale units (METRES vs COLMAP_SFM_UNITS) and bounding extents.
    - Missing camera evidence remains missing; NO neutral or fabricated positive fallback.
    - 3D points behind the camera plane (z_cam <= 0.1) are rejected as non-visible.
    - Appearance evidence is an optional gate/modulator, never alone sufficient for identity.
    - Uncertainty-aware distance check is applied if position uncertainties are supplied.
    """
    if obs_a.source_id == obs_b.source_id:
        return PairScore(
            a=obs_a.observation_id,
            b=obs_b.observation_id,
            score=0.0,
            basis={"reason": "same_image_conflict"},
        )

    # Validate finite coordinates
    if not (all(np.isfinite(obs_a.position)) and all(np.isfinite(obs_b.position))):
        return PairScore(
            a=obs_a.observation_id,
            b=obs_b.observation_id,
            score=0.0,
            basis={"reason": "non_finite_geometry"},
        )

    p_a = np.array(obs_a.position, dtype=np.float64)
    p_b = np.array(obs_b.position, dtype=np.float64)
    dist_3d = float(np.linalg.norm(p_a - p_b))

    # Unit-aware and extent-aware 3D distance threshold
    dim_a = np.array(obs_a.dimensions, dtype=np.float64)
    dim_b = np.array(obs_b.dimensions, dtype=np.float64)
    extent_a = float(np.linalg.norm(dim_a)) if len(dim_a) == 3 else 1.0
    extent_b = float(np.linalg.norm(dim_b)) if len(dim_b) == 3 else 1.0
    mean_extent = max(0.1, 0.5 * (extent_a + extent_b))

    if max_distance_3d is not None:
        tol_3d = max_distance_3d
    elif obs_a.scale_units == "METRES" and obs_b.scale_units == "METRES":
        # Physical meters: distance tolerance bounded by physical object extent
        tol_3d = max(0.35, min(2.5, 1.2 * mean_extent))
    else:
        # COLMAP SfM gauge units: adaptive tolerance derived from reconstructed bounding extent
        tol_3d = max(0.5, 1.5 * mean_extent)

    sigma_3d = max(0.1, tol_3d / 2.5)

    # Uncertainty-aware geometric compatibility (Mahalanobis distance)
    if obs_a.position_uncertainty is not None and obs_b.position_uncertainty is not None:
        u_a = np.array(obs_a.position_uncertainty, dtype=np.float64)
        u_b = np.array(obs_b.position_uncertainty, dtype=np.float64)
        var_comb = u_a ** 2 + u_b ** 2
        if np.all(var_comb > 1e-6):
            m_dist = float(np.sqrt(np.sum((p_a - p_b) ** 2 / var_comb)))
            if m_dist > 3.5:
                return PairScore(
                    a=obs_a.observation_id,
                    b=obs_b.observation_id,
                    score=0.0,
                    basis={"reason": "uncertainty_incompatible", "mahalanobis_dist": round(m_dist, 2)},
                )

    if dist_3d > tol_3d:
        return PairScore(
            a=obs_a.observation_id,
            b=obs_b.observation_id,
            score=0.0,
            basis={"reason": "excessive_3d_distance", "dist_3d": round(dist_3d, 3), "tol_3d": round(tol_3d, 3)},
        )

    score_3d = math.exp(-0.5 * (dist_3d / sigma_3d) ** 2)

    # Cross-View Reprojection Error with Visibility Validation
    reproj_errors: list[float] = []

    # Check projection of A into camera B
    if obs_b.camera is not None and len(obs_b.box_2d) == 4:
        p_cam_b = obs_b.camera.extrinsics.world_to_camera(Vec3(p_a[0], p_a[1], p_a[2]))
        if p_cam_b.z > 0.1:  # Point is in front of camera B
            uv_a_in_b = obs_b.camera.project(Vec3(p_a[0], p_a[1], p_a[2]))
            if uv_a_in_b is not None:
                cx_b = 0.5 * (obs_b.box_2d[0] + obs_b.box_2d[2])
                cy_b = 0.5 * (obs_b.box_2d[1] + obs_b.box_2d[3])
                reproj_errors.append(math.hypot(uv_a_in_b[0] - cx_b, uv_a_in_b[1] - cy_b))

    # Check projection of B into camera A
    if obs_a.camera is not None and len(obs_a.box_2d) == 4:
        p_cam_a = obs_a.camera.extrinsics.world_to_camera(Vec3(p_b[0], p_b[1], p_b[2]))
        if p_cam_a.z > 0.1:  # Point is in front of camera A
            uv_b_in_a = obs_a.camera.project(Vec3(p_b[0], p_b[1], p_b[2]))
            if uv_b_in_a is not None:
                cx_a = 0.5 * (obs_a.box_2d[0] + obs_a.box_2d[2])
                cy_a = 0.5 * (obs_a.box_2d[1] + obs_a.box_2d[3])
                reproj_errors.append(math.hypot(uv_b_in_a[0] - cx_a, uv_b_in_a[1] - cy_a))

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
        has_reproj = True
    else:
        # Missing camera evidence remains missing; DO NOT INVENT A NEUTRAL 0.5 FALLBACK
        mean_reproj_err = None
        score_reproj = None
        has_reproj = False

    # Appearance Compatibility Check (Optional Gate/Modulator)
    sim_app = None
    if obs_a.appearance_features is not None and obs_b.appearance_features is not None:
        fa = np.array(obs_a.appearance_features, dtype=np.float64)
        fb = np.array(obs_b.appearance_features, dtype=np.float64)
        norm_a = float(np.linalg.norm(fa))
        norm_b = float(np.linalg.norm(fb))
        if norm_a > 1e-6 and norm_b > 1e-6:
            sim_app = float(np.dot(fa, fb) / (norm_a * norm_b))
            if sim_app < min_appearance_sim:
                return PairScore(
                    a=obs_a.observation_id,
                    b=obs_b.observation_id,
                    score=0.0,
                    basis={"reason": "appearance_dissimilarity", "similarity": round(sim_app, 4)},
                )

    # Dynamic Composite Score Calculation
    if has_reproj and score_reproj is not None:
        base_score = (score_3d ** 0.6) * (score_reproj ** 0.4)
    else:
        # Score is purely geometric when camera projection is not available
        base_score = score_3d

    if sim_app is not None:
        total_score = base_score * (0.7 + 0.3 * max(0.0, sim_app))
    else:
        total_score = base_score

    return PairScore(
        a=obs_a.observation_id,
        b=obs_b.observation_id,
        score=round(total_score, 4),
        basis={
            "dist_3d": round(dist_3d, 3),
            "tol_3d": round(tol_3d, 3),
            "score_3d": round(score_3d, 4),
            "reprojection_available": has_reproj,
            "mean_reproj_px": round(mean_reproj_err, 2) if mean_reproj_err is not None else None,
            "score_reproj": round(score_reproj, 4) if score_reproj is not None else None,
            "appearance_similarity": round(sim_app, 4) if sim_app is not None else None,
            "coordinate_units": obs_a.scale_units,
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
    has_conf = [m.confidence is not None for m in members]
    if any(has_conf):
        confs = np.array([m.confidence if m.confidence is not None else 0.5 for m in members], dtype=np.float64)
        weights = confs / np.sum(confs)
    else:
        weights = np.ones(len(members)) / len(members)

    pos_fused = np.sum(positions * weights[:, None], axis=0)

    # 3. Robust median dimensions
    dims = np.array([m.dimensions for m in members], dtype=np.float64)
    dims_fused = np.median(dims, axis=0)

    # 4. Orientation from best-confidence member with known orientation
    best_orientation = None
    best_conf = -1.0
    for m in members:
        if m.orientation is not None:
            c_val = m.confidence if m.confidence is not None else 0.5
            if c_val > best_conf:
                best_conf = c_val
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
    valid_confs = [m.confidence for m in members if m.confidence is not None]
    if valid_confs:
        mean_det_conf = float(np.mean(valid_confs))
        k_views = len(cluster.sources)
        view_factor = 1.0 - (1.0 / (k_views + 1.0))

        if mean_reproj_px is not None:
            reproj_factor = math.exp(-0.5 * (min(mean_reproj_px, 120.0) / 35.0) ** 2)
        else:
            reproj_factor = 0.75  # Unverified reprojection penalty

        metric_scale_factor = 1.0 if all(m.is_physically_metric for m in members) else 0.85
        calibrated_conf = min(0.99, max(0.05, mean_det_conf * (0.5 + 0.5 * view_factor) * (0.6 + 0.4 * reproj_factor) * metric_scale_factor))
        conf_basis = f"CALIBRATED: mean_det={mean_det_conf:.3f}, views={k_views}, reproj_factor={reproj_factor:.3f}, scale_factor={metric_scale_factor:.2f}"
    else:
        mean_det_conf = None
        calibrated_conf = None
        conf_basis = "UNMEASURED: No detector confidence supplied"

    parent_obs_ids = [m.parent_observation_id for m in members if m.parent_observation_id]
    is_part = any(m.is_part for m in members)

    return ReconciledPhysicalEntity(
        entity_id=cluster.entity_key,
        primary_class=primary_class,
        member_observation_ids=cluster.members,
        sources=cluster.sources,
        observation_count=len(members),
        is_multiview=len(cluster.sources) >= 2,
        position=(float(pos_fused[0]), float(pos_fused[1]), float(pos_fused[2])),
        dimensions=(float(dims_fused[0]), float(dims_fused[1]), float(dims_fused[2])),
        orientation=best_orientation,
        mean_detector_confidence=mean_det_conf,
        calibrated_confidence=calibrated_conf,
        confidence_basis=conf_basis,
        reprojection_residual_px=mean_reproj_px,
        parent_entity_id=parent_obs_ids[0] if parent_obs_ids else None,
        is_part=is_part,
    )


class MultiViewAssociator:
    """End-to-end multi-view physical identity associator."""

    def __init__(
        self,
        max_distance_3d: float | None = None,
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

        # 2. Build pairwise cross-view scores and log non-qualifying refusals
        pair_scores: list[PairScore] = []
        extra_refused_decisions: list[Decision] = []
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
                if ps is not None:
                    if ps.score >= self.min_match_score:
                        pair_scores.append(ps)
                    else:
                        reason = str(ps.basis.get("reason", "low_match_score"))
                        extra_refused_decisions.append(
                            Decision(
                                a=ps.a,
                                b=ps.b,
                                score=ps.score,
                                outcome=REFUSED,
                                reason=reason,
                                basis=ps.basis,
                            )
                        )
                else:
                    extra_refused_decisions.append(
                        Decision(
                            a=o_a.observation_id,
                            b=o_b.observation_id,
                            score=0.0,
                            outcome=REFUSED,
                            reason="uncomputable_pair_score",
                            basis={},
                        )
                    )

        # 3. Run rigorous association union-find with hard constraints
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

        # 5. Build unified decision ledger and breakdown
        all_decisions: list[Decision] = []
        for d in result.decisions:
            if d.outcome == REFUSED and d.reason == "non_positive_score" and d.basis.get("reason"):
                all_decisions.append(Decision(d.a, d.b, d.score, d.outcome, str(d.basis["reason"]), d.basis))
            else:
                all_decisions.append(d)
        all_decisions.extend(extra_refused_decisions)

        refusal_counter: Counter[str] = Counter()
        accepted_count = 0
        for d in all_decisions:
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

        return tuple(reconciled_entities), summary, tuple(all_decisions)


def evaluate_association_benchmark(
    ground_truth_entities: Mapping[str, str],  # obs_id -> true_physical_id
    reconciled_entities: Sequence[ReconciledPhysicalEntity],
    observations: Sequence[MultiViewEntityObservation],
    occluded_obs_ids: set[str] | None = None,
    repeated_pattern_classes: set[str] | None = None,
) -> AssociationBenchmarkReport:
    """Evaluates multi-view identity association against ground truth physical identities.

    Strict Evaluation Rules:
    - Independent Ground Truth: Uses an external mapping of observation -> true physical object ID.
    - If ground truth is empty or missing, returns evaluation_status='NOT_EVALUATED' with None metrics.
    - Never produce perfect (1.0) precision or recall from an empty positive set; returns None.
    - Reports unmatched observations and unlabelled cases separately.
    """
    if not ground_truth_entities or not any(v for v in ground_truth_entities.values() if str(v).strip() not in ("", "none", "unknown", "unlabelled")):
        return AssociationBenchmarkReport(
            total_pairs_evaluated=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=None,
            recall=None,
            f1_score=None,
            false_merge_rate=None,
            false_split_rate=None,
            per_class_metrics={},
            performance_under_occlusion={"precision": None, "recall": None, "f1": None},
            performance_repeated_structures={"precision": None, "recall": None, "f1": None},
            performance_missing_calibration={"precision": None, "recall": None, "f1": None},
            evaluation_status="NOT_EVALUATED",
            unmatched_observations_count=0,
            unlabelled_observations_count=len(observations),
        )

    obs_to_pred = {}
    for rent in reconciled_entities:
        for oid in rent.member_observation_ids:
            obs_to_pred[oid] = rent.entity_id

    # Evaluate pairwise associations among observations that have valid ground truth labels
    obs_ids = [
        o.observation_id
        for o in observations
        if o.observation_id in ground_truth_entities and str(ground_truth_entities[o.observation_id]).strip() not in ("", "none", "unknown", "unlabelled")
    ]
    obs_by_id = {o.observation_id: o for o in observations}
    n = len(obs_ids)

    unlabelled_count = len(observations) - len(obs_ids)
    unmatched_count = sum(1 for oid in obs_ids if oid not in obs_to_pred)

    tp, fp, fn = 0, 0, 0
    per_class_tp: Counter[str] = Counter()
    per_class_fp: Counter[str] = Counter()
    per_class_fn: Counter[str] = Counter()

    occl_tp, occl_fp, occl_fn = 0, 0, 0
    rep_tp, rep_fp, rep_fn = 0, 0, 0
    nocal_tp, nocal_fp, nocal_fn = 0, 0, 0

    occluded_set = occluded_obs_ids or set()
    repeated_set = repeated_pattern_classes or {"window", "door", "column", "table", "chair", "socket"}

    pairs_evaluated = 0

    for i in range(n):
        for j in range(i + 1, n):
            id_a, id_b = obs_ids[i], obs_ids[j]
            o_a, o_b = obs_by_id[id_a], obs_by_id[id_b]
            if o_a.source_id == o_b.source_id:
                continue  # Pairs from same image are excluded by protocol

            pairs_evaluated += 1
            gt_same = (ground_truth_entities[id_a] == ground_truth_entities[id_b])
            pred_same = (obs_to_pred.get(id_a) == obs_to_pred.get(id_b) and obs_to_pred.get(id_a) is not None)

            c_label = o_a.class_name
            is_occl = (id_a in occluded_set or id_b in occluded_set)
            is_rep = (c_label in repeated_set)
            is_nocal = (o_a.camera is None or o_b.camera is None)

            if gt_same and pred_same:
                tp += 1
                per_class_tp[c_label] += 1
                if is_occl:
                    occl_tp += 1
                if is_rep:
                    rep_tp += 1
                if is_nocal:
                    nocal_tp += 1
            elif not gt_same and pred_same:
                fp += 1
                per_class_fp[c_label] += 1
                if is_occl:
                    occl_fp += 1
                if is_rep:
                    rep_fp += 1
                if is_nocal:
                    nocal_fp += 1
            elif gt_same and not pred_same:
                fn += 1
                per_class_fn[c_label] += 1
                if is_occl:
                    occl_fn += 1
                if is_rep:
                    rep_fn += 1
                if is_nocal:
                    nocal_fn += 1

    # Never produce 1.0 from an empty set
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (None if (tp + fn) == 0 else 0.0)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    f1 = float(2 * precision * recall / (precision + recall)) if (precision is not None and recall is not None and (precision + recall) > 0) else (0.0 if (precision is not None or recall is not None) else None)

    false_merge_rate = float(fp / (tp + fp)) if (tp + fp) > 0 else (None if (tp + fn) == 0 else 0.0)
    false_split_rate = float(fn / (tp + fn)) if (tp + fn) > 0 else None

    # Per-class metrics
    all_classes = set(per_class_tp.keys()) | set(per_class_fp.keys()) | set(per_class_fn.keys())
    per_class_metrics: dict[str, dict[str, float | None]] = {}
    for c in all_classes:
        c_tp, c_fp, c_fn = per_class_tp[c], per_class_fp[c], per_class_fn[c]
        c_prec = float(c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else (None if (c_tp + c_fn) == 0 else 0.0)
        c_rec = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else None
        c_f1 = float(2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec is not None and c_rec is not None and (c_prec + c_rec) > 0) else (0.0 if (c_prec is not None or c_rec is not None) else None)
        per_class_metrics[c] = {
            "precision": round(c_prec, 4) if c_prec is not None else None,
            "recall": round(c_rec, 4) if c_rec is not None else None,
            "f1": round(c_f1, 4) if c_f1 is not None else None,
        }

    def _calc_sub(t: int, f_p: int, f_n: int) -> dict[str, float | None]:
        p = float(t / (t + f_p)) if (t + f_p) > 0 else (None if (t + f_n) == 0 else 0.0)
        r = float(t / (t + f_n)) if (t + f_n) > 0 else None
        s_f1 = float(2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else (0.0 if (p is not None or r is not None) else None)
        return {
            "precision": round(p, 4) if p is not None else None,
            "recall": round(r, 4) if r is not None else None,
            "f1": round(s_f1, 4) if s_f1 is not None else None,
        }

    return AssociationBenchmarkReport(
        total_pairs_evaluated=pairs_evaluated,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_merge_rate=false_merge_rate,
        false_split_rate=false_split_rate,
        per_class_metrics=per_class_metrics,
        performance_under_occlusion=_calc_sub(occl_tp, occl_fp, occl_fn),
        performance_repeated_structures=_calc_sub(rep_tp, rep_fp, rep_fn),
        performance_missing_calibration=_calc_sub(nocal_tp, nocal_fp, nocal_fn),
        evaluation_status="COMPLETED",
        unmatched_observations_count=unmatched_count,
        unlabelled_observations_count=unlabelled_count,
    )
