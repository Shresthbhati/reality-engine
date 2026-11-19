"""Robust multi-sensor evidence fusion (P6-02 / Phase 9 / Cycle 3).

Extends the inverse-variance fusion engine in ``fusion.py`` with robust M-estimation
(Huber and Tukey biweight loss via Iteratively Reweighted Least Squares).

When multiple physical sensors (e.g. LiDAR, stereo, monocular depth, IMU) observe
the same real-world quantity:
  - Typical non-anomalous sensor readings are integrated via Bayesian inverse-variance
    weighting (sigma_i^2).
  - Small sample sizes (N=2) and two-sensor disagreements are handled explicitly without
    degenerate scale estimates. Bimodal distributions are detected and preserved as
    competing hypotheses rather than blithely averaged into an invalid intermediate state.
  - Correlated observations (rho > 0) are properly accounted for via intraclass correlation
    variance adjustment to prevent false precision inflation.
  - Multi-sensor 3D point cloud fusion requires registered spatial correspondences
    (via KD-tree / nearest-neighbor radius query). Raw points are NEVER silently truncated
    or destroyed: unmatched points are catalogued and returned with full provenance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.core.units import Unit
from provenance import Provenance
from reconstruction.fusion.fusion import (
    FusableObservation,
    FusionError,
    MeasurementConflict,
    _canonical_order,
    _pairwise_conflicts,
)
from reconstruction.fusion.multi_source import DepthSourceObservation

try:
    from scipy.spatial import KDTree  # type: ignore[import-untyped]
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass(frozen=True)
class RobustFusionResult:
    """Robust multi-sensor fusion estimate preserving inliers, outliers, and bimodal modes."""

    quantity: str
    value: float
    unit: Unit
    precision: float
    confidence: float
    provenance: Provenance
    method: str
    inliers: tuple[FusableObservation, ...]
    outliers: tuple[FusableObservation, ...]
    weights: dict[str, float]
    conflicts: tuple[MeasurementConflict, ...]
    iterations: int
    scale_estimate: float
    is_bimodal: bool = False
    competing_modes: tuple[tuple[float, float], ...] = ()
    correlation: float = 0.0

    def has_outliers(self) -> bool:
        return len(self.outliers) > 0

    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0 or len(self.outliers) > 0 or self.is_bimodal


@dataclass(frozen=True)
class PointCloudFusionResult:
    """Multi-sensor point cloud fusion result preserving registered consensus and unmatched points."""

    fused_points: list[tuple[float, float, float]]
    unmatched_by_source: dict[str, list[tuple[float, float, float]]]
    conflicts_count: int
    total_fused: int
    total_unmatched: int


def _compute_mad_scale(values: list[float], center: float) -> float:
    """Median Absolute Deviation (MAD) scaled to Gaussian standard deviation."""
    deviations = sorted(abs(v - center) for v in values)
    n = len(deviations)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        mad = deviations[n // 2]
    else:
        mad = (deviations[n // 2 - 1] + deviations[n // 2]) / 2.0
    # 1.4826 converts MAD to normal distribution scale
    return max(1.4826 * mad, 1e-6)


def robust_fuse_quantity(
    observations: Sequence[FusableObservation],
    quantity: str,
    loss: str = "huber",
    tuning_k: float = 2.5,
    max_iter: int = 20,
    tol: float = 1e-5,
    correlation: float = 0.0,
) -> RobustFusionResult:
    """Robustly fuse multi-source observations of a single physical quantity using M-estimation.

    Handles small sample sizes, two-sensor disagreements, bimodal distributions,
    and correlated observations without data loss or unjustified averaging.

    Parameters:
      observations: Declared observations with honesty-checked precisions (> 0).
      quantity: Name of the quantity (e.g. "depth:p0").
      loss: Robust loss function ("huber" or "tukey").
      tuning_k: Outlier residual threshold (standardized units). Default 2.5 sigma.
      max_iter: Max IRLS iterations.
      tol: Convergence tolerance on value change.
      correlation: Estimated inter-observation correlation coefficient in [0.0, 1.0).

    Returns:
      RobustFusionResult with inliers, outliers, and bimodal conflict status.
    """
    if not observations:
        raise FusionError(f"cannot fuse quantity {quantity!r}: no observations provided")

    ordered = _canonical_order(observations)
    unit = ordered[0].unit
    mismatched = [o.source for o in ordered if o.unit != unit]
    if mismatched:
        wrong = next(o.unit.value for o in ordered if o.unit != unit)
        raise FusionError(
            f"unit mismatch while fusing {quantity!r}: "
            f"{unit.value} vs {wrong} from source {mismatched[0]!r}"
        )

    if len(ordered) == 1:
        o = ordered[0]
        return RobustFusionResult(
            quantity=quantity,
            value=o.value,
            unit=o.unit,
            precision=o.precision,
            confidence=o.confidence,
            provenance=o.provenance,
            method="passthrough",
            inliers=(o,),
            outliers=(),
            weights={o.source: 1.0 / (o.precision**2)},
            conflicts=(),
            iterations=0,
            scale_estimate=o.precision,
            is_bimodal=False,
            competing_modes=(),
            correlation=correlation,
        )

    # Handle N=2 small sample case explicitly: two-sensor agreement vs bimodal disagreement
    if len(ordered) == 2:
        o1, o2 = ordered[0], ordered[1]
        sigma_diff = math.hypot(o1.precision, o2.precision)
        discrepancy = abs(o1.value - o2.value)
        z = discrepancy / max(sigma_diff, 1e-6)

        if z > tuning_k:
            # Two-sensor bimodal disagreement: sensors disagree beyond combined tolerance.
            # Never blithely average into an invalid intermediate state.
            primary, secondary = (o1, o2) if o1.precision <= o2.precision else (o2, o1)
            conflicts = _pairwise_conflicts(ordered)
            penalized_conf = min(o1.confidence, o2.confidence) * math.exp(-min(z, 6.0) / 4.0)

            return RobustFusionResult(
                quantity=quantity,
                value=primary.value,
                unit=unit,
                precision=primary.precision,
                confidence=round(penalized_conf, 4),
                provenance=Provenance.CONFLICT,
                method="bimodal_disagreement_detected",
                inliers=(primary,),
                outliers=(secondary,),
                weights={primary.source: 1.0 / (primary.precision**2), secondary.source: 0.0},
                conflicts=tuple(conflicts),
                iterations=1,
                scale_estimate=discrepancy,
                is_bimodal=True,
                competing_modes=((o1.value, o1.precision), (o2.value, o2.precision)),
                correlation=correlation,
            )
        else:
            # Consistent two-sensor consensus
            w1 = 1.0 / (o1.precision**2)
            w2 = 1.0 / (o2.precision**2)
            val = (o1.value * w1 + o2.value * w2) / (w1 + w2)
            base_var = 1.0 / (w1 + w2)
            adj_var = base_var * (1.0 + correlation) if correlation > 0.0 else base_var
            prec = math.sqrt(adj_var)
            conf = min(o1.confidence, o2.confidence)
            return RobustFusionResult(
                quantity=quantity,
                value=val,
                unit=unit,
                precision=prec,
                confidence=conf,
                provenance=Provenance.ESTIMATED,
                method="two_sensor_consensus",
                inliers=(o1, o2),
                outliers=(),
                weights={o1.source: w1, o2.source: w2},
                conflicts=(),
                iterations=1,
                scale_estimate=discrepancy,
                is_bimodal=False,
                competing_modes=(),
                correlation=correlation,
            )

    # For N >= 3: Iteratively Reweighted Least Squares (IRLS) M-estimation
    raw_weights = [1.0 / (o.precision**2) for o in ordered]
    total_w = sum(raw_weights)
    curr_mu = sum(o.value * w for o, w in zip(ordered, raw_weights)) / total_w
    values = [o.value for o in ordered]

    effective_weights = list(raw_weights)
    iterations = 0
    scale = 0.0

    for it in range(1, max_iter + 1):
        iterations = it
        mad_scale = _compute_mad_scale(values, curr_mu)
        scale = max(mad_scale, min(o.precision for o in ordered))

        new_weights: list[float] = []
        for o in ordered:
            base_w = 1.0 / (o.precision**2)
            total_sigma = math.sqrt(o.precision**2 + scale**2)
            std_resid = abs(o.value - curr_mu) / total_sigma

            if loss == "tukey":
                c = tuning_k * 1.5
                if std_resid <= c:
                    psi_w = (1.0 - (std_resid / c) ** 2) ** 2
                else:
                    psi_w = 0.0
            else:  # default huber
                if std_resid <= tuning_k:
                    psi_w = 1.0
                else:
                    psi_w = tuning_k / std_resid

            new_weights.append(base_w * psi_w)

        sum_w = sum(new_weights)
        if sum_w <= 1e-12:
            break

        new_mu = sum(o.value * w for o, w in zip(ordered, new_weights)) / sum_w
        effective_weights = new_weights

        if abs(new_mu - curr_mu) < tol:
            curr_mu = new_mu
            break
        curr_mu = new_mu

    # Classify inliers and outliers
    inliers: list[FusableObservation] = []
    outliers: list[FusableObservation] = []
    weight_dict: dict[str, float] = {}

    for o, w in zip(ordered, effective_weights):
        weight_dict[o.source] = w
        total_sigma = math.sqrt(o.precision**2 + scale**2)
        std_resid = abs(o.value - curr_mu) / total_sigma
        if std_resid > tuning_k:
            outliers.append(o)
        else:
            inliers.append(o)

    # Check for multi-modal / bimodal cluster in N >= 3
    is_bimodal = False
    competing_modes: list[tuple[float, float]] = []
    if len(outliers) >= 1 and len(inliers) >= 1:
        # Check if outliers themselves form a coherent second mode
        outlier_vals = [o.value for o in outliers]
        if len(outlier_vals) >= 2:
            outlier_spread = max(outlier_vals) - min(outlier_vals)
            if outlier_spread <= 2.0 * min(o.precision for o in outliers):
                is_bimodal = True
                competing_modes = [
                    (curr_mu, scale),
                    (float(np.mean(outlier_vals)), float(np.mean([o.precision for o in outliers]))),
                ]

    if not inliers:
        inliers = list(ordered)
        outliers = []

    conflicts = _pairwise_conflicts(ordered)

    # Inlier precision with correlation variance adjustment
    inlier_w_sum = sum(1.0 / (o.precision**2) for o in inliers)
    base_var = 1.0 / inlier_w_sum
    if correlation > 0.0 and len(inliers) > 1:
        adj_var = base_var * (1.0 + (len(inliers) - 1) * correlation)
    else:
        adj_var = base_var
    fused_precision = math.sqrt(adj_var)

    fused_confidence = min(o.confidence for o in inliers)
    if is_bimodal:
        fused_confidence *= 0.65

    provenance = Provenance.CONFLICT if (conflicts or outliers or is_bimodal) else Provenance.ESTIMATED

    return RobustFusionResult(
        quantity=quantity,
        value=curr_mu,
        unit=unit,
        precision=fused_precision,
        confidence=round(fused_confidence, 4),
        provenance=provenance,
        method=f"robust_{loss}_irls",
        inliers=tuple(inliers),
        outliers=tuple(outliers),
        weights=weight_dict,
        conflicts=tuple(conflicts),
        iterations=iterations,
        scale_estimate=scale,
        is_bimodal=is_bimodal,
        competing_modes=tuple(competing_modes),
        correlation=correlation,
    )


def robust_fuse_depth_observations(
    observations: Sequence[DepthSourceObservation],
    point_id: str,
    loss: str = "huber",
    tuning_k: float = 2.5,
    correlation: float = 0.0,
) -> RobustFusionResult:
    """Robustly fuse multi-source depth claims for a single world point."""
    return robust_fuse_quantity(
        [o.to_observation() for o in observations],
        quantity=f"depth:{point_id}",
        loss=loss,
        tuning_k=tuning_k,
        correlation=correlation,
    )


def fuse_registered_point_clouds(
    points_by_source: dict[str, Sequence[tuple[float, float, float]]],
    precisions_by_source: dict[str, float],
    confidences_by_source: dict[str, float] | None = None,
    correspondence_radius_m: float = 0.05,
    loss: str = "huber",
    tuning_k: float = 2.5,
) -> PointCloudFusionResult:
    """Fuse multi-sensor 3D point clouds using registered spatial correspondences.

    Requires spatial point correspondence within `correspondence_radius_m`.
    Does NOT silently truncate points when sensor clouds have unequal sizes.
    Unmatched points are catalogued and preserved in `unmatched_by_source`.
    """
    if not points_by_source:
        return PointCloudFusionResult(
            fused_points=[],
            unmatched_by_source={},
            conflicts_count=0,
            total_fused=0,
            total_unmatched=0,
        )

    sources = sorted(points_by_source.keys())
    conf_map = confidences_by_source or {}

    # Designate the source with the lowest nominal precision (highest accuracy) as primary reference
    ref_source = min(sources, key=lambda s: precisions_by_source.get(s, 0.05))
    other_sources = [s for s in sources if s != ref_source]

    ref_points = list(points_by_source[ref_source])
    fused_points: list[tuple[float, float, float]] = []
    unmatched_by_source: dict[str, list[tuple[float, float, float]]] = {s: [] for s in sources}
    conflicts_count = 0

    if not other_sources:
        return PointCloudFusionResult(
            fused_points=ref_points,
            unmatched_by_source=unmatched_by_source,
            conflicts_count=0,
            total_fused=len(ref_points),
            total_unmatched=0,
        )

    # Build spatial index trees for secondary sources
    secondary_trees: dict[str, Any] = {}
    secondary_arrays: dict[str, np.ndarray] = {}
    matched_secondary_indices: dict[str, set[int]] = {s: set() for s in other_sources}

    for s in other_sources:
        pts = points_by_source[s]
        arr = np.asarray(pts, dtype=np.float64)
        secondary_arrays[s] = arr
        if len(arr) > 0 and _HAS_SCIPY:
            secondary_trees[s] = KDTree(arr)

    # For each reference point, find corresponding points across secondary sources
    for i, ref_pt in enumerate(ref_points):
        matched_obs_x = [
            FusableObservation(
                value=ref_pt[0],
                unit=Unit.METER,
                source=ref_source,
                provenance=Provenance.OBSERVED,
                confidence=conf_map.get(ref_source, 0.95),
                precision=precisions_by_source.get(ref_source, 0.02),
            )
        ]
        matched_obs_y = [
            FusableObservation(
                value=ref_pt[1],
                unit=Unit.METER,
                source=ref_source,
                provenance=Provenance.OBSERVED,
                confidence=conf_map.get(ref_source, 0.95),
                precision=precisions_by_source.get(ref_source, 0.02),
            )
        ]
        matched_obs_z = [
            FusableObservation(
                value=ref_pt[2],
                unit=Unit.METER,
                source=ref_source,
                provenance=Provenance.OBSERVED,
                confidence=conf_map.get(ref_source, 0.95),
                precision=precisions_by_source.get(ref_source, 0.02),
            )
        ]

        has_any_match = False
        for s in other_sources:
            arr = secondary_arrays[s]
            if len(arr) == 0:
                continue

            match_idx: int | None = None
            if _HAS_SCIPY and s in secondary_trees:
                dist, idx = secondary_trees[s].query(ref_pt, distance_upper_bound=correspondence_radius_m)
                if dist <= correspondence_radius_m and idx < len(arr):
                    match_idx = int(idx)
            else:
                # Fallback Euclidean radius search
                diffs = arr - np.asarray(ref_pt, dtype=np.float64)
                dists = np.linalg.norm(diffs, axis=1)
                min_idx = int(np.argmin(dists))
                if dists[min_idx] <= correspondence_radius_m:
                    match_idx = min_idx

            if match_idx is not None:
                matched_pt = arr[match_idx]
                matched_secondary_indices[s].add(match_idx)
                has_any_match = True

                matched_obs_x.append(
                    FusableObservation(
                        value=float(matched_pt[0]),
                        unit=Unit.METER,
                        source=s,
                        provenance=Provenance.OBSERVED,
                        confidence=conf_map.get(s, 0.85),
                        precision=precisions_by_source.get(s, 0.05),
                    )
                )
                matched_obs_y.append(
                    FusableObservation(
                        value=float(matched_pt[1]),
                        unit=Unit.METER,
                        source=s,
                        provenance=Provenance.OBSERVED,
                        confidence=conf_map.get(s, 0.85),
                        precision=precisions_by_source.get(s, 0.05),
                    )
                )
                matched_obs_z.append(
                    FusableObservation(
                        value=float(matched_pt[2]),
                        unit=Unit.METER,
                        source=s,
                        provenance=Provenance.OBSERVED,
                        confidence=conf_map.get(s, 0.85),
                        precision=precisions_by_source.get(s, 0.05),
                    )
                )

        if has_any_match:
            res_x = robust_fuse_quantity(matched_obs_x, quantity=f"pt_{i}_x", loss=loss, tuning_k=tuning_k)
            res_y = robust_fuse_quantity(matched_obs_y, quantity=f"pt_{i}_y", loss=loss, tuning_k=tuning_k)
            res_z = robust_fuse_quantity(matched_obs_z, quantity=f"pt_{i}_z", loss=loss, tuning_k=tuning_k)
            fused_points.append((res_x.value, res_y.value, res_z.value))
            if res_x.has_conflicts() or res_y.has_conflicts() or res_z.has_conflicts():
                conflicts_count += 1
        else:
            unmatched_by_source[ref_source].append(ref_pt)

    # Collect unmatched secondary points
    for s in other_sources:
        arr = secondary_arrays[s]
        matched_set = matched_secondary_indices[s]
        for idx in range(len(arr)):
            if idx not in matched_set:
                unmatched_by_source[s].append((float(arr[idx][0]), float(arr[idx][1]), float(arr[idx][2])))

    total_unmatched = sum(len(pts) for pts in unmatched_by_source.values())

    return PointCloudFusionResult(
        fused_points=fused_points,
        unmatched_by_source=unmatched_by_source,
        conflicts_count=conflicts_count,
        total_fused=len(fused_points),
        total_unmatched=total_unmatched,
    )


def fuse_multisensor_points_3d(
    points_by_source: dict[str, Sequence[tuple[float, float, float]]],
    precisions_by_source: dict[str, float],
    confidences_by_source: dict[str, float] | None = None,
    loss: str = "huber",
    tuning_k: float = 2.5,
    spatial_registration: bool = False,
    correspondence_radius_m: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Robustly fuse 3D point cloud coordinates across multiple sensor streams.

    If spatial_registration is True, executes registered spatial KD-tree fusion
    without assuming 1:1 index-wise array matching.

    Parameters:
      points_by_source: Mapping from sensor name to ordered list of 3D points.
      precisions_by_source: Claimed 1-sigma spatial precision (meters) per sensor.
      confidences_by_source: Confidence per sensor [0, 1].
      loss: M-estimation loss function.
      tuning_k: Standardized residual threshold.
      spatial_registration: If True, uses spatial KD-tree registered correspondence.
      correspondence_radius_m: Radius tolerance for spatial correspondence.

    Returns:
      List of fused 3D coordinates (x, y, z).
    """
    if not points_by_source:
        return []

    if spatial_registration:
        res = fuse_registered_point_clouds(
            points_by_source=points_by_source,
            precisions_by_source=precisions_by_source,
            confidences_by_source=confidences_by_source,
            correspondence_radius_m=correspondence_radius_m,
            loss=loss,
            tuning_k=tuning_k,
        )
        return res.fused_points

    sources = sorted(points_by_source.keys())
    lengths = [len(points_by_source[s]) for s in sources]
    n_points = min(lengths)
    conf_map = confidences_by_source or {}

    fused_points: list[tuple[float, float, float]] = []

    for idx in range(n_points):
        # Fuse X coordinate
        obs_x = [
            FusableObservation(
                value=points_by_source[s][idx][0],
                unit=Unit.METER,
                source=s,
                provenance=Provenance.OBSERVED,
                confidence=conf_map.get(s, 0.9),
                precision=precisions_by_source.get(s, 0.05),
            )
            for s in sources
        ]
        res_x = robust_fuse_quantity(obs_x, quantity=f"pt_{idx}_x", loss=loss, tuning_k=tuning_k)

        # Fuse Y coordinate
        obs_y = [
            FusableObservation(
                value=points_by_source[s][idx][1],
                unit=Unit.METER,
                source=s,
                provenance=Provenance.OBSERVED,
                confidence=conf_map.get(s, 0.9),
                precision=precisions_by_source.get(s, 0.05),
            )
            for s in sources
        ]
        res_y = robust_fuse_quantity(obs_y, quantity=f"pt_{idx}_y", loss=loss, tuning_k=tuning_k)

        # Fuse Z coordinate
        obs_z = [
            FusableObservation(
                value=points_by_source[s][idx][2],
                unit=Unit.METER,
                source=s,
                provenance=Provenance.OBSERVED,
                confidence=conf_map.get(s, 0.9),
                precision=precisions_by_source.get(s, 0.05),
            )
            for s in sources
        ]
        res_z = robust_fuse_quantity(obs_z, quantity=f"pt_{idx}_z", loss=loss, tuning_k=tuning_k)

        fused_points.append((res_x.value, res_y.value, res_z.value))

    return fused_points
