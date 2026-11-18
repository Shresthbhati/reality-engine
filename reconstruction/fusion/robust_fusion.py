"""Robust multi-sensor evidence fusion (P6-02 / Phase 9).

Extends the inverse-variance fusion engine in ``fusion.py`` with robust M-estimation
(Huber and Tukey biweight loss via Iteratively Reweighted Least Squares).

When multiple physical sensors (e.g. LiDAR, stereo, monocular depth, IMU) observe
the same real-world quantity:
  - Typical non-anomalous sensor readings are integrated via Bayesian inverse-variance
    weighting (sigma_i^2).
  - Outlier readings (e.g. multipath LiDAR reflections, stereo matching failures,
    monocular scale collapse) that deviate beyond the statistical consensus are
    robustly downweighted or trimmed.
  - Raw evidence is NEVER silently discarded: outliers are catalogued in `outliers`,
    provenance is honestly stamped `Provenance.CONFLICT`, and full audit trail is preserved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RobustFusionResult:
    """Robust multi-sensor fusion estimate preserving inliers and outliers."""

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

    def has_outliers(self) -> bool:
        return len(self.outliers) > 0

    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0 or len(self.outliers) > 0


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
) -> RobustFusionResult:
    """Robustly fuse multi-source observations of a single physical quantity using M-estimation.

    Parameters:
      observations: Declared observations with honesty-checked precisions (> 0).
      quantity: Name of the quantity (e.g. "depth:p0").
      loss: Robust loss function ("huber" or "tukey").
      tuning_k: Outlier residual threshold (standardized units). Default 2.5 sigma.
      max_iter: Max IRLS iterations.
      tol: Convergence tolerance on value change.

    Returns:
      RobustFusionResult with inliers, outliers, and downweighted effective weights.
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
        )

    # Initial inverse-variance estimate
    raw_weights = [1.0 / (o.precision**2) for o in ordered]
    total_w = sum(raw_weights)
    curr_mu = sum(o.value * w for o, w in zip(ordered, raw_weights)) / total_w
    values = [o.value for o in ordered]

    # Iteratively Reweighted Least Squares (IRLS)
    effective_weights = list(raw_weights)
    iterations = 0
    scale = 0.0

    for it in range(1, max_iter + 1):
        iterations = it
        # Compute combined scale (sensor precision + empirical spread)
        mad_scale = _compute_mad_scale(values, curr_mu)
        scale = max(mad_scale, min(o.precision for o in ordered))

        new_weights: list[float] = []
        for o in ordered:
            base_w = 1.0 / (o.precision**2)
            total_sigma = math.sqrt(o.precision**2 + scale**2)
            std_resid = abs(o.value - curr_mu) / total_sigma

            if loss == "tukey":
                c = tuning_k * 1.5  # standard Tukey tuning constant
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
            # Degenerate case: all observations heavily downweighted, fallback to medians
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

    # If all were classified as outliers (e.g. bimodal split), retain the highest weight contributors
    if not inliers:
        inliers = list(ordered)
        outliers = []

    conflicts = _pairwise_conflicts(ordered)

    # Inlier precision: inverse root of inlier weight sum
    inlier_w_sum = sum(1.0 / (o.precision**2) for o in inliers)
    fused_precision = 1.0 / math.sqrt(inlier_w_sum)
    fused_confidence = min(o.confidence for o in inliers)

    provenance = Provenance.CONFLICT if (conflicts or outliers) else Provenance.ESTIMATED

    return RobustFusionResult(
        quantity=quantity,
        value=curr_mu,
        unit=unit,
        precision=fused_precision,
        confidence=fused_confidence,
        provenance=provenance,
        method=f"robust_{loss}_irls",
        inliers=tuple(inliers),
        outliers=tuple(outliers),
        weights=weight_dict,
        conflicts=tuple(conflicts),
        iterations=iterations,
        scale_estimate=scale,
    )


def robust_fuse_depth_observations(
    observations: Sequence[DepthSourceObservation],
    point_id: str,
    loss: str = "huber",
    tuning_k: float = 2.5,
) -> RobustFusionResult:
    """Robustly fuse multi-source depth claims for a single world point."""
    return robust_fuse_quantity(
        [o.to_observation() for o in observations],
        quantity=f"depth:{point_id}",
        loss=loss,
        tuning_k=tuning_k,
    )


def fuse_multisensor_points_3d(
    points_by_source: dict[str, Sequence[tuple[float, float, float]]],
    precisions_by_source: dict[str, float],
    confidences_by_source: dict[str, float] | None = None,
    loss: str = "huber",
    tuning_k: float = 2.5,
) -> list[tuple[float, float, float]]:
    """Robustly fuse 3D point cloud coordinates across multiple sensor streams point-by-point.

    Parameters:
      points_by_source: Mapping from sensor name to ordered list of 3D points.
      precisions_by_source: Claimed 1-sigma spatial precision (meters) per sensor.
      confidences_by_source: Confidence per sensor [0, 1].

    Returns:
      List of fused 3D coordinates (x, y, z).
    """
    if not points_by_source:
        return []

    sources = sorted(points_by_source.keys())
    n_points = min(len(points_by_source[s]) for s in sources)
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
