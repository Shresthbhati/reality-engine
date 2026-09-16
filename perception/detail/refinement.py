"""ROI refinement executor (universal-perception directive sections
11/13; P7-06's declared open item): pending RegionOfInterest work
orders become LOCALLY REFINED geometry or HONEST REFUSALS -- never
invented geometry.

Position in the universal pipeline:

    detail discovery -> ROI generation (perception/detail/roi.py)
      -> (this module) local reconstruction / refinement
      -> adaptive refinement / WorldIR integration

Contract:
  - Only REAL points refine. The executor resolves the ROI's
    point_ids through the caller-supplied `point_lookup`; it NEVER
    fabricates geometry from ROI metadata (bounds, curvature,
    budget) alone. Evidence that cannot be resolved is refused with
    a diagnostic naming the missing ids.
  - The compute_tier SPENDS effort (documented policy):
        none      -> refuse (zero local effort)
        survey    -> plane only
        light     -> plane only
        standard  -> plane
        high      -> plane -> cylinder -> sphere
        full      -> plane -> cylinder -> sphere
    (survey/light tier is measured but cannot justify escalation;
    high/full add the curved backends, ordered by expected cost.)
    Measured counters record what actually ran.
  - The WINNER is the measured-residual minimum across succeeded
    backends; its rms/max are the outcome's residuals and its
    quality is the documented monotone map 1/(1+(rms/tol)^2).
    Quality is never set to a constant, and never reported when no
    fit succeeded.
  - Status transitions carry the evidence: success -> "refined" +
    outcome; any failure -> "refused" + diagnostic reason. The
    input work orders are never mutated (outcomes are new records).
  - Deterministic: same input -> byte-identical outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from perception.detail.backends import (
    FitRefused,
    PlaneFit,
    fit_cylinder_on,
    fit_plane,
    fit_sphere_on,
)
from perception.detail.roi import RegionOfInterest
from perception.architecture.parametric import CylinderFit, SphereFit

__all__ = [
    "RefinementOutcome",
    "RefinementResult",
    "refine_rois",
    "apply_outcomes",
]

#: Quality tolerance scale for the documented map
#: quality = 1 / (1 + (rms/tol)^2) -- at rms == tol the quality is
#: 0.5. Same measurement-tolerance convention as parametric.py.
_QUALITY_TOLERANCE_M = 0.1

#: ROI metadata alone is not evidence: below this resolved-support
#: count a work order refuses rather than fits (the fit backends
#: have their own, stricter minimums; this is the executor's floor).
_MIN_RESOLVED_POINTS = 5

#: Backends per compute tier, in deterministic spend order. Levels
#: below "standard" cannot justify more than the cheap plane probe.
_TIER_BACKENDS: Dict[str, Tuple[str, ...]] = {
    "none": (),
    "survey": ("plane",),
    "light": ("plane",),
    "standard": ("plane",),
    "high": ("plane", "cylinder", "sphere"),
    "full": ("plane", "cylinder", "sphere"),
}


@dataclass(frozen=True)
class RefinementResult:
    """Measured outcome of ONE successful backend attempt."""

    backend: str            # "plane" | "cylinder" | "sphere"
    rms_residual_m: float
    max_residual_m: float
    quality: float
    n_points: int
    fit: object             # the winning backend's fit record

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "rms_residual_m": self.rms_residual_m,
            "max_residual_m": self.max_residual_m,
            "quality": self.quality,
            "n_points": self.n_points,
            "fit": self.fit.to_dict(),
        }


@dataclass(frozen=True)
class RefinementOutcome:
    """Evidence-carrying status transition for one ROI work order."""

    roi_id: str
    status: str             # "refined" | "refused"
    reason: Optional[str]   # diagnostic when refused
    refinement: Optional[RefinementResult]
    fits_attempted: int     # measured effort counters (outcome-level)
    fits_succeeded: int
    backends_run: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "roi_id": self.roi_id,
            "status": self.status,
            "reason": self.reason,
            "refinement": (
                self.refinement.to_dict() if self.refinement else None
            ),
            "fits_attempted": self.fits_attempted,
            "fits_succeeded": self.fits_succeeded,
            "backends_run": list(self.backends_run),
        }


def apply_outcomes(
    rois: Sequence[RegionOfInterest],
    outcomes: Sequence[RefinementOutcome],
) -> List[RegionOfInterest]:
    """Apply refinement outcomes to work orders: the state-transition
    ownership `roi.py` declares (pending -> refined/refused belongs to
    local refinement).

    Returns NEW RegionOfInterest records (input is never mutated).
    Outcomes for unknown roi_ids are ignored; a ROI with no outcome
    stays pending. For "refined", the outcome rides along as the
    record's refinement_outcome evidence; for "refused", the
    diagnostic rides along as status_reason.
    """
    by_id = {o.roi_id: o for o in outcomes}
    updated: List[RegionOfInterest] = []
    for roi in rois:
        outcome = by_id.get(roi.roi_id)
        if outcome is None:
            updated.append(roi)
            continue
        if outcome.status == "refined":
            updated.append(RegionOfInterest(
                roi_id=roi.roi_id,
                parent_entity_id=roi.parent_entity_id,
                bounds=roi.bounds,
                n_cells=roi.n_cells,
                n_points=roi.n_points,
                detail_cells=roi.detail_cells,
                point_ids=roi.point_ids,
                budget=roi.budget,
                max_curvature=roi.max_curvature,
                status="refined",
                provenance=dict(roi.provenance),
                refinement_outcome=outcome.refinement,
                status_reason=None,
            ))
        else:
            updated.append(RegionOfInterest(
                roi_id=roi.roi_id,
                parent_entity_id=roi.parent_entity_id,
                bounds=roi.bounds,
                n_cells=roi.n_cells,
                n_points=roi.n_points,
                detail_cells=roi.detail_cells,
                point_ids=roi.point_ids,
                budget=roi.budget,
                max_curvature=roi.max_curvature,
                status="refused",
                provenance=dict(roi.provenance),
                refinement_outcome=None,
                status_reason=outcome.reason,
            ))
    return updated


def _quality_from_rms(rms: float) -> float:
    return 1.0 / (1.0 + (rms / _QUALITY_TOLERANCE_M) ** 2)


def _run_backend(
    name: str,
    pts: Sequence[Tuple[float, float, float]],
    up: Tuple[float, float, float],
):
    if name == "plane":
        return fit_plane(pts)
    if name == "cylinder":
        return fit_cylinder_on(pts, up)
    if name == "sphere":
        return fit_sphere_on(pts)
    raise ValueError(f"unknown backend: {name}")


def refine_rois(
    rois: Sequence[RegionOfInterest],
    point_lookup: Callable[[str], Optional[Tuple[float, float, float]]],
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> List[RefinementOutcome]:
    """Execute pending ROI work orders against resolvable evidence.

    `point_lookup` maps a point id to its position (or None when the
    id cannot be resolved). Returns one outcome per input ROI, in
    input order; input ROIs are never mutated.
    """
    outcomes: List[RefinementOutcome] = []
    for roi in rois:
        outcomes.append(_refine_one(roi, point_lookup, up))
    return outcomes


def _refine_one(
    roi: RegionOfInterest,
    point_lookup: Callable[[str], Optional[Tuple[float, float, float]]],
    up: Tuple[float, float, float],
) -> RefinementOutcome:
    tier = roi.budget.compute_tier if roi.budget is not None else "none"
    if tier not in _TIER_BACKENDS:
        return RefinementOutcome(
            roi_id=roi.roi_id,
            status="refused",
            reason=f"unknown compute_tier {tier!r}",
            refinement=None,
            fits_attempted=0,
            fits_succeeded=0,
            backends_run=(),
        )
    if tier == "none":
        return RefinementOutcome(
            roi_id=roi.roi_id,
            status="refused",
            reason="compute_tier=none: zero local effort justified by "
                   "evidence budget",
            refinement=None,
            fits_attempted=0,
            fits_succeeded=0,
            backends_run=(),
        )

    # Resolve evidence: real points only.
    positions: List[Tuple[float, float, float]] = []
    missing: List[str] = []
    for pid in roi.point_ids:
        pos = point_lookup(pid)
        if pos is None:
            missing.append(pid)
        else:
            positions.append(tuple(float(c) for c in pos))
    if missing:
        return RefinementOutcome(
            roi_id=roi.roi_id,
            status="refused",
            reason=(
                f"evidence unresolvable: {len(missing)} of "
                f"{len(roi.point_ids)} point_ids missing "
                f"(first: {missing[0]!r})"
            ),
            refinement=None,
            fits_attempted=0,
            fits_succeeded=0,
            backends_run=(),
        )
    if len(positions) < _MIN_RESOLVED_POINTS:
        return RefinementOutcome(
            roi_id=roi.roi_id,
            status="refused",
            reason=(
                f"insufficient resolved support: {len(positions)} "
                f"points (minimum {_MIN_RESOLVED_POINTS})"
            ),
            refinement=None,
            fits_attempted=0,
            fits_succeeded=0,
            backends_run=(),
        )

    # Spend the tier's effort, in documented order.
    attempted = 0
    succeeded: List[Tuple[str, object, float, float]] = []
    backends_run: List[str] = []
    for name in _TIER_BACKENDS[tier]:
        backends_run.append(name)
        attempted += 1
        try:
            fit = _run_backend(name, positions, up)
        except FitRefused as exc:
            continue  # measured refusal: recorded in counters only
        rms = float(fit.rms_residual_m)
        mx = float(getattr(fit, "max_residual_m", rms))
        succeeded.append((name, fit, rms, mx))

    if not succeeded:
        return RefinementOutcome(
            roi_id=roi.roi_id,
            status="refused",
            reason=(
                f"all {attempted} backend fit(s) refused: no defensible "
                "local model for this support"
            ),
            refinement=None,
            fits_attempted=attempted,
            fits_succeeded=0,
            backends_run=tuple(backends_run),
        )

    # Winner: measured-rms minimum. Ties broken deterministically by
    # documented spend order (stable min over the ordered list).
    best_name, best_fit, best_rms, best_mx = min(
        succeeded, key=lambda t: t[2]
    )
    result = RefinementResult(
        backend=best_name,
        rms_residual_m=best_rms,
        max_residual_m=best_mx,
        quality=_quality_from_rms(best_rms),
        n_points=len(positions),
        fit=best_fit,
    )
    return RefinementOutcome(
        roi_id=roi.roi_id,
        status="refined",
        reason=None,
        refinement=result,
        fits_attempted=attempted,
        fits_succeeded=len(succeeded),
        backends_run=tuple(backends_run),
    )
