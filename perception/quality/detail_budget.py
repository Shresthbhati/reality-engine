"""Detail-budget derivation (universal-perception directive Phase 4,
sections 10-11): the FIRST CONSUMER of the measured evidence-quality
report (perception/quality/assessment.py, P7-04).

The question answered: for a region whose evidence measured at GSD g
with view counts V and observed fraction f, what is the MAXIMUM detail
level that can be DEFENSIBLY reconstructed, and how much compute does
the region deserve?

Position in the universal pipeline: quality analysis -> (this) ->
detail discovery / ROI generation -> adaptive local reconstruction.
ROI generation consumes `compute_tier` to decide where to spend
effort; the justified level caps what local reconstruction may claim.

Honesty rules:
  - The budget is DERIVED from measured facts by a DOCUMENTED mapping
    (below). Deriving by a documented rule is not fabricating; the
    level claims what the evidence supports, nothing more.
  - No cameras/points (the assessor's empty-scene report) maps to an
    honest unsupported region with zero allocation, NOT an error.
  - A report with observed points but no measured GSD cannot come
    from `assess_evidence_quality` (it refuses intrinsics-less input);
    if one arrives here anyway it means somebody bypassed the
    assessor -- refuse (ValueError) rather than invent a budget.
  - Multi-view coverage CAPS the level: fine pixels seen by a single
    camera cannot defensibly support fine detail (single-view depth
    and relief are unreliable).

Documented GSD-level mapping (mm/px -> multi-scale level L0..L4):
      gsd <=   5  -> L4   (high-frequency: ornament, relief, engraving)
      gsd <=  25  -> L2   (meso: components, openings, fixtures)
      gsd <= 100  -> L1   (structural surfaces)
      gsd >  100  -> L0   (global/macro massing only)
Coverage caps: min observed view count >= 2 required to hold the
GSD-derived level; with < 2 the level is capped at L2; with 0
observed points the level is L0 with basis "unsupported".
Compute tiers derive from the justified level: L4 -> "full", L3 ->
"high", L2 -> "standard", L1 -> "light", L0 -> "survey" (or "none"
when nothing is observed). These tiers are inputs to ROI/adaptive
compute (directive sections 9-10), not claims of accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from perception.quality.assessment import (
    DETAIL_TIER_THRESHOLDS,
    EvidenceQualityReport,
)

#: GSD thresholds (mm/px) mapping to the maximum justified multi-scale
#: level. Aligned with the assessor's fine/medium boundaries (5/25) so
#: the two layers agree; L1/L0 extend the scale. NOT tuned against
#: real datasets -- tuning is deferred until real-capture benchmarks
#: exist (the same honesty as every other threshold in this repo).
GSD_LEVEL_THRESHOLDS = (
    (DETAIL_TIER_THRESHOLDS["fine_gsd_mm"], "L4"),
    (DETAIL_TIER_THRESHOLDS["medium_gsd_mm"], "L2"),
    (100.0, "L1"),
)

#: Minimum distinct-view count a region's points must show for the
#: GSD-derived level to hold (multi-view corroboration).
_MIN_VIEWS_FOR_LEVEL = 2

_LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4"]

_COMPUTE_TIER_BY_LEVEL = {
    "L4": "full",
    "L3": "high",
    "L2": "standard",
    "L1": "light",
    "L0": "survey",
}


@dataclass(frozen=True)
class DetailBudget:
    """Maximum defensible detail for one region, derived from measured
    evidence quality. Every field is derived by a documented rule."""

    justified_level: str  # "L0".."L4" (multi-scale levels, directive section 9)
    compute_tier: str     # "none"|"survey"|"light"|"standard"|"high"|"full"
    max_gsd_mm_per_px: Optional[float]
    coverage_capped: bool
    basis: str            # "measured" | "unsupported"

    def to_dict(self) -> dict:
        return {
            "justified_level": self.justified_level,
            "compute_tier": self.compute_tier,
            "max_gsd_mm_per_px": self.max_gsd_mm_per_px,
            "coverage_capped": self.coverage_capped,
            "basis": self.basis,
        }


def _level_from_gsd(gsd: float) -> str:
    for threshold, level in GSD_LEVEL_THRESHOLDS:
        if gsd <= threshold:
            return level
    return "L0"


def _cap_by_views(level: str, min_views: int) -> str:
    if min_views >= _MIN_VIEWS_FOR_LEVEL:
        return level
    # Single-view (or unviewed) evidence caps at L2: meso geometry is
    # defensible from one view; fine/high-frequency claims are not.
    cap = "L2"
    if _LEVEL_ORDER.index(level) > _LEVEL_ORDER.index(cap):
        return cap
    return level


def detail_budget_for(report: EvidenceQualityReport) -> DetailBudget:
    """Derive the region's detail budget from a measured quality report.

    Raises ValueError for a report that claims observed points but
    carries no measured GSD (impossible from `assess_evidence_quality`;
    means the assessor contract was bypassed)."""
    if report.gsd_mm_per_px is None:
        if report.view_counts:
            # Observed points exist but no measured GSD: the assessor
            # cannot produce this (it refuses to guess intrinsics), so
            # this report bypassed the contract. Refuse, don't invent.
            raise ValueError(
                "report has observed points but no measured GSD -- "
                "not producible by assess_evidence_quality; refusing "
                "to derive a budget from unmeasurable evidence"
            )
        return DetailBudget(
            justified_level="L0",
            compute_tier="none",
            max_gsd_mm_per_px=None,
            coverage_capped=False,
            basis="unsupported",
        )

    level = _level_from_gsd(report.gsd_mm_per_px)
    min_views = min(report.view_counts.values()) if report.view_counts else 0
    capped_level = _cap_by_views(level, min_views)
    tier = _COMPUTE_TIER_BY_LEVEL[capped_level]
    return DetailBudget(
        justified_level=capped_level,
        compute_tier=tier,
        max_gsd_mm_per_px=report.gsd_mm_per_px,
        coverage_capped=(capped_level != level),
        basis="measured",
    )
