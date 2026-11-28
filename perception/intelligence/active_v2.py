"""Active Perception 2.0 (A4.10).

Identifies missing views, poorly observed walls, uncertain openings/furniture,
incomplete masks, conflicting identities, and low-quality photographs, and
prioritizes targeted actions by information gain per unit compute cost with
an explicit stopping criterion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from perception.intelligence.data_model import (
    ArchitecturalObservation,
    CrossViewIdentityHypothesis,
    NormalizedPerceptionObservation,
)
from perception.intelligence.refinement import RefinementAction, RefinementPlan, plan_active_refinement


@dataclass(frozen=True)
class ActivePerceptionReport:
    base_plan: RefinementPlan
    low_quality_images: tuple[str, ...]
    poorly_observed_walls: tuple[str, ...]
    uncertain_openings: tuple[str, ...]
    uncertain_furniture: tuple[str, ...]
    recommended_next_step: str


def run_active_perception_v2(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    architectural: Sequence[ArchitecturalObservation] = (),
    identities: Sequence[CrossViewIdentityHypothesis] = (),
    image_qualities: Mapping[str, Mapping[str, Any]] | None = None,
    compute_budget: float = 12.0,
) -> ActivePerceptionReport:
    """Compute comprehensive active refinement priorities and stopping decision."""
    base = plan_active_refinement(
        observations,
        architectural=architectural,
        identities=identities,
        compute_budget=compute_budget,
    )
    low_q = tuple(
        img_id for img_id, q in sorted((image_qualities or {}).items())
        if q.get("quality_grade") in ("degraded", "unusable_blank") or q.get("is_blurry") or q.get("is_low_contrast")
    )
    poor_walls = tuple(
        ao.observation.observation_id for ao in architectural
        if ao.observation.canonical_label in ("wall", "wall_segment") and (ao.boundary_continuity_score or 0.0) < 0.30
    )
    unc_openings = tuple(
        ao.observation.observation_id for ao in architectural
        if ao.structural_role == "opening_candidate"
    )
    unc_furn = tuple(
        o.observation_id for o in observations
        if o.category == "furniture" and ((o.scores.raw_score is not None and o.scores.raw_score < 0.40) or bool(o.competing_labels))
    )

    if not base.should_continue and not low_q and not poor_walls:
        next_step = "stop_sufficient_evidence"
    elif unc_openings or unc_furn:
        next_step = "run_targeted_crops_and_second_opinion"
    elif low_q:
        next_step = "request_sharper_views"
    else:
        next_step = "execute_queued_refinement_actions"

    return ActivePerceptionReport(
        base_plan=base,
        low_quality_images=low_q,
        poorly_observed_walls=poor_walls,
        uncertain_openings=unc_openings,
        uncertain_furniture=unc_furn,
        recommended_next_step=next_step,
    )


__all__ = [
    "ActivePerceptionReport",
    "run_active_perception_v2",
]
