"""Active Perception & Refinement Controller (A3.10).

Identifies:
  - Missing architectural evidence (e.g., room with furniture but no wall/floor/opening)
  - Low-confidence / uncalibrated uncertain detections
  - Conflicting detections (competing labels)
  - Incomplete / missing masks
  - Ambiguous cross-view identities
  - Regions needing higher-resolution processing

Prioritizes actions by expected information gain vs computational cost and knows
when to stop (diminishing returns / budget exhausted).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from perception.intelligence.data_model import (
    ArchitecturalObservation,
    CrossViewIdentityHypothesis,
    NormalizedPerceptionObservation,
)


@dataclass(frozen=True)
class RefinementAction:
    action_id: str
    action_kind: str  # "reprocess_crop" | "run_segmenter" | "second_opinion_detector" | "request_camera_view" | "resolve_identity"
    target_id: str
    expected_info_gain: float
    estimated_cost: float
    priority_score: float
    reason: str
    roi_px: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "target_id": self.target_id,
            "expected_info_gain": self.expected_info_gain,
            "estimated_cost": self.estimated_cost,
            "priority_score": self.priority_score,
            "reason": self.reason,
            "roi_px": list(self.roi_px) if self.roi_px else None,
        }


@dataclass(frozen=True)
class RefinementPlan:
    actions: tuple[RefinementAction, ...]
    should_continue: bool
    stop_reason: str
    summary: dict[str, Any]


def plan_active_refinement(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    architectural: Sequence[ArchitecturalObservation] = (),
    identities: Sequence[CrossViewIdentityHypothesis] = (),
    compute_budget: float = 10.0,
    min_priority_ratio: float = 0.25,
) -> RefinementPlan:
    """Rank refinement actions by info_gain / cost and halt when diminishing returns are reached."""
    candidates: list[RefinementAction] = []

    # 1. Missing architectural baseline check
    arch_labels = {o.canonical_label for o in observations if o.category == "architecture"}
    if observations and not (arch_labels & {"wall", "floor", "ceiling"}):
        candidates.append(RefinementAction(
            action_id="ref_missing_arch_envelope",
            action_kind="request_camera_view",
            target_id="scene_envelope",
            expected_info_gain=0.90,
            estimated_cost=2.0,
            priority_score=0.45,
            reason="objects_detected_without_wall_floor_or_ceiling_context",
        ))

    # 2. Per-observation checks: missing masks, competing labels, low-score crops
    for o in observations:
        if o.segmentation is None:
            x1, y1, x2, y2 = (int(round(v)) for v in o.bbox_px)
            candidates.append(RefinementAction(
                action_id=f"ref_mask_{o.observation_id}",
                action_kind="run_segmenter",
                target_id=o.observation_id,
                expected_info_gain=0.65,
                estimated_cost=0.8,
                priority_score=round(0.65 / 0.8, 4),
                reason=f"missing_instance_mask_for_{o.canonical_label}",
                roi_px=(x1, y1, x2, y2),
            ))
        if o.competing_labels:
            x1, y1, x2, y2 = (int(round(v)) for v in o.bbox_px)
            candidates.append(RefinementAction(
                action_id=f"ref_conflict_{o.observation_id}",
                action_kind="second_opinion_detector",
                target_id=o.observation_id,
                expected_info_gain=0.80,
                estimated_cost=1.5,
                priority_score=round(0.80 / 1.5, 4),
                reason=f"competing_labels_on_{o.canonical_label}",
                roi_px=(x1, y1, x2, y2),
            ))
        raw = o.scores.raw_score
        if raw is not None and raw < 0.35 and (o.bbox_norm[2] - o.bbox_norm[0]) < 0.15:
            x1, y1, x2, y2 = (int(round(v)) for v in o.bbox_px)
            candidates.append(RefinementAction(
                action_id=f"ref_crop_{o.observation_id}",
                action_kind="reprocess_crop",
                target_id=o.observation_id,
                expected_info_gain=0.55,
                estimated_cost=1.2,
                priority_score=round(0.55 / 1.2, 4),
                reason=f"small_low_score_{o.canonical_label}",
                roi_px=(x1, y1, x2, y2),
            ))

    # 3. Unverified architectural openings needing frame/view verification
    for ao in architectural:
        if ao.structural_role == "opening_candidate":
            x1, y1, x2, y2 = (int(round(v)) for v in ao.observation.bbox_px)
            candidates.append(RefinementAction(
                action_id=f"ref_opening_{ao.observation.observation_id}",
                action_kind="reprocess_crop",
                target_id=ao.observation.observation_id,
                expected_info_gain=0.70,
                estimated_cost=1.0,
                priority_score=0.70,
                reason="unverified_opening_candidate_needs_frame_or_mask",
                roi_px=(x1, y1, x2, y2),
            ))

    # 4. Ambiguous cross-view identities
    for ident in identities:
        if ident.status == "ambiguous":
            candidates.append(RefinementAction(
                action_id=f"ref_id_{ident.identity_id}",
                action_kind="resolve_identity",
                target_id=ident.identity_id,
                expected_info_gain=0.75,
                estimated_cost=1.8,
                priority_score=round(0.75 / 1.8, 4),
                reason="ambiguous_cross_view_identity",
            ))

    # Filter by minimum priority ratio and pack within compute_budget
    viable = [c for c in candidates if c.priority_score >= min_priority_ratio]
    viable.sort(key=lambda c: c.priority_score, reverse=True)

    selected: list[RefinementAction] = []
    spent = 0.0
    for act in viable:
        if spent + act.estimated_cost <= compute_budget:
            selected.append(act)
            spent += act.estimated_cost

    should_continue = len(selected) > 0
    stop_reason = (
        "actions_queued_within_budget"
        if should_continue
        else ("no_viable_refinement_needed" if not viable else "budget_exhausted")
    )
    return RefinementPlan(
        actions=tuple(selected),
        should_continue=should_continue,
        stop_reason=stop_reason,
        summary={
            "total_candidates": len(candidates),
            "selected_actions": len(selected),
            "estimated_cost_total": round(spent, 2),
            "budget": compute_budget,
        },
    )


__all__ = [
    "RefinementAction",
    "RefinementPlan",
    "plan_active_refinement",
]
