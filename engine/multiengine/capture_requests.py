"""Active capture: turn an unresolved problem into a specific, structured request for more evidence.

Nothing here invents a number. The "why existing images are insufficient" text is derived from measured facts about the problem
(how many registered views observed it, whether it touches the frame edge, whether a scale reference exists, ...); a rule fires only
when its fact is known. ``expected_gain`` is always ``None`` -- the system has no measured basis for predicting one -- and
``overlap_with`` names the registered images a new photo should overlap rather than quoting a percentage nobody measured.

``to_row`` maps a request onto the existing ``capture_requests`` table (apps/api/models.py ``CaptureRequest``) so the frontend lists
it through the existing contract; WorldIR remains the authoritative home of the request.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

_ARCHITECTURAL_KINDS = frozenset(
    {"incomplete_wall", "missing_corner", "missing_surface", "missing_architectural_feature", "poor_registration"}
)


@dataclass(frozen=True)
class ProblemEvidence:
    """Measured facts about the place a problem is, as far as they are known (None = not measured)."""

    n_registered_views: int | None = None  # registered cameras that observe the location
    view_ids: Tuple[str, ...] = ()  # which images those are
    truncated_at_frame_edge: bool | None = None
    scale_reference_present: bool | None = None
    pixels_on_target: float | None = None
    pixels_on_target_needed: float | None = (
        None  # supplied by the engine/benchmark that needs it; None = unknown
    )
    # dimension axes (w/d/h) with NO usable measurement; () = none known to be missing
    unmeasured_dimension_axes: Tuple[str, ...] = ()
    # True/False only when the number of independent viewpoints is known; "rear unobserved" is a statement about the EVIDENCE,
    # never that the rear is absent
    rear_observed: bool | None = None
    # True when independent estimates of the same architectural element disagree and nothing separates them
    inconsistent_estimates: bool | None = None


@dataclass(frozen=True)
class ViewSuggestion:
    kind: str  # "wider_context" | "close_up" | "new_viewpoint" | "scale_reference"
    description: str
    direction: Tuple[float, float, float] | None = (
        None  # only when it was computed from measured geometry
    )


@dataclass(frozen=True)
class CaptureRequestSpec:
    reason: str
    target_region: Dict[str, Any]  # where, in the world frame it is expressed in
    target_ids: Tuple[str, ...]
    why_insufficient: Tuple[
        str, ...
    ]  # one entry per measured fact that makes the existing evidence insufficient
    suggested_views: Tuple[ViewSuggestion, ...]
    overlap_with: Tuple[str, ...]  # registered image ids a new photo should overlap
    scale_reference_needed: bool
    instruction: str
    basis: Tuple[str, ...] = ()  # observation / measurement ids this request rests on
    expected_gain: float | None = None  # never invented

    def to_row(
        self, *, world_id: str, request_id: str, created_version: str | None = None
    ) -> Dict[str, Any]:
        return {
            "id": request_id,
            "world_id": world_id,
            "reason": self.reason[:300],
            "target_region": self.target_region,
            "target_ids": list(self.target_ids),
            "suggested_poses": [
                {
                    "kind": v.kind,
                    "description": v.description,
                    **({"direction": list(v.direction)} if v.direction else {}),
                }
                for v in self.suggested_views
            ],
            "expected_gain": self.expected_gain,
            "instruction": self.instruction,
            "status": "open",
            "created_version": created_version,
        }


def request_for_problem(
    *,
    problem_kind: str,
    location: Dict[str, Any],
    target_ids: Sequence[str],
    evidence: ProblemEvidence,
    basis: Sequence[str] = (),
) -> CaptureRequestSpec | None:
    """A structured request, or None when no measured fact shows the existing images to be insufficient (then more photos are not
    known to help and the caller should keep the problem as UNRESOLVED instead of asking for evidence it cannot justify)."""
    why: List[str] = []
    views: List[ViewSuggestion] = []
    scale_needed = False
    if evidence.n_registered_views is not None and evidence.n_registered_views < 2:
        why.append(
            f"only {evidence.n_registered_views} registered view(s) observe this location; a 3D position needs at least two"
        )
        views.append(
            ViewSuggestion(
                "new_viewpoint",
                "a photo of this location from a clearly different position than the existing "
                "view, keeping the surrounding structure in frame so it registers",
            )
        )
    if evidence.truncated_at_frame_edge:
        why.append("the structure is cut off at the edge of the frame in every view that sees it")
        views.append(
            ViewSuggestion(
                "wider_context",
                "a wider photo that includes the whole structure and the room around it",
            )
        )
    if (
        evidence.pixels_on_target is not None
        and evidence.pixels_on_target_needed is not None
        and evidence.pixels_on_target < evidence.pixels_on_target_needed
    ):
        why.append(
            f"the target covers {evidence.pixels_on_target:g} px but the engine that needs it requires "
            f"{evidence.pixels_on_target_needed:g} px"
        )
        views.append(
            ViewSuggestion(
                "close_up",
                "a close-up of the target with its surroundings still visible for registration",
            )
        )
    if evidence.scale_reference_present is False and problem_kind in (
        "incorrect_dimension",
        "metric_scale",
    ):
        why.append(
            "no scale reference of known length is visible, so the dimension cannot be made metric"
        )
        views.append(
            ViewSuggestion(
                "scale_reference",
                "a photo including an object or marker of measured length next to the target",
            )
        )
        scale_needed = True
    if evidence.unmeasured_dimension_axes:
        axes = ", ".join(evidence.unmeasured_dimension_axes)
        why.append(f"no usable measurement exists for dimension axis/axes {axes}, so the object's size is not determined")
        views.append(
            ViewSuggestion(
                "side_view",
                "a photo taken from the side of the object, level with it, so its depth and height are visible "
                "together with a surface or object of known size",
            )
        )
    if evidence.rear_observed is False:
        why.append(
            "no view observes the rear of the object; its rear geometry is UNOBSERVED (this does not mean it is absent)"
        )
        views.append(
            ViewSuggestion(
                "reverse_view",
                "a photo from the opposite side, showing the back of the object with some of the surrounding room in frame",
            )
        )
    if evidence.inconsistent_estimates and problem_kind in _ARCHITECTURAL_KINDS:
        why.append("independent estimates of this architectural element disagree and the existing views do not separate them")
        views.append(
            ViewSuggestion(
                "architectural_view",
                "another photo of the wall or opening from a different angle, with its corners and the adjoining wall visible",
            )
        )
    if not why:
        return None
    overlap = tuple(evidence.view_ids)
    instruction = "; ".join(v.description for v in views)
    if overlap:
        instruction += f". Keep part of each new photo overlapping with the existing image(s) {', '.join(overlap)}"
    return CaptureRequestSpec(
        reason=f"{problem_kind}: {why[0]}",
        target_region=dict(location),
        target_ids=tuple(target_ids),
        why_insufficient=tuple(why),
        suggested_views=tuple(views),
        overlap_with=overlap,
        scale_reference_needed=scale_needed,
        instruction=instruction,
        basis=tuple(basis),
    )
