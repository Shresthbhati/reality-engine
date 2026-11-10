"""Intelligent capture director (N-1).

Combines capture quality assessment with per-surface coverage analysis to
guide users through incomplete room captures. Identifies missing views
(corners, ceilings, floors, windows, doors, obstructed areas) and explains
what to photograph next.

Pure, testable, no hardware dependency. Honesty rules:
* Coverage estimates are plan-view approximations; furniture occlusion and
  lens FOV are not modelled. Every result is labelled provisional.
* Missing views are inferred from coverage gaps, never measured directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.capture_quality import OK, WARN, DROP, assess_capture, wall_coverage
from world_ir import Entity, EntityType


@dataclass(frozen=True)
class MissingView:
    """One view the director recommends the user take next."""
    target: str                    # "wall_3" | "object:door_001" | "ceiling" | "floor" | "corner_A" | "general"
    kind: str                      # "wall" | "object" | "ceiling" | "floor" | "corner" | "general"
    reason: str                    # human-readable, specific reason
    suggested_position: str        # e.g. "stand in the middle of the room facing wall 3"
    overlap_with: Optional[str]    # evidence id of a neighbouring photo to overlap with
    priority: int                  # 1 = do this first, higher numbers later


@dataclass(frozen=True)
class CapturePlan:
    """Complete capture state for one room/world."""
    room_id: str
    status: str                    # "incomplete" | "complete" | "no_evidence"
    missing_views: List[MissingView]
    next_best: Optional[MissingView]     # the single most important view to take next
    wall_coverage: dict            # output of wall_coverage()
    image_quality: dict            # output of assess_capture() summary
    provisional: bool              # always True until validated on real data
    notes: List[str]


@dataclass(frozen=True)
class RouteStep:
    """One step in a recommended capture route."""
    step: int
    target: str
    instruction: str
    expected_coverage_gain: str


@dataclass(frozen=True)
class CaptureRoute:
    """Ordered capture route through one room or property."""
    room_id: str
    steps: List[RouteStep]
    estimated_photos: int
    notes: List[str]


class CaptureDirector:
    """Decides what photos are still needed and in what order to take them."""

    NOT_SEEN = 0.20
    WEAK = 0.80

    def missing_views(
        self,
        room_id: str,
        footprint_xyz: Sequence[Sequence[float]],
        cameras: Sequence[dict],
        basis: Optional[dict] = None,
        world_entities: Optional[Sequence[Entity]] = None,
        photo_assessments: Optional[dict] = None,
    ) -> CapturePlan:
        """Return every missing view for a room plus the single next-best view."""
        coverage = wall_coverage(footprint_xyz, cameras, basis)
        missing: List[MissingView] = []

        if not coverage.get("available"):
            return CapturePlan(
                room_id=room_id, status="no_evidence", missing_views=[],
                next_best=None, wall_coverage=coverage,
                image_quality=photo_assessments or {"available": False},
                provisional=True,
                notes=["No room footprint available: cannot plan capture views."],
            )

        # Wall-level missing views
        for w in coverage.get("walls", []):
            if w["status"] == "not_seen":
                missing.append(MissingView(
                    target=f"wall_{w['index']}", kind="wall",
                    reason=w["hint"] or f"{w['label']} has no camera coverage.",
                    suggested_position=self._wall_position_suggestion(w),
                    overlap_with=None, priority=1,
                ))
            elif w["status"] == "weak":
                missing.append(MissingView(
                    target=f"wall_{w['index']}", kind="wall",
                    reason=f"{w['label']} has partial coverage ({w['covered_fraction']:.0%} seen, "
                           f"{w['multi_view_fraction']:.0%} multi-view).",
                    suggested_position=self._wall_position_suggestion(w),
                    overlap_with=None, priority=2,
                ))

        # Object-specific missing views
        if world_entities:
            seen_wall_ids = {w["index"] for w in coverage.get("walls", [])
                             if w["status"] == "seen"}
            for ent in world_entities:
                wall_ref = ent.custom_properties.get("in_wall", {}).get("wall_id")
                if not wall_ref:
                    continue
                try:
                    wall_index = int(wall_ref.rsplit("-", 1)[-1])
                except (ValueError, IndexError):
                    continue
                if wall_index in seen_wall_ids:
                    continue
                kind_map = {
                    EntityType.DOOR: "door", EntityType.WINDOW: "window",
                    EntityType.SWITCH: "switch", EntityType.SOCKET: "socket",
                }
                obj_kind = kind_map.get(ent.type, "object")
                missing.append(MissingView(
                    target=f"object:{ent.id}", kind="object",
                    reason=f"{obj_kind} '{ent.name or ent.id}' sits on {wall_ref} "
                           f"which has no reliable coverage yet.",
                    suggested_position=(f"Photograph {wall_ref} squarely so the "
                                        f"{obj_kind} at {wall_ref} is in frame."),
                    overlap_with=None, priority=1,
                ))

        # Ceiling / floor prompts
        missing.append(MissingView(
            target="ceiling", kind="ceiling",
            reason="Ceiling plane not yet observed. Photograph ceiling edges from the middle of the room.",
            suggested_position="Stand in the middle of the room, tilt the camera up.",
            overlap_with=None, priority=3,
        ))
        missing.append(MissingView(
            target="floor", kind="floor",
            reason="Floor plane not yet observed. Photograph floor edges from standing height.",
            suggested_position="Stand in the middle of the room, point the camera down.",
            overlap_with=None, priority=3,
        ))

        # Corner prompts (one corner view covers two unseen walls)
        corners = coverage.get("corners", [])
        unseen_walls = {w["index"] for w in coverage.get("walls", [])
                        if w["status"] == "not_seen"}
        for i, c in enumerate(corners):
            w1, w2 = i + 1, (i % len(corners)) + 1
            if w1 in unseen_walls and w2 in unseen_walls:
                missing.append(MissingView(
                    target=f"corner_{c['label']}", kind="corner",
                    reason=f"Corner {c['label']} joins two unseen walls (wall {w1} and wall {w2}). "
                           f"One corner photograph covers both.",
                    suggested_position=f"Stand so corner {c['label']} fills the frame.",
                    overlap_with=None, priority=1,
                ))
                break

        # Next best view
        missing.sort(key=lambda m: (m.priority, m.target))
        next_best = missing[0] if missing else None

        # Fold image quality into notes
        quality_notes: List[str] = []
        if photo_assessments and photo_assessments.get("available"):
            s = photo_assessments.get("summary", {})
            if s.get("DROP", 0):
                quality_notes.append(
                    f"{s['DROP']} photo(s) will probably be dropped "
                    f"(blurry, overexposed, or no overlap): retake those views."
                )
            if s.get("WARN", 0):
                quality_notes.append(
                    f"{s['WARN']} photo(s) could be better: consider retaking "
                    f"the weakest ones after the missing views above."
                )

        status = "complete" if not missing else "incomplete"
        notes = list(coverage.get("notes", [])) + quality_notes
        if not cameras:
            notes.append("No registered cameras yet: take the first photos from "
                         "the middle of the room, overlapping neighbouring views.")

        return CapturePlan(
            room_id=room_id, status=status, missing_views=missing,
            next_best=next_best, wall_coverage=coverage,
            image_quality=photo_assessments or {"available": False},
            provisional=True, notes=notes,
        )

    def route(
        self,
        room_id: str,
        footprint_xyz: Sequence[Sequence[float]],
        cameras: Sequence[dict],
        basis: Optional[dict] = None,
        world_entities: Optional[Sequence[Entity]] = None,
    ) -> CaptureRoute:
        """An ordered capture route through one room."""
        plan = self.missing_views(
            room_id, footprint_xyz, cameras, basis, world_entities)
        steps: List[RouteStep] = []
        n = 0
        for mv in plan.missing_views:
            n += 1
            steps.append(RouteStep(
                step=n, target=mv.target,
                instruction=mv.reason,
                expected_coverage_gain=mv.suggested_position,
            ))
        est_photos = len(steps)
        notes = list(plan.notes)
        notes.append("Route is a suggestion: skip steps you cannot take, and the "
                     "next-best view recomputes after every photo.")
        return CaptureRoute(
            room_id=room_id, steps=steps, estimated_photos=est_photos, notes=notes,
        )

    @staticmethod
    def _wall_position_suggestion(wall: dict) -> str:
        mid_x = (wall["p0"][0] + wall["p1"][0]) / 2
        mid_y = (wall["p0"][1] + wall["p1"][1]) / 2
        return (f"Stand roughly {wall['length']:.1f} m back from {wall['label']} "
                f"near plan position ({mid_x:.2f}, {mid_y:.2f}), facing the wall "
                f"squarely so it fills most of the frame.")


__all__ = ["CaptureDirector", "CapturePlan", "CaptureRoute", "MissingView", "RouteStep"]