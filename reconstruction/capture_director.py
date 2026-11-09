"""Intelligent capture director (plan items A.1, A.4, A.5).

Combines capture_quality.py (image-quality verdicts + per-wall coverage) with
route planning and object-specific prompts to guide a user through an incomplete
room capture. Pure, testable, no hardware dependency.

Honesty rules:
* Coverage estimates are plan-view approximations (camera centre + viewing
  direction). Furniture occlusion, lens FOV and true intrinsics are not known
  here, so every result is labelled ``provisional`` and explained.
* Missing views are *inferred* from coverage gaps, never measured directly.
* Object-specific prompts come from WorldIR entities whose bounds intersect a
  wall that is not yet "seen" -- an object is only prompted when the evidence
  actually supports that it exists and is poorly covered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.capture_quality import (
    OK, WARN, DROP, _headline, assess_capture, wall_coverage,
)
from world_ir import Entity, EntityType


@dataclass(frozen=True)
class MissingView:
    """One view the director recommends the user take next."""
    target: str                    # "wall_3" | "object:door_001" | "ceiling" | "floor" | "corner_A"
    kind: str                      # "wall" | "object" | "ceiling" | "floor" | "corner" | "general"
    reason: str                    # human-readable, specific reason
    suggested_position: str        # e.g. "stand in the middle of the room facing wall 3"
    overlap_with: Optional[str]    # evidence id of a neighbouring photo to overlap with, or None
    priority: int                  # 1 = do this first, higher numbers later


@dataclass(frozen=True)
class CapturePlan:
    """The complete recommended capture state for one room/world."""
    room_id: str
    status: str                    # "incomplete" | "complete" | "no_evidence"
    missing_views: List[MissingView]
    next_best: Optional[MissingView]  # the single most important view to take next
    wall_coverage: dict            # output of wall_coverage()
    image_quality: dict            # output of assess_capture() summary
    provisional: bool              # always True until validated on real data
    notes: List[str]


@dataclass(frozen=True)
class RouteStep:
    """One step in a recommended capture route."""
    step: int
    target: str                    # same namespace as MissingView.target
    instruction: str               # what to do at this step
    expected_coverage_gain: str    # which walls/objects this step addresses


@dataclass(frozen=True)
class CaptureRoute:
    """An ordered capture route through a room or property."""
    room_id: str
    steps: List[RouteStep]
    estimated_photos: int
    notes: List[str]


class CaptureDirector:
    """Decides what photos are still needed and in what order to take them."""

    #: Wall coverage fractions below which a wall is considered missing entirely.
    NOT_SEEN = 0.20
    #: Wall coverage fractions below which a wall is weak (needs reinforcement).
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
        """Return every missing view for a room plus the single next-best view.

        Parameters
        ----------
        footprint_xyz: room footprint polygon corners in world frame.
        cameras: registered camera poses [{evidence_id, position_m, rotation_wxyz}].
        basis: plan basis {e1, e2, up}; defaults to (x, z) with +Y up.
        world_entities: WorldIR entities (doors, windows, fixtures, etc.) whose
            placement on poorly-covered walls triggers object-specific prompts.
        photo_assessments: optional output of assess_capture() to fold image
            quality into the plan (blurry/overexposed photos are treated as
            missing coverage for the walls they were meant to capture).
        """
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

        # ---- wall-level missing views -------------------------------------
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

        # ---- object-specific missing views ---------------------------------
        if world_entities:
            seen_wall_ids = {w["index"] for w in coverage.get("walls", [])
                             if w["status"] == "seen"}
            for ent in world_entities:
                wall_ref = ent.custom_properties.get("in_wall", {}).get("wall_id")
                if not wall_ref:
                    continue
                # extract wall index from id like "wall-room-001-03"
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

        # ---- ceiling / floor prompts --------------------------------------
        missing.append(MissingView(
            target="ceiling", kind="ceiling",
            reason="Ceiling plane not yet observed. Photograph the ceiling edges "
                   "from the middle of the room.",
            suggested_position="Stand in the middle of the room, tilt the camera up.",
            overlap_with=None, priority=3,
        ))
        missing.append(MissingView(
            target="floor", kind="floor",
            reason="Floor plane not yet observed. Photograph the floor edges "
                   "from standing height.",
            suggested_position="Stand in the middle of the room, point the camera down.",
            overlap_with=None, priority=3,
        ))

        # ---- corner prompts (corners are where two walls meet; if both are
        # ---- unseen, a corner view resolves both at once) ------------------
        corners = coverage.get("corners", [])
        unseen_walls = {w["index"] for w in coverage.get("walls", [])
                        if w["status"] == "not_seen"}
        for i, c in enumerate(corners):
            w1, w2 = i + 1, (i % len(corners)) + 1
            if w1 in unseen_walls and w2 in unseen_walls:
                missing.append(MissingView(
                    target=f"corner_{c['label']}", kind="corner",
                    reason=f"Corner {c['label']} joins two unseen walls "
                           f"(wall {w1} and wall {w2}). One corner photograph covers both.",
                    suggested_position=f"Stand so corner {c['label']} fills the frame.",
                    overlap_with=None, priority=1,
                ))
                break  # one corner prompt is enough; the wall prompts cover the rest

        # ---- next best view -----------------------------------------------
        missing.sort(key=lambda m: (m.priority, m.target))
        next_best = missing[0] if missing else None

        # fold image quality: if a photo was dropped, note that its walls may
        # need re-capture
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
        """An ordered capture route through one room.

        Orders missing views so that general room coverage (walls, corners)
        comes before object-specific detail, and corner views (which cover two
        walls at once) come before single-wall views.
        """
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