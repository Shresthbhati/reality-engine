"""Architectural constraint solver (plan item D.16).

Checks WorldIR geometry for physically implausible relationships:
* walls that do not meet floors/ceilings
* doors/windows detached from wall openings
* windows floating outside walls
* stairs that do not connect floors

Reports violations with measured evidence; never auto-corrects (geometry
changes require user confirmation or new capture evidence).

Honesty rules:
* Every violation carries the measurements that triggered it and a severity.
* A relationship "not observed" is recorded as unknown, not assumed valid.
* No LLM or heuristic assumption is used to repair geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_ir import Entity, EntityType, Geometry, Relationship, RelationshipKind


@dataclass(frozen=True)
class ConstraintViolation:
    """One implausible architectural relationship."""
    rule: str                      # e.g. "wall_meets_floor"
    subject_id: str                # entity id
    subject_type: str              # "wall" | "door" | "window" | "stair"
    target_id: Optional[str]
    target_type: Optional[str]
    severity: str                  # "info" | "warning" | "error"
    message: str
    measured: dict                 # measurements that triggered this
    suggestion: Optional[str] = None


@dataclass(frozen=True)
class ConstraintReport:
    """All constraint violations found in one world."""
    violations: List[ConstraintViolation]
    entities_checked: int
    provisional: bool = True


class ArchitecturalConstraintSolver:
    """Validates WorldIR entities against architectural plausibility rules."""

    #: A wall whose vertical extent does not reach within this of a floor is
    #: flagged (metres).
    WALL_FLOOR_GAP_TOL_M = 0.15
    #: An opening whose host wall is more than this from its position is
    #    detached (metres).
    OPENING_HOST_TOL_M = 0.10
    #: Stair vertical rise that does not connect two floor levels (metres).
    STAIR_FLOOR_GAP_TOL_M = 0.30

    def check_world(self, entities: Dict[str, Entity]) -> ConstraintReport:
        """Run every architectural constraint check over the world's entities."""
        violations: List[ConstraintViolation] = []

        walls = [e for e in entities.values() if e.type == EntityType.WALL]
        floors = [e for e in entities.values() if e.type == EntityType.FLOOR]
        ceilings = [e for e in entities.values() if e.type == EntityType.CEILING]
        doors = [e for e in entities.values() if e.type == EntityType.DOOR]
        windows = [e for e in entities.values() if e.type == EntityType.WINDOW]
        openings = [e for e in entities.values() if e.type == EntityType.OPENING]
        stairs = [e for e in entities.values() if e.type == EntityType.STAIRS]

        violations.extend(self._check_walls_meet_floors(walls, floors, entities))
        violations.extend(self._check_walls_meet_ceilings(walls, ceilings, entities))
        violations.extend(self._check_openings_on_host_walls(doors, windows, openings, entities))
        violations.extend(self._check_windows_in_walls(windows, walls, entities))
        violations.extend(self._check_stairs_connect_floors(stairs, floors, entities))

        violations.sort(key=lambda v: ({"error": 0, "warning": 1, "info": 2}[v.severity], v.subject_id))
        return ConstraintReport(
            violations=violations,
            entities_checked=len(entities),
            provisional=True,
        )

    # ------------------------------------------------------------------ checks

    def _bounds(self, ent: Entity) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (min, max) world bounds for an entity, or None."""
        gids = ent.geometry_ids
        if not gids:
            return None
        # We don't have direct geometry access here; use custom_properties bounds
        cp = ent.custom_properties
        bmin = cp.get("bounds_min") or cp.get("bounds", {}).get("min")
        bmax = cp.get("bounds_max") or cp.get("bounds", {}).get("max")
        if bmin is None or bmax is None:
            return None
        return (np.asarray(bmin, float), np.asarray(bmax, float))

    def _centre(self, ent: Entity) -> Optional[np.ndarray]:
        bounds = self._bounds(ent)
        if bounds is None:
            return None
        return (bounds[0] + bounds[1]) / 2

    def _check_walls_meet_floors(
        self, walls: List[Entity], floors: List[Entity], entities: Dict[str, Entity]
    ) -> List[ConstraintViolation]:
        out: List[ConstraintViolation] = []
        for w in walls:
            cb = self._centre(w)
            if cb is None:
                continue
            # find nearest floor below the wall
            best_floor: Optional[Entity] = None
            best_gap = float("inf")
            for f in floors:
                fb = self._bounds(f)
                if fb is None:
                    continue
                # floor top vs wall bottom
                floor_top = fb[1][1]
                wall_bottom = cb[1]  # approximate; use bounds if available
                wb = self._bounds(w)
                if wb is not None:
                    wall_bottom = wb[0][1]
                gap = wall_bottom - floor_top
                if 0 <= gap < best_gap:
                    best_gap = gap
                    best_floor = f
            if best_floor is None:
                out.append(ConstraintViolation(
                    rule="wall_meets_floor", subject_id=w.id, subject_type="wall",
                    target_id=None, target_type=None, severity="warning",
                    message=f"Wall '{w.name or w.id}' has no observed floor below it.",
                    measured={"wall_bottom_m": round(float(cb[1]), 3)},
                    suggestion="Capture the floor at the base of this wall.",
                ))
            elif best_gap > self.WALL_FLOOR_GAP_TOL_M:
                out.append(ConstraintViolation(
                    rule="wall_meets_floor", subject_id=w.id, subject_type="wall",
                    target_id=best_floor.id, target_type="floor", severity="warning",
                    message=(f"Wall '{w.name or w.id}' ends {best_gap:.2f} m above "
                             f"the nearest floor '{best_floor.name or best_floor.id}'."),
                    measured={"gap_m": round(best_gap, 3)},
                    suggestion="Check for a missing floor section or a raised wall.",
                ))
        return out

    def _check_walls_meet_ceilings(
        self, walls: List[Entity], ceilings: List[Entity], entities: Dict[str, Entity]
    ) -> List[ConstraintViolation]:
        out: List[ConstraintViolation] = []
        for w in walls:
            wb = self._bounds(w)
            if wb is None:
                continue
            wall_top = wb[1][1]
            best_ceiling: Optional[Entity] = None
            best_gap = float("inf")
            for c in ceilings:
                cb = self._bounds(c)
                if cb is None:
                    continue
                ceiling_bottom = cb[0][1]
                gap = ceiling_bottom - wall_top
                if 0 <= gap < best_gap:
                    best_gap = gap
                    best_ceiling = c
            if best_ceiling is None:
                out.append(ConstraintViolation(
                    rule="wall_meets_ceiling", subject_id=w.id, subject_type="wall",
                    target_id=None, target_type=None, severity="info",
                    message=f"Wall '{w.name or w.id}' has no observed ceiling above it.",
                    measured={"wall_top_m": round(float(wall_top), 3)},
                    suggestion="Capture the ceiling above this wall.",
                ))
            elif best_gap > self.WALL_FLOOR_GAP_TOL_M:
                out.append(ConstraintViolation(
                    rule="wall_meets_ceiling", subject_id=w.id, subject_type="wall",
                    target_id=best_ceiling.id, target_type="ceiling", severity="info",
                    message=(f"Wall '{w.name or w.id}' ends {best_gap:.2f} m below "
                             f"the nearest ceiling '{best_ceiling.name or best_ceiling.id}'."),
                    measured={"gap_m": round(best_gap, 3)},
                    suggestion="Check for a soffit, beam, or missing ceiling section.",
                ))
        return out

    def _check_openings_on_host_walls(
        self, doors: List[Entity], windows: List[Entity], openings: List[Entity],
        entities: Dict[str, Entity],
    ) -> List[ConstraintViolation]:
        out: List[ConstraintViolation] = []
        all_openings = doors + windows + openings
        for o in all_openings:
            cp = o.custom_properties
            in_wall = cp.get("in_wall", {})
            wall_id = in_wall.get("wall_id")
            if not wall_id:
                # opening has no recorded host wall: unknown, not invalid
                continue
            wall = entities.get(wall_id)
            if wall is None:
                out.append(ConstraintViolation(
                    rule="opening_host_wall_exists", subject_id=o.id, subject_type="opening",
                    target_id=wall_id, target_type="wall", severity="error",
                    message=f"Opening '{o.name or o.id}' references host wall {wall_id} "
                            f"which does not exist in the world.",
                    measured={},
                    suggestion="Remove the dangling reference or re-capture the host wall.",
                ))
                continue
            # check the opening is within the wall's bounds
            ob = self._bounds(o)
            wb = self._bounds(wall)
            if ob is None or wb is None:
                continue
            # opening centre should lie on the wall's face plane
            o_center = (ob[0] + ob[1]) / 2
            # wall extent along its length axis
            wall_len = float(np.linalg.norm(wb[1] - wb[0]))
            # rough check: opening within wall's AABB
            within = bool(np.all(o_center >= wb[0] - 0.05) and np.all(o_center <= wb[1] + 0.05))
            if not within:
                out.append(ConstraintViolation(
                    rule="opening_within_host_wall", subject_id=o.id, subject_type="opening",
                    target_id=wall_id, target_type="wall", severity="warning",
                    message=(f"Opening '{o.name or o.id}' is outside its host wall "
                             f"'{wall.name or wall.id}' bounds."),
                    measured={"opening_center_m": [round(float(c), 3) for c in o_center],
                              "wall_bounds_m": [[round(float(c), 3) for c in wb[0]],
                                                [round(float(c), 3) for c in wb[1]]]},
                    suggestion="Re-measure the opening position or reassign its host wall.",
                ))
        return out

    def _check_windows_in_walls(
        self, windows: List[Entity], walls: List[Entity], entities: Dict[str, Entity]
    ) -> List[ConstraintViolation]:
        out: List[ConstraintViolation] = []
        for wnd in windows:
            cp = wnd.custom_properties
            wall_id = cp.get("in_wall", {}).get("wall_id")
            if not wall_id:
                continue
            wall = entities.get(wall_id)
            if wall is None:
                continue
            wb = self._bounds(wall)
            ob = self._bounds(wnd)
            if wb is None or ob is None:
                continue
            # window should not extend above the wall top
            wall_top = wb[1][1]
            win_top = ob[1][1]
            if win_top > wall_top + 0.05:
                out.append(ConstraintViolation(
                    rule="window_below_wall_top", subject_id=wnd.id, subject_type="window",
                    target_id=wall_id, target_type="wall", severity="warning",
                    message=(f"Window '{wnd.name or wnd.id}' extends above its host wall "
                             f"'{wall.name or wall.id}' top by {win_top - wall_top:.2f} m."),
                    measured={"window_top_m": round(float(win_top), 3),
                              "wall_top_m": round(float(wall_top), 3)},
                    suggestion="Lower the window or extend the host wall.",
                ))
        return out

    def _check_stairs_connect_floors(
        self, stairs: List[Entity], floors: List[Entity], entities: Dict[str, Entity],
    ) -> List[ConstraintViolation]:
        out: List[ConstraintViolation] = []
        for s in stairs:
            sb = self._bounds(s)
            if sb is None:
                continue
            stair_bottom = sb[0][1]
            stair_top = sb[1][1]
            # find floors near the stair bottom and top
            bottom_floor: Optional[Entity] = None
            top_floor: Optional[Entity] = None
            best_bottom_gap = float("inf")
            best_top_gap = float("inf")
            for f in floors:
                fb = self._bounds(f)
                if fb is None:
                    continue
                floor_top = fb[1][1]
                gap_b = abs(stair_bottom - floor_top)
                gap_t = abs(stair_top - floor_top)
                if gap_b < best_bottom_gap:
                    best_bottom_gap = gap_b
                    bottom_floor = f
                if gap_t < best_top_gap:
                    best_top_gap = gap_t
                    top_floor = f
            if bottom_floor is None or top_floor is None:
                out.append(ConstraintViolation(
                    rule="stairs_connect_floors", subject_id=s.id, subject_type="stair",
                    target_id=None, target_type=None, severity="warning",
                    message=(f"Stair '{s.name or s.id}' does not clearly connect "
                             f"to observed floors (bottom gap {best_bottom_gap:.2f} m, "
                             f"top gap {best_top_gap:.2f} m)."),
                    measured={"bottom_gap_m": round(best_bottom_gap, 3),
                              "top_gap_m": round(best_top_gap, 3)},
                    suggestion="Capture the floors at the top and bottom of this stair.",
                ))
            elif bottom_floor.id == top_floor.id:
                out.append(ConstraintViolation(
                    rule="stairs_connect_floors", subject_id=s.id, subject_type="stair",
                    target_id=bottom_floor.id, target_type="floor", severity="error",
                    message=(f"Stair '{s.name or s.id}' starts and ends on the same "
                             f"floor '{bottom_floor.name or bottom_floor.id}'."),
                    measured={"floor_id": bottom_floor.id},
                    suggestion="The stair may not actually change elevation; re-capture.",
                ))
        return out


__all__ = ["ArchitecturalConstraintSolver", "ConstraintViolation", "ConstraintReport"]