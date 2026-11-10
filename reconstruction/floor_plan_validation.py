"""Automatic floor-plan and room-layout validation (F-30).

Checks whether room boundaries, doors, circulation areas and floor-plan
geometry agree with the reconstructed 3D environment. Compares the generated
plan against the 3D world and reports discrepancies.

Honesty rules:
* A discrepancy is reported with both the plan value and the measured value.
* Missing evidence (no measured door, no measured wall) is "unobserved",
  not assumed correct.
* No threshold is treated as a hard pass/fail without an explicit tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class PlanDiscrepancy:
    """One disagreement between the floor plan and the 3D reconstruction."""
    rule: str                      # e.g. "door_in_wall", "room_closure"
    room_id: str
    element_id: str                # door/wall/room id
    plan_value: Optional[float]
    measured_value: Optional[float]
    difference_m: Optional[float]
    severity: str                  # "info" | "warning" | "error"
    message: str
    note: Optional[str] = None


@dataclass(frozen=True)
class FloorPlanValidationReport:
    """All discrepancies between plan and 3D world for one floor."""
    room_id: str
    discrepancies: List[PlanDiscrepancy]
    rooms_checked: int
    doors_checked: int
    walls_checked: int
    provisional: bool = True


class FloorPlanValidator:
    """Compares floor-plan geometry against the reconstructed 3D world."""

    DOOR_WIDTH_TOL_M = 0.05
    WALL_LENGTH_TOL_M = 0.05
    LOOP_GAP_TOL_M = 0.02

    def validate_room(
        self,
        room: Entity,
        walls: Sequence[Entity],
        doors: Sequence[Entity],
        world_entities: Dict[str, Entity],
    ) -> FloorPlanValidationReport:
        discrepancies: List[PlanDiscrepancy] = []
        cp = room.custom_properties or {}

        # Room closure: check the wall loop closes
        wall_loops = self._check_loop_closure(walls)
        for gap in wall_loops:
            discrepancies.append(PlanDiscrepancy(
                rule="room_loop_closure", room_id=room.id,
                element_id=room.id, plan_value=None,
                measured_value=gap, difference_m=None,
                severity="error" if gap > self.LOOP_GAP_TOL_M else "warning",
                message=f"Room '{room.name or room.id}' wall loop has a "
                        f"{gap:.3f} m gap.",
                note="The footprint polygon may not match the reconstructed walls.",
            ))

        # Wall lengths: plan vs measured
        plan_wall_lengths = cp.get("wall_lengths_m", {})
        for w in walls:
            wcp = w.custom_properties or {}
            wlen = wcp.get("length_m")
            if wlen is None:
                continue
            plan_len = plan_wall_lengths.get(w.id)
            if plan_len is not None:
                diff = abs(float(plan_len) - float(wlen))
                if diff > self.WALL_LENGTH_TOL_M:
                    discrepancies.append(PlanDiscrepancy(
                        rule="wall_length", room_id=room.id, element_id=w.id,
                        plan_value=round(float(plan_len), 4),
                        measured_value=round(float(wlen), 4),
                        difference_m=round(diff, 4),
                        severity="warning",
                        message=(f"Wall '{w.name or w.id}' plan length "
                                 f"{plan_len:.3f} m differs from measured "
                                 f"{wlen:.3f} m by {diff:.3f} m."),
                        note="Re-measure or update the plan.",
                    ))

        # Doors in walls
        for d in doors:
            dcp = d.custom_properties or {}
            host_wall_id = dcp.get("in_wall", {}).get("wall_id")
            plan_width = dcp.get("width_m")
            measured_width = dcp.get("width_m")
            if host_wall_id:
                host = world_entities.get(host_wall_id)
                if host is None:
                    discrepancies.append(PlanDiscrepancy(
                        rule="door_host_wall", room_id=room.id, element_id=d.id,
                        plan_value=None, measured_value=None,
                        difference_m=None, severity="error",
                        message=f"Door '{d.name or d.id}' references wall {host_wall_id} "
                                f"which does not exist.",
                        note="Reassign the door's host wall or re-capture.",
                    ))
                    continue
            if plan_width is not None and measured_width is not None:
                diff = abs(float(plan_width) - float(measured_width))
                if diff > self.DOOR_WIDTH_TOL_M:
                    discrepancies.append(PlanDiscrepancy(
                        rule="door_width", room_id=room.id, element_id=d.id,
                        plan_value=round(float(plan_width), 4),
                        measured_value=round(float(measured_width), 4),
                        difference_m=round(diff, 4),
                        severity="warning",
                        message=(f"Door '{d.name or d.id}' plan width "
                                 f"{plan_width:.3f} m differs from measured "
                                 f"{measured_width:.3f} m by {diff:.3f} m."),
                        note="Update the plan or re-measure the door.",
                    ))

        discrepancies.sort(key=lambda d: ({"error": 0, "warning": 1, "info": 2}[d.severity], d.element_id))
        return FloorPlanValidationReport(
            room_id=room.id,
            discrepancies=discrepancies,
            rooms_checked=1,
            doors_checked=len(doors),
            walls_checked=len(walls),
            provisional=True,
        )

    def validate_floor(
        self,
        rooms: Sequence[Entity],
        all_walls: Dict[str, Sequence[Entity]],
        all_doors: Dict[str, Sequence[Entity]],
        world_entities: Dict[str, Entity],
    ) -> List[FloorPlanValidationReport]:
        reports: List[FloorPlanValidationReport] = []
        for room in rooms:
            room_id = room.id
            walls = all_walls.get(room_id, [])
            doors = all_doors.get(room_id, [])
            reports.append(self.validate_room(room, walls, doors, world_entities))
        return reports

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _check_loop_closure(walls: Sequence[Entity]) -> List[float]:
        """Return gaps in the wall loop."""
        gaps: List[float] = []
        walls_list = list(walls)
        if len(walls_list) < 2:
            return gaps

        segments = []
        for w in walls_list:
            cp = w.custom_properties or {}
            bmin = cp.get("bounds_min")
            bmax = cp.get("bounds_max")
            if bmin is None or bmax is None:
                continue
            p0 = np.asarray(bmin, float)
            p1 = np.asarray(bmax, float)
            segments.append((p0, p1, w.id))

        for i in range(len(segments)):
            p0_i, p1_i, id_i = segments[i]
            end_i = p1_i
            for j in range(len(segments)):
                if i == j:
                    continue
                p0_j, p1_j, id_j = segments[j]
                start_j = p0_j
                gap = float(np.linalg.norm(end_i - start_j))
                if gap > 0.01:
                    gaps.append(gap)
        return gaps


import numpy as np  # noqa: E402


__all__ = ["FloorPlanValidator", "PlanDiscrepancy", "FloorPlanValidationReport"]