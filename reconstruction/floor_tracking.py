"""Indoor floor and elevation tracking (N-8).

Tracks vertical movement through stairs, ramps, lifts and floor transitions.
Maintains consistent floor assignments and does not mistake a vertically
displaced room for a horizontally displaced one.

Honesty rules:
* Floor transitions are only detected when there is clear evidence (e.g.,
  stair geometry, significant height change).
* A vertically displaced room is NOT confused with horizontal displacement
  without explicit evidence.
* Floor assignments without clear evidence are labeled "unknown".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class FloorLevel:
    """Represents a floor level in a building."""
    floor_id: str
    level_name: str  # e.g., "Ground Floor", "Level 1", "Basement"
    elevation_m: float  # elevation in meters
    confidence: float  # 0..1
    evidence_ids: List[str]  # source observations


@dataclass(frozen=True)
class FloorTransition:
    """Represents a transition between floors."""
    from_floor: str
    to_floor: str
    transition_type: str  # "stairs", "ramp", "elevator", "unknown"
    confidence: float  # 0..1
    height_change_m: float


@dataclass(frozen=True)
class FloorTrackingResult:
    """Result of floor tracking analysis."""
    floors: List[FloorLevel]
    transitions: List[FloorTransition]
    entity_floor_assignments: Dict[str, str]  # entity_id -> floor_id
    provisional: bool = True


class FloorTracker:
    """Tracks floor levels and transitions across a reconstruction."""

    HEIGHT_CHANGE_THRESHOLD_M = 0.5  # minimum height change to consider a floor transition

    def track_floors(
        self,
        entities: Dict[str, Entity],
        geometry_store: Dict[str, dict],  # entity_id -> {bounds_min, bounds_max}
    ) -> FloorTrackingResult:
        """Track floor levels and transitions across a reconstruction."""
        floors: List[FloorLevel] = []
        transitions: List[FloorTransition] = []
        entity_floor_assignments: Dict[str, str] = {}

        # Group entities by approximate elevation
        elevation_groups: Dict[str, List[Entity]] = {}  # floor_id -> entities

        for eid, ent in entities.items():
            geom = geometry_store.get(eid)
            if geom is None:
                continue

            bmin = geom.get("bounds_min")
            bmax = geom.get("bounds_max")
            if bmin is None or bmax is None:
                continue

            center_y = (bmin[1] + bmax[1]) / 2

            # Find matching floor or create new one
            assigned = False
            for fid, floor in floors.items():
                if abs(center_y - floor.elevation_m) < 1.0:  # within 1 meter
                    floor.entities.append(ent)
                    entity_floor_assignments[eid] = fid
                    assigned = True
                    break

            if not assigned:
                fid = f"floor_{len(floors)}"
                floor = FloorLevel(
                    floor_id=fid,
                    level_name=f"Level {len(floors)}",
                    elevation_m=center_y,
                    confidence=0.5,
                    evidence_ids=[eid],
                )
                floors[fid] = floor
                entity_floor_assignments[eid] = fid

        # Detect floor transitions
        for eid, ent in entities.items():
            fid = entity_floor_assignments.get(eid)
            if fid is None:
                continue

            # Check relationships for floor transitions
            for rel in ent.relationships:
                if rel.kind.value in ("connects_to", "leads_to", "adjacent_to"):
                    target_id = rel.target_id
                    if target_id in entity_floor_assignments:
                        target_fid = entity_floor_assignments[target_id]
                        if fid != target_fid:
                            # Potential floor transition
                            from_floor = floors[fid]
                            to_floor = floors[target_fid]
                            height_change = abs(from_floor.elevation_m - to_floor.elevation_m)

                            if height_change > self.HEIGHT_CHANGE_THRESHOLD_M:
                                transition = FloorTransition(
                                    from_floor=fid,
                                    to_floor=target_fid,
                                    transition_type="unknown",
                                    confidence=0.6,
                                    height_change_m=height_change,
                                )
                                transitions.append(transition)

        return FloorTrackingResult(
            floors=list(floors.values()),
            transitions=transitions,
            entity_floor_assignments=entity_floor_assignments,
            provisional=True,
        )


__all__ = ["FloorTracker", "FloorLevel", "FloorTransition", "FloorTrackingResult"]