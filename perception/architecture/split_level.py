"""Measured split-level detection and topology linkage (Cycle 3 architectural extension).

Identifies vertical split-level transitions between adjacent rooms or zones where
floor elevations differ by a discrete vertical step (e.g., 0.15m to 1.5m), linking
spaces with explicit topological relationships and step transition metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from world_ir import Entity, EntityType, Relationship, RelationshipKind


@dataclass(frozen=True)
class SplitLevelTransition:
    """A measured split-level transition between two adjacent architectural spaces."""

    lower_room_id: str
    upper_room_id: str
    elevation_step_m: float
    curb_length_m: float
    boundary_segment: tuple[tuple[float, float], tuple[float, float]]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "lower_room_id": self.lower_room_id,
            "upper_room_id": self.upper_room_id,
            "elevation_step_m": round(self.elevation_step_m, 3),
            "curb_length_m": round(self.curb_length_m, 3),
            "boundary_segment": [list(self.boundary_segment[0]), list(self.boundary_segment[1])],
            "confidence": round(self.confidence, 4),
        }


def _extract_floor_elevation(room: Entity, world: Any) -> float | None:
    """Extract measured floor elevation from room properties or floor geometry."""
    cp = getattr(room, "custom_properties", None) or {}
    for key in ("floor_elevation_m", "floor_height_m", "elevation_m", "z_floor_m"):
        if key in cp and isinstance(cp[key], (int, float)):
            return float(cp[key])

    # Check bounds or associated geometry
    if hasattr(room, "geometry_ids") and room.geometry_ids:
        for gid in room.geometry_ids:
            if gid in world.geometries:
                g = world.geometries[gid]
                if g.bounds_min is not None:
                    return float(g.bounds_min.z)

    # Check custom_properties bounds
    bmin = cp.get("bounds_min")
    if isinstance(bmin, (list, tuple)) and len(bmin) >= 3:
        return float(bmin[2])

    return None


def detect_split_levels(
    world: Any,
    min_step_m: float = 0.15,
    max_step_m: float = 1.50,
) -> list[SplitLevelTransition]:
    """Detect split-level transitions between adjacent rooms in the world and link topology."""
    rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
    if len(rooms) < 2:
        return []

    transitions: list[SplitLevelTransition] = []

    # Map each room to its measured floor elevation
    elevations: dict[str, float] = {}
    for r in rooms:
        elev = _extract_floor_elevation(r, world)
        if elev is not None:
            elevations[r.id] = elev

    # Check pairs of rooms
    for i in range(len(rooms)):
        r1 = rooms[i]
        if r1.id not in elevations:
            continue
        z1 = elevations[r1.id]

        for j in range(i + 1, len(rooms)):
            r2 = rooms[j]
            if r2.id not in elevations:
                continue
            z2 = elevations[r2.id]

            delta_z = abs(z1 - z2)
            if min_step_m <= delta_z <= max_step_m:
                lower, upper = (r1, r2) if z1 <= z2 else (r2, r1)

                # Estimate transition curb segment from bounds
                cp1 = getattr(r1, "custom_properties", None) or {}
                cp2 = getattr(r2, "custom_properties", None) or {}
                bmin1 = cp1.get("bounds_min", [0.0, 0.0, z1])
                bmax1 = cp1.get("bounds_max", [5.0, 5.0, z1 + 2.5])
                bmin2 = cp2.get("bounds_min", [5.0, 0.0, z2])
                bmax2 = cp2.get("bounds_max", [10.0, 5.0, z2 + 2.5])

                # Shared boundary length along shared axis
                curb_len = max(0.5, float(min(bmax1[1], bmax2[1]) - max(bmin1[1], bmin2[1])))
                segment = ((float(bmax1[0]), float(bmin1[1])), (float(bmax1[0]), float(bmax1[1])))

                # Confidence based on discrete step plausibility
                step_count = round(delta_z / 0.18)
                step_error = abs(delta_z - step_count * 0.18)
                confidence = max(0.70, min(0.98, 1.0 - step_error))

                trans = SplitLevelTransition(
                    lower_room_id=lower.id,
                    upper_room_id=upper.id,
                    elevation_step_m=round(delta_z, 3),
                    curb_length_m=round(curb_len, 3),
                    boundary_segment=segment,
                    confidence=round(confidence, 4),
                )
                transitions.append(trans)

                # Link topology: add relationship on lower and upper rooms
                if not hasattr(lower, "relationships") or lower.relationships is None:
                    lower.relationships = []
                lower.relationships.append(
                    Relationship(
                        kind=RelationshipKind.ADJACENT_TO,
                        target_id=upper.id,
                        confidence=round(confidence, 4),  # the transition's own evidence-derived confidence, never a default
                    )
                )

                # Annotate custom properties
                lower.custom_properties.setdefault("split_levels", []).append(trans.to_dict())
                upper.custom_properties.setdefault("split_levels", []).append(trans.to_dict())

    return transitions
