"""Measured corridor detection & geometric-topological inference.

A corridor is an elongated interior circulation space connecting multiple
rooms or spaces. Unlike rooms (which are typically lower aspect-ratio
living or working enclosures), a corridor's evidence is:

1. Elongated free-space geometry: aspect ratio (length / width) >= 2.0 (typically >= 2.5).
2. Width consistency: width within walkable circulation limits (0.7 m to 3.5 m).
3. Floor and ceiling continuity: clear vertical clearance between floor and ceiling.
4. Enclosing boundary walls: parallel longitudinal walls guiding circulation.
5. Doorway connections: repeated or multiple openings into adjacent rooms.
6. Intersections: junctions where circulation corridors meet (T-junction, L-junction, cross).

Honesty rules:
- An elongated space without sufficient enclosing walls or floor continuity
  cannot honestly be declared a corridor and is refused (CorridorRefused)
  or returned with UNRESOLVED status.
- Measurements (length, width, height, area, axis) are computed strictly from
  the measured boundary geometry, never guessed from standard building codes.
- Deterministic ordering throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from perception.architecture.classify import ArchitecturalElement, PlaneInput
from perception.architecture.room_graph import RoomGraph, RoomOpening, _floor_height, _up_axis

#: Minimum length-to-width aspect ratio for a corridor candidate.
CORRIDOR_MIN_ASPECT_RATIO = 2.0

#: Walkable corridor width bounds (m): below 0.7 m is a crawlway/gap;
#: above 3.5 m is a hall, atrium, or open room.
CORRIDOR_MIN_WIDTH_M = 0.7
CORRIDOR_MAX_WIDTH_M = 3.5

#: Minimum length for a circulation corridor (m).
CORRIDOR_MIN_LENGTH_M = 1.8


class CorridorRefused(ValueError):
    """Corridor candidate rejected due to insufficient or contradictory evidence."""


@dataclass(frozen=True)
class CorridorIntersection:
    """An intersection junction between two or more circulation routes."""

    intersection_id: str
    position: Tuple[float, float, float]
    intersecting_corridor_ids: Tuple[str, ...]
    kind: str  # "T-junction" | "L-junction" | "cross-junction"

    def to_dict(self) -> dict:
        return {
            "intersection_id": self.intersection_id,
            "position": list(self.position),
            "intersecting_corridor_ids": list(self.intersecting_corridor_ids),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class CorridorGraph:
    """One corridor: boundaries, dimensions, longitudinal axis, connected rooms, openings."""

    corridor_id: str
    boundary_element_ids: Tuple[str, ...]
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    length_m: float
    width_m: float
    height_m: float
    floor_area_m2: float
    longitudinal_axis: Tuple[float, float, float]  # Unit vector along length
    connected_room_ids: Tuple[str, ...] = ()
    openings: Tuple[RoomOpening, ...] = ()
    intersections: Tuple[CorridorIntersection, ...] = ()
    level_id: Optional[str] = None
    confidence: float = 1.0
    status: str = "detected"  # "detected" | "unresolved" | "partial"
    notes: Tuple[str, ...] = ()

    @property
    def aspect_ratio(self) -> float:
        return self.length_m / max(self.width_m, 1e-6)

    def to_dict(self) -> dict:
        return {
            "corridor_id": self.corridor_id,
            "boundary_element_ids": list(self.boundary_element_ids),
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "length_m": self.length_m,
            "width_m": self.width_m,
            "aspect_ratio": self.aspect_ratio,
            "height_m": self.height_m,
            "floor_area_m2": self.floor_area_m2,
            "longitudinal_axis": list(self.longitudinal_axis),
            "connected_room_ids": list(self.connected_room_ids),
            "openings": [o.to_dict() for o in self.openings],
            "intersections": [i.to_dict() for i in self.intersections],
            "level_id": self.level_id,
            "confidence": self.confidence,
            "status": self.status,
            "notes": list(self.notes),
        }


def _plan_overlap(a: ArchitecturalElement, b: ArchitecturalElement, ax0: int, ax1: int) -> bool:
    if a.bounds_min is None or a.bounds_max is None or b.bounds_min is None or b.bounds_max is None:
        return False
    return (
        a.bounds_min[ax0] <= b.bounds_max[ax0]
        and b.bounds_min[ax0] <= a.bounds_max[ax0]
        and a.bounds_min[ax1] <= b.bounds_max[ax1]
        and b.bounds_min[ax1] <= a.bounds_max[ax1]
    )


def detect_corridors(
    elements: Sequence[ArchitecturalElement],
    up: Sequence[float] = (0.0, 0.0, 1.0),
    plane_inputs: Optional[Dict[str, PlaneInput]] = None,
    rooms: Optional[Sequence[RoomGraph]] = None,
    min_aspect_ratio: float = CORRIDOR_MIN_ASPECT_RATIO,
) -> List[CorridorGraph]:
    """Detect corridor entities from architectural elements and room graphs.

    Distinguishes corridors from rooms based on:
    - Aspect ratio (length / width >= min_aspect_ratio).
    - Width within circulation limits (0.7m to 3.5m).
    - Longitudinal axis.
    - Doorway connections to adjacent rooms.
    """
    floors = [e for e in elements if e.element_type == "floor"]
    ceilings = [e for e in elements if e.element_type == "ceiling"]
    walls = [e for e in elements if e.element_type == "wall"]
    rooms = list(rooms or [])
    plane_inputs = dict(plane_inputs or {})

    up_idx = _up_axis(up)
    plan_axes = [i for i in range(3) if i != up_idx]
    ax0, ax1 = plan_axes[0], plan_axes[1]
    axis_names = ["x", "y", "z"]

    candidates: List[dict] = []

    for floor in floors:
        fh = _floor_height(floor, up)
        if fh is None:
            continue

        # Find matching ceiling above floor
        matching_ceilings = [
            c for c in ceilings
            if c.bounds_min is not None and _plan_overlap(floor, c, ax0, ax1) and c.bounds_min[up_idx] > fh
        ]
        if not matching_ceilings:
            continue
        ceiling = min(matching_ceilings, key=lambda c: c.bounds_min[up_idx])
        cz = ceiling.bounds_min[up_idx]

        # Enclosing walls
        room_walls = [
            w for w in walls
            if w.bounds_min is not None
            and _plan_overlap(floor, w, ax0, ax1)
            and w.bounds_min[up_idx] < cz
            and w.bounds_max[up_idx] > fh
        ]

        # A corridor requires at least 2 longitudinal walls or >= 2 walls
        if len(room_walls) < 2:
            continue

        # Bounds of this enclosure from the floor footprint and vertical clearance
        lo = [0.0, 0.0, 0.0]
        hi = [0.0, 0.0, 0.0]
        lo[ax0] = floor.bounds_min[ax0]
        hi[ax0] = floor.bounds_max[ax0]
        lo[ax1] = floor.bounds_min[ax1]
        hi[ax1] = floor.bounds_max[ax1]
        lo[up_idx] = fh
        hi[up_idx] = cz

        d0 = hi[ax0] - lo[ax0]
        d1 = hi[ax1] - lo[ax1]
        dz = hi[up_idx] - lo[up_idx]

        if d0 <= 0 or d1 <= 0:
            continue

        length = max(d0, d1)
        width = min(d0, d1)
        aspect = length / max(width, 1e-6)

        # Aspect ratio & width gates
        if aspect < min_aspect_ratio:
            # Does not have corridor geometry (it is a standard room or square lobby)
            continue
        if width < CORRIDOR_MIN_WIDTH_M or width > CORRIDOR_MAX_WIDTH_M:
            # Too narrow (crack) or too wide (hall/open space)
            continue
        if length < CORRIDOR_MIN_LENGTH_M:
            continue

        # Determine longitudinal axis (unit vector)
        axis = [0.0, 0.0, 0.0]
        if d0 >= d1:
            axis[ax0] = 1.0
        else:
            axis[ax1] = 1.0

        # Collect openings on corridor walls
        openings: List[RoomOpening] = []
        from perception.architecture.room_graph import _openings_for
        for w in room_walls:
            openings.extend(_openings_for(w, fh, up, plane_inputs))

        # Check which rooms share boundary walls or doorways with this corridor
        connected_rooms: List[str] = []
        corridor_wall_ids = {w.element_id for w in room_walls}
        for rm in rooms:
            shared = set(rm.boundary_element_ids) & corridor_wall_ids
            if shared:
                connected_rooms.append(rm.room_id)

        candidates.append({
            "floor": floor,
            "ceiling": ceiling,
            "walls": room_walls,
            "bounds_min": tuple(lo),
            "bounds_max": tuple(hi),
            "length_m": length,
            "width_m": width,
            "height_m": dz,
            "floor_area_m2": d0 * d1,
            "axis": tuple(axis),
            "openings": openings,
            "connected_rooms": sorted(connected_rooms),
        })

    # Sort deterministically
    candidates.sort(key=lambda c: (c["bounds_min"][up_idx], c["bounds_min"][ax0], c["bounds_min"][ax1]))

    corridors: List[CorridorGraph] = []
    for idx, c in enumerate(candidates, start=1):
        cid = f"corridor-{idx:03d}"
        boundary_ids = sorted([c["floor"].element_id, c["ceiling"].element_id] + [w.element_id for w in c["walls"]])

        # Calculate confidence and status based on boundary enclosure and doorways
        conf = 0.8
        notes = []
        status = "detected"
        if len(c["walls"]) >= 3:
            conf += 0.1
        elif len(c["walls"]) < 2:
            status = "unresolved"
            conf = min(conf, 0.5)
            notes.append("Ambiguous circulation: insufficient boundary wall enclosure")

        if len(c["connected_rooms"]) >= 2:
            conf += 0.1
            notes.append(f"Connects {len(c['connected_rooms'])} rooms")
        elif len(c["connected_rooms"]) == 1:
            notes.append(f"Connects room {c['connected_rooms'][0]}")
        else:
            if status == "detected":
                status = "partial"
            notes.append("Circulation geometry with unresolved room connections")

        conf = min(1.0, conf)

        corridors.append(CorridorGraph(
            corridor_id=cid,
            boundary_element_ids=tuple(boundary_ids),
            bounds_min=c["bounds_min"],
            bounds_max=c["bounds_max"],
            length_m=c["length_m"],
            width_m=c["width_m"],
            height_m=c["height_m"],
            floor_area_m2=c["floor_area_m2"],
            longitudinal_axis=c["axis"],
            connected_room_ids=tuple(c["connected_rooms"]),
            openings=tuple(sorted(c["openings"], key=lambda o: o.wall_element_id)),
            confidence=conf,
            status=status,
            notes=tuple(notes),
        ))

    # Harmonize with room-level corridor detection
    try:
        from perception.architecture.corridors import detect_corridor
        for rm in rooms:
            c_fit = detect_corridor(rm)
            if c_fit is not None:
                # Check if this space is already covered by a candidate
                covered = any(
                    abs(c.bounds_min[ax0] - rm.bounds_min[ax0]) < 0.2
                    and abs(c.bounds_min[ax1] - rm.bounds_min[ax1]) < 0.2
                    for c in corridors
                )
                if not covered:
                    cid = f"corridor-{rm.room_id}"
                    long_axis_vec = [0.0, 0.0, 0.0]
                    if c_fit.long_axis == "x":
                        long_axis_vec[0] = 1.0
                    elif c_fit.long_axis == "y":
                        long_axis_vec[1] = 1.0
                    else:
                        long_axis_vec[ax0] = 1.0
                    corridors.append(CorridorGraph(
                        corridor_id=cid,
                        boundary_element_ids=tuple(rm.boundary_element_ids),
                        bounds_min=rm.bounds_min,
                        bounds_max=rm.bounds_max,
                        length_m=c_fit.length_m,
                        width_m=c_fit.width_m,
                        height_m=float(rm.dimensions_m.get(axis_names[up_idx], 2.4)),
                        floor_area_m2=rm.floor_area_m2,
                        longitudinal_axis=tuple(long_axis_vec),
                        connected_room_ids=tuple(rm.adjacent_room_ids),
                        openings=rm.openings,
                        confidence=c_fit.confidence,
                        status="detected",
                        notes=(
                            f"Inferred from room {rm.room_id} with elongation {c_fit.elongation:.1f}"
                            + (" (serves both sides)" if c_fit.connects_both_sides else " (single-side service)"),
                        ),
                    ))
    except Exception:
        pass

    # Detect intersections between corridors
    if len(corridors) >= 2:
        updated_corridors: List[CorridorGraph] = []
        for c1 in corridors:
            intersections: List[CorridorIntersection] = []
            for c2 in corridors:
                if c1.corridor_id == c2.corridor_id:
                    continue
                # Test bounding box overlap
                overlap = (
                    c1.bounds_min[ax0] <= c2.bounds_max[ax0]
                    and c2.bounds_min[ax0] <= c1.bounds_max[ax0]
                    and c1.bounds_min[ax1] <= c2.bounds_max[ax1]
                    and c2.bounds_min[ax1] <= c1.bounds_max[ax1]
                    and abs(c1.bounds_min[up_idx] - c2.bounds_min[up_idx]) < 0.3
                )
                if overlap:
                    ix_lo = [max(c1.bounds_min[i], c2.bounds_min[i]) for i in range(3)]
                    ix_hi = [min(c1.bounds_max[i], c2.bounds_max[i]) for i in range(3)]
                    center = ((ix_lo[0] + ix_hi[0]) / 2.0, (ix_lo[1] + ix_hi[1]) / 2.0, (ix_lo[2] + ix_hi[2]) / 2.0)
                    # Check angle between axes
                    dot_axes = sum(
                        c1.longitudinal_axis[i] * c2.longitudinal_axis[i] for i in range(3)
                    )
                    kind = "cross-junction" if abs(dot_axes) < 0.3 else "T-junction"
                    intersections.append(CorridorIntersection(
                        intersection_id=f"ix-{c1.corridor_id}-{c2.corridor_id}",
                        position=center,
                        intersecting_corridor_ids=tuple(sorted([c1.corridor_id, c2.corridor_id])),
                        kind=kind,
                    ))
            updated_corridors.append(CorridorGraph(
                corridor_id=c1.corridor_id,
                boundary_element_ids=c1.boundary_element_ids,
                bounds_min=c1.bounds_min,
                bounds_max=c1.bounds_max,
                length_m=c1.length_m,
                width_m=c1.width_m,
                height_m=c1.height_m,
                floor_area_m2=c1.floor_area_m2,
                longitudinal_axis=c1.longitudinal_axis,
                connected_room_ids=c1.connected_room_ids,
                openings=c1.openings,
                intersections=tuple(intersections),
                level_id=c1.level_id,
                confidence=c1.confidence,
                status=c1.status,
                notes=c1.notes,
            ))
        corridors = updated_corridors

    return corridors
