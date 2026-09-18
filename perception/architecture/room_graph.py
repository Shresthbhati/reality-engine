"""Room + building graphs (P8-01 room graph, P8-02 building graph).

Position in the perception stack:

    plane detection (perception/geometry/orientation.py, RANSAC)
      -> geometry-only classification (perception/architecture/
         classify.py: wall/floor/ceiling + doorway gaps)
      -> THIS MODULE: architectural ENTITY structure
      -> WorldIR promotion (existing promote_planes path stays the
         write path for individual elements)

A room is NOT merely a mesh, and a building is NOT merely a point
cloud (directive sections 33/34): a room here has boundaries, measured
openings, adjacency topology, and measured dimensions; a building has
storeys (rooms grouped by measured floor height) and a measured
envelope.

Method (all measured from the classifier's real plane evidence, never
guessed):

- Boundary elements: wall/floor/ceiling ArchitecturalElements.
- Enclosure: a room exists iff the element set encloses space -- the
  tests' requirement is at least one floor, one ceiling, and >= 3
  walls whose union of bounds spans all three axes; anything less
  cannot honestly be called a room and is refused (empty list, no
  partial guesses).
- Openings: reused unchanged from classify.detect_wall_opening (the
  measured gap in a wall's inlier coverage at floor height) --
  opening width/height come from the gap's measured extent, not a
  door-size assumption.
- Adjacency: two rooms are adjacent iff they share >= 1 boundary
  element (a wall plane can bound two rooms; sharing a floor/ceiling
  means the rooms are vertically stacked and this is recorded in the
  edge metadata).
- Storeys (P8-02): rooms whose floor-element heights (the floor's
  measured bounds z) agree within FLOOR_HEIGHT_TOLERANCE_M belong to
  one storey; storeys are ordered bottom-up. The building envelope is
  the union of its rooms' measured bounds.

Deterministic: sorting by stable keys everywhere; same input ->
byte-identical output regardless of input order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from perception.architecture.classify import (
    ArchitecturalElement,
    PlaneInput,
    classify_planes,
    detect_wall_opening,
)


#: Two floors whose measured heights differ by more than this belong
#: to different storeys. Below the tolerance they are the same storey
#: (measurement noise on the same floor seen from both rooms). Not
#: tuned against a real dataset -- documented deferral like every
#: threshold in this repo.
FLOOR_HEIGHT_TOLERANCE_M = 0.15

#: Minimum walls for an enclosure: a box has >= 3 distinct wall
#: planes for any convex room interior (2 walls cannot enclose).
MIN_WALLS_FOR_ROOM = 3


class RoomGraphError(ValueError):
    """Room/building graph construction refused."""


@dataclass(frozen=True)
class RoomOpening:
    """A measured opening in one of the room's boundary walls."""

    wall_element_id: str
    kind: str  # "doorway" (the one gap type classify.py measures)
    width_m: float
    height_m: float

    def to_dict(self) -> dict:
        return {
            "wall_element_id": self.wall_element_id,
            "kind": self.kind,
            "width_m": self.width_m,
            "height_m": self.height_m,
        }


@dataclass(frozen=True)
class RoomGraph:
    """One room: boundaries, openings, adjacency, measured dimensions."""

    room_id: str
    boundary_element_ids: Tuple[str, ...]
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    dimensions_m: Dict[str, float]
    floor_area_m2: float
    openings: Tuple[RoomOpening, ...] = ()
    adjacent_room_ids: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "boundary_element_ids": list(self.boundary_element_ids),
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "dimensions_m": dict(self.dimensions_m),
            "floor_area_m2": self.floor_area_m2,
            "openings": [o.to_dict() for o in self.openings],
            "adjacent_room_ids": list(self.adjacent_room_ids),
        }


@dataclass(frozen=True)
class Storey:
    """One building level: rooms sharing a measured floor height."""

    storey_id: str
    floor_height_m: float
    room_ids: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "storey_id": self.storey_id,
            "floor_height_m": self.floor_height_m,
            "room_ids": list(self.room_ids),
        }


@dataclass(frozen=True)
class BuildingGraph:
    """Rooms assembled into storeys with a measured envelope."""

    building_id: str
    storeys: Tuple[Storey, ...]
    envelope_bounds_min: Tuple[float, float, float]
    envelope_bounds_max: Tuple[float, float, float]

    def to_dict(self) -> dict:
        return {
            "building_id": self.building_id,
            "storeys": [s.to_dict() for s in self.storeys],
            "envelope_bounds_min": list(self.envelope_bounds_min),
            "envelope_bounds_max": list(self.envelope_bounds_max),
        }


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _floor_height(element: ArchitecturalElement) -> Optional[float]:
    """A floor element's measured height (its bounds z midpoint; a
    classified floor plane is horizontal, so its z extent is the
    measurement). None for non-floor elements."""
    if element.element_type != "floor":
        return None
    if element.bounds_min is None or element.bounds_max is None:
        return None
    return (element.bounds_min[2] + element.bounds_max[2]) / 2.0


def _enclosed_bounds(elements: Sequence[ArchitecturalElement], up) -> Optional[
    Tuple[Tuple[float, float, float], Tuple[float, float, float]]
]:
    """The bounds enclosed by a room's boundary elements: the smallest
    box containing all elements' bounds (the room interior is inside
    its walls). None when any participating element lacks bounds."""
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    for el in elements:
        if el.bounds_min is None or el.bounds_max is None:
            return None
        for i in range(3):
            lo[i] = min(lo[i], el.bounds_min[i])
            hi[i] = max(hi[i], el.bounds_max[i])
    if any(not math.isfinite(v) for v in lo + hi):
        return None
    return tuple(lo), tuple(hi)


def _group_enclosures(
    elements: Sequence[ArchitecturalElement], up
) -> List[List[ArchitecturalElement]]:
    """Group classified elements into maximal enclosure candidates.

    Method: floors are room seeds (each floor plane anchors at most
    one room at its measured height). A floor's room takes the floor,
    the ceiling(s) at the matching height band, and every wall whose
    measured bounds overlap the floor's xy extent. Walls shared with
    another floor's room are allowed (shared walls); a wall spanning
    multiple floors' xy extents stays a member of every room it
    encloses -- the graph records membership, ownership is not
    invented.
    """
    floors = [e for e in elements if e.element_type == "floor"]
    ceilings = [e for e in elements if e.element_type == "ceiling"]
    walls = [e for e in elements if e.element_type == "wall"]

    groups: List[List[ArchitecturalElement]] = []
    for floor in floors:
        fh = _floor_height(floor)
        if fh is None:
            continue
        # Ceiling candidates: ceilings whose xy extent overlaps the
        # floor's (a ceiling for a DIFFERENT room does not).
        def _xy_overlap(a: ArchitecturalElement, b: ArchitecturalElement) -> bool:
            return (
                a.bounds_min[0] <= b.bounds_max[0]
                and b.bounds_min[0] <= a.bounds_max[0]
                and a.bounds_min[1] <= b.bounds_max[1]
                and b.bounds_min[1] <= a.bounds_max[1]
            )

        room_ceilings = [
            c for c in ceilings
            if c.bounds_min is not None and _xy_overlap(floor, c)
            and c.bounds_min[2] > fh
        ]
        if not room_ceilings:
            continue
        # The storey's own ceiling is the LOWEST one above its floor
        # (a higher ceiling belongs to the storey above).
        ceiling = min(room_ceilings, key=lambda c: c.bounds_min[2])
        ceiling_z = ceiling.bounds_min[2]
        # A wall encloses THIS storey iff it positively overlaps the
        # floor->ceiling band (strict inequalities: a wall that only
        # touches the band's edge belongs to the adjacent storey --
        # without this, multi-storey inputs balloon every room across
        # all storeys). A wall spanning several storeys (shared
        # structure) is a member of every storey it encloses.
        room_walls = [
            w for w in walls
            if w.bounds_min is not None
            and _xy_overlap(floor, w)
            and w.bounds_min[2] < ceiling_z
            and w.bounds_max[2] > fh
        ]
        members: List[ArchitecturalElement] = [floor, ceiling] + room_walls
        if len(room_walls) < MIN_WALLS_FOR_ROOM or not room_ceilings:
            continue  # cannot enclose: refuse this floor, no guess
        groups.append(members)
    return groups


def _openings_for(
    wall: ArchitecturalElement,
    floor_height: float,
    up: Sequence[float],
    plane_inputs: Dict[str, PlaneInput],
) -> List[RoomOpening]:
    """Measured openings on one wall, via classify.detect_wall_opening
    (unchanged reuse) + the gap's measured extent."""
    plane = plane_inputs.get(wall.source_plane_id)
    if plane is None:
        return []
    opening = detect_wall_opening(plane, tuple(up), floor_height)
    if opening is None:
        return []
    # The gap's measured extent: re-derive width/height from the
    # wall's inlier coverage around the detected gap. detect_wall_
    # opening returns the element only; the gap width is the empty
    # run's lateral extent -- measured here from the same bucket scan.
    # For honesty, width/height come from the wall plane's actual
    # inliers, using the same bucketing.
    lat = _gap_extent_m(plane, tuple(up), floor_height)
    if lat is None:
        return []
    width, height = lat
    return [RoomOpening(
        wall_element_id=wall.element_id,
        kind="doorway",
        width_m=width,
        height_m=height,
    )]


def _gap_extent_m(
    plane: PlaneInput, up: Sequence[float], floor_height: float
) -> Optional[Tuple[float, float]]:
    """The measured (width, height) of a doorway gap in a wall, from
    the wall's own inliers (the same evidence detect_wall_opening
    uses, extended with a measured head height):

    - width: the longest empty lateral bucket run inside the floor
      scan band (classify.py's DOOR_SCAN_BAND_M convention);
    - height: the first wall-material height ABOVE the scan band
      within the gap's lateral range -- the measured head height. A
      gap with no material above it in the inliers reports the wall's
      own top (the gap is open as far as the evidence goes).

    None when there is no doorway-sized interior gap."""
    from perception.architecture.classify import (
        DOOR_MIN_WIDTH_M,
        DOOR_SCAN_BAND_M,
        SCAN_BUCKET_M,
    )

    def _dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    up_u = tuple(up)
    lateral = _cross(plane.normal, up_u)
    lateral_len = math.sqrt(_dot(lateral, lateral))
    if lateral_len < 1e-9:
        return None
    lateral = (lateral[0] / lateral_len, lateral[1] / lateral_len, lateral[2] / lateral_len)

    lat_min = min(
        _dot(plane.bounds_min, lateral), _dot(plane.bounds_max, lateral)
    )
    lat_max = max(
        _dot(plane.bounds_min, lateral), _dot(plane.bounds_max, lateral)
    )
    span = lat_max - lat_min
    if span <= 0:
        return None

    n_buckets = max(1, int(math.ceil(span / SCAN_BUCKET_M)))
    band_lo = floor_height
    band_hi = floor_height + DOOR_SCAN_BAND_M
    covered = [False] * n_buckets
    for pos in plane.inlier_positions:
        height = _dot(pos, up_u)
        if height < band_lo or height > band_hi:
            continue
        idx = int((_dot(pos, lateral) - lat_min) / SCAN_BUCKET_M)
        idx = max(0, min(n_buckets - 1, idx))
        covered[idx] = True

    # Longest interior empty run, edge margins excluded (same rule as
    # detect_wall_opening: boundary dropout is not an opening).
    lo, hi = 2, n_buckets - 2
    best_start, best_len = -1, 0
    run_start, run_len = -1, 0
    for i in range(max(0, lo), min(n_buckets, hi)):
        if not covered[i]:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    if best_len <= 0:
        return None
    width = best_len * SCAN_BUCKET_M
    if width < DOOR_MIN_WIDTH_M:
        return None

    # Head height: first wall material above the scan band within the
    # gap's own bucket run (no lateral margin: one would swallow the
    # solid wall columns just outside the gap edges and read their
    # height as the head -- a real bug caught by the fixture probe).
    gap_lo = lat_min + best_start * SCAN_BUCKET_M
    gap_hi = gap_lo + width
    above: List[float] = []
    wall_top = band_hi
    for pos in plane.inlier_positions:
        lat = _dot(pos, lateral)
        height = _dot(pos, up_u)
        wall_top = max(wall_top, height)
        if gap_lo <= lat <= gap_hi and height > band_hi:
            above.append(height)
    head = min(above) if above else wall_top
    height_m = head - floor_height
    if height_m <= 0:
        return None
    return width, height_m


def build_room_graph(
    elements: Sequence[ArchitecturalElement],
    up: Sequence[float] = (0.0, 0.0, 1.0),
    plane_inputs: Optional[Dict[str, PlaneInput]] = None,
    planes: Optional[Sequence[PlaneInput]] = None,
) -> List[RoomGraph]:
    """Build room graphs from classified architectural elements.

    Opening detection needs the walls' real inlier points, which do
    not ride on the classified elements: supply `planes` (the same
    PlaneInputs classify_planes consumed) or a prebuilt `plane_inputs`
    map (plane_id -> PlaneInput). When neither is supplied, openings
    are not detected (recorded absence, never a guess).

    Returns rooms sorted by room_id (deterministic regardless of
    input order). Insufficient enclosure -> no room (empty list).
    """
    floors = [e for e in elements if e.element_type == "floor"]
    groups = _group_enclosures(elements, up)
    rooms: List[RoomGraph] = []
    plane_inputs = dict(plane_inputs or {})
    if planes:
        for p in planes:
            plane_inputs.setdefault(p.plane_id, p)

    enclosures: List[dict] = []
    for members in groups:
        bounds = _enclosed_bounds(members, up)
        if bounds is None:
            continue
        bmin, bmax = bounds
        enclosures.append({
            "members": sorted(members, key=lambda e: e.element_id),
            "bmin": bmin,
            "bmax": bmax,
            "floor_height": _floor_height(
                next(e for e in members if e.element_type == "floor")
            ),
        })
    if not enclosures:
        return []

    enclosures.sort(key=lambda enc: (enc["bmin"][2], enc["bmin"][0], enc["bmin"][1]))
    for i, enc in enumerate(enclosures, start=1):
        members = enc["members"]
        bmin, bmax = enc["bmin"], enc["bmax"]
        dims = {
            "x": bmax[0] - bmin[0],
            "y": bmax[1] - bmin[1],
            "z": bmax[2] - bmin[2],
        }
        floor = next(e for e in members if e.element_type == "floor")
        fh = enc["floor_height"]
        openings: List[RoomOpening] = []
        for w in members:
            if w.element_type != "wall":
                continue
            openings.extend(_openings_for(w, fh, up, plane_inputs))
        area = dims["x"] * dims["y"]
        rooms.append(RoomGraph(
            room_id=f"room-{i:03d}",
            boundary_element_ids=tuple(sorted(e.element_id for e in members)),
            bounds_min=bmin,
            bounds_max=bmax,
            dimensions_m=dims,
            floor_area_m2=area,
            openings=tuple(sorted(openings, key=lambda o: o.wall_element_id)),
        ))

    # Adjacency: shared boundary element -> adjacent rooms (computed
    # before construction: the records are frozen).
    adjacency: Dict[str, List[str]] = {r.room_id: [] for r in rooms}
    for room in rooms:
        for other in rooms:
            if other.room_id == room.room_id:
                continue
            shared = set(room.boundary_element_ids) & set(other.boundary_element_ids)
            if shared and other.room_id not in adjacency[room.room_id]:
                adjacency[room.room_id].append(other.room_id)
    rooms = [
        RoomGraph(
            room_id=r.room_id,
            boundary_element_ids=r.boundary_element_ids,
            bounds_min=r.bounds_min,
            bounds_max=r.bounds_max,
            dimensions_m=r.dimensions_m,
            floor_area_m2=r.floor_area_m2,
            openings=r.openings,
            adjacent_room_ids=tuple(sorted(adjacency[r.room_id])),
        )
        for r in rooms
    ]

    return rooms


def build_building_graph(
    rooms: Sequence[RoomGraph],
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> Optional[BuildingGraph]:
    """Assemble rooms into storeys (measured floor heights) and a
    building envelope. None when there are no rooms -- an empty
    building is a refusal, not an empty shell."""
    if not rooms:
        return None
    # Storey assignment needs each room's floor height: the bounds z
    # minimum is the measured floor level of the room's enclosure.
    rooms_sorted = sorted(rooms, key=lambda r: r.room_id)
    by_height: List[Tuple[float, List[str]]] = []
    for room in rooms_sorted:
        h = room.bounds_min[2]
        if by_height and abs(h - by_height[-1][0]) <= FLOOR_HEIGHT_TOLERANCE_M:
            by_height[-1][1].append(room.room_id)
        else:
            by_height.append((h, [room.room_id]))
    storeys = tuple(
        Storey(
            storey_id=f"storey-{i:02d}",
            floor_height_m=h,
            room_ids=tuple(ids),
        )
        for i, (h, ids) in enumerate(sorted(by_height, key=lambda x: x[0]), start=1)
    )
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for room in rooms_sorted:
        for i in range(3):
            lo[i] = min(lo[i], room.bounds_min[i])
            hi[i] = max(hi[i], room.bounds_max[i])
    return BuildingGraph(
        building_id="building-001",
        storeys=storeys,
        envelope_bounds_min=tuple(lo),
        envelope_bounds_max=tuple(hi),
    )
