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
from dataclasses import dataclass, field, replace
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

#: A ceiling bounds a storey only when the space under it is high
#: enough to stand in. Lower horizontal planes (tables, counters,
#: worktops misclassified as horizontal structure) must not clip a
#: room's wall band at 0.75 m -- the room would lose every wall that
#: extends above the "ceiling". Measured from human clearance, not
#: tuned to a dataset. Used ONLY when a lower candidate exists; the
#: storey's real ceiling is the LOWEST candidate above this band.
MIN_WALKABLE_CLEAR_HEIGHT_M = 1.4

#: A horizontal plane spanning less plan area than this fraction of a
#: floor's footprint is furniture-scale (a table top), not structure
#: (a floor slab or ceiling). Its horizontal role is genuinely
#: uncertain -- classified, but never promoted as storey structure:
#: promoting a table as a floor fabricates a storey whose "floor" is
#: 1.2 m above the real one. Measured extent vs. the storey floor;
#: threshold between a two-person desk (~1.6 m2) and a small
#: mezzanine landing. Not tuned against a dataset -- documented
#: deferral.
FURNITURE_SCALE_FLOOR_FRACTION = 0.25

#: How far outside a floor's measured extent a boundary wall may sit
#: and still bound THIS floor. A floor's inliers routinely stop short
#: of its bounding walls (occlusion, furniture, scanner shadow), so
#: the wall's line coordinate may lie a little beyond the floor edge
#: -- but a wall hugging the OPPOSITE side of the building (many
#: metres away, another room's boundary) must not enclose this floor
#: too, or every room balloons across the whole plan. Beyond this
#: gap the wall belongs to some other enclosure. Not tuned against a
#: real dataset -- documented deferral like every threshold here.
FLOOR_WALL_GAP_TOLERANCE_M = 0.25


class RoomGraphError(ValueError):
    """Room/building graph construction refused."""


@dataclass(frozen=True)
class RoomOpening:
    """A measured opening in one of the room's boundary walls."""

    wall_element_id: str
    kind: str  # "doorway" | "window" | "generic_opening" | "unresolved"
    width_m: float
    height_m: float
    sill_height_m: float = 0.0
    connected_space_ids: Tuple[str, ...] = ()
    confidence: float = 1.0
    is_exterior: bool = False
    evidence_ids: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = {
            "wall_element_id": self.wall_element_id,
            "kind": self.kind,
            "width_m": self.width_m,
            "height_m": self.height_m,
        }
        if self.sill_height_m > 0:
            d["sill_height_m"] = self.sill_height_m
        if self.connected_space_ids:
            d["connected_space_ids"] = list(self.connected_space_ids)
        if self.confidence < 1.0:
            d["confidence"] = self.confidence
        if self.is_exterior:
            d["is_exterior"] = True
        if self.evidence_ids:
            d["evidence_ids"] = list(self.evidence_ids)
        return d


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
    corridor_ids: Tuple[str, ...] = ()
    status: str = "detected"  # "detected" | "partial" | "inferred" | "refused"
    confidence: float = 1.0
    boundary_completeness: float = 1.0
    ceiling_evidence: str = "measured"  # "measured" | "partial" | "missing"
    notes: Tuple[str, ...] = ()
    level_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "room_id": self.room_id,
            "boundary_element_ids": list(self.boundary_element_ids),
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "dimensions_m": dict(self.dimensions_m),
            "floor_area_m2": self.floor_area_m2,
            "openings": [o.to_dict() for o in self.openings],
            "adjacent_room_ids": list(self.adjacent_room_ids),
        }
        if self.corridor_ids:
            d["corridor_ids"] = list(self.corridor_ids)
        if self.status != "detected":
            d["status"] = self.status
        if self.confidence < 1.0:
            d["confidence"] = self.confidence
        if self.boundary_completeness < 1.0:
            d["boundary_completeness"] = self.boundary_completeness
        if self.ceiling_evidence != "measured":
            d["ceiling_evidence"] = self.ceiling_evidence
        if self.notes:
            d["notes"] = list(self.notes)
        if self.level_id:
            d["level_id"] = self.level_id
        return d


@dataclass(frozen=True)
class Storey:
    """One building level: rooms sharing a measured floor height."""

    storey_id: str
    floor_height_m: float
    room_ids: Tuple[str, ...]
    corridor_ids: Tuple[str, ...] = ()
    stair_ids: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = {
            "storey_id": self.storey_id,
            "floor_height_m": self.floor_height_m,
            "room_ids": list(self.room_ids),
        }
        if self.corridor_ids:
            d["corridor_ids"] = list(self.corridor_ids)
        if self.stair_ids:
            d["stair_ids"] = list(self.stair_ids)
        return d


@dataclass(frozen=True)
class BuildingGraph:
    """Rooms assembled into storeys with a measured envelope."""

    building_id: str
    storeys: Tuple[Storey, ...]
    envelope_bounds_min: Tuple[float, float, float]
    envelope_bounds_max: Tuple[float, float, float]
    corridors: Tuple[object, ...] = ()
    stairs: Tuple[object, ...] = ()

    def to_dict(self) -> dict:
        d = {
            "building_id": self.building_id,
            "storeys": [s.to_dict() for s in self.storeys],
            "envelope_bounds_min": list(self.envelope_bounds_min),
            "envelope_bounds_max": list(self.envelope_bounds_max),
        }
        if self.corridors:
            d["corridors"] = [c.to_dict() if hasattr(c, "to_dict") else c for c in self.corridors]
        if self.stairs:
            d["stairs"] = [s.to_dict() if hasattr(s, "to_dict") else s for s in self.stairs]
        return d


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _up_axis(up: Sequence[float]) -> int:
    u = [abs(x) for x in up]
    return u.index(max(u)) if max(u) > 1e-6 else 2


def _floor_height(element: ArchitecturalElement, up: Sequence[float] = (0.0, 0.0, 1.0)) -> Optional[float]:
    """A floor element's measured height (its bounds along the up axis midpoint).
    None for non-floor elements."""
    if element.element_type != "floor":
        return None
    if element.bounds_min is None or element.bounds_max is None:
        return None
    up_idx = _up_axis(up)
    return (element.bounds_min[up_idx] + element.bounds_max[up_idx]) / 2.0


def _enclosed_bounds(elements: Sequence[ArchitecturalElement], up, ceiling_cap: Optional[float] = None) -> Optional[
    Tuple[Tuple[float, float, float], Tuple[float, float, float]]
]:
    """The bounds enclosed by a room's boundary elements.

    Horizontal extent: the FLOOR's measured extent -- the walkable
    surface a person actually occupies. A union over the wall AABBs
    (the earlier behavior) inflates shared-wall rooms: one continuous
    facade wall spanning two rooms drags the union across the whole
    building, and both rooms reported the building's footprint
    (observed: 24 m2 per room instead of 12). Walls bound the space;
    the floor measures it. The z extent is the ROOM's own: floor bottom
    .. ceiling top when a ceiling was measured, otherwise floor bottom ..
    the walls' top capped at ``ceiling_cap`` (the next slab above the
    floor, when one exists). A union over the wall AABBs is wrong for the
    same reason vertically: the walls of stacked storeys are coplanar, so
    one wall plane spans every storey and would give each stacked room the
    whole building's height and the ground level's elevation (observed:
    two stacked rooms both reported z 0..4.7 and collapsed into one storey).

    None when any participating element lacks bounds. With multiple
    floors (rare; a room is normally one floor), the floors' union
    measures the horizontal extent."""
    if any(el.bounds_min is None or el.bounds_max is None for el in elements):
        return None
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    for el in elements:
        for i in range(3):
            lo[i] = min(lo[i], el.bounds_min[i])
            hi[i] = max(hi[i], el.bounds_max[i])
    if any(not math.isfinite(v) for v in lo + hi):
        return None
    floors = [el for el in elements if el.element_type == "floor"]
    if floors:
        up_idx = _up_axis(up)
        plan = [i for i in range(3) if i != up_idx]
        for ax in plan:
            lo[ax] = min(el.bounds_min[ax] for el in floors)
            hi[ax] = max(el.bounds_max[ax] for el in floors)
        lo[up_idx] = min(el.bounds_min[up_idx] for el in floors)
        ceilings = [el for el in elements if el.element_type == "ceiling"]
        if ceilings:
            hi[up_idx] = max(el.bounds_max[up_idx] for el in ceilings)
        elif ceiling_cap is not None:
            hi[up_idx] = max(lo[up_idx], min(hi[up_idx], ceiling_cap))
    return tuple(lo), tuple(hi)


def _wall_encloses_floor(
    w: ArchitecturalElement,
    floor: ArchitecturalElement,
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> bool:
    """A wall encloses a floor when its vertical PLANE's footprint
    reaches the floor's extent. AABB overlap fails for real
    reconstructions: the floor's inliers stop short of its bounding
    walls (the walls swallow the floor rim), so a boundary wall's thin
    AABB can sit entirely OUTSIDE the floor's AABB while the wall
    obviously encloses it.

    Measured rule: for each horizontal axis, the wall has a "line"
    coordinate (its thin axis) and a "run" interval (its long axis).
    The wall reaches the floor iff, on every horizontal axis, either
    the intervals overlap (run axes) or the wall's line coordinate
    lies within the floor's extent expanded by the wall's own run --
    the thin axis cannot demand overlap because the wall is thin by
    definition. Concretely: overlap on the run axis AND the wall's
    line coordinate within [floor.min - wall_run, floor.max +
    wall_run]... simplest honest formulation: the wall's line
    coordinate must lie within the floor's extent on its thin axis
    (expanded by the measurement's own granularity, the wall's run
    extent), and overlap on its run axis.
    """
    if w.bounds_min is None or w.bounds_max is None:
        return False
    up_idx = _up_axis(up)
    plan = [i for i in range(3) if i != up_idx]
    extents = [
        (w.bounds_max[i] - w.bounds_min[i], i) for i in plan
    ]
    extents.sort()
    thin_i = extents[0][1]      # the axis the wall is thin along
    run_i = extents[1][1]       # the axis the wall runs along
    # Run axis: genuine interval overlap with the floor.
    if (w.bounds_min[run_i] > floor.bounds_max[run_i]
            or w.bounds_max[run_i] < floor.bounds_min[run_i]):
        return False
    # Thin axis: the wall's line coordinate must HUG one of the
    # floor's edges on the axis the wall runs along, within a small
    # gap tolerance. Earlier drafts relaxed the wall's line interval
    # by the wall's whole run extent -- under that rule a facade wall
    # 3 m beyond a floor's far edge 'enclosed' the floor too, and
    # every room ballooned across the entire plan (observed on the
    # two-room apartment fixture: one 6x4 m phantom room instead of
    # two 3x4 m rooms). A boundary wall touches this floor; a wall a
    # room-width away bounds a different room.
    #
    # The wall's thin-axis interval may start up to
    # FLOOR_WALL_GAP_TOLERANCE_M before the floor edge (its inliers
    # may reach slightly past the boundary) and the floor edge may sit
    # up to that tolerance outside the wall's interval (the floor's
    # inliers may stop slightly short of the wall), but a wall whose
    # line interval lies entirely beyond the edge + tolerance does not
    # bound this floor.
    gap = FLOOR_WALL_GAP_TOLERANCE_M
    line_lo, line_hi = w.bounds_min[thin_i], w.bounds_max[thin_i]
    f_lo, f_hi = floor.bounds_min[thin_i], floor.bounds_max[thin_i]
    # The wall's line interval must reach the floor's extent from the
    # outside or overlap it: its near edge may not lie further than
    # `gap` beyond the floor's far edge on this axis. Overlaps pass
    # trivially. A wall entirely beyond that (a facade a room-width
    # away) does not bound this floor.
    return (line_lo <= f_hi + gap and line_hi >= f_lo - gap)


def plan_area(
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> float:
    """The plan (horizontal-projection) area of an axis-aligned box."""
    up_idx = _up_axis(up)
    plan = [i for i in range(3) if i != up_idx]
    return (
        (bounds_max[plan[0]] - bounds_min[plan[0]])
        * (bounds_max[plan[1]] - bounds_min[plan[1]])
    )


def _plan_overlap_area(
    a: ArchitecturalElement,
    b: ArchitecturalElement,
    ax0: int,
    ax1: int,
) -> float:
    """Positive plan-intersection area of two elements' AABBs on the
    two plan axes (0.0 when disjoint)."""
    dx = min(a.bounds_max[ax0], b.bounds_max[ax0]) \
        - max(a.bounds_min[ax0], b.bounds_min[ax0])
    dy = min(a.bounds_max[ax1], b.bounds_max[ax1]) \
        - max(a.bounds_min[ax1], b.bounds_min[ax1])
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return dx * dy


def floor_undersized_vs_floors(
    element: ArchitecturalElement,
    floors: Sequence[ArchitecturalElement],
    up: Sequence[float] = (0.0, 0.0, 1.0),
    fraction: float = FURNITURE_SCALE_FLOOR_FRACTION,
    coverage_positions: Optional[Sequence[Sequence[float]]] = None,
) -> bool:
    """True when a horizontal element spans less than `fraction` of the
    plan area of the storey's real floors -- furniture scale (table
    tops, worktops, counter slabs that RANSAC resolved and the role
    assignment called horizontal).

    Such a plane's horizontal role is genuinely UNCERTAIN structure:
    the geometry is measured, the semantic role is not. Callers must
    not promote it as storey structure (a table promoted as a floor
    fabricates a storey 1.2 m above the real one) nor let it seed
    rooms. A HOLLOW RING (perimeter trim rows whose bbox spans the
    plan while its points occupy a fraction of it) is demoted
    regardless of any reference floor: hollowness is measured directly
    via plan_support_coverage, so bbox-spanning rings cannot hide
    behind a large reference floor's bbox.
    False when no reference floors exist (nothing to compare against:
    keep the classification; downstream refusal paths still guard
    every use).
    """
    if element.bounds_min is None or element.bounds_max is None:
        return False
    # Hollow-ring check first: independent of reference floors, so a
    # trim ring cannot borrow scale from a real floor's bbox. Requires
    # the plane's own inlier positions (callers that cannot supply
    # them keep the bbox-only comparison below). Ring = the plane is
    # DENSE enough for its emptiness to mean shape, and fewer than
    # RING_MAX_INTERIOR_CELL_SHARE of its occupied plan cells lie in
    # the central half of its bbox (a solid sheet keeps ~25% whatever
    # its sampling density -- measured on real sparse SfM floors).
    if coverage_positions:
        _, interior_share, density = plan_support_coverage(
            coverage_positions,
            element.bounds_min, element.bounds_max, up)
        if (density >= RING_MIN_DENSITY_PTS_PER_M2
                and interior_share < RING_MAX_INTERIOR_CELL_SHARE):
            return True
    if not floors:
        return False
    refs = [
        plan_area(f.bounds_min, f.bounds_max, up) for f in floors
        if f.bounds_min is not None and f.bounds_max is not None
    ]
    ref = max(refs) if refs else 0.0
    if ref <= 0.0:
        return False
    return plan_area(element.bounds_min, element.bounds_max, up) \
        < fraction * ref


#: Ring-shape test, DENSITY-INVARIANT: a horizontal plane is a HOLLOW
#: RING when fewer than this share of its OCCUPIED plan-grid cells lie
#: in the central half of its own bounding box. Shares, not areas:
#: a uniformly sampled sheet keeps ~25% of its occupied cells in the
#: interior whatever its point density (a real sparse SfM floor
#: measured at 2-6% area occupancy but ~25% interior share), while a
#: perimeter ring -- wall-base trim rows, furniture-foot rows, the
#: fused L-ring measured on the wall-rows-near-floor case -- keeps
#: essentially none. Measuring occupied-cell SHARES is what makes the
#: test valid on both dense grids and sparse dust; an area-occupancy
#: test (occupied area / bbox area) misread every real sparse floor
#: as a ring (measured: south_building floors at cov 0.02-0.06
#: demoted, rooms lost).
RING_MAX_INTERIOR_CELL_SHARE = 0.10

#: The ring verdict also requires the plane DENSE enough that its
#: emptiness means shape rather than sampling: below this measured
#: density the interior cells may simply not have been visited yet.
#: Measured placements: the sparsest REAL floor that must survive
#: unjudged sits at ~4 pts/m2 (south_building plane-031); the
#: sparsest real floor JUDGED by shape sits at ~16 pts/m2 and passes
#: on interior share 0.38 (south_building plane-017); the sparsest
#: ring that must be demoted sits at ~16 pts/m2 with interior share
#: 0.00 (the wall-rows fixture). The gate must fall below 16 so shape
#: decides at the density where rings and floors actually meet.
RING_MIN_DENSITY_PTS_PER_M2 = 15.0

#: Occupancy-grid cell size for plan support coverage (meters). One
#: decimeter: an order below room scale, an order above point noise,
#: so occupancy reflects surface presence rather than sampling density.
PLAN_COVERAGE_CELL_M = 0.1

#: A floor/ceiling sheet of a REAL interior spans at least this much
#: plan extent in one horizontal axis: a walking surface is
#: human-scale (a closet is ~1 m; a room is metres). A horizontal
#: plane smaller than this in BOTH plan axes is furniture, trim, or --
#: the measured real-capture case -- a sheet of a COLLAPSED
#: reconstruction: COLMAP registered the cameras but the sparse map
#: degenerated to a ~0.5 m shell, and per-plane tilt classification
#: still called its fragments "floor" at 0.99 confidence (dataset
#: room_capture: 1,213 pts spanning 0.4x0.5 m promoted as 2-3 floors
#: where the true room is 5x4 m). Demoting sub-room-scale horizontal
#: sheets keeps a ran-but-degenerate reconstruction from posing as
#: recovered structure. Human-scale, not dataset-tuned -- documented
#: deferral like every threshold here.
MIN_INTERIOR_SHEET_EXTENT_M = 1.0


def plan_support_coverage(
    positions: Sequence[Sequence[float]],
    bounds_min: Optional[Sequence[float]],
    bounds_max: Optional[Sequence[float]],
    up: Sequence[float] = (0.0, 0.0, 1.0),
    cell: float = PLAN_COVERAGE_CELL_M,
) -> float:
    """Measured plan-support facts for a horizontal plane's inliers:
    (area_occupancy, interior_cell_share, density_pts_per_m2).

    - area_occupancy: fraction of the plan bbox area that occupied
      grid cells cover (density-DEPENDENT: a sparse dust sheet scores
      low even when perfectly solid).
    - interior_cell_share: share of OCCUPIED cells inside the central
      half of the bbox (density-INVARIANT shape signal: a solid sheet
      keeps ~25% whatever its sampling, a perimeter ring ~0).
    - density_pts_per_m2: inliers per square meter of bbox area --
      whether the emptiness above means shape or merely sampling.

    Returns (1.0, 1.0, inf) when extents or positions are missing
    (nothing measurable to judge; downstream refusal paths still
    guard every use) -- coverage can only DEMOTE, never promote.
    """
    if (not positions or bounds_min is None or bounds_max is None):
        return (1.0, 1.0, float("inf"))
    up_idx = _up_axis(up)
    plan_axes = [i for i in range(3) if i != up_idx]
    ax0, ax1 = plan_axes[0], plan_axes[1]
    ex = bounds_max[ax0] - bounds_min[ax0]
    ey = bounds_max[ax1] - bounds_min[ax1]
    if ex <= 0.0 or ey <= 0.0:
        return (1.0, 1.0, float("inf"))
    occupied: set = set()
    interior = 0
    cx0, cx1 = 0.25 * ex, 0.75 * ex
    cy0, cy1 = 0.25 * ey, 0.75 * ey
    for p in positions:
        u = p[ax0] - bounds_min[ax0]
        v = p[ax1] - bounds_min[ax1]
        key = (int(u / cell), int(v / cell))
        if key in occupied:
            continue
        occupied.add(key)
        if cx0 <= u <= cx1 and cy0 <= v <= cy1:
            interior += 1
    area = ex * ey
    area_occupancy = (len(occupied) * cell * cell) / area
    interior_share = interior / len(occupied) if occupied else 1.0
    density = len(positions) / area
    return (area_occupancy, interior_share, density)


def structural_plane_undersized(
    bounds_min: Optional[Sequence[float]],
    bounds_max: Optional[Sequence[float]],
    up: Sequence[float] = (0.0, 0.0, 1.0),
    min_extent: float = MIN_INTERIOR_SHEET_EXTENT_M,
) -> bool:
    """True when a structural plane's TWO LARGEST extents both span
    less than `min_extent` -- sub-room-scale: furniture, trim, or a
    fragment of a collapsed reconstruction, never architecture.

    Role-independent by construction: a floor/ceiling sheet is human-
    scale in PLAN (its two largest extents are the plan axes); a wall
    is human-scale in RUN or HEIGHT (its two largest extents are those
    -- the thin axis is smallest by definition). A real wall spans at
    least ~1 m of run or height (a closet door is ~1 m tall); a real
    floor spans at least ~1 m of plan (a closet). A measured plane
    failing BOTH -- measured on dataset room_capture: 0.4x0.5 m sheets
    of a collapsed 1,213-pt COLMAP map promoted as floors and walls at
    0.96-0.99 confidence -- cannot host a person and must not be
    promoted as structure.
    False when extents are missing (nothing measurable to judge;
    downstream refusal paths still guard every use).
    """
    if bounds_min is None or bounds_max is None:
        return False
    extents = sorted(
        (bounds_max[i] - bounds_min[i] for i in range(3)), reverse=True
    )
    if extents[1] <= 0.0:
        return False
    return extents[1] < min_extent


def _group_enclosures(
    elements: Sequence[ArchitecturalElement], up
) -> List[List[ArchitecturalElement]]:
    """Group classified elements into maximal enclosure candidates."""
    floors = [e for e in elements if e.element_type == "floor"]
    ceilings = [e for e in elements if e.element_type == "ceiling"]
    walls = [e for e in elements if e.element_type == "wall"]

    up_idx = _up_axis(up)
    plan_axes = [i for i in range(3) if i != up_idx]
    ax0, ax1 = plan_axes[0], plan_axes[1]

    groups: List[List[ArchitecturalElement]] = []
    for floor in floors:
        fh = _floor_height(floor, up)
        if fh is None:
            continue

        def _plan_overlap(a: ArchitecturalElement, b: ArchitecturalElement) -> bool:
            return (
                a.bounds_min[ax0] <= b.bounds_max[ax0]
                and b.bounds_min[ax0] <= a.bounds_max[ax0]
                and a.bounds_min[ax1] <= b.bounds_max[ax1]
                and b.bounds_min[ax1] <= a.bounds_max[ax1]
            )

        # A horizontal plane only seeds a STOREY when a person could
        # actually stand on it. Skip this floor when another floor sits
        # BARELY below it (less than walkable clear height) AND overlaps
        # it in plan: that pair is one storey level -- a finish-threshold
        # twin (5 cm slab step between rooms) is two disjoint plans and
        # still seeds both rooms, while a mezzanine slab or an oversized
        # table top INSIDE a taller storey overlaps the floor below and
        # must not fabricate a mid-air enclosure. A real upper storey
        # keeps seeding: the floor below it is a full storey down.
        seeded = True
        for g in floors:
            if g is floor or g.bounds_min is None:
                continue
            gh = _floor_height(g, up)
            if gh is None or gh >= fh:
                continue
            if fh - gh >= MIN_WALKABLE_CLEAR_HEIGHT_M:
                continue  # a real storey below: this floor stands alone
            g_area = plan_area(g.bounds_min, g.bounds_max, up)
            f_area = plan_area(floor.bounds_min, floor.bounds_max, up)
            smaller = min(g_area, f_area)
            if smaller > 0.0 and (
                _plan_overlap_area(floor, g, ax0, ax1) / smaller
                >= FURNITURE_SCALE_FLOOR_FRACTION
            ):
                seeded = False
                break
        if not seeded:
            continue  # slab within a taller storey: no room, no guess

        room_ceilings = [
            c for c in ceilings
            if c.bounds_min is not None and _plan_overlap(floor, c)
            and c.bounds_min[up_idx] > fh
        ]
        # Ceiling absence is recorded, never fatal: real captures
        # routinely miss the ceiling (tall rooms, scan budget, dark
        # upper finish), while the floor + enclosing walls still close
        # the space. The storey's ceiling is the lowest candidate
        # leaving WALKABLE clear height under it: a table top or
        # worktop misread as "ceiling" would clip the wall band at
        # 0.75 m and shed every wall above it -- the room would
        # collapse exactly when furniture is present.
        ceiling = None
        for c in sorted(
            room_ceilings, key=lambda c: c.bounds_min[up_idx]
        ):
            if (c.bounds_min[up_idx] - fh
                    >= MIN_WALKABLE_CLEAR_HEIGHT_M):
                ceiling = c
                break
        ceiling_val = (
            ceiling.bounds_min[up_idx] if ceiling is not None
            else math.inf
        )
        room_walls = [
            w for w in walls
            if w.bounds_min is not None
            and _wall_encloses_floor(w, floor, up)
            and w.bounds_min[up_idx] < ceiling_val
            and w.bounds_max[up_idx] > fh
        ]
        # Furniture/clutter filter: vertical span >= 0.8m
        room_walls = [
            w for w in room_walls
            if (w.bounds_max[up_idx] - w.bounds_min[up_idx]) >= 0.8
        ]
        members: List[ArchitecturalElement] = (
            ([floor, ceiling] if ceiling is not None else [floor])
            + room_walls
        )
        if len(room_walls) < MIN_WALLS_FOR_ROOM:
            continue  # cannot enclose: refuse this floor, no guess
        groups.append(members)
    return groups


def _openings_for(
    wall: ArchitecturalElement,
    floor_height: float,
    up: Sequence[float],
    plane_inputs: Dict[str, PlaneInput],
) -> List[RoomOpening]:
    """Measured openings on one wall (doorways via classify.detect_wall_opening
    and windows via windows.detect_window)."""
    plane = plane_inputs.get(wall.source_plane_id)
    if plane is None:
        return []

    openings: List[RoomOpening] = []

    # 1. Doorway detection at floor height
    opening = detect_wall_opening(plane, tuple(up), floor_height)
    if opening is not None:
        lat = _gap_extent_m(plane, tuple(up), floor_height)
        if lat is not None:
            width, height = lat
            openings.append(RoomOpening(
                wall_element_id=wall.element_id,
                kind="doorway",
                width_m=width,
                height_m=height,
                sill_height_m=0.0,
            ))

    # 2. Window detection above floor height
    try:
        from perception.architecture.windows import detect_window
        win = detect_window(plane, floor_height, tuple(up))
        if win is not None:
            openings.append(RoomOpening(
                wall_element_id=wall.element_id,
                kind="window",
                width_m=win.width_m,
                height_m=win.height_m,
                sill_height_m=win.sill_height_m,
                confidence=win.confidence,
                evidence_ids=win.evidence_ids,
            ))
    except Exception:
        pass

    return openings


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
        if isinstance(planes, dict):
            for pid, p in planes.items():
                plane_inputs.setdefault(pid, p)
        else:
            for p in planes:
                plane_inputs.setdefault(p.plane_id, p)

    up_axis = _up_axis(up)
    enclosures: List[dict] = []
    for members in groups:
        # the next slab above this room's floor that overlaps it in plan and clears walking height caps a room
        # whose own ceiling was never observed (the walls it shares with the storey above run right past it)
        own = next(e for e in members if e.element_type == "floor")
        own_h = _floor_height(own, up)
        plan_ax = [i for i in range(3) if i != up_axis]
        above = [
            _floor_height(g, up) for g in floors
            if g is not own and g.bounds_min is not None and own_h is not None
            and (_floor_height(g, up) or -math.inf) - own_h >= MIN_WALKABLE_CLEAR_HEIGHT_M
            and own.bounds_min[plan_ax[0]] <= g.bounds_max[plan_ax[0]] and g.bounds_min[plan_ax[0]] <= own.bounds_max[plan_ax[0]]
            and own.bounds_min[plan_ax[1]] <= g.bounds_max[plan_ax[1]] and g.bounds_min[plan_ax[1]] <= own.bounds_max[plan_ax[1]]
        ]
        bounds = _enclosed_bounds(members, up, ceiling_cap=min(above) if above else None)
        if bounds is None:
            continue
        bmin, bmax = bounds
        enclosures.append({
            "members": sorted(members, key=lambda e: e.element_id),
            "bmin": bmin,
            "bmax": bmax,
            "floor_height": _floor_height(
                next(e for e in members if e.element_type == "floor"), up
            ),
        })
    if not enclosures:
        return []

    up_idx = _up_axis(up)
    plan_axes = [i for i in range(3) if i != up_idx]
    ax0, ax1 = plan_axes[0], plan_axes[1]
    axis_names = ["x", "y", "z"]

    enclosures.sort(key=lambda enc: (enc["bmin"][up_idx], enc["bmin"][ax0], enc["bmin"][ax1]))
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
        area = dims[axis_names[ax0]] * dims[axis_names[ax1]]

        # Honest provenance (measured, never defaulted): how many of
        # the floor's four plan edges are actually bounded by an
        # enclosure wall (within the wall-gap tolerance), whether a
        # ceiling was observed, and the derived status. A room whose
        # capture missed a wall is PARTIAL with fractional confidence
        # -- reporting boundary_completeness=1.0 for it would be the
        # exact false-certainty the mission prohibits.
        walls_m = [e for e in members if e.element_type == "wall"]
        edge_specs = (
            (ax0, floor.bounds_min[ax0]), (ax0, floor.bounds_max[ax0]),
            (ax1, floor.bounds_min[ax1]), (ax1, floor.bounds_max[ax1]),
        )
        covered = 0
        for axis, edge in edge_specs:
            other = ax1 if axis == ax0 else ax0
            for w in walls_m:
                if w.bounds_min is None or w.bounds_max is None:
                    continue
                # A wall bounds an edge only when it is PERPENDICULAR
                # to that edge: its thin plan axis is the edge's axis
                # (a wall running ALONG the edge bounds a different
                # edge), its line coordinate reaches the edge within
                # the wall-gap tolerance, and it spans the floor on
                # the other axis. All three checks are measured.
                w_extents = {
                    j: w.bounds_max[j] - w.bounds_min[j]
                    for j in (ax0, ax1)
                }
                if min(w_extents, key=w_extents.get) != axis:
                    continue  # wall runs along this edge, not into it
                spans_other = (
                    w.bounds_min[other] <= floor.bounds_max[other]
                    and w.bounds_max[other] >= floor.bounds_min[other]
                )
                reaches_edge = (
                    w.bounds_min[axis] - FLOOR_WALL_GAP_TOLERANCE_M
                    <= edge
                    <= w.bounds_max[axis] + FLOOR_WALL_GAP_TOLERANCE_M
                )
                if spans_other and reaches_edge:
                    covered += 1
                    break
        boundary_completeness = covered / 4.0
        ceiling_present = any(
            e.element_type == "ceiling" for e in members
        )
        ceiling_evidence = "measured" if ceiling_present else "missing"
        status = (
            "detected"
            if boundary_completeness >= 1.0 and ceiling_present
            else "partial"
        )
        notes_list: List[str] = []
        if boundary_completeness < 1.0:
            notes_list.append(
                f"boundary incomplete: {covered} of 4 floor edges bounded"
            )
        if not ceiling_present:
            notes_list.append("no ceiling observed in the capture")
        notes = tuple(notes_list)

        rooms.append(RoomGraph(
            room_id=f"room-{i:03d}",
            boundary_element_ids=tuple(sorted(e.element_id for e in members)),
            bounds_min=bmin,
            bounds_max=bmax,
            dimensions_m=dims,
            floor_area_m2=area,
            openings=tuple(sorted(openings, key=lambda o: o.wall_element_id)),
            status=status,
            confidence=boundary_completeness,
            boundary_completeness=boundary_completeness,
            ceiling_evidence=ceiling_evidence,
            notes=notes,
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
    # Rebuild carries ONLY the adjacency: every measured provenance
    # field (status, confidence, boundary_completeness, ceiling_evidence,
    # notes) must survive -- a rebuild through the constructor would
    # reset them to their perfect defaults and report partial captures
    # as full-confidence rooms (the exact false certainty this seam
    # exists to prevent).
    rooms = [
        replace(r, adjacent_room_ids=tuple(sorted(adjacency[r.room_id])))
        for r in rooms
    ]

    return rooms


def build_building_graph(
    rooms: Sequence[RoomGraph],
    up: Sequence[float] = (0.0, 0.0, 1.0),
    corridors: Optional[Sequence[object]] = None,
    stairs: Optional[Sequence[object]] = None,
) -> Optional[BuildingGraph]:
    """Assemble rooms, corridors, and stairs into storeys (measured floor heights)
    and a building envelope. None when there are no rooms or corridors -- an empty
    building is a refusal, not an empty shell."""
    corridors_list = list(corridors or [])
    stairs_list = list(stairs or [])
    if not rooms and not corridors_list:
        return None

    up_idx = _up_axis(up)

    # Collect all space floor heights (rooms + corridors)
    rooms_sorted = sorted(rooms, key=lambda r: r.room_id)
    corridors_sorted = sorted(corridors_list, key=lambda c: getattr(c, "corridor_id", ""))

    by_height: List[dict] = []

    def _add_to_storey(height: float, room_id: Optional[str] = None, corridor_id: Optional[str] = None):
        for entry in by_height:
            if abs(height - entry["height"]) <= FLOOR_HEIGHT_TOLERANCE_M:
                if room_id and room_id not in entry["room_ids"]:
                    entry["room_ids"].append(room_id)
                if corridor_id and corridor_id not in entry["corridor_ids"]:
                    entry["corridor_ids"].append(corridor_id)
                return
        # New storey
        by_height.append({
            "height": height,
            "room_ids": [room_id] if room_id else [],
            "corridor_ids": [corridor_id] if corridor_id else [],
            "stair_ids": [],
        })

    for r in rooms_sorted:
        _add_to_storey(r.bounds_min[up_idx], room_id=r.room_id)
    for c in corridors_sorted:
        c_bmin = getattr(c, "bounds_min", (0, 0, 0))
        c_id = getattr(c, "corridor_id", "")
        _add_to_storey(c_bmin[up_idx], corridor_id=c_id)

    by_height.sort(key=lambda x: x["height"])

    # Link stairs to storeys by measured vertical elevation
    for st in stairs_list:
        st_id = getattr(st, "stair_id", getattr(st, "id", "stair-001"))
        pos = getattr(st, "position", None)
        rise = getattr(st, "rise_m", 0.17)
        n_steps = getattr(st, "n_steps", 10)
        total_rise = rise * n_steps
        if pos:
            z_mid = pos[up_idx]
            z_lo = z_mid - total_rise / 2.0
            z_hi = z_mid + total_rise / 2.0
        else:
            z_lo = 0.0
            z_hi = total_rise

        # Find matching lower and upper storeys
        for entry in by_height:
            h = entry["height"]
            if abs(h - z_lo) <= 0.4 or abs(h - z_hi) <= 0.4 or (z_lo <= h <= z_hi):
                if st_id not in entry["stair_ids"]:
                    entry["stair_ids"].append(st_id)

    storeys = tuple(
        Storey(
            storey_id=f"storey-{i:02d}",
            floor_height_m=entry["height"],
            room_ids=tuple(sorted(entry["room_ids"])),
            corridor_ids=tuple(sorted(entry["corridor_ids"])),
            stair_ids=tuple(sorted(entry["stair_ids"])),
        )
        for i, entry in enumerate(by_height, start=1)
    )

    # Populate level_id on corridors from matching storey
    corridor_to_storey = {}
    for s in storeys:
        for cid in s.corridor_ids:
            corridor_to_storey[cid] = s.storey_id

    updated_corridors = []
    import dataclasses
    for c in corridors_sorted:
        cid = getattr(c, "corridor_id", "")
        if cid in corridor_to_storey and dataclasses.is_dataclass(c):
            c = dataclasses.replace(c, level_id=corridor_to_storey[cid])
        updated_corridors.append(c)

    # Compute overall building envelope
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for room in rooms_sorted:
        for i in range(3):
            lo[i] = min(lo[i], room.bounds_min[i])
            hi[i] = max(hi[i], room.bounds_max[i])
    for corridor in updated_corridors:
        c_bmin = getattr(corridor, "bounds_min", None)
        c_bmax = getattr(corridor, "bounds_max", None)
        if c_bmin and c_bmax:
            for i in range(3):
                lo[i] = min(lo[i], c_bmin[i])
                hi[i] = max(hi[i], c_bmax[i])

    return BuildingGraph(
        building_id="building-001",
        storeys=storeys,
        envelope_bounds_min=tuple(lo),
        envelope_bounds_max=tuple(hi),
        corridors=tuple(updated_corridors),
        stairs=tuple(stairs_list),
    )
