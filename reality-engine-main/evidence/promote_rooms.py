"""Room inference from plane structure -> WorldIR promotion (the next
geometric-reasoning loop after planes: spec sec 17 "rooms", sec 13
"room dimensions", sec 18 scene-graph relationships).

The planes slice (perception/geometry/planes.py + orientation.py,
promoted by evidence/promote_planes.py) produced typed FLOOR / WALL /
CEILING entities. This module closes the next loop: a ROOM exists where
the geometry actually supports one -- walls resting on a floor and
enclosing it with a closed boundary ring. The derivation is pure
geometry over the same evidence the planes were promoted from; no LLM,
no invented extents.

How the ring is derived (and why this way):

  - Each boundary wall contributes the LINE wall-plane intersect
    floor-plane, expressed in floor-plane 2D coordinates. The LINE is
    the agreed structure -- two planes' intersection is exact even when
    the wall's inlier extent is ragged (in real reconstructions the
    floor plane swallows the wall bases and adjacent wall planes swallow
    each wall's edge columns, so AABB endpoints do NOT meet at shared
    corners; observed and handled here).
  - Each wall also contributes its inlier SUPPORT interval along its
    line (projection of its inlier AABB) -- the honest extent its
    points actually cover.
  - Corners are intersections of consecutive boundary lines; every
    corner must be reachable from both walls' support intervals (within
    CORNER_REACH_FRACTION of the wall's span -- a wall whose points
    NOWHERE approach the corner it is supposed to form does not form
    it), and consecutive lines must not be parallel (parallel
    "neighbors" never enclose anything).
  - The floor itself must be a walkable surface: at least
    MIN_FLOOR_INTERIOR_INLIERS strictly inside the ring. The wall-base
    ring of a room is itself a horizontal plane spanning the room's
    full extent -- with zero interior points -- and must not spawn a
    duplicate room.
  - Anything unmet is reported honestly -- NO_CLOSED_RING /
    UNCLOSED_RING / FLOOR_NOT_INTERIOR with per-test notes -- and NO
    room is inferred. A half-built room, an open facade, or two walls
    produce no ROOM entity: absence is honest, a guessed rectangle is
    fabrication.

Measurements, all derived from the same real inlier geometry (ESTIMATED,
precision from the plane fit RMS like promote_planes): floor area
(shoelace over the ring), floor dimensions (ring extents along the
floor's two in-plane axes), room height (floor plane to the highest
wall point). Every measurement lands in the promoted entity's
custom_properties and the RoomPromotionResult.

Relationships: CONTAINS from the room to each wall/floor/ceiling
entity, PART_OF from each part back to the room (world_ir
RelationshipKind vocabulary; INFERRED provenance -- the edges are
derived, and each records the derivation in metadata so "why does the
engine believe this wall is part of this room?" is answerable from the
WorldIR alone).

Known limitation, labelled: the ring walk assumes a simply-connected,
convex boundary (corners in angular order). Non-convex rooms (L-shapes)
will produce a conservatively wrong-or-failed ring rather than a silent
lie, but proper non-convex ring tracing is undone work.

Deterministic: canonical ordering (planes, lines, corners, outputs),
pure float math, no clocks, no RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from provenance import Provenance
from world_ir import (
    Entity,
    EntityType,
    Measurement,
    Observation,
    Relationship,
    RelationshipKind,
    WorldIR,
)

#: Minimum number of contacting walls for a closed interior boundary.
MIN_WALLS = 3

#: A wall bounds a room only if it actually rests on the floor: its
#: inliers reach within this tolerance of the floor plane. Real
#: reconstructions legitimately lose a wall's bottom row to the floor
#: plane (shared boundary points are claimed by whichever plane extracts
#: first -- observed with grid fixtures: walls started 0.2 m up), so
#: this must tolerate the loss while still rejecting a genuinely
#: floating wall (a loft band 1.5 m up bounds nothing).
WALL_FLOOR_CONTACT_TOLERANCE_M = 0.3

#: All walls of one room must top out within this tolerance of each
#: other -- mixed top heights mean the boundary does not close honestly.
WALL_TOP_TOLERANCE_M = 0.25

#: A ring corner must lie within this fraction of a wall's support span
#: (measured along the wall's line, beyond the interval its inliers
#: actually cover) for that wall to count as forming the corner. Generous
#: because edge-column loss shrinks support intervals by whole grid
#: steps; still bounded, so a wall nowhere near its claimed corner is
#: rejected.
CORNER_REACH_FRACTION = 0.6

#: Two lines are the same boundary line (co-planar wall patches) when
#: their directions differ by at most this many degrees AND their
#: perpendicular offsets differ by at most this many meters; their
#: support intervals union into one segment.
LINE_MERGE_ANGLE_DEG = 1.0
LINE_MERGE_OFFSET_M = 0.02

#: A floor candidate must have at least this many inliers strictly
#: INSIDE the boundary ring to count as a walkable surface.
MIN_FLOOR_INTERIOR_INLIERS = 3


class RoomInferenceError(ValueError):
    pass


# ----------------------------------------------------------------------
# Plane summary: the minimal facts room inference needs, resolved once
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PlaneSummary:
    """Minimal plane facts for room inference, built by the caller from
    an OrientedPlane plus its resolved inlier positions/bounds (see
    evidence.promote_planes) so this module never re-derives point
    membership and never silently shrinks a plane's evidence set.
    """

    plane_id: str
    role: str  # "floor" | "ceiling" | "wall" | "unknown"
    normal: Tuple[float, float, float]  # unit
    d: float
    inlier_rms_m: float
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    #: The plane's actual inlier positions -- required because the
    #: walkable-floor test (is a floor inlier INSIDE the boundary ring?)
    #: cannot be answered from a bounding box: the wall-base ring of a
    #: room spans the room's full extent but has no interior points.
    inlier_positions: Tuple[Tuple[float, float, float], ...]
    entity_id: str  # the WorldIR entity this plane was promoted as


def plane_summary_from(oriented, positions) -> PlaneSummary:
    """Build a PlaneSummary from an OrientedPlane and its resolved
    inlier positions (one call per plane; convenience over manual
    field assembly)."""
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    return PlaneSummary(
        plane_id=oriented.plane.plane_id,
        role=oriented.role,
        normal=oriented.normal,
        d=oriented.d,
        inlier_rms_m=oriented.plane.inlier_rms_distance_m,
        bounds_min=(min(xs), min(ys), min(zs)),
        bounds_max=(max(xs), max(ys), max(zs)),
        inlier_positions=tuple(positions),
        entity_id="",  # filled by the caller after plane promotion
    )


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def _plane_signed_distance(
    normal: Tuple[float, float, float], d: float, point: Tuple[float, float, float]
) -> float:
    return normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2] + d


def _aabb_corners(
    bounds_min: Tuple[float, float, float], bounds_max: Tuple[float, float, float]
) -> List[Tuple[float, float, float]]:
    return [
        (x, y, z)
        for x in (bounds_min[0], bounds_max[0])
        for y in (bounds_min[1], bounds_max[1])
        for z in (bounds_min[2], bounds_max[2])
    ]


def _floor_basis(
    normal: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Two orthonormal in-plane axes for a floor's unit normal
    (deterministic: the world axis least aligned with the normal seeds
    the first basis vector)."""
    seed = (1.0, 0.0, 0.0) if abs(normal[0]) <= abs(normal[2]) else (0.0, 0.0, 1.0)
    dot = sum(n * s for n, s in zip(normal, seed))
    u = tuple(s - dot * n for s, n in zip(seed, normal))
    u_len = math.sqrt(sum(c * c for c in u))
    u = (u[0] / u_len, u[1] / u_len, u[2] / u_len)
    v = (
        normal[1] * u[2] - normal[2] * u[1],
        normal[2] * u[0] - normal[0] * u[2],
        normal[0] * u[1] - normal[1] * u[0],
    )
    return u, v


@dataclass(frozen=True)
class _WallLine:
    """A wall-intersection-floor boundary line in floor-plane 2D
    coordinates: point + unoriented direction, plus the support interval
    the wall's inliers actually cover along the line."""

    point: Tuple[float, float]
    direction: Tuple[float, float]  # unit
    support_lo: float  # param along direction
    support_hi: float
    wall_plane_ids: Tuple[str, ...]

    def span(self) -> float:
        return self.support_hi - self.support_lo

    def angle(self) -> float:
        """Unoriented (folded) line angle in [0, pi)."""
        theta = math.atan2(self.direction[1], self.direction[0])
        if theta < 0.0:
            theta += math.pi
        if theta >= math.pi:
            theta -= math.pi
        return theta

    def directed_angle(self) -> float:
        """Oriented direction angle in [0, 2pi) -- the ring walk orders
        by THIS, because opposite walls of a room have directions pi
        apart and must sit opposite in the cyclic order (sorting by the
        folded angle places parallel walls adjacent, where they falsely
        read as a parallel non-enclosure; observed and fixed)."""
        theta = math.atan2(self.direction[1], self.direction[0])
        if theta < 0.0:
            theta += 2.0 * math.pi
        return theta

    def offset(self) -> float:
        """Signed perpendicular offset of the line through `point` with
        `direction`: offset = point x direction (scalar cross)."""
        return self.point[0] * self.direction[1] - self.point[1] * self.direction[0]

    def covers(self, parameter: float) -> bool:
        margin = CORNER_REACH_FRACTION * max(self.span(), 1e-9)
        return self.support_lo - margin <= parameter <= self.support_hi + margin


def _wall_lines_in_floor(
    floor: PlaneSummary, walls: List[PlaneSummary]
) -> List[_WallLine]:
    """Per wall, its boundary line (wall plane intersect floor plane) in
    floor-plane coordinates, with the wall's inlier support interval.

    Line direction = n_floor x n_wall; a point on it solved as
    p0 = alpha*n_floor + beta*n_wall with n.p0 = -d for both planes
    (exact 2x2 solve). Support = projection of the wall's inlier AABB
    corners onto the line.
    """
    n1, d1 = floor.normal, floor.d
    u_axis, v_axis = _floor_basis(n1)
    origin = tuple(-d1 * n1[i] for i in range(3))  # a point on the floor plane

    lines: List[_WallLine] = []
    for wall in walls:
        n2, d2 = wall.normal, wall.d
        cross = (
            n1[1] * n2[2] - n1[2] * n2[1],
            n1[2] * n2[0] - n1[0] * n2[2],
            n1[0] * n2[1] - n1[1] * n2[0],
        )
        cross_len = math.sqrt(sum(c * c for c in cross))
        if cross_len < 1e-9:
            continue  # wall parallel to floor: not a boundary wall
        direction = (cross[0] / cross_len, cross[1] / cross_len, cross[2] / cross_len)
        c = sum(n1[i] * n2[i] for i in range(3))
        denom = 1.0 - c * c
        if abs(denom) < 1e-12:
            continue
        beta = (d1 * c - d2) / denom
        alpha = -d1 - beta * c
        p0 = tuple(alpha * n1[i] + beta * n2[i] for i in range(3))
        t_values = [
            sum((corner[i] - p0[i]) * direction[i] for i in range(3))
            for corner in _aabb_corners(wall.bounds_min, wall.bounds_max)
        ]
        t_lo, t_hi = min(t_values), max(t_values)
        end_a = tuple(p0[i] + t_lo * direction[i] for i in range(3))
        end_b = tuple(p0[i] + t_hi * direction[i] for i in range(3))
        a2 = (
            sum((end_a[i] - origin[i]) * u_axis[i] for i in range(3)),
            sum((end_a[i] - origin[i]) * v_axis[i] for i in range(3)),
        )
        b2 = (
            sum((end_b[i] - origin[i]) * u_axis[i] for i in range(3)),
            sum((end_b[i] - origin[i]) * v_axis[i] for i in range(3)),
        )
        du, dv = b2[0] - a2[0], b2[1] - a2[1]
        seg_len = math.hypot(du, dv)
        if seg_len < 1e-9:
            continue
        lines.append(_WallLine(
            point=a2,
            direction=(du / seg_len, dv / seg_len),
            support_lo=0.0,
            support_hi=seg_len,
            wall_plane_ids=(wall.plane_id,),
        ))
    return lines


def _merge_coplanar_lines(lines: List[_WallLine]) -> List[_WallLine]:
    """Union lines that are the same boundary line (co-planar wall
    patches): same angle, same perpendicular offset, overlapping
    support. Deterministic."""
    remaining = sorted(lines, key=lambda l: (round(l.angle(), 9), round(l.offset(), 9), l.wall_plane_ids))
    merged: List[_WallLine] = []
    while remaining:
        line = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                angle_diff = math.degrees(abs(line.angle() - other.angle()))
                if angle_diff > LINE_MERGE_ANGLE_DEG:
                    continue
                if abs(line.offset() - other.offset()) > LINE_MERGE_OFFSET_M:
                    continue
                # Parameterize other's endpoints on line's axis.
                d = line.direction
                t_other_lo = (other.point[0] - line.point[0]) * d[0] + (other.point[1] - line.point[1]) * d[1]
                t_other_hi = t_other_lo + other.span() * (
                    1.0 if (other.direction[0] * d[0] + other.direction[1] * d[1]) >= 0 else -1.0
                )
                lo, hi = min(t_other_lo, t_other_hi), max(t_other_lo, t_other_hi)
                union_lo, union_hi = min(line.support_lo, lo), max(line.support_hi, hi)
                line = _WallLine(
                    point=line.point,
                    direction=line.direction,
                    support_lo=union_lo,
                    support_hi=union_hi,
                    wall_plane_ids=tuple(sorted(set(line.wall_plane_ids) | set(other.wall_plane_ids))),
                )
                remaining.remove(other)
                changed = True
        merged.append(line)
    return merged


def _line_intersection(
    a: _WallLine, b: _WallLine
) -> Optional[Tuple[float, float]]:
    """Intersection of two unoriented 2D lines, or None if parallel."""
    d1, d2 = a.direction, b.direction
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-12:
        return None
    # Solve a.point + t*d1 == b.point + s*d2.
    dx, dy = b.point[0] - a.point[0], b.point[1] - a.point[1]
    t = (dx * d2[1] - dy * d2[0]) / det
    return (a.point[0] + t * d1[0], a.point[1] + t * d1[1])


def _parameter_on(line: _WallLine, point: Tuple[float, float]) -> float:
    return (point[0] - line.point[0]) * line.direction[0] + (
        point[1] - line.point[1]
    ) * line.direction[1]


def _point_in_polygon(
    point: Tuple[float, float], vertices: Tuple[Tuple[float, float], ...]
) -> bool:
    """Even-odd ray-cast point-in-polygon (strictly interior points
    only -- boundary points are not interior)."""
    x, y = point
    inside = False
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


@dataclass(frozen=True)
class _Ring:
    """A closed boundary loop in floor-plane coordinates."""

    vertices: Tuple[Tuple[float, float], ...]
    segment_wall_ids: Tuple[str, ...]  # wall(s) backing each edge

    def shoelace_area(self) -> float:
        area2 = 0.0
        n = len(self.vertices)
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            area2 += x1 * y2 - x2 * y1
        return abs(area2) / 2.0

    def extents(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        us = [v[0] for v in self.vertices]
        vs = [v[1] for v in self.vertices]
        return (min(us), max(us)), (min(vs), max(vs))


def _trace_ring_from_lines(
    lines: List[_WallLine], interior_point: Tuple[float, float]
) -> Optional[_Ring]:
    """Order boundary lines into a closed loop of corners.

    Each line is oriented so the interior point (the floor's inlier
    centroid -- real evidence, not a guess) lies on its LEFT; oriented
    lines are then sorted by directed angle, which walks a convex ring
    counter-clockwise with opposite walls opposite in the order. Every
    consecutive pair must intersect at a corner reachable from BOTH
    walls' support intervals (within CORNER_REACH_FRACTION of their
    span). Any parallel consecutive pair or unreachable corner fails
    honestly (None).
    """
    if len(lines) < MIN_WALLS:
        return None

    oriented: List[_WallLine] = []
    for line in lines:
        w = (interior_point[0] - line.point[0], interior_point[1] - line.point[1])
        cross = line.direction[0] * w[1] - line.direction[1] * w[0]
        if cross < 0.0:
            # Interior on the right: flip the walk direction. Support
            # parameters re-express on the flipped axis by negation.
            oriented.append(_WallLine(
                point=line.point,
                direction=(-line.direction[0], -line.direction[1]),
                support_lo=-line.support_hi,
                support_hi=-line.support_lo,
                wall_plane_ids=line.wall_plane_ids,
            ))
        else:
            oriented.append(line)

    ordered = sorted(oriented, key=lambda l: (round(l.directed_angle(), 9), round(l.offset(), 9)))
    vertices: List[Tuple[float, float]] = []
    edge_walls: List[str] = []
    n = len(ordered)
    for i in range(n):
        current = ordered[i]
        nxt = ordered[(i + 1) % n]
        corner = _line_intersection(current, nxt)
        if corner is None:
            return None  # consecutive boundary lines parallel: no enclosure
        if not current.covers(_parameter_on(current, corner)):
            return None  # current wall's points never approach this corner
        if not nxt.covers(_parameter_on(nxt, corner)):
            return None
        vertices.append(corner)
        edge_walls.append("|".join(current.wall_plane_ids))
    return _Ring(vertices=tuple(vertices), segment_wall_ids=tuple(edge_walls))


# ----------------------------------------------------------------------
# Detection (pure -- no WorldIR mutation here)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class DetectedRoom:
    """A room candidate inferred from plane structure.

    status "detected" means a closed boundary ring exists and promotion
    is allowed; every other status (NO_CLOSED_RING, UNCLOSED_RING,
    FLOOR_NOT_INTERIOR) comes with notes explaining exactly which
    geometric test failed -- and no ring, no measurements, no entity.
    """

    room_id: str
    floor: PlaneSummary
    ceiling: Optional[PlaneSummary]
    walls: Tuple[PlaneSummary, ...]
    ring: Optional[_Ring]
    status: str
    notes: Tuple[str, ...]

    def floor_area_m2(self) -> float:
        if self.ring is None:
            raise RoomInferenceError(f"{self.room_id}: no ring -- no honest area")
        return self.ring.shoelace_area()

    def floor_dimensions_m(self) -> Tuple[float, float]:
        if self.ring is None:
            raise RoomInferenceError(f"{self.room_id}: no ring -- no honest dimensions")
        (u_lo, u_hi), (v_lo, v_hi) = self.ring.extents()
        return (u_hi - u_lo, v_hi - v_lo)

    def height_m(self) -> float:
        """Room height along the floor normal.

        With an observed ceiling plane: the exact floor-to-ceiling
        parallel-plane distance (re-signing the ceiling's constant into
        the floor's normal direction, as promote_planes._plane_gap
        does). Without one: floor plane to the highest wall point -- an
        honest LOWER bound on the true height, since the ceiling may be
        higher than the walls' surviving inliers reach (real
        reconstructions lose wall top rows to the ceiling plane).
        """
        if self.ceiling is not None:
            f, c = self.floor, self.ceiling
            dot = f.normal[0] * c.normal[0] + f.normal[1] * c.normal[1] + f.normal[2] * c.normal[2]
            sign = 1.0 if dot >= 0.0 else -1.0
            return abs(f.d - sign * c.d)
        distances = [
            _plane_signed_distance(self.floor.normal, self.floor.d, corner)
            for w in self.walls
            for corner in _aabb_corners(w.bounds_min, w.bounds_max)
        ]
        return abs(max(distances))


def _failed_candidate(
    floor: PlaneSummary, room_id: str, status: str, notes: List[str]
) -> DetectedRoom:
    return DetectedRoom(
        room_id=room_id, floor=floor, ceiling=None, walls=(),
        ring=None, status=status, notes=tuple(notes),
    )


def detect_rooms(
    planes: Sequence[PlaneSummary],
    up: Tuple[float, float, float],
) -> List[DetectedRoom]:
    """Infer rooms from classified plane summaries.

    `up` is the world up vector (same discipline as orientation.py:
    non-degenerate or raise). Walls must rest on the floor, share a top
    height, close a boundary ring of corners their inliers support, and
    the floor must have interior support -- anything less yields a
    candidate with an honest failure status and notes, never a
    fabricated room.

    Deterministic: floors ordered by plane_id, walls sorted by plane_id,
    ring lines ordered by (angle, offset).
    """
    up_len = math.sqrt(sum(c * c for c in up))
    if up_len == 0.0 or not math.isfinite(up_len):
        raise RoomInferenceError(f"up vector must be non-degenerate, got {up!r}")

    floors = sorted((p for p in planes if p.role == "floor"), key=lambda p: p.plane_id)
    walls_by_id = {p.plane_id: p for p in planes if p.role == "wall"}
    ceilings = sorted((p for p in planes if p.role == "ceiling"), key=lambda p: p.plane_id)

    rooms: List[DetectedRoom] = []
    for floor in floors:
        room_id = f"room-candidate-{floor.plane_id}"
        notes: List[str] = []
        walls = sorted(walls_by_id.values(), key=lambda p: p.plane_id)

        # Contact: every wall considered must actually rest on the floor.
        # The test uses each wall's LOWEST SIGNED distance to the floor
        # plane: a resting wall's lowest point sits at the floor (within
        # tolerance -- real reconstructions legitimately lose a wall's
        # bottom row to the floor plane, observed with grid fixtures), a
        # floating band sits above it, and a wall PIERCING the plane (a
        # horizontal slab crossing the room) has corners far BELOW it.
        # min-of-|distance| would wrongly accept piercing walls (a corner
        # exactly on the plane reads as contact); the signed minimum does
        # not.
        contacting: List[PlaneSummary] = []
        for wall in walls:
            lowest = min(
                _plane_signed_distance(floor.normal, floor.d, c)
                for c in _aabb_corners(wall.bounds_min, wall.bounds_max)
            )
            if abs(lowest) <= WALL_FLOOR_CONTACT_TOLERANCE_M:
                contacting.append(wall)
            else:
                notes.append(
                    f"wall {wall.plane_id} does not rest on this floor "
                    f"(lowest point {lowest:+.3f} m relative to the plane) -- excluded"
                )
        if len(contacting) < MIN_WALLS:
            notes.append(
                f"only {len(contacting)} wall(s) rest on this floor "
                f"(need >= {MIN_WALLS}): NO_CLOSED_RING"
            )
            rooms.append(_failed_candidate(floor, room_id, "NO_CLOSED_RING", notes))
            continue

        # Shared top height along the floor normal.
        wall_tops = {
            w.plane_id: max(
                _plane_signed_distance(floor.normal, floor.d, c)
                for c in _aabb_corners(w.bounds_min, w.bounds_max)
            )
            for w in contacting
        }
        top_spread = max(wall_tops.values()) - min(wall_tops.values())
        if top_spread > WALL_TOP_TOLERANCE_M:
            notes.append(
                f"wall top heights spread {top_spread:.3f} m > "
                f"{WALL_TOP_TOLERANCE_M} m: NO_CLOSED_RING"
            )
            rooms.append(_failed_candidate(floor, room_id, "NO_CLOSED_RING", notes))
            continue

        # Boundary ring from wall lines + support-validated corners.
        # Interior reference: the floor's inlier centroid -- a real,
        # evidence-derived point (for a genuine floor it is inside the
        # room; the walkable-floor test below independently verifies
        # interior support).
        if not floor.inlier_positions:
            notes.append("floor plane has no inliers: NO_CLOSED_RING")
            rooms.append(_failed_candidate(floor, room_id, "NO_CLOSED_RING", notes))
            continue
        centroid = (
            sum(p[0] for p in floor.inlier_positions) / len(floor.inlier_positions),
            sum(p[1] for p in floor.inlier_positions) / len(floor.inlier_positions),
            sum(p[2] for p in floor.inlier_positions) / len(floor.inlier_positions),
        )
        u_axis, v_axis = _floor_basis(floor.normal)
        origin = tuple(-floor.d * floor.normal[i] for i in range(3))
        interior_2d = (
            sum((centroid[i] - origin[i]) * u_axis[i] for i in range(3)),
            sum((centroid[i] - origin[i]) * v_axis[i] for i in range(3)),
        )
        lines = _merge_coplanar_lines(_wall_lines_in_floor(floor, contacting))
        ring = _trace_ring_from_lines(lines, interior_2d)
        if ring is None:
            notes.append(
                f"{len(lines)} boundary line(s) do not form a closed ring of "
                "supported corners: UNCLOSED_RING"
            )
            rooms.append(_failed_candidate(floor, room_id, "UNCLOSED_RING", notes))
            continue

        # Walkable-floor check: interior support inside the ring.
        interior_count = 0
        for position in floor.inlier_positions:
            uv = (
                sum((position[i] - origin[i]) * u_axis[i] for i in range(3)),
                sum((position[i] - origin[i]) * v_axis[i] for i in range(3)),
            )
            if _point_in_polygon(uv, ring.vertices):
                interior_count += 1
                if interior_count >= MIN_FLOOR_INTERIOR_INLIERS:
                    break
        if interior_count < MIN_FLOOR_INTERIOR_INLIERS:
            notes.append(
                f"floor plane has {interior_count} interior inlier(s) inside the "
                f"boundary ring (need >= {MIN_FLOOR_INTERIOR_INLIERS}): a "
                "boundary line, not a walkable floor: FLOOR_NOT_INTERIOR"
            )
            rooms.append(_failed_candidate(floor, room_id, "FLOOR_NOT_INTERIOR", notes))
            continue

        # Ceiling: the ceiling plane sitting at the shared wall top
        # (within tolerance). None is honest -- walls can enclose a room
        # whose ceiling was never observed.
        wall_top = max(wall_tops.values())
        ceiling: Optional[PlaneSummary] = None
        for candidate in ceilings:
            ceiling_height = min(
                _plane_signed_distance(floor.normal, floor.d, c)
                for c in _aabb_corners(candidate.bounds_min, candidate.bounds_max)
            )
            if abs(ceiling_height - wall_top) <= WALL_TOP_TOLERANCE_M:
                ceiling = candidate
                break
        if ceiling is None:
            notes.append("no ceiling plane at the wall-top height: room open above (honest)")

        # Ring-participating walls, in ring order (merged lines carry
        # '|'joined plane ids).
        ring_wall_ids: List[str] = []
        for seg_ids in ring.segment_wall_ids:
            for pid in seg_ids.split("|"):
                if pid not in ring_wall_ids:
                    ring_wall_ids.append(pid)
        ring_walls = tuple(walls_by_id[pid] for pid in ring_wall_ids)

        rooms.append(DetectedRoom(
            room_id=f"room-{floor.plane_id}",
            floor=floor,
            ceiling=ceiling,
            walls=ring_walls,
            ring=ring,
            status="detected",
            notes=tuple(notes),
        ))
    return rooms


# ----------------------------------------------------------------------
# Promotion (the write path -- mutates WorldIR like promote_planes does)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RoomPromotionResult:
    entity: Entity
    contains_ids: Tuple[str, ...]
    measurements: Tuple[Tuple[str, Measurement], ...]


def promote_room_to_entity(
    room: DetectedRoom,
    world: WorldIR,
    entity_id: str,
    entity_name: str = "",
) -> RoomPromotionResult:
    """Promote one detected room into WorldIR.

    Refuses (raises RoomInferenceError) anything that is not a detected
    room -- an UNCLOSED_RING candidate has no honest entity, only notes.

    Writes: a ROOM Entity (INFERRED -- the type is derived from
    geometry; the underlying planes stay RECONSTRUCTED on their own
    geometries), CONTAINS room->part edges plus PART_OF part->room edges
    for every wall/floor/ceiling in the boundary, and ESTIMATED
    floor-area / floor-dimension / height Measurements derived from the
    ring and the wall tops. Deterministic; no clocks, no RNG.
    """
    if room.status != "detected" or room.ring is None:
        raise RoomInferenceError(
            f"refusing to promote {room.room_id!r}: status={room.status!r} -- "
            "only a closed-ring room becomes a WorldIR entity"
        )

    part_ids: List[str] = [room.floor.entity_id]
    part_ids.extend(w.entity_id for w in room.walls)
    if room.ceiling is not None:
        part_ids.append(room.ceiling.entity_id)

    # Every part must exist: promotion builds on promote_planes output;
    # a missing entity would dangle the relationship graph.
    missing = [pid for pid in part_ids if pid not in world.entities]
    if missing:
        raise RoomInferenceError(
            f"room parts not present in WorldIR: {missing[:5]} -- promote the "
            "planes first (evidence.promote_planes.promote_plane_to_entity)"
        )

    confidence = min(world.entities[pid].confidence for pid in part_ids)

    relationships: List[Relationship] = []
    for pid in part_ids:
        relationships.append(Relationship(
            kind=RelationshipKind.CONTAINS,
            target_id=pid,
            confidence=confidence,
            provenance=Provenance.INFERRED,
            metadata={"derived_from": "wall_floor_ring_closure"},
        ))
        world.entities[pid].relationships.append(Relationship(
            kind=RelationshipKind.PART_OF,
            target_id=entity_id,
            confidence=confidence,
            provenance=Provenance.INFERRED,
            metadata={"derived_from": "wall_floor_ring_closure"},
        ))

    # Measurements from the ring (hand-checkable in tests).
    area = room.floor_area_m2()
    dim_u, dim_v = room.floor_dimensions_m()
    height = room.height_m()
    precision = max(room.floor.inlier_rms_m, 0.01)
    measurements: Tuple[Tuple[str, Measurement], ...] = (
        ("floor_area_m2", Measurement(
            value=area, unit="meter^2", precision=precision * max(dim_u, dim_v),
            provenance=Provenance.ESTIMATED, confidence=confidence,
        )),
        ("floor_dimension_u_m", Measurement(
            value=dim_u, unit="meter", precision=precision,
            provenance=Provenance.ESTIMATED, confidence=confidence,
        )),
        ("floor_dimension_v_m", Measurement(
            value=dim_v, unit="meter", precision=precision,
            provenance=Provenance.ESTIMATED, confidence=confidence,
        )),
        ("height_m", Measurement(
            value=height, unit="meter", precision=precision,
            provenance=Provenance.ESTIMATED, confidence=confidence,
        )),
    )

    entity = Entity(
        id=entity_id,
        name=entity_name or entity_id,
        type=EntityType.ROOM,
        geometry_ids=[],  # the room's extent lives in its parts, not new geometry
        relationships=relationships,
        provenance=Provenance.INFERRED,
        confidence=confidence,
    )
    for key, measurement in measurements:
        entity.custom_properties[key] = measurement.value
    entity.observations.append(Observation(
        id=f"obs-room-{entity_id}",
        sensor_type="room_inference",
        confidence=confidence,
        metadata={
            "floor_plane_id": room.floor.plane_id,
            "wall_plane_ids": [w.plane_id for w in room.walls],
            "ceiling_plane_id": room.ceiling.plane_id if room.ceiling else None,
            "boundary_vertex_count": len(room.ring.vertices),
            "floor_area_m2": area,
            "floor_dimensions_m": [dim_u, dim_v],
            "height_m": height,
            "notes": list(room.notes),
        },
    ))

    world.entities[entity_id] = entity
    return RoomPromotionResult(
        entity=entity,
        contains_ids=tuple(part_ids),
        measurements=measurements,
    )
