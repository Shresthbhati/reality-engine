"""Pure measurement math -- the single source of truth for saved measurements.

The viewport previews values client-side, but the application API recomputes
every saved value from the stored points with these functions and rejects a
client value that disagrees. Nothing here knows about units: a value is in the
world's coordinate unit (metres for metric/authored worlds, "model units" for
unscaled reconstructions); the caller decides the label.

Frame: Y-up (the canonical WorldIR / viewport frame). "Vertical" is axis 1.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

KINDS = ("distance", "height", "area", "elevation_difference")
#: kinds whose value is an area (unit squared); the rest are lengths.
AREA_KINDS = frozenset({"area"})
POINTS_MIN = {"distance": 2, "height": 2, "elevation_difference": 2, "area": 3}
POINTS_MAX = {"distance": 2, "height": 2, "elevation_difference": 2, "area": 64}
VERTICAL_AXIS = 1
#: magnitudes beyond this are never a real coordinate; rejects overflow-in-waiting values.
COORD_LIMIT = 1e9
#: an area polygon must be planar to this fraction of its bounding diagonal (float32 pick noise is ~1e-6).
PLANARITY_REL_TOL = 1e-4
PLANARITY_ABS_TOL = 1e-6
#: a polygon whose area is below this fraction of (diagonal squared) is a sliver/collinear, not a measurable area.
DEGENERATE_REL = 1e-9

Point = tuple[float, float, float]


class MeasureError(ValueError):
    """The points cannot honestly produce a value of this kind."""


def validate_points(kind: str, points: Any) -> list[Point]:
    """Return clean (x, y, z) float triples or raise MeasureError (NaN/inf/bool/wrong arity rejected)."""
    if kind not in KINDS:
        raise MeasureError(f"unknown measurement kind {kind!r}; expected one of {', '.join(KINDS)}")
    if not isinstance(points, (list, tuple)):
        raise MeasureError("points must be a list of [x, y, z]")
    if len(points) < POINTS_MIN[kind]:
        raise MeasureError(f"{kind} needs at least {POINTS_MIN[kind]} points, got {len(points)}")
    if len(points) > POINTS_MAX[kind]:
        raise MeasureError(f"{kind} accepts at most {POINTS_MAX[kind]} points, got {len(points)}")
    clean: list[Point] = []
    for i, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 3:
            raise MeasureError(f"point {i} must be [x, y, z]")
        for v in p:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise MeasureError(f"point {i} has a non-numeric coordinate")
            if not math.isfinite(v) or abs(v) > COORD_LIMIT:
                raise MeasureError(f"point {i} has a non-finite or out-of-range coordinate")
        clean.append((float(p[0]), float(p[1]), float(p[2])))
    return clean


def distance(a: Point, b: Point) -> float:
    return math.dist(a, b)


def vertical_delta(a: Point, b: Point) -> float:
    """Signed vertical difference b - a along the up axis."""
    return b[VERTICAL_AXIS] - a[VERTICAL_AXIS]


def _cross(u: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _segments_cross(p1, p2, p3, p4) -> bool:
    """Proper intersection of 2D segments p1p2 and p3p4 (touching endpoints of neighbours is handled by the caller)."""
    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0


def polygon_area(points: Sequence[Point]) -> float:
    """Area of a planar 3D polygon (Newell's method).

    Rejects: fewer than 3 points, collinear/zero-area polygons, polygons that are
    not planar within tolerance, and self-intersecting outlines (whose 'area' is
    not a well-defined region).
    """
    n = len(points)
    if n < 3:
        raise MeasureError("an area needs at least 3 points")
    nx = ny = nz = 0.0
    for i in range(n):
        c = _cross(points[i], points[(i + 1) % n])
        nx, ny, nz = nx + c[0], ny + c[1], nz + c[2]
    normal_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    xs, ys, zs = zip(*points)
    diag = math.dist((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
    area = normal_len / 2.0
    if diag == 0.0 or area <= DEGENERATE_REL * diag * diag:
        raise MeasureError("degenerate polygon: the points are collinear or coincident (zero area)")
    unit = (nx / normal_len, ny / normal_len, nz / normal_len)
    cx, cy, cz = sum(xs) / n, sum(ys) / n, sum(zs) / n
    tol = max(PLANARITY_REL_TOL * diag, PLANARITY_ABS_TOL)
    for p in points:
        off = abs((p[0] - cx) * unit[0] + (p[1] - cy) * unit[1] + (p[2] - cz) * unit[2])
        if off > tol:
            raise MeasureError("polygon is not planar (a point is off the polygon's plane)")
    # Self-intersection, in the plane: drop the axis the normal points along most.
    drop = max(range(3), key=lambda a: abs(unit[a]))
    keep = [a for a in range(3) if a != drop]
    flat = [(p[keep[0]], p[keep[1]]) for p in points]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges share a vertex
            if _segments_cross(flat[i], flat[(i + 1) % n], flat[j], flat[(j + 1) % n]):
                raise MeasureError("polygon outline crosses itself")
    return area


def compute(kind: str, points: Any) -> float:
    """Validate then compute. Length kinds return a length; 'area' returns an area; elevation_difference is signed."""
    pts = validate_points(kind, points)
    if kind == "distance":
        return distance(pts[0], pts[1])
    if kind == "height":
        return abs(vertical_delta(pts[0], pts[1]))
    if kind == "elevation_difference":
        return vertical_delta(pts[0], pts[1])
    return polygon_area(pts)


def values_agree(a: float, b: float, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


# ---- unit / scale honesty -------------------------------------------------

SCALE_STATES = ("metric", "authored", "unknown")


def unit_for(kind: str, scale_state: str) -> str:
    """'m' / 'm2' only for metric or authored worlds; an unknown/relative scale is NEVER labelled metres."""
    if scale_state not in SCALE_STATES:
        raise MeasureError(f"scale_state must be one of {', '.join(SCALE_STATES)}")
    base = "m" if scale_state in ("metric", "authored") else "model_units"
    return base + ("2" if kind in AREA_KINDS else "")
