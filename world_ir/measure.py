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

KINDS = ("distance", "height", "area", "elevation_difference", "angle")
#: kinds whose value is an area (unit squared); the rest are lengths.
AREA_KINDS = frozenset({"area"})
POINTS_MIN = {"distance": 2, "height": 2, "elevation_difference": 2, "area": 3, "angle": 3}
POINTS_MAX = {"distance": 2, "height": 2, "elevation_difference": 2, "area": 64, "angle": 3}
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


def angle(a: Point, b: Point, c: Point) -> float:
    """Angle in degrees at vertex b formed by rays ba and bc (0.0 to 180.0 deg)."""
    v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    l1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
    l2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    if l1 == 0.0 or l2 == 0.0:
        raise MeasureError("degenerate angle: coincident vertex and ray endpoint")
    dot = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (l1 * l2)
    clamped = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(clamped))


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
    if kind == "angle":
        return angle(pts[0], pts[1], pts[2])
    return polygon_area(pts)


def values_agree(a: float, b: float, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


# ---- unit / scale honesty -------------------------------------------------

SCALE_STATES = ("metric", "authored", "unknown")


def unit_for(kind: str, scale_state: str) -> str:
    """'m' / 'm2' only for metric or authored worlds; an unknown/relative scale is NEVER labelled metres."""
    if scale_state not in SCALE_STATES:
        raise MeasureError(f"scale_state must be one of {', '.join(SCALE_STATES)}")
    if kind == "angle":
        return "deg"
    base = "m" if scale_state in ("metric", "authored") else "model_units"
    return base + ("2" if kind in AREA_KINDS else "")


# ---- dimension state (plan D1) ----------------------------------------------
# Extends the measured/reconstructed/inferred distinction (backlog P0.16); it is not a second vocabulary.
#: weakest -> strongest.
DIMENSION_STATES = ("ESTIMATED", "RECONSTRUCTED", "MEASURED", "VERIFIED")


def scale_state_of(world_metadata: Any) -> str:
    """'metric' | 'unknown' from a WorldIR ``metadata`` dict (metadata.scale.state). RELATIVE/missing => 'unknown'."""
    scale = (world_metadata or {}).get("scale") if isinstance(world_metadata, dict) else None
    state = str((scale or {}).get("state") or "").lower()
    return "metric" if state == "metric" else "unknown"


def effective_state(requested: str, scale_state: str, scale_dimension_state: str | None = None) -> str:
    """Cap a requested state by what the world can honestly support.

    A world without a metric scale has no metric evidence, so nothing is MEASURED (cap RECONSTRUCTED); an accepted
    typical-size prior (metadata.scale.dimension_state == ESTIMATED) caps everything but a user tape VERIFY at ESTIMATED.
    """
    if requested not in DIMENSION_STATES:
        raise MeasureError(f"state must be one of {', '.join(DIMENSION_STATES)}")
    if requested == "VERIFIED":
        if scale_state == "unknown":
            raise MeasureError("a tape-verified dimension needs a metric scale; this world is in model units")
        return requested
    cap = "RECONSTRUCTED" if scale_state == "unknown" else None
    if scale_dimension_state == "ESTIMATED":
        cap = "ESTIMATED"
    if cap and DIMENSION_STATES.index(requested) > DIMENSION_STATES.index(cap):
        return cap
    return requested


def make_dimension(
    value: float,
    *,
    state: str,
    basis: str,
    scale_state: str,
    scale_dimension_state: str | None = None,
    uncertainty_m: float | None = None,
    kind: str = "distance",
) -> dict:
    """The one dimension shape: {value, unit, state, basis, uncertainty_m}.

    ``unit`` is "m"/"m2" only for a metric/authored world, else "model_units"/"model_units2" (never metres for a
    RELATIVE world). ``state`` is capped by ``effective_state``. ``uncertainty_m`` stays None unless the data supplied one
    (it is in the dimension's own unit; the name is the plan's).
    """
    if not math.isfinite(value):
        raise MeasureError("a dimension value must be finite")
    if uncertainty_m is not None and (not math.isfinite(uncertainty_m) or uncertainty_m < 0):
        raise MeasureError("uncertainty must be a finite number >= 0 (or omitted)")
    return {
        "value": float(value),
        "unit": unit_for(kind, scale_state),
        "state": effective_state(state, scale_state, scale_dimension_state),
        "basis": basis,
        "uncertainty_m": None if uncertainty_m is None else float(uncertainty_m),
    }


def dimension_from_metadata(value: float, world_metadata: Any, *, state: str, basis: str,
                            uncertainty_m: float | None = None, kind: str = "distance") -> dict:
    """make_dimension reading the world's scale state + dimension_state from WorldIR ``metadata``.

    The entry point for room_semantics / room_layout: pass the dimension, the world's ``metadata`` dict, the state the
    geometry supports (inferred/layout-closed => ESTIMATED, derived => RECONSTRUCTED, direct metric evidence => MEASURED).
    """
    scale = (world_metadata or {}).get("scale") if isinstance(world_metadata, dict) else None
    return make_dimension(value, state=state, basis=basis, scale_state=scale_state_of(world_metadata),
                          scale_dimension_state=(scale or {}).get("dimension_state"),
                          uncertainty_m=uncertainty_m, kind=kind)


#: which dimension state a property of an entity with this provenance can honestly claim
_STATE_FOR_PROVENANCE = {
    "OBSERVED": "MEASURED",
    "RECONSTRUCTED": "RECONSTRUCTED",
    "INFERRED": "ESTIMATED",
    "ESTIMATED": "ESTIMATED",
    "GENERATED": "ESTIMATED",
    "CONFLICT": "ESTIMATED",
    "UNKNOWN": "ESTIMATED",
}


def _dimension_basis_and_state(entity: Any, key: str) -> tuple[str, str]:
    """(state, basis) for one ``*_m`` / ``*_m2`` property, from the entity's own provenance and the property's own
    measurement basis. Never upgrades a labelled default to a measurement."""
    cp = getattr(entity, "custom_properties", None) or {}
    prov = getattr(getattr(entity, "provenance", None), "value", None) or str(getattr(entity, "provenance", ""))
    state = _STATE_FOR_PROVENANCE.get(str(prov).upper(), "ESTIMATED")
    if key in ("thickness_m",) and cp.get("thickness_basis") in (None, "default_unmeasured"):
        return "ESTIMATED", "labelled default: no pipeline stage measures wall/ceiling thickness"
    if key in ("floor_dimension_u_m", "floor_dimension_v_m", "height_m", "floor_area_m2") and "footprint_method" in cp:
        return "RECONSTRUCTED", "derived from the measured room footprint"
    if cp.get("wall_slab"):
        return state, ("derived from the solved wall slab" if state == "RECONSTRUCTED"
                       else "inferred wall segment (closes the loop where no point supports it)")
    if "polygon_slab" in (getattr(entity, "semantic_labels", None) or []):
        return "RECONSTRUCTED", "derived from the measured room outline"
    if key.endswith("_m2"):
        return state, "derived from reconstructed floor/ceiling geometry"
    return state, "derived from reconstructed geometry"


def stamp_world_dimensions(world: Any) -> dict:
    """Give every ``*_m`` / ``*_m2`` numeric property on every entity the D1 dimension shape.

    Stores ``custom_properties['dimensions'] = {property: {value, unit, state, basis, uncertainty_m}}`` and leaves the
    flat property in place, so existing readers keep working. Idempotent: re-run it after a scale becomes known to
    upgrade units/states. A user-VERIFIED value already stored for the same value is preserved (F6).

    Returns ``{'dimensions': n, 'with_state': n, 'entities': n}`` for the caller's report.
    """
    scale_meta = (getattr(world, "metadata", None) or {}).get("scale") or {}
    meta = {"scale": scale_meta}
    total = stated = entities = 0
    for entity in (getattr(world, "entities", None) or {}).values():
        cp = getattr(entity, "custom_properties", None)
        if not isinstance(cp, dict):
            continue
        existing = cp.get("dimensions") if isinstance(cp.get("dimensions"), dict) else {}
        dims = dict(existing)
        for key, value in list(cp.items()):
            if key == "uncertainty_m" or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if not key.endswith(("_m", "_m2")):
                continue
            prev = existing.get(key)
            if isinstance(prev, dict) and prev.get("state") == "VERIFIED" and prev.get("value") == float(value):
                dims[key] = prev          # a tape-verified value survives a rebuild (F6)
                continue
            state, basis = _dimension_basis_and_state(entity, key)
            dims[key] = dimension_from_metadata(float(value), meta, state=state, basis=basis,
                                                kind="area" if key.endswith("_m2") else "distance")
        if dims:
            cp["dimensions"] = dims
            total += len(dims)
            stated += sum(1 for d in dims.values() if isinstance(d, dict) and d.get("state"))
            entities += 1
    return {"dimensions": total, "with_state": stated, "entities": entities}
