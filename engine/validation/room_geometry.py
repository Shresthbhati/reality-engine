"""Coherent-room geometry validation on a compiled ``WorldIR`` (sprint workstream 2).

``validate_room_geometry(world, camera_centres=...)`` returns a ``GeometryReport``: typed ``Finding`` records (severity,
element ids, evidence refs, measured values) plus one ``CheckOutcome`` per check. It is READ-ONLY: nothing is repaired,
moved or deleted -- a violation is data for the stage report, never a silent fix.

Principles
  - General, never tuned to one scene. Tolerances are FRACTIONS of the world's characteristic length ``L`` (median
    horizontal long extent of the walls it could read), so a unit-less relative reconstruction and a metric one are
    judged the same way; absolute metre floors apply only when ``metadata.scale.state == "metric"``.
  - Evidence-based, unknown stays unknown. Geometry here is the axis-aligned bounds the WorldIR stores, so a wall's
    orientation is readable only when its box is thin; anything else (tilted wall, no bounds, no up axis, no cameras
    supplied, no footprint polygon) is reported as an UNKNOWN finding and the check is not allowed to say "pass".
    Room closure is reported ``closed`` / ``open`` / ``unknown`` and is never assumed closed.
  - Severity: ``error`` = a decided violation of a hard invariant (degrades the stage), ``warning`` = decided but
    plausibly a capture limitation, ``info`` = could not decide.

Checks (``CHECKS``): duplicate_walls, room_closure, footprint_validity, openings, surface_bounding, attachments,
camera_conflicts, topology. Each "cannot decide" case is listed in its docstring.

Limitations, stated once: bounds-only geometry (no mesh/plane normals), axis-aligned wall reading, furniture is
``EntityType.FURNITURE`` only (wall-mounted fixtures have no separate type), camera frame is assumed to be the
world frame (the caller passes centres it knows to be in it), and ``footprint_polygon_xyz_m`` is a plan-view
occupancy grid with a cell-sized error that the tolerances absorb.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from world_ir import WorldIR
from world_ir.constraints import _host_id
from world_ir.measure import scale_state_of
from world_ir.schema_v1 import EntityType, RelationshipKind
from world_ir.validation import _entity_aabb

ERROR, WARNING, INFO = "error", "warning", "info"
CHECKS = ("duplicate_walls", "room_closure", "footprint_validity", "openings", "surface_bounding",
          "attachments", "camera_conflicts", "topology")
_OPENINGS = (EntityType.DOOR, EntityType.WINDOW, EntityType.OPENING)
_SUPPORT_RELS = {RelationshipKind.RESTS_ON, RelationshipKind.SUPPORTED_BY, RelationshipKind.MOUNTED_ON,
                 RelationshipKind.ATTACHED_TO, RelationshipKind.EMBEDDED_IN}
_ELEMENT_TYPES = (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING, EntityType.DOOR, EntityType.WINDOW,
                  EntityType.OPENING, EntityType.FURNITURE)


@dataclass(frozen=True)
class GeometryConfig:
    """Fractions of the characteristic length L (see module docstring); ``min_m`` floors apply to metric worlds only."""
    thin_wall_ratio: float = 0.06       # short/long horizontal extent below which a wall reads as axis-aligned (~3.4 deg tilt)
    duplicate_offset: float = 0.03      # collinear walls closer than this (perpendicular) are the same wall
    duplicate_overlap: float = 0.5      # ... and overlapping by this fraction of the shorter one
    gap: float = 0.08                   # wall-loop endpoint farther than this from every other wall is an open gap
    opening: float = 0.04               # opening position tolerance along/perpendicular to its host wall
    surface: float = 0.10               # floor/ceiling overhang / separation tolerance (absorbs footprint cell size)
    attach: float = 0.08                # furniture floating / below-floor / above-ceiling tolerance
    penetration: float = 0.05           # furniture depth into a wall beyond which it is penetrating
    camera: float = 0.01                # camera centre within this of a wall solid counts as inside it
    camera_room: float = 0.05           # camera claimed in a room may sit this far outside its footprint
    area_mismatch: float = 0.10         # declared floor area vs polygon area relative difference
    min_m: float = 0.02                 # metric floor on every tolerance


@dataclass(frozen=True)
class Finding:
    code: str
    check: str
    severity: str                       # error | warning | info
    status: str                         # "violation" (decided) | "unknown" (could not decide)
    message: str
    element_ids: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict:
        return {"code": self.code, "check": self.check, "severity": self.severity, "status": self.status,
                "message": self.message, "element_ids": list(self.element_ids), "evidence": self.evidence}


@dataclass
class CheckOutcome:
    check: str
    state: str = "not_applicable"       # pass | fail | partial | unknown | not_applicable
    evaluated: int = 0                  # elements decided (pass or violation)
    violations: int = 0
    undecided: int = 0
    reasons: list[str] = field(default_factory=list)

    def finish(self) -> CheckOutcome:
        if self.violations:
            self.state = "fail"
        elif self.evaluated and not self.undecided:
            self.state = "pass"
        elif self.evaluated:
            self.state = "partial"
        elif self.undecided:
            self.state = "unknown"
        else:
            self.state = "not_applicable"
        return self

    def to_dict(self) -> dict:
        return {"check": self.check, "state": self.state, "evaluated": self.evaluated, "violations": self.violations,
                "undecided": self.undecided, "reasons": self.reasons[:8]}


@dataclass
class GeometryReport:
    findings: list[Finding]
    checks: dict[str, CheckOutcome]
    rooms: dict[str, dict]
    context: dict[str, Any]

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def unknowns(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "unknown"]

    @property
    def verdict(self) -> str:
        if self.errors:
            return "violations"
        if any(c.state in ("partial", "unknown") for c in self.checks.values()) or not any(
                c.state == "pass" for c in self.checks.values()):
            return "no_violations_found_with_unknowns"
        return "no_violations_found"

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "errors": len(self.errors), "warnings": len(self.warnings),
                "unknowns": len(self.unknowns), "checks": {k: v.to_dict() for k, v in self.checks.items()},
                "rooms": self.rooms, "context": self.context, "findings": [f.to_dict() for f in self.findings]}


# ------------------------------------------------------------------------------------------------ geometry helpers
def _seg_dist(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / n))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_seg(a, b, c, tol) -> bool:
    return (min(a[0], b[0]) - tol <= c[0] <= max(a[0], b[0]) + tol
            and min(a[1], b[1]) - tol <= c[1] <= max(a[1], b[1]) + tol)


def _segs_touch_or_cross(p1, p2, p3, p4, eps) -> bool:
    d1, d2, d3, d4 = _orient(p3, p4, p1), _orient(p3, p4, p2), _orient(p1, p2, p3), _orient(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    tol = math.sqrt(eps)
    return ((abs(d1) <= eps and _on_seg(p3, p4, p1, tol)) or (abs(d2) <= eps and _on_seg(p3, p4, p2, tol))
            or (abs(d3) <= eps and _on_seg(p1, p2, p3, tol)) or (abs(d4) <= eps and _on_seg(p1, p2, p4, tol)))


def _shoelace(poly: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))


def _point_in_poly(p, poly) -> bool:
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > p[1]) != (y2 > p[1]) and p[0] < (x2 - x1) * (p[1] - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _dist_to_poly(p, poly) -> float:
    if _point_in_poly(p, poly):
        return 0.0
    return min(_seg_dist(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def _finite3(v) -> tuple[float, float, float] | None:
    try:
        t = tuple(float(x) for x in v)
    except (TypeError, ValueError):
        return None
    return t if len(t) == 3 and all(math.isfinite(x) for x in t) else None


@dataclass
class _Wall:
    id: str
    ent: Any
    box: tuple | None
    reason: str | None = None        # why orientation is unreadable (None = readable)
    axis: int = 0                       # 0: runs along the first horizontal axis, 1: along the second
    c: float = 0.0                      # perpendicular centre coordinate
    t0: float = 0.0
    t1: float = 0.0
    thick: float = 0.0
    v0: float = 0.0
    v1: float = 0.0

    @property
    def readable(self) -> bool:
        return self.reason is None

    def endpoints(self):
        return (((self.t0, self.c), (self.t1, self.c)) if self.axis == 0
                else ((self.c, self.t0), (self.c, self.t1)))

    def body_dist(self, p) -> float:
        a, b = self.endpoints()
        return max(0.0, _seg_dist(p, a, b) - self.thick / 2.0)


class _Ctx:
    def __init__(self, world: WorldIR, cfg: GeometryConfig, cameras: Mapping[str, Any] | None):
        self.world, self.cfg = world, cfg
        self.findings: list[Finding] = []
        self.metric = scale_state_of(world.metadata) == "metric"
        up = self._up_axis()
        self.up_ok = up is not None
        self.v: int = 1 if up is None else up           # vertical axis; meaningless unless up_ok
        self.ha: int = 0 if self.v != 0 else 1          # first / second horizontal axis
        self.hb: int = 2 if self.v != 2 else 1
        self.cameras: dict[str, tuple[float, float, float]] = {}
        for k, c in (cameras or {}).items():
            p = _finite3(c)
            if p is not None:
                self.cameras[str(k)] = p
        self.entities = {k: world.entities[k] for k in sorted(world.entities)}
        self.by_type: dict[EntityType, list[str]] = {}
        for k, e in self.entities.items():
            self.by_type.setdefault(e.type, []).append(k)
        self.walls = {k: self._wall(k) for k in self.by_type.get(EntityType.WALL, [])}
        readable = [w for w in self.walls.values() if w.readable]
        longs = [w.t1 - w.t0 for w in readable]
        self.L: float | None = statistics.median(longs) if longs else None
        if self.L is None:
            self.L = self._footprint_scale()
        self.rooms = {k: self.entities[k] for k in self.by_type.get(EntityType.ROOM, [])}
        self.element_room = self._link_elements()
        self.footprints = {k: self._footprint(self.entities[k]) for k in self.rooms}

    # -- frame
    def _up_axis(self) -> int | None:
        cf = self.world.metadata.get("compiled_from") if isinstance(self.world.metadata, dict) else None
        up = _finite3(cf.get("up")) if isinstance(cf, dict) and cf.get("up") is not None else None
        if up is None:
            return None
        i = max(range(3), key=lambda j: abs(up[j]))
        return i if abs(up[i]) >= 0.9 else None

    def hp(self, p) -> tuple[float, float]:
        return (p[self.ha], p[self.hb])

    def vc(self, eid: str) -> float | None:
        """Vertical centre of an entity's bounds (None without bounds or up axis)."""
        box = _entity_aabb(self.world, self.entities[eid])
        return None if box is None or not self.up_ok else 0.5 * (box[0][self.v] + box[1][self.v])

    def tol(self, frac: float) -> float:
        base = frac * (self.L if self.L else 1.0)
        return max(base, self.cfg.min_m) if self.metric else base

    # -- walls
    def _wall(self, eid: str) -> _Wall:
        e = self.entities[eid]
        w = _Wall(eid, e, _entity_aabb(self.world, e))
        cp = e.custom_properties or {}
        if not self.up_ok:
            w.reason = "up axis not recorded (metadata.compiled_from.up)"
        elif w.box is None:
            w.reason = "no geometry bounds"
        elif cp.get("canted"):
            w.reason = "wall is marked canted; its axis-aligned box does not give its line"
        else:
            lo, hi = w.box
            ea, eb = hi[self.ha] - lo[self.ha], hi[self.hb] - lo[self.hb]
            long_, short = max(ea, eb), min(ea, eb)
            if long_ <= 1e-9:
                w.reason = "degenerate horizontal extent"
            elif short > self.cfg.thin_wall_ratio * long_ and not (
                    isinstance(cp.get("thickness_m"), (int, float)) and short <= 1.25 * float(cp["thickness_m"])):
                w.reason = (f"axis-aligned box is not thin ({short:.3g} x {long_:.3g}): a tilted or compound wall's "
                            "line cannot be read from bounds")
            else:
                w.axis = 0 if ea >= eb else 1
                pa, pb = (self.ha, self.hb) if w.axis == 0 else (self.hb, self.ha)
                w.t0, w.t1 = lo[pa], hi[pa]
                w.c, w.thick = 0.5 * (lo[pb] + hi[pb]), hi[pb] - lo[pb]
                w.v0, w.v1 = lo[self.v], hi[self.v]
        return w

    def _footprint_scale(self) -> float | None:
        areas = []
        for k in self.by_type.get(EntityType.ROOM, []):
            a = (self.entities[k].custom_properties or {}).get("floor_area_m2")
            if isinstance(a, (int, float)) and a > 0:
                areas.append(float(a))
        return math.sqrt(max(areas)) if areas else None

    # -- room association
    def _link_elements(self) -> dict[str, set[str]]:
        m: dict[str, set[str]] = {}
        for rid in self.by_type.get(EntityType.ROOM, []):
            r = self.entities[rid]
            cp = r.custom_properties or {}
            ids = {x.target_id for x in r.relationships if x.kind is RelationshipKind.CONTAINS}
            for key in ("boundary_element_ids", "wall_slab_ids"):
                ids |= {i for i in (cp.get(key) or ()) if isinstance(i, str)}
            for i in ids:
                if i in self.entities:
                    m.setdefault(i, set()).add(rid)
        for k, e in self.entities.items():
            cp = e.custom_properties or {}
            for key in ("room_id", "space_id"):
                if cp.get(key) in self.rooms:
                    m.setdefault(k, set()).add(cp[key])
            for x in e.relationships:
                if x.target_id in self.rooms and x.kind in (RelationshipKind.PART_OF, RelationshipKind.INSIDE):
                    m.setdefault(k, set()).add(x.target_id)
        for k in self.by_type.get(EntityType.WALL, []):   # slab walls inherit the room of the evidence plane they sit on
            for x in self.entities[k].relationships:
                if x.kind is RelationshipKind.ATTACHED_TO and x.target_id in m and k not in m:
                    m[k] = set(m[x.target_id])
        for k in [i for t in _OPENINGS for i in self.by_type.get(t, [])]:
            h = _host_id(self.world, self.entities[k])
            if h in m:
                m.setdefault(k, set()).update(m[h])
        return m

    def _footprint(self, room) -> list[tuple[float, float]] | None:
        cp = room.custom_properties or {}
        xyz = cp.get("footprint_polygon_xyz_m")
        if self.up_ok and isinstance(xyz, list) and len(xyz) >= 3:
            pts = [_finite3(p) for p in xyz]
            if all(p is not None for p in pts):
                return [self.hp(p) for p in pts if p is not None]
        return None

    # -- findings
    def refs(self, ids: Iterable[str]) -> dict[str, Any]:
        geoms: list[str] = []
        evid: set[str] = set()
        for i in ids:
            e = self.entities.get(i)
            if e is None:
                continue
            geoms += list(e.geometry_ids)
            evid |= {s for s in ((e.custom_properties or {}).get("supporting_evidence_ids") or ()) if isinstance(s, str)}
        return {"entity_ids": list(ids), "geometry_ids": sorted(set(geoms)), "supporting_evidence_ids": sorted(evid)}

    def add(self, out: CheckOutcome, code: str, severity: str, message: str, ids: Sequence[str], **ev) -> None:
        ids = tuple(ids)
        self.findings.append(Finding(code, out.check, severity, "violation", message, ids,
                                     {**self.refs(ids), **{k: _jsonable(v) for k, v in ev.items()}}))
        out.violations += 1

    def unknown(self, out: CheckOutcome, code: str, message: str, ids: Sequence[str], **ev) -> None:
        ids = tuple(ids)
        self.findings.append(Finding(code, out.check, INFO, "unknown", message, ids,
                                     {**self.refs(ids), **{k: _jsonable(v) for k, v in ev.items()}}))
        out.undecided += 1
        if message not in out.reasons:
            out.reasons.append(message)


def _jsonable(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (list, tuple)):
        return [_jsonable(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(i) for k, i in v.items()}
    return v


# ------------------------------------------------------------------------------------------------ checks
def _linked(ctx: _Ctx, a: _Wall, b: _Wall) -> bool:
    """A layout slab and the evidence plane it was built from are one physical wall, not a duplicate."""
    for x, y in ((a, b), (b, a)):
        if any(r.target_id == y.id and r.kind is RelationshipKind.ATTACHED_TO for r in x.ent.relationships):
            return True
        cp = x.ent.custom_properties or {}
        if y.id in (cp.get("evidence_plane_ids") or ()) or cp.get("slab_wall_id") == y.id:
            return True
    return False


def check_duplicate_walls(ctx: _Ctx) -> CheckOutcome:
    """Near-coincident, collinear, overlapping walls. Cannot decide: walls whose orientation is unreadable."""
    out = CheckOutcome("duplicate_walls")
    ws = list(ctx.walls.values())
    if len(ws) < 2:
        return out.finish()
    for w in ws:
        if not w.readable:
            ctx.unknown(out, "wall_orientation_unreadable", f"wall {w.id}: {w.reason}", [w.id])
    rd = [w for w in ws if w.readable]
    off = ctx.tol(ctx.cfg.duplicate_offset)
    for i, a in enumerate(rd):
        for b in rd[i + 1:]:
            out.evaluated += 1
            if a.axis != b.axis or abs(a.c - b.c) > off or _linked(ctx, a, b):
                continue
            ov = min(a.t1, b.t1) - max(a.t0, b.t0)
            short = min(a.t1 - a.t0, b.t1 - b.t0)
            vo = min(a.v1, b.v1) - max(a.v0, b.v0)
            if short <= 0 or ov < ctx.cfg.duplicate_overlap * short or vo <= 0.5 * min(a.v1 - a.v0, b.v1 - b.v0):
                continue
            ra, rb = ctx.element_room.get(a.id, set()), ctx.element_room.get(b.id, set())
            shared_partition = bool(ra and rb and not (ra & rb))   # two rooms each stored their side of one partition
            ctx.add(out, "duplicate_wall_shared_partition" if shared_partition else "duplicate_wall",
                    WARNING if shared_partition else ERROR,
                    f"walls {a.id} and {b.id} are collinear (offset {abs(a.c - b.c):.3g}, overlap {ov:.3g} of "
                    f"{short:.3g})", [a.id, b.id], offset=abs(a.c - b.c), overlap=ov, tolerance=off)
    return out.finish()


def _room_walls(ctx: _Ctx, rid: str) -> list[_Wall]:
    return [ctx.walls[k] for k in sorted(ctx.walls) if rid in ctx.element_room.get(k, set())]


def check_room_closure(ctx: _Ctx, report_rooms: dict[str, dict]) -> CheckOutcome:
    """Gaps in each room's wall loop -> closed / open / unknown. Cannot decide: no linked walls, or any linked wall
    with unreadable orientation (then ``unknown``, never ``closed``)."""
    out = CheckOutcome("room_closure")
    gap = ctx.tol(ctx.cfg.gap)
    for rid in sorted(ctx.rooms):
        ws = _room_walls(ctx, rid)
        info = report_rooms.setdefault(rid, {})
        if not ws:
            info["closure"] = {"state": "unknown", "reason": "no wall entity is linked to this room"}
            ctx.unknown(out, "closure_unknown", f"room {rid}: no wall is linked to it, closure cannot be assessed", [rid])
            continue
        bad = [w for w in ws if not w.readable]
        rd = [w for w in ws if w.readable]
        gaps: list[dict[str, Any]] = []
        for w in rd:
            for end in w.endpoints():
                others = [o for o in rd if o.id != w.id]
                d = min((o.body_dist(end) for o in others), default=math.inf)
                if d > gap:
                    declared = bool((w.ent.custom_properties or {}).get("unobserved_spans_m"))
                    near = min((math.hypot(end[0] - e2[0], end[1] - e2[1]) for o in others for e2 in o.endpoints()),
                               default=math.inf)
                    gaps.append({"wall": w.id, "endpoint": [round(end[0], 4), round(end[1], 4)],
                                 "distance_to_nearest_wall": round(d, 4) if math.isfinite(d) else None,
                                 "distance_to_nearest_endpoint": round(near, 4) if math.isfinite(near) else None,
                                 "declared_unobserved": declared})
        if bad:
            info["closure"] = {"state": "unknown", "reason": f"{len(bad)} of {len(ws)} walls have unreadable orientation",
                               "unreadable": [w.id for w in bad], "gaps_among_readable": gaps}
            ctx.unknown(out, "closure_unknown", f"room {rid}: walls {[w.id for w in bad]} cannot be read as line "
                        "segments, so the loop cannot be closed or opened", [rid, *[w.id for w in bad]])
            continue
        out.evaluated += 1
        if len(rd) < 3:
            info["closure"] = {"state": "open", "reason": f"{len(rd)} wall segment(s) cannot enclose a room", "gaps": gaps}
            ctx.add(out, "room_boundary_open", ERROR, f"room {rid} has only {len(rd)} readable wall(s); a loop needs 3+",
                    [rid, *[w.id for w in rd]], wall_count=len(rd))
        elif gaps:
            undeclared = [g for g in gaps if not g["declared_unobserved"]]
            info["closure"] = {"state": "open", "gaps": gaps, "tolerance": gap}
            ctx.add(out, "room_boundary_open", ERROR if undeclared else WARNING,
                    f"room {rid} wall loop has {len(gaps)} open endpoint(s) (> {gap:.3g} from any other wall)"
                    + ("" if undeclared else "; all on walls that declare unobserved spans"),
                    [rid, *sorted({g['wall'] for g in gaps})], gaps=gaps, tolerance=gap)
        else:
            info["closure"] = {"state": "closed", "wall_count": len(rd), "tolerance": gap}
    return out.finish()


def check_footprint_validity(ctx: _Ctx, report_rooms: dict[str, dict]) -> CheckOutcome:
    """Zero-area / degenerate / self-intersecting room footprint. Cannot decide: no footprint polygon."""
    out = CheckOutcome("footprint_validity")
    for rid in sorted(ctx.rooms):
        cp = ctx.rooms[rid].custom_properties or {}
        poly = ctx.footprints[rid]
        if poly is None:
            b = cp.get("boundary_polygon_2d")
            if isinstance(b, list) and len(b) >= 3 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                poly = [(float(p[0]), float(p[1])) for p in b]    # floor-plane 2D frame: an isometry, valid for shape tests
        if poly is None:
            ctx.unknown(out, "footprint_unknown", f"room {rid} has no footprint polygon", [rid])
            continue
        out.evaluated += 1
        scale = max(max(abs(c) for p in poly for c in p), 1e-9)
        clean = [p for i, p in enumerate(poly) if math.hypot(p[0] - poly[i - 1][0], p[1] - poly[i - 1][1]) > 1e-9 * scale]
        area = abs(_shoelace(clean)) if len(clean) >= 3 else 0.0
        ref = ctx.L * ctx.L if ctx.L else scale * scale
        report_rooms.setdefault(rid, {})["footprint"] = {"vertices": len(clean), "area": round(area, 6)}
        if len(clean) < 3:
            ctx.add(out, "footprint_degenerate", ERROR, f"room {rid} footprint has {len(clean)} distinct vertices", [rid])
            continue
        eps = 1e-9 * scale * scale
        n = len(clean)
        crossing = [(i, j) for i in range(n) for j in range(i + 2, n)
                    if not (i == 0 and j == n - 1)
                    and _segs_touch_or_cross(clean[i], clean[(i + 1) % n], clean[j], clean[(j + 1) % n], eps)]
        if crossing:
            ctx.add(out, "footprint_self_intersecting", ERROR,
                    f"room {rid} footprint edges {crossing[:3]} cross or touch", [rid], edge_pairs=crossing[:5])
            continue
        if area <= 1e-3 * ref:
            ctx.add(out, "footprint_zero_area", ERROR, f"room {rid} footprint area {area:.3g} is ~0", [rid], area=area)
            continue
        declared = cp.get("floor_area_m2")
        if isinstance(declared, (int, float)) and declared > 0 and abs(declared - area) / declared > ctx.cfg.area_mismatch:
            ctx.add(out, "footprint_area_mismatch", WARNING,
                    f"room {rid} declares floor area {declared:.4g} but its footprint polygon encloses {area:.4g}",
                    [rid], declared=float(declared), polygon=area)
    return out.finish()


def _opening_box(ctx: _Ctx, e):
    """Bounds from geometry, else the ``bounds_min``/``bounds_max`` the opening promoters also record in properties."""
    box = _entity_aabb(ctx.world, e)
    if box is not None:
        return box
    cp = e.custom_properties or {}
    lo, hi = _finite3(cp.get("bounds_min") or ()), _finite3(cp.get("bounds_max") or ())
    return None if lo is None or hi is None else (lo, hi)


def _opening_span(ctx: _Ctx, box, host: _Wall):
    pa = ctx.ha if host.axis == 0 else ctx.hb
    pb = ctx.hb if host.axis == 0 else ctx.ha
    return box[0][pa], box[1][pa], 0.5 * (box[0][pb] + box[1][pb])


def check_openings(ctx: _Ctx) -> CheckOutcome:
    """Door/window hosting: no host, off the wall plane, outside the host's extent, wider than the host, outside its
    height, overlapping a sibling. Cannot decide: host orientation unreadable, or opening without bounds."""
    out = CheckOutcome("openings")
    per_host: dict[str, list[tuple[str, float, float, float, float]]] = {}
    tol = ctx.tol(ctx.cfg.opening)
    for t in _OPENINGS:
        for oid in ctx.by_type.get(t, []):
            e = ctx.entities[oid]
            hid = _host_id(ctx.world, e)
            host = ctx.walls.get(hid) if hid else None
            if host is None:
                out.evaluated += 1
                ctx.add(out, "opening_not_hosted", ERROR,
                        f"{t.value} {oid} has no host wall (host={hid!r})", [oid], host=hid)
                continue
            box = _opening_box(ctx, e)
            if not host.readable or box is None:
                ctx.unknown(out, "opening_undecidable",
                            f"{t.value} {oid}: " + (f"host {host.id} unreadable ({host.reason})" if not host.readable
                                                     else "no bounds"), [oid, host.id])
                continue
            out.evaluated += 1
            o0, o1, perp = _opening_span(ctx, box, host)
            width = o1 - o0
            bad = False
            if width > (host.t1 - host.t0) + tol:
                bad = True
                ctx.add(out, "opening_host_too_short", ERROR,
                        f"{t.value} {oid} is {width:.3g} wide but host wall {host.id} is {host.t1 - host.t0:.3g} long",
                        [oid, host.id], opening_width=width, host_length=host.t1 - host.t0)
            elif o0 < host.t0 - tol or o1 > host.t1 + tol:
                bad = True
                ctx.add(out, "opening_outside_host_extent", ERROR,
                        f"{t.value} {oid} spans {o0:.3g}..{o1:.3g} but host {host.id} covers {host.t0:.3g}..{host.t1:.3g}",
                        [oid, host.id], span=[o0, o1], host_span=[host.t0, host.t1], tolerance=tol)
            if abs(perp - host.c) > host.thick / 2.0 + tol:
                bad = True
                ctx.add(out, "opening_off_wall_plane", ERROR,
                        f"{t.value} {oid} sits {abs(perp - host.c):.3g} from host {host.id}'s plane "
                        f"(thickness {host.thick:.3g})", [oid, host.id], offset=abs(perp - host.c), tolerance=tol)
            if box[0][ctx.v] < host.v0 - tol or box[1][ctx.v] > host.v1 + tol:
                # evidence walls routinely lose their edge rows, so this is a warning: decided but often a capture limit
                ctx.add(out, "opening_outside_wall_height", WARNING,
                        f"{t.value} {oid} vertical {box[0][ctx.v]:.3g}..{box[1][ctx.v]:.3g} exceeds host "
                        f"{host.id} {host.v0:.3g}..{host.v1:.3g}", [oid, host.id])
            if not bad:
                per_host.setdefault(host.id, []).append((oid, o0, o1, box[0][ctx.v], box[1][ctx.v]))
    for hid, lst in sorted(per_host.items()):
        for i, (a, a0, a1, av0, av1) in enumerate(lst):
            for b, b0, b1, bv0, bv1 in lst[i + 1:]:
                if min(a1, b1) - max(a0, b0) > tol and min(av1, bv1) - max(av0, bv0) > tol:
                    ctx.add(out, "openings_overlap", ERROR, f"openings {a} and {b} overlap on wall {hid}", [a, b, hid],
                            overlap=min(a1, b1) - max(a0, b0))
    return out.finish()


def _horiz_rect(ctx: _Ctx, box):
    return (box[0][ctx.ha], box[0][ctx.hb], box[1][ctx.ha], box[1][ctx.hb])


def check_surface_bounding(ctx: _Ctx) -> CheckOutcome:
    """Floor/ceiling slabs not bounded by their room's walls (overhang, floating away from every wall) and ceiling not
    above floor. Cannot decide: slab without a room, walls unreadable/fewer than 3, no up axis."""
    out = CheckOutcome("surface_bounding")
    tol = ctx.tol(ctx.cfg.surface)
    for t in (EntityType.FLOOR, EntityType.CEILING):
        for sid in ctx.by_type.get(t, []):
            e = ctx.entities[sid]
            box = _entity_aabb(ctx.world, e)
            rooms = sorted(ctx.element_room.get(sid, set()))
            if not ctx.up_ok or box is None:
                ctx.unknown(out, "surface_undecidable", f"{t.value} {sid}: no bounds or up axis", [sid])
                continue
            if not rooms:
                ctx.unknown(out, "surface_no_room", f"{t.value} {sid} is not linked to a room, so its bounding walls are unknown", [sid])
                continue
            walls = [w for r in rooms for w in _room_walls(ctx, r)]
            rd = [w for w in walls if w.readable]
            if len(rd) < 3 or len(rd) != len(walls):
                ctx.unknown(out, "surface_walls_unreadable",
                            f"{t.value} {sid}: room {rooms[0]} has {len(rd)} readable of {len(walls)} walls", [sid, *rooms])
                continue
            out.evaluated += 1
            x0, y0, x1, y1 = _horiz_rect(ctx, box)
            # envelope of the wall footprint rectangles in (first, second) horizontal coordinates
            rects = [((w.t0, w.c - w.thick / 2, w.t1, w.c + w.thick / 2) if w.axis == 0
                      else (w.c - w.thick / 2, w.t0, w.c + w.thick / 2, w.t1)) for w in rd]
            e0, f0 = min(r[0] for r in rects), min(r[1] for r in rects)
            e1, f1 = max(r[2] for r in rects), max(r[3] for r in rects)
            over = max(e0 - x0, f0 - y0, x1 - e1, y1 - f1)
            if over > tol:
                ctx.add(out, "surface_exceeds_walls", ERROR,
                        f"{t.value} {sid} extends {over:.3g} beyond its room's walls (tolerance {tol:.3g})",
                        [sid, *rooms], overhang=over, tolerance=tol)
                continue
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            nearest = min(w.body_dist((cx, cy)) for w in rd)
            reach = 0.5 * math.hypot(x1 - x0, y1 - y0) + tol
            if nearest > reach:
                ctx.add(out, "surface_not_bounded_by_walls", ERROR,
                        f"{t.value} {sid} is {nearest:.3g} from the nearest wall (half-diagonal {reach:.3g})",
                        [sid, *rooms], nearest_wall=nearest)
    floors = {r: [ctx.vc(i) for i in ctx.by_type.get(EntityType.FLOOR, []) if r in ctx.element_room.get(i, set())]
              for r in ctx.rooms} if ctx.up_ok else {}
    for cid in ctx.by_type.get(EntityType.CEILING, []):
        cbox = _entity_aabb(ctx.world, ctx.entities[cid])
        for r in sorted(ctx.element_room.get(cid, set())):
            fl = [f for f in floors.get(r, []) if f is not None]
            if cbox is None or not fl:
                continue
            if 0.5 * (cbox[0][ctx.v] + cbox[1][ctx.v]) <= max(fl):
                ctx.add(out, "ceiling_not_above_floor", ERROR, f"ceiling {cid} is not above the floor of room {r}", [cid, r])
    return out.finish()


def _floor_level(ctx: _Ctx, rid: str) -> tuple[float, float] | None:
    cp = ctx.rooms[rid].custom_properties or {}
    fh = cp.get("floor_height_m")
    h = max((float(x) for x in (cp.get("height_m"), cp.get("ceiling_height_max_m")) if isinstance(x, (int, float))), default=None)
    if not isinstance(fh, (int, float)) or h is None:
        return None
    return float(fh), float(fh) + h


def check_attachments(ctx: _Ctx) -> CheckOutcome:
    """Furniture floating above the floor, below it, above the ceiling, or crossing a wall. Cannot decide: no bounds,
    no up axis, no room floor level (floor/ceiling tests), unreadable walls (penetration test)."""
    out = CheckOutcome("attachments")
    fl = ctx.tol(ctx.cfg.attach)
    pen = ctx.tol(ctx.cfg.penetration)
    furn = ctx.by_type.get(EntityType.FURNITURE, [])
    boxes = {i: _entity_aabb(ctx.world, ctx.entities[i]) for i in furn}
    no_level: list[str] = []
    wall_test_partial: list[str] = []
    unread_ids = [w.id for w in ctx.walls.values() if not w.readable]
    for fid in furn:
        e, box = ctx.entities[fid], boxes[fid]
        if not ctx.up_ok or box is None:
            ctx.unknown(out, "attachment_undecidable", f"furniture {fid}: no bounds or up axis", [fid])
            continue
        rooms = sorted(ctx.element_room.get(fid, set()))
        if not rooms:
            c = ctx.hp(tuple(0.5 * (box[0][i] + box[1][i]) for i in range(3)))
            inside = [r for r in sorted(ctx.rooms) if ctx.footprints.get(r) and _dist_to_poly(c, ctx.footprints[r]) == 0.0
                      and _floor_level(ctx, r) is not None]
            # stacked storeys: the room whose floor is nearest the piece's base
            rooms = sorted(inside, key=lambda r: abs(box[0][ctx.v] - (_floor_level(ctx, r) or (0.0, 0.0))[0]))
        decided = False
        levels = [_floor_level(ctx, r) for r in rooms[:1]]
        if levels and levels[0] is not None:
            f0, c0 = levels[0]
            decided = True
            supported = any(x.kind in _SUPPORT_RELS for x in e.relationships) or any(
                b is not None and j != fid and abs(b[1][ctx.v] - box[0][ctx.v]) <= fl
                and min(box[1][ctx.ha], b[1][ctx.ha]) > max(box[0][ctx.ha], b[0][ctx.ha])
                and min(box[1][ctx.hb], b[1][ctx.hb]) > max(box[0][ctx.hb], b[0][ctx.hb]) for j, b in boxes.items())
            if box[0][ctx.v] < f0 - fl:
                ctx.add(out, "furniture_below_floor", ERROR,
                        f"furniture {fid} bottom {box[0][ctx.v]:.3g} is below room {rooms[0]} floor {f0:.3g}",
                        [fid, rooms[0]], bottom=box[0][ctx.v], floor=f0, tolerance=fl)
            elif box[0][ctx.v] > f0 + fl and not supported:
                ctx.add(out, "furniture_floating", ERROR,
                        f"furniture {fid} bottom is {box[0][ctx.v] - f0:.3g} above floor with no support relation",
                        [fid, rooms[0]], height_above_floor=box[0][ctx.v] - f0, tolerance=fl)
            if box[1][ctx.v] > c0 + fl:
                ctx.add(out, "furniture_above_ceiling", ERROR,
                        f"furniture {fid} top {box[1][ctx.v]:.3g} is above room {rooms[0]} ceiling {c0:.3g}",
                        [fid, rooms[0]], top=box[1][ctx.v], ceiling=c0)
        else:
            no_level.append(fid)
        pierced = False
        for w in ctx.walls.values():
            if not w.readable:
                continue
            lo_t, hi_t = (box[0][ctx.ha], box[1][ctx.ha]) if w.axis == 0 else (box[0][ctx.hb], box[1][ctx.hb])
            lo_p, hi_p = (box[0][ctx.hb], box[1][ctx.hb]) if w.axis == 0 else (box[0][ctx.ha], box[1][ctx.ha])
            if min(hi_t, w.t1) - max(lo_t, w.t0) <= pen or min(box[1][ctx.v], w.v1) - max(box[0][ctx.v], w.v0) <= pen:
                continue
            # intrusion measured from the wall face the box approaches (the face on the side of the box's centre):
            # 0 when it rests against the face, thickness when it fills the slab, more when it passes through
            half = w.thick / 2
            depth = (hi_p - (w.c - half)) if 0.5 * (lo_p + hi_p) < w.c else ((w.c + half) - lo_p)
            depth = max(0.0, depth)
            if depth > pen:
                pierced = True
                centre_in_wall = abs(0.5 * (lo_p + hi_p) - w.c) <= half + pen
                # an object's box is only an upper bound of its extent (a yawed object's axis-aligned box overstates it),
                # so a decided ERROR needs the object's CENTRE inside the wall; a box that merely intrudes is a WARNING
                ctx.add(out, "furniture_penetrates_wall" if centre_in_wall else "furniture_bounds_intrude_wall",
                        ERROR if centre_in_wall else WARNING,
                        f"furniture {fid} " + ("has its centre inside" if centre_in_wall else "bounds intrude")
                        + f" {depth:.3g} into wall {w.id} (tolerance {pen:.3g})",
                        [fid, w.id], depth=depth, tolerance=pen, centre_in_wall=centre_in_wall)
        if unread_ids:
            wall_test_partial.append(fid)
        if decided or pierced or any(w.readable for w in ctx.walls.values()):
            out.evaluated += 1
    if no_level:
        ctx.unknown(out, "attachment_no_floor_level",
                    f"{len(no_level)} furniture piece(s) have no room with a known floor/ceiling level, so floating/"
                    "below-floor/above-ceiling is not decidable", no_level)
    if wall_test_partial:
        ctx.unknown(out, "attachment_wall_test_partial",
                    f"{len(unread_ids)} wall(s) with unreadable orientation were not tested for penetration "
                    f"against {len(wall_test_partial)} furniture piece(s)", [*wall_test_partial, *unread_ids[:5]])
    return out.finish()


def check_camera_conflicts(ctx: _Ctx) -> CheckOutcome:
    """Registered camera centre inside a wall solid, or outside the room it is claimed to be in. Cannot decide: no
    camera centres supplied; walls unreadable (inside-wall test); a claimed room without footprint (outside test)."""
    out = CheckOutcome("camera_conflicts")
    if not ctx.cameras:
        if ctx.walls or ctx.rooms:
            ctx.unknown(out, "cameras_not_supplied", "no registered camera centres were supplied to the validator", [])
        return out.finish()
    if not ctx.up_ok:
        ctx.unknown(out, "camera_undecidable", "up axis not recorded; camera/surface frame cannot be resolved", [])
        return out.finish()
    tol = ctx.tol(ctx.cfg.camera)
    unread = [w.id for w in ctx.walls.values() if not w.readable]
    for cid in sorted(ctx.cameras):
        p = ctx.cameras[cid]
        h = ctx.hp(p)
        out.evaluated += 1
        for w in ctx.walls.values():
            if not w.readable:
                continue
            tpos, ppos = (h[0], h[1]) if w.axis == 0 else (h[1], h[0])
            if (w.t0 - tol <= tpos <= w.t1 + tol and abs(ppos - w.c) <= w.thick / 2 + tol
                    and w.v0 - tol <= p[ctx.v] <= w.v1 + tol):
                ctx.add(out, "camera_inside_wall", ERROR, f"camera {cid} centre lies inside wall {w.id}", [w.id],
                        camera=cid, centre=list(p), offset_from_wall_plane=abs(ppos - w.c), wall_thickness=w.thick)
    if unread:
        ctx.unknown(out, "camera_wall_test_partial",
                    f"{len(unread)} wall(s) have unreadable orientation, so the inside-wall test is incomplete for "
                    f"{len(ctx.cameras)} camera(s)", unread[:5])
    for rid in sorted(ctx.rooms):
        claimed = [s for s in ((ctx.rooms[rid].custom_properties or {}).get("camera_stations") or ()) if isinstance(s, str)]
        poly, lvl = ctx.footprints.get(rid), _floor_level(ctx, rid)
        for cid in sorted(set(claimed)):
            if cid not in ctx.cameras:
                continue
            p = ctx.cameras[cid]
            if poly is None:
                ctx.unknown(out, "camera_room_unknown", f"camera {cid} is claimed in room {rid} which has no footprint", [rid])
                continue
            d = _dist_to_poly(ctx.hp(p), poly)
            if d > ctx.tol(ctx.cfg.camera_room):
                ctx.add(out, "camera_outside_claimed_room", ERROR,
                        f"camera {cid} is {d:.3g} outside room {rid}'s footprint although it is listed as a station in it",
                        [rid], camera=cid, distance_outside=d)
            elif lvl is not None and not (lvl[0] - ctx.tol(ctx.cfg.camera_room) <= p[ctx.v] <= lvl[1] + ctx.tol(ctx.cfg.camera_room)):
                ctx.add(out, "camera_outside_room_height", ERROR,
                        f"camera {cid} height {p[ctx.v]:.3g} is outside room {rid} {lvl[0]:.3g}..{lvl[1]:.3g}",
                        [rid], camera=cid)
    return out.finish()


def check_topology(ctx: _Ctx) -> CheckOutcome:
    """Orphan elements and rooms unreachable through doors/CONNECTS within a storey. Cannot decide: orphans with no rooms
    in the world; connectivity with no door/opening/CONNECTS evidence at all."""
    out = CheckOutcome("topology")
    if not ctx.rooms:
        if any(ctx.by_type.get(t) for t in _ELEMENT_TYPES):
            ctx.unknown(out, "orphans_undecidable", "the world has no room entity, so no element can be called an orphan", [])
        return out.finish()
    for t in _ELEMENT_TYPES:
        for eid in ctx.by_type.get(t, []):
            out.evaluated += 1
            if ctx.element_room.get(eid):
                continue
            if t is EntityType.FURNITURE:
                box = _entity_aabb(ctx.world, ctx.entities[eid])
                if box is None or not ctx.up_ok or not any(ctx.footprints.values()):
                    ctx.unknown(out, "orphan_undecidable", f"furniture {eid} has no room link and no footprint to test against", [eid])
                    continue
                c = ctx.hp(tuple(0.5 * (box[0][i] + box[1][i]) for i in range(3)))
                if any(p and _dist_to_poly(c, p) <= ctx.tol(ctx.cfg.camera_room) for p in ctx.footprints.values()):
                    continue
            ctx.add(out, "orphan_element", WARNING, f"{t.value} {eid} belongs to no room", [eid])
    storey_of: dict[str, str] = {}
    for sid in ctx.by_type.get(EntityType.STOREY, []):
        for x in ctx.entities[sid].relationships:
            if x.kind is RelationshipKind.CONTAINS and x.target_id in ctx.rooms:
                storey_of[x.target_id] = sid
    parent = {r: r for r in ctx.rooms}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    evidence = False
    for k, e in ctx.entities.items():
        for x in e.relationships:
            if x.kind is RelationshipKind.CONNECTS and k in ctx.rooms and x.target_id in ctx.rooms:
                evidence = True
                parent[find(k)] = find(x.target_id)
        if e.type in _OPENINGS:
            evidence = True
            rs = sorted(ctx.element_room.get(k, set()) | {x.target_id for x in e.relationships
                                                          if x.kind is RelationshipKind.CONNECTS and x.target_id in ctx.rooms})
            for r in rs[1:]:
                parent[find(rs[0])] = find(r)
    groups: dict[str | None, list[str]] = {}
    for r in sorted(ctx.rooms):
        groups.setdefault(storey_of.get(r), []).append(r)
    if not any(len(g) >= 2 for g in groups.values()):
        return out.finish()
    if not evidence:
        ctx.unknown(out, "connectivity_unknown", "no door/opening/CONNECTS evidence exists, room connectivity cannot be assessed", sorted(ctx.rooms))
        return out.finish()
    for gsid, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(g) < 2:
            continue
        comps: dict[str, list[str]] = {}
        for r in g:
            comps.setdefault(find(r), []).append(r)
        out.evaluated += 1
        if len(comps) > 1:
            ctx.add(out, "rooms_not_connected", WARNING,
                    f"rooms on storey {gsid} form {len(comps)} groups with no door/opening between them: "
                    f"{sorted(map(sorted, comps.values()))}", g, groups=sorted(map(sorted, comps.values())))
    return out.finish()


# ------------------------------------------------------------------------------------------------ entry points
def validate_room_geometry(world: WorldIR, *, camera_centres: Mapping[str, Any] | None = None,
                           config: GeometryConfig | None = None) -> GeometryReport:
    """Run every check on ``world`` and return the report. ``camera_centres`` maps evidence id -> (x, y, z) in the WORLD
    frame; omit it and the camera check reports unknown."""
    ctx = _Ctx(world, config or GeometryConfig(), camera_centres)
    rooms: dict[str, dict] = {}
    outcomes = {
        "duplicate_walls": check_duplicate_walls(ctx),
        "room_closure": check_room_closure(ctx, rooms),
        "footprint_validity": check_footprint_validity(ctx, rooms),
        "openings": check_openings(ctx),
        "surface_bounding": check_surface_bounding(ctx),
        "attachments": check_attachments(ctx),
        "camera_conflicts": check_camera_conflicts(ctx),
        "topology": check_topology(ctx),
    }
    context = {
        "up_axis": None if not ctx.up_ok else "xyz"[ctx.v], "scale_state": "metric" if ctx.metric else "unknown_or_relative",
        "characteristic_length": None if ctx.L is None else round(ctx.L, 6),
        "walls": len(ctx.walls), "walls_readable": sum(w.readable for w in ctx.walls.values()),
        "rooms": len(ctx.rooms), "cameras": len(ctx.cameras),
        "tolerances_are": "fractions of characteristic_length" + (" with metric floors" if ctx.metric else ""),
        "config": {k: getattr(ctx.cfg, k) for k in ctx.cfg.__dataclass_fields__},
        "frame_note": "walls are read from axis-aligned bounds; camera centres are assumed to be in the world frame",
    }
    return GeometryReport(findings=ctx.findings, checks=outcomes, rooms=rooms, context=context)


def report_to_segment_fields(report: GeometryReport) -> tuple[list[str], dict[str, float]]:
    """Stage fields (same contract as ``world_ir.constraints.violations_to_segment_fields``): decided ERROR findings become
    stage warnings (=> ``degraded``); counts of every severity/unknown go to the stage metrics."""
    warnings = [f"geometry {f.code}: {f.message}" for f in report.errors]
    metrics = {"geometry_errors": float(len(report.errors)), "geometry_warnings": float(len(report.warnings)),
               "geometry_unknown": float(len(report.unknowns)),
               "geometry_checks_failed": float(sum(c.state == "fail" for c in report.checks.values())),
               "geometry_checks_passed": float(sum(c.state == "pass" for c in report.checks.values()))}
    return warnings, metrics


def camera_centres_from_poses(camera_poses: Iterable[Any]) -> dict[str, tuple[float, float, float]]:
    """``(evidence_id, position, rotation)`` tuples (``prog.camera_poses``) or objects with those attributes."""
    out: dict[str, tuple[float, float, float]] = {}
    for p in camera_poses or ():
        try:
            eid, pos = (p[0], p[1]) if isinstance(p, (tuple, list)) else (p.evidence_id, p.position)
        except (AttributeError, IndexError, TypeError):
            continue
        c = _finite3(pos)
        if c is not None:
            out[str(eid)] = c
    return out


def validate_for_stage(world: WorldIR, camera_poses: Iterable[Any] = ()) -> tuple[list[str], dict[str, float]]:
    """The ``compile_validate`` hook: validate, record the full report on ``world.metadata['room_geometry_validation']``
    (so it is persisted with the world and visible in the stage report), return ``(stage warnings, stage metrics)``.
    A validator crash is recorded and degrades the stage; it never fails the job and is never silent."""
    try:
        report = validate_room_geometry(world, camera_centres=camera_centres_from_poses(camera_poses))
    except Exception as exc:  # noqa: BLE001 -- a validator bug must be visible, not fatal
        msg = f"{type(exc).__name__}: {exc}"
        world.metadata["room_geometry_validation"] = {"verdict": "validator_error", "error": msg}
        return [f"geometry validator failed: {msg}"], {"geometry_validator_error": 1.0}
    world.metadata["room_geometry_validation"] = report.to_dict()
    return report_to_segment_fields(report)
