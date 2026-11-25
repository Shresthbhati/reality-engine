"""Planar wall graph, room (enclosure) extraction, and topology validation that separates ERRORS from INCOMPLETENESS.

Severity classes -- the distinction this module exists to make:
  STRUCTURAL_ERROR  the observed evidence is impossible as stated (duplicate entity, walls that cross with no
                    junction, a room boundary that self-intersects, an opening beyond a fully-observed host,
                    a broken reference, a stair joining one storey to itself, a non-finite coordinate).
  CONTRADICTION     two pieces of evidence disagree; both are kept and surfaced (never averaged away).
  INCOMPLETE        the world is simply not fully observed: an occluded wall end, an opening with no known
                    host, a ring that does not close because an end was never seen, a single-view entity.
  INFO              legitimate and unremarkable (a free-standing partition end, an isolated wall).
An incomplete reconstruction therefore stays representable and `valid` -- it is merely not `complete`.

Rooms are the bounded faces of the planar graph formed by wall junctions (half-edge traversal), so a room is
found only when its boundary is actually closed by evidence; a boundary left open is reported with the gap size
and which side is unobserved, never silently closed. Room area carries a first-order propagated sigma.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .association import line_through, signed
from .constraints import CKind, ConstraintConfig, ConstraintReport, CStatus
from .model import Conflict, Kind, SpatialEntity


class Severity(str, Enum):
    STRUCTURAL_ERROR = "structural_error"
    CONTRADICTION = "contradiction"
    INCOMPLETE = "incomplete"
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    code: str
    severity: Severity
    entities: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    message: str
    magnitude: float | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity.value, "entities": list(self.entities),
                "evidence_ids": list(self.evidence_ids), "message": self.message, "magnitude": self.magnitude}


@dataclass(frozen=True)
class Room:
    room_id: str
    storey: str | None
    wall_ids: tuple[str, ...]
    polygon: tuple[tuple[float, float], ...]
    area_m2: float
    area_sigma_m2: float | None
    boundary_has_unobserved_end: bool
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"room_id": self.room_id, "storey": self.storey, "wall_ids": list(self.wall_ids),
                "polygon": [list(p) for p in self.polygon], "area_m2": self.area_m2,
                "area_sigma_m2": self.area_sigma_m2, "boundary_has_unobserved_end": self.boundary_has_unobserved_end,
                "evidence_ids": list(self.evidence_ids)}


@dataclass
class TopologyReport:
    issues: list[Issue]
    rooms: list[Room]
    room_adjacency: list[tuple[str, str, str, tuple[str, ...]]]   # (room_a, room_b, shared wall, openings on it)
    nodes: dict[int, tuple[float, float]]
    edges: list[tuple[int, int, str]]
    dangling: list[tuple[str, str]]                               # (wall id, end 'a'|'b') with degree-1 node

    def by_severity(self, sev: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity is sev]

    @property
    def valid(self) -> bool:
        return not self.by_severity(Severity.STRUCTURAL_ERROR)

    @property
    def complete(self) -> bool:
        return not self.by_severity(Severity.INCOMPLETE) and not self.by_severity(Severity.CONTRADICTION)

    def status(self) -> str:
        if not self.valid:
            return "invalid"
        if self.by_severity(Severity.CONTRADICTION):
            return "contradictory"
        return "incomplete" if self.by_severity(Severity.INCOMPLETE) else "consistent"


# ----------------------------------------------------------------------------------------------- planar graph

class _DSU:
    def __init__(self):
        self.p: dict[int, int] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build_graph(walls: Sequence[SpatialEntity], junctions: Sequence[tuple[str, str, str, str]], node_tol: float = 0.1):
    """Nodes = wall ends + the places other walls' ends land on a wall; edges = wall pieces between nodes.
    Only junctions backed by evidence (constraints.py) create contact -- proximity alone never does."""
    byid = {w.entity_id: w for w in walls}
    pts: list[tuple[str, float, float, tuple[float, float]]] = []   # (wall, t, tag, point)
    for w in walls:
        fit = line_through(w.a, w.b)
        pts.append((w.entity_id, signed(fit, w.a)[0], "a", w.a))
        pts.append((w.entity_id, signed(fit, w.b)[0], "b", w.b))
    for wi, which, wj, _ in junctions:
        p = byid[wi].a if which == "a" else byid[wi].b
        pts.append((wj, signed(line_through(byid[wj].a, byid[wj].b), p)[0], "j", p))
    dsu = _DSU()
    for i in range(len(pts)):
        dsu.find(i)
        for j in range(i):
            if byid[pts[i][0]].storey == byid[pts[j][0]].storey and math.dist(pts[i][3], pts[j][3]) <= node_tol:
                dsu.union(i, j)
    # a junction point's partner on its own wall is that wall's end node: union (wi, which) point with the placed one
    index = {}
    for k, (wid, _, tag, _) in enumerate(pts):
        if tag in "ab":
            index[(wid, tag)] = k
    n_end = 2 * len(walls)
    for q, (wi, which, wj, _) in enumerate(junctions):
        dsu.union(n_end + q, index[(wi, which)])
    roots: dict[int, int] = {}
    nodes: dict[int, list[tuple[float, float]]] = {}
    for k, (_, _, _, p) in enumerate(pts):
        r = roots.setdefault(dsu.find(k), len(roots))
        nodes.setdefault(r, []).append(p)
    node_xy = {r: (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps)) for r, ps in nodes.items()}
    per_wall: dict[str, list[tuple[float, int]]] = {}
    for k, (wid, t, _, _) in enumerate(pts):
        per_wall.setdefault(wid, []).append((t, roots[dsu.find(k)]))
    edges: list[tuple[int, int, str]] = []
    for wid, lst in per_wall.items():
        seq = []
        for t, r in sorted(lst):
            if not seq or seq[-1] != r:
                seq.append(r)
        edges += [(a, b, wid) for a, b in zip(seq, seq[1:]) if a != b]
    return node_xy, edges, roots, pts, dsu


def _faces(node_xy: dict[int, tuple[float, float]], edges: list[tuple[int, int, str]]):
    """Bounded faces by half-edge traversal after pruning degree-1 spurs. Returns [(node cycle, wall ids per edge)]."""
    adj: dict[int, set[int]] = {}
    wall_of: dict[tuple[int, int], str] = {}
    for a, b, w in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        wall_of[(a, b)] = wall_of[(b, a)] = w
    changed = True
    while changed:
        changed = False
        for v in [v for v, nb in adj.items() if len(nb) <= 1]:
            for u in adj.pop(v):
                adj[u].discard(v)
            changed = True
    ang = lambda a, b: math.atan2(node_xy[b][1] - node_xy[a][1], node_xy[b][0] - node_xy[a][0])  # noqa: E731
    order = {v: sorted(nb, key=lambda u, v=v: ang(v, u)) for v, nb in adj.items()}
    seen, faces = set(), []
    for u in adj:
        for v in adj[u]:
            if (u, v) in seen:
                continue
            cyc, a, b = [], u, v
            while (a, b) not in seen:
                seen.add((a, b))
                cyc.append((a, b))
                lst = order[b]
                a, b = b, lst[(lst.index(a) - 1) % len(lst)]   # next clockwise from the twin: face stays on the left
            faces.append(cyc)
    out = []
    for cyc in faces:
        poly = [node_xy[a] for a, _ in cyc]
        area = 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                         for i in range(len(poly)))
        if area > 1e-6:                                         # CCW = bounded; the outer face is negative
            out.append(([a for a, _ in cyc], [wall_of[e] for e in cyc]))
    return out, adj


def _segments_cross(p1, p2, p3, p4, margin: float) -> float | None:
    """Interior crossing of two segments; returns the smaller distance from the crossing to any of the four
    end points (so a junction-like touch near an end is not a crossing), else None."""
    d1, d2 = (p2[0] - p1[0], p2[1] - p1[1]), (p4[0] - p3[0], p4[1] - p3[1])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / den
    if not (0 < t < 1 and 0 < u < 1):
        return None
    x = (p1[0] + t * d1[0], p1[1] + t * d1[1])
    near = min(math.dist(x, q) for q in (p1, p2, p3, p4))
    return near if near > margin else None


def _area_sigma(poly, sig) -> float | None:
    if any(s is None for s in sig):
        return None
    n = len(poly)
    var = 0.0
    for i in range(n):
        dx = (poly[(i + 1) % n][1] - poly[i - 1][1]) / 2.0
        dy = (poly[i - 1][0] - poly[(i + 1) % n][0]) / 2.0
        var += (dx * dx + dy * dy) * sig[i] ** 2
    return math.sqrt(var)


# --------------------------------------------------------------------------------------------------- validation

def validate_topology(
    entities: Sequence[SpatialEntity], report: ConstraintReport | None = None,
    conflicts: Sequence[Conflict] = (), *, known_evidence_ids: set[str] | None = None,
    cfg: ConstraintConfig = ConstraintConfig(), gap_search_m: float = 1.5, min_room_area_m2: float = 0.25,
) -> TopologyReport:
    issues: list[Issue] = []
    byid = {e.entity_id: e for e in entities}
    walls = [e for e in entities if e.kind is Kind.WALL]
    openings = [e for e in entities if e.kind in (Kind.DOOR, Kind.WINDOW, Kind.OPENING)]
    junctions = report.junctions if report else []

    def add(code, sev, ents, msg, mag=None):
        ev = tuple(sorted({x for i in ents if i in byid for x in byid[i].evidence_ids}))
        issues.append(Issue(code, sev, tuple(ents), ev, msg, mag))

    for e in entities:   # coordinate frame / finiteness
        if not all(math.isfinite(c) for c in (*e.a, *e.b)):
            add("non_finite_coordinates", Severity.STRUCTURAL_ERROR, [e.entity_id], f"{e.entity_id} has non-finite coordinates")
    # references and evidence traceability (orphan geometry)
    for e in entities:
        if e.host_id is not None and e.host_id not in byid:
            add("broken_reference", Severity.STRUCTURAL_ERROR, [e.entity_id], f"{e.entity_id} references missing host {e.host_id}")
        if not e.evidence_ids:
            add("orphan_geometry", Severity.STRUCTURAL_ERROR, [e.entity_id], f"{e.entity_id} has no supporting evidence")
        elif known_evidence_ids is not None and not set(e.evidence_ids) <= known_evidence_ids:
            miss = sorted(set(e.evidence_ids) - known_evidence_ids)
            add("unresolvable_evidence", Severity.STRUCTURAL_ERROR, [e.entity_id], f"{e.entity_id} cites unknown evidence {miss[:3]}")
        if len(e.image_ids) < 2:
            add("single_view_entity", Severity.INCOMPLETE, [e.entity_id], f"{e.entity_id} rests on a single view")
        if e.host_id is None and e.kind in (Kind.DOOR, Kind.WINDOW, Kind.OPENING):
            add("opening_host_unknown", Severity.INCOMPLETE, [e.entity_id],
                f"{e.entity_id} has no determined host wall" + (f" (candidates {list(e.host_alternatives)})" if e.host_alternatives else ""))
        if e.host_id in byid and byid[e.host_id].storey != e.storey:
            add("cross_storey_attachment", Severity.STRUCTURAL_ERROR, [e.entity_id, e.host_id], f"{e.entity_id} is attached to a wall on another storey")
    # duplicates: same kind & storey, both ends inside each other's gate
    for i, a in enumerate(entities):
        for b in entities[i + 1:]:
            if a.kind is b.kind and a.storey == b.storey and a.kind is Kind.WALL:
                tol = cfg.k * math.sqrt((a.beliefs["position"].sigma or 0) ** 2 + (b.beliefs["position"].sigma or 0) ** 2 + cfg.floor_m ** 2)
                if (math.dist(a.a, b.a) <= tol and math.dist(a.b, b.b) <= tol) or (math.dist(a.a, b.b) <= tol and math.dist(a.b, b.a) <= tol):
                    add("duplicate_entity", Severity.STRUCTURAL_ERROR, [a.entity_id, b.entity_id], f"{a.entity_id} and {b.entity_id} describe the same wall")
    # surfaced conflicts
    for c in conflicts:
        sev = Severity.STRUCTURAL_ERROR if c.code == "untransformable_frame" else Severity.CONTRADICTION
        issues.append(Issue(c.code, sev, c.entities, c.evidence_ids, c.message, c.magnitude))
    if report:
        for c in report.contradictions:
            issues.append(Issue(c.code, Severity.CONTRADICTION, c.entities, c.evidence_ids, c.message, c.magnitude))
        sev_for = {CKind.OPENING_ATTACHMENT: Severity.STRUCTURAL_ERROR, CKind.OPENING_DIMENSIONS: Severity.STRUCTURAL_ERROR,
                   CKind.FLOOR_CEILING: Severity.STRUCTURAL_ERROR, CKind.STAIR_CONNECTIVITY: Severity.STRUCTURAL_ERROR,
                   CKind.COORDINATE_FRAME: Severity.STRUCTURAL_ERROR, CKind.GEOMETRIC_CONTINUITY: Severity.INCOMPLETE,
                   CKind.WALL_CONNECTIVITY: Severity.CONTRADICTION, CKind.WALL_INTERSECTION: Severity.CONTRADICTION}
        for c in report.constraints:
            if c.status is CStatus.VIOLATED and c.kind in sev_for and c.kind not in (CKind.WALL_CONNECTIVITY, CKind.WALL_INTERSECTION):
                code = {CKind.GEOMETRIC_CONTINUITY: "observed_gap_between_collinear_walls"}.get(c.kind, f"{c.kind.value}_violated")
                issues.append(Issue(code, sev_for[c.kind], c.participants, c.evidence_ids, c.message, c.measured))

    node_xy, edges, roots, pts, dsu = build_graph(walls, junctions)
    # unjoined crossings: two walls pass through each other with no junction
    for i, (a1, b1, w1) in enumerate(edges):
        for (a2, b2, w2) in edges[i + 1:]:
            if w1 == w2 or {a1, b1} & {a2, b2} or byid[w1].storey != byid[w2].storey:
                continue
            near = _segments_cross(node_xy[a1], node_xy[b1], node_xy[a2], node_xy[b2], 0.1)
            if near is not None:
                add("unjoined_crossing", Severity.STRUCTURAL_ERROR, [w1, w2], f"{w1} and {w2} cross {near:.2f} m from the nearest end with no junction", near)

    deg: dict[int, int] = {}
    for a, b, _ in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    dangling: list[tuple[str, str]] = []
    end_node = {(w.entity_id, tag): roots[dsu.find(k)] for k, (wid, _, tag, _) in enumerate(pts) if tag in "ab"
                for w in [byid[wid]]}
    for (wid, tag), nd in sorted(end_node.items()):
        if deg.get(nd, 0) <= 1:
            dangling.append((wid, tag))
            w = byid[wid]
            seen = w.end_a_observed if tag == "a" else w.end_b_observed
            p = w.a if tag == "a" else w.b
            others = [o for o in walls if o is not w and o.storey == w.storey]
            gaps = []
            for o in others:
                f = line_through(o.a, o.b)
                t, d = signed(f, p)
                if f.t_lo - gap_search_m <= t <= f.t_hi + gap_search_m and abs(d) <= gap_search_m:
                    gaps.append((abs(d), o))
            if not seen:
                add("dangling_end_unobserved", Severity.INCOMPLETE, [wid], f"end '{tag}' of {wid} was never seen as a termination (occluded / cut by the image border)")
            elif gaps:
                d, o = min(gaps, key=lambda g: g[0])
                far_seen = (o.end_a_observed and o.end_b_observed)
                add("open_boundary_observed_gap", Severity.INCOMPLETE, [wid, o.entity_id],
                    f"end '{tag}' of {wid} stops {d:.2f} m short of {o.entity_id}; a passage or a missing wall cannot be told apart from the evidence"
                    + ("" if far_seen else " (the other wall's extent is partly unobserved)"), d)
            else:
                add("free_wall_end", Severity.INFO, [wid], f"end '{tag}' of {wid} is an observed free termination")
    for w in walls:
        if not any(w.entity_id in (a, b) for a, b, _ in [(j[0], j[2], 0) for j in junctions]) and len(walls) > 1:
            add("isolated_wall", Severity.INFO, [w.entity_id], f"{w.entity_id} has no junction evidence with any other wall")

    # rooms from bounded faces
    faces, _ = _faces(node_xy, edges)
    node_sigma: dict[int, list[float]] = {}
    for k, (wid, _, tag, p) in enumerate(pts):
        if tag in "ab":
            s = byid[wid].end_sigma(tag)
            node_sigma.setdefault(roots[dsu.find(k)], []).append(s if s is not None else float("nan"))
    rooms: list[Room] = []
    for cyc, wids in faces:
        poly = tuple(node_xy[v] for v in cyc)
        area = 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
        if area < min_room_area_m2:
            continue
        uniq = tuple(sorted(set(wids)))
        sig = [min((s for s in node_sigma.get(v, [float("nan")]) if s == s), default=None) for v in cyc]
        ev = tuple(sorted({x for w in uniq for x in byid[w].evidence_ids}))
        unobs = any(not (byid[w].end_a_observed and byid[w].end_b_observed) for w in uniq)
        rid = "room-" + hashlib.sha1("|".join(uniq).encode()).hexdigest()[:10]
        rooms.append(Room(rid, byid[uniq[0]].storey, uniq, poly, area, _area_sigma(poly, sig), unobs, ev))
        # simple-polygon check
        n = len(poly)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if _segments_cross(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n], 0.0) is not None:
                    add("invalid_room_boundary", Severity.STRUCTURAL_ERROR, list(uniq), f"room boundary {rid} self-intersects")
                    break
    adjacency = []
    for i, ra in enumerate(rooms):
        for rb in rooms[i + 1:]:
            if ra.storey != rb.storey:
                continue
            for w in sorted(set(ra.wall_ids) & set(rb.wall_ids)):
                adjacency.append((ra.room_id, rb.room_id, w, tuple(sorted(o.entity_id for o in openings if o.host_id == w))))
    # stairs joining the same storey twice are an impossible connectivity
    return TopologyReport(sorted(issues, key=lambda i: (i.severity.value, i.code, i.entities)), rooms, adjacency,
                          node_xy, edges, dangling)
