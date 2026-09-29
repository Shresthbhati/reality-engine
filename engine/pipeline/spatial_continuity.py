"""Geometry-aware continuity of structural entities between two versions of the SAME world.

Rebuilding a world from more evidence can change how it is REPRESENTED without changing the
physical world: one plane splits into several, several merge, a plane shifts a little, entity
ids are re-derived. Comparing entity counts or ids cannot tell those apart from a real change.
This module compares the GEOMETRY that is actually stored on the version report:

    per entity:  type, centre, inlier point sample (<= MAX_POINTS), bounds, supporting evidence ids

and, for planar structure (wall / floor / ceiling with enough points), a plane fitted to those
points (normal, in-plane axes, extents). Two patches are the same surface when

    * they have the same type
    * their normals agree (|cos| within ANGLE_TOL_DEG)
    * their planes are close (mean offset <= MOVE_TOL_REL of the scene extent)
    * their IN-PLANE FOOTPRINTS overlap (rasterised occupancy, one dilation cell of tolerance)

Relations (each carries the measured coverage and its explanation, nothing is guessed):

    preserved   one-to-one, both cover >= COVER_SAME, negligible shift/tilt
    refined     one-to-one, both cover >= COVER_SAME, but moved / tilted / resized
    extended    the previous surface is covered by the candidate, which reaches further (new area)
    reduced     the candidate covers only part of the previous surface (the rest is NOT reproduced)
    split       one previous surface -> several candidate fragments that jointly cover it
    merge       several previous surfaces -> one candidate that covers them
    regrouped   several <-> several, where the UNION of each side covers the other's footprint (stacked or
                overlapping planes grouped differently): same area carried, grouping changed
    ambiguous  overlapping but not classifiable (many-to-many, or partial both ways): reported, not forced
    removed     nothing in the candidate overlaps it
    new         nothing in the previous version overlaps it

Signals that do not exist are skipped, never fabricated: without stored points an entity falls back to
centre distance (``signal: center_only``) and can only be preserved / refined / removed / new. A move
is only called a CONFLICT when supporting-evidence ids exist on both sides and no new evidence
supports the moved candidate; otherwise it is recorded as a refinement whose justification is unknown.

Generic by construction: no dataset, coordinate, or building-part names appear here.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

MAX_POINTS = 160
MIN_PLANE_POINTS = 8
PLANAR_TYPES = ("wall", "floor", "ceiling")

COVER_MIN = 0.15            # two patches overlap at all
COVER_SAME = 0.60           # one covers most of the other
ANGLE_TOL_DEG = 20.0        # normals further apart than this are different surfaces
MOVE_TOL_REL = 0.10         # planes further apart than this (x scene extent) are different surfaces, not a moved one
PRESERVED_SHIFT_REL = 0.03  # <= this offset, ...
PRESERVED_ANGLE_DEG = 4.0   # ... and <= this tilt = unchanged
CONFLICT_SHIFT_REL = 0.05   # an unsupported move beyond either of these opens a geometry conflict
CONFLICT_ANGLE_DEG = 8.0
CENTER_MATCH_REL = 0.20     # fallback (no points): centre distance for a match
REGION_CLUSTER_REL = 0.30   # patches closer than this (x scene extent) share a region
PLANE_RMS_TO_WIDTH = 0.10   # rms out-of-plane <= this x the short in-plane extent to count as planar


@dataclass
class Patch:
    id: str
    type: str
    center: np.ndarray
    size: float
    points: Optional[np.ndarray]
    evidence: Optional[FrozenSet[str]]
    provenance: str
    confidence: float
    planar: bool = False
    normal: Optional[np.ndarray] = None
    u: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    extents: Tuple[float, float] = (0.0, 0.0)

    @property
    def area(self) -> float:
        return self.extents[0] * self.extents[1]


def fit_plane(points: np.ndarray) -> Optional[dict]:
    """PCA plane through ``points`` (N,3): centre, unit normal, in-plane axes, rms offset, extents."""
    if points is None or len(points) < MIN_PLANE_POINTS:
        return None
    c = points.mean(0)
    d = points - c
    try:
        _, _, vt = np.linalg.svd(d, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    u, v, n = vt[0], vt[1], vt[2]
    eu, ev = float(np.ptp(d @ u)), float(np.ptp(d @ v))
    if ev <= 1e-9:
        return None
    return {"center": c, "normal": n, "u": u, "v": v, "rms": float(np.sqrt(np.mean((d @ n) ** 2))),
            "extents": (eu, ev)}


def sample_points(points: Sequence[Sequence[float]], limit: int = MAX_POINTS) -> List[List[float]]:
    """Deterministic, evenly strided sample (rounded) so stored reports stay small."""
    pts = list(points)
    if len(pts) > limit:
        step = len(pts) / float(limit)
        pts = [pts[int(i * step)] for i in range(limit)]
    return [[round(float(c), 4) for c in p] for p in pts]


def build_patch(rec: dict, to_prev: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                scale: float = 1.0) -> Patch:
    """A record (as stored on the version report) -> Patch, optionally mapped into the previous frame."""
    raw = rec.get("points")
    pts = np.asarray(raw, float) if raw else None
    if pts is not None and to_prev is not None:
        pts = to_prev(pts)
    center = np.asarray(rec["center"], float)
    if to_prev is not None:
        center = to_prev(center.reshape(1, 3))[0]
    b = rec.get("bounds")
    size = float(np.linalg.norm(np.asarray(b[1], float) - np.asarray(b[0], float))) * scale if b else 0.0
    ev = rec.get("evidence")
    patch = Patch(id=rec["id"], type=rec["type"], center=center, size=size, points=pts,
                  evidence=frozenset(ev) if ev is not None else None,
                  provenance=rec.get("provenance", "UNKNOWN"), confidence=float(rec.get("confidence", 0.0)))
    if rec["type"] in PLANAR_TYPES and pts is not None:
        fit = fit_plane(pts)
        if fit is not None and fit["rms"] <= PLANE_RMS_TO_WIDTH * fit["extents"][1]:
            patch.planar = True
            patch.center, patch.normal, patch.u, patch.v = fit["center"], fit["normal"], fit["u"], fit["v"]
            patch.extents = fit["extents"]
            patch.size = max(patch.size, math.hypot(*fit["extents"]))
    return patch


def _occupancy(uv: np.ndarray, lo: np.ndarray, h: float):
    cells = set(map(tuple, np.floor((uv - lo) / h).astype(int).tolist()))
    dilated = {(i + di, j + dj) for (i, j) in cells for di in (-1, 0, 1) for dj in (-1, 0, 1)}
    return cells, dilated


def footprint_cover(a: Patch, b: Patch) -> Tuple[float, float]:
    """(share of a's footprint touched by b, share of b's footprint touched by a), in a's plane basis."""
    def uv(p):
        d = p - a.center
        return np.c_[d @ a.u, d @ a.v]

    A, B = uv(a.points), uv(b.points)

    def spacing(P):
        w = np.ptp(P, axis=0)
        return math.sqrt(max(float(w[0] * w[1]), 1e-12) / len(P))

    h = 1.5 * max(spacing(A), spacing(B), 1e-9)
    lo = np.minimum(A.min(0), B.min(0))
    ca, da = _occupancy(A, lo, h)
    cb, db = _occupancy(B, lo, h)
    return len(ca & db) / max(len(ca), 1), len(cb & da) / max(len(cb), 1)


def union_cover(a: Patch, others: Sequence[Patch]) -> float:
    """Share of a's footprint touched by ANY of ``others`` (one common grid, so overlaps are not double-counted)."""
    def uv(p):
        d = p - a.center
        return np.c_[d @ a.u, d @ a.v]

    A, Bs = uv(a.points), [uv(b.points) for b in others]
    if not Bs:
        return 0.0

    def spacing(P):
        w = np.ptp(P, axis=0)
        return math.sqrt(max(float(w[0] * w[1]), 1e-12) / len(P))

    h = 1.5 * max([spacing(A)] + [spacing(B) for B in Bs] + [1e-9])
    lo = np.minimum(A.min(0), np.min([B.min(0) for B in Bs], axis=0))
    cells, _ = _occupancy(A, lo, h)
    touched = set()
    for B in Bs:
        touched |= _occupancy(B, lo, h)[1]
    return len(cells & touched) / max(len(cells), 1)


def _pair(a: Patch, b: Patch, ext: float) -> Optional[dict]:
    """How patch ``a`` (previous) relates to ``b`` (candidate, in a's frame), or None when unrelated."""
    if a.type != b.type:
        return None
    if a.planar and b.planar:
        ang = math.degrees(math.acos(min(1.0, abs(float(a.normal @ b.normal)))))
        if ang > ANGLE_TOL_DEG:
            return None
        gap = float(np.mean(np.abs((b.points - a.center) @ a.normal)))
        if gap > MOVE_TOL_REL * ext:
            return None
        ca, cb = footprint_cover(a, b)
        if max(ca, cb) < COVER_MIN:
            return None
        shift = float(np.linalg.norm((b.center - a.center) - ((b.center - a.center) @ a.normal) * a.normal))
        return {"signal": "plane_footprint", "cover_prev": ca, "cover_cand": cb, "angle_deg": ang,
                "gap_rel": gap / ext, "shift_rel": shift / ext,
                "area_ratio": (b.area / a.area) if a.area > 1e-12 else None}
    d = float(np.linalg.norm(a.center - b.center))
    if d > max(CENTER_MATCH_REL * ext, 0.5 * max(a.size, b.size)):
        return None
    return {"signal": "center_only", "cover_prev": 1.0, "cover_cand": 1.0, "angle_deg": None,
            "gap_rel": d / ext, "shift_rel": d / ext, "area_ratio": None, "dist": d}


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def reconcile(prev_records: List[dict], cand_records: List[dict], *, ext_prev: float,
              to_prev: Callable[[np.ndarray], np.ndarray], scale: float,
              new_evidence_ids: Sequence[str] = ()) -> dict:
    """Classify how the previous version's structure is carried by the candidate. See module docstring."""
    P = [build_patch(r) for r in prev_records]
    Q = [build_patch(r, to_prev, 1.0 / scale) for r in cand_records]
    new_ev = frozenset(new_evidence_ids)

    edges: Dict[Tuple[int, int], dict] = {}
    center_only: List[Tuple[float, int, int, dict]] = []
    for i, a in enumerate(P):
        for j, b in enumerate(Q):
            info = _pair(a, b, ext_prev)
            if info is None:
                continue
            if info["signal"] == "center_only":
                center_only.append((info["dist"], i, j, info))
            else:
                edges[(i, j)] = info
    # without shape data an entity may match only ONE counterpart: greedy nearest, one-to-one
    used_i, used_j = set(), set()
    for _, i, j, info in sorted(center_only, key=lambda t: t[0]):
        if i not in used_i and j not in used_j:
            edges[(i, j)] = info
            used_i.add(i)
            used_j.add(j)

    uf = _UF(len(P) + len(Q))
    for (i, j) in edges:
        uf.union(i, len(P) + j)
    comps: Dict[int, Tuple[List[int], List[int]]] = {}
    for i in range(len(P)):
        comps.setdefault(uf.find(i), ([], []))[0].append(i)
    for j in range(len(Q)):
        comps.setdefault(uf.find(len(P) + j), ([], []))[1].append(j)

    relations: List[dict] = []
    for pi, qj in comps.values():
        pids = [P[i].id for i in pi]
        qids = [Q[j].id for j in qj]
        typ = (P[pi[0]] if pi else Q[qj[0]]).type
        base = {"type": typ, "prev": pids, "cand": qids}
        if not qj:
            relations.append({**base, "kind": "removed", "signal": None,
                              "provenance": P[pi[0]].provenance,
                              "explanation": f"no candidate {typ} overlaps this previous {typ}"})
            continue
        if not pi:
            ev = Q[qj[0]].evidence
            relations.append({**base, "kind": "new", "signal": None,
                              "supporting_evidence": sorted(ev & new_ev) if ev is not None else None,
                              "explanation": f"a {typ} with no counterpart in the previous version"})
            continue
        sub = {(i, j): edges[(i, j)] for i in pi for j in qj if (i, j) in edges}
        signals = sorted({e["signal"] for e in sub.values()})
        rel = {**base, "signal": "+".join(signals)}
        if len(pi) == 1 and len(qj) == 1:
            i, j = pi[0], qj[0]
            e = sub[(i, j)]
            ca, cb = e["cover_prev"], e["cover_cand"]
            rel.update(cover_prev=round(ca, 3), cover_cand=round(cb, 3), angle_deg=e["angle_deg"],
                       gap_rel=round(e["gap_rel"], 4), shift_rel=round(e["shift_rel"], 4))
            p, q = P[i], Q[j]
            support = None
            if p.evidence is not None and q.evidence is not None:
                support = sorted(q.evidence - p.evidence)
            rel["supporting_new_evidence"] = support
            # both sides of the comparison, in the previous frame, so a conflict can keep both hypotheses
            rel.update(prev_center=[round(float(c), 4) for c in p.center],
                       cand_center=[round(float(c), 4) for c in q.center],
                       prev_evidence=sorted(p.evidence) if p.evidence is not None else None,
                       cand_evidence=sorted(q.evidence) if q.evidence is not None else None,
                       prev_confidence=p.confidence, cand_confidence=q.confidence)
            if ca >= COVER_SAME and cb >= COVER_SAME:
                still = e["gap_rel"] <= PRESERVED_SHIFT_REL and (e["angle_deg"] or 0.0) <= PRESERVED_ANGLE_DEG
                rel["kind"] = "preserved" if still else "refined"
                if not still:
                    moved = e["gap_rel"] > CONFLICT_SHIFT_REL or (e["angle_deg"] or 0.0) > CONFLICT_ANGLE_DEG
                    rel["moved_beyond_tolerance"] = bool(moved)
                    rel["unsupported_move"] = bool(moved and support is not None and not support)
                rel["explanation"] = ("same surface, same position within tolerance" if still else
                                      "same surface, position or orientation changed within the matching tolerance")
            elif ca >= COVER_SAME:
                rel["kind"] = "extended"
                rel["explanation"] = (f"the previous {typ} is covered by a candidate that reaches "
                                      f"further (only {cb:.0%} of the candidate overlaps the previous one)")
            elif cb >= COVER_SAME:
                rel["kind"] = "reduced"
                rel["explanation"] = f"only {ca:.0%} of the previous {typ} is reproduced by the candidate"
            else:
                rel["kind"] = "ambiguous"
                rel["explanation"] = f"partial overlap both ways ({ca:.0%} / {cb:.0%})"
        elif len(pi) == 1:
            i = pi[0]
            linked = [Q[j] for j in qj if (i, j) in sub]
            total = (union_cover(P[i], linked) if P[i].planar and all(q.planar for q in linked)
                     else min(1.0, sum(sub[(i, j)]["cover_prev"] for j in qj if (i, j) in sub)))
            grew = any(sub[(i, j)]["cover_cand"] < COVER_SAME for j in qj if (i, j) in sub)
            rel.update(cover_prev=round(total, 3))
            if total >= COVER_SAME:
                rel["kind"] = "split"
                rel["grew"] = bool(grew)
                rel["explanation"] = (f"one previous {typ} is now {len(qj)} fragments that together cover "
                                      f"{total:.0%} of it: a change of representation, not of the world")
            else:
                rel["kind"] = "ambiguous"
                rel["explanation"] = f"one previous {typ} overlaps {len(qj)} fragments but they cover only {total:.0%} of it"
        elif len(qj) == 1:
            j = qj[0]
            each = [sub[(i, j)]["cover_prev"] for i in pi if (i, j) in sub]
            total_c = min(1.0, sum(sub[(i, j)]["cover_cand"] for i in pi if (i, j) in sub))
            rel.update(cover_cand=round(total_c, 3))
            if each and min(each) >= COVER_SAME:
                rel["kind"] = "merge"
                rel["explanation"] = (f"{len(pi)} previous {typ}s are now one surface that covers each of them: "
                                      "a change of representation, not of the world")
            else:
                rel["kind"] = "ambiguous"
                rel["explanation"] = f"{len(pi)} previous {typ}s overlap one candidate but not all are covered by it"
        else:
            # many-to-many. Ask the physical question instead of giving up: does the UNION of one side cover
            # the footprint of the other? (segmentation often stacks several overlapping planes per surface)
            planar = all(P[i].planar for i in pi) and all(Q[j].planar for j in qj)
            if planar:
                def wmean(vals_weights):
                    tot = sum(w for _, w in vals_weights) or 1.0
                    return sum(v * w for v, w in vals_weights) / tot

                prev_cov = wmean([(union_cover(P[i], [Q[j] for j in qj if (i, j) in sub]) if any((i, j) in sub for j in qj) else 0.0,
                                   len(P[i].points)) for i in pi])
                cand_cov = wmean([(union_cover(Q[j], [P[i] for i in pi if (i, j) in sub]) if any((i, j) in sub for i in pi) else 0.0,
                                   len(Q[j].points)) for j in qj])
                rel.update(cover_prev=round(prev_cov, 3), cover_cand=round(cand_cov, 3))
                if prev_cov >= COVER_SAME and cand_cov >= COVER_SAME:
                    rel["kind"] = "regrouped"
                    rel["explanation"] = (f"{len(pi)} previous and {len(qj)} candidate {typ}s cover the same area "
                                          f"({prev_cov:.0%} / {cand_cov:.0%}): the surfaces are grouped differently, "
                                          "the world is not")
                elif prev_cov >= COVER_SAME:
                    rel["kind"] = "extended"
                    rel["explanation"] = (f"the {len(pi)} previous {typ}s are covered ({prev_cov:.0%}) by {len(qj)} "
                                          f"candidates that reach further ({cand_cov:.0%} overlap)")
                elif cand_cov >= COVER_SAME:
                    rel["kind"] = "reduced"
                    rel["explanation"] = f"only {prev_cov:.0%} of the previous {typ}s' area is reproduced"
                else:
                    rel["kind"] = "ambiguous"
                    rel["explanation"] = (f"{len(pi)} previous and {len(qj)} candidate {typ}s overlap only partly "
                                          f"({prev_cov:.0%} / {cand_cov:.0%})")
            else:
                rel["kind"] = "ambiguous"
                rel["explanation"] = f"{len(pi)} previous and {len(qj)} candidate {typ}s overlap in a many-to-many pattern"
        relations.append(rel)

    counts = Counter(r["kind"] for r in relations)
    return {
        "matched_by": "footprint overlap of fitted planes where points exist, else type + centre distance",
        "relations": relations,
        "counts": {k: counts.get(k, 0) for k in
                   ("preserved", "refined", "extended", "reduced", "split", "merge", "regrouped", "ambiguous",
                    "removed", "new")},
        "signals": dict(Counter(r.get("signal") for r in relations if r.get("signal"))),
        "regions": regions(P, Q, relations, ext_prev, new_ev),
        "prev_total": len(P), "cand_total": len(Q),
    }


def regions(P: List[Patch], Q: List[Patch], relations: List[dict], ext: float,
            new_evidence: FrozenSet[str]) -> List[dict]:
    """Smallest useful spatial grouping: relations whose patches are near each other (in the previous
    frame) form one region. A region says WHERE the world changed and which evidence supports it."""
    by_id = {p.id: p for p in P}
    by_id.update({("cand:" + q.id): q for q in Q})

    def members(rel):
        return [by_id[i] for i in rel["prev"]] + [by_id["cand:" + i] for i in rel["cand"]]

    centres = [np.mean([m.center for m in members(r)], axis=0) for r in relations]
    uf = _UF(len(relations))
    for a in range(len(relations)):
        for b in range(a + 1, len(relations)):
            if float(np.linalg.norm(centres[a] - centres[b])) <= REGION_CLUSTER_REL * ext:
                uf.union(a, b)
    groups: Dict[int, List[int]] = {}
    for k in range(len(relations)):
        groups.setdefault(uf.find(k), []).append(k)

    out = []
    for n, idx in enumerate(sorted(groups.values(), key=lambda g: (-len(g), g[0])), start=1):
        rels = [relations[k] for k in idx]
        kinds = Counter(r["kind"] for r in rels)
        ev: set = set()
        unknown_ev = False
        for r in rels:
            for i in r["cand"]:
                e = by_id["cand:" + i].evidence
                if e is None:
                    unknown_ev = True
                else:
                    ev |= e
        cen = np.mean([centres[k] for k in idx], axis=0)
        changed = [k for k in kinds if k != "preserved"]
        status = ("new" if set(kinds) == {"new"} else "unchanged" if not changed else "changed")
        out.append({
            "id": f"region-{n}", "center": [round(float(c), 4) for c in cen],
            "types": dict(Counter(r["type"] for r in rels)), "relations": dict(kinds), "status": status,
            "supporting_evidence": None if unknown_ev and not ev else sorted(ev),
            "affected_by_new_evidence": sorted(ev & new_evidence) if not (unknown_ev and not ev) else None,
        })
    return out
