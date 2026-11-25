"""Evidence-supported spatial constraint engine.

Principle: a constraint is only IMPOSED when the observations themselves support it (an observed wall end lies
within its k-sigma gate of another wall's line); otherwise it is only EVALUATED and reported. Nothing here assumes
a rectangular or Manhattan building: parallelism / perpendicularity are measured and reported, and are applied to
the solution only if the caller opts in (`apply_alignment_priors`) -- by default they are priors, not evidence.

Solver: weighted nonlinear least squares (scipy `least_squares`, robust soft-L1) over every wall end point.
  prior residuals    each end point is pulled to its fitted position with its along / across sigma,
  junction residuals the signed distance of an observed end to the line of the wall it touches, / junction_sigma.
Contradictions are SURFACED, never silently resolved: a junction that can only be satisfied by moving a wall
more than k sigma from its own evidence is returned as a `Conflict("constraint_contradiction")` and the
constraint is marked VIOLATED. Posterior end-point sigmas come from (J^T J)^-1 at the solution, so a corner
confirmed by two walls is more certain than either wall alone, and an end with no junction keeps its sigma.

Constraint inventory: wall connectivity (corner), wall intersection (T-junction), parallelism/perpendicularity,
geometric continuity (collinear walls with a gap), room enclosure (see topology.py), door/window-to-wall
attachment, opening dimensions, floor/ceiling relationship, stair connectivity between storeys, object
containment/adjacency, coordinate-frame consistency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from scipy.optimize import least_squares

from .association import line_through, signed
from .model import Belief, Conflict, Kind, SpatialEntity, Status, angle_diff


class CKind(str, Enum):
    WALL_CONNECTIVITY = "wall_connectivity"
    WALL_INTERSECTION = "wall_intersection"
    WALL_ALIGNMENT = "wall_alignment"
    GEOMETRIC_CONTINUITY = "geometric_continuity"
    ROOM_ENCLOSURE = "room_enclosure"
    OPENING_ATTACHMENT = "opening_attachment"
    OPENING_DIMENSIONS = "opening_dimensions"
    FLOOR_CEILING = "floor_ceiling"
    STAIR_CONNECTIVITY = "stair_connectivity"
    OBJECT_CONTAINMENT = "object_containment"
    OBJECT_ADJACENCY = "object_adjacency"
    COORDINATE_FRAME = "coordinate_frame"


class CStatus(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNDETERMINED = "undetermined"     # the evidence is insufficient to decide -- NOT a failure


@dataclass(frozen=True)
class Constraint:
    kind: CKind
    participants: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    status: CStatus
    message: str
    applied: bool = False             # True = it shaped the solution
    measured: float | None = None  # metres / radians / degrees as stated in the message
    normalized: float | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "participants": list(self.participants), "evidence_ids": list(self.evidence_ids),
                "status": self.status.value, "message": self.message, "applied": self.applied,
                "measured": self.measured, "normalized": self.normalized}


@dataclass(frozen=True)
class ConstraintConfig:
    k: float = 3.5
    floor_m: float = 0.03              # un-modelled systematic error, added in quadrature to every gate
    junction_sigma_m: float = 0.01     # how hard a supported junction is held
    continuity_gap_m: float = 1.5
    z_tolerance_m: float = 0.10
    apply_alignment_priors: bool = False
    min_along_sigma_unobserved_m: float = 1.0
    angle_floor_rad: float = 0.02


@dataclass
class ConstraintReport:
    entities: list[SpatialEntity]                      # refined (new objects; inputs untouched)
    constraints: list[Constraint]
    contradictions: list[Conflict]
    junctions: list[tuple[str, str, str, str]]         # (wall_i, end 'a'|'b', wall_j, 'corner'|'t')
    solver: dict[str, float]

    def by_kind(self, kind: CKind) -> list[Constraint]:
        return [c for c in self.constraints if c.kind is kind]

    def violated(self) -> list[Constraint]:
        return [c for c in self.constraints if c.status is CStatus.VIOLATED]


def _fit_of(e: SpatialEntity):
    return line_through(e.a, e.b)


def _end_sigma(e: SpatialEntity, which: str, cfg: ConstraintConfig) -> tuple[float, float]:
    """(along, across) 1-sigma of an end point. An unobserved end is a lower bound: weakly held along the wall."""
    s = e.sigma_a if which == "a" else e.sigma_b
    obs = e.end_a_observed if which == "a" else e.end_b_observed
    across = e.beliefs["position"].sigma or cfg.floor_m
    along = (s if (s is not None and obs) else cfg.min_along_sigma_unobserved_m)
    return max(along, 1e-3), max(across, 1e-3)


def _candidate_junctions(walls: list[SpatialEntity], cfg: ConstraintConfig):
    """Evidence-gated 'observed end of i lies on wall j' relations."""
    out = []
    for i, wi in enumerate(walls):
        for which, p, seen in (("a", wi.a, wi.end_a_observed), ("b", wi.b, wi.end_b_observed)):
            if not seen:
                continue                      # a lower-bound end proves nothing about where the wall stops
            s_e = (wi.sigma_a if which == "a" else wi.sigma_b) or cfg.floor_m
            for j, wj in enumerate(walls):
                if i == j or wi.storey != wj.storey:
                    continue
                s_th_i, s_th_j = (w.beliefs["orientation"].sigma or 0.0 for w in (wi, wj))
                if angle_diff(wi.angle % math.pi, wj.angle % math.pi) <= cfg.k * math.sqrt(
                        s_th_i ** 2 + s_th_j ** 2 + cfg.angle_floor_rad ** 2):
                    continue                  # (nearly) parallel: continuity, not a junction
                fit = _fit_of(wj)
                t, d = signed(fit, p)
                s_d = wj.beliefs["position"].sigma or 0.0
                s_t = wj.beliefs["orientation"].sigma or 0.0
                gate = cfg.k * math.sqrt(s_e ** 2 + s_d ** 2 + (t * s_t) ** 2 + cfg.floor_m ** 2)
                if abs(d) > gate:
                    continue
                g = cfg.k * math.sqrt(s_e ** 2 + cfg.floor_m ** 2)
                if t < fit.t_lo - g or t > fit.t_hi + g:
                    continue                  # beyond j's extent: no contact (a near-miss is a gap; see topology.py)
                near_a = abs(t - fit.t_lo) <= g and wj.end_a_observed
                near_b = abs(t - fit.t_hi) <= g and wj.end_b_observed
                out.append((i, which, j, "corner" if (near_a or near_b) else "t", abs(d) / gate * cfg.k))
    return out


def solve_spatial_constraints(
    entities: Sequence[SpatialEntity], config: ConstraintConfig = ConstraintConfig(),
    *, storeys: Mapping[str, float] | None = None, conflicts: Sequence[Conflict] = (),
) -> ConstraintReport:
    cfg = config
    ents = [replace(e, beliefs=dict(e.beliefs), alternatives=list(e.alternatives), end_post=dict(e.end_post)) for e in entities]
    walls = [e for e in ents if e.kind is Kind.WALL]
    constraints: list[Constraint] = []
    contradictions: list[Conflict] = []

    cands = _candidate_junctions(walls, cfg)
    n = len(walls)
    x0 = np.array([c for w in walls for c in (*w.a, *w.b)], float)
    prior = x0.copy()
    sig_along = np.zeros(2 * n)
    sig_across = np.zeros(2 * n)
    dirs = []
    for i, w in enumerate(walls):
        a, b = np.asarray(w.a), np.asarray(w.b)
        u = (b - a) / np.linalg.norm(b - a)
        dirs.append((u, np.array([-u[1], u[0]])))
        for k, which in enumerate("ab"):
            sig_along[2 * i + k], sig_across[2 * i + k] = _end_sigma(w, which, cfg)

    def pts(x):
        return x.reshape(n, 4)

    align_pairs = []
    if cfg.apply_alignment_priors:   # only pairs statistically indistinguishable from parallel / perpendicular
        for i in range(n):
            for j in range(i + 1, n):
                if walls[i].storey != walls[j].storey:
                    continue
                s_ij = math.sqrt(sum((w.beliefs["orientation"].sigma or 0.0) ** 2 for w in (walls[i], walls[j]))
                                 + cfg.angle_floor_rad ** 2)
                d = angle_diff(walls[i].angle % math.pi, walls[j].angle % math.pi)
                if d / s_ij < 1.0:
                    align_pairs.append((i, j, "par", s_ij))
                elif abs(d - math.pi / 2) / s_ij < 1.0:
                    align_pairs.append((i, j, "perp", s_ij))

    def residual(x):
        P = pts(x)
        r = []
        for i in range(n):
            u, nr = dirs[i]
            for k in (0, 1):
                d = P[i, 2 * k:2 * k + 2] - prior.reshape(n, 4)[i, 2 * k:2 * k + 2]
                r += [float(d @ u) / sig_along[2 * i + k], float(d @ nr) / sig_across[2 * i + k]]
        for (i, which, j, _, _) in cands:
            p = P[i, 0:2] if which == "a" else P[i, 2:4]
            a, b = P[j, 0:2], P[j, 2:4]
            e = b - a
            r.append(float((p[0] - a[0]) * (-e[1]) + (p[1] - a[1]) * e[0]) / float(np.linalg.norm(e)) / cfg.junction_sigma_m)
        for (i, j, kind_, s_ij) in align_pairs:      # fixed set decided from the initial geometry (constant length)
            ui = (P[i, 2:4] - P[i, 0:2]) / np.linalg.norm(P[i, 2:4] - P[i, 0:2])
            uj = (P[j, 2:4] - P[j, 0:2]) / np.linalg.norm(P[j, 2:4] - P[j, 0:2])
            r.append((float(ui[0] * uj[1] - ui[1] * uj[0]) if kind_ == "par" else float(ui @ uj)) / s_ij)
        return np.asarray(r, float)

    cost0 = float(0.5 * np.sum(residual(x0) ** 2)) if n else 0.0
    if n and cands:
        sol = least_squares(residual, x0, loss="soft_l1", f_scale=1.5, method="trf", max_nfev=200)
        xs, cost1, nfev = sol.x, float(sol.cost), int(sol.nfev)
        J = sol.jac
        cov = np.linalg.pinv(J.T @ J)
    else:
        xs, cost1, nfev, cov = x0.copy(), cost0, 0, None
    P, P0 = pts(xs), pts(x0)

    # posterior end sigmas + contradiction check
    constrained_ends = {(i, which) for (i, which, _, _, _) in cands}
    for i, w in enumerate(walls):
        for k, which in enumerate("ab"):
            idx = 4 * i + 2 * k
            moved = float(np.linalg.norm(P[i, 2 * k:2 * k + 2] - P0[i, 2 * k:2 * k + 2]))
            if cov is not None and (i, which) in constrained_ends:
                post = math.sqrt(max(0.0, (cov[idx, idx] + cov[idx + 1, idx + 1]) / 2.0))
                w.end_post[which] = post
            if k == 0:
                w.a = (float(P[i, 0]), float(P[i, 1]))
            else:
                w.b = (float(P[i, 2]), float(P[i, 3]))
            dv = P[i, 2 * k:2 * k + 2] - P0[i, 2 * k:2 * k + 2]
            u_, n_ = dirs[i]
            s_iso = moved / max(math.hypot(float(dv @ u_) / sig_along[2 * i + k], float(dv @ n_) / sig_across[2 * i + k]), 1e-12)
            if (i, which) in constrained_ends and moved > cfg.k * s_iso:
                contradictions.append(Conflict(
                    "constraint_contradiction", (w.entity_id,), w.evidence_ids,
                    f"junction at end '{which}' of {w.entity_id} needs it moved {moved:.3f} m "
                    f"({moved / s_iso:.1f} sigma) from its own evidence; evidence kept as alternative",
                    moved, moved / s_iso))
                w.alternatives.append({"kind": "pre_constraint_position", "end": which,
                                       "point": [float(P0[i, 2 * k]), float(P0[i, 2 * k + 1])], "moved_m": moved})
                w.beliefs["position"] = Belief(Status.CONFLICTING, w.beliefs["position"].support,
                                               w.beliefs["position"].value, w.beliefs["position"].sigma,
                                               w.beliefs["position"].confidence,
                                               "a topological constraint contradicts this wall's own evidence")

    byid = {w.entity_id: w for w in walls}
    junctions = []
    joined: dict[str, list[str]] = {}
    contradicted = {c.entities[0] for c in contradictions}
    for (i, which, j, typ, zn) in cands:
        wi, wj = walls[i], walls[j]
        junctions.append((wi.entity_id, which, wj.entity_id, typ))
        joined.setdefault(wi.entity_id, []).append(wj.entity_id)
        joined.setdefault(wj.entity_id, []).append(wi.entity_id)
        bad = wi.entity_id in contradicted
        constraints.append(Constraint(
            CKind.WALL_CONNECTIVITY if typ == "corner" else CKind.WALL_INTERSECTION, (wi.entity_id, wj.entity_id),
            tuple(sorted(set(wi.evidence_ids) | set(wj.evidence_ids))),
            CStatus.VIOLATED if bad else CStatus.SATISFIED,
            f"observed end '{which}' of {wi.entity_id} lies on {wj.entity_id} ({typ}), supported by its own evidence",
            applied=True, measured=None, normalized=zn))
    for w in walls:
        if w.entity_id in joined:
            others = sorted(set(joined[w.entity_id]))
            w.beliefs["topology"] = Belief(Status.GEOMETRICALLY_INFERRED, w.evidence_ids,
                                           note=f"junction evidence with {others}")
        elif not (w.end_a_observed or w.end_b_observed):
            w.beliefs["topology"] = Belief(Status.UNKNOWN, w.evidence_ids, note="no observed termination: connectivity unknown")
        else:
            w.beliefs["topology"] = Belief(Status.UNKNOWN, w.evidence_ids, note="no junction evidence for its observed ends")

    constraints += _continuity(walls, ents, cfg)
    constraints += _attachments(ents, byid, cfg)
    constraints += _vertical(ents, byid, storeys, cfg)
    constraints += [Constraint(CKind.COORDINATE_FRAME, (), c.evidence_ids, CStatus.VIOLATED, c.message)
                    for c in conflicts if c.code == "untransformable_frame"]
    return ConstraintReport(ents, constraints, contradictions, junctions,
                            {"cost_before": cost0, "cost_after": cost1, "nfev": float(nfev),
                             "n_walls": float(n), "n_junction_constraints": float(len(cands))})


# ------------------------------------------------------------------------------------- evaluated-only constraints

def _continuity(walls, ents, cfg: ConstraintConfig) -> list[Constraint]:
    out = []
    openings = [e for e in ents if e.kind in (Kind.DOOR, Kind.WINDOW, Kind.OPENING)]
    for i, wi in enumerate(walls):
        for wj in walls[i + 1:]:
            if wi.storey != wj.storey:
                continue
            s_i, s_j = (w.beliefs["orientation"].sigma or 0.0 for w in (wi, wj))
            if angle_diff(wi.angle % math.pi, wj.angle % math.pi) > cfg.k * math.sqrt(s_i ** 2 + s_j ** 2 + cfg.angle_floor_rad ** 2):
                continue
            fit = _fit_of(wi)
            if abs(signed(fit, wj.a)[1]) > cfg.k * math.sqrt((wi.beliefs["position"].sigma or 0) ** 2 + (wj.beliefs["position"].sigma or 0) ** 2 + cfg.floor_m ** 2):
                continue
            best = None
            for ea, sa_, pa in (("a", wi.end_a_observed, wi.a), ("b", wi.end_b_observed, wi.b)):
                for eb, sb_, pb in (("a", wj.end_a_observed, wj.a), ("b", wj.end_b_observed, wj.b)):
                    g = math.dist(pa, pb)
                    if best is None or g < best[0]:
                        best = (g, sa_, sb_, pa, pb)
            g, sa_, sb_, pa, pb = best
            if g > cfg.continuity_gap_m or g < 1e-9:
                continue
            ev = tuple(sorted(set(wi.evidence_ids) | set(wj.evidence_ids)))
            mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            spanned = [o.entity_id for o in openings if o.storey == wi.storey and
                       min(math.dist(mid, o.a), math.dist(mid, o.b)) <= max(g, 0.3) and math.dist(o.a, o.b) >= 0.5 * g]
            if spanned:
                st, msg = CStatus.SATISFIED, f"gap of {g:.2f} m between collinear walls is an opening ({spanned})"
            elif not (sa_ and sb_):
                st, msg = CStatus.UNDETERMINED, f"{g:.2f} m gap between collinear walls but an end is not observed (occlusion?)"
            else:
                st, msg = CStatus.VIOLATED, f"both ends observed, {g:.2f} m gap between collinear walls with no opening seen"
            out.append(Constraint(CKind.GEOMETRIC_CONTINUITY, (wi.entity_id, wj.entity_id), ev, st, msg, measured=g))
    return out


def _attachments(ents, byid, cfg: ConstraintConfig) -> list[Constraint]:
    out = []
    for o in ents:
        if o.kind not in (Kind.DOOR, Kind.WINDOW, Kind.OPENING):
            continue
        if o.host_id is None:
            why = ("ambiguous between " + ", ".join(o.host_alternatives)) if o.host_alternatives else "no wall within gate"
            out.append(Constraint(CKind.OPENING_ATTACHMENT, (o.entity_id,), o.evidence_ids, CStatus.UNDETERMINED,
                                  f"{o.entity_id} has no host wall: {why}"))
            continue
        if o.host_id not in byid:   # the host is a single-view (tentative) wall that was not admitted to the solve
            out.append(Constraint(CKind.OPENING_ATTACHMENT, (o.entity_id, o.host_id), o.evidence_ids, CStatus.UNDETERMINED,
                                  f"{o.entity_id}: host {o.host_id} is only tentatively supported (single view)"))
            continue
        h = byid[o.host_id]
        fit = _fit_of(h)
        ta, tb = signed(fit, o.a)[0], signed(fit, o.b)[0]
        lo, hi = min(ta, tb), max(ta, tb)
        tol = cfg.k * math.sqrt((o.sigma_a or cfg.floor_m) ** 2 + cfg.floor_m ** 2)
        over_lo, over_hi = max(0.0, fit.t_lo - lo), max(0.0, hi - fit.t_hi)
        over = max(over_lo, over_hi)
        unobs = (over_lo > tol and not h.end_a_observed) or (over_hi > tol and not h.end_b_observed)
        ev = tuple(sorted(set(o.evidence_ids) | set(h.evidence_ids)))
        if over <= tol:
            st, msg = CStatus.SATISFIED, f"{o.entity_id} lies within host {h.entity_id}"
        elif unobs:
            st, msg = CStatus.UNDETERMINED, f"{o.entity_id} extends {over:.2f} m past the observed part of {h.entity_id}, whose end is unobserved"
        else:
            st, msg = CStatus.VIOLATED, f"{o.entity_id} extends {over:.2f} m beyond host {h.entity_id} (both ends observed)"
        out.append(Constraint(CKind.OPENING_ATTACHMENT, (o.entity_id, h.entity_id), ev, st, msg, measured=over))
        width = o.length
        if width > h.length + tol and h.end_a_observed and h.end_b_observed:
            out.append(Constraint(CKind.OPENING_DIMENSIONS, (o.entity_id, h.entity_id), ev, CStatus.VIOLATED,
                                  f"opening width {width:.2f} m exceeds host length {h.length:.2f} m", measured=width))
        elif o.z0 is not None and o.z1 is not None and h.z0 is not None and h.z1 is not None:
            ok = (o.z1 - o.z0) <= (h.z1 - h.z0) + cfg.z_tolerance_m
            out.append(Constraint(CKind.OPENING_DIMENSIONS, (o.entity_id, h.entity_id), ev,
                                  CStatus.SATISFIED if ok else CStatus.VIOLATED,
                                  f"opening height {o.z1 - o.z0:.2f} m vs host height {h.z1 - h.z0:.2f} m",
                                  measured=o.z1 - o.z0))
        else:
            out.append(Constraint(CKind.OPENING_DIMENSIONS, (o.entity_id, h.entity_id), ev, CStatus.UNDETERMINED,
                                  "heights not observed: opening height vs host height cannot be checked"))
    return out


def _vertical(ents, byid, storeys: Mapping[str, float] | None, cfg: ConstraintConfig) -> list[Constraint]:
    out = []
    for e in ents:
        if e.kind is Kind.WALL:
            if e.z0 is None or e.z1 is None or not storeys or e.storey not in storeys:
                out.append(Constraint(CKind.FLOOR_CEILING, (e.entity_id,), e.evidence_ids, CStatus.UNDETERMINED,
                                      f"{e.entity_id}: wall z-extent or storey floor height not observed"))
                continue
            floor = storeys[e.storey]
            ok = abs(e.z0 - floor) <= cfg.z_tolerance_m and e.z1 > e.z0
            out.append(Constraint(CKind.FLOOR_CEILING, (e.entity_id,), e.evidence_ids,
                                  CStatus.SATISFIED if ok else CStatus.VIOLATED,
                                  f"wall base z={e.z0:.2f} m vs storey {e.storey} floor {floor:.2f} m",
                                  measured=e.z0 - floor))
        elif e.kind is Kind.STAIR:
            if e.z0 is None or e.z1 is None or not storeys or len(storeys) < 2:
                out.append(Constraint(CKind.STAIR_CONNECTIVITY, (e.entity_id,), e.evidence_ids, CStatus.UNDETERMINED,
                                      f"{e.entity_id}: stair z-extent or the storeys it should join are not known"))
                continue
            lo = min(storeys, key=lambda s: abs(storeys[s] - e.z0))
            hi = min(storeys, key=lambda s: abs(storeys[s] - e.z1))
            ok = (lo != hi and abs(storeys[lo] - e.z0) <= cfg.z_tolerance_m
                  and abs(storeys[hi] - e.z1) <= cfg.z_tolerance_m and storeys[hi] > storeys[lo])
            out.append(Constraint(CKind.STAIR_CONNECTIVITY, (e.entity_id,), e.evidence_ids,
                                  CStatus.SATISFIED if ok else CStatus.VIOLATED,
                                  f"stair spans z {e.z0:.2f}->{e.z1:.2f} m; nearest storeys {lo}->{hi}"
                                  + ("" if ok else " (does not connect two distinct storeys)"),
                                  measured=e.z1 - e.z0))
    return out


# ------------------------------------------------------------------------------------------ alignment & objects

def evaluate_alignment(entities: Sequence[SpatialEntity], groups: Sequence[Sequence[str]],
                       cfg: ConstraintConfig = ConstraintConfig()) -> list[Constraint]:
    """Measure (never impose) parallelism / perpendicularity between walls that bound the same space.
    A pair not aligned is reported UNDETERMINED with its measured angle: non-Manhattan layouts are legitimate."""
    byid = {e.entity_id: e for e in entities}
    out, seen = [], set()
    for grp in groups:
        for ai, aid in enumerate(grp):
            for bid in grp[ai + 1:]:
                if (aid, bid) in seen or aid not in byid or bid not in byid:
                    continue
                seen.add((aid, bid))
                wa, wb = byid[aid], byid[bid]
                s = math.sqrt(sum((w.beliefs["orientation"].sigma or 0.0) ** 2 for w in (wa, wb)) + cfg.angle_floor_rad ** 2)
                d = angle_diff(wa.angle % math.pi, wb.angle % math.pi)
                ev = tuple(sorted(set(wa.evidence_ids) | set(wb.evidence_ids)))
                if d / s <= cfg.k:
                    out.append(Constraint(CKind.WALL_ALIGNMENT, (aid, bid), ev, CStatus.SATISFIED,
                                          f"parallel within {math.degrees(d):.1f} deg ({d / s:.1f} sigma)",
                                          applied=cfg.apply_alignment_priors, measured=math.degrees(d), normalized=d / s))
                elif abs(d - math.pi / 2) / s <= cfg.k:
                    out.append(Constraint(CKind.WALL_ALIGNMENT, (aid, bid), ev, CStatus.SATISFIED,
                                          f"perpendicular within {math.degrees(abs(d - math.pi / 2)):.1f} deg",
                                          applied=cfg.apply_alignment_priors, measured=math.degrees(d),
                                          normalized=abs(d - math.pi / 2) / s))
                else:
                    out.append(Constraint(CKind.WALL_ALIGNMENT, (aid, bid), ev, CStatus.UNDETERMINED,
                                          f"neither parallel nor perpendicular ({math.degrees(d):.1f} deg): "
                                          "no alignment imposed", measured=math.degrees(d)))
    return out


def point_in_polygon(pt, poly) -> bool:
    x, y = pt
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def check_objects(objects: Mapping[str, tuple[float, float, float, float]], rooms: Mapping[str, Sequence[tuple[float, float]]],
                  walls: Sequence[SpatialEntity], adjacency_m: float = 0.15) -> list[Constraint]:
    """Containment (object footprint inside a room polygon) and adjacency (footprint touches a wall).
    Footprints are (minx, miny, maxx, maxy). With no reconstructed room nothing is claimed (UNDETERMINED)."""
    out = []
    for oid, (x0, y0, x1, y1) in sorted(objects.items()):
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        home = [rid for rid, poly in rooms.items() if all(point_in_polygon(c, poly) for c in corners)]
        if not rooms:
            out.append(Constraint(CKind.OBJECT_CONTAINMENT, (oid,), (), CStatus.UNDETERMINED, f"{oid}: no room reconstructed"))
        elif home:
            out.append(Constraint(CKind.OBJECT_CONTAINMENT, (oid, home[0]), (), CStatus.SATISFIED, f"{oid} inside {home[0]}"))
        else:
            out.append(Constraint(CKind.OBJECT_CONTAINMENT, (oid,), (), CStatus.VIOLATED,
                                  f"{oid} footprint is not inside any reconstructed room"))
        for w in walls:
            if w.kind is not Kind.WALL:
                continue
            fit = _fit_of(w)
            best = min(
                (abs(signed(fit, c)[1]) if fit.t_lo <= signed(fit, c)[0] <= fit.t_hi else float("inf")) for c in corners)
            if best <= adjacency_m:
                out.append(Constraint(CKind.OBJECT_ADJACENCY, (oid, w.entity_id), w.evidence_ids, CStatus.SATISFIED,
                                      f"{oid} is {best:.2f} m from {w.entity_id}", measured=best))
    return out
