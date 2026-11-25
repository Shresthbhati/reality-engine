"""Evidence-to-world association for structural entities (walls, doors, windows).

Question answered: "which observations, across images, are the same physical wall / opening, which are
different, which disagree, and which are too thin to establish anything?"

Method (deterministic, no RNG):
  * Walls are clustered by a greedy, strongest-evidence-first assignment against the CURRENT FIT of each
    cluster, never pairwise between members -- so A~B and B~C can never chain A with C (the failure of the legacy
    union-find resolver). Every gate is expressed in sigmas, with a small floor in quadrature for un-modelled
    systematic lift error, so noisy (monocular) observations match loosely and precise ones tightly.
  * Gates: orientation, perpendicular offset of both end points, along-axis overlap-or-small-gap, and "one image
    cannot contribute two overlapping extents to one wall".
  * If an observation passes the gates for two clusters with scores inside `ambiguity_margin` it is NOT merged:
    it stays a provisional entity, state AMBIGUOUS, with the candidates recorded.
  * End points are lower bounds unless the end was actually seen as a termination (occlusion / image border).
  * Doors/windows attach to a host wall by line proximity + along-wall containment; several plausible hosts =>
    no host is chosen, the alternatives are kept. Conflicting classes on one opening => one entity with a
    CONFLICTING semantic belief and the alternatives, not an average.
  * Nothing is manufactured: every entity lists the evidence ids that support it; an entity with fewer than
    `min_images_to_establish` distinct images is returned as tentative (existence WEAKLY_SUPPORTED).

Thresholds are starting points (floors, k-sigma gates), NOT tuned on any dataset; `benchmarks/spatial` reports
what they produce on synthetic ground truth.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from world_ir.observations import AssociationState, Visibility

from .model import (
    UNKNOWN_BELIEF,
    AssociationRecord,
    Belief,
    Conflict,
    Kind,
    SpatialEntity,
    Status,
    StructuralObservation,
    angle_diff,
)


@dataclass(frozen=True)
class AssociationConfig:
    k_off: float = 3.5                 # offset gate, sigmas (3.5 => ~5e-4 false-split chance per criterion)
    k_ang: float = 3.5                 # orientation gate, sigmas
    offset_floor_m: float = 0.03       # un-modelled systematic lift error, added in quadrature
    angle_floor_rad: float = 0.02      # ~1.1 degrees
    gap_tol_m: float = 0.15            # collinear segments farther apart than this are different entities
    ambiguity_margin: float = 2.0      # chi^2 units: best vs second-best candidate
    min_images_to_establish: int = 2
    opening_offset_floor_m: float = 0.10   # openings are lifted less precisely than wall planes
    opening_min_iou: float = 0.30      # along-wall interval overlap to be the same opening
    max_wall_thickness_m: float = 0.60  # parallel neighbours closer than this are a conflict, not two walls
    max_reassign_iterations: int = 4


# --------------------------------------------------------------------------------------------- line fitting

@dataclass(frozen=True)
class LineFit:
    c: tuple[float, float]          # weighted centroid
    u: tuple[float, float]          # unit direction
    theta: float                    # [0, pi)
    sigma_d: float                  # 1-sigma of the perpendicular offset
    sigma_theta: float | None    # None when the extent along the line is zero
    t_lo: float
    t_hi: float
    reduced_chi: float | None    # sqrt(sum w r^2 / dof); ~1 when the stated sigmas are right
    n_points: int


def fit_line(obs: Sequence[StructuralObservation]) -> LineFit:
    """Weighted total-least-squares line through the end points (weight 1/sigma^2).

    sigma_d = 1/sqrt(sum w); sigma_theta = 1/sqrt(sum w t^2) (t = along-axis distance from the centroid): the
    standard first-order covariance of an offset/angle line fit under independent end-point errors."""
    pts, w = [], []
    for o in obs:
        for p in (o.p0, o.p1):
            pts.append(p)
            w.append(1.0 / (o.sigma_m ** 2))
    P, W = np.asarray(pts, float), np.asarray(w, float)
    c = (W[:, None] * P).sum(0) / W.sum()
    D = P - c
    M = (W[:, None, None] * (D[:, :, None] * D[:, None, :])).sum(0) / W.sum()
    _, vecs = np.linalg.eigh(M)
    u = vecs[:, 1]
    if u[0] < 0 or (abs(u[0]) < 1e-12 and u[1] < 0):
        u = -u
    n = np.array([-u[1], u[0]])
    t, r = D @ u, D @ n
    swt2 = float((W * t * t).sum())
    dof = len(pts) - 2
    chi = math.sqrt(float((W * r * r).sum()) / dof) if dof > 0 else None
    return LineFit(
        c=(float(c[0]), float(c[1])), u=(float(u[0]), float(u[1])),
        theta=float(math.atan2(u[1], u[0]) % math.pi), sigma_d=1.0 / math.sqrt(float(W.sum())),
        sigma_theta=(1.0 / math.sqrt(swt2)) if swt2 > 1e-12 else None,
        t_lo=float(t.min()), t_hi=float(t.max()), reduced_chi=chi, n_points=len(pts))


def line_through(a, b) -> LineFit:
    """Exact geometric line through two points (unit sigma; only direction/extent are used)."""
    return fit_line([StructuralObservation("_", "_", Kind.WALL, tuple(a), tuple(b), 1.0)])


def signed(fit: LineFit, p) -> tuple[float, float]:
    """(along, perpendicular) coordinates of p relative to the fit."""
    dx, dy = p[0] - fit.c[0], p[1] - fit.c[1]
    return dx * fit.u[0] + dy * fit.u[1], -dx * fit.u[1] + dy * fit.u[0]


def _span(o: StructuralObservation, fit: LineFit) -> tuple[float, float]:
    return tuple(sorted((signed(fit, o.p0)[0], signed(fit, o.p1)[0])))  # type: ignore[return-value]


@dataclass(frozen=True)
class _Gate:
    passed: bool
    z_off: float
    z_ang: float
    gap: float
    reason: str

    @property
    def score(self) -> float:
        return self.z_off ** 2 + self.z_ang ** 2


def _gate(o: StructuralObservation, fit: LineFit, members: Sequence[StructuralObservation],
          cfg: AssociationConfig) -> _Gate:
    sig_th_o = math.sqrt(2.0) * o.sigma_m / o.length
    dth = angle_diff(math.atan2(o.p1[1] - o.p0[1], o.p1[0] - o.p0[0]) % math.pi, fit.theta)
    s_ang = math.sqrt(sig_th_o ** 2 + (fit.sigma_theta or 0.0) ** 2 + cfg.angle_floor_rad ** 2)
    z_ang = dth / s_ang
    # the fitted line's own uncertainty at each end point: offset sigma plus angle sigma times the lever arm
    (ta_, da), (tb_, db) = signed(fit, o.p0), signed(fit, o.p1)
    st = fit.sigma_theta or 0.0
    z_off = max(abs(d) / math.sqrt(o.sigma_m ** 2 + fit.sigma_d ** 2 + (t * st) ** 2 + cfg.offset_floor_m ** 2)
                for t, d in ((ta_, da), (tb_, db)))
    lo, hi = _span(o, fit)
    gap = max(0.0, lo - fit.t_hi, fit.t_lo - hi)
    if z_ang > cfg.k_ang:
        return _Gate(False, z_off, z_ang, gap, f"orientation differs by {math.degrees(dth):.1f} deg ({z_ang:.1f} sigma)")
    if z_off > cfg.k_off:
        return _Gate(False, z_off, z_ang, gap, f"offset {max(abs(da), abs(db)):.3f} m ({z_off:.1f} sigma)")
    if gap > cfg.gap_tol_m + 3.0 * o.sigma_m:
        return _Gate(False, z_off, z_ang, gap, f"collinear but {gap:.2f} m apart (distinct segments)")
    for m in members:   # one image cannot show the same wall twice over the same stretch
        if m.image_id == o.image_id:
            mlo, mhi = _span(m, fit)
            if min(hi, mhi) - max(lo, mlo) > 2.0 * o.sigma_m:
                return _Gate(False, z_off, z_ang, gap, "same image already supplies this stretch")
    return _Gate(True, z_off, z_ang, gap, "ok")


# -------------------------------------------------------------------------------------------- coordinate frames

def to_common_frame(
    observations: Sequence[StructuralObservation], target_frame: str,
    transforms: Mapping[str, tuple[float, float, float]] | None = None,
) -> tuple[list[StructuralObservation], list[Conflict]]:
    """Map every observation into `target_frame` with a 2D rigid transform {frame: (theta_rad, tx, ty)} meaning
    p_target = R(theta) p_frame + t. An observation in a frame with no transform is NOT guessed into the target:
    it is returned in a Conflict and left out. Transform uncertainty is not propagated (declared limitation)."""
    transforms = transforms or {}
    out, missing = [], {}
    for o in observations:
        if o.frame_id == target_frame:
            out.append(o)
            continue
        tr = transforms.get(o.frame_id)
        if tr is None:
            missing.setdefault(o.frame_id, []).append(o.evidence_id)
            continue
        th, tx, ty = tr
        cs, sn = math.cos(th), math.sin(th)

        def f(p):
            return (cs * p[0] - sn * p[1] + tx, sn * p[0] + cs * p[1] + ty)

        out.append(StructuralObservation(**{**o.__dict__, "p0": f(o.p0), "p1": f(o.p1), "frame_id": target_frame}))
    conflicts = [Conflict("untransformable_frame", (), tuple(sorted(ids)),
                          f"{len(ids)} observation(s) are in frame {fr!r} with no transform to {target_frame!r}; "
                          "excluded, not guessed") for fr, ids in sorted(missing.items())]
    return out, conflicts


# ------------------------------------------------------------------------------------------------ association

@dataclass
class AssociationResult:
    entities: list[SpatialEntity]
    records: list[AssociationRecord]
    conflicts: list[Conflict]
    unassociated: list[str] = field(default_factory=list)   # evidence ids that fed no entity
    config: AssociationConfig = field(default_factory=AssociationConfig)

    def established(self) -> list[SpatialEntity]:
        return [e for e in self.entities if len(e.image_ids) >= self.config.min_images_to_establish]

    def tentative(self) -> list[SpatialEntity]:
        return [e for e in self.entities if len(e.image_ids) < self.config.min_images_to_establish]

    def by_id(self) -> dict[str, SpatialEntity]:
        return {e.entity_id: e for e in self.entities}


def _eid(prefix: str, evidence_ids: Sequence[str]) -> str:
    return f"{prefix}-{hashlib.sha1('|'.join(sorted(evidence_ids)).encode()).hexdigest()[:10]}"


def _view_confidences(members: Sequence[StructuralObservation]) -> list[float | None]:
    """One confidence per distinct image (max over that image's detections): views, not detections, are the
    independent units, so a duplicate detection in one image cannot inflate belief."""
    per: dict[str, float | None] = {}
    for o in members:
        cur = per.get(o.image_id)
        per[o.image_id] = o.label_confidence if cur is None else (cur if o.label_confidence is None
                                                                 else max(cur, o.label_confidence))
    return list(per.values())


def _noisy_or(values: Sequence[float | None]) -> tuple[float | None, str]:
    given = [v for v in values if v is not None]
    if not given:
        return None, "no producer confidence supplied"
    p = 1.0
    for v in given:
        p *= 1.0 - min(max(v, 0.0), 1.0)
    lb = f" (lower bound: {len(given)} of {len(values)} views supplied confidence)" if len(given) < len(values) else ""
    return 1.0 - p, "noisy-OR over independent views" + lb


def associate_structure(
    observations: Sequence[StructuralObservation],
    config: AssociationConfig = AssociationConfig(),
    *, target_frame: str = "world", frame_transforms: Mapping[str, tuple[float, float, float]] | None = None,
) -> AssociationResult:
    cfg = config
    ids = [o.evidence_id for o in observations]
    if len(set(ids)) != len(ids):
        raise ValueError("evidence ids must be unique")
    obs, conflicts = to_common_frame(observations, target_frame, frame_transforms)
    unassoc = sorted(set(ids) - {o.evidence_id for o in obs})
    records: list[AssociationRecord] = []

    walls: list[SpatialEntity] = []
    stairs: list[SpatialEntity] = []
    for kind, bucket in ((Kind.WALL, walls), (Kind.STAIR, stairs)):   # both are linear elements fitted the same way
        by_storey: dict[str | None, list[StructuralObservation]] = {}
        for o in obs:
            if o.kind is kind:
                by_storey.setdefault(o.storey, []).append(o)
        for storey in sorted(by_storey, key=lambda s: s or ""):
            ents, recs = _cluster_walls(by_storey[storey], storey, cfg, kind)
            bucket += ents
            records += recs
    conflicts += _near_duplicate_conflicts(walls, cfg)

    openings, orecs, oconf = _associate_openings(
        [o for o in obs if o.kind in (Kind.DOOR, Kind.WINDOW, Kind.OPENING)], walls, cfg)
    records += orecs
    conflicts += oconf
    records.sort(key=lambda r: r.evidence_id)
    return AssociationResult(walls + stairs + openings, records, conflicts, unassoc, cfg)


# --- walls

def _cluster_walls(wobs: Sequence[StructuralObservation], storey: str | None, cfg: AssociationConfig,
                   kind: Kind = Kind.WALL):
    order = sorted(wobs, key=lambda o: (-(o.length / o.sigma_m), o.evidence_id))
    clusters: list[list[StructuralObservation]] = []
    ambiguous: dict[str, tuple[str, ...]] = {}
    gate_of: dict[str, _Gate] = {}

    fit_cache: dict[tuple[str, ...], LineFit] = {}

    def fit_of(ms):
        # a cluster's fit only changes when its membership does; memoised by the (sorted) evidence ids it contains
        key = tuple(sorted(m.evidence_id for m in ms))
        f = fit_cache.get(key)
        if f is None:
            f = fit_cache[key] = fit_line(ms)
        return f

    def candidates(o, excl=None):
        out = []
        for ci, members in enumerate(clusters):
            ms = [m for m in members if m is not excl]
            if ms:
                g = _gate(o, fit_of(ms), ms, cfg)
                if g.passed:
                    out.append((g.score, ci, g))
        return sorted(out, key=lambda x: (x[0], x[1]))

    def is_ambiguous(cands):
        return len(cands) >= 2 and cands[1][0] - cands[0][0] < cfg.ambiguity_margin

    for o in order:
        cands = candidates(o)
        if not cands:
            clusters.append([o])
        elif is_ambiguous(cands):
            ambiguous[o.evidence_id] = tuple(f"cluster{c[1]}" for c in cands)
            clusters.append([o])
        else:
            clusters[cands[0][1]].append(o)
            gate_of[o.evidence_id] = cands[0][2]

    for _ in range(cfg.max_reassign_iterations):   # settling pass; stops when nothing moves
        moved = False
        for o in order:
            cur = next(i for i, m in enumerate(clusters) if o in m)
            cands = candidates(o, excl=o)
            if not cands or is_ambiguous(cands):
                continue
            best = cands[0][1]
            if best != cur:
                clusters[cur].remove(o)
                clusters[best].append(o)
                gate_of[o.evidence_id] = cands[0][2]
                ambiguous.pop(o.evidence_id, None)
                moved = True
        clusters = [c for c in clusters if c]
        if not moved:
            break

    entities, records = [], []
    for members in clusters:
        ent = _build_wall(members, storey, cfg, kind)
        entities.append(ent)
        multi = len(ent.image_ids) >= cfg.min_images_to_establish
        for m in members:
            g = gate_of.get(m.evidence_id)
            if m.evidence_id in ambiguous:
                st, why = AssociationState.AMBIGUOUS, "fits several clusters within the ambiguity margin; not merged"
            elif m.visibility is Visibility.OCCLUDED:
                st, why = AssociationState.IMAGE_OCCLUSION, "occluded view: extent is a lower bound"
            elif not multi:
                st, why = AssociationState.INSUFFICIENT_EVIDENCE, "single view: insufficient to establish the entity"
            elif m.visibility is Visibility.PARTIAL:
                st, why = AssociationState.PARTIAL_OVERLAP, "partially visible; joined by orientation/offset/overlap"
            else:
                st, why = AssociationState.SAME_ENTITY_CONFIRMED, "orientation, offset and overlap gates passed"
            records.append(AssociationRecord(
                m.evidence_id, ent.entity_id, st.value, why, candidates=ambiguous.get(m.evidence_id, ()),
                normalized_offset=g.z_off if g else None, normalized_angle=g.z_ang if g else None))
    return entities, records


def _extreme(points: list[tuple[float, bool, float]], take_min: bool) -> tuple[float, bool, float]:
    """Merge the end points defining one extreme of the extent. points = (t, seen, sigma). If any end near the
    extreme was actually seen, only seen ends are averaged; otherwise the extreme is a lower bound (not seen)."""
    ext = min(points) if take_min else max(points)
    span = max(p[0] for p in points) - min(p[0] for p in points)
    # never let the two extremes' merge windows swallow the extent (very noisy, very short elements)
    near = [p for p in points if abs(p[0] - ext[0]) <= min(3.0 * max(p[2], ext[2]), 0.25 * span)]
    seen = [p for p in near if p[1]]
    use = seen or near
    w = [1.0 / p[2] ** 2 for p in use]
    return sum(wi * p[0] for wi, p in zip(w, use)) / sum(w), bool(seen), 1.0 / math.sqrt(sum(w))


def _build_wall(members: Sequence[StructuralObservation], storey, cfg: AssociationConfig,
                kind: Kind = Kind.WALL) -> SpatialEntity:
    fit = fit_line(members)
    pts = [(signed(fit, p)[0], seen, o.sigma_m)
           for o in members for p, seen in ((o.p0, o.end0_observed), (o.p1, o.end1_observed))]
    ta, a_obs, sa = _extreme(pts, True)
    tb, b_obs, sb = _extreme(pts, False)
    nrm = (-fit.u[1], fit.u[0])
    a = (fit.c[0] + ta * fit.u[0], fit.c[1] + ta * fit.u[1])
    b = (fit.c[0] + tb * fit.u[0], fit.c[1] + tb * fit.u[1])
    ev = tuple(sorted(o.evidence_id for o in members))
    imgs = tuple(sorted({o.image_id for o in members}))
    zs0 = [o.z0 for o in members if o.z0 is not None]
    zs1 = [o.z1 for o in members if o.z1 is not None]
    n_views = len(imgs)
    conf, conf_note = _noisy_or(_view_confidences(members))
    established = n_views >= cfg.min_images_to_establish
    exist = (Belief(Status.DIRECTLY_OBSERVED, ev, confidence=conf, note=f"{n_views} views; {conf_note}") if established
             else Belief(Status.WEAKLY_SUPPORTED, ev, note="single view: insufficient to establish this entity"))
    pos = Belief(Status.DIRECTLY_OBSERVED, ev, value=fit.c[0] * nrm[0] + fit.c[1] * nrm[1], sigma=fit.sigma_d,
                 note="perpendicular offset of the fitted line from the origin (m)")
    length = math.dist(a, b)
    dim = (Belief(Status.DIRECTLY_OBSERVED, ev, value=length, sigma=math.hypot(sa, sb),
                  note="length between two observed terminations; height is not covered by this belief")
           if a_obs and b_obs else
           Belief(Status.WEAKLY_SUPPORTED, ev, value=length,
                  note="lower bound: an end is occluded / cut by the image border, true length >= value"))
    ori = (Belief(Status.DIRECTLY_OBSERVED, ev, value=fit.theta, sigma=fit.sigma_theta)
           if fit.sigma_theta is not None else Belief(Status.UNKNOWN, ev, note="no extent along the line"))
    if fit.reduced_chi is not None and fit.reduced_chi > 3.0:
        geo = Belief(Status.CONFLICTING, ev, value=fit.reduced_chi,
                     note="views disagree with each other beyond their stated sigmas")
    elif n_views >= 2:
        geo = Belief(Status.GEOMETRICALLY_INFERRED, ev, value=fit.reduced_chi, note="multi-view line fit")
    else:
        geo = Belief(Status.DIRECTLY_OBSERVED, ev, note="single-view line")
    sem = (Belief(Status.SEMANTICALLY_INFERRED, ev, confidence=conf, note="class from detector; " + conf_note)
           if established else Belief(Status.WEAKLY_SUPPORTED, ev, note="class from a single view"))
    return SpatialEntity(
        entity_id=_eid(kind.value, ev), kind=kind, a=a, b=b,
        z0=float(np.median(zs0)) if zs0 else None, z1=float(np.median(zs1)) if zs1 else None,
        evidence_ids=ev, image_ids=imgs,
        beliefs={"existence": exist, "position": pos, "dimensions": dim, "orientation": ori, "geometry": geo,
                 "topology": UNKNOWN_BELIEF, "semantic": sem},
        sigma_a=sa, sigma_b=sb, end_a_observed=a_obs, end_b_observed=b_obs, storey=storey,
        partially_visible=any(o.visibility is Visibility.PARTIAL for o in members),
        occluded=any(o.visibility is Visibility.OCCLUDED for o in members))


def _near_duplicate_conflicts(walls: list[SpatialEntity], cfg: AssociationConfig) -> list[Conflict]:
    out = []
    for i, w1 in enumerate(walls):
        for w2 in walls[i + 1:]:
            if w1.storey != w2.storey:
                continue
            dth = angle_diff(w1.angle % math.pi, w2.angle % math.pi)
            s1, s2 = (w.beliefs["orientation"].sigma or 0.0 for w in (w1, w2))
            if dth > cfg.k_ang * math.sqrt(s1 ** 2 + s2 ** 2 + cfg.angle_floor_rad ** 2):
                continue
            fit = line_through(w1.a, w1.b)
            (ta, da), (tb, db) = signed(fit, w2.a), signed(fit, w2.b)
            off = (abs(da) + abs(db)) / 2.0
            ov = min(max(ta, tb), fit.t_hi) - max(min(ta, tb), fit.t_lo)
            if off <= cfg.max_wall_thickness_m and ov > 0.5 * min(w1.length, w2.length):
                s = math.sqrt(sum((w.beliefs["position"].sigma or 0.0) ** 2 for w in (w1, w2)) + cfg.offset_floor_m ** 2)
                out.append(Conflict(
                    "parallel_near_duplicate", (w1.entity_id, w2.entity_id),
                    tuple(sorted(w1.evidence_ids + w2.evidence_ids)),
                    f"{w1.entity_id} and {w2.entity_id} are parallel, {off:.3f} m apart and overlap {ov:.2f} m: "
                    "either two faces of one thick wall or conflicting measurements of one wall", off, off / s))
                for w, other in ((w1, w2), (w2, w1)):
                    p = w.beliefs["position"]
                    w.beliefs["position"] = Belief(Status.CONFLICTING, p.support, p.value, p.sigma, p.confidence,
                                                   f"{p.note}; conflicts with {other.entity_id}")
                    w.alternatives.append({"kind": "parallel_near_duplicate", "other": other.entity_id,
                                           "offset_m": off})
    return out


# --- openings

def _associate_openings(oobs: Sequence[StructuralObservation], walls: list[SpatialEntity], cfg: AssociationConfig):
    records: list[AssociationRecord] = []
    conflicts: list[Conflict] = []
    hosted: dict[tuple[str | None, str | None], list[StructuralObservation]] = {}
    host_alts: dict[str, tuple[str, ...]] = {}
    wall_fit = {w.entity_id: line_through(w.a, w.b) for w in walls}
    for o in sorted(oobs, key=lambda o: o.evidence_id):
        cands = []
        for w in walls:
            if w.storey != o.storey:
                continue
            fit = wall_fit[w.entity_id]
            (ta, da), (tb, db) = signed(fit, o.p0), signed(fit, o.p1)
            s_off = math.sqrt(o.sigma_m ** 2 + (w.beliefs["position"].sigma or 0.0) ** 2 + cfg.opening_offset_floor_m ** 2)
            z = max(abs(da), abs(db)) / s_off
            # no separate orientation gate: both end points lying within the host's offset band already bounds the
            # opening's angle, and a ~1 m opening's own angle is far noisier than its end-point offsets
            if z > cfg.k_off:
                continue
            lo, hi, tol = min(ta, tb), max(ta, tb), 3.0 * o.sigma_m
            # beyond an end only acceptable when that end of the wall was not observed (wall extent is a lower bound)
            if (lo >= fit.t_lo - tol or not w.end_a_observed) and (hi <= fit.t_hi + tol or not w.end_b_observed):
                cands.append((z ** 2, w.entity_id))
        cands.sort()
        if not cands:
            key = (None, o.storey)
        elif len(cands) >= 2 and cands[1][0] - cands[0][0] < cfg.ambiguity_margin:
            key, host_alts[o.evidence_id] = (None, o.storey), tuple(c[1] for c in cands)
        else:
            key = (cands[0][1], o.storey)
        hosted.setdefault(key, []).append(o)

    byw = {w.entity_id: w for w in walls}
    entities: list[SpatialEntity] = []
    for (host, _), group in sorted(hosted.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")):
        hw = byw.get(host)
        for members in _group_openings(group, hw, wall_fit.get(host) if host else None, cfg):
            ent = _build_opening(members, hw, wall_fit.get(host) if host else None, host_alts, cfg)
            entities.append(ent)
            multi = len(ent.image_ids) >= cfg.min_images_to_establish
            for m in members:
                if host is None and m.evidence_id in host_alts:
                    st, why = AssociationState.AMBIGUOUS, "several plausible host walls; none chosen"
                elif host is None:
                    st, why = AssociationState.INSUFFICIENT_EVIDENCE, "no host wall within gate: attachment unknown"
                elif not multi:
                    st, why = AssociationState.INSUFFICIENT_EVIDENCE, "single view: insufficient to establish the opening"
                else:
                    st, why = AssociationState.SAME_ENTITY_CONFIRMED, f"attached to host {host}"
                records.append(AssociationRecord(m.evidence_id, ent.entity_id, st.value, why,
                                                 candidates=host_alts.get(m.evidence_id, ())))
            if len({m.kind for m in members}) > 1:
                conflicts.append(Conflict(
                    "opening_class_conflict", (ent.entity_id,), tuple(sorted(m.evidence_id for m in members)),
                    f"views classify this opening as {sorted({m.kind.value for m in members})}"))
    return entities, records, conflicts


def _group_openings(group, host, host_fit, cfg: AssociationConfig):
    """Cluster one host's opening observations by along-wall interval IoU (strongest first). Without a host the
    group must also pass the collinearity gate (there is no wall line to project onto)."""
    groups: list[list[StructuralObservation]] = []
    for o in sorted(group, key=lambda o: (-o.length, o.evidence_id)):
        best = None
        for g in groups:
            if any(m.image_id == o.image_id for m in g):
                continue
            f = host_fit or fit_line(g)
            if host_fit is None and not _gate(o, f, g, cfg).passed:
                continue
            spans = [_span(m, f) for m in g]
            lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
            so = _span(o, f)
            inter = min(hi, so[1]) - max(lo, so[0])
            v = inter / (max(hi, so[1]) - min(lo, so[0])) if inter > 0 else 0.0
            if v >= cfg.opening_min_iou and (best is None or v > best[0]):
                best = (v, g)
        if best:
            best[1].append(o)
        else:
            groups.append([o])
    return groups


def _build_opening(members: Sequence[StructuralObservation], host: SpatialEntity | None, host_fit,
                   host_alts: dict[str, tuple[str, ...]], cfg: AssociationConfig) -> SpatialEntity:
    ref = host_fit or fit_line(members[:1])
    pts = [(signed(ref, p)[0], seen, o.sigma_m)
           for o in members for p, seen in ((o.p0, o.end0_observed), (o.p1, o.end1_observed))]
    ta, a_obs, sa = _extreme(pts, True)
    tb, b_obs, sb = _extreme(pts, False)
    off = 0.0   # an attached opening lies ON its host's line; an unattached one on its own first view's line
    nrm = (-ref.u[1], ref.u[0])

    def at(t):
        return (ref.c[0] + t * ref.u[0] + off * nrm[0], ref.c[1] + t * ref.u[1] + off * nrm[1])

    a, b = at(ta), at(tb)
    ev = tuple(sorted(o.evidence_id for o in members))
    imgs = tuple(sorted({o.image_id for o in members}))
    votes: dict[Kind, int] = {}
    for im in imgs:
        ks = [o.kind for o in members if o.image_id == im]
        k = max(sorted(set(ks), key=lambda x: x.value), key=ks.count)
        votes[k] = votes.get(k, 0) + 1
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0].value))
    conflict_cls = len(top) > 1
    kind = Kind.OPENING if (conflict_cls and top[0][1] == top[1][1]) else top[0][0]
    n_views = len(imgs)
    conf, note = _noisy_or(_view_confidences(members))
    established = n_views >= cfg.min_images_to_establish
    exist = (Belief(Status.DIRECTLY_OBSERVED, ev, confidence=conf, note=f"{n_views} views; {note}") if established
             else Belief(Status.WEAKLY_SUPPORTED, ev, note="single view: insufficient to establish this opening"))
    length = math.dist(a, b)
    dim = (Belief(Status.DIRECTLY_OBSERVED, ev, value=length, sigma=math.hypot(sa, sb), note="opening width (m)")
           if a_obs and b_obs else
           Belief(Status.WEAKLY_SUPPORTED, ev, value=length, note="lower bound: an edge is not observed"))
    hsig = host.beliefs["position"].sigma if host else None
    if hsig is not None:
        # across-wall position inherits the host's offset uncertainty (propagated, not reset)
        pos = Belief(Status.GEOMETRICALLY_INFERRED, ev, sigma=hsig,
                     note="across-wall position inherited from the host wall; along-wall sigma on the end points")
    else:
        pos = Belief(Status.WEAKLY_SUPPORTED, ev, sigma=sa, note="no host wall: position from the opening's own views")
    alt_hosts = tuple(sorted({h for o in members for h in host_alts.get(o.evidence_id, ())}))
    topo = (Belief(Status.GEOMETRICALLY_INFERRED, ev, note=f"attached to {host.entity_id}") if host else
            Belief(Status.UNKNOWN, ev, note="host wall unknown or ambiguous: " + (",".join(alt_hosts) or "none within gate")))
    if conflict_cls:
        sem = Belief(Status.CONFLICTING, ev, note=f"class votes {[(k.value, c) for k, c in top]}")
    elif established:
        sem = Belief(Status.SEMANTICALLY_INFERRED, ev, confidence=conf, note="class from detector; " + note)
    else:
        sem = Belief(Status.WEAKLY_SUPPORTED, ev, note="class from a single view")
    ori = (Belief(Status.GEOMETRICALLY_INFERRED, ev, value=host.angle % math.pi, note="inherited from host wall")
           if host else UNKNOWN_BELIEF)
    zs0 = [o.z0 for o in members if o.z0 is not None]
    zs1 = [o.z1 for o in members if o.z1 is not None]
    return SpatialEntity(
        entity_id=_eid(kind.value, ev), kind=kind, a=a, b=b,
        z0=float(np.median(zs0)) if zs0 else None, z1=float(np.median(zs1)) if zs1 else None,
        evidence_ids=ev, image_ids=imgs,
        beliefs={"existence": exist, "position": pos, "dimensions": dim, "orientation": ori,
                 "geometry": Belief(Status.DIRECTLY_OBSERVED, ev), "topology": topo, "semantic": sem},
        sigma_a=sa, sigma_b=sb, end_a_observed=a_obs, end_b_observed=b_obs,
        host_id=host.entity_id if host else None, host_alternatives=alt_hosts, storey=members[0].storey,
        alternatives=[{"kind": k.value, "views": c} for k, c in top] if conflict_cls else [],
        partially_visible=any(o.visibility is Visibility.PARTIAL for o in members),
        occluded=any(o.visibility is Visibility.OCCLUDED for o in members))
