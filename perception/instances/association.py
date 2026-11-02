"""Gated association of observations into persistent entities (plan §4 Phase 1).

Replaces the failure mode of `object_resolution.merge_hypotheses` (union-find:
A~B and B~C merges A with C even when A and C are different objects) with:

  gate    label compatibility, 3D Mahalanobis distance (uncertainty-aware, so
          relative-depth observations are matched loosely and metric ones
          tightly), optional normal difference;
  score   log-likelihood, geometry dominant, optional appearance only as a
          small secondary term INSIDE the gate (never merges on looks alone);
  assign  per image, one Hungarian assignment against the entities that
          existed BEFORE that image -- two detections in the same image can
          never be the same entity, and an entity takes at most one detection
          per image, so there is no transitive chaining;
  doubt   when the best and second-best candidate are within a likelihood
          margin the observation is NOT merged: it becomes a provisional
          entity marked AMBIGUOUS, with both candidates recorded.

Every observation yields an `AssociationDecision` with its reason. Not done
here (see PENDING_IMPLEMENTATION.md): epipolar/reprojection verification
(`geom_verified` stays None) and negative visibility evidence.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from world_ir.observations import (
    AssociationDecision,
    AssociationState,
    DepthSource,
    Observation,
    attach_ledger,
    sigma_for,
)

#: chi-square, 3 degrees of freedom, 99% -- squared Mahalanobis distance above this fails the distance gate
GATE_CHI2_3DOF_99 = 11.34
MAX_NORMAL_DIFF_DEG = 30.0
NORMAL_SIGMA_DEG = 15.0
#: best and second-best candidate closer than this in log-likelihood (a 2:1 likelihood ratio) => AMBIGUOUS
AMBIGUITY_LL_MARGIN = math.log(2.0)
APPEARANCE_WEIGHT = 0.5
FUSED_SIGMA_FLOOR_M = 0.02  # fusing never claims better than this: repeated agreement is not infinite precision
_BIG = 1e6

#: child label -> parent label; a child may match its parent (a detector saying "chair" vs "armchair")
LABEL_PARENTS: Dict[str, str] = {"armchair": "chair", "office_chair": "chair", "sofa": "couch", "loveseat": "couch"}


def labels_compatible(a: str, b: str) -> bool:
    if a == b:
        return True
    return LABEL_PARENTS.get(a) == b or LABEL_PARENTS.get(b) == a


@dataclass
class EntityTrack:
    entity_id: str
    label: str
    position: np.ndarray
    sigma_m: float
    normal: Optional[np.ndarray]
    bounds_min: Optional[np.ndarray]
    bounds_max: Optional[np.ndarray]
    observation_ids: List[str] = field(default_factory=list)
    image_ids: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    ambiguous_with: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "position": [float(v) for v in self.position],
            "sigma_m": self.sigma_m,
            "normal": None if self.normal is None else [float(v) for v in self.normal],
            "observation_ids": list(self.observation_ids),
            "image_ids": sorted(self.image_ids),
            "confidence": self.confidence,
            "ambiguous_with": list(self.ambiguous_with),
        }


@dataclass(frozen=True)
class AssociationResult:
    entities: Tuple[EntityTrack, ...]
    decisions: Tuple[AssociationDecision, ...]
    assignment: Dict[str, str]  # observation_id -> entity_id
    #: the ledger this run produced: one Observation per input, one AssociationDecision per observation
    observations: Tuple[Observation, ...] = ()


def _unit(v: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else None


def _iou3d(o: Observation, e: EntityTrack) -> Optional[float]:
    if o.bounds_min is None or o.bounds_max is None or e.bounds_min is None or e.bounds_max is None:
        return None
    lo = np.maximum(np.asarray(o.bounds_min), e.bounds_min)
    hi = np.minimum(np.asarray(o.bounds_max), e.bounds_max)
    inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
    vol = lambda a, b: float(np.prod(np.clip(np.asarray(b) - np.asarray(a), 0.0, None)))
    union = vol(o.bounds_min, o.bounds_max) + vol(e.bounds_min, e.bounds_max) - inter
    return inter / union if union > 0 else None


def _candidate(o: Observation, e: EntityTrack, appearance_sim: Optional[float]):
    """(log-likelihood, d², dist, normal_diff_deg) if the entity passes every gate, else None."""
    if not labels_compatible(o.label, e.label):
        return None
    diff = np.asarray(o.world_pos, dtype=float) - e.position
    dist = float(np.linalg.norm(diff))
    m2 = dist * dist / (o.sigma_m ** 2 + e.sigma_m ** 2)
    if m2 > GATE_CHI2_3DOF_99:
        return None
    ll = -0.5 * m2
    ang = None
    on = _unit(o.normal)
    if on is not None and e.normal is not None:
        # a wall seen from the other side has a flipped normal: compare orientation, not sign
        ang = math.degrees(math.acos(min(1.0, abs(float(on @ e.normal)))))
        if ang > MAX_NORMAL_DIFF_DEG:
            return None
        ll -= 0.5 * (ang / NORMAL_SIGMA_DEG) ** 2
    if appearance_sim is not None:
        ll += APPEARANCE_WEIGHT * (appearance_sim - 1.0)
    return ll, m2, dist, ang


def _new_entity(o: Observation, index: int, ambiguous_with: Tuple[str, ...] = ()) -> EntityTrack:
    return EntityTrack(
        entity_id=f"ent-{o.label}-{index:04d}", label=o.label, position=np.asarray(o.world_pos, dtype=float),
        sigma_m=o.sigma_m, normal=_unit(o.normal),
        bounds_min=None if o.bounds_min is None else np.asarray(o.bounds_min, dtype=float),
        bounds_max=None if o.bounds_max is None else np.asarray(o.bounds_max, dtype=float),
        observation_ids=[o.observation_id], image_ids={o.image_id}, confidence=o.confidence,
        ambiguous_with=ambiguous_with,
    )


def _fuse(e: EntityTrack, o: Observation) -> None:
    we, wo = 1.0 / e.sigma_m ** 2, 1.0 / o.sigma_m ** 2
    e.position = (we * e.position + wo * np.asarray(o.world_pos, dtype=float)) / (we + wo)
    e.sigma_m = max(FUSED_SIGMA_FLOOR_M, 1.0 / math.sqrt(we + wo))
    on = _unit(o.normal)
    if on is not None:
        if e.normal is None:
            e.normal = on
        else:
            if float(on @ e.normal) < 0:
                on = -on
            e.normal = _unit(e.normal * we + on * wo)
    if o.bounds_min is not None and o.bounds_max is not None:
        lo, hi = np.asarray(o.bounds_min, dtype=float), np.asarray(o.bounds_max, dtype=float)
        e.bounds_min = lo if e.bounds_min is None else np.minimum(e.bounds_min, lo)
        e.bounds_max = hi if e.bounds_max is None else np.maximum(e.bounds_max, hi)
    e.observation_ids.append(o.observation_id)
    e.image_ids.add(o.image_id)
    e.confidence = max(e.confidence, 1.0 - (1.0 - e.confidence) * (1.0 - o.confidence))


def associate(
    observations: Sequence[Observation],
    appearance: Optional[Dict[str, Sequence[float]]] = None,
) -> AssociationResult:
    """Associate observations into entities. Deterministic: images and observations are processed in sorted id order.
    `appearance` (observation_id -> embedding) only adds a small score term to candidates that already passed the
    geometric gates."""
    appearance = appearance or {}
    entities: List[EntityTrack] = []
    decisions: List[AssociationDecision] = []
    assignment: Dict[str, str] = {}

    by_image: Dict[str, List[Observation]] = {}
    for o in observations:
        by_image.setdefault(o.image_id, []).append(o)

    for image_id in sorted(by_image):
        obs = sorted(by_image[image_id], key=lambda o: o.observation_id)
        existing = list(entities)  # entities as they were BEFORE this image: nothing in this image can chain
        cands: List[List[Tuple[int, float, float, float, Optional[float]]]] = []
        for o in obs:
            row = []
            for j, e in enumerate(existing):
                sim = None
                if o.observation_id in appearance and e.observation_ids and e.observation_ids[0] in appearance:
                    a, b = np.asarray(appearance[o.observation_id], float), np.asarray(appearance[e.observation_ids[0]], float)
                    sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) or 1.0))
                c = _candidate(o, e, sim)
                if c is not None:
                    row.append((j, *c))
            cands.append(row)

        ambiguous: Dict[int, Tuple[str, ...]] = {}
        for i, row in enumerate(cands):
            ranked = sorted(row, key=lambda c: -c[1])
            if len(ranked) >= 2 and ranked[0][1] - ranked[1][1] < AMBIGUITY_LL_MARGIN:
                ambiguous[i] = tuple(existing[c[0]].entity_id for c in ranked if ranked[0][1] - c[1] < AMBIGUITY_LL_MARGIN)

        matched: Dict[int, Tuple[int, float, float, float, Optional[float]]] = {}
        rows = [i for i in range(len(obs)) if i not in ambiguous and cands[i]]
        if rows and existing:
            cost = np.full((len(rows), len(existing)), _BIG)
            for r, i in enumerate(rows):
                for c in cands[i]:
                    cost[r, c[0]] = -c[1]
            for r, j in zip(*linear_sum_assignment(cost)):
                if cost[r, j] < _BIG:
                    i = rows[r]
                    matched[i] = next(c for c in cands[i] if c[0] == j)

        for i, o in enumerate(obs):
            row_c = tuple(sorted(((existing[c[0]].entity_id, round(math.exp(c[1]), 6)) for c in cands[i]), key=lambda t: -t[1]))
            if i in matched:
                j, ll, m2, dist, ang = matched[i]
                e = existing[j]
                iou = _iou3d(o, e)
                _fuse(e, o)
                assignment[o.observation_id] = e.entity_id
                decisions.append(AssociationDecision(
                    o.observation_id, AssociationState.SAME_ENTITY_CONFIRMED, e.entity_id, round(math.exp(ll), 6),
                    f"within the distance gate (d²={m2:.2f}) of {e.entity_id}; depth={o.depth_source.value}; "
                    "no epipolar check run", dist, m2, ang, iou, None, row_c))
                continue
            new = _new_entity(o, len(entities), ambiguous.get(i, ()))
            entities.append(new)
            assignment[o.observation_id] = new.entity_id
            if i in ambiguous:
                decisions.append(AssociationDecision(
                    o.observation_id, AssociationState.AMBIGUOUS, new.entity_id, 0.0,
                    f"cannot tell {', '.join(ambiguous[i])} apart (likelihoods within 2:1): kept as a separate provisional "
                    "entity, not merged", candidates=row_c))
            elif cands[i]:
                decisions.append(AssociationDecision(
                    o.observation_id, AssociationState.DIFFERENT_ENTITY, new.entity_id, 0.0,
                    "its best candidate was claimed by another detection in the same image", candidates=row_c))
            else:
                decisions.append(AssociationDecision(
                    o.observation_id, AssociationState.DIFFERENT_ENTITY, new.entity_id, 0.0,
                    "no existing entity passed the label/distance/normal gates: new entity"))

    return AssociationResult(tuple(entities), tuple(decisions), assignment, tuple(observations))


def observation_from_hypothesis(
    h, depth_source: DepthSource, camera_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    model: str = "unknown@unknown", license: str = "unknown",
) -> Observation:
    """Lift an `ObjectHypothesis3D` into the ledger. Its uncertainty comes from the depth source and the distance to the
    camera (a monocular-depth lift is therefore matched loosely, a metric one tightly)."""
    # a hypothesis lifted from relative depth is relative whatever the caller claims (scale gate, blueprint 3.3)
    if getattr(h, "depth_source", None) is DepthSource.RELATIVE:
        depth_source = DepthSource.RELATIVE
    pos = (h.position.x, h.position.y, h.position.z)
    rng = float(np.linalg.norm(np.asarray(pos) - np.asarray(camera_position)))
    return Observation(
        observation_id=h.region_id, image_id=h.evidence_id, label=h.label, world_pos=pos,
        sigma_m=sigma_for(depth_source, rng), depth_source=depth_source, model=model, license=license,
        bounds_min=(h.bounds_min.x, h.bounds_min.y, h.bounds_min.z),
        bounds_max=(h.bounds_max.x, h.bounds_max.y, h.bounds_max.z), confidence=h.confidence,
    )


def resolve_hypotheses(
    hypotheses,
    depth_source: DepthSource = DepthSource.MVS,
    camera_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    model: str = "unknown@unknown",
    license: str = "unknown",
    appearance: Optional[Dict[str, Sequence[float]]] = None,
    world=None,
):
    """The object-resolution path (Hungarian, no transitive chaining, AMBIGUOUS kept apart): same `MergedObjectCandidate`
    output as the legacy `merge_hypotheses`. Returns (candidates, AssociationResult) so callers can report the
    decisions. `camera_positions` (evidence_id -> xyz) sets each observation's range, hence its uncertainty.
    `appearance` maps "evidence_id|region_id" -> embedding. When `world` is given, the observations and EVERY association
    decision are stored in its ledger (`world_ir.observations.attach_ledger`), so they are saved with the version."""
    from perception.instances.object_resolution import MergedObjectCandidate

    camera_positions = camera_positions or {}
    by_obs = {}
    observations = []
    for h in hypotheses:
        o = observation_from_hypothesis(h, depth_source, camera_positions.get(h.evidence_id, (0.0, 0.0, 0.0)), model, license)
        o = dataclasses.replace(o, observation_id=f"{h.evidence_id}|{h.region_id}")  # region ids are not unique across images
        by_obs[o.observation_id] = h
        observations.append(o)
    result = associate(observations, appearance)
    if world is not None:
        attach_ledger(world, result.observations, result.decisions)

    groups: Dict[str, list] = {}
    for oid, eid in result.assignment.items():
        groups.setdefault(eid, []).append(by_obs[oid])

    from engine.math import Vec3

    candidates = []
    for n, (_, members) in enumerate(sorted(groups.items())):
        members.sort(key=lambda h: (h.evidence_id, h.region_id))
        w = sum(h.confidence for h in members) or 1.0
        pos = Vec3(*(sum(getattr(h.position, a) * h.confidence for h in members) / w for a in "xyz"))
        lo = Vec3(*(min(getattr(h.bounds_min, a) for h in members) for a in "xyz"))
        hi = Vec3(*(max(getattr(h.bounds_max, a) for h in members) for a in "xyz"))
        best = max(h.confidence for h in members)
        candidates.append(MergedObjectCandidate(
            candidate_id=f"obj-{members[0].label}-{n:04d}", label=members[0].label, position=pos, bounds_min=lo,
            bounds_max=hi, source_hypotheses=tuple(members), confidence=max(best, min(1.0, 1.0 - (1.0 - best) ** len(members)))))
    candidates.sort(key=lambda c: (c.label, c.position.x, c.position.y, c.position.z))
    return candidates, result


def overlap_relation(bounds_a, bounds_b, camera=None) -> AssociationState:
    """Why two boxes overlap. Touching/penetrating in 3D is PHYSICAL_INTERSECTION (wall-floor, frame-in-wall, mounted
    on). Disjoint in 3D but overlapping in `camera`'s image is IMAGE_OCCLUSION: one is in front of the other, they are not
    the same thing and do not touch. Otherwise DIFFERENT_ENTITY."""
    (amin, amax), (bmin, bmax) = ((np.asarray(b[0], float), np.asarray(b[1], float)) for b in (bounds_a, bounds_b))
    if np.all(np.minimum(amax, bmax) - np.maximum(amin, bmin) >= -1e-9):
        return AssociationState.PHYSICAL_INTERSECTION
    if camera is None:
        return AssociationState.DIFFERENT_ENTITY

    def rect(lo, hi):
        from engine.math import Vec3

        pts = [camera.project(Vec3(x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        if any(p is None for p in pts):
            return None
        us, vs = [p[0] for p in pts], [p[1] for p in pts]
        return min(us), min(vs), max(us), max(vs)

    ra, rb = rect(amin, amax), rect(bmin, bmax)
    if ra and rb and min(ra[2], rb[2]) > max(ra[0], rb[0]) and min(ra[3], rb[3]) > max(ra[1], rb[1]):
        return AssociationState.IMAGE_OCCLUSION
    return AssociationState.DIFFERENT_ENTITY


def surface_relation(plane_a, plane_b, normal_tol_deg: float = 10.0, dist_tol_m: float = 0.1) -> AssociationState:
    """Plane parameter test, plane = (normal, d) with n·x = d. Same orientation AND same offset => one surface; parallel
    but offset (two walls 4 m apart, repeated floor patterns) or differently oriented => different."""
    na, da = _unit(plane_a[0]), float(plane_a[1])
    nb, db = _unit(plane_b[0]), float(plane_b[1])
    dot = float(na @ nb)
    if math.degrees(math.acos(min(1.0, abs(dot)))) > normal_tol_deg:
        return AssociationState.DIFFERENT_ENTITY
    if dot < 0:  # same plane seen from the other side
        db = -db
    return AssociationState.SAME_SURFACE_CONFIRMED if abs(da - db) <= dist_tol_m else AssociationState.DIFFERENT_ENTITY


def identity_metrics(assignment: Dict[str, str], truth: Dict[str, str]) -> dict:
    """Pairwise identity metrics against ground truth ids (observation_id -> true object id). wrong_merge_rate = share of
    predicted same-entity pairs that are really different objects; wrong_split_rate = share of truly-same pairs the engine
    separated; duplicate_entity_rate = (predicted entities - true objects) / true objects."""
    ids = sorted(set(assignment) & set(truth))
    tp = fp = fn = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            same_pred = assignment[ids[a]] == assignment[ids[b]]
            same_true = truth[ids[a]] == truth[ids[b]]
            tp += same_pred and same_true
            fp += same_pred and not same_true
            fn += same_true and not same_pred
    n_true = len({truth[i] for i in ids})
    n_pred = len({assignment[i] for i in ids})
    return {
        "pair_precision": tp / (tp + fp) if tp + fp else 1.0,
        "pair_recall": tp / (tp + fn) if tp + fn else 1.0,
        "wrong_merge_rate": fp / (tp + fp) if tp + fp else 0.0,
        "wrong_split_rate": fn / (tp + fn) if tp + fn else 0.0,
        "duplicate_entity_rate": (n_pred - n_true) / n_true if n_true else 0.0,
    }
