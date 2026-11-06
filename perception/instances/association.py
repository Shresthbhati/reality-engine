"""Gated association of observations into persistent entities (plan §4 Phase 1, Epic E05-3).

Replaces the failure mode of `object_resolution.merge_hypotheses` (union-find:
A~B and B~C merges A with C even when A and C are different objects) with:

  gate    label compatibility, 3D Mahalanobis distance (uncertainty-aware, so
          relative-depth observations are matched loosely and metric ones
          tightly), normal difference, visual appearance consistency, and
          two-view epipolar geometric verification;
  score   log-likelihood, geometry dominant, appearance similarity term
          inside the gate (tracks running appearance embeddings across views);
  assign  per image, one Hungarian assignment against the entities that
          existed BEFORE that image -- two detections in the same image can
          never be the same entity, and an entity takes at most one detection
          per image, so there is no transitive chaining;
  doubt   when the best and second-best candidate are within a likelihood
          margin without appearance disambiguation, the observation is NOT
          merged: it becomes a provisional entity marked AMBIGUOUS, with both
          candidates recorded.

Every observation yields an `AssociationDecision` with its reason and
`geom_verified` flag reflecting two-view epipolar geometric validation.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from engine.math import Vec3
from perception.instances.epipolar import epipolar_consistent
from world_ir.camera import PinholeCamera
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
DEFAULT_APPEARANCE_WEIGHT = 1.5
MIN_APPEARANCE_SIM_GATE = 0.20  # Observations below this cosine similarity fail the appearance gate
FUSED_SIGMA_FLOOR_M = 0.02  # fusing never claims better than this: repeated agreement is not infinite precision
_BIG = 1e6

#: child label -> parent label; a child may match its parent (a detector saying "chair" vs "armchair")
LABEL_PARENTS: Dict[str, str] = {
    "armchair": "chair",
    "office_chair": "chair",
    "sofa": "couch",
    "loveseat": "couch",
}


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
    appearance_embedding: Optional[np.ndarray] = None  # Normalized fused visual descriptor

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


def _candidate(
    o: Observation,
    e: EntityTrack,
    appearance_sim: Optional[float],
    geom_verified: Optional[bool] = None,
    min_appearance_sim: float = MIN_APPEARANCE_SIM_GATE,
    appearance_weight: float = DEFAULT_APPEARANCE_WEIGHT,
):
    """(log-likelihood, d², dist, normal_diff_deg, geom_verified) if entity passes every gate, else None."""
    if not labels_compatible(o.label, e.label):
        return None

    # Geometric distance gate
    diff = np.asarray(o.world_pos, dtype=float) - e.position
    dist = float(np.linalg.norm(diff))
    m2 = dist * dist / (o.sigma_m ** 2 + e.sigma_m ** 2)
    if m2 > GATE_CHI2_3DOF_99:
        return None

    # Normal orientation gate
    ang = None
    on = _unit(o.normal)
    if on is not None and e.normal is not None:
        ang = math.degrees(math.acos(min(1.0, abs(float(on @ e.normal)))))
        if ang > MAX_NORMAL_DIFF_DEG:
            return None

    # Appearance gate: reject visually incompatible candidates when embeddings are available
    if appearance_sim is not None:
        if appearance_sim < min_appearance_sim:
            return None

    # Epipolar / 2-view geometric verification gate
    if geom_verified is False:
        return None

    # Log-likelihood computation
    ll = -0.5 * m2
    if ang is not None:
        ll -= 0.5 * (ang / NORMAL_SIGMA_DEG) ** 2

    if appearance_sim is not None:
        # Appearance term rewards strong visual similarity, penalizes low similarity
        ll += appearance_weight * (appearance_sim - 0.5)

    return ll, m2, dist, ang, geom_verified


def _new_entity(
    o: Observation,
    index: int,
    ambiguous_with: Tuple[str, ...] = (),
    appearance_embedding: Optional[np.ndarray] = None,
) -> EntityTrack:
    return EntityTrack(
        entity_id=f"ent-{o.label}-{index:04d}",
        label=o.label,
        position=np.asarray(o.world_pos, dtype=float),
        sigma_m=o.sigma_m,
        normal=_unit(o.normal),
        bounds_min=None if o.bounds_min is None else np.asarray(o.bounds_min, dtype=float),
        bounds_max=None if o.bounds_max is None else np.asarray(o.bounds_max, dtype=float),
        observation_ids=[o.observation_id],
        image_ids={o.image_id},
        confidence=o.confidence,
        ambiguous_with=ambiguous_with,
        appearance_embedding=appearance_embedding,
    )


def _fuse(
    e: EntityTrack,
    o: Observation,
    obs_embedding: Optional[np.ndarray] = None,
) -> None:
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

    # Fuse normalized visual appearance embedding
    if obs_embedding is not None:
        o_norm = np.linalg.norm(obs_embedding)
        if o_norm > 1e-12:
            o_unit = obs_embedding / o_norm
            if e.appearance_embedding is None:
                e.appearance_embedding = o_unit
            else:
                fused_emb = e.appearance_embedding * len(e.observation_ids) + o_unit
                fnorm = np.linalg.norm(fused_emb)
                e.appearance_embedding = fused_emb / fnorm if fnorm > 1e-12 else o_unit

    e.observation_ids.append(o.observation_id)
    e.image_ids.add(o.image_id)
    e.confidence = max(e.confidence, 1.0 - (1.0 - e.confidence) * (1.0 - o.confidence))


def _compute_appearance_sim(
    o: Observation,
    e: EntityTrack,
    appearance: Dict[str, Sequence[float]],
) -> Optional[float]:
    """Compute cosine similarity between observation embedding and entity track descriptor."""
    if o.observation_id not in appearance:
        return None

    a = np.asarray(appearance[o.observation_id], dtype=float)
    a_norm = np.linalg.norm(a)
    if a_norm < 1e-12:
        return None
    a = a / a_norm

    # Prefer comparing to entity's running fused embedding
    if e.appearance_embedding is not None:
        return float(np.dot(a, e.appearance_embedding))

    # Fallback to first observation with an embedding
    for oid in e.observation_ids:
        if oid in appearance:
            b = np.asarray(appearance[oid], dtype=float)
            b_norm = np.linalg.norm(b)
            if b_norm > 1e-12:
                return float(np.dot(a, b / b_norm))

    return None


def _check_epipolar(
    o: Observation,
    e: EntityTrack,
    obs_by_id: Dict[str, Observation],
    cameras: Optional[Dict[str, PinholeCamera]],
    epipolar_tol_px: float = 8.0,
) -> Optional[bool]:
    """Check two-view epipolar geometric consistency between observation and track."""
    if not cameras or o.image_id not in cameras:
        return None

    cam_b = cameras[o.image_id]
    # Center pixel of observation in image B
    if o.bbox_px is not None:
        pix_b = ((o.bbox_px[0] + o.bbox_px[2]) * 0.5, (o.bbox_px[1] + o.bbox_px[3]) * 0.5)
    else:
        proj_b = cam_b.project(Vec3(*o.world_pos))
        if proj_b is None:
            return None
        pix_b = proj_b

    # Compare against an observation from a DIFFERENT image in track e
    for ref_oid in e.observation_ids:
        ref_o = obs_by_id.get(ref_oid)
        if ref_o is None or ref_o.image_id == o.image_id:
            continue
        if ref_o.image_id not in cameras:
            continue

        cam_a = cameras[ref_o.image_id]
        # Coincident camera centres give no baseline: the essential matrix is zero, so the test cannot say
        # "inconsistent", only "unmeasurable". Leave it unrun (None) so the other gates decide.
        if (cam_a.extrinsics.position - cam_b.extrinsics.position).length() < 1e-9:
            continue
        if ref_o.bbox_px is not None:
            pix_a = ((ref_o.bbox_px[0] + ref_o.bbox_px[2]) * 0.5, (ref_o.bbox_px[1] + ref_o.bbox_px[3]) * 0.5)
        else:
            proj_a = cam_a.project(Vec3(*ref_o.world_pos))
            if proj_a is None:
                continue
            pix_a = proj_a

        # Evaluate epipolar consistency
        return epipolar_consistent(cam_a, pix_a, cam_b, pix_b, tolerance_px=epipolar_tol_px)

    return None


def associate(
    observations: Sequence[Observation],
    appearance: Optional[Dict[str, Sequence[float]]] = None,
    cameras: Optional[Dict[str, PinholeCamera]] = None,
    epipolar_tolerance_px: float = 8.0,
    min_appearance_sim: float = MIN_APPEARANCE_SIM_GATE,
    appearance_weight: float = DEFAULT_APPEARANCE_WEIGHT,
) -> AssociationResult:
    """Associate multi-view observations into persistent entities via Hungarian matching.

    Args:
        observations: Input sequence of 3D observations.
        appearance: Optional mapping observation_id -> visual embedding vector.
        cameras: Optional mapping image_id -> PinholeCamera for epipolar verification.
        epipolar_tolerance_px: Epipolar line distance threshold in pixels.
        min_appearance_sim: Minimum cosine similarity required to pass appearance gate.
        appearance_weight: Weight applied to appearance log-likelihood term.
    """
    appearance = appearance or {}
    entities: List[EntityTrack] = []
    decisions: List[AssociationDecision] = []
    assignment: Dict[str, str] = {}

    obs_by_id: Dict[str, Observation] = {o.observation_id: o for o in observations}
    by_image: Dict[str, List[Observation]] = {}
    for o in observations:
        by_image.setdefault(o.image_id, []).append(o)

    for image_id in sorted(by_image):
        obs = sorted(by_image[image_id], key=lambda o: o.observation_id)
        existing = list(entities)  # entities as they were BEFORE this image: nothing in this image can chain

        cands: List[List[Tuple[int, float, float, float, Optional[float], Optional[bool]]]] = []
        app_sims_per_row: List[Dict[int, float]] = []

        for o in obs:
            row = []
            sim_dict: Dict[int, float] = {}
            for j, e in enumerate(existing):
                sim = _compute_appearance_sim(o, e, appearance)
                if sim is not None:
                    sim_dict[j] = sim

                geom_ver = _check_epipolar(o, e, obs_by_id, cameras, epipolar_tolerance_px)
                c = _candidate(
                    o,
                    e,
                    sim,
                    geom_verified=geom_ver,
                    min_appearance_sim=min_appearance_sim,
                    appearance_weight=appearance_weight,
                )
                if c is not None:
                    row.append((j, *c))
            cands.append(row)
            app_sims_per_row.append(sim_dict)

        # Ambiguity detection with appearance disambiguation
        ambiguous: Dict[int, Tuple[str, ...]] = {}
        for i, row in enumerate(cands):
            ranked = sorted(row, key=lambda c: -c[1])
            if len(ranked) >= 2 and ranked[0][1] - ranked[1][1] < AMBIGUITY_LL_MARGIN:
                # Check if appearance breaks the ambiguity
                cand0_idx, cand1_idx = ranked[0][0], ranked[1][0]
                sim0 = app_sims_per_row[i].get(cand0_idx)
                sim1 = app_sims_per_row[i].get(cand1_idx)

                # If appearance clearly favors candidate 0 over candidate 1, resolve ambiguity
                if sim0 is not None and sim1 is not None and (sim0 - sim1 >= 0.25) and sim0 >= 0.65:
                    # Ambiguity resolved by visual appearance!
                    pass
                else:
                    ambiguous[i] = tuple(existing[c[0]].entity_id for c in ranked if ranked[0][1] - c[1] < AMBIGUITY_LL_MARGIN)

        # Hungarian assignment
        matched: Dict[int, Tuple[int, float, float, float, Optional[float], Optional[bool]]] = {}
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
            o_emb = np.asarray(appearance[o.observation_id], dtype=float) if o.observation_id in appearance else None

            if i in matched:
                j, ll, m2, dist, ang, geom_ver = matched[i]
                e = existing[j]
                iou = _iou3d(o, e)
                _fuse(e, o, o_emb)
                assignment[o.observation_id] = e.entity_id

                geom_text = "epipolar verified" if geom_ver is True else "no epipolar check run"
                decisions.append(
                    AssociationDecision(
                        o.observation_id,
                        AssociationState.SAME_ENTITY_CONFIRMED,
                        e.entity_id,
                        round(math.exp(ll), 6),
                        f"within the distance gate (d²={m2:.2f}) of {e.entity_id}; depth={o.depth_source.value}; {geom_text}",
                        dist,
                        m2,
                        ang,
                        iou,
                        geom_ver,
                        row_c,
                    )
                )
                continue

            # New entity creation
            new = _new_entity(
                o,
                len(entities),
                ambiguous.get(i, ()),
                appearance_embedding=_unit(o_emb),
            )
            entities.append(new)
            assignment[o.observation_id] = new.entity_id

            if i in ambiguous:
                decisions.append(
                    AssociationDecision(
                        o.observation_id,
                        AssociationState.AMBIGUOUS,
                        new.entity_id,
                        0.0,
                        f"cannot tell {', '.join(ambiguous[i])} apart (likelihoods within 2:1): kept as a separate provisional entity, not merged",
                        candidates=row_c,
                    )
                )
            elif cands[i]:
                decisions.append(
                    AssociationDecision(
                        o.observation_id,
                        AssociationState.DIFFERENT_ENTITY,
                        new.entity_id,
                        0.0,
                        "its best candidate was claimed by another detection in the same image or failed appearance/epipolar gate",
                        candidates=row_c,
                    )
                )
            else:
                decisions.append(
                    AssociationDecision(
                        o.observation_id,
                        AssociationState.DIFFERENT_ENTITY,
                        new.entity_id,
                        0.0,
                        "no existing entity passed the label/distance/normal/appearance/epipolar gates: new entity",
                    )
                )

    return AssociationResult(tuple(entities), tuple(decisions), assignment, tuple(observations))


def observation_from_hypothesis(
    h,
    depth_source: DepthSource,
    camera_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    model: str = "unknown@unknown",
    license: str = "unknown",
) -> Observation:
    """Lift an `ObjectHypothesis3D` into the ledger."""
    if getattr(h, "depth_source", None) is DepthSource.RELATIVE:
        depth_source = DepthSource.RELATIVE
    pos = (h.position.x, h.position.y, h.position.z)
    rng = float(np.linalg.norm(np.asarray(pos) - np.asarray(camera_position)))
    return Observation(
        observation_id=h.region_id,
        image_id=h.evidence_id,
        label=h.label,
        world_pos=pos,
        sigma_m=sigma_for(depth_source, rng),
        depth_source=depth_source,
        model=model,
        license=license,
        bounds_min=(h.bounds_min.x, h.bounds_min.y, h.bounds_min.z),
        bounds_max=(h.bounds_max.x, h.bounds_max.y, h.bounds_max.z),
        confidence=h.confidence,
    )


def resolve_hypotheses(
    hypotheses,
    depth_source: DepthSource = DepthSource.MVS,
    camera_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    cameras: Optional[Dict[str, PinholeCamera]] = None,
    model: str = "unknown@unknown",
    license: str = "unknown",
    appearance: Optional[Dict[str, Sequence[float]]] = None,
    world=None,
):
    """The object-resolution path (Hungarian, appearance embeddings, epipolar gating)."""
    from perception.instances.object_resolution import MergedObjectCandidate

    camera_positions = camera_positions or {}
    by_obs = {}
    observations = []
    for h in hypotheses:
        o = observation_from_hypothesis(h, depth_source, camera_positions.get(h.evidence_id, (0.0, 0.0, 0.0)), model, license)
        o = dataclasses.replace(o, observation_id=f"{h.evidence_id}|{h.region_id}")
        by_obs[o.observation_id] = h
        observations.append(o)

    result = associate(observations, appearance=appearance, cameras=cameras)
    if world is not None:
        attach_ledger(world, result.observations, result.decisions)

    groups: Dict[str, list] = {}
    for oid, eid in result.assignment.items():
        groups.setdefault(eid, []).append(by_obs[oid])

    candidates = []
    for n, (_, members) in enumerate(sorted(groups.items())):
        members.sort(key=lambda h: (h.evidence_id, h.region_id))
        w = sum(h.confidence for h in members) or 1.0
        pos = Vec3(*(sum(getattr(h.position, a) * h.confidence for h in members) / w for a in "xyz"))
        lo = Vec3(*(min(getattr(h.bounds_min, a) for h in members) for a in "xyz"))
        hi = Vec3(*(max(getattr(h.bounds_max, a) for h in members) for a in "xyz"))
        best = max(h.confidence for h in members)
        candidates.append(
            MergedObjectCandidate(
                candidate_id=f"obj-{members[0].label}-{n:04d}",
                label=members[0].label,
                position=pos,
                bounds_min=lo,
                bounds_max=hi,
                source_hypotheses=tuple(members),
                confidence=max(best, min(1.0, 1.0 - (1.0 - best) ** len(members))),
            )
        )
    candidates.sort(key=lambda c: (c.label, c.position.x, c.position.y, c.position.z))
    return candidates, result


def overlap_relation(bounds_a, bounds_b, camera=None) -> AssociationState:
    """Why two boxes overlap in 3D or 2D image plane."""
    (amin, amax), (bmin, bmax) = ((np.asarray(b[0], float), np.asarray(b[1], float)) for b in (bounds_a, bounds_b))
    if np.all(np.minimum(amax, bmax) - np.maximum(amin, bmin) >= -1e-9):
        return AssociationState.PHYSICAL_INTERSECTION
    if camera is None:
        return AssociationState.DIFFERENT_ENTITY

    def rect(lo, hi):
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
    """Plane parameter test, plane = (normal, d) with n·x = d."""
    na, da = _unit(plane_a[0]), float(plane_a[1])
    nb, db = _unit(plane_b[0]), float(plane_b[1])
    dot = float(na @ nb)
    if math.degrees(math.acos(min(1.0, abs(dot)))) > normal_tol_deg:
        return AssociationState.DIFFERENT_ENTITY
    if dot < 0:
        db = -db
    return AssociationState.SAME_SURFACE_CONFIRMED if abs(da - db) <= dist_tol_m else AssociationState.DIFFERENT_ENTITY


def identity_metrics(assignment: Dict[str, str], truth: Dict[str, str]) -> dict:
    """Pairwise identity metrics against ground truth ids (observation_id -> true object id)."""
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
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "identity_consistency": f1,
        "wrong_merge_rate": fp / (tp + fp) if tp + fp else 0.0,
        "wrong_split_rate": fn / (tp + fn) if tp + fn else 0.0,
        "duplicate_entity_rate": (n_pred - n_true) / n_true if n_true else 0.0,
    }
