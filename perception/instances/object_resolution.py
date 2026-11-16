"""Multi-view object entity resolution (semantic perception campaign,
Phase 5 "Object Resolution" / evidence-fusion campaign's entity-
resolution stage): merge `ObjectHypothesis3D` observations of the same
physical object seen from different frames/views into one candidate,
without discarding any source hypothesis.

Before this module, `perception/instances/lifting.py` produced one
`ObjectHypothesis3D` per (region, frame) -- nothing combined the N
hypotheses a real multi-image capture would produce for the same chair
seen in 5 different photos into a single candidate object. This is that
missing stage, sitting between per-frame lifting and WorldIR promotion
(which does not exist yet and is separate, later work -- this module
produces a fused *candidate*, not a WorldIR entity, matching how
`evidence/promote_planes.py` sits downstream of pure plane detection).

DEPRECATED (blueprint 3.2): the union-find merge below is the LEGACY resolver. The
only supported resolver is `perception.instances.association` (gated Hungarian,
no transitive chaining, every decision recorded in the ledger) -- use
`resolve_objects()` here or `association.resolve_hypotheses()`. `merge_hypotheses`
is kept only so old callers/tests can opt in to the old behaviour explicitly; it
is never the default anywhere and warns on every call.

Merge criterion (deterministic, geometry + semantics, "same class is
necessary but never sufficient" -- the same rule `world_ir/entity_reid.py`
already applies for cross-session matching):
  - same `label` (no cross-class merging, ever), AND
  - centroid-to-centroid distance within `distance_threshold_m`.
Transitive: if A merges with B and B merges with C, all three end up in
one candidate (union-find), which is the honest behavior for many
overlapping observations of one object across many frames -- a purely
pairwise rule would arbitrarily depend on iteration order.

This is NOT appearance-based (no embeddings — see
`docs/WORLD_MEMORY_LEARNING_AUDIT.md`'s standing rule against fake
embeddings) and NOT multi-view-consistency-verified (no epipolar/
reprojection check against the source cameras) — both are real,
larger, separate additions once a caller needs them; this module is
explicit about only using label + 3D proximity.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from engine.math import Vec3
from perception.instances.appearance import AppearanceDescriptor, appearance_similarity
from perception.instances.epipolar import (
    DEFAULT_EPIPOLAR_TOLERANCE_PX,
    epipolar_consistent,
)
from perception.instances.lifting import ObjectHypothesis3D
from world_ir.camera import PinholeCamera

#: Minimum appearance-histogram similarity (see perception/instances/
#: appearance.py) required to merge two hypotheses when descriptors are
#: supplied. Not tuned against a real dataset -- same "starting point a
#: caller can override" honesty as DEFAULT_MERGE_DISTANCE_M.
DEFAULT_APPEARANCE_SIMILARITY_MIN = 0.3

#: Two same-label hypotheses within this distance are treated as
#: observations of the same object. Not tuned against any real dataset
#: -- a starting point a caller can override, same honesty as
#: world_ir/entity_reid.py's MATCH_DISTANCE_M.
DEFAULT_MERGE_DISTANCE_M = 0.5


@dataclass(frozen=True)
class MergedObjectCandidate:
    candidate_id: str
    label: str
    #: Confidence-weighted centroid of all contributing hypotheses.
    position: Vec3
    #: Union AABB across all contributing hypotheses -- the fused
    #: object's extent is at least as large as any single view's,
    #: never smaller (a merge can only add evidence, never subtract it).
    bounds_min: Vec3
    bounds_max: Vec3
    #: Every hypothesis that fed this candidate -- never discarded.
    source_hypotheses: Tuple[ObjectHypothesis3D, ...]
    confidence: float

    @property
    def observation_count(self) -> int:
        return len(self.source_hypotheses)

    @property
    def evidence_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({h.evidence_id for h in self.source_hypotheses}))

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "position": self.position.to_dict(),
            "bounds_min": self.bounds_min.to_dict(),
            "bounds_max": self.bounds_max.to_dict(),
            "observation_count": self.observation_count,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "source_region_ids": [h.region_id for h in self.source_hypotheses],
        }


class _UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def _distance(a: Vec3, b: Vec3) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _box_iou(a: "MergedObjectCandidate", b: "MergedObjectCandidate") -> float:
    inter = 1.0
    for lo_a, hi_a, lo_b, hi_b in ((a.bounds_min.x, a.bounds_max.x, b.bounds_min.x, b.bounds_max.x),
                                   (a.bounds_min.y, a.bounds_max.y, b.bounds_min.y, b.bounds_max.y),
                                   (a.bounds_min.z, a.bounds_max.z, b.bounds_min.z, b.bounds_max.z)):
        inter *= max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    vol = lambda c: ((c.bounds_max.x - c.bounds_min.x) * (c.bounds_max.y - c.bounds_min.y)  # noqa: E731
                     * (c.bounds_max.z - c.bounds_min.z))
    union = vol(a) + vol(b) - inter
    return inter / union if union > 0 else 0.0


#: same-class candidates whose centres are closer than this share of the larger one's longest side are one object
#: (observed on a ten-photo room: per-view depth misalignment moved one sofa's box by about half its length)
FUSE_CENTRE_SHARE = 0.5


def _longest_side(c: "MergedObjectCandidate") -> float:
    return max(c.bounds_max.x - c.bounds_min.x, c.bounds_max.y - c.bounds_min.y, c.bounds_max.z - c.bounds_min.z)


def consensus_bounds(centre: Vec3, boxes: List[Tuple[Vec3, Vec3]]) -> Tuple[Vec3, Vec3]:
    """One box for several views of one object: ``centre`` +- the per-axis MEDIAN half-extent of the views' boxes.

    The union of per-view boxes grows with every view whose depth scale is a little off (each view's box sits somewhere
    else along its own rays), so a sofa seen in six photographs came out 7 m long. The median is what most views agree on
    and cannot be inflated by one misaligned view. One box is returned unchanged."""
    if len(boxes) == 1:
        return boxes[0]
    half = [sorted((hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)[ax] for lo, hi in boxes) for ax in range(3)]
    mid = len(boxes) // 2
    h = [(s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])) / 2.0 for s in half]
    return Vec3(centre.x - h[0], centre.y - h[1], centre.z - h[2]), Vec3(centre.x + h[0], centre.y + h[1], centre.z + h[2])


def fuse_overlapping_candidates(
    candidates: List["MergedObjectCandidate"], iou_min: Optional[float] = None,
    centre_share: float = FUSE_CENTRE_SHARE,
) -> List["MergedObjectCandidate"]:
    """Fuse same-label candidates whose bounds overlap by more than ``iou_min`` (default: the world validator's own
    duplicate rule, ``world_ir.validation._DUPLICATE_IOU``).

    Per-view depth is only approximately aligned, so one sofa seen from several photographs can come out of the
    association resolver as several candidates that overlap heavily, or sit closer than ``centre_share`` of the larger
    one's longest side. The validator calls the overlapping ones duplicates (``invariant_6_duplicate_entity``); they are
    one object, so they are fused here: every contributing hypothesis is kept, the extent is the consensus box (``consensus_bounds``), the position the
    confidence-weighted centroid. Order is stable."""
    if iou_min is None:
        from world_ir.validation import _DUPLICATE_IOU as iou_min
    n = len(candidates)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i], candidates[j]
            if a.label != b.label:
                continue
            near = _distance(a.position, b.position) < centre_share * max(_longest_side(a), _longest_side(b))
            if near or _box_iou(a, b) > iou_min:
                uf.union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    fused: List[MergedObjectCandidate] = []
    for root in sorted(groups):
        members = [candidates[i] for i in groups[root]]
        if len(members) == 1:
            fused.append(members[0])
            continue
        total = sum(max(m.confidence, 1e-9) for m in members)
        centroid = Vec3(*(sum(max(m.confidence, 1e-9) * getattr(m.position, ax) for m in members) / total
                          for ax in ("x", "y", "z")))
        lo, hi = consensus_bounds(centroid, [(m.bounds_min, m.bounds_max) for m in members])
        fused.append(MergedObjectCandidate(
            candidate_id=members[0].candidate_id,
            label=members[0].label,
            position=centroid,
            bounds_min=lo,
            bounds_max=hi,
            source_hypotheses=tuple(h for m in members for h in m.source_hypotheses),
            confidence=max(m.confidence for m in members),
        ))
    return fused


def absorb_scale_drift_duplicates(
    candidates: List["MergedObjectCandidate"], cameras: Dict[str, object], box_margin: float = 0.15,
) -> Tuple[List["MergedObjectCandidate"], int]:
    """Fold a single-photograph candidate into a multi-photograph one of the same label when geometry says they are one object.

    A per-photograph depth-scale error moves a detection along THAT camera's viewing ray, so the same sofa can come out of one
    photograph metres from where the others put it, and 3D distance cannot link them. A viewing ray can: if the confirmed
    object's position, projected into the orphan's photograph, falls inside the orphan's own 2D footprint (grown by
    ``box_margin``), and the confirmed object has NO detection of its own in that photograph, the orphan is that object seen
    with a wrong depth. It is absorbed as evidence (its hypotheses join ``source_hypotheses``); the confirmed object's position
    and extent are NOT moved by the drifted points. An orphan whose footprint contains no such object stays as it was.
    Returns ``(candidates, n_absorbed)``; order of the survivors is stable.

    UNPROVEN, NOT USED BY THE PIPELINE: measured on the 10-photo room it absorbed 0 orphans, because the confirmed objects'
    positions project between near-correct and thousands of pixels away in the orphans' photographs (the same depth-scale drift
    this is meant to repair), so the test only works once per-view scale is consistent."""
    confirmed = [i for i, c in enumerate(candidates) if len(c.evidence_ids) >= 2]
    orphans = [i for i, c in enumerate(candidates) if len(c.evidence_ids) == 1]
    if not confirmed or not orphans:
        return list(candidates), 0

    extra: Dict[int, List["MergedObjectCandidate"]] = {}
    absorbed = set()
    for oi in orphans:
        o = candidates[oi]
        view = o.evidence_ids[0]
        cam = cameras.get(view)
        if cam is None:
            continue
        pix = [q for q in (cam.project(p) for h in o.source_hypotheses for p in h.points) if q is not None]
        if len(pix) < 3:
            continue
        us, vs = [q[0] for q in pix], [q[1] for q in pix]
        du, dv = (max(us) - min(us)) * box_margin, (max(vs) - min(vs)) * box_margin
        u0, u1, v0, v1 = min(us) - du, max(us) + du, min(vs) - dv, max(vs) + dv
        cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
        best, best_d = None, None
        for ei in confirmed:
            e = candidates[ei]
            if e.label != o.label or view in e.evidence_ids:
                continue        # another label, or e already has its own detection in this photograph: o is a second instance
            q = cam.project(e.position)
            if q is None or not (u0 <= q[0] <= u1 and v0 <= q[1] <= v1):
                continue
            d = (q[0] - cu) ** 2 + (q[1] - cv) ** 2
            if best_d is None or d < best_d:
                best, best_d = ei, d
        if best is not None:
            extra.setdefault(best, []).append(o)
            absorbed.add(oi)

    out: List["MergedObjectCandidate"] = []
    for i, c in enumerate(candidates):
        if i in absorbed:
            continue
        if i in extra:
            hyps = tuple(c.source_hypotheses) + tuple(h for o in extra[i] for h in o.source_hypotheses)
            c = MergedObjectCandidate(
                candidate_id=c.candidate_id, label=c.label, position=c.position, bounds_min=c.bounds_min,
                bounds_max=c.bounds_max, source_hypotheses=hyps, confidence=c.confidence)
        out.append(c)
    return out, len(absorbed)


def resolve_objects(hypotheses: List[ObjectHypothesis3D], **kwargs):
    """The default object-resolution entry point: gated Hungarian association (see `association.resolve_hypotheses`,
    same keyword arguments). Returns (candidates, AssociationResult)."""
    from perception.instances.association import resolve_hypotheses

    return resolve_hypotheses(hypotheses, **kwargs)


def merge_hypotheses(
    hypotheses: List[ObjectHypothesis3D],
    distance_threshold_m: float = DEFAULT_MERGE_DISTANCE_M,
    cameras: Optional[Dict[str, PinholeCamera]] = None,
    appearance: Optional[Dict[str, AppearanceDescriptor]] = None,
    appearance_similarity_min: float = DEFAULT_APPEARANCE_SIMILARITY_MIN,
    epipolar_tolerance_px: float = DEFAULT_EPIPOLAR_TOLERANCE_PX,
) -> List[MergedObjectCandidate]:
    """DEPRECATED legacy union-find resolver (transitive chaining merges distinct neighbours); use `resolve_objects`.

    Cluster same-label, spatially-close hypotheses into candidates.

    Same-label + 3D proximity is the base gate (as before) but is not
    sufficient identity on its own -- per
    docs/future/perception/MULTI_VIEW_IDENTITY.md, symmetric scenes
    (e.g. four identical chairs) need a discriminating signal geometry
    alone can't give. Two more gates strengthen the merge decision, both
    opt-in via keyword arguments so existing callers/tests are
    unaffected when they aren't supplied:

    - `cameras` (evidence_id -> PinholeCamera): when both hypotheses'
      source cameras are known, require epipolar consistency (see
      perception/instances/epipolar.py) between their (recomputed)
      detection pixels -- rejects two hypotheses whose 3D
      reconstructions happen to land close together but that the
      cameras' known relative geometry says could not be the same
      physical point.
    - `appearance` (region_id -> AppearanceDescriptor): when both
      hypotheses' color histograms are known (see
      perception/instances/appearance.py), require histogram similarity
      at or above `appearance_similarity_min`.

    A hypothesis with no entry in `cameras`/`appearance` is not gated by
    that check (missing evidence degrades to the old label+proximity
    behavior for that pair rather than silently refusing every merge) --
    honest opt-in strengthening, not a change to the base contract.

    Deterministic: hypotheses are sorted by (label, region_id) before
    clustering, and output candidates are sorted by (label, position) so
    the same input list always produces the same output regardless of
    the caller's original ordering.
    """
    warnings.warn(
        "merge_hypotheses is the legacy union-find resolver and is deprecated: it chains A~B, B~C into one object. "
        "Use perception.instances.object_resolution.resolve_objects (gated Hungarian association).",
        DeprecationWarning, stacklevel=2,
    )
    if distance_threshold_m < 0:
        raise ValueError(f"distance_threshold_m must be non-negative, got {distance_threshold_m}")
    if not hypotheses:
        return []

    cameras = cameras or {}
    appearance = appearance or {}

    ordered = sorted(hypotheses, key=lambda h: (h.label, h.region_id))
    uf = _UnionFind(len(ordered))
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            if a.label != b.label:
                continue
            if _distance(a.position, b.position) > distance_threshold_m:
                continue

            cam_a, cam_b = cameras.get(a.evidence_id), cameras.get(b.evidence_id)
            if cam_a is not None and cam_b is not None:
                pixel_a = cam_a.project(a.position)
                pixel_b = cam_b.project(b.position)
                # A hypothesis' own camera can't project its own
                # reconstructed position only in a genuine
                # behind-the-camera degeneracy -- treat that as
                # inconsistent rather than skipping the check.
                if pixel_a is None or pixel_b is None:
                    continue
                if not epipolar_consistent(cam_a, pixel_a, cam_b, pixel_b, epipolar_tolerance_px):
                    continue

            desc_a, desc_b = appearance.get(a.region_id), appearance.get(b.region_id)
            if desc_a is not None and desc_b is not None:
                if appearance_similarity(desc_a, desc_b) < appearance_similarity_min:
                    continue

            uf.union(i, j)

    groups: Dict[int, List[ObjectHypothesis3D]] = {}
    for i, hyp in enumerate(ordered):
        groups.setdefault(uf.find(i), []).append(hyp)

    candidates: List[MergedObjectCandidate] = []
    for group_index, (root, members) in enumerate(sorted(groups.items())):
        total_weight = sum(h.confidence for h in members) or 1.0
        cx = sum(h.position.x * h.confidence for h in members) / total_weight
        cy = sum(h.position.y * h.confidence for h in members) / total_weight
        cz = sum(h.position.z * h.confidence for h in members) / total_weight

        bounds_min = Vec3(
            min(h.bounds_min.x for h in members),
            min(h.bounds_min.y for h in members),
            min(h.bounds_min.z for h in members),
        )
        bounds_max = Vec3(
            max(h.bounds_max.x for h in members),
            max(h.bounds_max.y for h in members),
            max(h.bounds_max.z for h in members),
        )

        # More independent observations agreeing raises confidence
        # (capped at 1.0), never lowers it below the best single view --
        # agreement is positive evidence, not a reason to doubt.
        best_single = max(h.confidence for h in members)
        agreement_boost = 1.0 - (1.0 - best_single) ** len(members)
        confidence = max(best_single, min(1.0, agreement_boost))

        candidates.append(MergedObjectCandidate(
            candidate_id=f"obj-{members[0].label}-{group_index:04d}",
            label=members[0].label,
            position=Vec3(cx, cy, cz),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            source_hypotheses=tuple(members),
            confidence=confidence,
        ))

    candidates.sort(key=lambda c: (c.label, c.position.x, c.position.y, c.position.z))
    return candidates
