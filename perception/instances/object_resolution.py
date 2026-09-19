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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from engine.math import Vec3
from perception.instances.appearance import AppearanceDescriptor, appearance_similarity
from perception.instances.epipolar import (
    DEFAULT_EPIPOLAR_TOLERANCE_PX,
    epipolar_consistent,
)
from perception.instances.lifting import ObjectHypothesis3D
from reconstruction.calibration.camera import PinholeCamera

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


def merge_hypotheses(
    hypotheses: List[ObjectHypothesis3D],
    distance_threshold_m: float = DEFAULT_MERGE_DISTANCE_M,
    cameras: Optional[Dict[str, PinholeCamera]] = None,
    appearance: Optional[Dict[str, AppearanceDescriptor]] = None,
    appearance_similarity_min: float = DEFAULT_APPEARANCE_SIMILARITY_MIN,
    epipolar_tolerance_px: float = DEFAULT_EPIPOLAR_TOLERANCE_PX,
) -> List[MergedObjectCandidate]:
    """Cluster same-label, spatially-close hypotheses into candidates.

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
