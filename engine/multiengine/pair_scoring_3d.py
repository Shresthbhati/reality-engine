"""Cross-view association from MEASURED 3D geometry.

Once detections carry a lifted position and dimensions in the same frame and scale state, "is this the same physical object seen from
another photograph?" has a direct geometric answer: do the two 3D boxes occupy the same space? This scorer answers it with the 3D box
IoU and leaves every pair it cannot judge to the base (epipolar / appearance) scorer, unchanged.

Honest limits, stated in every score's basis: the IoU threshold is an untuned starting value; boxes are axis-aligned extents lifted from
2D boxes (background pixels inside a box inflate them); two DIFFERENT objects of the same class that genuinely overlap in space would be
merged (the world validator's duplicate-entity invariant is the backstop in the other direction). Pairs whose boxes are in different
scale states are never compared. Labels are handled by the association layer, not here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from engine.multiengine.association import Candidate, PairScore
from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.lift_stage import lifted_members

MIN_IOU = 0.25  # untuned starting value
FLOOR_FRAC = 0.1  # an extent is never below this fraction of the box's largest extent (a fronto-parallel lift has ~0 depth extent)

BaseScorer = Callable[[Sequence[Candidate], EvidenceMatrix], Iterable[PairScore]]


def iou_3d(lo_a, hi_a, lo_b, hi_b) -> float:
    inter = np.prod(np.maximum(0.0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)))
    union = np.prod(np.asarray(hi_a) - np.asarray(lo_a)) + np.prod(np.asarray(hi_b) - np.asarray(lo_b)) - inter
    return 0.0 if union <= 0 else float(inter / union)


def _box(matrix: EvidenceMatrix, oid: str, floor_frac: float):
    """(lo, hi, scale_state) of one detection's lifted box, or None when it has no accepted position + dimensions."""
    ids = lifted_members(matrix, [matrix.store.get(oid)]).get(oid, ())
    pos = dim = None
    for lid in ids:
        o = matrix.store.get(lid)
        if o.quantity == "position" and pos is None:
            pos = o
        elif o.quantity == "dimensions" and dim is None:
            dim = o
    if pos is None or dim is None:
        return None
    p, d = np.asarray(pos.predicted["value"], float), np.asarray(dim.predicted["value"], float)
    d = np.maximum(d, floor_frac * float(d.max()))
    state = str(pos.predicted.get("scale_state") or "")
    if len(p) != 3 or len(d) != 3 or not state or state != str(dim.predicted.get("scale_state") or ""):
        return None
    return p - d / 2, p + d / 2, state


def make_geometry_pair_scorer(base: BaseScorer | None = None, *, min_iou: float = MIN_IOU, floor_frac: float = FLOOR_FRAC) -> BaseScorer:
    def scorer(cands: Sequence[Candidate], matrix: EvidenceMatrix) -> Iterable[PairScore]:
        base_by_pair = {}
        for s in (base(cands, matrix) if base is not None else []):
            base_by_pair[tuple(sorted((s.a, s.b)))] = s
        boxes = {c.observation_id: _box(matrix, c.observation_id, floor_frac) for c in cands}
        out: list[PairScore] = []
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                if a.source_id == b.source_id:
                    continue
                key = tuple(sorted((a.observation_id, b.observation_id)))
                ba, bb = boxes[a.observation_id], boxes[b.observation_id]
                if ba is None or bb is None or ba[2] != bb[2]:  # cannot judge by 3D: leave the pair to the base scorer
                    if key in base_by_pair:
                        out.append(base_by_pair[key])
                    continue
                iou = iou_3d(ba[0], ba[1], bb[0], bb[1])
                basis: dict[str, Any] = {
                    "method": "3D box IoU of lifted axis-aligned boxes (same frame and scale state)", "iou_3d": round(iou, 4),
                    "min_iou": min_iou, "scale_state": ba[2], "base_epipolar": dict(base_by_pair[key].basis) if key in base_by_pair else None,
                    "limitation": "untuned threshold; extents come from 2D boxes and can include background; same-class objects that truly "
                                  "overlap in space would be merged",
                }
                if iou >= min_iou:
                    out.append(PairScore(a.observation_id, b.observation_id, iou, basis))
                else:
                    out.append(PairScore(a.observation_id, b.observation_id, 0.0, {**basis, "refused": f"3D IoU {iou:.3f} below minimum {min_iou}"}))
        return out

    return scorer
