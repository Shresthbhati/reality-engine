"""Scene-object segmenter for the multi-view perception stage: Grounding DINO boxes -> depth-refined masks.

The multi-view stage lifts *masks* through each view's depth map. COCO Mask R-CNN has no coffee table, TV stand, rug or
potted plant, so a living room came back with zero objects. Grounding DINO takes the vocabulary as text; it yields boxes,
and a box lifted as-is would drag the wall behind the object into its 3D extent. So each box is refined against the view's
own depth: pixels whose depth is close to the box's central (foreground) depth stay, the rest are dropped.

Honest limits, recorded rather than hidden: the mask is a depth-band cut of a box, not a learned instance mask; a flat
object spanning a large depth range (a rug) keeps only the band near its centre. Model absence raises -- it never looks
like "nothing detected". The model is loaded from the local Hugging Face cache only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.segmentation.interface import SegmentationResult, SegmentedRegion

MODEL_NAME = "grounding-dino-tiny"
LICENSE = "Apache-2.0"
#: One query per class (a combined prompt returns merged, unassignable labels such as "tv stand bookshelf").
SCENE_PROMPTS: Tuple[str, ...] = (
    "sofa", "armchair", "chair", "dining table", "coffee table", "side table", "desk", "bed", "wardrobe", "bookshelf",
    "dresser", "nightstand", "tv stand", "television", "floor lamp", "rug", "potted plant", "picture frame", "curtain",
)
#: boxes covering more than this share of the frame are scene-level false positives; tiny ones are noise
MAX_BOX_SHARE = 0.80
MIN_BOX_SHARE = 0.004
NMS_IOU = 0.5
#: keep pixels whose depth is within this fraction of the box's foreground depth
DEPTH_BAND = 0.35
MIN_MASK_PIXELS = 20


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clean_boxes(
    raw: Sequence[Tuple[str, float, Tuple[float, float, float, float]]], src_size: Tuple[int, int]
) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    """Drop scene-sized and speck-sized boxes, then class-aware NMS (highest score wins)."""
    sw, sh = src_size
    kept: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    for label, score, box in sorted(raw, key=lambda r: -r[1]):
        share = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / float(sw * sh)
        if not (MIN_BOX_SHARE <= share <= MAX_BOX_SHARE):
            continue
        if any(k[0] == label and _iou(k[2], box) > NMS_IOU for k in kept):
            continue
        kept.append((label, score, box))
    return kept


def depth_refined_mask(
    depth_values: np.ndarray, box_xyxy: Tuple[float, float, float, float], src_size: Tuple[int, int],
    band: float = DEPTH_BAND,
) -> Optional[np.ndarray]:
    """Boolean mask at the depth map's resolution: the box (scaled from the source image) restricted to pixels whose depth
    is within ``band`` of the box's central depth. None when the box holds too little valid depth to say anything."""
    dh, dw = depth_values.shape
    sw, sh = src_size
    x0 = int(np.clip(np.floor(box_xyxy[0] * dw / sw), 0, dw - 1))
    x1 = int(np.clip(np.ceil(box_xyxy[2] * dw / sw), x0 + 1, dw))
    y0 = int(np.clip(np.floor(box_xyxy[1] * dh / sh), 0, dh - 1))
    y1 = int(np.clip(np.ceil(box_xyxy[3] * dh / sh), y0 + 1, dh))
    sub = depth_values[y0:y1, x0:x1]
    valid = np.isfinite(sub) & (sub > 0)
    if int(valid.sum()) < MIN_MASK_PIXELS:
        return None
    h, w = sub.shape
    centre = sub[h // 4: h - h // 4 or h, w // 4: w - w // 4 or w]
    centre = centre[np.isfinite(centre) & (centre > 0)]
    ref = float(np.median(centre if centre.size else sub[valid]))
    keep = valid & (np.abs(sub - ref) <= band * ref)
    if int(keep.sum()) < MIN_MASK_PIXELS:
        return None
    mask = np.zeros((dh, dw), dtype=bool)
    mask[y0:y1, x0:x1] = keep
    return mask


class GroundedSceneSegmenter:
    """``segment(evidence)`` like ``MaskRCNNDetector``, for the multi-view perception stage.

    ``depth_by_id``: evidence id -> DepthMap. A view without a depth map is skipped (never lifted against a guess)."""

    name = MODEL_NAME

    def __init__(self, depth_by_id: Dict[str, object], prompts: Sequence[str] = SCENE_PROMPTS) -> None:
        self._depth = depth_by_id
        self._prompts = tuple(prompts)

    def segment(self, evidence) -> List[SegmentationResult]:
        from engine.pipeline.open_vocab_furniture import _grounded_raw, item_path

        results: List[SegmentationResult] = []
        for item in evidence:
            eid = getattr(item, "id", None) or item.evidence_id
            depth = self._depth.get(eid)
            if depth is None:
                continue
            raw, src_size = _grounded_raw(item_path(item), self._prompts)
            values = np.asarray(depth.values, dtype=float)
            regions: List[SegmentedRegion] = []
            for j, (label, score, box) in enumerate(clean_boxes(raw, src_size)):
                mask = depth_refined_mask(values, box, src_size)
                if mask is None:
                    continue
                regions.append(SegmentedRegion(
                    region_id=f"seg-{eid}-{j:03d}", evidence_id=eid, label=label, mask=mask.tolist(),
                    confidence=float(min(1.0, max(0.0, score)))))
            results.append(SegmentationResult(evidence_id=eid, regions=regions, model_name=MODEL_NAME))
        return results

    def unload(self) -> None:
        from engine.pipeline.open_vocab_furniture import unload_open_vocab_furniture

        unload_open_vocab_furniture()
