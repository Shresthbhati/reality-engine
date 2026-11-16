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

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.segmentation.interface import SegmentationResult, SegmentedRegion

log = logging.getLogger(__name__)

MODEL_NAME = "grounding-dino-tiny"
LICENSE = "Apache-2.0"
#: One query per class (a combined prompt returns merged, unassignable labels such as "tv stand bookshelf").
SCENE_PROMPTS: Tuple[str, ...] = (
    "sofa", "armchair", "chair", "dining table", "coffee table", "side table", "desk", "bed", "wardrobe", "bookshelf",
    "dresser", "nightstand", "tv stand", "television", "floor lamp", "rug", "potted plant", "picture frame", "curtain",
)
#: Extra phrasings queried for a class; every hit is reported under the CANONICAL label. Grounding DINO is sensitive to the
#: exact words: on a ten-photo living room the plain prompt "coffee table" found nothing above threshold in 0/6 views, while
#: "wooden coffee table" found it in 6/6 at 0.81. Only variants that measured better than the plain prompt are listed.
PROMPT_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "coffee table": ("wooden coffee table",),
    "floor lamp": ("standing lamp",),     # 6/6 views at 0.73 vs 0.60
    "rug": ("area rug",),                 # 6/6 at 0.75 vs 5/6 at 0.65
    "bookshelf": ("shelving unit",),      # 6/6 vs 5/6
    "side table": ("end table",),         # 6/6 at 0.63 vs 3/6 at 0.44
}
_CANONICAL_OF_VARIANT = {v: k for k, vs in PROMPT_VARIANTS.items() for v in vs}


def query_prompts(prompts: Sequence[str]) -> Tuple[str, ...]:
    """``prompts`` plus the variants of every class in it (what is actually sent to the detector)."""
    return tuple(prompts) + tuple(v for p in prompts for v in PROMPT_VARIANTS.get(p, ()))


#: boxes covering more than this share of the frame are scene-level false positives; tiny ones are noise
MAX_BOX_SHARE = 0.80
MIN_BOX_SHARE = 0.004
NMS_IOU = 0.5
#: every class is queried on its own, so each prompt returns *something* (a "bed" in a living room at ~0.4): demand more
MIN_SCORE = 0.45
#: boxes of DIFFERENT classes on the same pixels are one object seen with two names: the higher score keeps it
CROSS_CLASS_IOU = 0.6
#: an object seen in a single photograph needs at least this confidence (score x share of valid depth) to enter the world
MIN_SINGLE_VIEW_CONFIDENCE = 0.6
#: with this many photographs or more, an object must be confirmed by >= 2 of them (one view cannot localise anything in 3D)
MANY_VIEWS = 4
#: scene box = 2nd..98th percentile of the sparse points, grown by this share of its diagonal
SCENE_MARGIN = 0.25
#: nothing in a room is longer than this share of the room's own extent (depth blow-ups are)
MAX_SIDE_SHARE = 0.7
#: below this many sparse points there is no scene to measure; the scene filters then do nothing
MIN_SCENE_POINTS = 50
#: WorldIR type per label; anything not listed stays UNKNOWN rather than being guessed
ENTITY_TYPE_BY_LABEL: Dict[str, str] = {
    **{k: "furniture" for k in ("sofa", "armchair", "chair", "dining table", "coffee table", "side table", "desk", "bed",
                                "wardrobe", "bookshelf", "dresser", "nightstand", "tv stand", "television", "floor lamp",
                                "rug")},
    "potted plant": "vegetation",
}
#: keep pixels whose depth is within this fraction of the box's foreground depth. At 0.35 a sofa 4 m away kept a 2.8 m deep
#: slab: floor in front and wall behind were lifted with it and a 6-view sofa came out 7 m long (measured on a ten-photo room).
DEPTH_BAND = 0.15
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
    """Drop weak, scene-sized and speck-sized boxes; then NMS, highest score first: a box is dropped when it overlaps a
    kept box of the same class (IoU > NMS_IOU) or of any other class (IoU > CROSS_CLASS_IOU)."""
    sw, sh = src_size
    kept: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    for label, score, box in sorted(raw, key=lambda r: -r[1]):
        if score < MIN_SCORE:
            continue
        share = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / float(sw * sh)
        if not (MIN_BOX_SHARE <= share <= MAX_BOX_SHARE):
            continue
        if any(_iou(k[2], box) > (NMS_IOU if k[0] == label else CROSS_CLASS_IOU) for k in kept):
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


#: depth band applied INSIDE a SAM 2 mask: only strips depth outliers, since the mask already follows the object's outline
SAM_DEPTH_BAND = 0.35


def sam_refined_mask(depth_values: np.ndarray, sam_mask: np.ndarray, band: float = SAM_DEPTH_BAND) -> Optional[np.ndarray]:
    """Boolean mask at the depth map's resolution from a full-resolution SAM 2 instance mask: the mask's valid-depth pixels
    within ``band`` of their median depth. None when the mask holds too little valid depth."""
    import cv2

    dh, dw = depth_values.shape
    small = cv2.resize(sam_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
    valid = small & np.isfinite(depth_values) & (depth_values > 0)
    if int(valid.sum()) < MIN_MASK_PIXELS:
        return None
    ref = float(np.median(depth_values[valid]))
    keep = valid & (np.abs(depth_values - ref) <= band * ref)
    return keep if int(keep.sum()) >= MIN_MASK_PIXELS else None


def scene_box(points) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Robust (lo, hi) of the sparse points (2nd..98th percentile per axis); None when there are too few to define one."""
    pts = np.array([tuple(getattr(p, "position", p)) for p in points], dtype=float)
    if len(pts) < MIN_SCENE_POINTS:
        return None
    return np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)


def filter_implausible(candidates, box, n_views: int, min_confidence: float = MIN_SINGLE_VIEW_CONFIDENCE):
    """Drop candidates that cannot be real objects of this scene; returns (kept, {reason: count}).

    Reasons, in order: ``outside_scene`` (centre beyond the sparse points' box grown by SCENE_MARGIN), ``scene_sized``
    (longest side > MAX_SIDE_SHARE of the scene's longest axis extent), ``single_view`` (one photograph, though the capture has
    >= MANY_VIEWS), ``weak_single_view`` (one photograph in a small capture and confidence < ``min_confidence``)."""
    from collections import Counter

    dropped: Counter = Counter()
    kept = []
    if box is not None:
        lo, hi = box
        diag = float(np.linalg.norm(hi - lo))
        longest = float(np.max(hi - lo))
        lo_m, hi_m = lo - SCENE_MARGIN * diag, hi + SCENE_MARGIN * diag
    for c in candidates:
        if box is not None:
            centre = np.array([c.position.x, c.position.y, c.position.z])
            side = max(c.bounds_max.x - c.bounds_min.x, c.bounds_max.y - c.bounds_min.y, c.bounds_max.z - c.bounds_min.z)
            if bool(np.any(centre < lo_m) or np.any(centre > hi_m)):
                dropped["outside_scene"] += 1
                continue
            if side > MAX_SIDE_SHARE * longest:
                dropped["scene_sized"] += 1
                continue
        if len(c.evidence_ids) < 2:
            if n_views >= MANY_VIEWS:
                dropped["single_view"] += 1
                continue
            if c.confidence < min_confidence:
                dropped["weak_single_view"] += 1
                continue
        kept.append(c)
    return kept, dict(dropped)


class GroundedSceneSegmenter:
    """``segment(evidence)`` like ``MaskRCNNDetector``, for the multi-view perception stage.

    ``depth_by_id``: evidence id -> DepthMap. A view without a depth map is skipped (never lifted against a guess)."""

    name = MODEL_NAME

    def __init__(self, depth_by_id: Dict[str, object], prompts: Sequence[str] = SCENE_PROMPTS,
                 mask_source: str = "depth_band", sam: object = None) -> None:
        if mask_source not in ("depth_band", "sam2"):
            raise ValueError(f"mask_source must be 'depth_band' or 'sam2', got {mask_source!r}")
        self._depth = depth_by_id
        self._prompts = tuple(prompts)
        self._mask_source = mask_source
        self._sam = sam  # a Sam2BoxPrompter (or any object with masks_for_boxes); created on first use for 'sam2'
        #: per run: how many masks came from SAM 2 vs fell back to the depth band (recorded, never hidden)
        self.mask_counts = {"sam2": 0, "depth_band": 0}
        self.mask_fallback: Optional[str] = None  # why SAM 2 was not used, when it was requested and unavailable

    def segment(self, evidence) -> List[SegmentationResult]:
        from engine.pipeline.open_vocab_furniture import _grounded_raw, item_path

        results: List[SegmentationResult] = []
        for item in evidence:
            eid = getattr(item, "id", None) or item.evidence_id
            depth = self._depth.get(eid)
            if depth is None:
                continue
            raw, src_size = _grounded_raw(item_path(item), query_prompts(self._prompts))
            raw = [(_CANONICAL_OF_VARIANT.get(label, label), score, box) for label, score, box in raw]
            values = np.asarray(depth.values, dtype=float)
            regions: List[SegmentedRegion] = []
            boxes = clean_boxes(raw, src_size)
            sam_masks = None
            if self._mask_source == "sam2" and boxes:
                from engine.core.image_io import open_oriented
                from perception.segmentation.sam2_box_prompter import Sam2BoxPrompter, Sam2Unavailable

                try:
                    if self._sam is None:
                        self._sam = Sam2BoxPrompter()
                    sam_masks = self._sam.masks_for_boxes(open_oriented(item_path(item)).convert("RGB"), [b[2] for b in boxes])
                except Sam2Unavailable as exc:
                    # the checkpoint is missing: use the depth band for the rest of the run, loudly and on the record
                    log.warning("SAM 2 unavailable, falling back to depth-band masks: %s", exc)
                    self._mask_source, self.mask_fallback = "depth_band", str(exc)
            for j, (label, score, box) in enumerate(boxes):
                mask = sam_refined_mask(values, sam_masks[j]) if sam_masks is not None else None
                if mask is not None:
                    self.mask_counts["sam2"] += 1
                else:
                    mask = depth_refined_mask(values, box, src_size)
                    if mask is not None:
                        self.mask_counts["depth_band"] += 1
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
        if self._sam is not None and hasattr(self._sam, "unload"):
            self._sam.unload()
