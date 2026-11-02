"""High-detail inspection for small objects: coarse-to-fine, targeted high-resolution tiles.

A whole-photo detector at ~1000 px sees a wall socket as ~15 pixels and misses it. So:
  pass 1 (coarse)  the whole image, downscaled, finds large classes and WHERE the walls/regions of interest are;
  pass 2 (fine)    full-resolution overlapping tiles, restricted to the regions of interest, run ONLY the small-class
                   prompts from the taxonomy -- the expensive work is spent where small fittings can be.
Detections from both passes are mapped to taxonomy classes, de-duplicated across tile overlaps (per-class NMS) and
carry the pass that produced them. A label that maps to no class is reported, never invented into one.
The detector is any object with ``detect(image_rgb, queries)`` (perception.detection.backends); none is bundled, and
without one this module returns an explicit "unavailable" result instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from perception.entities import taxonomy

Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SmallDetection:
    class_id: str
    box: Box                       # pixels in the ORIGINAL image
    score: float
    stage: str                     # "coarse" | "fine"
    raw_label: str


@dataclass
class InspectionResult:
    detections: List[SmallDetection]
    tiles_run: int = 0
    tiles_total: int = 0
    unmapped_labels: List[str] = field(default_factory=list)
    status: str = "ok"
    note: str = ""


def tile_boxes(width: int, height: int, tile: int = 640, overlap: float = 0.25) -> List[Tuple[int, int, int, int]]:
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, width - tile) + step, step))
    ys = list(range(0, max(1, height - tile) + step, step))
    out = []
    for y in ys:
        for x in xs:
            x0, y0 = min(x, max(0, width - tile)), min(y, max(0, height - tile))
            out.append((x0, y0, min(width, x0 + tile), min(height, y0 + tile)))
    return sorted(set(out))


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: Sequence[SmallDetection], iou: float = 0.5) -> List[SmallDetection]:
    """Per-class non-maximum suppression; the higher score wins and a fine-pass box is preferred on a tie."""
    keep: List[SmallDetection] = []
    for d in sorted(dets, key=lambda d: (-d.score, d.stage != "fine")):
        if all(k.class_id != d.class_id or _iou(k.box, d.box) < iou for k in keep):
            keep.append(d)
    return keep


def _intersects(t: Tuple[int, int, int, int], regions: Sequence[Box]) -> bool:
    return any(min(t[2], r[2]) > max(t[0], r[0]) and min(t[3], r[3]) > max(t[1], r[1]) for r in regions)


def _resize(img: np.ndarray, s: float) -> np.ndarray:
    import cv2

    return cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def inspect_small_objects(image_rgb: np.ndarray, detector, *, regions: Optional[Sequence[Box]] = None, tile: int = 640,
                          overlap: float = 0.25, coarse_max: int = 1024, min_score: float = 0.25) -> InspectionResult:
    """``regions``: pixel boxes (e.g. wall segments) the fine pass is limited to; None = the whole image."""
    if detector is None:
        return InspectionResult([], status="unavailable", note="no detector backend installed; nothing was inspected")
    ok, why = detector.available() if hasattr(detector, "available") else (True, "")
    if not ok:
        return InspectionResult([], status="unavailable", note=str(why))
    h, w = image_rgb.shape[:2]
    small_prompts = sorted({p for c in taxonomy.small_classes() for p in c.prompts})
    all_prompts = sorted({p for c in taxonomy.all_classes() for p in c.prompts})
    unmapped: List[str] = []
    dets: List[SmallDetection] = []

    def collect(raw, dx: float, dy: float, scale: float, stage: str) -> None:
        for r in raw:
            if r.score < min_score:
                continue
            cls = taxonomy.class_for_label(r.label)
            if cls is None:
                unmapped.append(r.label)
                continue
            b = r.box_xyxy
            dets.append(SmallDetection(cls.id, (b[0] / scale + dx, b[1] / scale + dy, b[2] / scale + dx, b[3] / scale + dy),
                                       float(r.score), stage, r.label))

    s = min(1.0, coarse_max / max(h, w))
    coarse = image_rgb if s >= 1.0 else _resize(image_rgb, s)
    collect(detector.detect(coarse, all_prompts), 0.0, 0.0, s, "coarse")

    tiles = tile_boxes(w, h, tile, overlap)
    run = [t for t in tiles if regions is None or _intersects(t, regions)]
    for (x0, y0, x1, y1) in run:
        collect(detector.detect(image_rgb[y0:y1, x0:x1], small_prompts), float(x0), float(y0), 1.0, "fine")
    return InspectionResult(nms(dets), tiles_run=len(run), tiles_total=len(tiles), unmapped_labels=sorted(set(unmapped)))
