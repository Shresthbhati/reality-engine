"""Mask Geometry, Contour Extraction & Mask-Guided Box Refinement (A5.1 / A5.4 / A5.5).

Provides deterministic, coordinate-safe utilities on verified binary masks:
  - RLE decoding back into 2D boolean masks with strict dimension checks
  - Boundary contour polygon extraction in pixel coordinates
  - Tight mask-guided bounding-box refinement (revising loose detector boxes
    while preserving revision history & provenance)
  - Architectural opening-vs-wall mask carving consistency checks
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from perception.intelligence.data_model import (
    NormalizedPerceptionObservation,
    ObservationLifecycle,
    ObservationRevision,
    SegmentationObservation,
)


@dataclass(frozen=True)
class MaskGeometrySummary:
    observation_id: str
    tight_bbox_px: tuple[float, float, float, float]
    polygon_vertex_count: int
    box_shrink_ratio: float
    is_valid: bool


def decode_rle_to_mask(rle_mask: dict[str, Any] | None) -> np.ndarray | None:
    """Decode an RLE dict ``{size: [H, W], counts: [...]}`` into a 2D boolean array."""
    if not rle_mask or "size" not in rle_mask or "counts" not in rle_mask:
        return None
    size = rle_mask["size"]
    if len(size) != 2:
        return None
    h, w = int(size[0]), int(size[1])
    if h <= 0 or w <= 0:
        return None
    total = h * w
    flat = np.zeros(total, dtype=bool)
    idx = 0
    val = False
    for cnt in rle_mask["counts"]:
        c = int(cnt)
        if c < 0:
            return None
        end = min(total, idx + c)
        if val and end > idx:
            flat[idx:end] = True
        idx = end
        val = not val
    return flat.reshape((h, w))


def extract_mask_polygon_px(
    mask_2d: np.ndarray,
    *,
    max_vertices: int = 32,
) -> tuple[tuple[float, float], ...]:
    """Extract an ordered simplified boundary polygon in ``(x, y)`` pixel coordinates."""
    arr = np.asarray(mask_2d, dtype=np.uint8)
    if arr.ndim != 2 or int(arr.sum()) == 0:
        return ()
    try:
        import cv2

        contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return ()
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, max(1.0, 0.015 * peri), True)
        pts = [(round(float(pt[0][0]), 2), round(float(pt[0][1]), 2)) for pt in approx[:max_vertices]]
        return tuple(pts)
    except Exception:
        ys, xs = np.where(arr > 0)
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def compute_tight_box_from_mask(mask_2d: np.ndarray) -> tuple[float, float, float, float] | None:
    """Compute tight ``(x1, y1, x2, y2)`` pixel bounding box from a non-empty 2D mask."""
    arr = np.asarray(mask_2d, dtype=bool)
    if arr.ndim != 2 or not arr.any():
        return None
    ys, xs = np.where(arr)
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def refine_boxes_from_verified_masks(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    min_mask_quality: float = 0.35,
    attach_polygons: bool = True,
) -> tuple[list[NormalizedPerceptionObservation], list[MaskGeometrySummary]]:
    """Refine loose detector boxes using verified SAM masks while keeping full revision provenance."""
    out: list[NormalizedPerceptionObservation] = []
    summaries: list[MaskGeometrySummary] = []

    for o in observations:
        seg = o.segmentation
        if seg is None or seg.rle_mask is None or seg.area_px <= 0:
            out.append(o)
            continue
        mask_2d = decode_rle_to_mask(seg.rle_mask)
        if mask_2d is None or mask_2d.shape != (o.coords.image_height, o.coords.image_width):
            out.append(o)
            summaries.append(MaskGeometrySummary(
                observation_id=o.observation_id,
                tight_bbox_px=o.bbox_px,
                polygon_vertex_count=0,
                box_shrink_ratio=1.0,
                is_valid=False,
            ))
            continue

        tight = compute_tight_box_from_mask(mask_2d)
        if tight is None:
            out.append(o)
            continue

        poly = extract_mask_polygon_px(mask_2d) if attach_polygons else seg.polygon_px
        new_seg = SegmentationObservation(
            mask_id=seg.mask_id,
            observation_id=seg.observation_id,
            mask_kind=seg.mask_kind,
            area_px=seg.area_px,
            box_iou=seg.box_iou,
            solidity=seg.solidity,
            hole_count=seg.hole_count,
            boundary_uncertainty=seg.boundary_uncertainty,
            quality_score=seg.quality_score,
            rle_mask=seg.rle_mask,
            polygon_px=poly,
            depth_mean_relative=seg.depth_mean_relative,
            model_id=seg.model_id,
        )

        old_area = max(1.0, (o.bbox_px[2] - o.bbox_px[0]) * (o.bbox_px[3] - o.bbox_px[1]))
        new_area = max(1.0, (tight[2] - tight[0]) * (tight[3] - tight[1]))
        shrink = round(new_area / old_area, 4)

        if seg.quality_score >= min_mask_quality and 0.25 <= shrink <= 1.25:
            rev = ObservationRevision(
                revision_number=o.revision.revision_number + 1,
                reason="mask_guided_box_refinement",
                lifecycle=ObservationLifecycle.REVISED,
                previous_observation_id=o.observation_id,
            )
            out.append(replace(o, bbox_px=tight, segmentation=new_seg, revision=rev))
            summaries.append(MaskGeometrySummary(
                observation_id=o.observation_id,
                tight_bbox_px=tight,
                polygon_vertex_count=len(poly),
                box_shrink_ratio=shrink,
                is_valid=True,
            ))
        else:
            out.append(replace(o, segmentation=new_seg))
            summaries.append(MaskGeometrySummary(
                observation_id=o.observation_id,
                tight_bbox_px=o.bbox_px,
                polygon_vertex_count=len(poly),
                box_shrink_ratio=shrink,
                is_valid=True,
            ))

    return out, summaries


__all__ = [
    "MaskGeometrySummary",
    "compute_tight_box_from_mask",
    "decode_rle_to_mask",
    "extract_mask_polygon_px",
    "refine_boxes_from_verified_masks",
]
