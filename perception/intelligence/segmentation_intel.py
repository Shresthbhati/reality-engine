"""Segmentation Intelligence Subsystem (A3.7).

Provides:
  - Mask quality scoring (solidity, mask-to-box IoU, hole count, boundary gradient alignment)
  - Overlapping instance separation (assigning contested pixels to highest-confidence/compact mask)
  - Mask kind classification: ``visible_object_mask``, ``occluded_object_hypothesis``,
    ``architectural_surface_mask``, ``structural_boundary``
  - Boundary uncertainty estimation and optional mask-to-depth association
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.multiengine.mask_codec import count_holes, encode_mask
from engine.multiengine.spatial_intelligence.detection_pipeline import (
    compute_box_iou,
    extract_mask_topology,
    extract_polygon_from_mask,
)
from perception.intelligence.architectural import BOUNDARY_CLASSES, PRIMARY_SURFACES
from perception.intelligence.data_model import SegmentationObservation


@dataclass(frozen=True)
class EvaluatedMask:
    observation: SegmentationObservation
    refined_mask: np.ndarray
    is_acceptable: bool
    rejection_reason: str | None = None


def classify_mask_kind(
    canonical_label: str,
    *,
    is_occluded: bool = False,
) -> str:
    lbl = canonical_label.lower().replace(" ", "_")
    if lbl in BOUNDARY_CLASSES:
        return "structural_boundary"
    if lbl in PRIMARY_SURFACES:
        return "architectural_surface_mask"
    if is_occluded:
        return "occluded_object_hypothesis"
    return "visible_object_mask"


def evaluate_instance_mask(
    mask_2d: np.ndarray,
    *,
    observation_id: str,
    box_px: tuple[float, float, float, float],
    canonical_label: str,
    image_rgb: np.ndarray | None = None,
    depth_map: np.ndarray | None = None,
    model_id: str = "sam",
    is_occluded: bool = False,
) -> EvaluatedMask:
    """Evaluate mask quality, consistency with box, boundary uncertainty, and depth statistics."""
    m = np.asarray(mask_2d, dtype=bool)
    topo = extract_mask_topology(m)
    area_px = int(topo["mask_area_px"])
    solidity = float(topo["solidity"])
    holes = int(topo["hole_count"])

    if not topo["is_valid"]:
        seg_obs = SegmentationObservation(
            mask_id=f"mask_{observation_id}",
            observation_id=observation_id,
            mask_kind=classify_mask_kind(canonical_label, is_occluded=is_occluded),
            area_px=area_px,
            box_iou=0.0,
            solidity=solidity,
            hole_count=holes,
            boundary_uncertainty=1.0,
            quality_score=0.0,
            model_id=model_id,
        )
        return EvaluatedMask(seg_obs, m, False, str(topo["validation_note"]))

    ys, xs = np.where(m)
    mask_box = (float(np.min(xs)), float(np.min(ys)), float(np.max(xs)) + 1.0, float(np.max(ys)) + 1.0)
    box_iou = float(compute_box_iou(box_px, mask_box))

    # Boundary uncertainty: higher when solidity is low or box_iou is poor, lower when RGB edges align
    edge_align = 0.5
    if image_rgb is not None and image_rgb.shape[:2] == m.shape[:2]:
        gray = (
            image_rgb[..., 0].astype(np.float32) * 0.299
            + image_rgb[..., 1].astype(np.float32) * 0.587
            + image_rgb[..., 2].astype(np.float32) * 0.114
        )
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        grad = np.hypot(gx, gy)
        # Mask perimeter pixels via 4-neighbor XOR
        perim = m ^ (
            np.roll(m, 1, axis=0) & np.roll(m, -1, axis=0) & np.roll(m, 1, axis=1) & np.roll(m, -1, axis=1)
        )
        if np.any(perim):
            edge_strength = float(np.mean(grad[perim]))
            edge_align = float(min(1.0, edge_strength / 25.0))

    quality = round(float(max(0.0, min(1.0, 0.45 * solidity + 0.35 * box_iou + 0.20 * edge_align))), 4)
    boundary_unc = round(float(max(0.0, min(1.0, 1.0 - quality))), 4)

    depth_mean: float | None = None
    if depth_map is not None and depth_map.shape[:2] == m.shape[:2]:
        vals = depth_map[m]
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            depth_mean = round(float(np.median(finite)), 4)

    poly = extract_polygon_from_mask(m) or ()
    rle = {**encode_mask(m), "holes": count_holes(m)}
    seg_obs = SegmentationObservation(
        mask_id=f"mask_{observation_id}",
        observation_id=observation_id,
        mask_kind=classify_mask_kind(canonical_label, is_occluded=is_occluded),
        area_px=area_px,
        box_iou=round(box_iou, 4),
        solidity=round(solidity, 4),
        hole_count=holes,
        boundary_uncertainty=boundary_unc,
        quality_score=quality,
        rle_mask=rle,
        polygon_px=poly,
        depth_mean_relative=depth_mean,
        model_id=model_id,
    )
    is_ok = quality >= 0.30 and box_iou >= 0.15
    return EvaluatedMask(seg_obs, m, is_ok, None if is_ok else "low_mask_box_consistency")


def separate_overlapping_masks(
    masks: Sequence[np.ndarray],
    scores: Sequence[float],
    *,
    is_structural: Sequence[bool] | None = None,
) -> list[np.ndarray]:
    """Resolve pixel ownership conflicts among overlapping object instance masks.

    Architectural background surfaces (walls/floors/ceilings) do not steal pixels
    from foreground objects; among foreground objects, higher-scoring smaller masks win.
    """
    if not masks:
        return []
    h, w = masks[0].shape[:2]
    flags = list(is_structural) if is_structural is not None else [False] * len(masks)
    out = [np.asarray(m, dtype=bool).copy() for m in masks]

    # Foreground objects claim pixels first in descending score order
    fg_indices = [i for i in range(len(masks)) if not flags[i]]
    fg_indices.sort(key=lambda i: (float(scores[i]), -int(np.sum(out[i]))), reverse=True)
    claimed = np.zeros((h, w), dtype=bool)
    for idx in fg_indices:
        m = out[idx] & (~claimed)
        out[idx] = m
        claimed |= m

    return out


__all__ = [
    "EvaluatedMask",
    "classify_mask_kind",
    "evaluate_instance_mask",
    "separate_overlapping_masks",
]
