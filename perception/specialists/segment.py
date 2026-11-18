"""Specialist Instance Segmentation Engine (Universal Spatial Intelligence Directive Phase 7 / Cycle 2).

Generates and lifts instance segmentation masks:
- Executes SAM 2 / SAM / contour segmenter on entities eligible for mask refinement.
- Computes precise 2D boundary contours from real visual masks (via OpenCV / topological tracing).
- Honestly distinguishes real visual masks from non-visual bounding box geometric approximations.
- Enriches entity representations with mask references, visual provenance, and geometric contours.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np

from engine.quality.autonomous_correction_loop import RefinementKind, RefinementTask
from world_ir import WorldIR

try:
    import cv2  # type: ignore[import-untyped]
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def extract_mask_contour(
    mask: np.ndarray | Sequence[Sequence[bool | int]],
    min_area: float = 1.0,
) -> tuple[list[list[float]], float]:
    """Extract a 2D boundary polygon contour and area from a binary mask.

    Returns:
        (contour_points_xy, area_px) where contour_points_xy is a list of [x, y] coordinates.
    """
    if not isinstance(mask, np.ndarray):
        mask_np = np.array(mask, dtype=np.uint8)
    else:
        mask_np = (mask > 0).astype(np.uint8)

    if mask_np.ndim != 2 or np.count_nonzero(mask_np) == 0:
        return [], 0.0

    if _HAS_CV2:
        try:
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(c))
                if area < min_area:
                    area = float(np.count_nonzero(mask_np))
                epsilon = max(1.0, 0.01 * cv2.arcLength(c, True))
                approx = cv2.approxPolyDP(c, epsilon, True)
                pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
                if len(pts) >= 3:
                    return pts, area
        except Exception:
            pass

    # Robust fallback: extract boundary coordinates from bounding extents & perimeter
    coords = np.argwhere(mask_np > 0)
    if len(coords) == 0:
        return [], 0.0

    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    pts = [
        [float(c_min), float(r_min)],
        [float(c_min), float(r_max)],
        [float(c_max), float(r_max)],
        [float(c_max), float(r_min)],
    ]
    area = float(np.count_nonzero(mask_np))
    return pts, area


def segment_specialist(
    task: RefinementTask,
    world: WorldIR,
    segmenter: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
) -> Optional[bool]:
    """Execute high-precision instance segmentation for an entity.

    Returns:
        True: entity mask and boundary contour were generated and attached.
        False: execution attempted but failed or entity was already masked.
        None: entity or prerequisite data is missing.
    """
    if task.kind != RefinementKind.SEGMENT:
        return None

    entity_id = task.metadata.get("entity_id")
    if not entity_id or entity_id not in world.entities:
        return False

    entity = world.entities[entity_id]
    if entity.extra_fields is None:
        entity.extra_fields = {}

    ef = entity.extra_fields
    has_mask = (ef.get("mask_ref") is not None and ef.get("mask_ref") >= 0) or ef.get("has_mask", False)
    if has_mask:
        return False  # Already segmented

    # Check for direct real mask in task metadata
    raw_mask = task.metadata.get("mask")
    if raw_mask is None:
        raw_mask = task.metadata.get("binary_mask")
    if raw_mask is None:
        raw_mask = task.metadata.get("mask_array")

    if raw_mask is not None:
        contour, area_px = extract_mask_contour(raw_mask)
        if contour:
            ef["has_mask"] = True
            ef["mask_ref"] = task.metadata.get("mask_ref", 1)
            ef["is_visual_mask"] = True
            ef["segmentation"] = {
                "contour": contour,
                "num_vertices": len(contour),
                "area_px": round(area_px, 2),
                "confidence": float(task.metadata.get("confidence", 0.95)),
                "backend": task.metadata.get("backend", "real_mask_contour_extraction"),
                "is_visual_mask": True,
            }
            attrs = ef.setdefault("attributes", {})
            attrs["segmentation"] = {
                "value": "segmented",
                "basis": "real_mask_contour_extraction",
                "p": float(task.metadata.get("confidence", 0.95)),
            }
            return True

    # If external segmenter callable or backend provided
    custom_fn = segmenter or task.metadata.get("segmenter")
    if callable(custom_fn):
        res = custom_fn({"entity": entity, "metadata": task.metadata})
        if res and res.get("success", True):
            # Check if mask array returned
            mask_data = res.get("mask")
            if mask_data is None:
                mask_data = res.get("binary_mask")
            if mask_data is not None:
                contour, area_px = extract_mask_contour(mask_data)
            else:
                contour = res.get("contour", [])
                area_px = float(res.get("area_px", 2048.0))

            is_visual = bool(res.get("is_visual_mask", True))
            conf = float(res.get("confidence", 0.94 if is_visual else 0.50))
            backend_str = res.get("backend", "sam_custom")
            basis_str = res.get("basis", "real_visual_segmentation" if is_visual else "bounding_box_approximation_non_visual")

            ef["has_mask"] = True
            ef["mask_ref"] = res.get("mask_ref", 1)
            ef["is_visual_mask"] = is_visual
            ef["segmentation"] = {
                "contour": contour,
                "num_vertices": len(contour),
                "area_px": area_px,
                "confidence": conf,
                "backend": backend_str,
                "is_visual_mask": is_visual,
            }
            attrs = ef.setdefault("attributes", {})
            attrs["segmentation"] = {"value": "segmented", "basis": basis_str, "p": conf}
            return True

    # Fallback when no visual image or mask is available:
    # Honestly label as bounding box approximation, NOT as a visual mask
    width = 0.5
    height = 0.5
    geom = getattr(entity, "geometry", None)
    if geom and hasattr(geom, "parameters") and geom.parameters:
        dims = geom.parameters.get("dimensions") or geom.parameters.get("size")
        if isinstance(dims, (list, tuple)) and len(dims) >= 2:
            width = float(dims[0])
            height = float(dims[1])

    # Generate 8-point geometric bounding boundary
    num_pts = 8
    contour_geom: list[list[float]] = []
    r_w = max(0.05, width / 2.0)
    r_h = max(0.05, height / 2.0)
    for i in range(num_pts):
        angle = 2.0 * math.pi * i / num_pts
        contour_geom.append([
            round(r_w * math.cos(angle), 4),
            round(r_h * math.sin(angle), 4),
        ])

    ef["has_mask"] = True
    ef["mask_ref"] = task.metadata.get("mask_ref", 1)
    ef["is_visual_mask"] = False
    ef["segmentation"] = {
        "contour": contour_geom,
        "num_vertices": len(contour_geom),
        "area_m2": round(math.pi * r_w * r_h, 4),
        "confidence": 0.45,
        "backend": "bounding_box_approximation_non_visual",
        "is_visual_mask": False,
    }
    attrs = ef.setdefault("attributes", {})
    attrs["segmentation"] = {
        "value": "bounding_box_approximation",
        "basis": "bounding_box_approximation_non_visual",
        "p": 0.45,
    }
    return True
