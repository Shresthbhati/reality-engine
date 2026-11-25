"""Architectural segmentation engine with mask validity, boundary quality,
overlap/occlusion analysis, and structural boundary extraction.

Uses ``Sam2BoxPrompter`` (``facebook/sam2.1-hiera-small`` via ``transformers``)
for box-prompted instance masks, and never invents masks or disguises
bounding boxes as segmentation masks when SAM2 is unavailable.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from perception.detection.normalized import NormalizedDetection
from perception.segmentation.sam2_box_prompter import Sam2BoxPrompter, Sam2Unavailable

STRUCTURAL_CATEGORIES = frozenset(
    {
        "wall",
        "floor",
        "ceiling",
        "column",
        "beam",
        "doorway",
        "door",
        "window",
        "window frame",
        "stairs",
        "railing",
        "balcony",
        "structural opening",
        "archway",
    }
)


@dataclass(frozen=True)
class EvaluatedMask:
    detection_id: str
    label: str
    canonical_label: str
    mask: np.ndarray | None
    validity: str  # "valid" | "degenerate_empty" | "degenerate_box_fill" | "unavailable"
    mask_area_px: int
    box_area_px: float
    mask_to_box_ratio: float
    boundary_sharpness: float
    segmentation_uncertainty: float
    occluded_by: Tuple[str, ...] = ()
    model_name: str = ""
    model_version: str = ""
    device: str = "cpu"
    note: str = ""


@dataclass(frozen=True)
class StructuralBoundaryCandidate:
    source_detection_id: str
    target_detection_id: str
    source_label: str
    target_label: str
    contact_pixels: int
    mean_gradient_strength: float
    bbox_px: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ArchitecturalSegmentationOutput:
    image_id: str
    masks: Tuple[EvaluatedMask, ...]
    structural_boundaries: Tuple[StructuralBoundaryCandidate, ...]
    overlap_pairs: Tuple[Dict[str, Any], ...]
    model_name: str
    status: str  # "ok" | "empty" | "unavailable" | "failed"
    inference_ms: float = 0.0
    error: str | None = None


def evaluate_mask_quality(
    mask: np.ndarray | None,
    bbox_px: Tuple[float, float, float, float],
    rgb: np.ndarray,
) -> Tuple[str, int, float, float, float, float]:
    """Compute validity, area, mask-to-box ratio, boundary sharpness, and uncertainty."""
    x0, y0, x1, y1 = bbox_px
    box_area = max(1.0, (x1 - x0) * (y1 - y0))
    if mask is None:
        return "unavailable", 0, box_area, 0.0, 0.0, 1.0

    m = np.asarray(mask, dtype=bool)
    area = int(m.sum())
    if area < 16:
        return "degenerate_empty", area, box_area, round(area / box_area, 4), 0.0, 0.95

    ratio = area / box_area
    if ratio > 0.995 and box_area > 400.0:
        # Suspiciously exact box fill -- flag explicitly rather than trusting blindly
        validity = "degenerate_box_fill"
        base_unc = 0.75
    elif ratio < 0.03:
        validity = "degenerate_empty"
        base_unc = 0.85
    else:
        validity = "valid"
        base_unc = 0.15

    # Boundary sharpness: mean Sobel gradient magnitude along 1px mask perimeter
    u8 = m.astype(np.uint8) * 255
    eroded = cv2.erode(u8, np.ones((3, 3), np.uint8), iterations=1)
    border = (u8 > 0) & (eroded == 0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    if border.any():
        sharpness = float(np.mean(grad[border])) / 255.0
    else:
        sharpness = 0.0
    sharpness = round(min(1.0, max(0.0, sharpness)), 4)
    unc = round(min(1.0, max(0.05, base_unc + 0.25 * (1.0 - min(1.0, sharpness * 2.5)))), 4)
    return validity, area, round(box_area, 2), round(ratio, 4), sharpness, unc


def refine_binary_mask(mask: np.ndarray, *, min_component_px: int = 24) -> np.ndarray:
    """Fill tiny interior holes (3x3 close) and drop tiny disconnected speckle islands."""
    u8 = np.asarray(mask, dtype=np.uint8)
    if u8.size == 0 or int(u8.sum()) == 0:
        return u8.astype(bool)
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return closed.astype(bool)
    out = np.zeros_like(closed, dtype=bool)
    for lid in range(1, num_labels):
        if int(stats[lid, cv2.CC_STAT_AREA]) >= min_component_px:
            out |= labels == lid
    return out if out.any() else closed.astype(bool)


def compare_masks_iou(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> Optional[float]:
    """Compute IoU between two masks from different models (None when either is absent)."""
    if mask_a is None or mask_b is None:
        return None
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return round(inter / union, 4) if union > 0 else 1.0


def segment_normalized_detections(
    rgb: np.ndarray,
    detections: Sequence[NormalizedDetection],
    *,
    image_id: str = "",
    depth_map: np.ndarray | None = None,
    prompter: Sam2BoxPrompter | None = None,
    unload_after: bool = True,
) -> ArchitecturalSegmentationOutput:
    """Run SAM2 box-prompted segmentation on `detections` and evaluate mask quality."""
    if not detections:
        return ArchitecturalSegmentationOutput(
            image_id=image_id,
            masks=(),
            structural_boundaries=(),
            overlap_pairs=(),
            model_name="sam2.1-hiera-small",
            status="empty",
        )

    from PIL import Image

    pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    owned_prompter = prompter is None
    sam = prompter or Sam2BoxPrompter()
    t0 = time.perf_counter()
    try:
        raw_masks = sam.masks_for_boxes(pil, [d.bbox_px for d in detections])
        inf_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        device = str(getattr(sam, "_device", "cpu") or "cpu")
    except Sam2Unavailable as exc:
        return ArchitecturalSegmentationOutput(
            image_id=image_id,
            masks=tuple(
                EvaluatedMask(
                    detection_id=d.detection_id,
                    label=d.label,
                    canonical_label=d.canonical_label,
                    mask=None,
                    validity="unavailable",
                    mask_area_px=0,
                    box_area_px=round(max(1.0, (d.bbox_px[2] - d.bbox_px[0]) * (d.bbox_px[3] - d.bbox_px[1])), 2),
                    mask_to_box_ratio=0.0,
                    boundary_sharpness=0.0,
                    segmentation_uncertainty=1.0,
                    model_name="sam2.1-hiera-small",
                    note=str(exc),
                )
                for d in detections
            ),
            structural_boundaries=(),
            overlap_pairs=(),
            model_name="sam2.1-hiera-small",
            status="unavailable",
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return ArchitecturalSegmentationOutput(
            image_id=image_id,
            masks=(),
            structural_boundaries=(),
            overlap_pairs=(),
            model_name="sam2.1-hiera-small",
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if owned_prompter and unload_after and hasattr(sam, "unload"):
            sam.unload()

    evaluated: List[EvaluatedMask] = []
    for det, raw_m in zip(detections, raw_masks):
        m = refine_binary_mask(raw_m) if raw_m is not None else None
        validity, area, box_area, ratio, sharpness, unc = evaluate_mask_quality(m, det.bbox_px, rgb)
        evaluated.append(
            EvaluatedMask(
                detection_id=det.detection_id,
                label=det.label,
                canonical_label=det.canonical_label,
                mask=m if validity == "valid" else m,
                validity=validity,
                mask_area_px=area,
                box_area_px=box_area,
                mask_to_box_ratio=ratio,
                boundary_sharpness=sharpness,
                segmentation_uncertainty=unc,
                model_name="sam2.1-hiera-small",
                model_version="ee5bba1d",
                device=device,
            )
        )

    # Overlap, occlusion, and structural boundary extraction
    overlap_pairs: List[Dict[str, Any]] = []
    boundaries: List[StructuralBoundaryCandidate] = []
    occluded_map: Dict[str, List[str]] = {e.detection_id: [] for e in evaluated}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    for i in range(len(evaluated)):
        mi = evaluated[i].mask
        if mi is None or evaluated[i].mask_area_px == 0:
            continue
        dil_i = cv2.dilate(mi.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        for j in range(i + 1, len(evaluated)):
            mj = evaluated[j].mask
            if mj is None or evaluated[j].mask_area_px == 0:
                continue
            inter = int((mi & mj).sum())
            if inter > 0:
                union = int((mi | mj).sum())
                iou = round(inter / max(1, union), 4)
                overlap_pairs.append(
                    {
                        "a": evaluated[i].detection_id,
                        "b": evaluated[j].detection_id,
                        "a_label": evaluated[i].canonical_label,
                        "b_label": evaluated[j].canonical_label,
                        "intersection_px": inter,
                        "iou": iou,
                    }
                )
                # Occlusion via depth if available
                if depth_map is not None:
                    di_vals = depth_map[mi]
                    dj_vals = depth_map[mj]
                    di_fin = di_vals[np.isfinite(di_vals)]
                    dj_fin = dj_vals[np.isfinite(dj_vals)]
                    if di_fin.size > 0 and dj_fin.size > 0:
                        med_i = float(np.median(di_fin))
                        med_j = float(np.median(dj_fin))
                        if med_i + 0.05 < med_j:
                            occluded_map[evaluated[j].detection_id].append(evaluated[i].detection_id)
                        elif med_j + 0.05 < med_i:
                            occluded_map[evaluated[i].detection_id].append(evaluated[j].detection_id)

            # Structural boundary contact between architectural elements
            if (
                evaluated[i].canonical_label in STRUCTURAL_CATEGORIES
                or evaluated[j].canonical_label in STRUCTURAL_CATEGORIES
            ):
                contact = dil_i & mj
                c_px = int(contact.sum())
                if c_px >= 12:
                    ys, xs = np.where(contact)
                    boundaries.append(
                        StructuralBoundaryCandidate(
                            source_detection_id=evaluated[i].detection_id,
                            target_detection_id=evaluated[j].detection_id,
                            source_label=evaluated[i].canonical_label,
                            target_label=evaluated[j].canonical_label,
                            contact_pixels=c_px,
                            mean_gradient_strength=round(float(np.mean(grad[contact])) / 255.0, 4),
                            bbox_px=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                        )
                    )

    final_masks = tuple(
        EvaluatedMask(
            **{**e.__dict__, "occluded_by": tuple(occluded_map.get(e.detection_id, ()))}
        )
        for e in evaluated
    )
    return ArchitecturalSegmentationOutput(
        image_id=image_id,
        masks=final_masks,
        structural_boundaries=tuple(boundaries),
        overlap_pairs=tuple(overlap_pairs),
        model_name="sam2.1-hiera-small",
        status="ok",
        inference_ms=inf_ms,
    )
