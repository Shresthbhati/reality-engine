"""Team B — Mask Refinement Engine.

Provides post-processing intelligence for segmentation masks produced by
SAM2, SAM-HQ, Mask R-CNN, and similar models.

Capabilities:
  - Boundary quality scoring (sharpness, smoothness, edge alignment)
  - Mask decomposition into semantic subparts
  - Semantic vs instance mask reconciliation
  - Boundary uncertainty field computation
  - RLE and polygon serialization helpers
  - Morphological cleaning (small-hole fill, isolated-pixel removal)

Design principles:
  - Never fabricates mask content — all outputs derived from input mask pixels.
  - Preserves uncertainty: low-quality masks are marked, not silently improved.
  - Coordinate-system safe: all outputs carry explicit (image_width, image_height).
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("reality.segmentation.mask_refinement")


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------
def encode_rle(binary_mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask (H x W bool) to COCO RLE format.

    Args:
        binary_mask: Boolean numpy array of shape (H, W).

    Returns:
        Dict with keys ``counts`` (list[int]) and ``size`` ([H, W]).
    """
    flat = binary_mask.flatten(order="F").astype(np.uint8)
    counts: list[int] = []
    current = 0
    current_val = 0
    for px in flat:
        if px == current_val:
            current += 1
        else:
            counts.append(current)
            current = 1
            current_val = px
    counts.append(current)
    # COCO RLE starts with the run of 0s
    if binary_mask.flatten(order="F")[0] == 1:
        counts.insert(0, 0)
    h, w = binary_mask.shape
    return {"counts": counts, "size": [h, w]}


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    """Decode a COCO RLE dict to a binary mask (H x W bool).

    Args:
        rle: Dict with ``counts`` and ``size`` keys.

    Returns:
        Boolean numpy array of shape (H, W).
    """
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        flat[idx : idx + run] = val
        idx += run
        val = 1 - val
    return flat.reshape((h, w), order="F").astype(bool)


def mask_to_polygon(binary_mask: np.ndarray, *, simplify_epsilon: float = 1.5) -> list[list[float]]:
    """Convert a binary mask to a list of polygon vertex coordinates.

    Returns the largest contour as a flat [x1, y1, x2, y2, ...] list.
    Falls back to bounding-box polygon when OpenCV is unavailable.

    Args:
        binary_mask: Boolean numpy array (H x W).
        simplify_epsilon: Polygon simplification tolerance in pixels.

    Returns:
        Flat list [x0, y0, x1, y1, ...] of polygon vertices, or [] on failure.
    """
    try:
        import cv2  # noqa: PLC0415
        mask_u8 = binary_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        simplified = cv2.approxPolyDP(largest, simplify_epsilon, closed=True)
        return simplified.flatten().tolist()
    except ImportError:
        # Fallback: return bounding-box polygon
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        if not rows.any():
            return []
        y1, y2 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
        x1, x2 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
        return [x1, y1, x2, y1, x2, y2, x1, y2]


# ---------------------------------------------------------------------------
# Mask quality metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaskQualityReport:
    """Quality assessment for a single segmentation mask."""
    mask_id: str
    pixel_coverage: float          # fraction of image area covered
    edge_sharpness: float          # 0–1, higher = crisper edges
    fill_ratio: float              # filled pixels / bounding box area
    aspect_ratio: float            # width / height of tight bounding box
    boundary_pixel_count: int      # number of boundary pixels
    has_holes: bool                # interior disconnected voids
    is_degenerate: bool            # True if mask is empty or single pixel
    quality_score: float           # composite 0–1


def _compute_edge_sharpness(binary_mask: np.ndarray) -> float:
    """Estimate boundary sharpness from gradient magnitude statistics.

    Uses finite differences on the mask to approximate a Sobel-like measure.
    Returns a value in [0, 1] where 1 = perfectly sharp rectangular boundary.
    """
    if binary_mask.sum() == 0:
        return 0.0
    m = binary_mask.astype(np.float32)
    # Finite differences
    dx = np.abs(np.diff(m, axis=1, prepend=m[:, :1]))
    dy = np.abs(np.diff(m, axis=0, prepend=m[:1, :]))
    edge_pixels = (dx + dy > 0).sum()
    filled_pixels = binary_mask.sum()
    if filled_pixels == 0:
        return 0.0
    # Ideal circle has perimeter ≈ 2π√(area/π) = 2√(π·area)
    # Jagged boundary → lower ratio
    ideal_perimeter = 2.0 * math.sqrt(math.pi * float(filled_pixels))
    sharpness = min(1.0, ideal_perimeter / max(float(edge_pixels), 1.0))
    return sharpness


def compute_mask_quality(
    binary_mask: np.ndarray,
    mask_id: str,
    image_width: int,
    image_height: int,
) -> MaskQualityReport:
    """Compute a full quality report for a binary segmentation mask.

    Args:
        binary_mask: Boolean numpy array (H x W).
        mask_id: Unique identifier for this mask.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        ``MaskQualityReport`` with quality metrics.
    """
    total_pixels = image_width * image_height
    filled = int(binary_mask.sum())

    if filled == 0:
        return MaskQualityReport(
            mask_id=mask_id, pixel_coverage=0.0, edge_sharpness=0.0,
            fill_ratio=0.0, aspect_ratio=1.0, boundary_pixel_count=0,
            has_holes=False, is_degenerate=True, quality_score=0.0,
        )

    # Bounding box
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    bbox_area = max(1, (rmax - rmin + 1) * (cmax - cmin + 1))
    bbox_w = max(1, cmax - cmin + 1)
    bbox_h = max(1, rmax - rmin + 1)

    # Boundary pixels
    m = binary_mask.astype(np.uint8)
    # Simple dilation minus original to get boundary
    boundary = np.zeros_like(m)
    boundary[:-1, :] |= (m[:-1, :] != m[1:, :])
    boundary[:, :-1] |= (m[:, :-1] != m[:, 1:])
    boundary_count = int(boundary.sum())

    # Holes: check if any background pixel is fully surrounded by mask
    # Approximate: if fill_ratio is high but perimeter is proportionally large → potential hole
    has_holes = False
    try:
        from scipy import ndimage  # noqa: PLC0415
        labeled_bg, n_bg = ndimage.label(~binary_mask)
        # Background regions not touching image border = holes
        border_labels: set[int] = set()
        border_labels.update(labeled_bg[0, :].tolist())
        border_labels.update(labeled_bg[-1, :].tolist())
        border_labels.update(labeled_bg[:, 0].tolist())
        border_labels.update(labeled_bg[:, -1].tolist())
        has_holes = any(i > 0 and i not in border_labels for i in range(1, n_bg + 1))
    except ImportError:
        has_holes = False

    pixel_coverage = filled / max(total_pixels, 1)
    fill_ratio = filled / bbox_area
    aspect_ratio = bbox_w / bbox_h
    edge_sharpness = _compute_edge_sharpness(binary_mask)
    is_degenerate = filled <= 4

    # Composite score (heuristic weights)
    quality_score = (
        0.35 * edge_sharpness
        + 0.30 * min(1.0, fill_ratio)
        + 0.20 * (1.0 - abs(1.0 - fill_ratio))  # penalize hollow masks
        + 0.15 * (1.0 if not is_degenerate else 0.0)
    )
    quality_score = max(0.0, min(1.0, quality_score))

    return MaskQualityReport(
        mask_id=mask_id,
        pixel_coverage=pixel_coverage,
        edge_sharpness=edge_sharpness,
        fill_ratio=fill_ratio,
        aspect_ratio=aspect_ratio,
        boundary_pixel_count=boundary_count,
        has_holes=has_holes,
        is_degenerate=is_degenerate,
        quality_score=quality_score,
    )


# ---------------------------------------------------------------------------
# Boundary uncertainty field
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundaryUncertaintyField:
    """Sparse representation of mask boundary uncertainty.

    Regions with high uncertainty (soft edges, gradients) are listed
    with their local uncertainty score. Downstream consumers may choose
    to widen or shrink the mask boundary conservatively.
    """
    mask_id: str
    uncertain_pixel_count: int
    mean_uncertainty: float          # 0–1
    max_uncertainty: float           # 0–1
    uncertainty_map_rle: dict[str, Any] | None   # RLE of uncertainty > 0.5


def compute_boundary_uncertainty(
    soft_mask: np.ndarray,
    mask_id: str,
    *,
    soft_threshold: float = 0.5,
    uncertainty_band: float = 0.25,
) -> BoundaryUncertaintyField:
    """Estimate boundary uncertainty from a soft (float) mask.

    Pixels in the range [soft_threshold - uncertainty_band,
    soft_threshold + uncertainty_band] are considered uncertain.

    Args:
        soft_mask: Float mask (H x W) with values in [0, 1].
        mask_id: Mask identifier.
        soft_threshold: Threshold for hard mask binarization.
        uncertainty_band: Width of uncertainty region around threshold.

    Returns:
        ``BoundaryUncertaintyField`` with uncertainty statistics.
    """
    lo = soft_threshold - uncertainty_band
    hi = soft_threshold + uncertainty_band
    uncertain_map = ((soft_mask >= lo) & (soft_mask <= hi))
    n_uncertain = int(uncertain_map.sum())

    if n_uncertain == 0:
        return BoundaryUncertaintyField(
            mask_id=mask_id,
            uncertain_pixel_count=0,
            mean_uncertainty=0.0,
            max_uncertainty=0.0,
            uncertainty_map_rle=None,
        )

    # Uncertainty value: distance from threshold (inverted, centered)
    uncertainty_values = 1.0 - np.abs(soft_mask - soft_threshold) / uncertainty_band
    uncertainty_values = np.clip(uncertainty_values, 0.0, 1.0)
    uncertainty_values[~uncertain_map] = 0.0

    mean_unc = float(uncertainty_values[uncertain_map].mean()) if n_uncertain > 0 else 0.0
    max_unc = float(uncertainty_values.max())

    # RLE of uncertain pixels (binary)
    uncertain_rle = encode_rle(uncertain_map)

    return BoundaryUncertaintyField(
        mask_id=mask_id,
        uncertain_pixel_count=n_uncertain,
        mean_uncertainty=mean_unc,
        max_uncertainty=max_unc,
        uncertainty_map_rle=uncertain_rle,
    )


# ---------------------------------------------------------------------------
# Semantic/instance reconciliation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaskReconciliationResult:
    """Result of reconciling semantic and instance masks for the same region."""
    region_id: str
    semantic_label: str
    instance_label: str | None
    final_label: str
    final_mask_rle: dict[str, Any]
    agreement_iou: float
    resolution_strategy: str
    notes: str


def reconcile_semantic_and_instance_masks(
    semantic_mask: np.ndarray,
    semantic_label: str,
    instance_mask: np.ndarray | None,
    instance_label: str | None,
    region_id: str,
    image_width: int,
    image_height: int,
    *,
    min_iou_for_agreement: float = 0.65,
) -> MaskReconciliationResult:
    """Reconcile a semantic mask with an instance mask for the same spatial region.

    When both masks cover the same area (IoU ≥ threshold), use the instance
    mask (finer boundary). When they disagree, use intersection with semantic
    label as authority for category.

    Args:
        semantic_mask: Semantic segmentation output (H x W bool).
        semantic_label: Predicted semantic class.
        instance_mask: Instance segmentation mask (H x W bool) or None.
        instance_label: Predicted instance class or None.
        region_id: Unique region identifier.
        image_width: Image width for coordinate validation.
        image_height: Image height for coordinate validation.
        min_iou_for_agreement: IoU threshold for agreeing to use instance mask.

    Returns:
        ``MaskReconciliationResult`` describing the reconciled output.
    """
    if instance_mask is None or instance_label is None:
        # No instance mask — use semantic as-is
        return MaskReconciliationResult(
            region_id=region_id,
            semantic_label=semantic_label,
            instance_label=None,
            final_label=semantic_label,
            final_mask_rle=encode_rle(semantic_mask),
            agreement_iou=1.0,
            resolution_strategy="semantic_only",
            notes="No instance mask available; using semantic mask directly.",
        )

    # Compute IoU
    intersection = (semantic_mask & instance_mask).sum()
    union = (semantic_mask | instance_mask).sum()
    iou = float(intersection) / max(float(union), 1.0)

    if iou >= min_iou_for_agreement:
        # Good agreement — use instance mask (finer boundaries)
        return MaskReconciliationResult(
            region_id=region_id,
            semantic_label=semantic_label,
            instance_label=instance_label,
            final_label=semantic_label,  # semantic label is authoritative
            final_mask_rle=encode_rle(instance_mask),
            agreement_iou=iou,
            resolution_strategy="instance_preferred",
            notes=f"IoU={iou:.2f} ≥ {min_iou_for_agreement:.2f}; instance mask preferred for boundary quality.",
        )
    else:
        # Disagreement — use intersection for conservative mask
        intersection_mask = semantic_mask & instance_mask
        if intersection_mask.sum() > 0:
            final_mask = intersection_mask
            strategy = "intersection"
            notes = f"IoU={iou:.2f} < {min_iou_for_agreement:.2f}; using mask intersection (conservative)."
        else:
            final_mask = semantic_mask
            strategy = "semantic_fallback"
            notes = f"IoU={iou:.2f} — no overlap; falling back to semantic mask."

        return MaskReconciliationResult(
            region_id=region_id,
            semantic_label=semantic_label,
            instance_label=instance_label,
            final_label=semantic_label,
            final_mask_rle=encode_rle(final_mask),
            agreement_iou=iou,
            resolution_strategy=strategy,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Subpart decomposition
# ---------------------------------------------------------------------------
_SUBPART_LABELS: dict[str, list[str]] = {
    "chair": ["seat", "backrest", "armrest", "leg", "cushion"],
    "sofa": ["seat", "backrest", "armrest", "cushion", "leg"],
    "table": ["top", "leg", "shelf", "drawer"],
    "desk": ["top", "leg", "drawer", "keyboard_tray"],
    "cabinet": ["door", "shelf", "handle", "body"],
    "wardrobe": ["door", "shelf", "rod", "handle", "body"],
    "bed": ["mattress", "headboard", "footboard", "frame", "leg"],
    "door": ["panel", "frame", "handle", "hinge", "lock"],
    "window": ["frame", "pane", "sill", "handle"],
    "refrigerator": ["door", "handle", "shelf", "freezer_door"],
    "bookshelf": ["shelf", "side_panel", "back_panel", "support"],
    "tv_stand": ["top", "shelf", "door", "leg"],
}


@dataclass(frozen=True)
class SubpartHypothesis:
    """A predicted subpart within a parent object mask."""
    subpart_id: str
    parent_object_id: str
    subpart_label: str
    relative_region: str    # "top" | "center" | "bottom" | "left" | "right"
    confidence: float
    bbox_px: tuple[float, float, float, float]  # subpart bbox within full image


def decompose_into_subparts(
    parent_label: str,
    parent_bbox_px: tuple[float, float, float, float],
    parent_object_id: str,
    image_width: int,
    image_height: int,
    *,
    min_confidence: float = 0.40,
) -> list[SubpartHypothesis]:
    """Generate structural subpart hypotheses from an object bounding box.

    Uses geometric heuristics (top/bottom/left/right regions of the bbox)
    to propose plausible subpart locations for known object categories.

    Args:
        parent_label: Canonical label of the parent object.
        parent_bbox_px: (x1, y1, x2, y2) of the parent in image pixels.
        parent_object_id: Identifier of the parent observation.
        image_width: Image width for clamping.
        image_height: Image height for clamping.
        min_confidence: Minimum subpart hypothesis confidence to include.

    Returns:
        List of ``SubpartHypothesis`` for the parent object.
    """
    subpart_labels = _SUBPART_LABELS.get(parent_label.lower())
    if not subpart_labels:
        return []

    x1, y1, x2, y2 = parent_bbox_px
    pw = x2 - x1
    ph = y2 - y1
    if pw <= 0 or ph <= 0:
        return []

    # Heuristic region layout per object type
    _region_specs: dict[str, dict[str, tuple[float, float, float, float]]] = {
        "chair": {
            "seat":     (0.1, 0.5, 0.9, 0.75),
            "backrest": (0.1, 0.0, 0.9, 0.5),
            "armrest":  (0.0, 0.3, 0.2, 0.7),
            "leg":      (0.1, 0.75, 0.9, 1.0),
            "cushion":  (0.1, 0.45, 0.9, 0.7),
        },
        "sofa": {
            "seat":     (0.0, 0.55, 1.0, 0.85),
            "backrest": (0.0, 0.0, 1.0, 0.55),
            "armrest":  (0.0, 0.1, 0.15, 0.9),
            "cushion":  (0.1, 0.5, 0.9, 0.85),
            "leg":      (0.1, 0.85, 0.9, 1.0),
        },
        "table": {
            "top":      (0.0, 0.0, 1.0, 0.25),
            "leg":      (0.1, 0.25, 0.9, 1.0),
            "shelf":    (0.1, 0.5, 0.9, 0.75),
            "drawer":   (0.1, 0.1, 0.9, 0.3),
        },
        "door": {
            "panel":    (0.05, 0.05, 0.95, 0.9),
            "frame":    (0.0, 0.0, 1.0, 1.0),
            "handle":   (0.75, 0.4, 0.95, 0.65),
            "hinge":    (0.0, 0.1, 0.1, 0.5),
            "lock":     (0.75, 0.45, 0.95, 0.6),
        },
        "window": {
            "frame":    (0.0, 0.0, 1.0, 1.0),
            "pane":     (0.05, 0.05, 0.95, 0.9),
            "sill":     (0.0, 0.85, 1.0, 1.0),
            "handle":   (0.8, 0.45, 0.95, 0.6),
        },
    }

    region_map = _region_specs.get(parent_label.lower(), {})
    result: list[SubpartHypothesis] = []

    for sp_label in subpart_labels:
        if sp_label in region_map:
            rx1_r, ry1_r, rx2_r, ry2_r = region_map[sp_label]
        else:
            # Default: center half
            rx1_r, ry1_r, rx2_r, ry2_r = 0.1, 0.1, 0.9, 0.9

        sp_x1 = max(0.0, min(float(image_width), x1 + rx1_r * pw))
        sp_y1 = max(0.0, min(float(image_height), y1 + ry1_r * ph))
        sp_x2 = max(0.0, min(float(image_width), x1 + rx2_r * pw))
        sp_y2 = max(0.0, min(float(image_height), y1 + ry2_r * ph))

        if sp_x2 <= sp_x1 or sp_y2 <= sp_y1:
            continue

        # Simple confidence: scale of the subpart relative to parent
        sp_area = (sp_x2 - sp_x1) * (sp_y2 - sp_y1)
        parent_area = pw * ph
        relative_size = sp_area / max(parent_area, 1.0)
        confidence = min(0.85, 0.3 + 0.7 * relative_size)

        if confidence < min_confidence:
            continue

        # Region label from vertical position
        cy_rel = (ry1_r + ry2_r) / 2
        if cy_rel < 0.33:
            region = "top"
        elif cy_rel > 0.66:
            region = "bottom"
        else:
            region = "center"

        sp_id = hashlib.sha1(
            f"{parent_object_id}:{sp_label}:{sp_x1:.0f}:{sp_y1:.0f}".encode()
        ).hexdigest()[:8]

        result.append(SubpartHypothesis(
            subpart_id=f"sp_{sp_id}",
            parent_object_id=parent_object_id,
            subpart_label=sp_label,
            relative_region=region,
            confidence=confidence,
            bbox_px=(sp_x1, sp_y1, sp_x2, sp_y2),
        ))

    return result


# ---------------------------------------------------------------------------
# Morphological cleaning
# ---------------------------------------------------------------------------
def clean_mask_morphologically(
    binary_mask: np.ndarray,
    *,
    min_component_pixels: int = 20,
    fill_hole_max_size: int = 200,
) -> np.ndarray:
    """Remove small isolated components and fill small holes in a mask.

    Args:
        binary_mask: Boolean numpy array (H x W).
        min_component_pixels: Components smaller than this are removed.
        fill_hole_max_size: Holes smaller than this area (pixels) are filled.

    Returns:
        Cleaned boolean mask of the same shape.
    """
    if binary_mask.sum() == 0:
        return binary_mask.copy()

    result = binary_mask.copy()

    try:
        from scipy import ndimage  # noqa: PLC0415

        # Remove small foreground components
        labeled, n_comp = ndimage.label(result)
        for c in range(1, n_comp + 1):
            comp_size = int((labeled == c).sum())
            if comp_size < min_component_pixels:
                result[labeled == c] = False

        # Fill small holes
        inverted = ~result
        labeled_holes, n_holes = ndimage.label(inverted)
        # Find background label (touches image border)
        border = set()
        border.update(labeled_holes[0, :].tolist())
        border.update(labeled_holes[-1, :].tolist())
        border.update(labeled_holes[:, 0].tolist())
        border.update(labeled_holes[:, -1].tolist())
        for h in range(1, n_holes + 1):
            if h in border:
                continue
            hole_size = int((labeled_holes == h).sum())
            if hole_size <= fill_hole_max_size:
                result[labeled_holes == h] = True

    except ImportError:
        # No scipy — return as-is
        pass

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
@dataclass
class MaskRefinementResult:
    """Complete result of mask refinement for one observation."""
    observation_id: str
    original_quality: MaskQualityReport
    final_mask_rle: dict[str, Any]
    boundary_uncertainty: BoundaryUncertaintyField | None
    subpart_hypotheses: list[SubpartHypothesis]
    cleaning_applied: bool
    notes: list[str]


def refine_mask(
    observation_id: str,
    binary_mask: np.ndarray,
    label: str,
    bbox_px: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    *,
    soft_mask: np.ndarray | None = None,
    min_quality_for_subparts: float = 0.30,
    apply_morphological_cleaning: bool = True,
) -> MaskRefinementResult:
    """Run the full mask refinement pipeline for one observation.

    Steps:
      1. Assess initial mask quality.
      2. Apply morphological cleaning if enabled.
      3. Compute boundary uncertainty (if soft_mask provided).
      4. Decompose into subpart hypotheses.

    Args:
        observation_id: Identifier of the parent observation.
        binary_mask: Input binary mask (H x W bool).
        label: Canonical label of the detected object.
        bbox_px: (x1, y1, x2, y2) in image pixels.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        soft_mask: Optional float probability mask for uncertainty analysis.
        min_quality_for_subparts: Minimum quality score to generate subparts.
        apply_morphological_cleaning: Whether to apply morphological cleaning.

    Returns:
        ``MaskRefinementResult`` with all refinement outputs.
    """
    notes: list[str] = []

    # Initial quality
    quality = compute_mask_quality(binary_mask, observation_id, image_width, image_height)

    # Morphological cleaning
    working_mask = binary_mask.copy()
    cleaning_applied = False
    if apply_morphological_cleaning and not quality.is_degenerate:
        cleaned = clean_mask_morphologically(working_mask)
        if cleaned.sum() != working_mask.sum():
            cleaning_applied = True
            notes.append(
                f"Morphological cleaning changed pixel count: "
                f"{working_mask.sum()} → {cleaned.sum()}"
            )
        working_mask = cleaned

    # Boundary uncertainty
    boundary_unc: BoundaryUncertaintyField | None = None
    if soft_mask is not None:
        boundary_unc = compute_boundary_uncertainty(soft_mask, observation_id)
        if boundary_unc.mean_uncertainty > 0.6:
            notes.append(
                f"High boundary uncertainty (mean={boundary_unc.mean_uncertainty:.2f}) — "
                f"treat mask edges as approximate."
            )

    # Subpart hypotheses
    subparts: list[SubpartHypothesis] = []
    if quality.quality_score >= min_quality_for_subparts:
        subparts = decompose_into_subparts(
            label, bbox_px, observation_id, image_width, image_height
        )
        if subparts:
            notes.append(f"Decomposed into {len(subparts)} subpart hypotheses.")
    else:
        notes.append(
            f"Quality score {quality.quality_score:.2f} < {min_quality_for_subparts:.2f} "
            "— skipping subpart decomposition."
        )

    return MaskRefinementResult(
        observation_id=observation_id,
        original_quality=quality,
        final_mask_rle=encode_rle(working_mask),
        boundary_uncertainty=boundary_unc,
        subpart_hypotheses=subparts,
        cleaning_applied=cleaning_applied,
        notes=notes,
    )
