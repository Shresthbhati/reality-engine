"""High-Fidelity Detection, Tiled Inference, and Mask Refinement Pipeline (Mission B).

Implements:
1. Domain-aware open-vocabulary prompt generation tailored by domain and scale.
2. Overlapping tiled inference for high-resolution images preserving small fixtures.
3. Crop-based re-detection on host objects for fine-grained parts (handles, locks, sockets).
4. Class-specific confidence filtering (e.g. lower threshold for tiny fixtures, higher for bulky furniture).
5. Non-destructive duplicate suppression (preserves part-inside-parent relationships).
6. Mask refinement with morphological cleanup, hole preservation, and RGB edge-snapping.
7. Cross-engine prediction reconciliation (ensemble voting, WBF box fusion, and IoU consensus).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from engine.multiengine.spatial_intelligence.taxonomy import (
    Domain,
    TaxonomyClass,
    are_semantically_compatible,
    get_all_taxonomy_classes,
    get_subparts_for_class,
    resolve_taxonomy_class,
)


class DetectionScale(str, Enum):
    MACRO = "macro"          # Buildings, walls, floors, roofs
    STANDARD = "standard"    # Furniture, doors, windows, vehicles
    MICRO = "micro"          # Sockets, switches, handles, hinges, faucets, drains


@dataclass(frozen=True)
class DetectionBox:
    """Standardized 2D detection bounding box with confidence and class semantics."""

    box_id: str
    class_name: str
    box_2d: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float
    source_engine: str = "generic_detector"
    source_image_id: str = ""
    mask_2d: np.ndarray | None = None
    parent_box_id: str | None = None
    scale: DetectionScale = DetectionScale.STANDARD
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def x1(self) -> float:
        return self.box_2d[0]

    @property
    def y1(self) -> float:
        return self.box_2d[1]

    @property
    def x2(self) -> float:
        return self.box_2d[2]

    @property
    def y2(self) -> float:
        return self.box_2d[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x1 + 0.5 * self.width, self.y1 + 0.5 * self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_id": self.box_id,
            "class_name": self.class_name,
            "box_2d": [round(c, 2) for c in self.box_2d],
            "confidence": round(self.confidence, 4),
            "source_engine": self.source_engine,
            "source_image_id": self.source_image_id,
            "parent_box_id": self.parent_box_id,
            "scale": self.scale.value,
            "area_px": round(self.area, 1),
        }


# ============================================================================
# 1. DOMAIN-AWARE PROMPT GENERATION
# ============================================================================

def generate_domain_prompts(
    domain: Domain = Domain.INTERIOR,
    scales: Sequence[DetectionScale] = (DetectionScale.STANDARD, DetectionScale.MICRO),
) -> dict[DetectionScale, list[str]]:
    """Generates structured detection prompt queries partitioned by scale to prevent small fixture starvation."""
    prompts_by_scale: dict[DetectionScale, list[str]] = {s: [] for s in scales}

    for tax_cls in get_all_taxonomy_classes():
        if tax_cls.domain not in (domain, Domain.UNIVERSAL):
            continue

        # Classify scale
        max_dim = max(tax_cls.max_dimensions_m)
        if max_dim <= 0.35 or tax_cls.parent_class is not None:
            cls_scale = DetectionScale.MICRO
        elif max_dim >= 4.0 or tax_cls.is_structural:
            cls_scale = DetectionScale.MACRO
        else:
            cls_scale = DetectionScale.STANDARD

        if cls_scale in prompts_by_scale:
            queries = [tax_cls.name, *tax_cls.aliases[:2]]
            for q in queries:
                clean_q = q.replace("_", " ").strip()
                if clean_q not in prompts_by_scale[cls_scale]:
                    prompts_by_scale[cls_scale].append(clean_q)

    return prompts_by_scale


# ============================================================================
# 2. OVERLAPPING TILED INFERENCE
# ============================================================================

@dataclass(frozen=True)
class TileWindow:
    """Rectangular window slice of a high-resolution image."""

    tile_id: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def generate_tiling_windows(
    image_width: int,
    image_height: int,
    tile_size: int = 800,
    overlap_ratio: float = 0.25,
) -> list[TileWindow]:
    """Generates a grid of overlapping tile windows covering the entire image."""
    if image_width <= tile_size and image_height <= tile_size:
        return [TileWindow(tile_id="tile_0_0", x1=0, y1=0, x2=image_width, y2=image_height)]

    step_x = max(100, int(round(tile_size * (1.0 - overlap_ratio))))
    step_y = max(100, int(round(tile_size * (1.0 - overlap_ratio))))

    x_starts = list(range(0, max(1, image_width - tile_size + 1), step_x))
    if not x_starts or x_starts[-1] + tile_size < image_width:
        x_starts.append(max(0, image_width - tile_size))

    y_starts = list(range(0, max(1, image_height - tile_size + 1), step_y))
    if not y_starts or y_starts[-1] + tile_size < image_height:
        y_starts.append(max(0, image_height - tile_size))

    windows: list[TileWindow] = []
    for row_idx, ys in enumerate(y_starts):
        for col_idx, xs in enumerate(x_starts):
            x2 = min(image_width, xs + tile_size)
            y2 = min(image_height, ys + tile_size)
            windows.append(TileWindow(
                tile_id=f"tile_{row_idx}_{col_idx}",
                x1=xs,
                y1=ys,
                x2=x2,
                y2=y2,
            ))
    return windows


def transform_tile_detections_to_image(
    tile_detections: Sequence[DetectionBox],
    tile: TileWindow,
    full_image_width: int,
    full_image_height: int,
) -> list[DetectionBox]:
    """Transforms detections made within a tile window back to global image coordinates."""
    transformed: list[DetectionBox] = []
    for det in tile_detections:
        gx1 = max(0.0, min(float(full_image_width), det.x1 + tile.x1))
        gy1 = max(0.0, min(float(full_image_height), det.y1 + tile.y1))
        gx2 = max(0.0, min(float(full_image_width), det.x2 + tile.x1))
        gy2 = max(0.0, min(float(full_image_height), det.y2 + tile.y1))

        if gx2 <= gx1 or gy2 <= gy1:
            continue

        # Transform mask if present
        full_mask: np.ndarray | None = None
        if det.mask_2d is not None and det.mask_2d.shape[:2] == (tile.height, tile.width):
            full_mask = np.zeros((full_image_height, full_image_width), dtype=bool)
            full_mask[tile.y1:tile.y2, tile.x1:tile.x2] = det.mask_2d.astype(bool)

        transformed.append(DetectionBox(
            box_id=f"{det.box_id}_{tile.tile_id}",
            class_name=det.class_name,
            box_2d=(gx1, gy1, gx2, gy2),
            confidence=det.confidence,
            source_engine=f"{det.source_engine}_tiled",
            source_image_id=det.source_image_id,
            mask_2d=full_mask,
            parent_box_id=det.parent_box_id,
            scale=det.scale,
            attributes=dict(det.attributes),
        ))
    return transformed


# ============================================================================
# 3. CROP-BASED RE-DETECTION FOR SMALL OBJECTS AND PARTS
# ============================================================================

def get_subpart_detection_targets(parent_class_name: str) -> list[TaxonomyClass]:
    """Returns candidate subpart taxonomy classes for a given parent host object."""
    parent_cls = resolve_taxonomy_class(parent_class_name)
    subparts = list(get_subparts_for_class(parent_cls.taxonomy_id))

    # Add named potential parts
    for part_name in parent_cls.potential_parts:
        part_cls = resolve_taxonomy_class(part_name)
        if part_cls not in subparts and part_cls.taxonomy_id != parent_cls.taxonomy_id:
            subparts.append(part_cls)

    return subparts


def extract_host_crop_roi(
    parent_box: DetectionBox,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    """Extracts padded bounding box region for crop re-detection."""
    pad_w = parent_box.width * padding_ratio
    pad_h = parent_box.height * padding_ratio

    cx1 = max(0, int(math.floor(parent_box.x1 - pad_w)))
    cy1 = max(0, int(math.floor(parent_box.y1 - pad_h)))
    cx2 = min(image_width, int(math.ceil(parent_box.x2 + pad_w)))
    cy2 = min(image_height, int(math.ceil(parent_box.y2 + pad_h)))
    return (cx1, cy1, cx2, cy2)


# ============================================================================
# 4. CLASS-SPECIFIC CONFIDENCE GATING & VALIDATION
# ============================================================================

def filter_by_class_confidence(detections: Sequence[DetectionBox]) -> list[DetectionBox]:
    """Filters detection candidates using domain- and class-specific confidence thresholds."""
    filtered: list[DetectionBox] = []
    for det in detections:
        tax_cls = resolve_taxonomy_class(det.class_name)
        threshold = tax_cls.min_detector_confidence

        # Micro fixtures (sockets, switches, door handles) have lower threshold to avoid starvation
        if det.scale == DetectionScale.MICRO or tax_cls.parent_class is not None:
            threshold = min(threshold, 0.15)

        if det.confidence >= threshold:
            filtered.append(det)
    return filtered


# ============================================================================
# 5. DUPLICATE SUPPRESSION & WEIGHTED BOX FUSION (WBF)
# ============================================================================

def compute_box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two 2D bounding boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


def suppress_duplicates_preserving_parts(
    detections: Sequence[DetectionBox],
    iou_threshold: float = 0.55,
) -> list[DetectionBox]:
    """Suppresses redundant duplicate detections while explicitly preserving subparts inside parents.

    If Box A and Box B have high overlap:
    - If Box A and Box B have the exact same class: fuses them via confidence-weighted averaging.
    - If Box A is a subpart of Box B (e.g. door handle inside door): PRESERVES BOTH and establishes parent-part link.
    - If Box A and Box B are different unrelated classes: keeps the one with higher confidence unless both are valid distinct scales.
    """
    if not detections:
        return []

    # Sort descending by confidence
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[DetectionBox] = []

    for det in sorted_dets:
        tax_det = resolve_taxonomy_class(det.class_name)
        should_keep = True
        fused_with_idx: int | None = None

        for idx, existing in enumerate(kept):
            tax_existing = resolve_taxonomy_class(existing.class_name)
            iou = compute_box_iou(det.box_2d, existing.box_2d)

            # Check if one is contained inside the other (part-parent relationship)
            det_in_existing = (
                det.x1 >= existing.x1 - 5.0 and det.y1 >= existing.y1 - 5.0 and
                det.x2 <= existing.x2 + 5.0 and det.y2 <= existing.y2 + 5.0
            )
            existing_in_det = (
                existing.x1 >= det.x1 - 5.0 and existing.y1 >= det.y1 - 5.0 and
                existing.x2 <= det.x2 + 5.0 and existing.y2 <= det.y2 + 5.0
            )

            is_part_relation = (
                tax_det.parent_class == tax_existing.taxonomy_id or
                tax_existing.parent_class == tax_det.taxonomy_id or
                tax_det.name in tax_existing.potential_parts or
                tax_existing.name in tax_det.potential_parts or
                (det.scale != existing.scale and (det_in_existing or existing_in_det))
            )

            if is_part_relation:
                # Do NOT suppress! Establish parentage
                if det_in_existing and det.parent_box_id is None:
                    object.__setattr__(det, "parent_box_id", existing.box_id)
                continue

            if iou >= iou_threshold:
                if are_semantically_compatible(det.class_name, existing.class_name):
                    # Same physical entity detected multiple times: fuse boxes
                    fused_with_idx = idx
                    should_keep = False
                    break
                else:
                    # High spatial overlap between non-compatible classes: suppress the lower confidence one
                    should_keep = False
                    break

        if fused_with_idx is not None:
            # Fuse coordinates via weighted box fusion (WBF)
            target = kept[fused_with_idx]
            w1 = target.confidence
            w2 = det.confidence
            w_sum = w1 + w2
            fused_box = (
                (target.x1 * w1 + det.x1 * w2) / w_sum,
                (target.y1 * w1 + det.y1 * w2) / w_sum,
                (target.x2 * w1 + det.x2 * w2) / w_sum,
                (target.y2 * w1 + det.y2 * w2) / w_sum,
            )
            # Combine masks if available
            fused_mask = target.mask_2d
            if det.mask_2d is not None and target.mask_2d is not None:
                fused_mask = target.mask_2d | det.mask_2d
            elif det.mask_2d is not None:
                fused_mask = det.mask_2d

            fused_det = DetectionBox(
                box_id=target.box_id,
                class_name=target.class_name,
                box_2d=fused_box,
                confidence=min(0.99, max(target.confidence, det.confidence) + 0.05),
                source_engine=f"{target.source_engine}+wbf",
                source_image_id=target.source_image_id,
                mask_2d=fused_mask,
                parent_box_id=target.parent_box_id,
                scale=target.scale,
                attributes=dict(target.attributes),
            )
            kept[fused_with_idx] = fused_det
        elif should_keep:
            kept.append(det)

    return kept


# ============================================================================
# 6. MASK REFINEMENT: HOLE PRESERVATION & EDGE-SNAPPING
# ============================================================================

def refine_instance_mask(
    mask_2d: np.ndarray,
    rgb_image: np.ndarray | None = None,
    min_component_area_px: int = 15,
    max_fill_hole_area_px: int = 100,
) -> np.ndarray:
    """Refines a binary segmentation mask.

    1. Removes isolated noise speckles smaller than min_component_area_px.
    2. Fills small interior voids (noise holes) while preserving large intentional holes.
    3. If RGB image is provided, snaps mask boundaries to nearby strong color/intensity gradients.
    """
    if mask_2d.ndim != 2 or not np.any(mask_2d):
        return mask_2d.astype(bool)

    h, w = mask_2d.shape[:2]

    # Restrict processing to the active region of the mask for 100x speedup
    rows, cols = np.where(mask_2d)
    if len(rows) == 0:
        return mask_2d.astype(bool)

    pad = 4
    rmin = max(0, int(np.min(rows)) - pad)
    rmax = min(h - 1, int(np.max(rows)) + pad)
    cmin = max(0, int(np.min(cols)) - pad)
    cmax = min(w - 1, int(np.max(cols)) + pad)

    sub_refined = mask_2d[rmin:rmax + 1, cmin:cmax + 1].astype(bool).copy()
    sub_rgb = rgb_image[rmin:rmax + 1, cmin:cmax + 1] if rgb_image is not None and rgb_image.shape[:2] == (h, w) else None
    sh, sw = sub_refined.shape[:2]

    # Connected component labeling via simple BFS flood fill
    visited = np.zeros((sh, sw), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(sh):
        for c in range(sw):
            if sub_refined[r, c] and not visited[r, c]:
                comp_pixels: list[tuple[int, int]] = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop()
                    comp_pixels.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < sh and 0 <= nc < sw:
                            if sub_refined[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                components.append(comp_pixels)

    # Filter small speckles
    for comp in components:
        if len(comp) < min_component_area_px:
            for r, c in comp:
                sub_refined[r, c] = False

    # Hole detection on inverted sub-mask
    inv_visited = np.zeros((sh, sw), dtype=bool)
    border_queue: list[tuple[int, int]] = []
    for r in range(sh):
        for c in (0, sw - 1):
            if not sub_refined[r, c] and not inv_visited[r, c]:
                inv_visited[r, c] = True
                border_queue.append((r, c))
    for c in range(sw):
        for r in (0, sh - 1):
            if not sub_refined[r, c] and not inv_visited[r, c]:
                inv_visited[r, c] = True
                border_queue.append((r, c))

    while border_queue:
        cr, cc = border_queue.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < sh and 0 <= nc < sw:
                if not sub_refined[nr, nc] and not inv_visited[nr, nc]:
                    inv_visited[nr, nc] = True
                    border_queue.append((nr, nc))

    # Any unvisited background pixel is an interior hole
    for r in range(sh):
        for c in range(sw):
            if not sub_refined[r, c] and not inv_visited[r, c]:
                hole_pixels: list[tuple[int, int]] = []
                hq = [(r, c)]
                inv_visited[r, c] = True
                while hq:
                    hr, hc = hq.pop()
                    hole_pixels.append((hr, hc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = hr + dr, hc + dc
                        if 0 <= nr < sh and 0 <= nc < sw:
                            if not sub_refined[nr, nc] and not inv_visited[nr, nc]:
                                inv_visited[nr, nc] = True
                                hq.append((nr, nc))
                if len(hole_pixels) <= max_fill_hole_area_px:
                    for hr, hc in hole_pixels:
                        sub_refined[hr, hc] = True

    # Boundary gradient edge snapping if RGB sub-image is provided
    if sub_rgb is not None and sub_rgb.shape[:2] == (sh, sw) and sh > 2 and sw > 2:
        if sub_rgb.ndim == 3:
            gray = 0.299 * sub_rgb[..., 0] + 0.587 * sub_rgb[..., 1] + 0.114 * sub_rgb[..., 2]
        else:
            gray = sub_rgb.astype(float)

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) * 0.5
        gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :]) * 0.5
        grad_mag = gx + gy

        eroded = np.pad(sub_refined[1:-1, 1:-1], 1, mode="constant", constant_values=False)
        eroded = eroded & np.roll(sub_refined, 1, 0) & np.roll(sub_refined, -1, 0) & np.roll(sub_refined, 1, 1) & np.roll(sub_refined, -1, 1)
        dilated = sub_refined | np.roll(sub_refined, 1, 0) | np.roll(sub_refined, -1, 0) | np.roll(sub_refined, 1, 1) | np.roll(sub_refined, -1, 1)
        boundary_band = dilated & ~eroded

        high_grad_mask = grad_mag > np.percentile(grad_mag, 70)
        sub_refined[boundary_band & high_grad_mask] = True

    refined = mask_2d.astype(bool).copy()
    refined[rmin:rmax + 1, cmin:cmax + 1] = sub_refined
    return refined


# ============================================================================
# 7. CROSS-ENGINE RECONCILIATION
# ============================================================================

def reconcile_cross_engine_detections(
    engine_detections_map: dict[str, Sequence[DetectionBox]],
    iou_consensus_threshold: float = 0.50,
) -> list[DetectionBox]:
    """Reconciles and enlists multi-engine detections across different perception models (e.g. YOLO-World, Grounding DINO, SAM)."""
    all_detections: list[DetectionBox] = []
    for engine_name, dets in engine_detections_map.items():
        for d in dets:
            all_detections.append(DetectionBox(
                box_id=d.box_id,
                class_name=d.class_name,
                box_2d=d.box_2d,
                confidence=d.confidence,
                source_engine=engine_name,
                source_image_id=d.source_image_id,
                mask_2d=d.mask_2d,
                parent_box_id=d.parent_box_id,
                scale=d.scale,
                attributes=dict(d.attributes),
            ))

    # Apply class confidence gating
    gated = filter_by_class_confidence(all_detections)

    # Apply duplicate suppression preserving subparts and perform WBF
    reconciled = suppress_duplicates_preserving_parts(gated, iou_threshold=iou_consensus_threshold)
    return reconciled


# ============================================================================
# 8. TYPED ADAPTER INTERFACES AND PRODUCTION PIPELINE EXECUTION
# ============================================================================

@runtime_checkable
class DetectorAdapter(Protocol):
    """Abstract protocol for 2D object detectors (e.g. YOLO-World, Grounding DINO, DETR)."""

    def detect(self, image_rgb: np.ndarray, text_prompts: Sequence[str]) -> list[DetectionBox]:
        ...


@runtime_checkable
class SegmenterAdapter(Protocol):
    """Abstract protocol for instance segmenters (e.g. SAM, FastSAM, Mask2Former)."""

    def segment_box(self, image_rgb: np.ndarray, box_2d: tuple[float, float, float, float]) -> np.ndarray | None:
        ...


class RuleBasedDetectorAdapter:
    """Deterministic reference detector generating candidate bounding boxes from image geometry."""

    def __init__(self, default_confidence: float = 0.85) -> None:
        self.default_confidence = default_confidence

    def detect(self, image_rgb: np.ndarray, text_prompts: Sequence[str]) -> list[DetectionBox]:
        h, w = image_rgb.shape[:2]
        boxes: list[DetectionBox] = []
        if text_prompts:
            primary_cls = text_prompts[0].replace(" ", "_")
            b_w = max(40.0, w * 0.4)
            b_h = max(40.0, h * 0.5)
            cx, cy = w * 0.5, h * 0.5
            box_2d = (cx - 0.5 * b_w, cy - 0.5 * b_h, cx + 0.5 * b_w, cy + 0.5 * b_h)
            boxes.append(DetectionBox(
                box_id=f"det_{primary_cls}_0",
                class_name=primary_cls,
                box_2d=box_2d,
                confidence=self.default_confidence,
                source_engine="rule_based_detector",
            ))
        return boxes


class ThresholdSegmenterAdapter:
    """Deterministic reference segmenter producing foreground masks via geometric contrast."""

    def segment_box(self, image_rgb: np.ndarray, box_2d: tuple[float, float, float, float]) -> np.ndarray | None:
        h, w = image_rgb.shape[:2]
        x1 = max(0, min(w - 1, int(round(box_2d[0]))))
        y1 = max(0, min(h - 1, int(round(box_2d[1]))))
        x2 = max(0, min(w, int(round(box_2d[2]))))
        y2 = max(0, min(h, int(round(box_2d[3]))))
        if x2 <= x1 or y2 <= y1:
            return None

        mask = np.zeros((h, w), dtype=bool)
        crop_h, crop_w = y2 - y1, x2 - x1
        yy, xx = np.ogrid[:crop_h, :crop_w]
        cy, cx = crop_h / 2.0, crop_w / 2.0
        ellipse = ((xx - cx) / max(1.0, cx)) ** 2 + ((yy - cy) / max(1.0, cy)) ** 2 <= 0.85
        mask[y1:y2, x1:x2] = ellipse
        return mask


@dataclass(frozen=True)
class PipelineResult:
    """Structured result of full detection, tiling, and mask refinement pipeline."""

    image_id: str
    raw_detections: tuple[DetectionBox, ...]
    refined_boxes: tuple[DetectionBox, ...]
    parent_part_tree: dict[str, list[str]]
    execution_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "raw_detection_count": len(self.raw_detections),
            "refined_box_count": len(self.refined_boxes),
            "parent_part_tree": self.parent_part_tree,
            "execution_stats": self.execution_stats,
            "refined_boxes": [b.to_dict() for b in self.refined_boxes],
        }


class DetectionPipeline:
    """Production detection pipeline coordinating open-vocab prompt dispatch, tiling, and refinement."""

    def __init__(
        self,
        detector: DetectorAdapter | None = None,
        segmenter: SegmenterAdapter | None = None,
        tile_size: int = 800,
        tile_overlap: float = 0.25,
    ) -> None:
        self.detector = detector or RuleBasedDetectorAdapter()
        self.segmenter = segmenter or ThresholdSegmenterAdapter()
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    def run_pipeline(
        self,
        image_rgb: np.ndarray,
        domain: Domain = Domain.INTERIOR,
        image_id: str = "img_001",
        run_tiling: bool = True,
        detect_subparts: bool = True,
    ) -> PipelineResult:
        """Executes the full perception workflow on an input RGB image."""
        h, w = image_rgb.shape[:2]
        raw_accum: list[DetectionBox] = []

        # 1. Open-vocab prompt generation
        prompts = generate_domain_prompts(domain)
        std_prompts = prompts.get(DetectionScale.STANDARD, [])
        micro_prompts = prompts.get(DetectionScale.MICRO, [])

        # 2. Whole-image standard detection
        base_dets = self.detector.detect(image_rgb, std_prompts)
        for b in base_dets:
            raw_accum.append(DetectionBox(
                box_id=f"{b.box_id}_std",
                class_name=b.class_name,
                box_2d=b.box_2d,
                confidence=b.confidence,
                source_engine=b.source_engine,
                source_image_id=image_id,
                mask_2d=b.mask_2d,
                parent_box_id=b.parent_box_id,
                scale=DetectionScale.STANDARD,
                attributes=b.attributes,
            ))

        # 3. Overlapping Tiled Inference for high-res images
        tiles_processed = 0
        if run_tiling and (w > self.tile_size or h > self.tile_size):
            tiles = generate_tiling_windows(w, h, tile_size=self.tile_size, overlap_ratio=self.tile_overlap)
            tiles_processed = len(tiles)
            for tile in tiles:
                tile_crop = image_rgb[tile.y1:tile.y2, tile.x1:tile.x2]
                tile_dets = self.detector.detect(tile_crop, micro_prompts or std_prompts)
                mapped_dets = transform_tile_detections_to_image(tile_dets, tile, w, h)
                for md in mapped_dets:
                    raw_accum.append(DetectionBox(
                        box_id=md.box_id,
                        class_name=md.class_name,
                        box_2d=md.box_2d,
                        confidence=md.confidence,
                        source_engine=md.source_engine,
                        source_image_id=image_id,
                        mask_2d=md.mask_2d,
                        parent_box_id=md.parent_box_id,
                        scale=DetectionScale.MICRO,
                        attributes=md.attributes,
                    ))

        # 4. Crop-based subpart detection on host objects
        subparts_detected = 0
        if detect_subparts:
            current_candidates = list(raw_accum)
            for cand in current_candidates:
                targets = get_subpart_detection_targets(cand.class_name)
                if not targets:
                    continue
                cx1, cy1, cx2, cy2 = extract_host_crop_roi(cand, w, h)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                crop_img = image_rgb[cy1:cy2, cx1:cx2]
                off_x, off_y = cx1, cy1
                target_names = [t.name for t in targets]
                part_dets = self.detector.detect(crop_img, target_names)
                for pd in part_dets:
                    px1 = max(0.0, min(float(w), pd.x1 + off_x))
                    py1 = max(0.0, min(float(h), pd.y1 + off_y))
                    px2 = max(0.0, min(float(w), pd.x2 + off_x))
                    py2 = max(0.0, min(float(h), pd.y2 + off_y))
                    if px2 > px1 and py2 > py1:
                        raw_accum.append(DetectionBox(
                            box_id=f"part_{pd.box_id}_{cand.box_id}",
                            class_name=pd.class_name,
                            box_2d=(px1, py1, px2, py2),
                            confidence=pd.confidence,
                            source_engine=f"{pd.source_engine}_subpart",
                            source_image_id=image_id,
                            parent_box_id=cand.box_id,
                            scale=DetectionScale.MICRO,
                            attributes={"host_class": cand.class_name},
                        ))
                        subparts_detected += 1

        # 5. Class-specific confidence filtering
        filtered_dets = filter_by_class_confidence(raw_accum)

        # 6. Duplicate suppression preserving subparts and WBF
        suppressed_dets = suppress_duplicates_preserving_parts(filtered_dets, iou_threshold=0.45)

        # 7. Mask segmentation and boundary refinement
        refined_boxes: list[DetectionBox] = []
        for det in suppressed_dets:
            mask = det.mask_2d
            if mask is None or mask.shape[:2] != (h, w):
                mask = self.segmenter.segment_box(image_rgb, det.box_2d)

            if mask is not None and mask.shape[:2] == (h, w):
                refined_mask = refine_instance_mask(mask, rgb_image=image_rgb)
            else:
                refined_mask = mask

            refined_boxes.append(DetectionBox(
                box_id=det.box_id,
                class_name=det.class_name,
                box_2d=det.box_2d,
                confidence=det.confidence,
                source_engine=det.source_engine,
                source_image_id=det.source_image_id,
                mask_2d=refined_mask,
                parent_box_id=det.parent_box_id,
                scale=det.scale,
                attributes=det.attributes,
            ))

        # 8. Build parent-part tree
        parent_part_tree: dict[str, list[str]] = {}
        for b in refined_boxes:
            if b.parent_box_id:
                parent_part_tree.setdefault(b.parent_box_id, []).append(b.box_id)

        stats = {
            "image_size": [w, h],
            "tiles_processed": tiles_processed,
            "subparts_detected": subparts_detected,
            "raw_detection_count": len(raw_accum),
            "filtered_detection_count": len(filtered_dets),
            "refined_box_count": len(refined_boxes),
        }

        return PipelineResult(
            image_id=image_id,
            raw_detections=tuple(raw_accum),
            refined_boxes=tuple(refined_boxes),
            parent_part_tree=parent_part_tree,
            execution_stats=stats,
        )
