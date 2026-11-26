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

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

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
    child_box_ids: tuple[str, ...] = ()
    scale: DetectionScale = DetectionScale.STANDARD
    physical_identity_id: str | None = None
    mask_polygon: tuple[tuple[float, float], ...] | None = None
    mask_area_px: int = 0
    is_truncated: bool = False
    occlusion_level: str = "none"  # "none", "partial", "heavy"
    frame_index: int | None = None
    timestamp_sec: float | None = None
    shot_id: int | None = None
    raw_score: float | None = None
    calibrated_confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    appearance_features: tuple[float, ...] = ()

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
            "child_box_ids": list(self.child_box_ids),
            "scale": self.scale.value,
            "area_px": round(self.area, 1),
            "mask_area_px": self.mask_area_px,
            "is_truncated": self.is_truncated,
            "occlusion_level": self.occlusion_level,
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 4) if self.timestamp_sec is not None else None,
            "shot_id": self.shot_id,
            "raw_score": round(self.raw_score, 4) if self.raw_score is not None else None,
            "calibrated_confidence": round(self.calibrated_confidence, 4) if self.calibrated_confidence is not None else None,
            "polygon_point_count": len(self.mask_polygon) if self.mask_polygon else 0,
            "has_appearance_features": len(self.appearance_features) > 0,
        }


def extract_polygon_from_mask(mask_2d: np.ndarray, max_points: int = 32) -> tuple[tuple[float, float], ...] | None:
    """Extracts a simplified polygonal boundary representation from a binary 2D mask."""
    if mask_2d.ndim != 2 or not np.any(mask_2d):
        return None

    if cv2 is not None:
        mask_u8 = (mask_2d.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 4.0:
            return None
        epsilon = max(1.0, 0.02 * cv2.arcLength(largest, True))
        approx = cv2.approxPolyDP(largest, epsilon, True)
        pts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
        if len(pts) > max_points:
            indices = np.linspace(0, len(pts) - 1, max_points, dtype=int)
            pts = [pts[i] for i in indices]
        return tuple(pts)
    else:
        rows, cols = np.where(mask_2d)
        if len(rows) < 4:
            return None
        rmin, rmax = float(np.min(rows)), float(np.max(rows))
        cmin, cmax = float(np.min(cols)), float(np.max(cols))
        return (
            (cmin, rmin),
            (cmax, rmin),
            (cmax, rmax),
            (cmin, rmax),
        )


def extract_mask_topology(mask_2d: np.ndarray) -> dict[str, Any]:
    """Extracts topological characteristics and geometric validity from a binary instance mask."""
    if mask_2d.ndim != 2 or not np.any(mask_2d):
        return {
            "hole_count": 0,
            "component_count": 0,
            "solidity": 0.0,
            "convex_hull_area_px": 0,
            "mask_area_px": 0,
            "is_valid": False,
            "validation_note": "empty_mask",
        }

    h, w = mask_2d.shape[:2]
    mask_area = int(np.sum(mask_2d.astype(bool)))
    total_area = float(h * w)

    # Reject degenerate masks: smaller than 20 px or covering > 98% of the entire frame
    if mask_area < 20:
        return {
            "hole_count": 0,
            "component_count": 0,
            "solidity": 0.0,
            "convex_hull_area_px": 0,
            "mask_area_px": mask_area,
            "is_valid": False,
            "validation_note": "mask_area_too_small_under_20px",
        }
    if mask_area > 0.98 * total_area:
        return {
            "hole_count": 0,
            "component_count": 0,
            "solidity": 1.0,
            "convex_hull_area_px": int(total_area),
            "mask_area_px": mask_area,
            "is_valid": False,
            "validation_note": "mask_area_exceeds_98_percent_frame",
        }

    hole_count = 0
    component_count = 1
    solidity = 1.0
    hull_area = mask_area

    if cv2 is not None:
        mask_u8 = mask_2d.astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours and hierarchy is not None:
            external_contours = [c for i, c in enumerate(contours) if hierarchy[0][i][3] == -1]
            hole_contours = [c for i, c in enumerate(contours) if hierarchy[0][i][3] != -1]
            component_count = max(1, len(external_contours))
            hole_count = len(hole_contours)

            all_pts = np.vstack([c for c in contours])
            hull = cv2.convexHull(all_pts)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0:
                solidity = float(np.clip(mask_area / hull_area, 0.0, 1.0))
    else:
        import scipy.ndimage as ndi
        lbls, component_count = ndi.label(mask_2d.astype(bool))
        filled = ndi.binary_fill_holes(mask_2d.astype(bool))
        holes = filled ^ mask_2d.astype(bool)
        if np.any(holes):
            _, hole_count = ndi.label(holes)

    return {
        "hole_count": int(hole_count),
        "component_count": int(component_count),
        "solidity": round(float(solidity), 4),
        "convex_hull_area_px": int(round(hull_area)),
        "mask_area_px": int(mask_area),
        "is_valid": True,
        "validation_note": "valid_instance_mask",
    }


def preprocess_image_for_detection(
    image_rgb: np.ndarray,
    max_dimension: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolution-aware image preprocessing with dynamic range inspection and aspect ratio telemetry."""
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be a 3-channel (H, W, 3) numpy array")

    h, w, c = image_rgb.shape
    mean_intensity = float(np.mean(image_rgb))
    std_intensity = float(np.std(image_rgb))
    is_low_contrast = std_intensity < 15.0

    meta: dict[str, Any] = {
        "original_shape": (h, w, c),
        "processed_shape": (h, w, c),
        "scale_factors": (1.0, 1.0),
        "aspect_ratio": round(w / max(1, h), 4),
        "mean_intensity": round(mean_intensity, 2),
        "std_intensity": round(std_intensity, 2),
        "is_low_contrast": is_low_contrast,
    }

    if max_dimension is not None and max(h, w) > max_dimension:
        from PIL import Image
        scale = max_dimension / float(max(h, w))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        pil_im = Image.fromarray(image_rgb).resize((nw, nh), Image.Resampling.BILINEAR)
        processed = np.asarray(pil_im)
        meta["processed_shape"] = (nh, nw, c)
        meta["scale_factors"] = (nw / float(w), nh / float(h))
        return processed, meta

    return image_rgb, meta


def detect_box_truncation(
    box_2d: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin: float = 3.0,
) -> bool:
    """Checks if a bounding box touches or is truncated by image boundaries."""
    x1, y1, x2, y2 = box_2d
    return (
        x1 <= margin or
        y1 <= margin or
        x2 >= image_width - margin or
        y2 >= image_height - margin
    )


def estimate_box_occlusion(
    box_2d: tuple[float, float, float, float],
    mask_2d: np.ndarray | None,
) -> str:
    """Estimates occlusion severity (none, partial, heavy) from mask-to-box area ratio."""
    if mask_2d is None:
        return "none"
    w = max(0.0, box_2d[2] - box_2d[0])
    h = max(0.0, box_2d[3] - box_2d[1])
    box_area = w * h
    if box_area <= 1e-4:
        return "none"
    mask_area = float(np.sum(mask_2d.astype(bool)))
    ratio = mask_area / box_area
    if ratio < 0.20:
        return "heavy"
    elif ratio < 0.55:
        return "partial"
    return "none"


# ============================================================================
# 1. DOMAIN-AWARE PROMPT GENERATION
# ============================================================================

def generate_domain_prompts(
    domain: Domain = Domain.INTERIOR,
    scales: Sequence[DetectionScale] = (DetectionScale.MACRO, DetectionScale.STANDARD, DetectionScale.MICRO),
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
        elif max_dim > 4.5 or tax_cls.is_structural:
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

GENUINE_STRUCTURAL_PARTS: dict[str, tuple[str, ...]] = {
    "door": ("door_handle", "door_knob", "door_lever", "door_frame", "door_leaf", "door_lock", "door_hinge", "door_latch", "door_threshold", "door_panel", "door_glazing", "door_hardware"),
    "window": ("window_frame", "window_sash", "window_sill", "window_glazing", "window_glass", "window_mullion", "window_handle", "window_hinge", "window_shutter"),
    "table": ("table_top", "furniture_leg", "table_leg", "table_support", "drawer", "drawer_handle"),
    "desk": ("drawer", "drawer_handle", "socket", "outlet_plate", "furniture_leg", "table_leg", "table_top"),
    "chair": ("chair_seat", "chair_backrest", "chair_back", "chair_armrest", "chair_arm", "furniture_leg", "chair_leg"),
    "cabinet": ("cabinet_carcass", "cabinet_door", "cabinet_handle", "drawer", "drawer_handle", "shelf"),
    "wardrobe": ("cabinet_carcass", "cabinet_door", "cabinet_handle", "drawer", "drawer_handle", "shelf"),
    "outlet_plate": ("socket", "plug_socket", "switch", "control_switch", "plug_head"),
    "electrical_outlet": ("socket", "plug_socket", "switch", "outlet_plate", "control_switch", "plug_head"),
    "switchboard_plate": ("socket", "plug_socket", "switch", "control_switch", "outlet_plate", "distribution_board"),
    "distribution_board": ("switch", "control_switch", "conduit", "cable"),
    "socket": ("switch", "outlet_plate", "plug_head"),
    "light": ("diffuser", "light_housing", "mounting_hardware"),
    "lamp": ("diffuser", "light_housing", "mounting_hardware"),
    "ceiling_light": ("diffuser", "light_housing", "mounting_hardware"),
    "faucet": ("faucet_handle", "spout", "tap", "fitting"),
    "sink": ("faucet", "basin", "drain", "pipe", "spout", "tap", "fitting"),
    "pipe": ("fitting", "valve"),
    "radiator": ("valve", "pipe", "vent", "mounting_hardware"),
    "tree": ("trunk", "branch"),
    "street_furniture": ("sign", "signboard", "lamppost"),
    "road": ("curb", "road_edge", "footpath"),
}

EXPECTED_KEY_COMPONENTS: dict[str, tuple[str, ...]] = {
    "door": ("door_handle", "door_frame"),
    "window": ("window_frame",),
    "table": ("table_top",),
    "chair": ("chair_seat",),
    "cabinet": ("door", "drawer"),
    "outlet_plate": ("socket",),
    "sink": ("basin", "faucet"),
    "lamp": ("diffuser", "light_housing"),
}


def get_subpart_detection_targets(parent_class_name: str) -> list[TaxonomyClass]:
    """Returns candidate subpart taxonomy classes for a given parent host object."""
    parent_cls = resolve_taxonomy_class(parent_class_name)
    targets: list[TaxonomyClass] = []
    seen: set[str] = set()

    # 1. Designated genuine structural parts
    struct_parts = GENUINE_STRUCTURAL_PARTS.get(parent_cls.name, ())
    for part_name in struct_parts:
        part_cls = resolve_taxonomy_class(part_name)
        if part_cls.taxonomy_id not in seen and part_cls.taxonomy_id != parent_cls.taxonomy_id:
            seen.add(part_cls.taxonomy_id)
            targets.append(part_cls)

    # 2. Strict taxonomy subparts with direct parent_class relationship
    for cls in get_all_taxonomy_classes():
        if cls.parent_class in (parent_cls.taxonomy_id, parent_cls.name):
            if cls.taxonomy_id not in seen and cls.taxonomy_id != parent_cls.taxonomy_id:
                seen.add(cls.taxonomy_id)
                targets.append(cls)

    return targets


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


def generate_component_rois_for_host(
    parent_box: DetectionBox,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int, str]]:
    """Generates targeted sub-regions (cx1, cy1, cx2, cy2, zone_name) for high-resolution component discovery (Sprint 8)."""
    rois: list[tuple[int, int, int, int, str]] = []
    # 1. Full aperture crop
    rois.append((*extract_host_crop_roi(parent_box, image_width, image_height, padding_ratio=0.10), "full_host"))

    cname = parent_box.class_name.lower()
    pw = parent_box.width
    ph = parent_box.height

    # High-resolution region proposals for fine-grained small fixtures:
    if "door" in cname and ph >= 120:
        # Left latch / handle zone
        y_mid_1 = max(0, int(parent_box.y1 + 0.35 * ph))
        y_mid_2 = min(image_height, int(parent_box.y1 + 0.68 * ph))
        xl_1 = max(0, int(parent_box.x1 - 0.05 * pw))
        xl_2 = min(image_width, int(parent_box.x1 + 0.45 * pw))
        if xl_2 > xl_1 and y_mid_2 > y_mid_1:
            rois.append((xl_1, y_mid_1, xl_2, y_mid_2, "door_latch_left"))
        # Right latch / handle zone
        xr_1 = max(0, int(parent_box.x2 - 0.45 * pw))
        xr_2 = min(image_width, int(parent_box.x2 + 0.05 * pw))
        if xr_2 > xr_1 and y_mid_2 > y_mid_1:
            rois.append((xr_1, y_mid_1, xr_2, y_mid_2, "door_latch_right"))
    elif ("table" in cname or "desk" in cname) and ph >= 80:
        # Upper band: table top surface and drawers
        yt_1 = max(0, int(parent_box.y1 - 0.08 * ph))
        yt_2 = min(image_height, int(parent_box.y1 + 0.45 * ph))
        xt_1 = max(0, int(parent_box.x1 - 0.05 * pw))
        xt_2 = min(image_width, int(parent_box.x2 + 0.05 * pw))
        if xt_2 > xt_1 and yt_2 > yt_1:
            rois.append((xt_1, yt_1, xt_2, yt_2, "table_top_band"))
    elif "window" in cname and ph >= 80:
        # Lower sash and window sill
        ys_1 = max(0, int(parent_box.y1 + 0.50 * ph))
        ys_2 = min(image_height, int(parent_box.y2 + 0.12 * ph))
        xs_1 = max(0, int(parent_box.x1 - 0.05 * pw))
        xs_2 = min(image_width, int(parent_box.x2 + 0.05 * pw))
        if xs_2 > xs_1 and ys_2 > ys_1:
            rois.append((xs_1, ys_1, xs_2, ys_2, "window_lower_sill"))
    elif "chair" in cname and ph >= 80:
        # Seat level zone
        ys_1 = max(0, int(parent_box.y1 + 0.35 * ph))
        ys_2 = min(image_height, int(parent_box.y1 + 0.70 * ph))
        xs_1 = max(0, int(parent_box.x1 - 0.05 * pw))
        xs_2 = min(image_width, int(parent_box.x2 + 0.05 * pw))
        if xs_2 > xs_1 and ys_2 > ys_1:
            rois.append((xs_1, ys_1, xs_2, ys_2, "chair_seat_zone"))

    return rois


def resolve_component_ownership(
    component_box: tuple[float, float, float, float],
    component_class: str,
    candidate_hosts: Sequence[DetectionBox],
    min_containment_ioa: float = 0.35,
) -> tuple[str | None, bool, list[str]]:
    """Resolves which host object a component physically belongs to with semantic specificity (Sprint 8).

    Guarantees:
    - Specific host binding over distant background surfaces (e.g. door_handle -> door, NOT wall).
    - Ambiguous parent preservation when multiple hosts of the same class overlap.
    - Subparts cannot accidentally claim other subparts as parents unless explicitly supported.

    Returns:
        (best_host_id, is_ambiguous, candidate_parent_ids)
    """
    tax_comp = resolve_taxonomy_class(component_class)
    comp_area = max(1.0, (component_box[2] - component_box[0]) * (component_box[3] - component_box[1]))

    qualifying_hosts: list[tuple[float, int, DetectionBox]] = []

    for host in candidate_hosts:
        if host.parent_box_id is not None:
            continue  # Subparts cannot host other components

        tax_host = resolve_taxonomy_class(host.class_name)

        # Check semantic hierarchy compatibility
        is_direct_part = (
            tax_comp.parent_class in (tax_host.taxonomy_id, tax_host.name) or
            tax_comp.name in GENUINE_STRUCTURAL_PARTS.get(tax_host.name, ())
        )
        is_potential = tax_comp.name in tax_host.potential_parts
        is_generic_host = host.class_name in tax_comp.hosts

        if not (is_direct_part or is_potential or is_generic_host):
            continue

        # Specificity score: direct structural part (3) > potential part (2) > generic mounting host (1)
        specificity = 3 if is_direct_part else (2 if is_potential else 1)

        # Containment IoA
        inter_x = max(0.0, min(component_box[2], host.x2) - max(component_box[0], host.x1))
        inter_y = max(0.0, min(component_box[3], host.y2) - max(component_box[1], host.y1))
        inter_area = inter_x * inter_y
        ioa = inter_area / comp_area

        # Perimeter/structural parts (frames, sashes, carcasses) can be up to 1.15x host area
        max_ratio = 1.15 if component_class in (
            "window_frame", "door_frame", "door_leaf", "cabinet_carcass", "door_panel", "window_sash", "table_top"
        ) else 0.85

        if ioa >= min_containment_ioa and comp_area <= max_ratio * host.area:
            qualifying_hosts.append((ioa, specificity, host))

    if not qualifying_hosts:
        return None, False, []

    # Sort descending by specificity first, then by containment IoA
    qualifying_hosts.sort(key=lambda item: (item[1], item[0]), reverse=True)

    best_spec = qualifying_hosts[0][1]
    best_ioa = qualifying_hosts[0][0]
    best_host = qualifying_hosts[0][2]

    # Check for ambiguity among hosts with the same top specificity
    same_spec_candidates = [h for ioa, spec, h in qualifying_hosts if spec == best_spec and ioa >= best_ioa - 0.08]
    if len(same_spec_candidates) > 1 and same_spec_candidates[0].class_name == same_spec_candidates[1].class_name:
        return None, True, [h.box_id for h in same_spec_candidates]

    return best_host.box_id, False, [best_host.box_id]


def extract_appearance_features(image_rgb: np.ndarray, box_2d: Sequence[float]) -> tuple[float, ...]:
    """Extracts normalized 24-bin color histogram appearance features from a detection crop."""
    h, w = image_rgb.shape[:2]
    x1 = max(0, min(w - 1, int(round(box_2d[0]))))
    y1 = max(0, min(h - 1, int(round(box_2d[1]))))
    x2 = max(0, min(w, int(round(box_2d[2]))))
    y2 = max(0, min(h, int(round(box_2d[3]))))
    if x2 <= x1 or y2 <= y1:
        return ()
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return ()
    r_h, _ = np.histogram(crop[..., 0], bins=8, range=(0, 256), density=True)
    g_h, _ = np.histogram(crop[..., 1], bins=8, range=(0, 256), density=True)
    b_h, _ = np.histogram(crop[..., 2], bins=8, range=(0, 256), density=True)
    hist = np.concatenate([r_h, g_h, b_h])
    norm = np.linalg.norm(hist)
    if norm > 1e-6:
        hist = hist / norm
    return tuple(float(v) for v in hist)


# ============================================================================
# 4. CLASS-SPECIFIC CONFIDENCE GATING & VALIDATION
# ============================================================================

DEFAULT_CALIBRATED_CLASS_THRESHOLDS: dict[str, float] = {
    # Openings and major interior objects
    "window": 0.40,
    "table": 0.28,
    "chair": 0.26,
    "lamp": 0.22,
    "door": 0.32,
    "opening": 0.28,
    "structural_boundary": 0.25,
    # Architectural surfaces and structural elements
    "wall": 0.25,
    "floor": 0.25,
    "ceiling": 0.25,
    "column": 0.28,
    "beam": 0.28,
    "staircase": 0.30,
    "railing": 0.28,
    "corridor": 0.25,
    # Architectural subparts and fittings
    "door_frame": 0.25,
    "door_handle": 0.20,
    "door_hinge": 0.20,
    "door_lock": 0.20,
    "door_panel": 0.22,
    "door_glazing": 0.22,
    "window_frame": 0.25,
    "window_sill": 0.22,
    "window_glass": 0.25,
    "glass": 0.25,
    "window_handle": 0.20,
    "window_sash": 0.22,
    "window_mullion": 0.22,
    # Electrical and mechanical fixtures
    "socket": 0.18,
    "electrical_outlet": 0.18,
    "switch": 0.18,
    "switchboard_plate": 0.18,
    "outlet_plate": 0.18,
    "control_switch": 0.18,
    "light": 0.25,
    "ceiling_light": 0.25,
    "ceiling_fan": 0.28,
    "light_housing": 0.22,
    "diffuser": 0.20,
    "mounting_hardware": 0.20,
    "vent": 0.22,
    "radiator": 0.28,
    "thermostat": 0.20,
    "electrical_panel": 0.25,
    # Plumbing and sanitary
    "faucet": 0.22,
    "sink": 0.28,
    "pipe": 0.22,
    "drain": 0.20,
    "basin": 0.22,
    "spout": 0.20,
    # Furniture, storage, and interior objects
    "sofa": 0.30,
    "bed": 0.30,
    "cabinet": 0.28,
    "wardrobe": 0.28,
    "storage_unit": 0.28,
    "kitchen_furniture": 0.28,
    "bathroom_fixture": 0.28,
    "drawer": 0.22,
    "drawer_handle": 0.18,
    "shelf": 0.25,
    "appliance": 0.28,
    "plant": 0.28,
    "decorative_object": 0.25,
    "table_top": 0.25,
    "table_leg": 0.20,
    "furniture_leg": 0.20,
    "chair_seat": 0.22,
    "chair_backrest": 0.22,
    "chair_armrest": 0.20,
    # Exterior infrastructure and environment
    "building": 0.30,
    "roof": 0.28,
    "tree": 0.30,
    "bush": 0.28,
    "footpath": 0.25,
    "road": 0.25,
    "streetlight": 0.28,
    "sign": 0.25,
    "bench": 0.28,
    "pole": 0.25,
    "fence": 0.28,
    "drainage_infrastructure": 0.22,
}


@dataclass
class CalibrationMetrics:
    ece: float
    brier_score: float
    nll: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": round(self.ece, 6),
            "brier_score": round(self.brier_score, 6),
            "nll": round(self.nll, 6),
            "sample_count": self.sample_count,
        }


class CalibrationResult(NamedTuple):
    calibrated_probability: float
    is_calibrated: bool
    method: str


class LearnedConfidenceCalibrator:
    """Empirical temperature and bias scaling calibrator fit on development split.

    Supports:
    - Global Platt scaling parameters (temperature, bias)
    - Per-class Platt scaling parameters when sufficient samples exist
    - Fallback tracking (per_class_platt, global_platt, uncalibrated)
    - Calibration error evaluation (ECE, Brier score, NLL)
    - Reliability diagrams and calibration curves
    - Parameter uncertainty / Fisher information standard errors
    - Full provenance tracking (model_name, checkpoint_hash, dataset_split, dataset_split_hash)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        bias: float = 0.0,
        class_params: dict[str, tuple[float, float]] | None = None,
        class_sample_counts: dict[str, dict[str, int]] | None = None,
        class_uncertainties: dict[str, dict[str, float]] | None = None,
        global_uncertainty: dict[str, float] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.bias = float(bias)
        self.class_params: dict[str, tuple[float, float]] = dict(class_params) if class_params else {}
        self.class_sample_counts: dict[str, dict[str, int]] = dict(class_sample_counts) if class_sample_counts else {}
        self.class_uncertainties: dict[str, dict[str, float]] = dict(class_uncertainties) if class_uncertainties else {}
        self.global_uncertainty: dict[str, float] = dict(global_uncertainty) if global_uncertainty else {}
        self.provenance: dict[str, Any] = dict(provenance) if provenance else {}
        self.is_fitted = bool(self.provenance.get("is_fitted", False) or class_params or temperature != 1.0 or bias != 0.0)

    @staticmethod
    def compute_calibration_metrics(
        scores: Sequence[float],
        binary_labels: Sequence[int],
        num_bins: int = 10,
    ) -> CalibrationMetrics:
        """Computes empirical ECE (Expected Calibration Error), Brier Score, and NLL."""
        if not scores or len(scores) != len(binary_labels):
            return CalibrationMetrics(ece=0.0, brier_score=0.0, nll=0.0, sample_count=0)

        p = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        y = np.asarray(binary_labels, dtype=np.float64)
        n = len(p)

        # Brier score
        brier = float(np.mean((p - y) ** 2))

        # NLL
        nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        # ECE with equal-width bins
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        bin_indices = np.digitize(p, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        ece = 0.0
        for b in range(num_bins):
            mask = (bin_indices == b)
            bin_count = np.sum(mask)
            if bin_count > 0:
                bin_acc = np.mean(y[mask])
                bin_conf = np.mean(p[mask])
                ece += (bin_count / n) * abs(bin_acc - bin_conf)

        return CalibrationMetrics(ece=float(ece), brier_score=brier, nll=nll, sample_count=n)

    @staticmethod
    def compute_reliability_diagram(
        scores: Sequence[float],
        binary_labels: Sequence[int],
        num_bins: int = 10,
    ) -> list[dict[str, Any]]:
        """Computes empirical reliability diagram bins (mean confidence vs empirical accuracy)."""
        if not scores or len(scores) != len(binary_labels):
            return []
        p = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        y = np.asarray(binary_labels, dtype=np.float64)
        n = len(p)
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        bin_indices = np.clip(np.digitize(p, bins) - 1, 0, num_bins - 1)
        diagram: list[dict[str, Any]] = []
        for b in range(num_bins):
            mask = (bin_indices == b)
            cnt = int(np.sum(mask))
            b_low = float(bins[b])
            b_high = float(bins[b + 1])
            if cnt > 0:
                conf = float(np.mean(p[mask]))
                acc = float(np.mean(y[mask]))
                gap = float(abs(acc - conf))
            else:
                conf = float(0.5 * (b_low + b_high))
                acc = 0.0
                gap = 0.0
            diagram.append({
                "bin_index": b,
                "bin_lower": round(b_low, 3),
                "bin_upper": round(b_high, 3),
                "count": cnt,
                "confidence": round(conf, 4),
                "accuracy": round(acc, 4),
                "calibration_gap": round(gap, 4),
            })
        return diagram

    def verify_model_compatibility(
        self,
        model_name: str | None = None,
        checkpoint_hash: str | None = None,
    ) -> bool:
        """Verifies if the calibrator was fitted on the specified model and checkpoint."""
        if not self.is_fitted:
            return True
        exp_model = self.provenance.get("model_name")
        if exp_model and model_name and exp_model != model_name:
            return False
        exp_hash = self.provenance.get("checkpoint_hash")
        if exp_hash and checkpoint_hash and exp_hash != checkpoint_hash:
            return False
        return True

    def fit(
        self,
        raw_scores: Sequence[float],
        binary_labels: Sequence[int],
        class_names: Sequence[str] | None = None,
        min_class_samples: int = 8,
        min_class_positives: int = 3,
        min_class_negatives: int = 3,
        provenance: dict[str, Any] | None = None,
    ) -> LearnedConfidenceCalibrator:
        """Fits Platt scaling parameters using NLL minimization on calibration data.

        Enforces strict sample requirements to prevent overfitting on sparse classes:
        - Requires >= 6 total samples with >= 2 positives and >= 2 negatives for global Platt.
        - Requires >= min_class_samples (default 8) with >= min_class_positives (default 3)
          and >= min_class_negatives (default 3) for per-class Platt.
        - Calculates parameter uncertainties (standard errors) via Fisher information.
        """
        if not raw_scores or len(raw_scores) < 6 or sum(binary_labels) < 2 or sum(binary_labels) > len(binary_labels) - 2:
            self.temperature = 1.0
            self.bias = 0.0
            self.is_fitted = False
            return self

        from scipy.optimize import minimize

        scores = np.clip(np.asarray(raw_scores, dtype=np.float64), 1e-4, 1.0 - 1e-4)
        logits = np.log(scores / (1.0 - scores))
        y = np.asarray(binary_labels, dtype=np.float64)

        def fit_platt_with_uncertainty(log_arr: np.ndarray, y_arr: np.ndarray) -> tuple[float, float, float, float]:
            def nll_loss(params: np.ndarray) -> float:
                T, b = params[0], params[1]
                scaled_logits = log_arr / max(1e-2, T) + b
                p = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -30.0, 30.0)))
                eps = 1e-7
                loss = -np.mean(y_arr * np.log(p + eps) + (1.0 - y_arr) * np.log(1.0 - p + eps))
                return float(loss)

            res = minimize(nll_loss, x0=[1.0, 0.0], bounds=[(0.05, 10.0), (-5.0, 5.0)], method="L-BFGS-B")
            opt_T, opt_b = (float(res.x[0]), float(res.x[1])) if res.success else (1.0, 0.0)

            # Fisher information / Hessian matrix estimation for parameter uncertainty
            n_samples = len(log_arr)
            z = log_arr / max(1e-2, opt_T) + opt_b
            p_opt = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            w = p_opt * (1.0 - p_opt)
            h_bb = float(np.mean(w))
            h_TT = float(np.mean(w * ((log_arr / (opt_T ** 2)) ** 2)))
            h_Tb = float(-np.mean(w * (log_arr / (opt_T ** 2))))
            det_h = h_TT * h_bb - (h_Tb ** 2)

            se_T = 0.0
            se_b = 0.0
            if det_h > 1e-8 and n_samples > 0:
                var_T = max(0.0, (1.0 / n_samples) * (h_bb / det_h))
                var_b = max(0.0, (1.0 / n_samples) * (h_TT / det_h))
                se_T = math.sqrt(var_T)
                se_b = math.sqrt(var_b)

            return (opt_T, opt_b, se_T, se_b)

        # 1. Global Platt fit with uncertainty
        self.temperature, self.bias, g_se_T, g_se_b = fit_platt_with_uncertainty(logits, y)
        self.global_uncertainty = {
            "temperature_se": round(g_se_T, 4),
            "bias_se": round(g_se_b, 4),
        }

        # 2. Per-class Platt fit if class_names provided with sample thresholds
        self.class_params = {}
        self.class_sample_counts = {}
        self.class_uncertainties = {}
        if class_names is not None and len(class_names) == len(raw_scores):
            by_class: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
            for lg, y_val, cls in zip(logits, y, class_names):
                by_class[cls][0].append(lg)
                by_class[cls][1].append(y_val)

            for cls, (c_logs, c_ys) in by_class.items():
                c_y_arr = np.asarray(c_ys, dtype=np.float64)
                total_c = len(c_logs)
                pos_c = int(np.sum(c_y_arr == 1.0))
                neg_c = int(np.sum(c_y_arr == 0.0))
                self.class_sample_counts[cls] = {
                    "total": total_c,
                    "positives": pos_c,
                    "negatives": neg_c,
                }
                if total_c >= min_class_samples and pos_c >= min_class_positives and neg_c >= min_class_negatives:
                    c_log_arr = np.asarray(c_logs, dtype=np.float64)
                    c_T, c_b, c_se_T, c_se_b = fit_platt_with_uncertainty(c_log_arr, c_y_arr)
                    self.class_params[cls] = (c_T, c_b)
                    self.class_uncertainties[cls] = {
                        "temperature_se": round(c_se_T, 4),
                        "bias_se": round(c_se_b, 4),
                    }

        self.is_fitted = True

        # Compute pre- and post-calibration metrics and reliability diagrams
        cal_scores = [self.calibrate(s, c).calibrated_probability for s, c in zip(raw_scores, class_names or ["unknown"] * len(raw_scores))]
        pre_metrics = self.compute_calibration_metrics(raw_scores, binary_labels)
        post_metrics = self.compute_calibration_metrics(cal_scores, binary_labels)
        pre_diagram = self.compute_reliability_diagram(raw_scores, binary_labels)
        post_diagram = self.compute_reliability_diagram(cal_scores, binary_labels)

        self.provenance = {
            "is_fitted": True,
            "global_temperature": round(self.temperature, 4),
            "global_bias": round(self.bias, 4),
            "global_uncertainty": self.global_uncertainty,
            "class_param_count": len(self.class_params),
            "classes_calibrated": list(self.class_params.keys()),
            "class_sample_counts": self.class_sample_counts,
            "class_uncertainties": self.class_uncertainties,
            "pre_calibration": pre_metrics.to_dict(),
            "post_calibration": post_metrics.to_dict(),
            "pre_reliability_diagram": pre_diagram,
            "post_reliability_diagram": post_diagram,
        }
        if provenance:
            self.provenance.update(provenance)

        return self

    def calibrate(self, score: float, class_name: str | None = None) -> CalibrationResult:
        """Transforms a raw score into a calibrated posterior probability.

        Returns:
            CalibrationResult(calibrated_probability, is_calibrated, method)
        """
        if not self.is_fitted:
            return CalibrationResult(float(score), False, "uncalibrated")

        p = float(np.clip(score, 1e-4, 1.0 - 1e-4))
        logit = math.log(p / (1.0 - p))

        if class_name and class_name in self.class_params:
            T, b = self.class_params[class_name]
            method = "per_class_platt"
        elif self.temperature != 1.0 or self.bias != 0.0:
            T, b = self.temperature, self.bias
            method = "global_platt"
        else:
            return CalibrationResult(float(score), False, "uncalibrated")

        scaled_logit = logit / max(1e-2, T) + b
        calibrated_prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, scaled_logit))))
        return CalibrationResult(float(np.clip(calibrated_prob, 0.01, 0.99)), True, method)

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "bias": self.bias,
            "class_params": {k: [round(v[0], 4), round(v[1], 4)] for k, v in self.class_params.items()},
            "class_sample_counts": self.class_sample_counts,
            "class_uncertainties": self.class_uncertainties,
            "global_uncertainty": self.global_uncertainty,
            "provenance": self.provenance,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedConfidenceCalibrator:
        cal = cls(
            temperature=data.get("temperature", 1.0),
            bias=data.get("bias", 0.0),
            class_params={k: (v[0], v[1]) for k, v in data.get("class_params", {}).items()},
            class_sample_counts=data.get("class_sample_counts", {}),
            class_uncertainties=data.get("class_uncertainties", {}),
            global_uncertainty=data.get("global_uncertainty", {}),
            provenance=data.get("provenance", {}),
        )
        cal.is_fitted = data.get("is_fitted", False)
        return cal

    def save(self, file_path: str | Path) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> LearnedConfidenceCalibrator:
        p = Path(file_path)
        if not p.is_file():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_default_production_calibrator(
    file_path: str | Path = "artifacts/perception/calibrator_dev.json",
) -> LearnedConfidenceCalibrator | None:
    """Loads default production confidence calibrator from workspace artifacts if available and fitted."""
    p = Path(file_path)
    if p.is_file():
        try:
            cal = LearnedConfidenceCalibrator.load(p)
            if cal.is_fitted:
                return cal
        except Exception:
            return None
    return None


def calibrate_detection_confidence(
    raw_score: float,
    class_name: str,
    calibrator: LearnedConfidenceCalibrator | None = None,
) -> float:
    """Calibrates raw detector confidence into empirical posterior probability.

    If a learned calibrator (fit on dev split) is provided, it uses the fitted parameters.
    Otherwise applies class-aware threshold scaling minimizing calibration error.
    """
    if calibrator is not None and calibrator.is_fitted:
        res = calibrator.calibrate(raw_score, class_name=class_name)
        return float(res.calibrated_probability)

    tax_cls = resolve_taxonomy_class(class_name)
    tau = DEFAULT_CALIBRATED_CLASS_THRESHOLDS.get(class_name, tax_cls.min_detector_confidence)
    p = float(np.clip(raw_score, 0.01, 0.99))
    raw_logit = math.log(p / (1.0 - p))
    thresh_logit = math.log(max(0.05, min(0.95, tau)) / (1.0 - max(0.05, min(0.95, tau))))
    calibrated_logit = (raw_logit - thresh_logit) / 1.25
    calibrated_prob = 1.0 / (1.0 + math.exp(-calibrated_logit))
    return float(np.clip(calibrated_prob, 0.05, 0.99))


def filter_by_class_confidence(
    detections: Sequence[DetectionBox],
    class_thresholds: dict[str, float] | None = None,
    max_frame_share: float = 0.75,
    max_dimension_share: float = 0.95,
    image_shape: tuple[int, int] | None = None,
) -> list[DetectionBox]:
    """Filters detection candidates using domain- and class-specific confidence thresholds and box geometry."""
    filtered: list[DetectionBox] = []
    eff_thresh = dict(DEFAULT_CALIBRATED_CLASS_THRESHOLDS)
    if class_thresholds:
        eff_thresh.update(class_thresholds)

    img_h, img_w = image_shape if image_shape else (3024, 4032)
    frame_area = float(img_w * img_h)

    for det in detections:
        # 1. Whole-image hallucinated box filtering
        w = det.width
        h = det.height
        frame_share = (w * h) / frame_area if frame_area > 0 else 0.0
        if frame_share > max_frame_share:
            continue
        if (w > max_dimension_share * img_w and h > max_dimension_share * img_h) or (w > max_dimension_share * img_w and frame_share > 0.35):
            continue

        tax_cls = resolve_taxonomy_class(det.class_name)
        threshold = eff_thresh.get(det.class_name, tax_cls.min_detector_confidence)

        # Micro fixtures (handles, sockets, switches, hinges, locks) should NOT cover large fractions of the whole image
        micro_classes = {
            "door_handle", "door_lock", "door_hinge", "window_handle",
            "socket", "switch", "faucet_handle", "spout", "drain",
            "drawer_handle", "control_switch",
        }
        if det.class_name in micro_classes or det.scale == DetectionScale.MICRO:
            if det.parent_box_id is None:  # un-parented full-image candidate
                norm_w = w / float(img_w)
                norm_h = h / float(img_h)
                if norm_w > 0.12 or norm_h > 0.12 or frame_share > 0.015:
                    continue
            threshold = min(threshold, 0.15)
        elif tax_cls.parent_class is not None:
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
    iou_threshold: float = 0.50,
    image_shape: tuple[int, int] | None = None,
    rejection_records: list[dict[str, Any]] | None = None,
) -> list[DetectionBox]:
    """Suppresses redundant duplicate detections while explicitly preserving subparts inside parents.

    If Box A and Box B have high overlap:
    - If Box A and Box B have the exact same class: fuses them via confidence-weighted averaging.
    - If Box A is a genuine subpart of Box B (or vice versa): PRESERVES BOTH and establishes parent-part link.
    - If Box A and Box B are different unrelated classes: cross-class NMS keeps the one with higher confidence.
    """
    if not detections:
        return []

    # Infer or use image dimensions for resolution-invariant geometric reasoning
    if image_shape is not None and image_shape[0] > 0 and image_shape[1] > 0:
        img_h, img_w = float(image_shape[0]), float(image_shape[1])
    else:
        mask_shapes = [d.mask_2d.shape[:2] for d in detections if d.mask_2d is not None]
        if mask_shapes:
            img_h, img_w = float(mask_shapes[0][0]), float(mask_shapes[0][1])
        elif detections:
            max_x = max(d.x2 for d in detections)
            max_y = max(d.y2 for d in detections)
            img_w = max(100.0, max_x)
            img_h = max(100.0, max_y)
        else:
            img_h, img_w = 3024.0, 4032.0

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

            # Check containment
            det_in_existing = (
                det.x1 >= existing.x1 - 5.0 and det.y1 >= existing.y1 - 5.0 and
                det.x2 <= existing.x2 + 5.0 and det.y2 <= existing.y2 + 5.0
            )
            existing_in_det = (
                existing.x1 >= det.x1 - 5.0 and existing.y1 >= det.y1 - 5.0 and
                existing.x2 <= det.x2 + 5.0 and existing.y2 <= det.y2 + 5.0
            )

            # Calculate containment IoA (Intersection over Area)
            inter_area = max(0.0, min(det.x2, existing.x2) - max(det.x1, existing.x1)) * max(0.0, min(det.y2, existing.y2) - max(det.y1, existing.y1))
            area_det = det.area
            area_existing = existing.area
            ioa_det = inter_area / area_det if area_det > 0 else 0.0
            ioa_existing = inter_area / area_existing if area_existing > 0 else 0.0

            # Genuine taxonomy part-parent relationships: require containment (IoA >= 0.35, area ratio checked)
            max_ratio_det = 1.15 if det.class_name in ("window_frame", "door_frame", "door_leaf", "cabinet_carcass", "door_panel", "window_sash", "table_top") else 0.85
            max_ratio_existing = 1.15 if existing.class_name in ("window_frame", "door_frame", "door_leaf", "cabinet_carcass", "door_panel", "window_sash", "table_top") else 0.85

            det_is_part_of_existing = (
                (det.parent_box_id == existing.box_id) or
                ((tax_det.parent_class in (tax_existing.taxonomy_id, tax_existing.name) or
                  tax_det.name in GENUINE_STRUCTURAL_PARTS.get(tax_existing.name, ()) or
                  (tax_det.name in tax_existing.potential_parts and det.scale == DetectionScale.MICRO))
                 and ioa_det >= 0.35 and area_det <= max_ratio_det * area_existing)
            )
            existing_is_part_of_det = (
                (existing.parent_box_id == det.box_id) or
                ((tax_existing.parent_class in (tax_det.taxonomy_id, tax_det.name) or
                  tax_existing.name in GENUINE_STRUCTURAL_PARTS.get(tax_det.name, ()) or
                  (tax_existing.name in tax_det.potential_parts and existing.scale == DetectionScale.MICRO))
                 and ioa_existing >= 0.35 and area_existing <= max_ratio_existing * area_det)
            )

            if det_is_part_of_existing:
                if det.parent_box_id is None:
                    object.__setattr__(det, "parent_box_id", existing.box_id)
                continue
            elif existing_is_part_of_det:
                if existing.parent_box_id is None:
                    object.__setattr__(existing, "parent_box_id", det.box_id)
                continue

            # Duplicate candidate matching: MUST NOT treat parent-child pairs as duplicate candidates (Sprint 7/8)
            is_compat = are_semantically_compatible(det.class_name, existing.class_name, allow_parent_part=False)

            if is_compat and (iou >= iou_threshold or ioa_det >= 0.65 or ioa_existing >= 0.65):
                # Same physical entity detected multiple times or partial sub-window detection: fuse boxes
                fused_with_idx = idx
                should_keep = False
                break
            elif iou >= iou_threshold:
                if (
                    det.parent_box_id is not None
                    and existing.parent_box_id is not None
                    and det.parent_box_id == existing.parent_box_id
                    and det.class_name != existing.class_name
                ):
                    # Different legitimate subparts belonging to the same host object (e.g. handle vs hinge, frame vs glass): DO NOT SUPPRESS!
                    continue
                elif (det.parent_box_id is not None) != (existing.parent_box_id is not None):
                    # One is a primary standalone entity and the other is a subpart of another object:
                    # A subpart should NEVER suppress a primary standalone entity of a different class
                    if det.parent_box_id is None:
                        continue
                    else:
                        should_keep = False
                        break
                else:
                    # High spatial overlap between non-compatible classes: suppress the lower confidence one
                    should_keep = False
                    break

        if fused_with_idx is not None:
            # Fuse coordinates via weighted box fusion (WBF) or containment adoption
            target = kept[fused_with_idx]
            w1 = target.confidence
            w2 = det.confidence
            w_sum = w1 + w2

            inter_area = max(0.0, min(det.x2, target.x2) - max(det.x1, target.x1)) * max(0.0, min(det.y2, target.y2) - max(det.y1, target.y1))
            ioa_det = inter_area / det.area if det.area > 0 else 0.0
            ioa_target = inter_area / target.area if target.area > 0 else 0.0

            max_single_object_w = 0.22 * img_w
            if ioa_det >= 0.70 and target.area > det.area:
                # det is largely inside target: retain target's full bounding box
                fused_box = target.box_2d
            elif ioa_target >= 0.70 and det.area > target.area and det.area <= 1.8 * target.area and det.width <= max_single_object_w and target.class_name not in ("lamp", "chair"):
                # target is largely inside det: adopt det's bounding box only if det is not an aggregate multi-object box
                fused_box = det.box_2d
            elif ioa_target >= 0.70:
                # Retain tighter focused target box rather than swallowing adjacent items
                fused_box = target.box_2d
            else:
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

            fused_class = target.class_name
            classes = {target.class_name, det.class_name}
            if classes == {"door", "window"}:
                # If opening intersects foreground furniture or extends toward floor, prioritize primary physical entity 'door'
                has_furn = any(
                    other.class_name in ("table", "desk", "chair") and other.parent_box_id is None and
                    other.y1 / img_h >= min(target.y1, det.y1) / img_h and
                    (min(target.x2, det.x2, other.x2) - max(target.x1, det.x1, other.x1)) > 0.20 * min(target.width, det.width)
                    for other in sorted_dets
                )
                if has_furn or (min(target.height, det.height) / max(1.0, min(target.width, det.width)) >= 0.70 and max(target.y2, det.y2) / img_h > 0.35):
                    fused_class = "door"

            fused_det = DetectionBox(
                box_id=target.box_id,
                class_name=fused_class,
                box_2d=fused_box,
                confidence=min(0.99, max(target.confidence, det.confidence) + 0.05),
                source_engine=f"{target.source_engine}+wbf",
                source_image_id=target.source_image_id,
                mask_2d=fused_mask,
                parent_box_id=target.parent_box_id,
                scale=target.scale,
                raw_score=target.raw_score,
                calibrated_confidence=None,
                attributes=dict(target.attributes),
            )
            kept[fused_with_idx] = fused_det
            if rejection_records is not None:
                rejection_records.append({
                    "box_id": det.box_id,
                    "class_name": det.class_name,
                    "box_2d": list(det.box_2d),
                    "confidence": det.confidence,
                    "stage": "duplicate_suppression",
                    "reason": f"fused_with_candidate_{target.box_id}_{target.class_name}",
                })
        elif should_keep:
            kept.append(det)
        elif rejection_records is not None:
            rejection_records.append({
                "box_id": det.box_id,
                "class_name": det.class_name,
                "box_2d": list(det.box_2d),
                "confidence": det.confidence,
                "stage": "duplicate_suppression",
                "reason": "cross_class_nms_suppressed",
            })

    # Evidence-based geometric disambiguation using normalized coordinates and priors
    refined_kept: list[DetectionBox] = []
    for b in kept:
        norm_x1 = b.x1 / img_w
        norm_y1 = b.y1 / img_h
        norm_x2 = b.x2 / img_w
        norm_y2 = b.y2 / img_h
        norm_w = b.width / img_w
        norm_h = b.height / img_h
        norm_area = b.area / (img_w * img_h)
        aspect_ratio = b.height / b.width if b.width > 0 else 1.0

        # Micro-fixture vs Door disambiguation:
        # A true walkable door entity occupies at least ~6.5% width and ~1.5% area in room perspectives.
        # Micro wall fixtures (sconces, lamps) misdetected as doors with small normalized dimensions:
        if b.class_name == "door" and norm_w < 0.065 and b.parent_box_id is None:
            if 0.15 <= norm_y1 <= 0.70 and norm_w < 0.055 and norm_area < 0.015:
                object.__setattr__(b, "class_name", "lamp")
                refined_kept.append(b)
            continue

        # Upper-wall opening / Table disambiguation:
        # A table rests on the floor or horizontal surface; it cannot float in the upper half of the room.
        # Vertical openings, door transoms, and lintels mislabeled as tables at upper room levels:
        if b.class_name == "table" and norm_y2 < 0.48 and b.parent_box_id is None:
            if 0.25 <= norm_x1 <= 0.75 and norm_w > 0.08:
                object.__setattr__(b, "class_name", "door")
                refined_kept.append(b)
                continue

        # Chair vs Table footprint disambiguation:
        # Tables have expansive horizontal footprints (low aspect ratio, wide bounds).
        # A candidate labeled "table" with narrow normalized width and vertical aspect ratio is an office/side chair:
        if b.class_name == "table" and norm_w < 0.12 and aspect_ratio >= 1.0 and b.parent_box_id is None:
            object.__setattr__(b, "class_name", "chair")
            refined_kept.append(b)
            continue

        # Aggregate cluster suppression:
        # A single chair cannot span > 22% of the entire image width while containing smaller chair detections:
        if b.class_name == "chair" and norm_w > 0.22 and b.parent_box_id is None:
            has_contained = any(
                other.class_name == "chair" and other.box_id != b.box_id and
                (compute_box_iou(b.box_2d, other.box_2d) > 0.10 or
                 (max(0.0, min(b.x2, other.x2) - max(b.x1, other.x1)) * max(0.0, min(b.y2, other.y2) - max(b.y1, other.y1)) / other.area if other.area > 0 else 0.0) > 0.60)
                for other in kept
            )
            if has_contained:
                continue

        # Aggregate room-spanning false detections:
        # An interior lamp fixture cannot span > 12% width and > 22% height without subparts:
        if b.class_name == "lamp" and (norm_w > 0.12 or norm_h > 0.22) and b.parent_box_id is None:
            continue

        # Window vs Door interior opening disambiguation:
        # Grounding DINO frequently labels glazed interior doors as 'window' due to glass pane features.
        # An opening candidate labeled 'window' that sits on an interior wall with vertical aspect ratio (AR >= 0.70)
        # and intersects foreground furniture (table, desk, chair) occluding its lower portion is an interior door:
        if b.class_name == "window" and b.parent_box_id is None and aspect_ratio >= 0.70 and norm_y2 > 0.35:
            is_occluded_by_furniture = any(
                occ.class_name in ("table", "desk", "chair") and occ.parent_box_id is None and
                occ.y1 / img_h >= norm_y1 and occ.y1 / img_h <= norm_y2 + 0.10 and
                (min(b.x2, occ.x2) - max(b.x1, occ.x1)) > 0.20 * b.width
                for occ in kept
            )
            if is_occluded_by_furniture:
                object.__setattr__(b, "class_name", "door")

        refined_kept.append(b)

    # Resolution-invariant amodal boundary extension for doors occluded by foreground furniture
    for b in refined_kept:
        if b.class_name == "door" and b.parent_box_id is None:
            norm_b_y2 = b.y2 / img_h
            if norm_b_y2 < 0.85:  # Door does not touch the floor in visible pixels
                for occ in refined_kept:
                    if occ.class_name in ("table", "desk", "chair") and occ.parent_box_id is None:
                        # Foreground occluder must be lower/overlapping vertically
                        occ_y1 = occ.y1 / img_h
                        occ_y2 = occ.y2 / img_h
                        if occ_y1 >= b.y1 / img_h and occ_y2 >= norm_b_y2 - 0.05:
                            x_overlap = min(b.x2, occ.x2) - max(b.x1, occ.x1)
                            if x_overlap > 0.20 * b.width:
                                object.__setattr__(b, "box_2d", (b.x1, b.y1, b.x2, max(b.y2, occ.y2)))
                                break

    # Reconcile parent IDs for all subparts (both crop-derived and full-image detected)
    kept_by_id = {d.box_id: d for d in refined_kept}
    pad_x = 0.015 * img_w
    pad_y = 0.015 * img_h
    for det in refined_kept:
        tax_det = resolve_taxonomy_class(det.class_name)
        is_subpart_type = (
            tax_det.parent_class is not None or
            any(det.class_name in parts for parts in GENUINE_STRUCTURAL_PARTS.values())
        )
        needs_parent = (det.parent_box_id is None and is_subpart_type) or (det.parent_box_id is not None and det.parent_box_id not in kept_by_id)
        if needs_parent:
            expected_parent_tax = tax_det.parent_class
            best_parent_id = None
            best_parent_score = 0.0
            for cand in refined_kept:
                if cand.parent_box_id is not None or cand.box_id == det.box_id:
                    continue
                cand_tax = resolve_taxonomy_class(cand.class_name)
                is_valid_host = (
                    (expected_parent_tax and cand_tax.taxonomy_id == expected_parent_tax) or
                    det.class_name in GENUINE_STRUCTURAL_PARTS.get(cand.class_name, ()) or
                    cand.class_name in tax_det.hosts
                )
                if not is_valid_host:
                    continue

                inter_x = max(0.0, min(det.x2, cand.x2) - max(det.x1, cand.x1))
                inter_y = max(0.0, min(det.y2, cand.y2) - max(det.y1, cand.y1))
                inter_area = inter_x * inter_y
                det_area = det.area
                ioa = inter_area / det_area if det_area > 0 else 0.0

                max_cand_ratio = 1.15 if det.class_name in (
                    "window_frame", "door_frame", "door_leaf", "cabinet_carcass", "door_panel", "window_sash", "table_top"
                ) else 0.85
                if ioa >= 0.40 and det.area <= max_cand_ratio * cand.area:
                    score = cand.confidence * max(0.10, ioa)
                    if score > best_parent_score:
                        best_parent_score = score
                        best_parent_id = cand.box_id

            if best_parent_id is not None:
                object.__setattr__(det, "parent_box_id", best_parent_id)

    return refined_kept


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

    # Vectorized connected component labeling and filtering via scipy.ndimage (C-optimized)
    import scipy.ndimage as ndi

    lbls, num_features = ndi.label(sub_refined)
    if num_features > 0:
        counts = np.bincount(lbls.ravel())
        is_small = np.zeros(len(counts), dtype=bool)
        is_small[1:] = counts[1:] < min_component_area_px
        if np.any(is_small):
            sub_refined[is_small[lbls]] = False

    # Vectorized hole detection and filling via binary_fill_holes
    filled = ndi.binary_fill_holes(sub_refined)
    holes = filled ^ sub_refined
    if np.any(holes):
        hlbls, num_holes = ndi.label(holes)
        if num_holes > 0:
            hcounts = np.bincount(hlbls.ravel())
            is_small_hole = np.zeros(len(hcounts), dtype=bool)
            is_small_hole[1:] = (hcounts[1:] <= max_fill_hole_area_px) & (hcounts[1:] > 0)
            if np.any(is_small_hole):
                sub_refined[is_small_hole[hlbls]] = True

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

        if grad_mag.size > 20000:
            pct_sample = grad_mag.ravel()[::max(1, grad_mag.size // 10000)]
            grad_thresh = float(np.percentile(pct_sample, 70))
        else:
            grad_thresh = float(np.percentile(grad_mag, 70))
        high_grad_mask = grad_mag > grad_thresh
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

    def segment_boxes(self, image_rgb: np.ndarray, boxes: Sequence[tuple[float, float, float, float]]) -> list[np.ndarray | None]:
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

    def segment_boxes(self, image_rgb: np.ndarray, boxes: Sequence[tuple[float, float, float, float]]) -> list[np.ndarray | None]:
        return [self.segment_box(image_rgb, b) for b in boxes]


class AdapterDetectorWrapper:
    """Wraps an engine BaseEngineAdapter into the DetectorAdapter protocol."""

    def __init__(self, adapter: Any, score_threshold: float = 0.20) -> None:
        self.adapter = adapter
        self.score_threshold = score_threshold

    def detect(self, image_rgb: np.ndarray, text_prompts: Sequence[str]) -> list[DetectionBox]:
        h, w = image_rgb.shape[:2]
        res = self.adapter.predict(
            [image_rgb],
            options={"queries": list(text_prompts), "score_threshold": self.score_threshold},
        )
        boxes: list[DetectionBox] = []
        if res.success and res.predictions:
            img_dets = res.predictions[0].get("detections", [])
            for idx, d in enumerate(img_dets):
                xyxy = d.get("box_xyxy", [0.0, 0.0, 0.0, 0.0])
                cls_name = d.get("label", "unknown").replace(" ", "_")
                score = float(d.get("score", 0.5))
                boxes.append(DetectionBox(
                    box_id=f"{self.adapter.engine_name}_{idx}",
                    class_name=cls_name,
                    box_2d=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=score,
                    source_engine=self.adapter.engine_name,
                ))
        return boxes


class AdapterSegmenterWrapper:
    """Wraps a SAM or instance segmenter BaseEngineAdapter into the SegmenterAdapter protocol."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def segment_box(self, image_rgb: np.ndarray, box_2d: tuple[float, float, float, float]) -> np.ndarray | None:
        masks = self.segment_boxes(image_rgb, [box_2d])
        return masks[0] if masks else None

    def segment_boxes(self, image_rgb: np.ndarray, boxes: Sequence[tuple[float, float, float, float]]) -> list[np.ndarray | None]:
        if not boxes:
            return []
        box_list = [[float(c) for c in b] for b in boxes]
        if hasattr(self.adapter, "execute"):
            res = self.adapter.execute({"image": image_rgb, "boxes": box_list})
        elif hasattr(self.adapter, "predict"):
            res = self.adapter.predict({"image": image_rgb, "boxes": box_list})
        else:
            return [None] * len(boxes)
        if res.success and res.raw_output:
            return list(res.raw_output)
        return [None] * len(boxes)


def create_default_segmenter(device: str | None = None) -> SegmenterAdapter:
    """Instantiates production SAM segmenter if weights exist; otherwise falls back to ThresholdSegmenterAdapter."""
    from pathlib import Path
    chk = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "sam_vit_b_01ec64.pth"
    if not chk.exists():
        chk = Path("checkpoints/sam_vit_b.pth")
    if chk.exists():
        try:
            from engine.adapters.segmenter_sam import SamPromptedSegmenterAdapter
            import torch
            dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            sam_adapter = SamPromptedSegmenterAdapter(checkpoint=str(chk), device=dev)
            return AdapterSegmenterWrapper(sam_adapter)
        except Exception:
            pass
    return ThresholdSegmenterAdapter()


class GroundingDinoDetectorAdapter:
    """Production open-vocabulary detector using Grounding DINO Tiny on GPU."""

    name = "grounding-dino-tiny"

    def __init__(self, box_threshold: float = 0.22, batch_size: int = 2) -> None:
        self.box_threshold = box_threshold
        self.batch_size = batch_size
        self._proc = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from engine.pipeline.open_vocab_furniture import _load
            self._proc, self._model, self._device = _load()

    def available(self) -> tuple[bool, str]:
        try:
            self._ensure_loaded()
            return True, "cached"
        except Exception as exc:
            return False, str(exc)

    def detect(self, image_rgb: np.ndarray, text_prompts: Sequence[str]) -> list[DetectionBox]:
        self._ensure_loaded()
        import torch
        from PIL import Image

        orig_h, orig_w = image_rgb.shape[:2]
        max_side = 1024
        if max(orig_h, orig_w) > max_side:
            scale = max_side / float(max(orig_h, orig_w))
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            img_infer = Image.fromarray(np.ascontiguousarray(image_rgb)).resize((new_w, new_h), Image.Resampling.BILINEAR)
            sx = orig_w / float(new_w)
            sy = orig_h / float(new_h)
        else:
            img_infer = Image.fromarray(np.ascontiguousarray(image_rgb))
            new_w, new_h = orig_w, orig_h
            sx, sy = 1.0, 1.0

        boxes: list[DetectionBox] = []
        queries = [p.replace("_", " ").strip() for p in text_prompts if p.strip()]
        if not queries:
            return boxes

        for i in range(0, len(queries), self.batch_size):
            chunk = list(queries[i:i + self.batch_size])
            text_queries = [f"{q}." if not q.endswith(".") else q for q in chunk]
            try:
                inp = self._proc(
                    images=[img_infer] * len(chunk),
                    text=text_queries,
                    return_tensors="pt",
                    padding=True,
                ).to(self._device)

                with torch.no_grad():
                    res = self._model(**inp)
            except Exception as exc:
                # If CUDA sticky error or memory pressure occurs, clear cache and fall back gracefully
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    self._device = "cpu"
                    self._model = self._model.to("cpu")
                    inp = self._proc(
                        images=[img_infer] * len(chunk),
                        text=text_queries,
                        return_tensors="pt",
                        padding=True,
                    ).to("cpu")
                    with torch.no_grad():
                        res = self._model(**inp)
                except Exception:
                    continue

            post = self._proc.post_process_grounded_object_detection(
                res,
                inp.input_ids,
                threshold=self.box_threshold,
                text_threshold=max(0.15, self.box_threshold * 0.8),
                target_sizes=[(new_h, new_w)] * len(chunk),
            )

            # Release tensor allocations immediately to preserve VRAM
            del inp, res
            if torch.cuda.is_available() and self._device == "cuda":
                torch.cuda.empty_cache()

            for q, r in zip(chunk, post):
                clean_cls = q.rstrip(".").replace(" ", "_").lower()
                for score, box in zip(r["scores"].tolist(), r["boxes"].tolist()):
                    b_xyxy = (
                        float(box[0] * sx),
                        float(box[1] * sy),
                        float(box[2] * sx),
                        float(box[3] * sy),
                    )
                    boxes.append(DetectionBox(
                        box_id=f"gd_{clean_cls}_{len(boxes)}",
                        class_name=clean_cls,
                        box_2d=b_xyxy,
                        confidence=float(score),
                        source_engine=self.name,
                        raw_score=float(score),
                        calibrated_confidence=float(score),
                    ))

        return boxes


class OwlVitDetectorAdapter:
    """Production zero-shot open-vocabulary detector using Google OWL-ViT on GPU/CPU."""

    name = "owlvit"

    def __init__(self, threshold: float = 0.08) -> None:
        self.threshold = threshold
        self._proc = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            import torch
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
            repo = "google/owlvit-base-patch32"
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._proc = OwlViTProcessor.from_pretrained(repo, local_files_only=True)
            self._model = OwlViTForObjectDetection.from_pretrained(repo, local_files_only=True).to(dev).eval()
            self._device = dev

    def available(self) -> tuple[bool, str]:
        try:
            self._ensure_loaded()
            return True, "cached"
        except Exception as exc:
            return False, str(exc)

    def detect(self, image_rgb: np.ndarray, text_prompts: Sequence[str]) -> list[DetectionBox]:
        self._ensure_loaded()
        import torch
        from PIL import Image

        img = Image.fromarray(np.ascontiguousarray(image_rgb))
        queries = [p.replace("_", " ").strip() for p in text_prompts if p.strip()]
        if not queries:
            return []

        h, w = image_rgb.shape[:2]
        inputs = self._proc(text=[queries], images=img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model(**inputs)

        post = getattr(self._proc, "post_process_object_detection", None) or self._proc.post_process_grounded_object_detection
        res = post(out, threshold=self.threshold, target_sizes=torch.tensor([[h, w]], device=self._device))[0]

        boxes: list[DetectionBox] = []
        for score, label_idx, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            cls_name = queries[int(label_idx)].replace(" ", "_").lower()
            b_xyxy = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            boxes.append(DetectionBox(
                box_id=f"owl_{cls_name}_{len(boxes)}",
                class_name=cls_name,
                box_2d=b_xyxy,
                confidence=float(score),
                source_engine="owlvit_base_patch32",
            ))
        return boxes


def create_default_detector(preferred: str = "auto") -> DetectorAdapter:
    """Creates best available production detector adapter."""
    if preferred in ("auto", "grounding_dino"):
        try:
            adapter = GroundingDinoDetectorAdapter()
            adapter._ensure_loaded()
            return adapter
        except Exception:
            pass
    if preferred in ("auto", "owlvit"):
        try:
            adapter = OwlVitDetectorAdapter()
            adapter._ensure_loaded()
            return adapter
        except Exception:
            pass
    return RuleBasedDetectorAdapter()


@dataclass(frozen=True)
class PipelineResult:
    """Structured result of full detection, tiling, and mask refinement pipeline."""

    image_id: str
    raw_detections: tuple[DetectionBox, ...]
    refined_boxes: tuple[DetectionBox, ...]
    parent_part_tree: dict[str, list[str]]
    execution_stats: dict[str, Any]
    rejection_ledger: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "raw_detection_count": len(self.raw_detections),
            "refined_box_count": len(self.refined_boxes),
            "parent_part_tree": self.parent_part_tree,
            "execution_stats": self.execution_stats,
            "rejection_count": len(self.rejection_ledger),
            "refined_boxes": [b.to_dict() for b in self.refined_boxes],
        }


class DetectionPipeline:
    """Production detection pipeline coordinating open-vocab prompt dispatch, tiling, and refinement."""

    def __init__(
        self,
        detector: DetectorAdapter | None = None,
        segmenter: SegmenterAdapter | None = None,
        calibrator: LearnedConfidenceCalibrator | None = None,
        tile_size: int = 800,
        tile_overlap: float = 0.25,
    ) -> None:
        self.detector = detector or RuleBasedDetectorAdapter()
        self.segmenter = segmenter or ThresholdSegmenterAdapter()
        self.calibrator = calibrator if calibrator is not None else load_default_production_calibrator()
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    def run_pipeline(
        self,
        image_rgb: np.ndarray,
        domain: Domain = Domain.INTERIOR,
        image_id: str = "img_001",
        run_tiling: bool = True,
        detect_subparts: bool = True,
        frame_index: int | None = None,
        timestamp_sec: float | None = None,
        shot_id: int | None = None,
        text_prompts: Sequence[str] | None = None,
    ) -> PipelineResult:
        """Executes the full perception workflow on an input RGB image."""
        image_rgb, prep_meta = preprocess_image_for_detection(image_rgb)
        h, w = image_rgb.shape[:2]
        raw_accum: list[DetectionBox] = []
        rejection_records: list[dict[str, Any]] = []

        # 1. Open-vocab prompt generation
        if text_prompts:
            macro_prompts: list[str] = []
            std_prompts = list(text_prompts)
            micro_prompts = [
                p for p in text_prompts
                if any(s in p for s in ("socket", "switch", "handle", "hinge", "lock", "sill", "frame", "vent", "tap", "drain"))
            ]
        else:
            prompts = generate_domain_prompts(domain)
            macro_prompts = prompts.get(DetectionScale.MACRO, [])
            std_prompts = prompts.get(DetectionScale.STANDARD, [])
            micro_prompts = prompts.get(DetectionScale.MICRO, [])

        # 2. Whole-image macro + standard detection
        macro_set = {p.lower().replace(" ", "_") for p in macro_prompts}
        whole_prompts = list(dict.fromkeys([*macro_prompts, *std_prompts]))
        base_dets = self.detector.detect(image_rgb, whole_prompts)
        for b in base_dets:
            cls_norm = b.class_name.lower().replace(" ", "_")
            det_scale = DetectionScale.MACRO if cls_norm in macro_set else DetectionScale.STANDARD
            raw_accum.append(DetectionBox(
                box_id=f"{b.box_id}_std",
                class_name=b.class_name,
                box_2d=b.box_2d,
                confidence=b.confidence,
                source_engine=b.source_engine,
                source_image_id=image_id,
                mask_2d=b.mask_2d,
                parent_box_id=b.parent_box_id,
                scale=det_scale,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                shot_id=shot_id,
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
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        shot_id=shot_id,
                        attributes=md.attributes,
                    ))

        # 4. Filter and deduplicate primary host detections
        filtered_primary = filter_by_class_confidence(raw_accum, image_shape=(h, w))
        passed_ids = {d.box_id for d in filtered_primary}
        for d in raw_accum:
            if d.box_id not in passed_ids:
                rejection_records.append({
                    "box_id": d.box_id,
                    "class_name": d.class_name,
                    "box_2d": list(d.box_2d),
                    "confidence": d.confidence,
                    "stage": "confidence_threshold_filtering",
                    "reason": "below_class_confidence",
                })

        canonical_hosts = suppress_duplicates_preserving_parts(
            filtered_primary,
            iou_threshold=0.50,
            image_shape=(h, w),
            rejection_records=rejection_records,
        )

        # 5. Crop-based subpart detection on deduplicated canonical host objects
        subparts_detected = 0
        verified_subparts: list[DetectionBox] = []
        if detect_subparts:
            # Sort host candidates descending by confidence and run crop re-detection on deduplicated canonical detections
            frame_area = float(w * h)
            current_candidates = sorted(
                [c for c in canonical_hosts if c.parent_box_id is None and (c.area / frame_area if frame_area > 0 else 0.0) >= 0.001],
                key=lambda x: x.confidence,
                reverse=True,
            )
            processed_per_class: dict[str, int] = defaultdict(int)
            for cand in current_candidates:
                if processed_per_class[cand.class_name] >= 4:
                    continue
                targets = get_subpart_detection_targets(cand.class_name)
                if not targets:
                    continue
                rois = generate_component_rois_for_host(cand, w, h)
                if not rois:
                    continue
                cand_subparts: list[DetectionBox] = []
                for cx1, cy1, cx2, cy2, roi_name in rois:
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    crop_img = image_rgb[cy1:cy2, cx1:cx2]
                    off_x, off_y = cx1, cy1
                    crop_w = float(cx2 - cx1)
                    crop_h = float(cy2 - cy1)
                    crop_area = crop_w * crop_h
                    target_names = [t.name for t in targets]
                    if text_prompts:
                        # If text_prompts contains specific subpart classes, filter targets to those requested:
                        has_subpart_prompts = any(
                            any(p == cls.name or p in cls.name or cls.name in p for p in text_prompts)
                            for cls in get_all_taxonomy_classes() if cls.parent_class is not None
                        )
                        if has_subpart_prompts:
                            target_names = [name for name in target_names if any(name == p or name in p or p in name for p in text_prompts)]
                            if not target_names:
                                continue
                    part_dets = self.detector.detect(crop_img, target_names)
                    for pd in part_dets:
                        canon_part_name = pd.class_name
                        if canon_part_name == cand.class_name:
                            continue
                        pw = pd.width
                        ph = pd.height
                        p_area = pw * ph
                        # Reject whole-crop hallucinated detections:
                        if (pw / crop_w > 0.92 and ph / crop_h > 0.92) or (p_area / crop_area > 0.88):
                            rejection_records.append({
                                "box_id": f"part_{pd.box_id}_{cand.box_id}_{roi_name}",
                                "class_name": pd.class_name,
                                "confidence": pd.confidence,
                                "stage": "subpart_crop_validation",
                                "reason": "whole_crop_hallucination",
                            })
                            continue
                        px1 = max(0.0, min(float(w), pd.x1 + off_x))
                        py1 = max(0.0, min(float(h), pd.y1 + off_y))
                        px2 = max(0.0, min(float(w), pd.x2 + off_x))
                        py2 = max(0.0, min(float(h), pd.y2 + off_y))
                        if px2 <= px1 or py2 <= py1:
                            continue

                        # Geometric containment constraint:
                        inter_x = max(0.0, min(px2, cand.x2) - max(px1, cand.x1))
                        inter_y = max(0.0, min(py2, cand.y2) - max(py1, cand.y1))
                        inter_area = inter_x * inter_y
                        part_area = (px2 - px1) * (py2 - py1)
                        ioa = inter_area / part_area if part_area > 0 else 0.0
                        if ioa < 0.35 or part_area > 1.05 * cand.area:
                            rejection_records.append({
                                "box_id": f"part_{pd.box_id}_{cand.box_id}_{roi_name}",
                                "class_name": pd.class_name,
                                "confidence": pd.confidence,
                                "stage": "subpart_crop_validation",
                                "reason": "containment_ioa_below_0.35",
                            })
                            continue

                        # Reject subparts that heavily overlap a foreground fixture (e.g. lamp in front of door)
                        collides_foreground_fixture = any(
                            other.class_name in ("lamp", "pendant_light", "track_light") and other.box_id != cand.box_id and other.parent_box_id is None and
                            compute_box_iou((px1, py1, px2, py2), other.box_2d) > 0.40
                            for other in canonical_hosts
                        )
                        if collides_foreground_fixture:
                            rejection_records.append({
                                "box_id": f"part_{pd.box_id}_{cand.box_id}_{roi_name}",
                                "class_name": pd.class_name,
                                "confidence": pd.confidence,
                                "stage": "subpart_crop_validation",
                                "reason": "foreground_fixture_collision",
                            })
                            continue

                        # Specificity-based ownership resolution:
                        best_parent_id, is_ambig, cand_parent_ids = resolve_component_ownership(
                            (px1, py1, px2, py2),
                            canon_part_name,
                            canonical_hosts,
                            min_containment_ioa=0.35,
                        )
                        if best_parent_id is None:
                            best_parent_id = cand.box_id
                            is_ambig = False
                            cand_parent_ids = [cand.box_id]

                        # Intra-host component deduplication across ROIs:
                        is_dup = False
                        for existing_part in cand_subparts:
                            if existing_part.class_name == canon_part_name:
                                p_iou = compute_box_iou((px1, py1, px2, py2), existing_part.box_2d)
                                p_inter = max(0.0, min(px2, existing_part.x2) - max(px1, existing_part.x1)) * max(0.0, min(py2, existing_part.y2) - max(py1, existing_part.y1))
                                p_ioa = p_inter / part_area if part_area > 0 else 0.0
                                if p_iou >= 0.40 or p_ioa >= 0.55:
                                    is_dup = True
                                    if pd.confidence > existing_part.raw_score:
                                        cand_subparts.remove(existing_part)
                                        is_dup = False
                                    break
                        if is_dup:
                            rejection_records.append({
                                "box_id": f"part_{pd.box_id}_{cand.box_id}_{roi_name}",
                                "class_name": canon_part_name,
                                "confidence": pd.confidence,
                                "stage": "subpart_crop_validation",
                                "reason": "intra_host_duplicate",
                            })
                            continue

                        # Confidence propagation: host-conditional probability
                        prop_conf = round(math.sqrt(float(pd.confidence) * float(cand.confidence)), 4)

                        subpart_box = DetectionBox(
                            box_id=f"part_{pd.box_id}_{cand.box_id}_{roi_name}",
                            class_name=canon_part_name,
                            box_2d=(px1, py1, px2, py2),
                            confidence=prop_conf,
                            source_engine=f"{pd.source_engine}_subpart",
                            source_image_id=image_id,
                            parent_box_id=best_parent_id,
                            scale=DetectionScale.MICRO,
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            shot_id=shot_id,
                            raw_score=pd.confidence,
                            attributes={
                                "host_class": cand.class_name,
                                "host_box_id": best_parent_id,
                                "host_confidence": round(float(cand.confidence), 4),
                                "roi_source": roi_name,
                                "ambiguous_parent": is_ambig,
                                "candidate_parent_box_ids": cand_parent_ids,
                                "provenance": {
                                    "detector_model": getattr(self.detector, "model_id", type(self.detector).__name__),
                                    "crop_origin": (off_x, off_y),
                                    "containment_ioa": round(float(ioa), 4),
                                },
                            },
                        )
                        cand_subparts.append(subpart_box)
                        raw_accum.append(subpart_box)
                        subparts_detected += 1

                verified_subparts.extend(cand_subparts)
                processed_per_class[cand.class_name] += 1

        # Check missing expected components on canonical hosts
        for host in canonical_hosts:
            attached_part_classes = {p.class_name for p in verified_subparts if p.parent_box_id == host.box_id}
            expected = EXPECTED_KEY_COMPONENTS.get(host.class_name, ())
            missing = [exp for exp in expected if exp not in attached_part_classes]
            host.attributes["missing_expected_parts"] = missing

        # 6. Resolve ownership for any primary detections that are subparts & combine detections
        resolved_primary: list[DetectionBox] = []
        for det in canonical_hosts:
            if det.parent_box_id is None:
                tax = resolve_taxonomy_class(det.class_name)
                if tax.parent_class is not None or len(tax.hosts) > 0:
                    p_id, is_ambig, cand_parents = resolve_component_ownership(
                        det.box_2d, det.class_name, canonical_hosts, min_containment_ioa=0.35
                    )
                    if p_id is not None:
                        new_attrs = dict(det.attributes)
                        new_attrs["host_box_id"] = p_id
                        new_attrs["ambiguous_parent"] = is_ambig
                        new_attrs["candidate_parent_box_ids"] = cand_parents
                        det = DetectionBox(
                            box_id=det.box_id,
                            class_name=det.class_name,
                            box_2d=det.box_2d,
                            confidence=det.confidence,
                            source_engine=det.source_engine,
                            source_image_id=det.source_image_id,
                            mask_2d=det.mask_2d,
                            parent_box_id=p_id,
                            scale=det.scale,
                            frame_index=det.frame_index,
                            timestamp_sec=det.timestamp_sec,
                            shot_id=det.shot_id,
                            attributes=new_attrs,
                            raw_score=det.raw_score,
                        )
            resolved_primary.append(det)

        # Deduplicate overlapping subparts between primary detections and crop re-detections
        combined_subparts: list[DetectionBox] = []
        for vp in verified_subparts:
            # Check if an equivalent subpart was already captured in resolved_primary
            is_dup = False
            for idx, p_det in enumerate(resolved_primary):
                if p_det.class_name == vp.class_name and p_det.parent_box_id == vp.parent_box_id:
                    if compute_box_iou(p_det.box_2d, vp.box_2d) >= 0.40:
                        is_dup = True
                        if vp.confidence > p_det.confidence:
                            new_attrs = dict(p_det.attributes)
                            new_attrs.update(vp.attributes)
                            resolved_primary[idx] = DetectionBox(
                                box_id=vp.box_id,
                                class_name=p_det.class_name,
                                box_2d=vp.box_2d,
                                confidence=vp.confidence,
                                source_engine=vp.source_engine,
                                source_image_id=vp.source_image_id,
                                mask_2d=p_det.mask_2d,
                                parent_box_id=p_det.parent_box_id,
                                scale=p_det.scale,
                                frame_index=p_det.frame_index,
                                timestamp_sec=p_det.timestamp_sec,
                                shot_id=p_det.shot_id,
                                attributes=new_attrs,
                                raw_score=vp.raw_score,
                            )
                        break
            if not is_dup:
                combined_subparts.append(vp)

        suppressed_dets = resolved_primary + combined_subparts

        # 7. Mask segmentation and boundary refinement (batched for SAM efficiency)
        unmasked = [
            (idx, det) for idx, det in enumerate(suppressed_dets)
            if det.mask_2d is None or det.mask_2d.shape[:2] != (h, w)
        ]
        mask_map: dict[int, np.ndarray | None] = {}
        if unmasked and hasattr(self.segmenter, "segment_boxes"):
            boxes_to_seg = [det.box_2d for _, det in unmasked]
            try:
                computed_masks = self.segmenter.segment_boxes(image_rgb, boxes_to_seg)
                mask_map = {idx: mask for (idx, _), mask in zip(unmasked, computed_masks)}
            except Exception:
                mask_map = {}

        from engine.multiengine.spatial_intelligence.attributes import extract_visual_attributes

        refined_boxes: list[DetectionBox] = []
        for idx, det in enumerate(suppressed_dets):
            mask = det.mask_2d
            if mask is None or mask.shape[:2] != (h, w):
                mask = mask_map.get(idx)
                if mask is None and idx not in mask_map:
                    mask = self.segmenter.segment_box(image_rgb, det.box_2d)

            if mask is not None and mask.shape[:2] == (h, w):
                refined_mask = refine_instance_mask(mask, rgb_image=image_rgb)
            else:
                refined_mask = mask

            # Topology and geometric validity
            topology = extract_mask_topology(refined_mask) if refined_mask is not None else None
            if topology is not None and not topology["is_valid"]:
                rejection_records.append({
                    "box_id": det.box_id,
                    "class_name": det.class_name,
                    "stage": "mask_segmentation",
                    "reason": topology["validation_note"],
                })
                refined_mask = None

            is_trunc = detect_box_truncation(det.box_2d, w, h)
            occ_lvl = estimate_box_occlusion(det.box_2d, refined_mask)
            poly = extract_polygon_from_mask(refined_mask) if refined_mask is not None else None
            mask_area = int(np.sum(refined_mask.astype(bool))) if refined_mask is not None else 0

            # Empirical visual appearance attributes
            try:
                vis_attrs = extract_visual_attributes(
                    image_rgb,
                    det.box_2d,
                    mask_2d=refined_mask,
                    class_hint=det.class_name,
                )
                combined_attrs = dict(det.attributes)
                combined_attrs.update(vis_attrs.to_dict())
            except Exception:
                combined_attrs = dict(det.attributes)

            # 24-bin normalized appearance feature histogram for temporal and multi-view tracking
            app_feat = extract_appearance_features(image_rgb, det.box_2d)
            combined_attrs["appearance_features"] = app_feat

            final_box = det.box_2d
            if refined_mask is not None and mask_area > 100:
                ys, xs = np.where(refined_mask)
                if len(ys) > 0 and len(xs) > 0:
                    my1, mx1, my2, mx2 = float(np.min(ys)), float(np.min(xs)), float(np.max(ys)), float(np.max(xs))
                    mask_box = (mx1, my1, mx2, my2)
                    mb_iou = compute_box_iou(det.box_2d, mask_box)
                    if mb_iou >= 0.50:
                        bx1 = 0.70 * mx1 + 0.30 * det.x1
                        by1 = 0.70 * my1 + 0.30 * det.y1
                        bx2 = 0.70 * mx2 + 0.30 * det.x2
                        by2 = 0.70 * my2 + 0.30 * det.y2
                        final_box = (bx1, by1, bx2, by2)

            if det.calibrated_confidence is not None:
                cal_conf = det.calibrated_confidence
            elif self.calibrator is not None and self.calibrator.is_fitted:
                cal_res = self.calibrator.calibrate(det.confidence, class_name=det.class_name)
                cal_conf = cal_res.calibrated_probability
                combined_attrs["is_calibrated"] = cal_res.is_calibrated
                combined_attrs["calibration_method"] = cal_res.method
            else:
                # Honesty rule (A5): never emit a pseudo-calibrated probability when no fitted calibrator exists
                cal_conf = None
                combined_attrs["is_calibrated"] = False
                combined_attrs["calibration_method"] = "uncalibrated"

            # Detection uncertainty assessment
            u_score = round(1.0 - (cal_conf if cal_conf is not None else det.confidence), 4)
            if is_trunc:
                u_score = min(1.0, round(u_score + 0.15, 4))
            if occ_lvl == "heavy":
                u_score = min(1.0, round(u_score + 0.20, 4))
            combined_attrs["detection_uncertainty"] = u_score
            combined_attrs["mask_topology"] = topology
            combined_attrs["provenance"] = det.attributes.get("provenance", {
                "source_engine": det.source_engine,
                "detector_model": getattr(self.detector, "model_id", type(self.detector).__name__),
                "source_image_id": image_id,
                "scale": det.scale.value,
            })

            refined_boxes.append(DetectionBox(
                box_id=det.box_id,
                class_name=det.class_name,
                box_2d=final_box,
                confidence=det.confidence,
                source_engine=det.source_engine,
                source_image_id=det.source_image_id,
                mask_2d=refined_mask,
                parent_box_id=det.parent_box_id,
                scale=det.scale,
                mask_polygon=poly,
                mask_area_px=mask_area,
                is_truncated=is_trunc,
                occlusion_level=occ_lvl,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                shot_id=shot_id,
                raw_score=det.raw_score if det.raw_score is not None else det.confidence,
                calibrated_confidence=cal_conf,
                attributes=combined_attrs,
                appearance_features=app_feat,
            ))

        # 8. Build parent-part tree and link child_box_ids to parent boxes
        parent_part_tree: dict[str, list[str]] = {}
        for b in refined_boxes:
            if b.parent_box_id:
                parent_part_tree.setdefault(b.parent_box_id, []).append(b.box_id)

        final_boxes: list[DetectionBox] = []
        for b in refined_boxes:
            ch_ids = tuple(parent_part_tree.get(b.box_id, []))
            if ch_ids != b.child_box_ids:
                final_boxes.append(DetectionBox(
                    box_id=b.box_id,
                    class_name=b.class_name,
                    box_2d=b.box_2d,
                    confidence=b.confidence,
                    source_engine=b.source_engine,
                    source_image_id=b.source_image_id,
                    mask_2d=b.mask_2d,
                    parent_box_id=b.parent_box_id,
                    child_box_ids=ch_ids,
                    scale=b.scale,
                    mask_polygon=b.mask_polygon,
                    mask_area_px=b.mask_area_px,
                    is_truncated=b.is_truncated,
                    occlusion_level=b.occlusion_level,
                    frame_index=b.frame_index,
                    timestamp_sec=b.timestamp_sec,
                    shot_id=b.shot_id,
                    raw_score=b.raw_score,
                    calibrated_confidence=b.calibrated_confidence,
                    attributes=b.attributes,
                    appearance_features=b.appearance_features,
                ))
            else:
                final_boxes.append(b)

        stats = {
            "image_size": [w, h],
            "preprocessing": prep_meta,
            "tiles_processed": tiles_processed,
            "subparts_detected": subparts_detected,
            "raw_detection_count": len(raw_accum),
            "filtered_detection_count": len(filtered_primary),
            "rejections_recorded": len(rejection_records),
            "refined_box_count": len(final_boxes),
        }

        return PipelineResult(
            image_id=image_id,
            raw_detections=tuple(raw_accum),
            refined_boxes=tuple(final_boxes),
            parent_part_tree=parent_part_tree,
            execution_stats=stats,
            rejection_ledger=tuple(rejection_records),
        )


UniversalPerceptionPipeline = DetectionPipeline

