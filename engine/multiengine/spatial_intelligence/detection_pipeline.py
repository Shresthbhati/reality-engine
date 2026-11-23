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

GENUINE_STRUCTURAL_PARTS: dict[str, tuple[str, ...]] = {
    "door": ("door_handle", "door_frame", "door_lock", "door_hinge", "door_panel", "door_glazing"),
    "window": ("window_frame", "window_sill", "window_glazing", "window_glass", "window_handle", "window_sash", "window_mullion"),
    "table": ("table_top", "furniture_leg", "table_leg", "drawer", "drawer_handle"),
    "desk": ("drawer", "drawer_handle", "socket", "furniture_leg", "table_leg", "table_top"),
    "chair": ("chair_seat", "chair_backrest", "chair_armrest", "furniture_leg"),
    "cabinet": ("cabinet_door", "cabinet_handle", "drawer", "drawer_handle", "shelf"),
    "electrical_outlet": ("socket", "switch", "outlet_plate", "control_switch"),
    "switchboard_plate": ("socket", "switch", "control_switch"),
    "socket": ("switch",),
    "light": ("light_fixture", "lamp_shade", "light_housing", "diffuser", "mounting_hardware"),
    "lamp": ("lamp_shade", "light_fixture", "diffuser", "light_housing"),
    "ceiling_light": ("light_fixture", "lamp_shade", "diffuser", "light_housing"),
    "faucet": ("faucet_handle", "spout"),
    "sink": ("faucet", "basin", "drain", "pipe"),
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

            # Genuine taxonomy part-parent relationships
            det_is_part_of_existing = (
                det.parent_box_id == existing.box_id or
                tax_det.parent_class == tax_existing.taxonomy_id or
                tax_det.name in GENUINE_STRUCTURAL_PARTS.get(tax_existing.name, ()) or
                (tax_det.name in tax_existing.potential_parts and det.scale == DetectionScale.MICRO)
            )
            existing_is_part_of_det = (
                existing.parent_box_id == det.box_id or
                tax_existing.parent_class == tax_det.taxonomy_id or
                tax_existing.name in GENUINE_STRUCTURAL_PARTS.get(tax_det.name, ()) or
                (tax_existing.name in tax_det.potential_parts and existing.scale == DetectionScale.MICRO)
            )

            if det_is_part_of_existing:
                if det.parent_box_id is None:
                    object.__setattr__(det, "parent_box_id", existing.box_id)
                continue
            elif existing_is_part_of_det:
                if existing.parent_box_id is None:
                    object.__setattr__(existing, "parent_box_id", det.box_id)
                continue

            # Calculate containment IoA (Intersection over Area)
            inter_area = max(0.0, min(det.x2, existing.x2) - max(det.x1, existing.x1)) * max(0.0, min(det.y2, existing.y2) - max(det.y1, existing.y1))
            area_det = det.area
            area_existing = existing.area
            ioa_det = inter_area / area_det if area_det > 0 else 0.0
            ioa_existing = inter_area / area_existing if area_existing > 0 else 0.0

            is_compat = are_semantically_compatible(det.class_name, existing.class_name)

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
            elif ioa_target >= 0.70 and det.area > target.area and det.width <= max_single_object_w:
                # target is largely inside det: adopt det's bounding box only if det is not an aggregate multi-object box
                fused_box = det.box_2d
            elif ioa_target >= 0.70 and target.width <= max_single_object_w:
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
                raw_score=target.raw_score,
                calibrated_confidence=min(0.99, max(target.confidence, det.confidence) + 0.05),
                attributes=dict(target.attributes),
            )
            kept[fused_with_idx] = fused_det
        elif should_keep:
            kept.append(det)

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
                compute_box_iou(b.box_2d, other.box_2d) > 0.20
                for other in kept
            )
            if has_contained:
                continue

        # Aggregate room-spanning false detections:
        # An interior lamp fixture cannot span > 12% width and > 22% height without subparts:
        if b.class_name == "lamp" and (norm_w > 0.12 or norm_h > 0.22) and b.parent_box_id is None:
            continue

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
                            if x_overlap > 0.25 * b.width:
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

                if ioa >= 0.20 or (det.x1 >= cand.x1 - pad_x and det.y1 >= cand.y1 - pad_y and det.x2 <= cand.x2 + pad_x and det.y2 <= cand.y2 + pad_y):
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
            inp = self._proc(
                images=[img_infer] * len(chunk),
                text=text_queries,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                res = self._model(**inp)

            post = self._proc.post_process_grounded_object_detection(
                res,
                inp.input_ids,
                threshold=self.box_threshold,
                text_threshold=max(0.15, self.box_threshold * 0.8),
                target_sizes=[(new_h, new_w)] * len(chunk),
            )

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
        h, w = image_rgb.shape[:2]
        raw_accum: list[DetectionBox] = []

        # 1. Open-vocab prompt generation
        if text_prompts:
            std_prompts = list(text_prompts)
            micro_prompts = [
                p for p in text_prompts
                if any(s in p for s in ("socket", "switch", "handle", "hinge", "lock", "sill", "frame", "vent", "tap", "drain"))
            ]
        else:
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

        # 4. Crop-based subpart detection on host objects
        subparts_detected = 0
        if detect_subparts:
            # Sort host candidates descending by confidence and run crop re-detection on prominent detections
            frame_area = float(w * h)
            current_candidates = sorted(
                [c for c in raw_accum if c.confidence >= 0.22 and (c.area / frame_area if frame_area > 0 else 0.0) >= 0.005],
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
                cx1, cy1, cx2, cy2 = extract_host_crop_roi(cand, w, h)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                crop_img = image_rgb[cy1:cy2, cx1:cx2]
                off_x, off_y = cx1, cy1
                crop_w = float(cx2 - cx1)
                crop_h = float(cy2 - cy1)
                crop_area = crop_w * crop_h
                target_names = [t.name for t in targets]
                part_dets = self.detector.detect(crop_img, target_names)
                for pd in part_dets:
                    if pd.class_name == cand.class_name:
                        continue
                    pw = pd.width
                    ph = pd.height
                    p_area = pw * ph
                    # Reject whole-crop hallucinated detections:
                    # A genuine subpart cannot cover the entire parent host crop
                    if (pw / crop_w > 0.80 and ph / crop_h > 0.80) or (p_area / crop_area > 0.65):
                        continue
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
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            shot_id=shot_id,
                            raw_score=pd.confidence,
                            attributes={"host_class": cand.class_name},
                        ))
                        subparts_detected += 1
                processed_per_class[cand.class_name] += 1

        # 5. Class-specific confidence filtering & whole-image false positive suppression
        filtered_dets = filter_by_class_confidence(raw_accum, image_shape=(h, w))

        # 6. Duplicate suppression preserving subparts and cross-class NMS
        suppressed_dets = suppress_duplicates_preserving_parts(filtered_dets, iou_threshold=0.50, image_shape=(h, w))

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
                if mask is None:
                    mask = self.segmenter.segment_box(image_rgb, det.box_2d)

            if mask is not None and mask.shape[:2] == (h, w):
                refined_mask = refine_instance_mask(mask, rgb_image=image_rgb)
            else:
                refined_mask = mask

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
                cal_conf = calibrate_detection_confidence(
                    det.confidence,
                    det.class_name,
                    calibrator=self.calibrator,
                )
                combined_attrs["is_calibrated"] = False
                combined_attrs["calibration_method"] = "uncalibrated"

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
            "tiles_processed": tiles_processed,
            "subparts_detected": subparts_detected,
            "raw_detection_count": len(raw_accum),
            "filtered_detection_count": len(filtered_dets),
            "refined_box_count": len(final_boxes),
        }

        return PipelineResult(
            image_id=image_id,
            raw_detections=tuple(raw_accum),
            refined_boxes=tuple(final_boxes),
            parent_part_tree=parent_part_tree,
            execution_stats=stats,
        )


UniversalPerceptionPipeline = DetectionPipeline

