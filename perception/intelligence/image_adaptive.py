"""Image Understanding & Adaptive Processing Subsystem (A3.6).

Analyzes:
  - Resolution & aspect ratio
  - Blur (Laplacian variance)
  - Exposure (underexposure / overexposure / clipping fractions)
  - Sensor noise (high-frequency median absolute deviation)
  - Contrast & texture density
  - Useful detection scale & adaptive region proposals

Plans targeted high-resolution crops only where evidence justifies extra compute
(e.g., dense host objects or high-texture regions), avoiding blind full-frame
reprocessing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    TileWindow,
    compute_box_iou,
    generate_tiling_windows,
)


@dataclass(frozen=True)
class ImageQualityProfile:
    width: int
    height: int
    mean_luma: float
    spatial_std_luma: float
    laplacian_var: float
    noise_sigma_est: float
    texture_density: float
    underexposed_fraction: float
    overexposed_fraction: float
    is_blank: bool
    is_blurry: bool
    is_low_contrast: bool
    is_underexposed: bool
    is_overexposed: bool
    recommended_scales: tuple[str, ...]
    quality_grade: str  # "good" | "degraded" | "unusable_blank"

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "mean_luma": round(self.mean_luma, 2),
            "spatial_std_luma": round(self.spatial_std_luma, 2),
            "laplacian_var": round(self.laplacian_var, 2),
            "noise_sigma_est": round(self.noise_sigma_est, 2),
            "texture_density": round(self.texture_density, 4),
            "underexposed_fraction": round(self.underexposed_fraction, 4),
            "overexposed_fraction": round(self.overexposed_fraction, 4),
            "is_blank": self.is_blank,
            "is_blurry": self.is_blurry,
            "is_low_contrast": self.is_low_contrast,
            "is_underexposed": self.is_underexposed,
            "is_overexposed": self.is_overexposed,
            "recommended_scales": list(self.recommended_scales),
            "quality_grade": self.quality_grade,
        }


def analyze_image_quality(image_rgb: np.ndarray) -> ImageQualityProfile:
    """Inspect resolution, blur, noise, exposure, contrast, and texture."""
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be (H, W, 3)")
    h, w = image_rgb.shape[:2]
    spatial_std = float(np.std(image_rgb, axis=(0, 1)).max()) if image_rgb.size > 0 else 0.0
    stride = max(1, max(h, w) // 256)
    sub = image_rgb[::stride, ::stride].astype(np.float32)
    luma = sub[..., 0] * 0.299 + sub[..., 1] * 0.587 + sub[..., 2] * 0.114
    mean_luma = float(np.mean(luma))
    std_luma = float(np.std(luma))
    under_frac = float(np.mean(luma < 25.0))
    over_frac = float(np.mean(luma > 235.0))

    lap_var = 0.0
    noise_est = 0.0
    tex_density = 0.0
    if luma.shape[0] >= 4 and luma.shape[1] >= 4:
        lap = (
            -4.0 * luma[1:-1, 1:-1]
            + luma[:-2, 1:-1]
            + luma[2:, 1:-1]
            + luma[1:-1, :-2]
            + luma[1:-1, 2:]
        )
        lap_var = float(np.var(lap))
        # High-frequency MAD estimator (Immerkaer-style)
        noise_est = float(np.median(np.abs(lap)) / (0.6745 * 4.0))
        gx = np.abs(luma[:, 1:] - luma[:, :-1])
        tex_density = float(np.mean(gx > 12.0))

    is_blank = spatial_std < 2.0
    is_blurry = (not is_blank) and lap_var < 25.0
    is_low_contrast = spatial_std < 15.0
    is_under = mean_luma < 35.0 or under_frac > 0.50
    is_over = mean_luma > 225.0 or over_frac > 0.50

    if is_blank:
        grade = "unusable_blank"
        scales: tuple[str, ...] = ()
    elif is_blurry or is_low_contrast or is_under or is_over:
        grade = "degraded"
        scales = ("macro", "standard")
    else:
        grade = "good"
        scales = ("macro", "standard", "micro") if max(w, h) >= 800 else ("macro", "standard")

    return ImageQualityProfile(
        width=w,
        height=h,
        mean_luma=mean_luma,
        spatial_std_luma=std_luma,
        laplacian_var=lap_var,
        noise_sigma_est=noise_est,
        texture_density=tex_density,
        underexposed_fraction=under_frac,
        overexposed_fraction=over_frac,
        is_blank=is_blank,
        is_blurry=is_blurry,
        is_low_contrast=is_low_contrast,
        is_underexposed=is_under,
        is_overexposed=is_over,
        recommended_scales=scales,
        quality_grade=grade,
    )


@dataclass(frozen=True)
class RegionProposal:
    proposal_id: str
    box_px: tuple[int, int, int, int]
    reason: str
    priority: float
    target_classes: tuple[str, ...] = ()


def plan_adaptive_crops(
    quality: ImageQualityProfile,
    coarse_detections: Sequence[DetectionBox],
    *,
    max_crops: int = 8,
    tile_size: int = 800,
) -> tuple[list[TileWindow], list[RegionProposal]]:
    """Select targeted high-res crops and adaptive tiling windows without blind over-tiling."""
    if quality.is_blank:
        return [], []

    tiles: list[TileWindow] = []
    if max(quality.width, quality.height) > tile_size and "micro" in quality.recommended_scales:
        tiles = generate_tiling_windows(quality.width, quality.height, tile_size=tile_size, overlap_ratio=0.20)

    proposals: list[RegionProposal] = []
    # Propose targeted crops around host objects (doors, windows, desks, cabinets, walls)
    host_targets = {
        "door": ("door_frame", "door_handle", "door_lock"),
        "window": ("window_frame", "window_sill", "window_handle"),
        "desk": ("drawer", "lamp", "monitor"),
        "cabinet": ("drawer", "furniture_handle"),
        "table": ("chair", "table_top"),
    }
    for idx, det in enumerate(sorted(coarse_detections, key=lambda d: d.confidence, reverse=True)):
        targets = host_targets.get(det.class_name)
        if not targets:
            continue
        x1 = max(0, int(det.x1 - 0.08 * det.width))
        y1 = max(0, int(det.y1 - 0.08 * det.height))
        x2 = min(quality.width, int(det.x2 + 0.08 * det.width))
        y2 = min(quality.height, int(det.y2 + 0.08 * det.height))
        if (x2 - x1) >= 48 and (y2 - y1) >= 48:
            proposals.append(RegionProposal(
                proposal_id=f"crop_{idx}_{det.box_id}",
                box_px=(x1, y1, x2, y2),
                reason=f"subpart_refinement_for_{det.class_name}",
                priority=round(float(det.confidence), 4),
                target_classes=targets,
            ))
        if len(proposals) >= max_crops:
            break

    return tiles, proposals


def transform_crop_box_to_image(
    box_crop_px: tuple[float, float, float, float],
    crop_origin_xy: tuple[int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Transform a crop-local XYXY box back into clamped full-image coordinates."""
    ox, oy = crop_origin_xy
    x1 = max(0.0, min(float(image_width), box_crop_px[0] + ox))
    y1 = max(0.0, min(float(image_height), box_crop_px[1] + oy))
    x2 = max(0.0, min(float(image_width), box_crop_px[2] + ox))
    y2 = max(0.0, min(float(image_height), box_crop_px[3] + oy))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def reconcile_crop_duplicates(
    boxes: Sequence[DetectionBox],
    *,
    iou_threshold: float = 0.55,
) -> list[DetectionBox]:
    """Deduplicate overlapping same-class detections across full-frame, tiles, and targeted crops."""
    ordered = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[DetectionBox] = []
    for b in ordered:
        if any(k.class_name == b.class_name and compute_box_iou(k.box_2d, b.box_2d) >= iou_threshold for k in kept):
            continue
        kept.append(b)
    return kept


__all__ = [
    "ImageQualityProfile",
    "RegionProposal",
    "analyze_image_quality",
    "plan_adaptive_crops",
    "reconcile_crop_duplicates",
    "transform_crop_box_to_image",
]
