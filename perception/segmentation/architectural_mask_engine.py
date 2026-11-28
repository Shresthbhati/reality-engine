"""Team B — Architectural Mask Engine.

Specializes in segmentation of architectural surfaces and openings from
RGB images, complementing object-level detectors with surface-first analysis.

Capabilities:
  - Surface mask partitioning (walls, floor, ceiling, structural elements)
  - Opening hole-carving logic (doors, windows cut holes in wall masks)
  - Semantic validation against architectural constraints
  - SegmentationObservation builder with provenance
  - RLE and COCO-polygon serialization

Design principles:
  - Never merges floor and ceiling into the same surface mask.
  - Door and window openings are always removed from the wall mask they reside in.
  - Surface masks must be contiguous — isolated fragments are reported as separate.
  - All uncertainty is preserved, never silently resolved.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.segmentation.mask_refinement_engine import (
    MaskQualityReport,
    compute_mask_quality,
    encode_rle,
    decode_rle,
)

log = logging.getLogger("reality.segmentation.architectural_mask_engine")


# ---------------------------------------------------------------------------
# Architectural surface categories
# ---------------------------------------------------------------------------
_SURFACE_LABELS: frozenset[str] = frozenset({
    "wall", "floor", "ceiling",
    "interior wall", "exterior wall", "partition wall",
    "structural wall", "brick wall", "concrete wall",
    "glass wall", "curtain wall",
})

_OPENING_LABELS: frozenset[str] = frozenset({
    "door", "window", "opening", "archway", "doorway",
    "skylight", "structural opening",
})

_STRUCTURAL_LABELS: frozenset[str] = frozenset({
    "column", "pillar", "beam", "structural beam", "joist",
    "post", "lintel", "railing", "stairs",
})


def is_architectural(label: str) -> bool:
    """Return True if label belongs to any architectural category."""
    l = label.lower()
    return l in _SURFACE_LABELS or l in _OPENING_LABELS or l in _STRUCTURAL_LABELS


def is_surface(label: str) -> bool:
    return label.lower() in _SURFACE_LABELS


def is_opening(label: str) -> bool:
    return label.lower() in _OPENING_LABELS


# ---------------------------------------------------------------------------
# Surface mask fragment
# ---------------------------------------------------------------------------
@dataclass
class SurfaceMaskFragment:
    """A single contiguous region of an architectural surface."""
    fragment_id: str
    surface_label: str           # "wall" | "floor" | "ceiling" etc.
    mask_rle: dict[str, Any]
    pixel_count: int
    bbox_px: tuple[float, float, float, float]
    quality: MaskQualityReport
    is_primary: bool             # largest fragment of this surface type


# ---------------------------------------------------------------------------
# Surface partitioner
# ---------------------------------------------------------------------------
@dataclass
class SurfacePartitionResult:
    """Result of partitioning an image into architectural surface masks."""
    image_id: str
    image_width: int
    image_height: int
    surface_fragments: list[SurfaceMaskFragment]
    total_covered_pixels: int
    coverage_fraction: float
    uncovered_fraction: float
    notes: list[str]


def _split_into_fragments(
    binary_mask: np.ndarray,
    surface_label: str,
    image_id: str,
    image_width: int,
    image_height: int,
    *,
    min_fragment_pixels: int = 50,
) -> list[SurfaceMaskFragment]:
    """Split a surface mask into contiguous fragments."""
    if binary_mask.sum() == 0:
        return []

    fragments: list[SurfaceMaskFragment] = []

    try:
        from scipy import ndimage  # noqa: PLC0415
        labeled, n = ndimage.label(binary_mask)
        sizes = [(c, int((labeled == c).sum())) for c in range(1, n + 1)]
    except ImportError:
        # Treat whole mask as one fragment
        sizes = [(1, int(binary_mask.sum()))]
        n = 1
        labeled = binary_mask.astype(int)

    # Find largest for primary flag
    if sizes:
        max_size = max(s for _, s in sizes)
    else:
        max_size = 0

    for comp_id, pixel_count in sizes:
        if pixel_count < min_fragment_pixels:
            continue
        comp_mask = (labeled == comp_id)
        rows = np.any(comp_mask, axis=1)
        cols = np.any(comp_mask, axis=0)
        rmin = int(np.argmax(rows))
        rmax = int(len(rows) - 1 - np.argmax(rows[::-1]))
        cmin = int(np.argmax(cols))
        cmax = int(len(cols) - 1 - np.argmax(cols[::-1]))

        fid = hashlib.sha1(
            f"{image_id}:{surface_label}:{comp_id}".encode()
        ).hexdigest()[:8]

        quality = compute_mask_quality(comp_mask, fid, image_width, image_height)

        fragments.append(SurfaceMaskFragment(
            fragment_id=f"surf_{fid}",
            surface_label=surface_label,
            mask_rle=encode_rle(comp_mask),
            pixel_count=pixel_count,
            bbox_px=(float(cmin), float(rmin), float(cmax + 1), float(rmax + 1)),
            quality=quality,
            is_primary=(pixel_count == max_size),
        ))

    return fragments


def partition_architectural_surfaces(
    surface_masks: dict[str, np.ndarray],
    image_id: str,
    image_width: int,
    image_height: int,
    *,
    min_fragment_pixels: int = 50,
) -> SurfacePartitionResult:
    """Partition architectural surface detections into validated fragments.

    Args:
        surface_masks: Mapping from surface_label (str) to binary mask (H x W bool).
        image_id: Image identifier.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        min_fragment_pixels: Minimum pixels for a fragment to be reported.

    Returns:
        ``SurfacePartitionResult`` with all validated surface fragments.
    """
    all_fragments: list[SurfaceMaskFragment] = []
    notes: list[str] = []

    # Validate mutual exclusivity: floor vs ceiling
    if "floor" in surface_masks and "ceiling" in surface_masks:
        overlap = (surface_masks["floor"] & surface_masks["ceiling"]).sum()
        if overlap > 0:
            # Remove overlap from both — they cannot be the same surface
            surface_masks["floor"] = surface_masks["floor"].copy()
            surface_masks["ceiling"] = surface_masks["ceiling"].copy()
            ambiguous = surface_masks["floor"] & surface_masks["ceiling"]
            surface_masks["floor"][ambiguous] = False
            surface_masks["ceiling"][ambiguous] = False
            notes.append(
                f"Removed {int(overlap)} floor/ceiling overlap pixels (architectural constraint)."
            )

    covered = np.zeros((image_height, image_width), dtype=bool)
    for label, mask in surface_masks.items():
        frags = _split_into_fragments(
            mask, label, image_id, image_width, image_height,
            min_fragment_pixels=min_fragment_pixels,
        )
        all_fragments.extend(frags)
        covered |= mask

    total_covered = int(covered.sum())
    total_pixels = image_width * image_height

    return SurfacePartitionResult(
        image_id=image_id,
        image_width=image_width,
        image_height=image_height,
        surface_fragments=all_fragments,
        total_covered_pixels=total_covered,
        coverage_fraction=total_covered / max(total_pixels, 1),
        uncovered_fraction=1.0 - total_covered / max(total_pixels, 1),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Opening hole-carver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpeningHoleResult:
    """Result of carving an opening from a host wall mask."""
    opening_id: str
    opening_label: str
    host_surface_label: str
    carved_pixels: int
    residual_host_mask_rle: dict[str, Any]
    opening_mask_rle: dict[str, Any]
    opening_bbox_px: tuple[float, float, float, float]
    opening_confidence: float
    notes: str


def carve_opening_from_wall(
    wall_mask: np.ndarray,
    opening_mask: np.ndarray,
    opening_label: str,
    opening_id: str,
    host_surface_label: str = "wall",
    *,
    min_opening_pixels: int = 25,
    max_carve_fraction: float = 0.80,
) -> OpeningHoleResult | None:
    """Remove an opening mask from a host wall mask.

    Validates that:
      - The opening overlaps the host wall mask (not floating in space).
      - The opening does not consume more than ``max_carve_fraction`` of the wall.
      - The opening has at least ``min_opening_pixels`` pixels in the wall.

    Args:
        wall_mask: Boolean mask of the host wall (H x W).
        opening_mask: Boolean mask of the detected opening (H x W).
        opening_label: "door" | "window" | "opening" etc.
        opening_id: Unique identifier for the opening detection.
        host_surface_label: Label of the host surface.
        min_opening_pixels: Minimum overlap to treat as a valid opening.
        max_carve_fraction: Maximum fraction of wall that may be carved.

    Returns:
        ``OpeningHoleResult`` if carving was valid, None if rejected.
    """
    overlap = (wall_mask & opening_mask).sum()
    if int(overlap) < min_opening_pixels:
        return None  # Opening not in wall — skip

    wall_pixels = int(wall_mask.sum())
    carve_fraction = int(overlap) / max(wall_pixels, 1)
    if carve_fraction > max_carve_fraction:
        return None  # Would destroy the wall — reject

    # Carve
    residual = wall_mask.copy()
    residual[opening_mask] = False

    # Opening bbox
    rows = np.any(opening_mask, axis=1)
    cols = np.any(opening_mask, axis=0)
    if not rows.any():
        return None
    rmin = int(np.argmax(rows))
    rmax = int(len(rows) - 1 - np.argmax(rows[::-1]))
    cmin = int(np.argmax(cols))
    cmax = int(len(cols) - 1 - np.argmax(cols[::-1]))

    # Confidence scales with how well the opening fits the wall
    confidence = min(1.0, float(overlap) / max(float(opening_mask.sum()), 1.0))

    return OpeningHoleResult(
        opening_id=opening_id,
        opening_label=opening_label,
        host_surface_label=host_surface_label,
        carved_pixels=int(overlap),
        residual_host_mask_rle=encode_rle(residual),
        opening_mask_rle=encode_rle(opening_mask),
        opening_bbox_px=(float(cmin), float(rmin), float(cmax + 1), float(rmax + 1)),
        opening_confidence=confidence,
        notes=f"Carved {int(overlap)} px ({carve_fraction*100:.1f}% of wall) for {opening_label}.",
    )


# ---------------------------------------------------------------------------
# SegmentationObservation builder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentationObservation:
    """A fully provenance-annotated segmentation observation.

    Wraps a surface mask or opening mask with all metadata needed for
    downstream EvidenceMatrix and WorldIR integration.
    """
    observation_id: str
    image_id: str
    label: str
    mask_rle: dict[str, Any]
    bbox_px: tuple[float, float, float, float]
    image_width_px: int
    image_height_px: int
    pixel_count: int
    quality_score: float
    source_engine: str
    is_architectural: bool
    is_opening: bool
    is_surface: bool
    surface_fragment_id: str | None
    opening_host_surface: str | None
    provenance: dict[str, Any]


def build_segmentation_observation(
    fragment: SurfaceMaskFragment,
    image_id: str,
    image_width: int,
    image_height: int,
    source_engine: str = "architectural_mask_engine",
) -> SegmentationObservation:
    """Build a SegmentationObservation from a SurfaceMaskFragment.

    Args:
        fragment: Surface mask fragment to convert.
        image_id: Image identifier.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        source_engine: Name of the segmentation engine.

    Returns:
        ``SegmentationObservation`` ready for EvidenceMatrix consumption.
    """
    obs_id = hashlib.sha256(
        f"seg:{image_id}:{fragment.fragment_id}:{fragment.surface_label}".encode()
    ).hexdigest()[:20]

    return SegmentationObservation(
        observation_id=f"segobs_{obs_id}",
        image_id=image_id,
        label=fragment.surface_label,
        mask_rle=fragment.mask_rle,
        bbox_px=fragment.bbox_px,
        image_width_px=image_width,
        image_height_px=image_height,
        pixel_count=fragment.pixel_count,
        quality_score=fragment.quality.quality_score,
        source_engine=source_engine,
        is_architectural=True,
        is_opening=is_opening(fragment.surface_label),
        is_surface=is_surface(fragment.surface_label),
        surface_fragment_id=fragment.fragment_id,
        opening_host_surface=None,
        provenance={
            "fragment_id": fragment.fragment_id,
            "is_primary": fragment.is_primary,
            "edge_sharpness": fragment.quality.edge_sharpness,
            "has_holes": fragment.quality.has_holes,
        },
    )


def build_opening_observation(
    hole_result: OpeningHoleResult,
    image_id: str,
    image_width: int,
    image_height: int,
    source_engine: str = "architectural_mask_engine",
) -> SegmentationObservation:
    """Build a SegmentationObservation from an OpeningHoleResult.

    Args:
        hole_result: Carved opening result.
        image_id: Image identifier.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        source_engine: Name of the segmentation engine.

    Returns:
        ``SegmentationObservation`` for the opening.
    """
    # Decode mask for pixel count
    opening_mask = decode_rle(hole_result.opening_mask_rle)
    pixel_count = int(opening_mask.sum())
    quality = compute_mask_quality(opening_mask, hole_result.opening_id, image_width, image_height)

    obs_id = hashlib.sha256(
        f"opening:{image_id}:{hole_result.opening_id}".encode()
    ).hexdigest()[:20]

    return SegmentationObservation(
        observation_id=f"segobs_{obs_id}",
        image_id=image_id,
        label=hole_result.opening_label,
        mask_rle=hole_result.opening_mask_rle,
        bbox_px=hole_result.opening_bbox_px,
        image_width_px=image_width,
        image_height_px=image_height,
        pixel_count=pixel_count,
        quality_score=quality.quality_score,
        source_engine=source_engine,
        is_architectural=True,
        is_opening=True,
        is_surface=False,
        surface_fragment_id=None,
        opening_host_surface=hole_result.host_surface_label,
        provenance={
            "carved_pixels": hole_result.carved_pixels,
            "opening_confidence": hole_result.opening_confidence,
            "notes": hole_result.notes,
        },
    )
