"""Voxel sizing for the detail chain that does not pretend arbitrary reconstruction units are metres.

A reconstruction without a measured scale reference is RELATIVE: its unit is whatever COLMAP's gauge happened to be
(South Building: the 1st-99th percentile footprint is 6.3 units, the bounding box 67.9). A voxel of "1.0" is then
neither a metre nor a defensible fraction of the scene. So:

    explicit size given            -> used as is (the caller knows better)
    scale is metric                -> METRIC_DEFAULT_VOXEL_M (a metre; the documented absolute default)
    scale is relative / unknown    -> RELATIVE_VOXEL_FRACTION of the ROBUST scene extent

The robust extent is the diagonal of the 1st-99th percentile box of the points. A plain bounding box is dominated
by a handful of far-flung tracks (measured on the real dataset: 10x too large), and a voxel derived from it would
silently land on the coarse side of the cliff that scripts/measure_detail_thresholds.py found.

RELATIVE_VOXEL_FRACTION is EMPIRICAL and PROVISIONAL: set from one real dataset (docs/engineering/DETAIL_CALIBRATION.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

#: fraction of the robust scene extent used as the voxel edge when the scale is not metric. Empirical (South Building).
RELATIVE_VOXEL_FRACTION = 0.16
#: the box that defines the robust extent
EXTENT_PERCENTILES = (1.0, 99.0)
#: below this a percentile box means nothing
MIN_POINTS_FOR_EXTENT = 50
#: the documented absolute default for a METRIC world (metres)
METRIC_DEFAULT_VOXEL_M = 1.0


@dataclass(frozen=True)
class VoxelChoice:
    size: float
    basis: str                     # "explicit" | "metric_default" | "scene_extent_fraction"
    scene_extent: Optional[float]  # robust extent in the reconstruction's own units (None when not needed/measurable)

    def to_dict(self) -> dict:
        return {"size": self.size, "basis": self.basis, "scene_extent": self.scene_extent,
                "fraction": RELATIVE_VOXEL_FRACTION if self.basis == "scene_extent_fraction" else None}


def robust_scene_extent(points: Sequence) -> Optional[float]:
    """Diagonal of the 1st-99th percentile box of ``points`` (objects with ``.position``); None when unmeasurable."""
    if len(points) < MIN_POINTS_FOR_EXTENT:
        return None
    import numpy as np

    pos = np.asarray([[float(c) for c in p.position] for p in points], float)
    lo = np.percentile(pos, EXTENT_PERCENTILES[0], axis=0)
    hi = np.percentile(pos, EXTENT_PERCENTILES[1], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return diag if math.isfinite(diag) and diag > 0.0 else None


def derive_voxel_size(points: Sequence, scale_state: str, explicit: Optional[float] = None) -> VoxelChoice:
    """Choose the detail-chain voxel edge. Raises ValueError when a relative-scale size is required but the scene
    extent cannot be measured (too few points): refusing is honest, a guessed size is not."""
    if explicit is not None:
        if explicit <= 0.0:
            raise ValueError("voxel size must be positive")
        return VoxelChoice(float(explicit), "explicit", None)
    if scale_state == "metric":
        return VoxelChoice(METRIC_DEFAULT_VOXEL_M, "metric_default", None)
    extent = robust_scene_extent(points)
    if extent is None:
        raise ValueError(
            f"scale is {scale_state!r} and the scene extent cannot be measured from {len(points)} point(s) "
            f"(need >= {MIN_POINTS_FOR_EXTENT}); refusing to assume a metric voxel size")
    return VoxelChoice(RELATIVE_VOXEL_FRACTION * extent, "scene_extent_fraction", extent)
