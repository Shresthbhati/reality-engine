"""WhiteboxTools & GDAL hydrological terrain analysis and geoprocessing.

Implements deterministic D8 flow routing, upstream drainage area accumulation,
Horn's slope/aspect raster kernels, and multi-directional analytical hillshade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from geo.terrain import TerrainElevationGrid


# Standard D8 direction encodings (ESRI / Whitebox convention)
#  32   64  128  (NW,  N, NE)
#  16    0    1  ( W,  .,  E)
#   8    4    2  (SW,  S, SE)
D8_CODES = np.array([
    [32, 64, 128],
    [16,  0,   1],
    [ 8,  4,   2],
], dtype=np.uint8)

# Neighbor row/col offsets for 8 directions: [dr, dc]
# Order: E, SE, S, SW, W, NW, N, NE
D8_OFFSETS = [
    (0, 1),    # 1: East
    (1, 1),    # 2: SE
    (1, 0),    # 4: South
    (1, -1),   # 8: SW
    (0, -1),   # 16: West
    (-1, -1),  # 32: NW
    (-1, 0),   # 64: North
    (-1, 1),   # 128: NE
]

D8_CODE_TO_OFFSET = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}


@dataclass(frozen=True)
class TerrainHydrologyResult:
    """Hydrological and geomorphic terrain analysis results."""
    flow_direction_d8: np.ndarray        # 2D uint8 grid with D8 codes
    flow_accumulation: np.ndarray       # 2D int32 grid of contributing cell count
    slope_percent: np.ndarray           # 2D float32 grid of maximum gradient percentage
    hillshade: np.ndarray               # 2D uint8 shaded relief [0, 255]
    stream_channel_mask: np.ndarray     # 2D boolean mask where accumulation >= threshold
    drainage_basins_count: int


def compute_d8_flow_direction(
    elevation: np.ndarray,
    cell_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate D8 flow direction and steepest slope percentage for each cell.
    
    Returns:
        (flow_dir, slope_pct): flow_dir has D8 codes, slope_pct has gradient drop / distance * 100.
    """
    rows, cols = elevation.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)
    slope_pct = np.zeros((rows, cols), dtype=np.float32)

    diag_dist = cell_size * math.sqrt(2.0)

    for r in range(rows):
        for c in range(cols):
            curr_elev = elevation[r, c]
            max_drop_rate = 0.0
            best_code = 0

            for code, (dr, dc) in D8_CODE_TO_OFFSET.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_elev = elevation[nr, nc]
                    dist = diag_dist if (dr != 0 and dc != 0) else cell_size
                    drop = curr_elev - neighbor_elev
                    drop_rate = drop / dist

                    if drop_rate > max_drop_rate:
                        max_drop_rate = drop_rate
                        best_code = code

            flow_dir[r, c] = best_code
            slope_pct[r, c] = max(0.0, max_drop_rate) * 100.0

    return flow_dir, slope_pct


def compute_flow_accumulation(flow_direction: np.ndarray) -> np.ndarray:
    """Compute contributing upstream drainage area (number of draining cells) using topological order."""
    rows, cols = flow_direction.shape
    accumulation = np.ones((rows, cols), dtype=np.int32)  # Each cell drains itself (value 1)

    # Compute in-degree (number of neighbors draining into each cell)
    in_degree = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            code = int(flow_direction[r, c])
            if code in D8_CODE_TO_OFFSET:
                dr, dc = D8_CODE_TO_OFFSET[code]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    in_degree[nr, nc] += 1

    # Queue of cells with zero in-degree (headwaters / ridge cells)
    queue = []
    for r in range(rows):
        for c in range(cols):
            if in_degree[r, c] == 0:
                queue.append((r, c))

    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1

        code = int(flow_direction[r, c])
        if code in D8_CODE_TO_OFFSET:
            dr, dc = D8_CODE_TO_OFFSET[code]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                accumulation[nr, nc] += accumulation[r, c]
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue.append((nr, nc))

    return accumulation


def compute_multidirectional_hillshade(
    elevation: np.ndarray,
    cell_size: float = 1.0,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> np.ndarray:
    """Generate shaded relief representation using Horn's finite difference algorithm."""
    rows, cols = elevation.shape
    hillshade = np.full((rows, cols), 180, dtype=np.uint8)

    zenith_rad = math.radians(90.0 - altitude_deg)
    azimuth_math_rad = math.radians(360.0 - azimuth_deg + 90.0)

    # Pad elevation for border handling
    padded = np.pad(elevation, 1, mode="edge")

    # Horn's 3x3 convolution kernels for dz/dx and dz/dy
    dz_dx = (
        (padded[0:rows, 2:cols+2] + 2.0 * padded[1:rows+1, 2:cols+2] + padded[2:rows+2, 2:cols+2]) -
        (padded[0:rows, 0:cols] + 2.0 * padded[1:rows+1, 0:cols] + padded[2:rows+2, 0:cols])
    ) / (8.0 * cell_size)

    dz_dy = (
        (padded[2:rows+2, 0:cols] + 2.0 * padded[2:rows+2, 1:cols+1] + padded[2:rows+2, 2:cols+2]) -
        (padded[0:rows, 0:cols] + 2.0 * padded[0:rows, 1:cols+1] + padded[0:rows, 2:cols+2])
    ) / (8.0 * cell_size)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect_rad = np.arctan2(dz_dy, -dz_dx)

    shaded = 255.0 * (
        (math.cos(zenith_rad) * np.cos(slope_rad)) +
        (math.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_math_rad - aspect_rad))
    )
    shaded_clamped = np.clip(shaded, 0.0, 255.0).astype(np.uint8)
    return shaded_clamped


def analyze_terrain_hydrology(
    grid: TerrainElevationGrid,
    stream_threshold: int = 10,
) -> TerrainHydrologyResult:
    """Execute complete hydrological analysis: D8 flow routing, accumulation, slope, and hillshade."""
    flow_dir, slope_pct = compute_d8_flow_direction(grid.elevation, cell_size=grid.resolution_m)
    accumulation = compute_flow_accumulation(flow_dir)
    hillshade = compute_multidirectional_hillshade(grid.elevation, cell_size=grid.resolution_m)

    stream_mask = accumulation >= stream_threshold
    # Count sinks / pit cells where flow_dir == 0
    sinks_count = int(np.sum(flow_dir == 0))

    return TerrainHydrologyResult(
        flow_direction_d8=flow_dir,
        flow_accumulation=accumulation,
        slope_percent=slope_pct,
        hillshade=hillshade,
        stream_channel_mask=stream_mask,
        drainage_basins_count=max(1, sinks_count),
    )
