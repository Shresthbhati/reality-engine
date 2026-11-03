"""Unit tests for hydrological terrain analysis (WhiteboxTools / GDAL algorithms)."""

import numpy as np
import pytest

from geo.terrain import TerrainElevationGrid
from geo.terrain_analysis import (
    analyze_terrain_hydrology,
    compute_d8_flow_direction,
    compute_flow_accumulation,
    compute_multidirectional_hillshade,
)


def test_d8_flow_direction_on_inclined_plane():
    """Verify flow direction points East (code 1) down an eastward inclined plane."""
    # Rows 10, Cols 10: Elevation drops along columns (x increases => elevation decreases)
    xs = np.linspace(100.0, 50.0, 10)
    elev = np.tile(xs, (10, 1))

    flow_dir, slope_pct = compute_d8_flow_direction(elev, cell_size=1.0)

    # For cols 0..8, steepest descent is directly East (code 1)
    assert np.all(flow_dir[:, :8] == 1)
    # Slope should be approximately 50m / 9m * 100% ~ 555%
    assert np.all(slope_pct[:, :8] > 500.0)


def test_flow_accumulation_concentrates_downstream():
    """Verify that flow accumulation grows strictly monotonically downstream."""
    xs = np.linspace(10.0, 0.0, 8)
    elev = np.tile(xs, (5, 1))
    flow_dir, _ = compute_d8_flow_direction(elev, cell_size=1.0)
    accumulation = compute_flow_accumulation(flow_dir)

    # In col 0, accumulation is 1 (headwater)
    assert np.all(accumulation[:, 0] == 1)
    # Downstream columns must have strictly greater accumulation than upstream
    for c in range(1, 7):
        assert np.all(accumulation[:, c] > accumulation[:, c - 1])
    # The outlet column has maximum accumulation
    assert np.all(accumulation[:, 6] == 7)


def test_hillshade_generation():
    """Verify hillshade values are within valid 8-bit image range [0, 255]."""
    elev = np.random.uniform(10.0, 50.0, size=(20, 20))
    hillshade = compute_multidirectional_hillshade(elev, cell_size=1.0)

    assert hillshade.shape == (20, 20)
    assert hillshade.dtype == np.uint8
    assert np.all(hillshade >= 0)
    assert np.all(hillshade <= 255)


def test_complete_terrain_watershed_analysis():
    """Verify analyze_terrain_hydrology pipeline integrates into TerrainElevationGrid."""
    grid = TerrainElevationGrid(
        elevation=np.outer(np.linspace(10.0, 0.0, 15), np.linspace(10.0, 0.0, 15)),
        bounds=(0.0, 0.0, 15.0, 15.0),
        resolution_m=1.0,
    )
    res = analyze_terrain_hydrology(grid, stream_threshold=5)

    assert res.flow_direction_d8.shape == (15, 15)
    assert res.flow_accumulation.shape == (15, 15)
    assert res.stream_channel_mask.shape == (15, 15)
    assert np.any(res.stream_channel_mask)  # Bottom corner should form stream channels
    assert res.drainage_basins_count >= 1
