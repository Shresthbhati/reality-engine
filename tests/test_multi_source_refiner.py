"""Tests for Multi-Source Geometric Refinement Engine.

Validates:
- Cross-checking 3D sparse points against multi-view depth maps to prune floaters
- Recovering global metric scale from depth constraints
- Rectifying wall plane normals relative to floor planes
- Estimating physical wall thickness from parallel plane pairs
"""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction.refinement import (
    MultiSourceGeometryRefiner,
    RefinementReport,
)


def test_point_to_camera_projection():
    # Camera at [0, 0, 0] looking along +Z
    extrinsic = np.eye(4)
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}

    # Point at [1.0, 0.5, 2.0]
    u, v, zc = MultiSourceGeometryRefiner.project_point_to_camera(
        np.array([1.0, 0.5, 2.0]), extrinsic, intrinsics
    )
    assert zc == 2.0
    # u = 500 * (1.0 / 2.0) + 320 = 250 + 320 = 570
    # v = 500 * (0.5 / 2.0) + 240 = 125 + 240 = 365
    assert abs(u - 570.0) < 1e-4
    assert abs(v - 365.0) < 1e-4


def test_filter_depth_consistency_pruning():
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
    extrinsic = np.eye(4)
    poses = {"cam_0": extrinsic}

    # Depth map constant 2.0 meters everywhere
    depth_map = np.ones((100, 100), dtype=np.float32) * 2.0
    depth_maps = {"cam_0": depth_map}

    points = [
        {"point_id": 1, "xyz": [0.0, 0.0, 2.0], "error": 0.3},       # Matches depth (residual = 0.0)
        {"point_id": 2, "xyz": [0.1, 0.1, 2.05], "error": 0.4},      # Small residual (0.05m <= 0.15m)
        {"point_id": 3, "xyz": [0.0, 0.0, 8.5], "error": 1.2},       # Wild floater (residual = 6.5m > 0.15m)
    ]

    valid_pts, depth_rmse, pruned_cnt = MultiSourceGeometryRefiner.filter_depth_consistency(
        points, depth_maps, poses, intrinsics, max_residual_m=0.15
    )
    assert pruned_cnt == 1
    assert len(valid_pts) == 2
    assert [p["point_id"] for p in valid_pts] == [1, 2]
    assert depth_rmse < 0.10


def test_align_metric_scale_from_depth():
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
    extrinsic = np.eye(4)
    poses = {"cam_0": extrinsic}

    # True metric depth is 3.0x larger than unscaled sparse SfM points
    # Suppose sparse points have z = 1.0, 1.2, 1.5
    # Measured depth maps have z = 3.0, 3.6, 4.5
    depth_map = np.ones((100, 100), dtype=np.float32) * 3.0
    depth_maps = {"cam_0": depth_map}

    points = [
        {"xyz": [0.0, 0.0, 1.0], "error": 0.2} for _ in range(12)
    ]

    scale = MultiSourceGeometryRefiner.align_metric_scale_from_depth(
        points, depth_maps, poses, intrinsics
    )
    # scale s should be ≈ 3.0
    assert abs(scale - 3.0) < 1e-4


def test_rectify_planes_and_wall_thickness():
    # Floor along Y = 0
    # Two parallel walls along X = 1.0 and X = 1.20 (20 cm wall thickness)
    planes = [
        {"type": "floor", "normal": [0.0, 1.0, 0.0], "d": 0.0},
        {"type": "wall", "normal": [1.0, 0.03, 0.0], "d": 1.0},
        {"type": "wall", "normal": [-1.0, -0.02, 0.0], "d": 1.20},
    ]

    rectified, thicknesses = MultiSourceGeometryRefiner.rectify_planes_and_estimate_thickness(planes)
    assert len(rectified) == 3
    # Walls should be rectified perpendicular to floor
    for p in rectified:
        if p["type"] == "wall":
            assert abs(p["normal"][1]) < 1e-5  # Y component eliminated

    assert len(thicknesses) == 1
    assert abs(thicknesses[0] - 0.20) < 1e-4


def test_execute_refinement_cycle_integration():
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
    extrinsic = np.eye(4)
    poses = {"cam_0": extrinsic}
    depth_map = np.ones((100, 100), dtype=np.float32) * 2.0
    depth_maps = {"cam_0": depth_map}

    points = [
        {"point_id": 1, "xyz": [0.0, 0.0, 2.0], "error": 0.2},
        {"point_id": 2, "xyz": [0.0, 0.0, 12.0], "error": 2.5},  # Floater
    ]
    planes = [
        {"type": "floor", "normal": [0.0, 1.0, 0.0], "d": 0.0},
        {"type": "wall", "normal": [1.0, 0.0, 0.0], "d": 2.0},
        {"type": "wall", "normal": [-1.0, 0.0, 0.0], "d": 2.15},  # 15 cm thickness
    ]

    filt_pts, rect_planes, report = MultiSourceGeometryRefiner.execute_refinement_cycle(
        points, depth_maps, poses, intrinsics, planes
    )
    assert isinstance(report, RefinementReport)
    assert report.initial_points_count == 2
    assert report.pruned_outliers_count == 1
    assert len(filt_pts) == 1
    assert len(report.wall_thickness_estimates_m) == 1
    assert report.wall_thickness_estimates_m[0] == pytest.approx(0.15, abs=1e-4)
