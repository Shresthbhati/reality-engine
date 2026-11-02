"""Unit tests for MVS dense point cloud ops and multi-view fusion (Master Development Plan Tasks 4.4, 4.5, 4.8)."""

import math
import numpy as np
import pytest

from perception.depth.interface import DepthMap
from reconstruction.calibration.camera import CameraIntrinsics, PinholeCamera
from reconstruction.mvs.dense_cloud import DenseCloudResult, MultiViewDenseCloudBuilder
from reconstruction.mvs.point_cloud_ops import (
    estimate_point_normals,
    radius_outlier_removal,
    statistical_outlier_removal,
    voxel_downsample,
)


def test_voxel_downsample_aggregation():
    # 4 points inside a 0.1m voxel at [0, 0, 0]
    pts = np.array([
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.02],
        [0.03, 0.03, 0.03],
        [0.04, 0.04, 0.04],
        # 1 point in a separate voxel at [1.0, 1.0, 1.0]
        [1.05, 1.05, 1.05],
    ])
    cols = np.array([
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [200, 200, 200],
    ], dtype=np.uint8)

    down_pts, down_cols, _ = voxel_downsample(pts, voxel_size_m=0.1, colors=cols)

    assert len(down_pts) == 2
    assert len(down_cols) == 2
    # Centroid of first 4 points
    assert np.allclose(down_pts[0], [0.025, 0.025, 0.025])
    assert np.allclose(down_cols[0], [100, 100, 100])
    # Second voxel
    assert np.allclose(down_pts[1], [1.05, 1.05, 1.05])
    assert np.allclose(down_cols[1], [200, 200, 200])


def test_statistical_outlier_removal():
    rng = np.random.default_rng(42)
    # Dense cluster of 100 points around origin (std = 0.05)
    cluster = rng.normal(0, 0.05, (100, 3))
    # 3 distant outlier noise points
    outliers = np.array([
        [5.0, 5.0, 5.0],
        [-4.0, 3.0, 2.0],
        [6.0, -5.0, 4.0],
    ])
    pts = np.vstack([cluster, outliers])

    inliers, _, _, mask = statistical_outlier_removal(pts, k_neighbors=15, std_ratio=2.0)

    assert len(inliers) < len(pts)
    # All 3 distant outliers must be filtered
    assert not mask[-1]
    assert not mask[-2]
    assert not mask[-3]
    # The vast majority of the dense cluster must be retained
    assert np.sum(mask[:100]) >= 95


def test_radius_outlier_removal():
    rng = np.random.default_rng(42)
    # Tight cluster of 20 points
    cluster = rng.uniform(-0.1, 0.1, (20, 3))
    # 1 isolated point far away
    isolated = np.array([[10.0, 10.0, 10.0]])
    pts = np.vstack([cluster, isolated])

    inliers, _, _, mask = radius_outlier_removal(pts, radius_m=0.3, min_neighbors=5)

    assert len(inliers) == 20
    assert not mask[-1]  # isolated point rejected
    assert np.all(mask[:20])  # cluster accepted


def test_estimate_point_normals_flat_plane():
    # Horizontal grid in XY plane at z = 2.0
    x = np.linspace(-1.0, 1.0, 10)
    y = np.linspace(-1.0, 1.0, 10)
    xx, yy = np.meshgrid(x, y)
    pts = np.stack([xx.flatten(), yy.flatten(), np.full(100, 2.0)], axis=1)

    # Viewpoint from above at (0, 0, 5)
    normals = estimate_point_normals(pts, k_neighbors=8, view_point=(0.0, 0.0, 5.0))

    assert normals.shape == (100, 3)
    for n in normals:
        # Unit length
        assert abs(np.linalg.norm(n) - 1.0) < 1e-4
        # Normal must point upward along +z toward the camera at z=5
        assert n[2] > 0.95
        assert abs(n[0]) < 0.1
        assert abs(n[1]) < 0.1


def test_multiview_dense_cloud_builder_fusion():
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=120.0, width=320, height=240)

    from reconstruction.calibration.camera import CameraExtrinsics
    from engine.math.math3 import Vec3, Quat

    ext1 = CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0))
    cam1 = PinholeCamera(intrinsics=intrinsics, extrinsics=ext1)

    ext2 = CameraExtrinsics(position=Vec3(0.2, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0))
    cam2 = PinholeCamera(intrinsics=intrinsics, extrinsics=ext2)

    # A flat wall at z = 3.0m in front of cameras
    # Depth map for cam1: all pixels have depth ~ 3.0 meters
    depth1_vals = [[3.0 for _ in range(320)] for _ in range(240)]
    depth1 = DepthMap(evidence_id="cam1", width=320, height=240, values=depth1_vals, unit="meters")

    # Depth map for cam2: same flat wall at z = 3.0 meters
    depth2_vals = [[3.0 for _ in range(320)] for _ in range(240)]
    depth2 = DepthMap(evidence_id="cam2", width=320, height=240, values=depth2_vals, unit="meters")

    # Synthetic RGB images
    img1 = np.ones((240, 320, 3), dtype=np.uint8) * 150
    img2 = np.ones((240, 320, 3), dtype=np.uint8) * 160

    builder = MultiViewDenseCloudBuilder(
        stride=16,
        consistency_check=True,
        depth_tolerance=0.08,
        min_consistent_views=1,
        voxel_size_m=0.05,
    )

    result = builder.fuse(
        cameras={"cam1": cam1, "cam2": cam2},
        depth_maps={"cam1": depth1, "cam2": depth2},
        images_rgb={"cam1": img1, "cam2": img2},
    )

    assert isinstance(result, DenseCloudResult)
    assert result.point_count > 0
    assert result.view_count == 2
    assert result.points.shape[1] == 3
    assert result.colors is not None
    assert result.normals is not None
    # All points should be around z = 3.0m
    assert abs(np.median(result.points[:, 2]) - 3.0) < 0.1
    assert result.diagnostics["status"] == "success"
