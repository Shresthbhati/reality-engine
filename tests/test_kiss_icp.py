"""Unit tests for KISS-ICP LiDAR odometry and registration extraction."""

import math
import numpy as np
import pytest

from reconstruction.geometry.lie_algebra import SE3, SO3
from reconstruction.registration.kiss_icp import KissIcpOdometry, VoxelHashMap


def test_voxel_hash_map_insertion_and_nn_query():
    """Verify points inserted into VoxelHashMap are retrievable via nearest neighbor."""
    vmap = VoxelHashMap(voxel_size=0.1)
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.05, 0.02, -0.01],
    ], dtype=np.float64)
    vmap.insert(pts)

    query = np.array([0.06, 0.01, 0.0], dtype=np.float64)
    closest, dist = vmap.find_nearest_neighbor(query, max_search_dist=0.2)

    assert closest is not None
    assert np.allclose(closest, pts[2])
    assert dist < 0.05


def test_kiss_icp_converges_to_known_rigid_transform():
    """Verify KISS-ICP registers source points to target transformed by known SE(3) pose."""
    rng = np.random.default_rng(42)
    # Generate 3D surface point cloud (e.g. wall/corner structure)
    num_pts = 400
    xs = rng.uniform(-3.0, 3.0, num_pts)
    ys = rng.uniform(-2.0, 2.0, num_pts)
    zs = np.sin(xs) * np.cos(ys) * 0.5
    source_pts = np.stack([xs, ys, zs], axis=-1)

    # Apply known inter-frame motion: 5 deg yaw + [0.15, -0.10, 0.05] translation
    rot_true = SO3.exp(np.array([0.0, 0.0, math.radians(5.0)]))
    trans_true = np.array([0.15, -0.10, 0.05], dtype=np.float64)
    pose_true = SE3(rot_true, trans_true)

    target_pts = pose_true.transform_points(source_pts)

    icp = KissIcpOdometry(
        voxel_size=0.1,
        initial_threshold=1.5,
        min_threshold=0.05,
        max_iterations=30,
    )
    result = icp.register(source_pts, target_pts)

    assert result.converged is True
    assert result.residual_rmse_m < 0.03
    assert result.inlier_ratio > 0.80

    # Verify recovered translation and rotation match ground truth
    assert np.allclose(result.pose.translation, trans_true, atol=0.02)
    rot_diff = result.pose.rotation.inverse() @ rot_true
    assert np.linalg.norm(rot_diff.log()) < 0.02
