"""Unit tests for the PCL classical point-cloud processing core."""

import numpy as np
import pytest

from perception.pointcloud.pcl_core import PclGeometricProcessor


def test_pcl_statistical_outlier_removal():
    """Verify SOR filter separates dense cluster points from isolated flying noise points."""
    np.random.seed(42)
    # Dense cluster (100 points within 1m radius)
    dense_pts = np.random.normal(loc=0.0, scale=0.2, size=(100, 3))
    # Isolated flying noise points
    outliers = np.array([
        [15.0, 20.0, 10.0],
        [-25.0, 18.0, -12.0],
        [30.0, -22.0, 15.0],
    ])
    pts = np.vstack([dense_pts, outliers])

    filtered = PclGeometricProcessor.statistical_outlier_removal(pts, k_neighbors=10, std_ratio=1.0)

    assert len(filtered.filtered_points) >= 95
    assert len(filtered.outlier_points) >= 3
    # Ensure extreme outliers are filtered out
    for out_pt in outliers:
        dists = np.linalg.norm(filtered.filtered_points - out_pt, axis=1)
        assert np.min(dists) > 5.0


def test_pcl_surface_normal_estimation():
    """Verify normal estimation on an XY horizontal plane yields [0, 0, 1] normals."""
    np.random.seed(42)
    # 2D grid on Z=0
    xs = np.linspace(-1, 1, 10)
    ys = np.linspace(-1, 1, 10)
    xx, yy = np.meshgrid(xs, ys)
    plane_pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(100)])

    normals = PclGeometricProcessor.estimate_surface_normals(plane_pts, k_neighbors=8)

    assert normals.shape == (100, 3)
    # For every point, normal should be parallel to Z axis (abs(nz) ≈ 1)
    for n in normals:
        assert abs(n[2]) > 0.95
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-4)


def test_pcl_dominant_plane_extraction_floor():
    """Verify RANSAC extracts a floor plane and correctly classifies it as FLOOR."""
    np.random.seed(42)
    xs = np.linspace(-2, 2, 20)
    ys = np.linspace(-2, 2, 20)
    xx, yy = np.meshgrid(xs, ys)
    floor_pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(400) + 0.05])
    # Add minor noise
    floor_pts += np.random.normal(0, 0.005, size=floor_pts.shape)

    # Add some random non-coplanar points
    random_pts = np.random.uniform(-3, 3, size=(50, 3))
    pts = np.vstack([floor_pts, random_pts])

    segment = PclGeometricProcessor.extract_dominant_plane(pts, distance_threshold=0.03)

    assert segment is not None
    assert segment.classification == "FLOOR"
    assert segment.inlier_rmse_m < 0.02
    assert abs(segment.plane_normal[2]) > 0.90
    assert len(segment.inlier_points) >= 380
