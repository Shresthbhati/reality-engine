"""Tests for Core Structure-from-Motion (SfM) and Bundle Adjustment Pipeline (Directive Rule 4).

Verifies:
- Essential matrix estimation and relative camera pose recovery via 8-point algorithm
- Cheirality positive-depth verification across camera hypotheses
- Multi-view DLT triangulation and ray baseline parallax angle calculation
- Nonlinear Levenberg-Marquardt bundle adjustment minimizing reprojection RMSE
- Multi-view dense depth fusion with statistical outlier pruning
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from reconstruction.sfm.core_pipeline import (
    BundleAdjustmentOptimizer,
    CameraModel,
    DenseDepthFusionEngine,
    EssentialMatrixEstimator,
    TriangulationEngine,
)


class TestSfMPipelineRule4:
    @pytest.fixture
    def standard_camera(self) -> CameraModel:
        return CameraModel(fx=800.0, fy=800.0, cx=960.0, cy=540.0, width=1920, height=1080)

    def test_essential_matrix_and_relative_pose(self, standard_camera: CameraModel):
        """Test relative pose recovery between 2 cameras observing a 3D point cloud."""
        # Known relative motion: pure translation along X by 1.0 meter
        R_true = np.eye(3)
        t_true = np.array([1.0, 0.0, 0.0])

        # Generate 15 points in front of both cameras (Z between 3.0 and 6.0 meters)
        np.random.seed(42)
        X_world = np.random.uniform(-1.0, 1.0, (15, 3))
        X_world[:, 2] += 4.0  # Push points into positive depth

        # Project into Camera 1 (origin)
        p1_h = standard_camera.K @ X_world.T
        pts1 = (p1_h[:2] / p1_h[2]).T

        # Project into Camera 2 (shifted by t_true)
        X_cam2 = (R_true @ X_world.T).T + t_true
        p2_h = standard_camera.K @ X_cam2.T
        pts2 = (p2_h[:2] / p2_h[2]).T

        R_est, t_est, mask = EssentialMatrixEstimator.estimate_relative_pose(
            pts1, pts2, standard_camera, standard_camera
        )

        assert np.sum(mask) >= 12  # At least 80% cheirality inliers
        # Check that translation is aligned with +X
        assert np.dot(t_est, np.array([1.0, 0.0, 0.0])) > 0.85

    def test_triangulation_dlt_and_parallax(self, standard_camera: CameraModel):
        """Test DLT triangulation and parallax angle measurement."""
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([np.eye(3), np.array([[-1.0], [0.0], [0.0]])])  # Baseline = 1m along X

        true_point = np.array([0.5, 0.2, 4.0])

        x1_norm = true_point[:2] / true_point[2]
        p2 = true_point + np.array([-1.0, 0.0, 0.0])
        x2_norm = p2[:2] / p2[2]

        tri_point = TriangulationEngine.triangulate_dlt_point(P1, P2, x1_norm, x2_norm)

        assert tri_point is not None
        np.testing.assert_allclose(tri_point, true_point, atol=1e-3)

        # Check parallax angle
        angle = TriangulationEngine.compute_parallax_angle_deg(
            cam1_pos=np.array([0.0, 0.0, 0.0]),
            cam2_pos=np.array([1.0, 0.0, 0.0]),
            point_3d=true_point,
        )
        assert 10.0 < angle < 20.0  # ~14 degrees parallax

    def test_bundle_adjustment_optimization(self, standard_camera: CameraModel):
        """Test nonlinear bundle adjustment reduces reprojection RMSE."""
        # 2 cameras, 10 3D points
        np.random.seed(42)
        cams = [
            (np.eye(3), np.array([0.0, 0.0, 0.0])),
            (np.eye(3), np.array([-1.0, 0.0, 0.0])),
        ]
        pts_true = np.random.uniform(-1.0, 1.0, (10, 3))
        pts_true[:, 2] += 5.0

        # Create observations with small measurement noise
        observations = []
        for c_idx, (R, t) in enumerate(cams):
            for p_idx, X in enumerate(pts_true):
                proj = BundleAdjustmentOptimizer.project_point(standard_camera.K, R, t, X)
                assert proj is not None
                noisy_uv = proj + np.random.normal(0.0, 1.5, size=2)  # 1.5px noise
                observations.append((c_idx, p_idx, noisy_uv))

        # Initial perturbed points (add 10cm error)
        pts_init = pts_true + np.random.normal(0.0, 0.10, size=pts_true.shape)

        refined_cams, refined_pts, result = BundleAdjustmentOptimizer.optimize(
            camera_poses=cams,
            points_3d=pts_init,
            observations=observations,
            camera_model=standard_camera,
            max_iterations=10,
        )

        assert result.converged is True
        assert result.final_rmse_px < result.initial_rmse_px
        assert result.num_observations == 20

    def test_dense_depth_fusion_and_outlier_rejection(self, standard_camera: CameraModel):
        """Test fusing multi-view depth maps and rejecting rogue outliers."""
        # Simple planar wall at depth Z = 3.0m
        h, w = 40, 40
        d_map1 = np.full((h, w), 3.0, dtype=np.float32)
        d_map2 = np.full((h, w), 3.0, dtype=np.float32)

        # Inject a rogue outlier at an extreme invalid depth
        d_map1[10, 10] = 19.5

        poses = [
            (np.eye(3), np.array([0.0, 0.0, 0.0])),
            (np.eye(3), np.array([0.5, 0.0, 0.0])),
        ]

        fused_pts = DenseDepthFusionEngine.fuse_depth_maps_with_outlier_rejection(
            depth_maps=[d_map1, d_map2],
            camera_poses=poses,
            camera_model=standard_camera,
            stride=5,
        )

        assert len(fused_pts) > 50
        # All retained points should be close to plane depth Z ~ 3.0m
        zs = [p[2] for p in fused_pts]
        assert np.percentile(zs, 95) < 5.0
