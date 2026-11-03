"""Unit tests for the self-contained TEASER++ registration solver."""

import math
import numpy as np
import pytest

from reconstruction.scale.teaser_solver import TeaserRegistrationSolver


def test_teaser_exact_rigid_registration():
    """Verify exact recovery of rotation and translation on noise-free points."""
    np.random.seed(42)
    source = np.random.uniform(-5.0, 5.0, size=(20, 3))

    # Known rotation (30 degrees around Z axis) and translation
    theta = math.radians(30.0)
    R_true = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t_true = np.array([1.5, -2.0, 0.75])

    target = (source @ R_true.T) + t_true

    solver = TeaserRegistrationSolver(noise_bound=0.01)
    result = solver.solve(source, target, estimate_scale=False)

    assert result.residual_rmse_m < 1e-4
    assert np.allclose(result.rotation, R_true, atol=1e-3)
    assert np.allclose(result.translation, t_true, atol=1e-3)
    assert result.inlier_count == 20


def test_teaser_with_heavy_outliers():
    """Verify TEASER++ rejects 75% correspondence outliers and recovers true transform."""
    np.random.seed(123)
    N_inliers = 15
    N_outliers = 45  # 75% outlier ratio
    N_total = N_inliers + N_outliers

    source_inliers = np.random.uniform(-3.0, 3.0, size=(N_inliers, 3))
    # Known 90 deg rotation around Y
    R_true = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    t_true = np.array([0.5, 1.2, -0.4])

    target_inliers = (source_inliers @ R_true.T) + t_true + np.random.normal(0, 0.005, size=(N_inliers, 3))

    # Outliers: pure uniform noise
    source_outliers = np.random.uniform(-10.0, 10.0, size=(N_outliers, 3))
    target_outliers = np.random.uniform(-10.0, 10.0, size=(N_outliers, 3))

    source = np.vstack([source_inliers, source_outliers])
    target = np.vstack([target_inliers, target_outliers])

    solver = TeaserRegistrationSolver(noise_bound=0.05)
    result = solver.solve(source, target, estimate_scale=False)

    assert result.inlier_count >= N_inliers
    assert result.residual_rmse_m < 0.05
    assert np.allclose(result.rotation, R_true, atol=0.02)
    assert np.allclose(result.translation, t_true, atol=0.05)
