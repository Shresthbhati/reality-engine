"""Comprehensive unit tests for Lighting & Environmental Illuminance Engine (Block 36)."""

import math
import numpy as np
import pytest

from perception.materials.lighting_estimator import (
    C0,
    compute_sh_basis_9,
    estimate_and_attach_world_lighting,
    evaluate_sh_irradiance,
    fit_spherical_harmonics_lighting,
)
from world_ir.world_v1 import WorldIR


def test_sh_basis_evaluation():
    """Verify order 0 and order 1 spherical harmonic basis values at canonical directions."""
    # Test +Z axis: [0, 0, 1]
    norm_z = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    basis_z = compute_sh_basis_9(norm_z)[0]

    assert abs(basis_z[0] - C0) < 1e-6, "Order 0 m=0 must equal C0"
    assert abs(basis_z[1]) < 1e-6, "Y component should be 0 on +Z"
    assert basis_z[2] > 0.48, "Z component should be ~0.4886 on +Z"
    assert abs(basis_z[3]) < 1e-6, "X component should be 0 on +Z"


def test_fit_spherical_harmonics_synthetic_directional():
    """Verify SH fitting recovers known ambient + directional light source across multi-directional surfaces."""
    # 100 surface normals distributed over the full sphere (simulating walls, floors, ceiling, furniture)
    rng = np.random.RandomState(42)
    raw_normals = rng.randn(100, 3)
    normals = raw_normals / np.linalg.norm(raw_normals, axis=1, keepdims=True)

    # Ground truth light:
    # Ambient = 0.3
    # Directional = 0.7 from light vector [0.6, 0.0, 0.8] (normalized)
    gt_dir = np.array([0.6, 0.0, 0.8], dtype=np.float64)
    gt_dir = gt_dir / np.linalg.norm(gt_dir)

    ambient = 0.3
    directional = 0.7

    # Lambertian diffuse lighting: L = ambient + directional * max(0, n . d)
    dots = np.maximum(0.0, normals @ gt_dir)
    obs_grey = ambient + directional * dots
    obs_rgb = np.stack([obs_grey, obs_grey * 0.9, obs_grey * 0.8], axis=1)  # Warm directional light

    estimate = fit_spherical_harmonics_lighting(normals, obs_rgb)

    # Recovered dominant direction should closely match gt_dir
    est_dir = np.array(estimate.dominant_light_direction)
    cos_sim = float(np.dot(est_dir, gt_dir))
    angle_err_deg = math.degrees(math.acos(min(1.0, max(-1.0, cos_sim))))

    assert angle_err_deg < 5.0, f"Dominant light angle error {angle_err_deg:.2f} deg exceeded threshold"
    assert estimate.ambient_irradiance_rgb[0] > 0.15
    assert estimate.diagnostics.total_rmse < 0.08


def test_delighting_albedo_recovery():
    """Verify de-lighting recovers uniform intrinsic diffuse albedo across shaded surfaces."""
    # Two walls of identical white paint (true albedo = 0.8):
    # Wall A faces toward light, Wall B is oblique to light
    normals = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.707, 0.707, 0.0],
        [-0.707, 0.707, 0.0],
        [0.707, 0.0, 0.707],
        [0.0, 0.707, 0.707],
        [0.577, 0.577, 0.577],
    ], dtype=np.float64)

    # Constant intrinsic albedo
    true_albedo = 0.75

    # Known lighting: Ambient 0.4, Directional from [0, 0, 1] intensity 0.6
    light_dir = np.array([0.0, 0.0, 1.0])
    irradiance = 0.4 + 0.6 * np.maximum(0.0, normals @ light_dir)
    observed_linear = np.stack([true_albedo * irradiance] * 3, axis=1)

    # Fit lighting
    estimate = fit_spherical_harmonics_lighting(normals, observed_linear)

    # Delight the observed surface samples
    delighted_albedo = estimate.delight(normals, observed_linear, epsilon=0.01)

    # Shaded variation across different orientations was high before:
    pre_std = np.std(observed_linear[:, 0])
    post_std = np.std(delighted_albedo[:, 0])

    # De-lighting must substantially reduce standard deviation across orientations
    assert post_std < pre_std * 0.4, f"De-lighting failed to homogenize albedo: pre_std={pre_std:.3f}, post_std={post_std:.3f}"


def test_estimate_and_attach_world_lighting():
    """Verify estimate_and_attach_world_lighting integrates into WorldIR.environment."""
    world = WorldIR()
    assert "lighting" not in world.environment

    # Generate 12 sample points
    rng = np.random.RandomState(123)
    raw_normals = rng.randn(12, 3)
    normals = raw_normals / np.linalg.norm(raw_normals, axis=1, keepdims=True)
    obs_rgb = rng.uniform(0.2, 0.8, (12, 3))

    estimate = estimate_and_attach_world_lighting(world, normals, obs_rgb)

    assert "lighting" in world.environment
    light_env = world.environment["lighting"]
    assert "sh_coefficients" in light_env
    assert "dominant_light_direction" in light_env
    assert len(light_env["sh_coefficients"]) == 9
    assert estimate.diagnostics.sample_count == 12
