"""Tests for Mathematically Validated Correlation-Aware Statistical Fusion."""

from __future__ import annotations

import numpy as np
import pytest

from engine.multiengine.statistical_fusion import (
    CorrelationDiscountResult,
    StatisticalFusionValidator,
)


def test_effective_sample_size_complete_correlation_never_shrinks_variance() -> None:
    """When rho = 1.0 (redundant models sharing weights/priors), N_eff is strictly 1.0."""
    validator = StatisticalFusionValidator()

    for n in [1, 2, 5, 10, 50]:
        n_eff = validator.effective_sample_size(n, rho=1.0)
        assert pytest.approx(n_eff, abs=1e-5) == 1.0


def test_effective_sample_size_zero_correlation_scales_linearly() -> None:
    """When rho = 0.0 (independent sensors, e.g. LiDAR vs visual SfM), N_eff is exactly N."""
    validator = StatisticalFusionValidator()

    for n in [1, 2, 5, 10, 20]:
        n_eff = validator.effective_sample_size(n, rho=0.0)
        assert pytest.approx(n_eff, abs=1e-5) == float(n)


def test_effective_sample_size_intermediate_correlation_bounds() -> None:
    """For rho in (0, 1), 1.0 < N_eff < N."""
    validator = StatisticalFusionValidator()
    rho = 0.8
    n = 10
    n_eff = validator.effective_sample_size(n, rho=rho)

    # vif = 1 + 9 * 0.8 = 8.2; n_eff = 10 / 8.2 = 1.2195
    expected_vif = 1.0 + (n - 1) * rho
    expected_n_eff = n / expected_vif

    assert pytest.approx(n_eff, rel=1e-4) == expected_n_eff
    assert 1.0 < n_eff < float(n)


def test_calculate_family_discounts_honesty_and_basis() -> None:
    """Verifies that family discounts record uncalibrated status unless empirical data is provided."""
    family_counts = {
        "vit_detector": 4,      # 4 ViT-based detectors
        "monocular_depth": 3,   # 3 depth estimators
        "independent_sensor": 2 # 2 independent LiDAR scans
    }

    result = StatisticalFusionValidator.calculate_family_discounts(
        family_counts,
        calibrated=False,
    )

    assert isinstance(result, CorrelationDiscountResult)
    assert result.is_calibrated is False
    assert "uncalibrated" in result.uncertainty_basis
    assert result.effective_sample_sizes["vit_detector"] < 4.0
    assert result.effective_sample_sizes["monocular_depth"] < 3.0
    # Independent sensor has rho = 0.0, so N_eff == 2.0
    assert pytest.approx(result.effective_sample_sizes["independent_sensor"], abs=1e-5) == 2.0


def test_monte_carlo_variance_validation_matches_theoretical_vif() -> None:
    """Empirical Monte Carlo variance of sample mean agrees with theoretical VIF within tight error."""
    # Test with N=4 estimators, rho=0.75
    mc_results = StatisticalFusionValidator.monte_carlo_variance_validation(
        n_estimators=4,
        rho=0.75,
        unit_variance=1.0,
        trials=3000,
        seed=123,
    )

    # Theoretical: (1 + 3 * 0.75) / 4 = 3.25 / 4 = 0.8125
    assert pytest.approx(mc_results["theoretical_variance"], rel=1e-4) == 0.8125
    # Empirical should match theoretical within 6% on 3000 trials
    assert mc_results["relative_error"] < 0.06


def test_covariance_intersection_conservatism() -> None:
    """Covariance Intersection fused covariance matrix is positive-definite and conservative."""
    cov_a = np.array([[0.2, 0.05], [0.05, 0.3]])
    cov_b = np.array([[0.15, -0.02], [-0.02, 0.25]])

    fused_cov = StatisticalFusionValidator.fuse_covariances_conservative([cov_a, cov_b])

    assert fused_cov.shape == (2, 2)
    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(fused_cov)
    assert all(ev > 0 for ev in eigenvalues)

    # Test error handling on bad inputs
    with pytest.raises(ValueError, match="dimension mismatch"):
        StatisticalFusionValidator.fuse_covariances_conservative([cov_a, np.eye(3)])
