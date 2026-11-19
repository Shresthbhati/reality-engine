r"""Mathematically Validated Correlation-Aware Statistical Fusion.

Provides rigorous information-theoretic and estimation-theoretic tools for
fusing observations from multiple models sharing intra-family or cross-family
correlations.

Mathematical Principles:
1. Effective Sample Size under Correlation:
   For $N$ estimators with pairwise correlation $\rho \in [0, 1]$ and common variance $\sigma^2$:
   $$\text{Var}(\bar{X}) = \sigma^2 \cdot \frac{1 + (N - 1)\rho}{N}$$
   The effective sample size is:
   $$N_{\text{eff}} = \frac{N}{1 + (N - 1)\rho}$$
   - When $\rho = 1.0$ (totally redundant models sharing identical priors/weights):
     $N_{\text{eff}} = 1.0$, $\text{Var}(\bar{X}) = \sigma^2$ (variance cannot shrink).
   - When $\rho = 0.0$ (independent physical sensors, e.g. LiDAR vs visual SfM):
     $N_{\text{eff}} = N$, $\text{Var}(\bar{X}) = \sigma^2 / N$ (standard error shrinks as $1/\sqrt{N}$).
   - The heuristic weight $1/\sqrt{N}$ is NOT a calibrated uncertainty without validation.
     It implicitly assumes $\rho = 1 / (\sqrt{N} + 1)$, which varies arbitrarily with $N$.

2. Covariance Intersection (CI):
   When the cross-correlation between two estimators $A$ and $B$ is unknown, the fused covariance
   is bounded conservatively by:
   $$P^{-1} = \omega P_A^{-1} + (1 - \omega) P_B^{-1}, \quad \omega \in [0, 1]$$
   guaranteeing that $P \ge \text{Cov}(\hat{x} - x_{\text{true}})$ for any true correlation.

3. Calibrated Confidence Requirement:
   A confidence value $c \in [0, 1]$ is uncalibrated unless derived from verified residual distributions,
   calibrated covariance matrices, or empirical isotonic/Platt scaling against a physical validation set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationDiscountResult:
    """Result of correlation-aware discounting across model families."""

    family_counts: dict[str, int]
    effective_sample_sizes: dict[str, float]
    family_weights: dict[str, float]
    assumed_correlations: dict[str, float]
    total_effective_sample_size: float
    total_weight: float
    is_calibrated: bool
    method: str
    uncertainty_basis: str


class StatisticalFusionValidator:
    """Validates and computes correlation-aware pooling weights and covariance bounds."""

    # Default conservative intra-family correlations based on architecture & training overlap
    DEFAULT_FAMILY_CORRELATIONS: dict[str, float] = {
        "vit_detector": 0.85,       # Grounding DINO, OWL-ViT, Florence-2 share ViT inductive bias
        "monocular_depth": 0.80,    # Depth Anything, MiDaS share scale ambiguity & single-view priors
        "sam_segmentation": 0.90,   # SAM, SAM 2 share prompt-conditioned mask heads
        "feature_tracker": 0.75,    # DINOv2, LightGlue share visual feature embedding spaces
        "sfm_geometry": 0.70,       # COLMAP, MapAnything share epipolar geometry constraints
        "dust3r_geometry": 0.85,    # DUSt3R, MASt3R share pointmap regression backbones
        "independent_sensor": 0.0,  # Physical LiDAR / Time-of-Flight vs optical camera
        "generic": 0.80,            # Conservative default when family correlation is unmeasured
    }

    @classmethod
    def effective_sample_size(cls, n: int, rho: float) -> float:
        """Computes effective sample size $N_{eff} = N / (1 + (N - 1) * rho)$.

        Args:
            n: Number of observations from the model family (n >= 1).
            rho: Pairwise correlation within the family (0.0 <= rho <= 1.0).

        Returns:
            Effective independent sample count (1.0 <= N_eff <= n).
        """
        if n <= 0:
            return 0.0
        if n == 1:
            return 1.0
        rho_clamped = max(0.0, min(1.0, float(rho)))
        vif = 1.0 + (n - 1) * rho_clamped  # Variance Inflation Factor
        return float(n / vif)

    @classmethod
    def calculate_family_discounts(
        cls,
        family_counts: dict[str, int],
        correlations: dict[str, float] | None = None,
        calibrated: bool = False,
        method: str = "sqrt_heuristic",
    ) -> CorrelationDiscountResult:
        """Calculates mathematically validated discounting weights across engine families.

        Args:
            family_counts: Map of model family name to number of observations.
            correlations: Optional custom intra-family correlation overrides.
            calibrated: Whether empirical calibration data exists for these families.
            method: Weight calculation strategy: 'sqrt_heuristic' (1/sqrt(N)) or 'effective_sample_size' (N_eff/N).

        Returns:
            CorrelationDiscountResult containing effective sizes and normalized weights.
        """
        corr_map = dict(cls.DEFAULT_FAMILY_CORRELATIONS)
        if correlations:
            corr_map.update(correlations)

        eff_sizes: dict[str, float] = {}
        family_weights: dict[str, float] = {}
        assumed_corrs: dict[str, float] = {}

        for fam, count in family_counts.items():
            if count <= 0:
                continue
            rho = corr_map.get(fam, corr_map["generic"])
            assumed_corrs[fam] = rho
            n_eff = cls.effective_sample_size(count, rho)
            eff_sizes[fam] = n_eff

            if method == "sqrt_heuristic":
                family_weights[fam] = 1.0 / math.sqrt(count)
            else:
                family_weights[fam] = float(n_eff / count)

        total_eff = sum(eff_sizes.values())
        total_w = sum(family_weights.values())

        if calibrated:
            basis = f"calibrated_effective_sample_size(rho_modeled) with sum(N_eff)={total_eff:.2f}"
        elif method == "sqrt_heuristic":
            basis = f"heuristic_uncalibrated_sqrt_N(weights_1_over_sqrt_N, sum(N_eff)={total_eff:.2f})"
        else:
            basis = f"uncalibrated_effective_sample_size(assumed_rho) with sum(N_eff)={total_eff:.2f}"

        return CorrelationDiscountResult(
            family_counts=dict(family_counts),
            effective_sample_sizes=eff_sizes,
            family_weights=family_weights,
            assumed_correlations=assumed_corrs,
            total_effective_sample_size=total_eff,
            total_weight=total_w,
            is_calibrated=calibrated,
            method=method,
            uncertainty_basis=basis,
        )

    @classmethod
    def fuse_covariances_conservative(
        cls,
        covariances: Sequence[np.ndarray],
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Fuses multiple covariance matrices using the Covariance Intersection (CI) principle.

        Guarantees that the resulting covariance matrix is conservative (does not overclaim
        precision) even if cross-correlations between estimators are completely unknown.

        Args:
            covariances: Sequence of (D, D) positive-definite covariance matrices.
            weights: Convex combination weights summing to 1.0 (defaults to uniform).

        Returns:
            (D, D) fused conservative covariance matrix.
        """
        if not covariances:
            raise ValueError("At least one covariance matrix is required for fusion.")

        d = covariances[0].shape[0]
        for c in covariances:
            if c.shape != (d, d):
                raise ValueError(f"Covariance dimension mismatch: expected ({d}, {d}), got {c.shape}")

        k = len(covariances)
        if weights is None:
            w = np.full(k, 1.0 / k, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if len(w) != k:
                raise ValueError(f"Weight count {len(w)} does not match covariance count {k}")
            w_sum = np.sum(w)
            if w_sum <= 0:
                raise ValueError("Weights must sum to a positive value.")
            w = w / w_sum

        # P_fused^-1 = sum_i (w_i * P_i^-1)
        precision_sum = np.zeros((d, d), dtype=float)
        for cov_i, w_i in zip(covariances, w):
            # Regularize slightly for numerical inversion stability
            reg_cov = cov_i + np.eye(d) * 1e-9
            inv_cov = np.linalg.inv(reg_cov)
            precision_sum += w_i * inv_cov

        fused_cov = np.linalg.inv(precision_sum)
        # Symmetrize
        return 0.5 * (fused_cov + fused_cov.T)

    @classmethod
    def monte_carlo_variance_validation(
        cls,
        n_estimators: int,
        rho: float,
        unit_variance: float = 1.0,
        trials: int = 2000,
        seed: int = 42,
    ) -> dict[str, float]:
        """Runs Monte Carlo simulation of N correlated estimators to empirically validate VIF.

        Returns:
            Dict containing theoretical variance, empirical variance, and relative error.
        """
        rng = np.random.default_rng(seed)
        # Construct equicorrelation matrix Sigma: Sigma_ii = 1, Sigma_ij = rho
        cov_matrix = np.full((n_estimators, n_estimators), rho * unit_variance, dtype=float)
        np.fill_diagonal(cov_matrix, unit_variance)

        # Generate correlated samples: shape (trials, n_estimators)
        samples = rng.multivariate_normal(
            mean=np.zeros(n_estimators),
            cov=cov_matrix,
            size=trials,
        )

        # Sample mean across estimators for each trial: shape (trials,)
        sample_means = np.mean(samples, axis=1)
        empirical_variance = float(np.var(sample_means, ddof=1))

        # Theoretical variance: sigma^2 * (1 + (n - 1) * rho) / n
        theoretical_variance = unit_variance * (1.0 + (n_estimators - 1) * rho) / n_estimators
        rel_error = abs(empirical_variance - theoretical_variance) / theoretical_variance

        return {
            "n_estimators": n_estimators,
            "rho": rho,
            "theoretical_variance": theoretical_variance,
            "empirical_variance": empirical_variance,
            "relative_error": rel_error,
            "effective_sample_size": cls.effective_sample_size(n_estimators, rho),
        }
