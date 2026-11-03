"""Self-contained Factor Graph Optimization Engine (Block 17 of 50-tech evaluation).

Implements nonlinear least-squares factor graph optimization fusing:
1. Odometry Between-Factors (relative camera/sensor motions)
2. Absolute Prior Factors (WGS84/ENU GNSS geodetic anchors or metric scale baselines)
3. Non-sequential Loop Closure Factors
Based on the factor-graph principles of Dellaert, Kaess, et al. (GTSAM / iSAM2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PriorFactor3D:
    """Prior constraint pinning or anchoring one pose index."""

    pose_idx: int
    measured_position: np.ndarray  # (3,)
    noise_sigma: float = 0.01  # std dev in meters


@dataclass(frozen=True)
class BetweenFactor3D:
    """Relative transformation constraint between pose_a and pose_b."""

    pose_a: int
    pose_b: int
    delta_position: np.ndarray  # (3,)
    noise_sigma: float = 0.05
    is_loop_closure: bool = False


@dataclass(frozen=True)
class OptimizationResult:
    """Output trajectory and convergence statistics."""

    optimized_positions: np.ndarray  # (K, 3)
    initial_cost: float
    final_cost: float
    iterations: int
    converged: bool
    loop_closures_enforced: int


class FactorGraphOptimizer:
    """Nonlinear factor graph optimizer for 3D trajectory refinement."""

    def __init__(
        self,
        max_iterations: int = 30,
        convergence_tol: float = 1e-5,
        damping_lambda: float = 1e-3,
    ) -> None:
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.damping_lambda = damping_lambda
        self._priors: List[PriorFactor3D] = []
        self._between: List[BetweenFactor3D] = []

    def add_prior_factor(self, pose_idx: int, position: np.ndarray, noise_sigma: float = 0.01) -> None:
        """Add absolute anchor prior factor."""
        self._priors.append(PriorFactor3D(pose_idx, np.asarray(position, dtype=np.float64), noise_sigma))

    def add_between_factor(
        self,
        pose_a: int,
        pose_b: int,
        delta: np.ndarray,
        noise_sigma: float = 0.05,
        is_loop_closure: bool = False,
    ) -> None:
        """Add relative motion or loop closure between factor."""
        self._between.append(BetweenFactor3D(pose_a, pose_b, np.asarray(delta, dtype=np.float64), noise_sigma, is_loop_closure))

    def _compute_cost(self, x: np.ndarray) -> float:
        """Compute sum of squared Mahalanobis residuals across all factors."""
        total_cost = 0.0
        for p in self._priors:
            res = x[p.pose_idx] - p.measured_position
            total_cost += float(np.sum((res / p.noise_sigma) ** 2))

        for b in self._between:
            pred_delta = x[b.pose_b] - x[b.pose_a]
            res = pred_delta - b.delta_position
            total_cost += float(np.sum((res / b.noise_sigma) ** 2))

        return 0.5 * total_cost

    def optimize(self, initial_positions: np.ndarray) -> OptimizationResult:
        """Run Levenberg-Marquardt nonlinear optimization to find optimal trajectory."""
        x = np.asarray(initial_positions, dtype=np.float64).copy()
        K = len(x)
        initial_cost = self._compute_cost(x)
        current_cost = initial_cost
        lam = self.damping_lambda
        converged = False
        iters = 0

        # Optimization loop
        for iters in range(1, self.max_iterations + 1):
            # Form linear system: H dx = -g
            # H: (3K, 3K), g: (3K,)
            dim = 3 * K
            H = np.zeros((dim, dim), dtype=np.float64)
            g = np.zeros(dim, dtype=np.float64)

            # 1. Accumulate priors
            for p in self._priors:
                idx = p.pose_idx * 3
                w = 1.0 / (p.noise_sigma**2)
                res = x[p.pose_idx] - p.measured_position
                H[idx : idx + 3, idx : idx + 3] += np.eye(3) * w
                g[idx : idx + 3] += res * w

            # 2. Accumulate between factors (odometry + loop closures)
            for b in self._between:
                ia = b.pose_a * 3
                ib = b.pose_b * 3
                w = 1.0 / (b.noise_sigma**2)
                res = (x[b.pose_b] - x[b.pose_a]) - b.delta_position

                # Jacobian J_a = -I, J_b = I
                # H_aa += J_a^T W J_a = W * I
                # H_bb += J_b^T W J_b = W * I
                # H_ab += J_a^T W J_b = -W * I
                H[ia : ia + 3, ia : ia + 3] += np.eye(3) * w
                H[ib : ib + 3, ib : ib + 3] += np.eye(3) * w
                H[ia : ia + 3, ib : ib + 3] -= np.eye(3) * w
                H[ib : ib + 3, ia : ia + 3] -= np.eye(3) * w

                g[ia : ia + 3] -= res * w
                g[ib : ib + 3] += res * w

            # Levenberg-Marquardt damping
            H_damped = H + (np.eye(dim) * lam)

            try:
                dx = -np.linalg.solve(H_damped, g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            dx_3d = dx.reshape((K, 3))
            candidate_x = x + dx_3d
            candidate_cost = self._compute_cost(candidate_x)

            if candidate_cost < current_cost:
                # Step accepted
                cost_reduction = current_cost - candidate_cost
                current_cost = candidate_cost
                x = candidate_x
                lam = max(1e-7, lam * 0.5)

                if cost_reduction < self.convergence_tol:
                    converged = True
                    break
            else:
                # Step rejected, increase damping
                lam *= 5.0

        loop_count = sum(1 for b in self._between if b.is_loop_closure)

        return OptimizationResult(
            optimized_positions=x,
            initial_cost=initial_cost,
            final_cost=current_cost,
            iterations=iters,
            converged=converged,
            loop_closures_enforced=loop_count,
        )
