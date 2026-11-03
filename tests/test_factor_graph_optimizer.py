"""Unit tests for the GTSAM factor graph trajectory optimization engine."""

import numpy as np
import pytest

from trajectories.optimization.factor_graph import FactorGraphOptimizer


def test_factor_graph_linear_trajectory_smoothing():
    """Verify factor graph eliminates odometry drift when pinned by start and end priors."""
    # True trajectory: 5 steps along X axis: [0, 1, 2, 3, 4]
    true_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])

    optimizer = FactorGraphOptimizer()
    # Prior at start (origin)
    optimizer.add_prior_factor(0, true_positions[0], noise_sigma=0.001)
    # Prior at end (GNSS fix at step 4)
    optimizer.add_prior_factor(4, true_positions[4], noise_sigma=0.001)

    # Odometry steps with cumulative drift (+0.2m drift on step 1-2)
    optimizer.add_between_factor(0, 1, np.array([1.2, 0.0, 0.0]), noise_sigma=0.05)
    optimizer.add_between_factor(1, 2, np.array([1.1, 0.0, 0.0]), noise_sigma=0.05)
    optimizer.add_between_factor(2, 3, np.array([0.9, 0.0, 0.0]), noise_sigma=0.05)
    optimizer.add_between_factor(3, 4, np.array([1.0, 0.0, 0.0]), noise_sigma=0.05)

    # Initial drifted estimate
    initial_est = np.array([
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [2.3, 0.0, 0.0],
        [3.2, 0.0, 0.0],
        [4.2, 0.0, 0.0],
    ])

    result = optimizer.optimize(initial_est)

    assert result.final_cost < result.initial_cost
    assert result.converged is True
    # Start and end must match priors within 2mm
    assert np.allclose(result.optimized_positions[0], true_positions[0], atol=2e-3)
    assert np.allclose(result.optimized_positions[4], true_positions[4], atol=2e-3)
    # Middle positions should be evenly smoothed
    assert result.optimized_positions[2, 0] == pytest.approx(2.20, abs=0.01)


def test_factor_graph_loop_closure_constraint():
    """Verify loop closure factor corrects circular trajectory closure error."""
    # 4 points forming a square: (0,0) -> (2,0) -> (2,2) -> (0,2) -> loop back to (0,0)
    optimizer = FactorGraphOptimizer()
    optimizer.add_prior_factor(0, np.array([0.0, 0.0, 0.0]), noise_sigma=0.001)

    # Odometry steps
    optimizer.add_between_factor(0, 1, np.array([2.0, 0.0, 0.0]), noise_sigma=0.05)
    optimizer.add_between_factor(1, 2, np.array([0.0, 2.0, 0.0]), noise_sigma=0.05)
    optimizer.add_between_factor(2, 3, np.array([-2.0, 0.0, 0.0]), noise_sigma=0.05)

    # Loop closure: step 3 should connect back to step 0 with delta [0, -2, 0]
    optimizer.add_between_factor(3, 0, np.array([0.0, -2.0, 0.0]), noise_sigma=0.01, is_loop_closure=True)

    initial_est = np.array([
        [0.0, 0.0, 0.0],
        [2.1, 0.1, 0.0],
        [2.2, 2.1, 0.0],
        [0.2, 2.2, 0.0],
    ])

    result = optimizer.optimize(initial_est)

    assert result.loop_closures_enforced == 1
    assert result.final_cost < result.initial_cost
    # Final step 3 to step 0 distance should be close to 2.0m along Y
    p3 = result.optimized_positions[3]
    p0 = result.optimized_positions[0]
    assert np.linalg.norm(p3 - p0) == pytest.approx(2.0, abs=0.05)
