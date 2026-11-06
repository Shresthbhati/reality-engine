"""RANSAC metric-scale fusion is deterministic: same constraints -> same fused scale (no unseeded global RNG)."""

from __future__ import annotations

from reconstruction.scale import MetricScaleFusionEngine, ScaleConstraint, ScaleConstraintType


def _constraints():
    # Two clusters of ratios (2.0 and 3.0 m/unit) with near-equal weight, so which one wins depends on the sampling.
    out = []
    for i, ratio in enumerate([2.0, 2.01, 1.99, 3.0, 3.01, 2.99, 2.0, 3.0]):
        out.append(ScaleConstraint(
            constraint_type=ScaleConstraintType.OPERATOR_BASELINE, source_id=f"s{i}", target_id=f"t{i}",
            real_distance_m=ratio, model_distance=1.0, weight=0.5, confidence=0.5,
        ))
    return out


def test_same_input_same_output_across_runs():
    runs = [MetricScaleFusionEngine(min_inlier_weight=0.1).fuse_constraints(_constraints()).to_dict() for _ in range(10)]
    assert all(r == runs[0] for r in runs)


def test_seed_is_overridable_and_stable():
    a = MetricScaleFusionEngine(min_inlier_weight=0.1, seed=7).fuse_constraints(_constraints()).to_dict()
    b = MetricScaleFusionEngine(min_inlier_weight=0.1, seed=7).fuse_constraints(_constraints()).to_dict()
    assert a == b
