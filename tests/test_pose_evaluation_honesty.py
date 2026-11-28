"""Regression (defect B2): Sim(3)-fitted residuals must never read as independent metric accuracy."""

import dataclasses

import numpy as np

from reconstruction.backend.interface import ReconstructedCameraPose
from reconstruction.evaluation import evaluate_absolute_poses

# Reference world->camera rows (qw,qx,qy,qz,tx,ty,tz): identity rotation, non-collinear centres.
_CENTERS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.5)]


def _gt():
    return {f"i{k}.jpg": (1.0, 0.0, 0.0, 0.0, -c[0], -c[1], -c[2]) for k, c in enumerate(_CENTERS)}


def _poses(scale):
    names = {f.name for f in dataclasses.fields(ReconstructedCameraPose)}
    extra = {k: v for k, v in {"confidence": 1.0, "reprojection_error": 0.0}.items() if k in names}
    return [
        ReconstructedCameraPose(evidence_id=f"i{k}.jpg", rotation=(1.0, 0.0, 0.0, 0.0),
                                position=tuple(scale * np.asarray(c)), **extra)
        for k, c in enumerate(_CENTERS)
    ]


def test_scaled_estimate_has_zero_fitted_residual_but_is_not_called_accurate():
    rep = evaluate_absolute_poses(_poses(10.0), _gt())
    assert rep["center_error_max"] < 1e-9           # Sim(3) scale fit hides the 10x error ...
    assert rep["residual_is_independent_metric_accuracy"] is False  # ... and the report says so
    assert "metric_scale_error_pct" not in rep
    assert any("not declared metric" in n for n in rep["notes"])


def test_declared_metric_estimate_exposes_the_scale_error():
    rep = evaluate_absolute_poses(_poses(10.0), _gt(), estimated_scale_state="metric")
    assert abs(rep["scale_ratio"] - 10.0) < 1e-6
    assert abs(rep["metric_scale_error_pct"] - 900.0) < 1e-3
    assert rep["metric_center_error_max"] > 1.0


def test_correctly_scaled_metric_estimate_has_near_zero_metric_error():
    rep = evaluate_absolute_poses(_poses(1.0), _gt(), estimated_scale_state="metric")
    assert rep["metric_scale_error_pct"] < 1e-6 and rep["metric_center_error_max"] < 1e-9
