"""F2: metric scale from device depth (LiDAR / structured light), estimated and gated honestly.

SYNTHETIC cameras and depth: a known metres-per-model-unit ratio. Asserts the estimator and the refusal path, not a real
LiDAR capture.
"""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult
from reconstruction.calibration.camera import CameraIntrinsics
from reconstruction.depth_scale import DepthScaleEstimate, estimate_scale_from_depth, scale_from_depth

W = H = 8
INTR = CameraIntrinsics(fx=60.0, fy=60.0, cx=W / 2, cy=H / 2, width=W, height=H)
POSE = ReconstructedCameraPose(evidence_id="cam", position=(0.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0))


def _points(n=40, z=2.0):
    # a small grid in front of the camera, all at z model units
    out = []
    k = 0
    for i in range(7):
        for j in range(7):
            out.append(ReconstructedPoint(position=((i - 3) * 0.01, (j - 3) * 0.01, z), track_id=f"p{k}",
                                          source_evidence_ids=["cam"]))
            k += 1
    return out[:n]


def _depth(mult=2.0, size=(H, W)):
    return np.full(size, mult * 2.0, float)     # model z = 2 -> depth = mult * 2 metres


def test_recovers_the_known_ratio_and_applies_it():
    est = estimate_scale_from_depth([POSE], _points(), {"cam": _depth(2.0)}, INTR, stride=1)
    assert est.accepted and est.meters_per_unit == pytest.approx(2.0, rel=1e-9)
    assert est.samples >= 30 and est.ratio_spread == pytest.approx(0.0)
    result = ReconstructionResult(points=_points(), camera_poses=[POSE], registration_status="success")
    scaled = scale_from_depth(result, est)
    assert scaled.state == "metric" and scaled.meters_per_unit == pytest.approx(2.0)
    assert scaled.result.points[0].position[2] == pytest.approx(4.0)
    assert scaled.diagnostics.basis == "device_depth" and scaled.diagnostics.dimension_state == "MEASURED"
    assert "device_depth" in scaled.diagnostics.note


def test_disagreeing_depth_is_refused_never_averaged_into_a_plausible_scale():
    noisy = _depth(2.0).copy()
    noisy[4:, :] *= 3.0                       # half the frame says 6 m where the model says 4 m
    est = estimate_scale_from_depth([POSE], _points(), {"cam": noisy}, INTR, stride=1)
    assert not est.accepted and est.meters_per_unit is None
    assert "spread" in est.reason
    result = ReconstructionResult(points=_points(), camera_poses=[POSE], registration_status="success")
    with pytest.raises(ValueError, match="spread"):
        scale_from_depth(result, est)


def test_too_few_samples_and_missing_frames_are_reported():
    est = estimate_scale_from_depth([POSE], _points(n=5), {"cam": _depth()}, INTR, stride=1)
    assert not est.accepted and "need 30" in est.reason
    est2 = estimate_scale_from_depth([POSE], _points(), {}, INTR)
    assert not est2.accepted and "0 valid" in est2.reason


def test_mismatched_depth_size_or_points_behind_the_camera_are_ignored():
    est = estimate_scale_from_depth([POSE], _points(), {"cam": _depth(size=(4, 4))}, INTR, stride=1)
    assert not est.accepted                       # intrinsics/depth size mismatch: refused, not resampled
    behind = [ReconstructedPoint(position=(0, 0, -2.0), track_id=f"b{i}", source_evidence_ids=["cam"]) for i in range(40)]
    assert not estimate_scale_from_depth([POSE], behind, {"cam": _depth()}, INTR, stride=1).accepted


def test_estimate_round_trips_to_json():
    est = estimate_scale_from_depth([POSE], _points(), {"cam": _depth()}, INTR, stride=1)
    d = est.to_dict()
    assert d["method"] == "device_depth" and d["accepted"] is True and d["used_views"] == ["cam"]
