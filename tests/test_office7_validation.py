"""The ground-truth validator's maths, checked against a KNOWN transform (synthetic points; proves the validator, not any reconstruction)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.measured_rooms.validate_office7 import gt_centre, umeyama, validate_cameras


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {"z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]])}[axis]


PTS = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1], [2, -1, 0.5]], float)


def test_umeyama_recovers_a_known_scale_rotation_and_translation():
    R, s, t = _rot("z", 40) @ _rot("x", 25), 2.5, np.array([1.0, -2.0, 0.3])
    s_hat, R_hat, t_hat = umeyama(PTS, s * PTS @ R.T + t)
    assert s_hat == pytest.approx(s) and np.allclose(R_hat, R) and np.allclose(t_hat, t)


def test_a_mirrored_trajectory_is_not_fitted_by_a_reflection():
    mirrored = PTS * np.array([1, 1, -1])  # a mirror image cannot be reached by a proper rotation
    s, R, t = umeyama(PTS, mirrored)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert np.linalg.norm(s * PTS @ R.T + t - mirrored, axis=1).max() > 0.1  # the residual stays visible instead of being absorbed


def test_validate_cameras_reports_scale_and_a_perturbed_camera():
    gt = {f"f{i}": p * 2.0 for i, p in enumerate(PTS)}  # ground truth in metres: 2 m per SfM unit
    sfm = {k: list(v / 2.0) for k, v in gt.items()}
    ok = validate_cameras(sfm, gt)
    assert ok["scale_metres_per_sfm_unit"] == pytest.approx(2.0) and ok["ate_rmse_m"] < 1e-9
    sfm["f3"] = [x + 0.5 for x in sfm["f3"]]  # one wrong camera must show up
    bad = validate_cameras(sfm, gt)
    assert bad["ate_rmse_m"] > 0.05 and bad["ate_rmse_m"] > ok["ate_rmse_m"]  # a least-squares fit smears one outlier, but never hides it


def test_too_few_matching_cameras_is_insufficient_not_a_fit():
    assert validate_cameras({"a": [0, 0, 0]}, {"a": np.zeros(3)})["status"] == "insufficient"
    with pytest.raises(ValueError):
        umeyama(np.zeros((5, 3)), PTS[:5])  # degenerate source


def test_pose_file_centre_is_the_translation_column(tmp_path):
    p = tmp_path / "frame-000000.pose.txt"
    p.write_text("1 0 0 1.5\n0 1 0 -2\n0 0 1 0.25\n0 0 0 1\n")
    assert np.allclose(gt_centre(p), [1.5, -2.0, 0.25])
