"""Geometry metrics checked against synthetic cases with analytically known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.metrics import geometry as g

SQUARE = (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), np.array([[0, 1, 2], [0, 2, 3]]))  # unit square in z=0


def _pose(r=np.eye(3), t=(0, 0, 0)):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


# ------------------------------------------------------------------ point clouds
def test_chamfer_known_translation():
    a = np.array([[0, 0, 0], [10, 0, 0]], float)
    b = a + [0, 0.5, 0]
    r = g.chamfer(a, b)
    assert r == pytest.approx({"accuracy_m": 0.5, "completeness_m": 0.5, "chamfer_m": 0.5})


def test_chamfer_asymmetric_partial_prediction():
    gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    pred = gt[:2]  # perfect accuracy, half the scene missing
    r = g.chamfer(pred, gt)
    assert r["accuracy_m"] == 0.0 and r["completeness_m"] == pytest.approx((0 + 0 + 1 + 2) / 4)


def test_fscore_thresholds_known_answer():
    gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    pred = np.array([[0.01, 0, 0], [1.2, 0, 0], [50, 0, 0]], float)
    out = g.fscore(pred, gt, (0.05, 0.25))
    # t=0.05: only the first pred is near a GT point (1/3 precision), only GT[0] is near a pred (1/4 recall)
    assert out["0.05"]["precision"] == pytest.approx(1 / 3) and out["0.05"]["recall"] == pytest.approx(1 / 4)
    assert out["0.05"]["fscore"] == pytest.approx(2 * (1 / 3) * (1 / 4) / (1 / 3 + 1 / 4))
    # t=0.25 additionally admits the 1.2 point and GT[1]
    assert out["0.25"]["precision"] == pytest.approx(2 / 3) and out["0.25"]["recall"] == pytest.approx(2 / 4)


def test_fscore_zero_when_nothing_matches_and_bad_threshold_rejected():
    assert g.fscore([[0, 0, 0]], [[9, 9, 9]], (0.1,))["0.1"]["fscore"] == 0.0
    with pytest.raises(ValueError):
        g.fscore([[0, 0, 0]], [[0, 0, 0]], (0,))
    with pytest.raises(ValueError):
        g.chamfer(np.empty((0, 3)), [[0, 0, 0]])


# ------------------------------------------------------------------------ meshes
@pytest.mark.parametrize("pt,expected", [
    ((0.5, 0.5, 2.0), 2.0),                     # above the interior
    ((0.5, 0.5, 0.0), 0.0),                     # on the surface
    ((3.0, 0.5, 0.0), 2.0),                     # beside an edge, in plane
    ((2.0, 2.0, 0.0), math.sqrt(2)),            # nearest to a corner vertex
    ((-1.0, -1.0, 1.0), math.sqrt(3)),          # off a corner and above
    ((0.5, -0.3, 0.4), math.hypot(0.3, 0.4)),   # beside an edge and above
])
def test_point_to_surface_distance_exact(pt, expected):
    assert g.point_to_surface_distance([pt], SQUARE)[0] == pytest.approx(expected, abs=1e-12)


def test_point_to_surface_distance_is_surface_not_vertex():
    # centre of a big quad is far from every vertex but ON the surface
    v = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], float)
    d = g.point_to_surface_distance([[5, 5, 0.3]], (v, np.array([[0, 1, 2], [0, 2, 3]])))
    assert d[0] == pytest.approx(0.3)


def test_mesh_accuracy_completeness_known_offset_and_missing_half():
    big = (np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], float), np.array([[0, 1, 2], [0, 2, 3]]))  # 2 x 1 at z=0
    half_up = (big[0] + [0, 0, 0.05], big[1])                       # same quad lifted 5 cm
    r = g.mesh_accuracy_completeness(half_up, big, n_samples=500, thresholds_m=(0.04, 0.06))
    assert r["accuracy_m"] == pytest.approx(0.05, abs=1e-9) and r["completeness_m"] == pytest.approx(0.05, abs=1e-9)
    assert r["fscore"]["0.04"]["fscore"] == 0.0 and r["fscore"]["0.06"]["fscore"] == 1.0
    half = (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), np.array([[0, 1, 2], [0, 2, 3]]))  # covers x in [0,1] only
    r2 = g.mesh_accuracy_completeness(half, big, n_samples=4000, thresholds_m=(0.01,))
    assert r2["accuracy_m"] == pytest.approx(0.0, abs=1e-12)
    assert r2["fscore"]["0.01"]["precision"] == 1.0
    assert r2["fscore"]["0.01"]["recall"] == pytest.approx(0.5, abs=0.04)      # half of GT area is covered
    assert r2["completeness_m"] == pytest.approx(0.25, abs=0.03)               # mean distance of uniform x in [1,2] to x=1 is 0.5, x half -> 0.25


def test_sample_surface_deterministic_and_on_surface():
    a, b = g.sample_surface(SQUARE, 50, seed=3), g.sample_surface(SQUARE, 50, seed=3)
    assert np.array_equal(a, b) and np.allclose(a[:, 2], 0) and (a >= 0).all() and (a <= 1).all()


# -------------------------------------------------------------------- camera poses
def _trajectory(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return [_pose(_rz(7 * i), (math.cos(i / 3) * 2, math.sin(i / 3) * 2, 0.1 * i + rng.normal(0, 0.0))) for i in range(n)]


def _similarity(poses, s, r, t):
    out = []
    for p in poses:
        q = np.eye(4)
        q[:3, :3] = r @ p[:3, :3]
        q[:3, 3] = s * r @ p[:3, 3] + t
        out.append(q)
    return out


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 3))
    r, t, s = _rz(40) @ np.array([[1, 0, 0], [0, math.cos(.3), -math.sin(.3)], [0, math.sin(.3), math.cos(.3)]]), np.array([1., -2., 3.]), 2.5
    s_, r_, t_ = g.umeyama_sim3(x, s * x @ r.T + t)
    assert s_ == pytest.approx(s) and np.allclose(r_, r) and np.allclose(t_, t)


def test_camera_registration_zero_error_after_similarity():
    gt = _trajectory()
    est = _similarity(gt, 0.37, _rz(-25), np.array([5., 1., -2.]))      # arbitrary frame AND scale
    r = g.camera_registration({str(i): p for i, p in enumerate(est)}, {str(i): p for i, p in enumerate(gt)})
    assert r["status"] == "OK" and r["ate_rmse_m"] < 1e-9 and r["rotation_err_max_deg"] < 1e-6
    assert r["scale"] == pytest.approx(1 / 0.37) and r["registered_fraction"] == 1.0


def test_camera_registration_known_noise_and_missing_cameras():
    gt = _trajectory()
    est = [p.copy() for p in gt]
    est[5][:3, 3] += [0.12, 0, 0]                                         # one camera 12 cm off
    r = g.camera_registration({str(i): p for i, p in enumerate(est) if i != 9}, {str(i): p for i, p in enumerate(gt)}, with_scale=False)
    assert r["registered_fraction"] == pytest.approx(11 / 12) and r["n_registered"] == 11
    assert 0.0 < r["ate_rmse_m"] < 0.12 and r["ate_max_m"] <= 0.12 + 1e-9    # alignment spreads (never inflates) the single outlier


def test_camera_registration_blocked_when_too_few_or_collinear():
    gt = {"a": _pose(), "b": _pose(t=(1, 0, 0))}
    assert g.camera_registration(gt, gt)["status"] == "BLOCKED"
    line = {str(i): _pose(t=(i, 0, 0)) for i in range(5)}
    r = g.camera_registration(line, line)
    assert r["status"] == "BLOCKED" and "collinear" in r["reason"]


def test_rotation_error_detects_known_rotation():
    gt = _trajectory()
    est = [p.copy() for p in gt]
    est[3][:3, :3] = est[3][:3, :3] @ _rz(10)
    r = g.camera_registration({str(i): p for i, p in enumerate(est)}, {str(i): p for i, p in enumerate(gt)}, with_scale=False)
    assert r["rotation_err_max_deg"] == pytest.approx(10.0, abs=2.0)  # the Sim3 fit absorbs a little of a single-camera rotation offset


def test_trajectory_rpe_perfect_and_drift():
    gt = _trajectory(15)
    perfect = g.trajectory_errors(_similarity(gt, 2.0, _rz(15), np.array([1., 2., 3.])), gt)
    assert perfect["rpe_trans_rmse_m"] < 1e-9 and perfect["rpe_rot_rmse_deg"] < 1e-6 and perfect["ate_rmse_m"] < 1e-9
    drift = []
    for i, p in enumerate(gt):                                            # accumulating drift along x
        q = p.copy()
        q[0, 3] += 0.01 * i
        drift.append(q)
    r = g.trajectory_errors(drift, gt, with_scale=False)
    assert r["rpe_trans_rmse_m"] > 1e-4 and r["ate_rmse_m"] > 0.01
    with pytest.raises(ValueError):
        g.trajectory_errors(gt[:3], gt)


# ------------------------------------------------------------------- structural
def _rect(kind, center, u, v, hu, hv):
    return {"kind": kind, "center": center, "u": u, "v": v, "half_u": hu, "half_v": hv}


FLOOR = _rect("floor", (2, 1.5, 0), (1, 0, 0), (0, 1, 0), 2, 1.5)          # 4 x 3
WALL = _rect("wall", (2, 0, 1.25), (1, 0, 0), (0, 0, 1), 2, 1.25)          # 4 x 2.5 on y=0


def test_structural_completeness_full_half_and_misplaced():
    full = g.structural_completeness([FLOOR, WALL], [FLOOR, WALL])
    assert full["floor"]["completeness"] == pytest.approx(1.0) and full["wall"]["completeness"] == pytest.approx(1.0)
    assert full["wall"]["spurious_area_fraction"] == 0.0 and full["ceiling"]["status"] == "BLOCKED"
    half_floor = _rect("floor", (1, 1.5, 0), (1, 0, 0), (0, 1, 0), 1, 1.5)   # x in [0, 2] only
    r = g.structural_completeness([half_floor], [FLOOR], kinds=("floor",))
    assert r["floor"]["completeness"] == pytest.approx(0.5, abs=0.03) and r["floor"]["spurious_area_fraction"] == 0.0
    moved = _rect("floor", (2, 1.5, 0.5), (1, 0, 0), (0, 1, 0), 2, 1.5)      # 50 cm off in height, tolerance 10 cm
    assert g.structural_completeness([moved], [FLOOR], kinds=("floor",))["floor"]["completeness"] == 0.0
    assert g.structural_completeness([moved], [FLOOR], kinds=("floor",), dist_tol_m=0.6)["floor"]["completeness"] == pytest.approx(1.0)


def test_structural_completeness_kind_and_orientation_must_match():
    tilted = _rect("wall", (2, 0, 1.25), (1, 0, 0), (0, math.sin(math.radians(30)), math.cos(math.radians(30))), 2, 1.25)
    assert g.structural_completeness([tilted], [WALL], kinds=("wall",))["wall"]["completeness"] == 0.0            # 30 deg off > 10 deg tol
    as_floor = dict(WALL, kind="floor")
    assert g.structural_completeness([as_floor], [WALL], kinds=("wall",))["wall"]["completeness"] == 0.0           # wrong kind
    extra = g.structural_completeness([WALL, _rect("wall", (9, 9, 1), (1, 0, 0), (0, 0, 1), 1, 1)], [WALL], kinds=("wall",))
    assert extra["wall"]["spurious_area_fraction"] == pytest.approx(4 / (10 + 4), abs=0.02)


def test_opening_completeness_matching_rules():
    gt = [{"kind": "door", "center": (1, 0, 1), "width_m": 0.9, "height_m": 2.0}, {"kind": "window", "center": (3, 0, 1.5), "width_m": 1.2, "height_m": 1.0}]
    pred = [{"kind": "door", "center": (1.05, 0, 1), "width_m": 0.85, "height_m": 2.1}, {"kind": "window", "center": (3, 0, 1.5), "width_m": 2.5, "height_m": 1.0}]
    r = g.opening_completeness(pred, gt)
    assert r["recall"] == 0.5 and r["precision"] == 0.5 and r["width_mae_m"] == pytest.approx(0.05)    # wrong-width window not matched
    assert g.opening_completeness([], gt)["recall"] == 0.0
    assert g.opening_completeness(pred, [])["status"] == "BLOCKED"


# --------------------------------------------------------------------- dimensions
def test_dimension_errors_only_independent_measurements():
    gt = [{"name": "w", "value_m": 3.0, "uncertainty_m": 0.005, "provenance_class": "measured"},
          {"name": "h", "value_m": 2.5, "uncertainty_m": 0.005, "provenance_class": "fitted"},
          {"name": "d", "value_m": 4.0, "provenance_class": "measured"},                          # no uncertainty -> incomplete
          {"name": "z", "value_m": 1.0, "uncertainty_m": 0.01, "provenance_class": "measured"}]   # no prediction
    r = g.dimension_errors({"w": 3.03, "h": 2.5, "d": 4.0}, gt)
    assert r["status"] == "OK" and r["n"] == 1 and r["mae_m"] == pytest.approx(0.03) and r["items"][0]["within_2_sigma"] is False
    assert {e["name"] for e in r["excluded"]} == {"h", "d", "z"}
    blocked = g.dimension_errors({"h": 2.5}, [gt[1]])
    assert blocked["status"] == "BLOCKED" and "mae_m" not in blocked
