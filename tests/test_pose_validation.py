"""perception.pose_validation: robust ATE/RPE, scale-state handling, reference-free validation."""
import numpy as np
import pytest

from perception.pose_validation import (evaluate_trajectory, robust_align, rotation_about_axis,
                                        validate_poses)


def _ring(n=24, radius=5.0):
    """Cameras on a circle looking at the origin, world->camera poses."""
    poses = {}
    for i in range(n):
        a = 2 * np.pi * i / n
        C = np.array([radius * np.cos(a), 0.3 * np.sin(3 * a), radius * np.sin(a)])
        z = -C / np.linalg.norm(C)
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        Rc2w = np.stack([x, y, z], 1)
        poses[f"f{i:03d}"] = (Rc2w.T, -Rc2w.T @ C)
    return poses


def _gauge(poses, s, R, t):
    """Express the same cameras in another world frame: X' = s R X + t."""
    out = {}
    for n, (Rw, tw) in poses.items():
        C = -Rw.T @ tw
        C2 = s * R @ C + t
        Rw2 = Rw @ R.T
        out[n] = (Rw2, -Rw2 @ C2)
    return out


G = (2.5, rotation_about_axis(37, (1, 2, 3)), np.array([4.0, -2.0, 9.0]))


def test_exact_gauge_recovered_zero_error():
    ref = _ring()
    est = _gauge(ref, *G)
    r = evaluate_trajectory(est, ref, est_scale_state="relative", ref_scale_state="relative",
                            order=sorted(ref))
    assert not r["refused"] and r["alignment"] == "sim3"
    assert r["gauge_scale"] == pytest.approx(1 / G[0], rel=1e-6)
    assert r["ate_translation_ref_units"]["all"]["max"] < 1e-9
    assert r["rotation_error_deg"]["all"]["max"] < 1e-3
    assert r["rpe"]["rot_deg"]["max"] < 1e-3 and r["rpe"]["trans_ref_units"]["max"] < 1e-9
    assert r["outliers"] == []


def test_113_degree_outlier_named_and_does_not_poison_good_cameras():
    ref = _ring()
    est = _gauge(ref, *G)
    bad = "f007"
    R, t = est[bad]
    est[bad] = (rotation_about_axis(113, (0, 1, 0)) @ R, t)
    r = evaluate_trajectory(est, ref, est_scale_state="relative", ref_scale_state="relative")
    assert [o["name"] for o in r["outliers"]] == [bad]
    assert r["outliers"][0]["rot_err_deg"] > 100
    # the other 23 stay exact: a non-robust fit would spread the outlier over all of them
    assert r["rotation_error_deg"]["inliers"]["max"] < 1e-3


def test_position_outlier_rejected_by_robust_alignment():
    ref = _ring()
    est = _gauge(ref, *G)
    R, t = est["f003"]
    C = -R.T @ t + np.array([20.0, 0, 0])
    est["f003"] = (R, -R @ C)
    r = evaluate_trajectory(est, ref, est_scale_state="relative", ref_scale_state="relative")
    assert "f003" in [o["name"] for o in r["outliers"]]
    assert r["alignment_inliers"] == len(ref) - 1


def test_metric_vs_metric_uses_se3_and_scale_error_is_not_absorbed():
    ref = _ring()
    est = _gauge(ref, 1.1, G[1], G[2])  # 10 % scale error, both claim metric
    r = evaluate_trajectory(est, ref, est_scale_state="metric", ref_scale_state="metric")
    assert r["alignment"] == "se3" and r["gauge_scale"] is None
    assert r["ate_translation_ref_units"]["all"]["rmse"] > 0.1
    r2 = evaluate_trajectory(est, ref, est_scale_state="relative", ref_scale_state="metric")
    assert r2["alignment"] == "sim3" and r2["ate_translation_ref_units"]["all"]["max"] < 1e-9


def test_relative_reference_is_flagged_not_metres():
    ref = _ring()
    r = evaluate_trajectory(_gauge(ref, *G), ref, est_scale_state="relative", ref_scale_state="relative")
    assert any("not metres" in n for n in r["notes"])


def test_refuses_instead_of_scoring_too_few_or_degenerate():
    ref = _ring()
    assert evaluate_trajectory({k: ref[k] for k in list(ref)[:3]}, ref, est_scale_state="relative",
                               ref_scale_state="relative")["refused"]
    same = {f"c{i}": (np.eye(3), np.zeros(3)) for i in range(6)}  # coincident centres
    assert evaluate_trajectory(same, same, est_scale_state="relative", ref_scale_state="relative")["refused"]
    with pytest.raises(ValueError):
        evaluate_trajectory(ref, ref, est_scale_state="maybe", ref_scale_state="relative")


def test_robust_align_degenerate_returns_none():
    p = np.zeros((5, 3))
    assert robust_align(p, p, with_scale=True) is None


def test_validate_flags_bad_frame_components_and_bad_rotation():
    poses = _ring(30)
    order = sorted(poses)
    R, t = poses["f010"]
    poses["f010"] = (rotation_about_axis(113, (0, 0, 1)) @ R, t)
    poses["bad"] = (np.diag([1, 1, -1.0]), np.zeros(3))  # reflection
    edges = [(order[i], order[i + 1]) for i in range(29) if i != 14]  # cut chain in two
    v = validate_poses(poses, edges=edges, order=order)
    kinds = {(i.get("name"), i["issue"]) for i in v["issues"]}
    assert ("f010", "trajectory_rotation_jump") in kinds
    assert ("bad", "not_a_rotation") in kinds
    assert sorted(v["components"]) == [1, 15, 15]
    assert any(i["issue"] == "disconnected_components" for i in v["issues"])


def test_validate_clean_ring_has_no_issues():
    poses = _ring()
    order = sorted(poses)
    v = validate_poses(poses, edges=list(zip(order[:-1], order[1:])), order=order)
    assert v["issues"] == [] and v["components"] == [len(poses)]
