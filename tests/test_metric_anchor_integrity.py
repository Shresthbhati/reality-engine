"""Metric-anchor and calibration integrity for depth scale alignment. The scenes are SYNTHETIC: they prove the acceptance logic
(what may and may not become metres), not any depth model's accuracy and not physical accuracy of any real reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from engine.math import Quat, Vec3
from engine.multiengine.depth_scale_alignment import (
    MetricAnchor,
    align_relative_depth_to_metric,
    validate_metric_anchor,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

TRUE_SCALE, TRUE_SHIFT = 0.5, 0.1
GAUGE = 0.5  # an arbitrary SfM gauge: SfM units = GAUGE * metres


def _camera():
    return PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )


def _scene(n=60, gauge=GAUGE, corrupt=None):
    """Points in the reconstruction's own gauge (z_sfm = gauge * z_metres) and the relative-depth map that matches them."""
    cam = _camera()
    depth = np.linspace(0.1, 2.0, 480 * 640, dtype=np.float32).reshape(480, 640)
    pts3, pts2, z_metres = [], [], []
    for i in range(n):
        z_m = 2.0 + (i / n) * 6.0
        z_sfm = gauge * z_m
        u, v = 40.0 + (i * 37) % 560, 40.0 + (i * 53) % 400  # spread across all four image quadrants
        p = cam.unproject(u, v, z_sfm)
        pts3.append((p.x, p.y, p.z))
        pts2.append((u, v))
        z_metres.append(z_m)
        depth[int(round(v)), int(round(u))] = float(((1.0 / z_sfm) - TRUE_SHIFT) / TRUE_SCALE)
    if corrupt is not None:
        for i in corrupt(n):
            u, v = pts2[i]
            depth[int(round(v)), int(round(u))] *= 3.0  # a held-out point the fit never saw, now wrong
    return cam, depth, pts3, pts2, np.asarray(z_metres)


def _align(anchor=None, **kw):
    cam, depth, pts3, pts2, _ = _scene(**kw.pop("scene", {}))
    return align_relative_depth_to_metric(depth, cam, pts3, pts2, metric_anchor=anchor, **kw), pts3


def _anchor(pts3, **kw):
    base = dict(anchor_type="laser_survey", measured_distance_m=float(np.linalg.norm(np.array(pts3[0]) - np.array(pts3[-1]))) / GAUGE,
                uncertainty_m=0.005, reference_point_a=pts3[0], reference_point_b=pts3[-1])
    base.update(kw)
    return MetricAnchor(**base)


def test_unanchored_sfm_is_never_metres_and_its_errors_are_labelled_sfm_units():
    res, _ = _align(None)
    assert res.is_aligned and res.units == "COLMAP_SFM_UNITS" and not res.is_physically_metric and res.error_units == "COLMAP_SFM_UNITS"
    assert res.metric_scale_factor is None and res.anchor_relative_uncertainty is None
    d = res.to_dict()
    assert not any(k.endswith("_m") for k in d if "error" in k or "residual" in k)  # no field names an SfM error "metres"


def test_a_positive_distance_without_reference_points_is_rejected_not_labelled_metres():
    _, pts3 = _align(None)
    res, _ = _align(MetricAnchor(anchor_type="laser_survey", measured_distance_m=6.0))
    assert not res.is_aligned and res.scale_state == "INVALID" and not res.is_physically_metric and res.units != "METRES"
    assert "REFERENCE_POINTS_MISSING" in res.rejection_reason and res.aligned_depth_map is None
    one = MetricAnchor(anchor_type="laser_survey", measured_distance_m=6.0, reference_point_a=pts3[0])
    assert "REFERENCE_POINTS_MISSING" in _align(one)[0].rejection_reason  # one point is as incomplete as none


@pytest.mark.parametrize("kw,needle", [
    (dict(measured_distance_m=0.0), "must be > 0"),
    (dict(measured_distance_m=-3.0), "must be > 0"),
    (dict(measured_distance_m=float("nan")), "finite"),
    (dict(measured_distance_m=float("inf")), "finite"),
    (dict(uncertainty_m=float("nan")), "uncertainty_m"),
    (dict(uncertainty_m=-0.1), "uncertainty_m"),
    (dict(uncertainty_m=1e6), "uncertainty_m"),  # an uncertainty larger than the measurement is no measurement
    (dict(reference_point_a=(float("nan"), 0.0, 1.0)), "finite 3-vectors"),
    (dict(reference_point_b=(1.0, 2.0)), "finite 3-vectors"),
    (dict(anchor_type="synthetic_ground_truth"), "SYNTHETIC_ANCHOR"),
    (dict(anchor_type="Synthetic_Lidar"), "SYNTHETIC_ANCHOR"),
    (dict(anchor_type="vibes"), "UNVERIFIED_ANCHOR_TYPE"),
])
def test_invalid_incomplete_or_synthetic_anchors_never_establish_metres(kw, needle):
    _, pts3 = _align(None)
    res, _ = _align(_anchor(pts3, **kw))
    assert not res.is_aligned and res.scale_state == "INVALID" and not res.is_physically_metric and res.units != "METRES"
    assert needle in res.rejection_reason, res.rejection_reason


def test_coincident_reference_points_are_rejected():
    _, pts3 = _align(None)
    res, _ = _align(_anchor(pts3, reference_point_a=pts3[3], reference_point_b=pts3[3]))
    assert not res.is_aligned and "Distance between reference points is zero" in res.rejection_reason


def test_reference_points_that_do_not_exist_in_the_reconstruction_are_rejected():
    _, pts3 = _align(None)
    far = (pts3[0][0] + 50.0, pts3[0][1], pts3[0][2])  # a 'measured' point that corresponds to nothing reconstructed
    res, _ = _align(_anchor(pts3, reference_point_a=far))
    assert not res.is_aligned and "ANCHOR_POINT_NOT_IN_RECONSTRUCTION" in res.rejection_reason


def test_validate_metric_anchor_is_the_single_decision_point():
    _, pts3 = _align(None)
    assert validate_metric_anchor(_anchor(pts3), pts3) is None
    assert validate_metric_anchor(_anchor(pts3, anchor_type="synthetic_ground_truth"), pts3) is not None


def test_a_correct_independent_measurement_recovers_true_metres_on_held_out_points():
    cam, depth, pts3, pts2, z_metres = _scene()
    res = align_relative_depth_to_metric(depth, cam, pts3, pts2, metric_anchor=_anchor(pts3))
    assert res.is_aligned and res.units == "METRES" and res.scale_state == "METRIC_SCALE_ALIGNED" and res.is_physically_metric
    assert res.error_units == "METRES" and res.metric_scale_factor == pytest.approx(1.0 / GAUGE, rel=1e-6)
    # the aligned map reproduces the TRUE metric depth at the correspondences (they live in the arbitrary gauge, the map in metres)
    got = np.array([res.aligned_depth_map[int(round(v)), int(round(u))] for u, v in pts2])
    assert np.median(np.abs(got - z_metres) / z_metres) < 0.02
    assert res.held_out_count >= 5 and res.held_out_relative_rmse < 0.05


def test_anchor_measurement_uncertainty_is_propagated_into_the_scale_uncertainty():
    cam, depth, pts3, pts2, _ = _scene()
    tight = align_relative_depth_to_metric(depth, cam, pts3, pts2, metric_anchor=_anchor(pts3, uncertainty_m=0.001))
    loose = align_relative_depth_to_metric(depth, cam, pts3, pts2, metric_anchor=_anchor(pts3, uncertainty_m=0.3))
    assert loose.anchor_relative_uncertainty > tight.anchor_relative_uncertainty > 0
    assert loose.scale_uncertainty > tight.scale_uncertainty  # a worse tape measure gives a less certain scale


def test_a_wrong_measurement_changes_the_result_in_proportion_so_independent_verification_is_required():
    cam, depth, pts3, pts2, _ = _scene()
    good = align_relative_depth_to_metric(depth, cam, pts3, pts2, metric_anchor=_anchor(pts3))
    d = _anchor(pts3).measured_distance_m
    bad = align_relative_depth_to_metric(depth, cam, pts3, pts2, metric_anchor=_anchor(pts3, measured_distance_m=d * 1.5))
    assert bad.metric_scale_factor == pytest.approx(good.metric_scale_factor * 1.5)  # no internal check can catch a wrong tape reading


def test_high_training_r2_with_poor_held_out_error_is_refused():
    n = 60
    n_train = max(int(round(n * 0.7)), min(10, n - 1))

    def test_idx(total):  # the module's own deterministic split: corrupt exactly the points the fit never saw
        return np.random.RandomState(42).permutation(total)[n_train:]

    cam, depth, pts3, pts2, _ = _scene(n=n, corrupt=test_idx)
    res = align_relative_depth_to_metric(depth, cam, pts3, pts2)
    assert not res.is_aligned and "POOR_HELD_OUT_FIT" in res.rejection_reason and "training R^2" in res.rejection_reason
    assert res.r_squared is not None and res.r_squared > 0.9  # the training fit alone looked excellent
    assert res.held_out_relative_rmse > res.max_held_out_relative_rmse and res.aligned_depth_map is None


def test_too_few_held_out_points_means_the_fit_cannot_be_validated():
    res, _ = _align(None, min_inliers=15, train_ratio=0.99, scene={"n": 20})
    assert not res.is_aligned and "INSUFFICIENT_HELD_OUT" in res.rejection_reason


def test_per_region_residuals_are_recorded_and_unvalidated_regions_are_named():
    res, _ = _align(None)
    regions = {r["region"]: r for r in res.region_residuals}
    assert set(regions) == {"top_left", "top_right", "bottom_left", "bottom_right"}
    assert all(r["status"] in ("validated", "unvalidated_too_few_points", "local_fit_failure") for r in regions.values())
    assert sum(r["n"] for r in regions.values()) == res.point_count
    assert res.to_dict()["region_residuals"]


def _damaged(factor, gate):
    cam, depth, pts3, pts2, _ = _scene(n=80)
    for i in [i for i, (u, v) in enumerate(pts2) if u < 320 and v < 240][:6]:  # damage ONLY top-left correspondences
        u, v = pts2[i]
        depth[int(round(v)), int(round(u))] *= factor
    return align_relative_depth_to_metric(depth, cam, pts3, pts2, max_held_out_rel_rmse=gate)


def test_a_clean_scene_validates_every_region_leave_one_out():
    res, _ = _align(None)
    assert all(r["status"] == "validated" and r["leave_region_out_rel_rmse"] < 1e-3 for r in res.region_residuals)


def test_good_on_average_but_wrong_in_one_region_is_refused_with_the_region_evidence():
    ok = _damaged(1.6, 0.25)
    assert ok.is_aligned and ok.held_out_relative_rmse < 0.15  # the GLOBAL held-out statistic alone would accept this fit
    res = _damaged(1.6, 0.15)  # same data, a stricter bound: the regional (leave-one-region-out) check now refuses it
    assert not res.is_aligned and "LOCAL_FIT_FAILURE" in res.rejection_reason and "top_left" in res.rejection_reason
    assert res.held_out_relative_rmse < 0.15  # i.e. the global gate passed; the refusal came from the spatial validation
    status = {r["region"]: r["status"] for r in res.region_residuals}
    assert status["top_left"] == "local_fit_failure" and res.aligned_depth_map is None  # the per-region evidence is attached
