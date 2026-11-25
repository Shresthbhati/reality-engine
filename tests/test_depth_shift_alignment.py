"""Relative -> metric depth alignment: a scale-only fit (z = s/rel) is the old model; adding a shift (z = s/rel + t) is used only when it
measurably helps and stays physical over the whole map."""
from __future__ import annotations

import numpy as np

from engine.math import Quat, Vec3
from perception.depth.interface import DepthMap
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from reconstruction.depth_to_points import metricize_relative_depth

W, H = 64, 48


def _camera():
    return PinholeCamera(intrinsics=CameraIntrinsics(fx=60.0, fy=60.0, cx=W / 2, cy=H / 2, width=W, height=H),
                         extrinsics=CameraExtrinsics(position=Vec3.zero(), rotation=Quat.identity()))


def _scene(s, t, extra_far_rel=None):
    """True depth ramp 2..5 m across the image; relative depth rel = s / (z - t); sparse points sampled on a grid at the true depth."""
    cam = _camera()
    z = 2.0 + 3.0 * (np.arange(W)[None, :] / (W - 1)) * np.ones((H, 1))
    rel = s / (z - t)
    if extra_far_rel is not None:
        rel[:, :6] = extra_far_rel          # a region the sparse points never touch
    pts = []
    for v in range(4, H - 4, 4):
        for u in range(10, W - 4, 3):
            p = cam.unproject(u + 0.5, v + 0.5, float(z[v, u]))
            pts.append((p.x, p.y, p.z))
    dm = DepthMap(evidence_id="e", width=W, height=H, values=rel.tolist(), unit="relative",
                  quantity="inverse_depth")
    return dm, cam, np.array(pts), z


def test_the_shift_is_off_by_default_it_measured_worse_for_objects():
    dm, cam, pts, z = _scene(s=1.5, t=0.8)
    _, al = metricize_relative_depth(dm, cam, pts)
    assert al.model == "scale" and al.shift == 0.0


def test_a_known_scale_and_shift_is_recovered():
    dm, cam, pts, z = _scene(s=1.5, t=0.8)
    metric, al = metricize_relative_depth(dm, cam, pts, use_shift=True)
    assert al.model == "scale+shift" and abs(al.scale - 1.5) < 0.05 and abs(al.shift - 0.8) < 0.05
    assert np.median(np.abs(np.asarray(metric.values) - z)) < 0.02


def test_scale_only_data_stays_scale_only():
    dm, cam, pts, z = _scene(s=1.5, t=0.0)
    metric, al = metricize_relative_depth(dm, cam, pts, use_shift=True)
    assert al.model == "scale" and al.shift == 0.0
    assert np.median(np.abs(np.asarray(metric.values) - z)) < 0.02


def test_the_shift_improves_on_the_old_fit_when_there_is_a_real_offset():
    dm, cam, pts, z = _scene(s=1.5, t=0.8)
    _, with_shift = metricize_relative_depth(dm, cam, pts, use_shift=True)
    _, old = metricize_relative_depth(dm, cam, pts)
    assert old.model == "scale" and with_shift.residual_median_m < 0.25 * old.residual_median_m


def test_a_shift_that_would_make_depth_non_positive_anywhere_is_refused():
    # the sparse points fit s/rel + t with t < 0 perfectly, but a region of the map has a huge rel -> s/rel + t < 0 there
    dm, cam, pts, z = _scene(s=1.5, t=-0.5, extra_far_rel=500.0)
    metric, al = metricize_relative_depth(dm, cam, pts, use_shift=True)
    assert al.model == "scale"                                  # fell back to the one-parameter fit, on the record
    assert float(np.min(np.asarray(metric.values))) > 0.0
