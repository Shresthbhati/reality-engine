"""Tests for LAND-01: single-image curved-structure reconstruction.

Spec rules under test:
  - Fits run on REAL 3D points; residuals are MEASURED, never fabricated.
  - A segment the points cannot support must DECLINE (FitRefused / kind
    == "none"), never best-effort a guessed shape.
  - Single-image scale is RELATIVE unless a trusted baseline is supplied.
  - Confidence is a measured deterministic function of fit quality.
  - Everything is deterministic: same points -> byte-identical output.
  - With no ground truth, the error report honestly says "not measured".
"""

from __future__ import annotations

import math

import pytest

from perception.architecture.landmark import (
    ArchFit,
    DomeFit,
    FitRefused,
    LandmarkErrorReport,
    LandmarkResult,
    TowerFit,
    fit_arch,
    fit_dome,
    fit_landmark,
    fit_tower,
    report_error,
)
UP = (0.0, 0.0, 1.0)


def _sphere_cap_points(center, radius, n_phi, n_theta, polar_max_rad):
    """Deterministic points on a spherical cap (dome) about +Z.

    polar_max_rad: max polar angle from +Z (e.g. pi/2 = hemisphere).
    """
    pts = []
    for i in range(n_phi):
        phi = math.pi * i / max(n_phi - 1, 1)
        if phi > polar_max_rad:
            continue
        for j in range(n_theta):
            theta = 2 * math.pi * j / n_theta
            x = center[0] + radius * math.sin(phi) * math.cos(theta)
            y = center[1] + radius * math.sin(phi) * math.sin(theta)
            z = center[2] + radius * math.cos(phi)
            pts.append((x, y, z))
    return pts


def _circle_arc_points(center, radius, start_ang, end_ang, n):
    """Deterministic points on a circular arc in the XZ plane (vertical)."""
    pts = []
    for i in range(n):
        a = start_ang + (end_ang - start_ang) * i / max(n - 1, 1)
        x = center[0] + radius * math.cos(a)
        z = center[2] + radius * math.sin(a)
        pts.append((x, center[1], z))
    return pts


def _cylinder_points(axis, origin, radius, ts, angles):
    helper = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    ax_, ay_, az_ = axis
    hx, hy, hz = helper
    # u = normalize(cross(axis, helper))
    cx = ay_ * hz - az_ * hy
    cy = az_ * hx - ax_ * hz
    cz = ax_ * hy - ay_ * hx
    nlen = math.sqrt(cx * cx + cy * cy + cz * cz)
    ux, uy, uz = cx / nlen, cy / nlen, cz / nlen
    # v = cross(axis, u)
    vx = ay_ * uz - az_ * uy
    vy = az_ * ux - ax_ * uz
    vz = ax_ * uy - ay_ * ux
    uvec = (ux, uy, uz)
    vvec = (vx, vy, vz)
    pts = []
    for t in ts:
        for ang in angles:
            p = [
                origin[i] + axis[i] * t + (uvec[i] * math.cos(ang) + vvec[i] * math.sin(ang)) * radius
                for i in range(3)
            ]
            pts.append(tuple(p))
    return pts


# ==================================================================
# Dome
# ==================================================================


def test_fit_dome_hemisphere_recovers_radius():
    pts = _sphere_cap_points((0.0, 0.0, 0.0), 5.0, 9, 16, math.pi / 2)
    fit = fit_dome(pts, UP)
    assert fit.n_points == len(pts)
    assert abs(fit.sphere.radius_m - 5.0) < 0.02
    assert 0.45 <= fit.upper_fraction <= 1.0
    assert fit.confidence > 0.5
    assert all(abs(fit.sphere.center[i]) < 0.1 for i in range(3))


def test_fit_dome_deterministic():
    pts = _sphere_cap_points((1.0, 2.0, 3.0), 3.0, 7, 12, math.pi / 2)
    a = fit_dome(pts, UP)
    b = fit_dome(pts, UP)
    assert a.to_dict() == b.to_dict()


def test_fit_dome_refuses_vertical_plane():
    """Points on a flat vertical wall are not a dome (sphere fit fails)."""
    pts = [(0.0, y, z) for y in range(-3, 4) for z in range(0, 7)]
    with pytest.raises(FitRefused):
        fit_dome(pts, UP)


def test_fit_dome_refuses_insufficient_points():
    with pytest.raises(FitRefused):
        fit_dome([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], UP)


def test_fit_dome_refuses_flat_patch():
    """A flat horizontal patch is not a dome."""
    pts = [(x, y, 0.0) for x in range(-2, 3) for y in range(-2, 3)]
    with pytest.raises(FitRefused):
        fit_dome(pts, UP)


# ==================================================================
# Arch
# ==================================================================


def test_fit_arch_semicircle_recovers_span():
    pts = _circle_arc_points((0.0, 0.0, 0.0), 3.0, 0.0, math.pi, 12)
    fit = fit_arch(pts, UP)
    assert fit.n_points == len(pts)
    assert abs(fit.angular_span_rad - math.pi) < 0.15
    assert abs(fit.span_m - 6.0) < 0.2
    assert abs(fit.rise_m - 3.0) < 0.2
    assert fit.confidence > 0.5


def test_fit_arch_deterministic():
    pts = _circle_arc_points((2.0, 1.0, 3.0), 2.5, 0.2, 2.9, 10)
    a = fit_arch(pts, UP)
    b = fit_arch(pts, UP)
    assert a.to_dict() == b.to_dict()


def test_fit_arch_refuses_shallow_arc():
    """An arc spanning only ~17 degrees is not an arch."""
    pts = _circle_arc_points((0.0, 0.0, 0.0), 4.0, 0.0, 0.3, 8)
    with pytest.raises(FitRefused):
        fit_arch(pts, UP)


def test_fit_arch_refuses_insufficient_points():
    with pytest.raises(FitRefused):
        fit_arch([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], UP)


# ==================================================================
# Tower
# ==================================================================


def test_fit_tower_vertical_cylinder_recovers_dims():
    axis = (0.0, 0.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    pts = _cylinder_points(axis, origin, 1.5, [0.0, 2.0, 4.0, 6.0, 8.0], [0, 2, 4, 6])
    fit = fit_tower(pts, UP)
    assert fit.n_points == len(pts)
    assert abs(fit.axis_up_dot - 1.0) < 1e-9
    assert abs(fit.cylinder.radius_m - 1.5) < 0.05
    assert fit.confidence > 0.5


def test_fit_tower_deterministic():
    axis = (0.0, 0.0, 1.0)
    pts = _cylinder_points(axis, (1.0, 1.0, 0.0), 2.0, [0.0, 3.0, 6.0], [0, 3, 6])
    a = fit_tower(pts, UP)
    b = fit_tower(pts, UP)
    assert a.to_dict() == b.to_dict()


def test_fit_tower_refuses_flat_patch():
    """A flat horizontal patch is not a tower (not elongated vertically)."""
    pts = [(x, y, 0.0) for x in range(-3, 4) for y in range(-3, 4)]
    with pytest.raises(FitRefused):
        fit_tower(pts, UP)


def test_fit_tower_refuses_insufficient_points():
    with pytest.raises(FitRefused):
        fit_tower([(0.0, 0.0, 0.0)], UP)


# ==================================================================
# fit_landmark auto-detection + scale
# ==================================================================


def test_fit_landmark_auto_detects_dome():
    pts = _sphere_cap_points((0.0, 0.0, 0.0), 4.0, 8, 14, math.pi / 2)
    res = fit_landmark(pts, UP)
    assert res.kind == "dome"
    assert res.dome is not None
    assert res.arch is None
    assert res.tower is None
    assert res.scale_state == "relative"
    assert res.measured_fact != ""
    assert "dome" in res.measured_fact


def test_fit_landmark_auto_detects_tower():
    axis = (0.0, 0.0, 1.0)
    pts = _cylinder_points(axis, (0.0, 0.0, 0.0), 2.0, [0.0, 5.0, 10.0], [0, 2, 4])
    res = fit_landmark(pts, UP)
    assert res.kind == "tower"
    assert res.tower is not None
    assert "tower" in res.measured_fact


def test_fit_landmark_hint_respected():
    pts = _sphere_cap_points((0.0, 0.0, 0.0), 3.0, 7, 12, math.pi / 2)
    res = fit_landmark(pts, UP, kind_hint="dome")
    assert res.kind == "dome"


def test_fit_landmark_measured_scale_with_baseline():
    pts = _sphere_cap_points((0.0, 0.0, 0.0), 2.0, 7, 12, math.pi / 2)
    res = fit_landmark(pts, UP, kind_hint="dome", baseline_m=10.0)
    assert res.scale_state == "measured"
    assert abs(res.scale_factor - 5.0) < 0.02


def test_fit_landmark_relative_scale_without_baseline():
    pts = _sphere_cap_points((0.0, 0.0, 0.0), 2.0, 7, 12, math.pi / 2)
    res = fit_landmark(pts, UP, kind_hint="dome")
    assert res.scale_state == "relative"
    assert res.scale_factor == 1.0


def test_fit_landmark_none_for_unfit_points():
    """Flat scattered points match no curved landmark -> kind 'none'."""
    pts = [(x, y, 0.0) for x in range(-2, 3) for y in range(-2, 3)]
    res = fit_landmark(pts, UP)
    assert res.kind == "none"
    assert res.refusal_reason is not None
    assert res.refusal_reason != ""


def test_fit_landmark_empty_points():
    res = fit_landmark([], UP)
    assert res.kind == "none"
    assert res.refusal_reason == "no points supplied"


def test_fit_landmark_deterministic():
    pts = _sphere_cap_points((0.0, 0.0, 0.0), 3.0, 7, 12, math.pi / 2)
    a = fit_landmark(pts, UP)
    b = fit_landmark(pts, UP)
    assert a.to_dict() == b.to_dict()


# ==================================================================
# Error report (acceptance criterion support)
# ==================================================================


def test_report_error_no_ground_truth_is_not_measured():
    rep = report_error("dome")
    assert rep.kind == "dome"
    assert rep.radius_error_m is None
    assert rep.height_error_m is None
    assert rep.position_rms_m is None
    assert rep.passes_threshold is None


def test_report_error_with_threshold():
    rep = report_error("arch", position_rms_m=0.02, threshold_m=0.05)
    assert rep.passes_threshold is True
    assert rep.position_rms_m == 0.02
    assert rep.threshold_m == 0.05


def test_report_error_fails_threshold():
    rep = report_error("tower", position_rms_m=0.10, threshold_m=0.05)
    assert rep.passes_threshold is False


def test_report_error_to_dict():
    rep = report_error("dome", radius_error_m=0.03, threshold_m=0.05)
    d = rep.to_dict()
    assert d["kind"] == "dome"
    assert d["radius_error_m"] == 0.03
    assert d["passes_threshold"] is None  # no position_rms_m -> cannot judge


# ==================================================================
# Honesty: measured residuals are recorded, never fabricated
# ==================================================================


def test_dome_fit_records_measured_residuals():
    pts = _sphere_cap_points((0.0, 0.0, 0.0), 4.0, 8, 14, math.pi / 2)
    fit = fit_dome(pts, UP)
    assert fit.sphere.rms_residual_m >= 0.0
    assert fit.sphere.max_residual_m >= 0.0
    assert fit.sphere.n_points == len(pts)


def test_arch_fit_records_measured_residuals():
    pts = _circle_arc_points((0.0, 0.0, 0.0), 3.0, 0.0, math.pi, 12)
    fit = fit_arch(pts, UP)
    assert fit.circle.rms_residual_m >= 0.0
    assert fit.circle.n_points == len(pts)