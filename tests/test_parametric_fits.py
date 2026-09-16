"""Tests for architectural parametric perception (P7-03 expansion:
directive sections 12-14 -- arch/column/dome fitting from real segment
points, deterministic, honest about evidence and confidence).

Spec rules under test:
  - Fits run on REAL points; residuals are MEASURED (rms/max), never
    fabricated; nothing is guessed where geometry does not support it.
  - A segment with insufficient/degenerate support must DECLINE
    (FitRefused), never best-effort a garbage model that downstream
    code would treat as a real column/dome/arch.
  - Axis orientation is canonicalized (positive up-dot) so two fits of
    the same column can never differ by sign.
  - Confidence is a measured function of fit quality, documented and
    deterministic -- not a made-up score.
  - Everything is closed-form/deterministic: the same points always
    produce byte-identical fit values.
"""

from __future__ import annotations

import math

import pytest

from perception.architecture.parametric import (
    CircleFit,
    CylinderFit,
    FitRefused,
    SphereFit,
    fit_circle,
    fit_cylinder,
    fit_sphere,
)

UP = (0.0, 0.0, 1.0)


def _cylinder_points(axis, origin, radius, ts, angles, wobble=0.0):
    """Deterministic points on (or wobbling off) a cylinder surface."""
    # Orthonormal basis perpendicular to the axis.
    helper = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = _cross(axis, helper)
    u = _norm(u)
    v = _cross(axis, u)
    pts = []
    for t in ts:
        for k, ang in enumerate(angles):
            r = radius * (1.0 + wobble * math.sin(7.0 * ang + t))
            p = [
                origin[i] + axis[i] * t + (u[i] * math.cos(ang) + v[i] * math.sin(ang)) * r
                for i in range(3)
            ]
            pts.append(tuple(p))
    return pts


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------------
# Cylinder (column) fitting
# ------------------------------------------------------------------


class TestCylinderFit:
    def test_perfect_vertical_cylinder_known_answer(self):
        pts = _cylinder_points(
            (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.3,
            ts=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            angles=[i * math.pi / 8 for i in range(12)],
        )
        fit = fit_cylinder(pts, up=UP)
        assert isinstance(fit, CylinderFit)
        assert abs(fit.radius_m - 0.3) < 1e-6
        assert abs(_dot(fit.axis, UP) - 1.0) < 1e-9
        # The axis LINE passes through the origin: axis_point is some
        # point on that line, so its perpendicular offset from the
        # origin must vanish (axial component is arbitrary).
        d = fit.axis_point
        perp = tuple(d[i] - _dot(d, fit.axis) * fit.axis[i] for i in range(3))
        assert math.sqrt(_dot(perp, perp)) < 1e-6
        assert fit.rms_residual_m < 1e-6
        assert fit.confidence > 0.99

    def test_residual_is_measured_not_fabricated(self):
        clean = fit_cylinder(
            _cylinder_points(
                (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.3,
                ts=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
                angles=[i * math.pi / 8 for i in range(12)],
            ),
            up=UP,
        )
        pts = _cylinder_points(
            (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.3,
            ts=[-1.0, 0.0, 1.0, 2.0],
            angles=[i * math.pi / 6 for i in range(12)],
            wobble=0.03,
        )
        fit = fit_cylinder(pts, up=UP)
        assert 0.0 < fit.rms_residual_m < 0.05
        assert fit.max_residual_m >= fit.rms_residual_m
        # Measured quality lowers confidence relative to the same
        # geometry without the wobble -- agreement is evidence.
        assert fit.confidence < clean.confidence

    def test_axis_sign_canonicalized_toward_up(self):
        # Same cylinder parameterized "downward": the fit must report
        # the same canonical axis either way.
        pts_a = _cylinder_points((0.0, 0.0, 1.0), (0, 0, 0), 0.3,
                                 ts=[0, 1, 2], angles=[0, 1, 2, 3, 4, 5])
        pts_b = _cylinder_points((0.0, 0.0, -1.0), (0, 0, 2), 0.3,
                                 ts=[0, 1, 2], angles=[0, 1, 2, 3, 4, 5])
        fit_a = fit_cylinder(pts_a, up=UP)
        fit_b = fit_cylinder(pts_b, up=UP)
        assert _dot(fit_a.axis, UP) > 0
        assert _dot(fit_b.axis, UP) > 0
        assert abs(_dot(fit_a.axis, fit_b.axis) - 1.0) < 1e-9

    def test_tilted_cylinder_axis_recovered(self):
        axis = _norm((1.0, 0.0, 1.0))
        pts = _cylinder_points(axis, (0, 0, 0), 0.25,
                               ts=[-1.0, 0.0, 1.0, 2.0],
                               angles=[i * math.pi / 6 for i in range(12)])
        fit = fit_cylinder(pts, up=UP)
        assert _dot(fit.axis, axis) > math.cos(math.radians(3.0))
        assert abs(fit.radius_m - 0.25) < 1e-6

    def test_height_extent_measured_along_axis(self):
        pts = _cylinder_points((0.0, 0.0, 1.0), (0, 0, 0), 0.3,
                               ts=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
                               angles=[0.0, 1.0, 2.0, 3.0])
        fit = fit_cylinder(pts, up=UP)
        assert abs((fit.height_max_m - fit.height_min_m) - 2.5) < 1e-6

    def test_too_few_points_refused(self):
        pts = _cylinder_points((0.0, 0.0, 1.0), (0, 0, 0), 0.3,
                               ts=[0.0], angles=[0.0, 1.0, 2.0])
        with pytest.raises(FitRefused) as exc:
            fit_cylinder(pts, up=UP)
        assert "at least" in str(exc.value).lower()

    def test_collinear_points_refused(self):
        pts = [(float(i), 0.0, 0.0) for i in range(10)]
        with pytest.raises(FitRefused):
            fit_cylinder(pts, up=UP)

    def test_radius_under_constrained_refused(self):
        # A flat patch of wall cannot constrain a cylinder: its best
        # "circle" explains only a fraction of the radial structure
        # (rms comparable to the lattice spacing, not to noise). The
        # fit must refuse rather than classify geometric nonsense.
        pts = [(x * 0.01, y * 0.01, 0.0) for x in range(5) for y in range(5)]
        with pytest.raises(FitRefused) as exc:
            fit_cylinder(pts, up=UP)
        assert "constrain" in str(exc.value).lower() or "shell" in str(exc.value).lower()

    def test_deterministic(self):
        pts = _cylinder_points((0.0, 0.0, 1.0), (0, 0, 0), 0.3,
                               ts=[0.0, 1.0, 2.0],
                               angles=[i * math.pi / 6 for i in range(12)])
        a = fit_cylinder(pts, up=UP)
        b = fit_cylinder(pts, up=UP)
        assert a.to_dict() == b.to_dict()

    def test_to_dict_carries_measured_fields(self):
        pts = _cylinder_points((0.0, 0.0, 1.0), (0, 0, 0), 0.3,
                               ts=[0.0, 1.0, 2.0],
                               angles=[i * math.pi / 6 for i in range(12)])
        d = fit_cylinder(pts, up=UP).to_dict()
        for key in ("radius_m", "rms_residual_m", "max_residual_m",
                    "confidence", "n_points", "axis", "axis_point"):
            assert key in d


# ------------------------------------------------------------------
# Sphere (dome) fitting
# ------------------------------------------------------------------


def _sphere_points(center, radius, dirs):
    return [
        tuple(center[i] + radius * d[i] for i in range(3)) for d in dirs
    ]


def _golden_dirs(n, z_min=-1.0):
    """Deterministic quasi-uniform directions on the sphere (golden
    angle spiral), restricted to z >= z_min."""
    phi = math.pi * (3.0 - math.sqrt(5.0))
    dirs = []
    for k in range(n):
        z = 1.0 - 2.0 * (k + 0.5) / n
        if z < z_min:
            continue
        r = math.sqrt(max(0.0, 1.0 - z * z))
        ang = phi * k
        dirs.append((r * math.cos(ang), r * math.sin(ang), z))
    return dirs


class TestSphereFit:
    def test_perfect_sphere_known_answer(self):
        dirs = _golden_dirs(60)
        pts = _sphere_points((1.0, 2.0, 3.0), 0.5, dirs)
        fit = fit_sphere(pts)
        assert abs(fit.radius_m - 0.5) < 1e-6
        assert all(abs(fit.center[i] - (1.0, 2.0, 3.0)[i]) < 1e-6 for i in range(3))
        assert fit.rms_residual_m < 1e-6
        assert fit.confidence > 0.99

    def test_partial_support_dome_like(self):
        # Upper hemisphere only -- a dome seen from below/outside.
        dirs = _golden_dirs(120, z_min=0.0)
        pts = _sphere_points((0.0, 0.0, 0.0), 2.0, dirs)
        fit = fit_sphere(pts)
        assert abs(fit.radius_m - 2.0) < 1e-6
        assert all(abs(c) < 1e-6 for c in fit.center)

    def test_coplanar_points_refused(self):
        pts = [(x, y, 0.0) for x in (0.0, 1.0, 2.0) for y in (0.0, 1.0, 2.0)]
        with pytest.raises(FitRefused):
            fit_sphere(pts)

    def test_too_few_points_refused(self):
        with pytest.raises(FitRefused):
            fit_sphere([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])

    def test_noisy_sphere_measures_residual(self):
        clean = fit_sphere(_sphere_points((0.0, 0.0, 0.0), 1.0, _golden_dirs(60)))
        dirs = [(dx * 1.02, dy * 0.99, dz) for dx, dy, dz in _golden_dirs(60)]
        pts = _sphere_points((0.0, 0.0, 0.0), 1.0, dirs)
        fit = fit_sphere(pts)
        assert 0.0 < fit.rms_residual_m < 0.05
        # Measured quality lowers confidence relative to the clean fit.
        assert fit.confidence < clean.confidence


# ------------------------------------------------------------------
# Circle / arch fitting
# ------------------------------------------------------------------


class TestCircleFit:
    def test_semicircular_arch_known_answer(self):
        # Semicircle radius 1 in the x-y plane (normal +Z), spanning
        # angles 0..pi; the largest gap (the opening) faces -Y.
        pts = [
            (math.cos(a), math.sin(a), 0.0)
            for a in [i * math.pi / 24 for i in range(25)]
        ]
        fit = fit_circle(pts, normal=(0.0, 0.0, 1.0))
        assert abs(fit.radius_m - 1.0) < 1e-6
        assert all(abs(c) < 1e-6 for c in fit.center)
        assert abs(fit.angular_span_rad - math.pi) < 0.02
        # Opening bisector points toward -Y (the missing half).
        opening = fit.opening_direction
        assert opening[1] < -0.99

    def test_extrusion_measured(self):
        pts = [
            (math.cos(a), math.sin(a), z)
            for a in [i * math.pi / 24 for i in range(25)]
            for z in (-0.2, 0.2)
        ]
        fit = fit_circle(pts, normal=(0.0, 0.0, 1.0))
        assert abs(fit.extrusion_depth_m - 0.4) < 1e-6

    def test_full_ring_span_is_2pi(self):
        n = 32
        pts = [
            (math.cos(i * math.pi / 16), math.sin(i * math.pi / 16), 0.0)
            for i in range(n)
        ]
        fit = fit_circle(pts, normal=(0.0, 0.0, 1.0))
        # Discrete sampling: the largest empty gap between adjacent
        # samples is 2*pi/n, so the measured span of n points is
        # 2*pi*(n-1)/n -- near-2pi is the honest reading of a ring.
        assert abs(fit.angular_span_rad - 2.0 * math.pi * (n - 1) / n) < 0.02
        assert fit.angular_span_rad > 6.0

    def test_off_plane_points_measured_in_residual(self):
        # 5 cm constant off-plane offset: the 3D residual must see it.
        pts = [
            (math.cos(a), math.sin(a), 0.05)
            for a in [i * math.pi / 24 for i in range(25)]
        ]
        fit = fit_circle(pts, normal=(0.0, 0.0, 1.0))
        assert fit.rms_residual_m >= 0.049

    def test_normal_derived_when_not_supplied(self):
        pts = [
            (math.cos(a), math.sin(a), 0.01 * math.sin(3 * a))
            for a in [i * math.pi / 24 for i in range(25)]
        ]
        fit = fit_circle(pts, normal=None)
        assert abs(_dot(fit.plane_normal, UP)) > 0.99

    def test_collinear_points_refused(self):
        pts = [(float(i), 0.0, 0.0) for i in range(8)]
        with pytest.raises(FitRefused):
            fit_circle(pts, normal=(0.0, 0.0, 1.0))

    def test_too_few_points_refused(self):
        with pytest.raises(FitRefused):
            fit_circle([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], normal=(0.0, 0.0, 1.0))

    def test_deterministic(self):
        pts = [
            (math.cos(a), math.sin(a), 0.0)
            for a in [i * math.pi / 24 for i in range(25)]
        ]
        a = fit_circle(pts, normal=(0.0, 0.0, 1.0))
        b = fit_circle(pts, normal=(0.0, 0.0, 1.0))
        assert a.to_dict() == b.to_dict()
