"""Uncertainty propagation (P10-02) — analytic known-answer tests.

Constitution discipline: every test verifies MEASURED/analytic behavior,
not implementation self-consistency. The analytic expectations are
hand-derived in comments (spec: docs/future/uncertainty/
UNCERTAINTY_PROPAGATION.md acceptance criteria):

- known-input covariance through scale/transform chains matches analytic
  results within tolerance;
- UNKNOWN input -> UNKNOWN output for EVERY operator;
- uncertainty is never silently zero, and confidence is never a
  substitute for a standard deviation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.math import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform
from uncertainty import (
    UNKNOWN,
    Uncertain,
    compose_pose_covariances,
    difference,
    linear_propagate,
    rotate_covariance,
    scale,
    sum as u_sum,
    transform_point_covariance,
)


def _quat_axis_angle(ax: Vec3, angle: float) -> Quat:
    """Half-angle axis-angle -> unit quaternion (the repo's Quat has no
    from_axis_angle; this is the standard construction)."""
    n = math.sqrt(ax.x ** 2 + ax.y ** 2 + ax.z ** 2)
    s = math.sin(angle / 2.0) / n
    return Quat(math.cos(angle / 2.0), ax.x * s, ax.y * s, ax.z * s).normalized()


def _quat_to_matrix(q: Quat) -> np.ndarray:
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class TestUncertainRepresentation:
    def test_unknown_basis_carries_no_sigma(self):
        u = Uncertain.unknown()
        assert u.basis == UNKNOWN
        assert u.sigma is None

    def test_roundtrip_preserves_value_sigma_basis(self):
        u = Uncertain(value=1.25, sigma=0.05, basis="measured")
        assert Uncertain.from_dict(u.to_dict()) == u

    def test_bad_sigma_rejected(self):
        with pytest.raises(ValueError):
            Uncertain(value=1.0, sigma=-0.1, basis="measured")
        with pytest.raises(ValueError):
            Uncertain(value=1.0, sigma=math.nan, basis="measured")

    def test_bad_basis_rejected(self):
        with pytest.raises(ValueError):
            Uncertain(value=1.0, sigma=0.1, basis="guessed")

    def test_missing_sigma_never_presented_as_zero(self):
        # A value with no spread estimate reports None, never 0.0
        # (zero would claim certainty that was never measured).
        u = Uncertain(value=1.0, sigma=None, basis="estimated")
        assert u.sigma is None
        assert u.to_dict()["sigma"] is None


class TestSumDifferenceScale:
    def test_sum_variance_adds(self):
        # a ~ sigma 1, b ~ sigma 2, independent -> sigma = sqrt(5).
        out = u_sum(Uncertain(0.0, 1.0, "measured"), Uncertain(0.0, 2.0, "measured"))
        assert out.sigma == pytest.approx(math.sqrt(5.0), rel=1e-12)

    def test_sum_with_correlation(self):
        # y = a + b: var(y) = var(a) + var(b) + 2cov(a,b)
        #             = 1 + 4 + 2*0.5 = 6.
        out = u_sum(
            Uncertain(0.0, 1.0, "measured"),
            Uncertain(0.0, 2.0, "measured"),
            covariance=0.5,
        )
        assert out.sigma == pytest.approx(math.sqrt(6.0), rel=1e-12)

    def test_difference_variance_adds(self):
        out = difference(Uncertain(5.0, 1.0, "measured"), Uncertain(2.0, 2.0, "measured"))
        assert out.value == pytest.approx(3.0)
        assert out.sigma == pytest.approx(math.sqrt(5.0), rel=1e-12)

    def test_scale_preserves_relative_uncertainty(self):
        # a = 10 +/- 1 (10%), k = 3 -> 30 +/- 3 (still 10%).
        a = Uncertain(10.0, 1.0, "measured")
        out = scale(a, 3.0)
        assert out.value == pytest.approx(30.0)
        assert out.sigma == pytest.approx(3.0)
        assert out.sigma / out.value == pytest.approx(a.sigma / a.value)

    def test_scale_by_zero_annihilates_uncertainty(self):
        out = scale(Uncertain(10.0, 1.0, "measured"), 0.0)
        assert out.value == 0.0
        assert out.sigma == 0.0

    def test_negative_scale_flips_value_not_sigma(self):
        out = scale(Uncertain(10.0, 1.0, "measured"), -2.0)
        assert out.value == pytest.approx(-20.0)
        assert out.sigma == pytest.approx(2.0)


class TestLinearPropagate:
    def test_diagonal_jacobian_matches_variance_algebra(self):
        # y = [1, 1] . x, x ~ diag(1, 4) -> var(y) = 5 (same as sum).
        cov_y = linear_propagate(np.array([[1.0, 1.0]]), np.diag([1.0, 4.0]))
        assert cov_y[0, 0] == pytest.approx(5.0, rel=1e-12)

    def test_rotation_jacobian_known_answer(self):
        # y = R x, R = 90-degree z rotation, cov = diag(1,2,3):
        # the x/y axes swap -> diag(2,1,3).
        r = _quat_to_matrix(_quat_axis_angle(Vec3(0, 0, 1), math.pi / 2))
        cov_y = rotate_covariance(np.diag([1.0, 2.0, 3.0]), r)
        assert np.allclose(cov_y, np.diag([2.0, 1.0, 3.0]), atol=1e-12)


class TestPoseComposition:
    def test_pure_translation_composition_is_analytic(self):
        # Both rotations exact (zero rotation covariance); t1 ~ 0.1 and
        # t2 ~ 0.2 isotropic. Composition translation = t1 + R1 t2, so
        # var(t_c) = var(t1) + var(t2) = 0.05 per axis.
        s1 = np.zeros((6, 6))
        s1[3:, 3:] = np.eye(3) * (0.1 ** 2)
        s2 = np.zeros((6, 6))
        s2[3:, 3:] = np.eye(3) * (0.2 ** 2)
        t1 = RigidTransform("a", "b", Quat.identity(), Vec3(0.0, 0.0, 0.0))
        t2 = RigidTransform("b", "c", Quat.identity(), Vec3(1.0, 0.0, 0.0))
        cov_c = compose_pose_covariances(t1, s1, t2, s2)
        expected = 0.1 ** 2 + 0.2 ** 2
        for i in range(3):
            assert cov_c[3 + i, 3 + i] == pytest.approx(expected, rel=1e-6)
        assert np.allclose(cov_c[:3, :3], 0.0, atol=1e-15)

    def test_rotated_inner_translation_swaps_variance_axes(self):
        # Repo convention: t_c = R2 t1 + t2. T1 carries x-axis
        # translation variance 0.2^2; T2 is an exact 90-degree z
        # rotation. R2 maps x-hat to y-hat, so t1's x-variance lands on
        # the output y axis.
        s1 = np.zeros((6, 6))
        s1[3, 3] = 0.2 ** 2
        s2 = np.zeros((6, 6))
        q = _quat_axis_angle(Vec3(0, 0, 1), math.pi / 2)
        t1 = RigidTransform("a", "b", Quat.identity(), Vec3(1.0, 0.0, 0.0))
        t2 = RigidTransform("b", "c", q, Vec3(0.0, 0.0, 0.0))
        cov_c = compose_pose_covariances(t1, s1, t2, s2)
        assert cov_c[3, 3] == pytest.approx(0.0, abs=1e-12)
        assert cov_c[4, 4] == pytest.approx(0.2 ** 2, rel=1e-6)

    def test_unknown_input_pose_propagates_unknown(self):
        t1 = RigidTransform("a", "b", Quat.identity(), Vec3(0, 0, 0))
        t2 = RigidTransform("b", "c", Quat.identity(), Vec3(1, 0, 0))
        cov = compose_pose_covariances(t1, None, t2, np.eye(6) * 0.1)
        assert cov is None

    def test_frame_mismatch_refused(self):
        t1 = RigidTransform("a", "b", Quat.identity(), Vec3(0, 0, 0))
        t2 = RigidTransform("x", "c", Quat.identity(), Vec3(1, 0, 0))
        with pytest.raises(ValueError):
            compose_pose_covariances(t1, np.eye(6) * 0.1, t2, np.eye(6) * 0.1)


def _perturbed_composition(t1, d1, t2, d2) -> np.ndarray:
    """Apply small perturbations (3 rotation-vector, 3 translation) to
    both transforms, compose, and return the 6-vector (rotvec, trans).
    This is the measurement model the Jacobian covariance must predict."""

    def perturbed(t: RigidTransform, d: np.ndarray) -> RigidTransform:
        axis_len = float(np.linalg.norm(d[:3]))
        if axis_len > 0:
            dq = _quat_axis_angle(Vec3(d[0], d[1], d[2]), axis_len)
        else:
            dq = Quat.identity()
        return RigidTransform(
            t.from_frame, t.to_frame,
            dq.multiply(t.rotation).normalized(),
            Vec3(t.translation.x + d[3], t.translation.y + d[4],
                 t.translation.z + d[5]),
        )

    tc = perturbed(t1, d1).compose(perturbed(t2, d2))
    w, x, y, z = tc.rotation.w, tc.rotation.x, tc.rotation.y, tc.rotation.z
    angle = 2.0 * math.acos(max(-1.0, min(1.0, w)))
    s = math.sqrt(max(1e-18, 1.0 - w * w))
    rotvec = np.array([x, y, z]) / s * angle
    return np.concatenate(
        [rotvec, [tc.translation.x, tc.translation.y, tc.translation.z]]
    )


class TestMonteCarloAgreement:
    def test_composition_matches_monte_carlo(self):
        # Beyond special cases: the first-order Jacobian covariance must
        # agree with the empirical covariance of the SAME composition
        # under the same input noise (tolerance reflects first-order +
        # finite-sample error, 15% on variances).
        rng = np.random.default_rng(42)
        s1 = np.diag([0.01, 0.01, 0.01, 0.0025, 0.0025, 0.0025])
        s2 = np.diag([0.04, 0.01, 0.01, 0.01, 0.01, 0.01])
        q1 = _quat_axis_angle(Vec3(0, 0, 1), 0.7)
        q2 = _quat_axis_angle(Vec3(0, 1, 0), 0.4)
        t1 = RigidTransform("a", "b", q1, Vec3(1.0, 2.0, 3.0))
        t2 = RigidTransform("b", "c", q2, Vec3(0.5, -1.0, 2.0))
        cov_analytic = compose_pose_covariances(t1, s1, t2, s2)
        n = 100_000
        d1 = rng.multivariate_normal(np.zeros(6), s1, size=n)
        d2 = rng.multivariate_normal(np.zeros(6), s2, size=n)
        samples = np.empty((n, 6))
        for k in range(n):
            samples[k] = _perturbed_composition(t1, d1[k], t2, d2[k])
        cov_mc = np.cov(samples.T)
        for i in range(6):
            assert cov_analytic[i, i] == pytest.approx(cov_mc[i, i], rel=0.15)


class TestPointTransform:
    def test_transform_point_covariance_rotates_blocks(self):
        cov = np.diag([1.0, 2.0, 3.0])
        q = _quat_axis_angle(Vec3(0, 0, 1), math.pi / 2)
        t = RigidTransform("a", "b", q, Vec3(10, 20, 30))
        out = transform_point_covariance(cov, t)
        assert np.allclose(out, np.diag([2.0, 1.0, 3.0]), atol=1e-12)
        # translation must not affect a point covariance
        assert np.allclose(
            out, rotate_covariance(cov, _quat_to_matrix(q)), atol=1e-12
        )

    def test_chain_returns_original_covariance(self):
        # rotate by +90z then -90z returns the original covariance.
        cov = np.diag([1.0, 2.0, 3.0])
        q1 = _quat_axis_angle(Vec3(0, 0, 1), math.pi / 2)
        q2 = _quat_axis_angle(Vec3(0, 0, 1), -math.pi / 2)
        t1 = RigidTransform("a", "b", q1, Vec3(0, 0, 0))
        t2 = RigidTransform("b", "c", q2, Vec3(0, 0, 0))
        out = transform_point_covariance(
            transform_point_covariance(cov, t1), t2
        )
        assert np.allclose(out, cov, atol=1e-9)

    def test_unknown_point_covariance_propagates_unknown(self):
        q = _quat_axis_angle(Vec3(0, 0, 1), 1.0)
        t = RigidTransform("a", "b", q, Vec3(0, 0, 0))
        assert transform_point_covariance(None, t) is None


class TestUnknownPropagation:
    def test_sum_unknown(self):
        out = u_sum(Uncertain.unknown(), Uncertain(0.0, 1.0, "measured"))
        assert out.basis == UNKNOWN and out.sigma is None

    def test_difference_unknown(self):
        out = difference(Uncertain(1.0, 1.0, "measured"), Uncertain.unknown())
        assert out.basis == UNKNOWN and out.sigma is None

    def test_scale_unknown(self):
        out = scale(Uncertain.unknown(), 2.0)
        assert out.basis == UNKNOWN and out.sigma is None

    def test_linear_unknown(self):
        assert linear_propagate(np.eye(2), None) is None

    def test_rotate_unknown(self):
        assert rotate_covariance(None, np.eye(3)) is None

    def test_unknown_sigma_is_contagious_not_zero(self):
        # The spec's silent-lie rule: an output derived from an input
        # with unknown spread must not present sigma=0 as if nothing
        # was uncertain.
        out = u_sum(
            Uncertain(1.0, None, "estimated"), Uncertain(1.0, 1.0, "measured")
        )
        assert out.sigma is None
        assert out.basis == UNKNOWN


class TestDeterminismAndBookkeeping:
    def test_repeat_calls_identical(self):
        a = Uncertain(1.0, 0.1, "measured")
        b = Uncertain(2.0, 0.2, "measured")
        assert u_sum(a, b) == u_sum(a, b)

    def test_derived_basis_recorded(self):
        out = u_sum(Uncertain(1.0, 0.1, "measured"), Uncertain(2.0, 0.2, "measured"))
        assert out.basis == "derived"

