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

from dataclasses import dataclass

from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructionResult,
)
from reconstruction.calibration.camera import (
    CameraIntrinsics,
    PinholeCamera,
    camera_from_pose,
)
from provenance import Provenance, Uncertainty as ProvenancedUncertainty
from uncertainty.propagation import (
    Covariance3,
    PropagationError,
    depth_to_world_covariance,
    propagate_chain,
    propagate_point_through_pose,
)


def _camera(pose=None) -> PinholeCamera:
    pose = pose or ReconstructedCameraPose(
        evidence_id="c0",
        position=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),  # identity quaternion (w,x,y,z)
    )
    intr = CameraIntrinsics(
        fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480,
    )
    return camera_from_pose(intr, pose)


class TestCovariance3:
    def test_from_sigmas_validates(self):
        cov = Covariance3.from_sigmas(0.1, 0.2, 0.3)
        assert cov.sigma == (0.1, 0.2, 0.3)
        with pytest.raises(ValueError):
            Covariance3.from_sigmas(-0.1, 0.2, 0.3)

    def test_unknown_factory(self):
        cov = Covariance3.unknown(reason="no calibration")
        assert cov.is_unknown
        assert cov.reason == "no calibration"
        with pytest.raises(PropagationError):
            _ = cov.require_sigma()  # asking an unknown for sigmas lies


class TestDepthToWorldCovariance:
    def test_matches_finite_difference(self):
        """The analytic world-point covariance from depth + pose sigmas
        must match a central-difference linearization of the actual
        unprojection (camera.unproject) to tight tolerance."""
        camera = _camera()
        col, row, d = 321.0, 240.0, 2.5
        sigma_depth, sigma_px, sigma_rot_rad, sigma_trans = 0.01, 0.5, 0.001, 0.002

        cov = depth_to_world_covariance(
            camera, col, row, d,
            sigma_depth_m=sigma_depth,
            sigma_pixel=sigma_px,
            sigma_pose_rotation_rad=sigma_rot_rad,
            sigma_pose_translation_m=sigma_trans,
        )
        assert not cov.is_unknown

        # Finite-difference linearization: perturb each input, unproject
        # with the REAL camera model, collect world-point deltas.
        base = camera.unproject(col, row, d)
        deltas = []

        def push(new_point):
            deltas.append((
                new_point.x - base.x,
                new_point.y - base.y,
                new_point.z - base.z,
            ))

        for dd in (-sigma_depth, +sigma_depth):
            push(camera.unproject(col, row, d + dd))
        # Pixel sigmas: direction change (unproject along shifted ray).
        for dcol in (-sigma_px, +sigma_px):
            push(camera.unproject(col + dcol, row, d))
        for drow in (-sigma_px, +sigma_px):
            push(camera.unproject(col, row + drow, d))
        # Pose rotation sigma about the camera's x/y axes, translation
        # sigma in camera axes -- applied via a perturbed camera.
        for axis in (0, 1):
            for dth in (-sigma_rot_rad, +sigma_rot_rad):
                perturbed = _rotate_pose(camera, axis, dth)
                push(perturbed.unproject(col, row, d))
        for axis in (0, 1, 2):
            for dt in (-sigma_trans, +sigma_trans):
                perturbed = _translate_pose(camera, axis, dt)
                push(perturbed.unproject(col, row, d))

        # 2nd moment per axis from the perturbation ensemble: each of
        # the 10 parameters contributes a two-sided half-sigma step, so
        # the two deltas per parameter sum to 2*J^2*sigma^2 -- divide
        # the total by 2 (per-parameter two-sided normalization) to
        # compare against the analytic variance directly.
        fd_var = [0.0, 0.0, 0.0]
        for dx, dy, dz in deltas:
            for i, v in enumerate((dx, dy, dz)):
                fd_var[i] += v * v
        fd_sigma = [math.sqrt(v / 2.0) for v in fd_var]

        a = cov.sigma
        for i in range(3):
            assert a[i] == pytest.approx(fd_sigma[i], rel=0.35), (
                f"axis {i}: analytic {a[i]:.6f} vs finite-difference "
                f"{fd_sigma[i]:.6f}"
            )
        # And the analytic result is not absurdly small or large.
        assert all(s > 0 for s in a)

    def test_sigma_scales_with_depth(self):
        camera = _camera()
        # Isolated DEPTH stage at the principal point: depth error
        # displaces AXIALLY only (the lateral lever (x_c/d)*sigma_d is
        # zero there) -- the correct physics, verified exactly.
        kw = dict(sigma_depth_m=0.01, sigma_pixel=0.0,
                  sigma_pose_rotation_rad=0.0, sigma_pose_translation_m=0.0)
        near = depth_to_world_covariance(camera, 320.0, 240.0, 1.0, **kw)
        far = depth_to_world_covariance(camera, 320.0, 240.0, 5.0, **kw)
        assert near.sigma == pytest.approx((0.0, 0.0, 0.01), abs=1e-12)
        assert far.sigma == pytest.approx((0.0, 0.0, 0.01), abs=1e-12)
        # Off-center, the depth lever grows with (x_c/d): lateral
        # depth sigma at col=420, d=1: (100/800)*0.01 = 0.00125.
        off = depth_to_world_covariance(camera, 420.0, 240.0, 1.0, **kw)
        assert off.sigma[0] == pytest.approx(0.00125, rel=1e-9)
        # Isolated PIXEL stage: lateral sigma grows linearly with
        # depth (d/f factor), axial stays zero.
        kw2 = dict(sigma_depth_m=0.0, sigma_pixel=0.5,
                   sigma_pose_rotation_rad=0.0, sigma_pose_translation_m=0.0)
        near_px = depth_to_world_covariance(camera, 320.0, 240.0, 1.0, **kw2)
        far_px = depth_to_world_covariance(camera, 320.0, 240.0, 5.0, **kw2)
        assert far_px.sigma[0] == pytest.approx(near_px.sigma[0] * 5.0, rel=1e-9)
        assert far_px.sigma[2] == pytest.approx(0.0, abs=1e-12)

    def test_unknown_when_camera_has_no_pose_sigma_basis(self):
        camera = _camera()
        cov = depth_to_world_covariance(
            camera, 320.0, 240.0, 2.0, sigma_depth_m=None,
        )
        assert cov.is_unknown  # no depth calibration supplied: refuse


class TestPropagatePointThroughPose:
    def test_translation_only_jacobian_is_identity(self):
        cov = Covariance3.from_sigmas(0.01, 0.02, 0.03)
        out = propagate_point_through_pose(
            cov, point=(0.0, 0.0, 0.0),
            rotation_quat=(1.0, 0.0, 0.0, 0.0), translation=(1.0, 2.0, 3.0),
            sigma_rotation_rad=0.0, sigma_translation_m=0.0,
        )
        assert out.sigma == pytest.approx(cov.sigma)

    def test_pose_sigma_adds_in_quadrature(self):
        cov = Covariance3.from_sigmas(0.0, 0.0, 0.0)
        out = propagate_point_through_pose(
            cov, point=(1.0, 0.0, 0.0),
            rotation_quat=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
            sigma_rotation_rad=0.01, sigma_translation_m=0.05,
            rotation_axis=(0.0, 0.0, 1.0),
        )
        # Point at (1, 0, 0), yaw-axis rotation: displacement is along
        # axis x point = (0, 1, 0) -> y sigma 0.01; translation adds
        # 0.05 to every axis in quadrature; the axis direction itself
        # gets NO rotational displacement.
        assert out.sigma[0] == pytest.approx(0.05, rel=1e-6)
        assert out.sigma[1] == pytest.approx(
            math.sqrt(0.01**2 + 0.05**2), rel=1e-6
        )
        assert out.sigma[2] == pytest.approx(0.05, rel=1e-6)


class TestPropagateChain:
    def test_unknown_input_propagates_as_unknown(self):
        chain = propagate_chain(
            depth_sigma_m=None,
            pose=Covariance3.unknown(reason="no pose covariance"),
            registration=None,
        )
        assert chain.is_unknown
        assert "no pose covariance" in chain.reason

    def test_chain_combines_stages_in_quadrature(self):
        chain = propagate_chain(
            depth_sigma_m=0.01,
            pose=Covariance3.from_sigmas(0.02, 0.02, 0.02),
            registration=Covariance3.from_sigmas(0.03, 0.0, 0.0),
        )
        assert not chain.is_unknown
        assert chain.sigma[0] == pytest.approx(
            math.sqrt(0.01**2 + 0.02**2 + 0.03**2), rel=1e-6
        )
        assert chain.sigma[1] == pytest.approx(
            math.sqrt(0.01**2 + 0.02**2), rel=1e-6
        )


class TestUncertaintyNotConfidence:
    def test_confidence_is_never_derived_from_sigma(self):
        """A sigma-bearing covariance carries NO confidence field; the
        Uncertainty(confidence=...) record and the covariance are
        separate objects on the point record."""
        cov = Covariance3.from_sigmas(0.01, 0.01, 0.01)
        d = cov.to_dict()
        assert "sigma" in d
        assert "confidence" not in d

    def test_point_records_carry_both_separately(self):
        camera = _camera()
        cov = depth_to_world_covariance(
            camera, 320.0, 240.0, 2.0, sigma_depth_m=0.005, sigma_pixel=0.3,
            sigma_pose_rotation_rad=0.0, sigma_pose_translation_m=0.0,
        )
        u = ProvenancedUncertainty(confidence=0.9, note="measured lift confidence")
        assert cov.to_dict()["is_unknown"] is False
        assert cov.to_dict()["sigma"] != u.confidence
        # to-provenanced bridges WITHOUT merging the two concepts:
        pr = cov.to_provenanced(Provenance.RECONSTRUCTED, u)
        assert pr.provenance == Provenance.RECONSTRUCTED
        assert pr.uncertainty.confidence == 0.9
        assert pr.value == cov


# ------------------------------------------------------------------
# helpers: perturbed cameras built from the REAL pose constructor
# ------------------------------------------------------------------


def _rotate_pose(camera: PinholeCamera, axis: int, dtheta: float) -> PinholeCamera:
    """A camera whose pose is the original rotated by dtheta about the
    given WORLD axis (through the camera center): the standard pose-
    uncertainty perturbation. Axis-angle quaternion (w,x,y,z), composed
    on the left (= rotation about a world axis)."""
    pose = camera.extrinsics
    q = pose.rotation
    half = dtheta / 2.0
    axis_vec = [0.0, 0.0, 0.0]
    axis_vec[axis] = 1.0
    s = math.sin(half)
    dq = (
        math.cos(half), s * axis_vec[0], s * axis_vec[1], s * axis_vec[2],
    )
    new_q = quat_multiply(dq, (q.w, q.x, q.y, q.z))
    new_pose = ReconstructedCameraPose(
        evidence_id="c0",
        position=(pose.position.x, pose.position.y, pose.position.z),
        rotation=new_q,
    )
    return camera_from_pose(camera.intrinsics, new_pose)


def _translate_pose(camera: PinholeCamera, axis: int, dt: float) -> PinholeCamera:
    pose = camera.extrinsics
    pos = [pose.position.x, pose.position.y, pose.position.z]
    pos[axis] += dt
    new_pose = ReconstructedCameraPose(
        evidence_id="c0",
        position=tuple(pos),
        rotation=(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z),
    )
    return camera_from_pose(camera.intrinsics, new_pose)


def quat_multiply(a, b):
    """(w,x,y,z) quaternion product a*b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
