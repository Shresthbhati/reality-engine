"""Unit tests for Sophus Lie algebra extraction: SO(3) and SE(3)."""

import math
import numpy as np
import pytest

from reconstruction.geometry.lie_algebra import SE3, SO3, hat, vee


def test_hat_and_vee_inversion():
    """Verify that hat and vee operators are exact inverses."""
    v = np.array([1.2, -3.4, 0.7], dtype=np.float64)
    v_hat = hat(v)
    assert v_hat.shape == (3, 3)
    assert np.allclose(v_hat, -v_hat.T)
    v_rec = vee(v_hat)
    assert np.allclose(v, v_rec)


def test_so3_exp_log_roundtrip():
    """Verify SO(3) exp and log round-trip for various rotation angles."""
    # Near zero (small angle test)
    w_small = np.array([1e-8, 2e-8, -1.5e-8])
    so3_small = SO3.exp(w_small)
    assert np.allclose(so3_small.log(), w_small, atol=1e-12)

    # Moderate rotation
    w_mod = np.array([0.3, -0.4, 0.5])
    so3_mod = SO3.exp(w_mod)
    assert np.allclose(so3_mod.log(), w_mod, atol=1e-10)

    # Large rotation (near pi)
    axis = np.array([0.0, 1.0, 0.0])
    w_pi = (math.pi - 1e-4) * axis
    so3_pi = SO3.exp(w_pi)
    assert np.allclose(so3_pi.log(), w_pi, atol=1e-5)


def test_so3_orthogonality_and_inverse():
    """Verify that SO3 matrices are orthogonal (R R^T = I) and inverse equals transpose."""
    w = np.array([0.5, 0.2, -0.7])
    rot = SO3.exp(w)
    r = rot.matrix
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-10)

    rot_inv = rot.inverse()
    assert np.allclose((rot @ rot_inv).matrix, np.eye(3), atol=1e-10)


def test_se3_exp_log_roundtrip():
    """Verify SE(3) exp and log roundtrip on combined translation and rotation."""
    xi = np.array([1.5, -2.0, 0.8, 0.2, -0.4, 0.1], dtype=np.float64)
    se3 = SE3.exp(xi)
    xi_rec = se3.log()
    assert np.allclose(xi, xi_rec, atol=1e-9)


def test_se3_point_transformation_and_inverse():
    """Verify rigid transformation of 3D point cloud and inversion."""
    rot = SO3.exp(np.array([0.0, 0.0, math.pi / 2.0]))  # 90 deg yaw
    trans = np.array([10.0, -5.0, 3.0])
    t = SE3(rot, trans)
    t_inv = t.inverse()

    pts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [3.0, 4.0, 5.0],
    ], dtype=np.float64)

    pts_transformed = t.transform_points(pts)
    pts_restored = t_inv.transform_points(pts_transformed)
    assert np.allclose(pts, pts_restored, atol=1e-10)

    # Check 90 deg around Z for [1, 0, 0] + [10, -5, 3] => [0, 1, 0] + [10, -5, 3] = [10, -4, 3]
    assert np.allclose(pts_transformed[0], [10.0, -4.0, 3.0], atol=1e-9)

    # Verify log/exp consistency with constructed SE(3)
    xi = t.log()
    t_rec = SE3.exp(xi)
    assert np.allclose(t.matrix, t_rec.matrix, atol=1e-9)


def test_se3_geodesic_interpolation():
    """Verify geodesic interpolation produces smooth midpoint in SE(3)."""
    t0 = SE3.identity()
    t1 = SE3.exp(np.array([2.0, 4.0, 6.0, 0.0, 0.0, 1.0]))

    mid = SE3.interpolate(t0, t1, 0.5)
    xi_mid = mid.log()

    # Midpoint tangent vector should be half of the full delta
    assert np.allclose(xi_mid, 0.5 * t1.log(), atol=1e-9)
