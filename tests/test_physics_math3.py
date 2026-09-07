import math

import pytest

from engine.physics.math3 import Mat3, Quat, Vec3


def test_vec3_arithmetic():
    a = Vec3(1, 2, 3)
    b = Vec3(4, 5, 6)
    assert a + b == Vec3(5, 7, 9)
    assert b - a == Vec3(3, 3, 3)
    assert a * 2 == Vec3(2, 4, 6)
    assert -a == Vec3(-1, -2, -3)


def test_vec3_dot_cross():
    x = Vec3(1, 0, 0)
    y = Vec3(0, 1, 0)
    assert x.dot(y) == 0.0
    assert x.cross(y) == Vec3(0, 0, 1)


def test_vec3_length_and_normalize():
    v = Vec3(3, 4, 0)
    assert v.length() == 5.0
    n = v.normalized()
    assert n.length() == pytest.approx(1.0)


def test_vec3_normalize_zero_vector_is_safe():
    assert Vec3(0, 0, 0).normalized() == Vec3(0, 0, 0)


def test_vec3_is_finite():
    assert Vec3(1, 2, 3).is_finite()
    assert not Vec3(float("nan"), 0, 0).is_finite()
    assert not Vec3(float("inf"), 0, 0).is_finite()


def test_mat3_diagonal_apply():
    m = Mat3.diagonal(2, 3, 4)
    v = Vec3(1, 1, 1)
    assert m.apply(v) == Vec3(2, 3, 4)


def test_mat3_inverse_diagonal():
    m = Mat3.diagonal(2, 4, 0)  # 0 => locked/static axis
    inv = m.inverse_diagonal()
    assert inv.rows[0][0] == pytest.approx(0.5)
    assert inv.rows[1][1] == pytest.approx(0.25)
    assert inv.rows[2][2] == 0.0


def test_quat_identity_rotate_is_noop():
    q = Quat.identity()
    v = Vec3(1, 2, 3)
    assert q.rotate(v) == v


def test_quat_normalize():
    q = Quat(2, 0, 0, 0).normalized()
    assert q.w == pytest.approx(1.0)


def test_quat_integrate_small_step_stays_normalized():
    q = Quat.identity()
    omega = Vec3(0.1, 0.0, 0.0)
    q2 = q.integrate(omega, dt=0.01)
    n = math.sqrt(q2.w**2 + q2.x**2 + q2.y**2 + q2.z**2)
    assert n == pytest.approx(1.0, abs=1e-9)


def test_quat_is_finite():
    assert Quat.identity().is_finite()
    assert not Quat(float("nan"), 0, 0, 1).is_finite()
