"""Regression test for the full Euler rigid-body equation (V11 spec sec
915: I*alpha + omega x (I*omega) = tau). A sphere can't exercise this --
its inertia is isotropic, so the gyroscopic term omega x (I*omega) is
identically zero for any omega. An asymmetric box is required.
"""

import pytest

from engine.physics.collision.shapes import Box
from engine.physics.materials import CANONICAL_MATERIALS
from engine.physics.rigid.body import RigidBody
from engine.physics.rigid.integrator import integrate
from engine.physics.math3 import Vec3


def make_asymmetric_box(**overrides) -> RigidBody:
    kwargs = dict(
        id="box",
        shape=Box(Vec3(0.2, 0.5, 1.0)),  # distinct extents on every axis -> distinct principal moments
        material=CANONICAL_MATERIALS["steel"],
        mass=2.0,
        angular_damping=0.0,
        linear_damping=0.0,
    )
    kwargs.update(overrides)
    return RigidBody(**kwargs)


def angular_momentum_sq(body: RigidBody) -> float:
    L = body.inertia.apply(body.angular_velocity)
    return L.length_sq()


def test_torque_free_rotation_conserves_body_frame_angular_momentum():
    body = make_asymmetric_box(angular_velocity=Vec3(1.0, 2.0, 0.3))
    initial = angular_momentum_sq(body)

    for _ in range(500):
        integrate(body, gravity=Vec3(0, 0, 0), dt=0.001)

    # Euler's equations conserve |I*omega| (and kinetic energy) for
    # torque-free rotation; angular velocity direction/magnitude alone
    # need not be constant for an asymmetric body (that's the point).
    assert angular_momentum_sq(body) == pytest.approx(initial, rel=1e-3)


def test_torque_free_rotation_conserves_rotational_kinetic_energy():
    body = make_asymmetric_box(angular_velocity=Vec3(0.5, 1.5, 2.0))
    initial_ke = body.kinetic_energy()

    for _ in range(500):
        integrate(body, gravity=Vec3(0, 0, 0), dt=0.001)

    assert body.kinetic_energy() == pytest.approx(initial_ke, rel=1e-3)


def test_single_axis_spin_has_no_gyroscopic_precession():
    """Spinning purely about one principal axis is a fixed point of
    Euler's equations (omega x (I*omega) = 0 when omega is parallel to a
    principal axis) -- omega should stay constant, unlike a
    multi-axis spin.
    """
    body = make_asymmetric_box(angular_velocity=Vec3(0.0, 3.0, 0.0))
    for _ in range(200):
        integrate(body, gravity=Vec3(0, 0, 0), dt=0.001)
    assert body.angular_velocity.x == pytest.approx(0.0, abs=1e-9)
    assert body.angular_velocity.z == pytest.approx(0.0, abs=1e-9)
    assert body.angular_velocity.y == pytest.approx(3.0, rel=1e-6)


def test_multi_axis_spin_direction_actually_changes():
    """Sanity check that the gyroscopic term is doing something: without
    it (alpha = I^-1 * tau with tau=0), a torque-free body's angular
    velocity would never change at all.
    """
    body = make_asymmetric_box(angular_velocity=Vec3(1.0, 2.0, 0.3))
    initial = body.angular_velocity
    for _ in range(200):
        integrate(body, gravity=Vec3(0, 0, 0), dt=0.001)
    assert body.angular_velocity != initial
