import pytest

from engine.physics.collision.shapes import Box, Sphere
from engine.physics.materials import CANONICAL_MATERIALS
from engine.physics.rigid.body import RigidBody, SleepState
from engine.physics.rigid.integrator import SLEEP_TIME_THRESHOLD, integrate, wake
from engine.physics.math3 import Vec3


def make_falling_body(**overrides) -> RigidBody:
    kwargs = dict(
        id="b1",
        shape=Sphere(0.5),
        material=CANONICAL_MATERIALS["steel"],
        mass=1.0,
    )
    kwargs.update(overrides)
    return RigidBody(**kwargs)


def test_static_body_has_zero_inverse_mass_and_never_moves():
    body = RigidBody(id="ground", shape=Box(Vec3(10, 1, 10)), material=CANONICAL_MATERIALS["concrete"], mass=0.0)
    assert body.is_static
    assert body.inv_mass == 0.0
    assert body.sleep_state == SleepState.STATIC
    integrate(body, gravity=Vec3(0, -9.81, 0), dt=1.0)
    assert body.position == Vec3(0, 0, 0)


def test_free_fall_matches_kinematics():
    body = make_falling_body(linear_damping=0.0)  # isolate gravity from damping
    gravity = Vec3(0, -9.81, 0)
    dt = 0.01
    for _ in range(100):  # 1 second
        integrate(body, gravity, dt)
    # semi-implicit Euler without damping: v = g*t, x = sum(v_i * dt)
    assert body.linear_velocity.y == pytest.approx(-9.81 * 1.0, rel=1e-6)
    assert body.position.y < 0


def test_damping_reduces_velocity_growth():
    damped = make_falling_body(id="damped", linear_damping=0.5)
    undamped = make_falling_body(id="undamped", linear_damping=0.0)
    gravity = Vec3(0, -9.81, 0)
    for _ in range(50):
        integrate(damped, gravity, 0.02)
        integrate(undamped, gravity, 0.02)
    assert abs(damped.linear_velocity.y) < abs(undamped.linear_velocity.y)


def test_torque_spins_free_body():
    body = make_falling_body(shape=Sphere(0.5))
    body.apply_torque(Vec3(0, 1.0, 0))
    integrate(body, gravity=Vec3(0, 0, 0), dt=0.1)
    assert body.angular_velocity.y > 0


def test_body_sleeps_when_still_long_enough():
    body = make_falling_body(linear_damping=1.0, angular_damping=1.0)
    body.linear_velocity = Vec3(0.001, 0, 0)  # already below sleep threshold
    t = 0.0
    dt = 0.05
    while t < SLEEP_TIME_THRESHOLD + 0.1:
        integrate(body, gravity=Vec3(0, 0, 0), dt=dt)
        t += dt
    assert body.sleep_state == SleepState.SLEEPING
    assert body.linear_velocity == Vec3.zero()


def test_sleeping_body_does_not_integrate_until_woken():
    body = make_falling_body()
    body.sleep_state = SleepState.SLEEPING
    integrate(body, gravity=Vec3(0, -9.81, 0), dt=1.0)
    assert body.position == Vec3(0, 0, 0)
    assert body.linear_velocity == Vec3.zero()

    wake(body)
    assert body.sleep_state == SleepState.AWAKE
    integrate(body, gravity=Vec3(0, -9.81, 0), dt=1.0)
    assert body.position.y < 0


def test_kinetic_energy_zero_at_rest_and_positive_when_moving():
    body = make_falling_body()
    assert body.kinetic_energy() == 0.0
    body.linear_velocity = Vec3(1, 0, 0)
    assert body.kinetic_energy() == pytest.approx(0.5 * body.mass * 1.0)


def test_serialization_roundtrip():
    body = make_falling_body()
    body.linear_velocity = Vec3(1, 2, 3)
    restored = RigidBody.from_dict(body.to_dict())
    assert restored.id == body.id
    assert restored.linear_velocity == body.linear_velocity
    assert restored.mass == body.mass
