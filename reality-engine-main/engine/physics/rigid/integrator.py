"""Semi-implicit Euler integration (spec sec 8 core equations) plus
sleeping (spec sec 8: "must support ... sleeping").

Angular integration uses the full Euler rigid-body equation (V11 spec
sec 915): I*alpha + omega x (I*omega) = tau, i.e.
alpha = I^-1 * (tau - omega x (I*omega))
not the simplified alpha = I^-1*tau this backend started with, which
silently drops the gyroscopic term and is only correct for a sphere
(isotropic inertia). Both omega and I here are in the BODY-LOCAL
(principal-axis) frame -- the only frame in which a constant diagonal I
stays valid without being re-derived from orientation every step (see
math3.Quat.integrate for the matching quaternion convention).
"""

from __future__ import annotations

from .body import RigidBody, SleepState
from engine.physics.math3 import Vec3

SLEEP_LINEAR_THRESHOLD = 0.05  # m/s
SLEEP_ANGULAR_THRESHOLD = 0.05  # rad/s
SLEEP_TIME_THRESHOLD = 0.5  # seconds of stillness before sleeping


def integrate(body: RigidBody, gravity: Vec3, dt: float) -> None:
    if body.is_static or body.sleep_state == SleepState.SLEEPING:
        body.clear_accumulators()
        return

    # v(t+dt) = v(t) + a dt   where a = g * gravity_scale + F/m
    linear_accel = gravity * body.gravity_scale + body.force * body.inv_mass
    body.linear_velocity = body.linear_velocity + linear_accel * dt
    # Damping is a rate, applied over dt -- not a fixed per-step fraction
    # (a fixed fraction would make damping strength depend on step size).
    body.linear_velocity = body.linear_velocity * max(0.0, 1.0 - body.linear_damping * dt)

    # omega(t+dt) = omega(t) + alpha dt
    # Euler's equation: alpha = I^-1 * (tau - omega x (I * omega))
    angular_momentum = body.inertia.apply(body.angular_velocity)
    gyroscopic_torque = body.angular_velocity.cross(angular_momentum)
    net_torque = body.torque - gyroscopic_torque
    angular_accel = body.inv_inertia.apply(net_torque)
    body.angular_velocity = body.angular_velocity + angular_accel * dt
    body.angular_velocity = body.angular_velocity * max(0.0, 1.0 - body.angular_damping * dt)

    body.position = body.position + body.linear_velocity * dt
    body.orientation = body.orientation.integrate(body.angular_velocity, dt)

    body.clear_accumulators()
    _update_sleep_state(body, dt)


def wake(body: RigidBody) -> None:
    if body.sleep_state == SleepState.SLEEPING:
        body.sleep_state = SleepState.AWAKE
        body._sleep_timer = 0.0


def _update_sleep_state(body: RigidBody, dt: float) -> None:
    is_still = (
        body.linear_velocity.length() < SLEEP_LINEAR_THRESHOLD
        and body.angular_velocity.length() < SLEEP_ANGULAR_THRESHOLD
    )
    if is_still:
        body._sleep_timer += dt
        if body._sleep_timer >= SLEEP_TIME_THRESHOLD:
            body.sleep_state = SleepState.SLEEPING
            body.linear_velocity = Vec3.zero()
            body.angular_velocity = Vec3.zero()
    else:
        body._sleep_timer = 0.0
