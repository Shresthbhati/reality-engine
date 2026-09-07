"""Sequential-impulse contact solver: restitution + Coulomb friction +
Baumgarte position correction. Single pass per contact per step (not an
iterative Gauss-Seidel solver) -- adequate for P1 gameplay fidelity
(spec sec 1.4); a persistent-contact iterative solver is a P2+ concern.

Convention: `normal` points from `a` toward `b`. `a` may be None to mean
a static, infinite-mass surface (a plane) -- every narrowphase function
in collision/narrowphase.py that involves a Plane returns a normal in
this same "plane is A" orientation, so callers can pass a=None directly.

Only applies LINEAR impulses (see rigid/body.py module docstring for why
angular contact response is deferred).
"""

from __future__ import annotations

from typing import Optional

from engine.physics.rigid.body import RigidBody
from engine.physics.math3 import Vec3
from engine.physics.rigid.integrator import wake

BAUMGARTE_FACTOR = 0.2
PENETRATION_SLOP = 0.005  # meters of allowed overlap before correction kicks in
REST_VELOCITY_THRESHOLD = 0.5  # m/s; below this, restitution is treated as 0

# Naive impulse solvers re-apply full restitution every frame on a resting
# contact -- gravity adds a small approach velocity each tick, restitution
# bounces it back, forever, so the body never settles. The standard fix
# (used by most real-time engines) is to zero restitution once the approach
# speed drops below a small threshold, since that residual "bounce" isn't a
# real collision, it's discretization noise.


def resolve_contact(
    a: Optional[RigidBody],
    b: RigidBody,
    normal: Vec3,
    penetration: float,
    restitution: float,
    friction: float,
) -> None:
    inv_mass_a = a.inv_mass if a is not None else 0.0
    inv_mass_b = b.inv_mass
    inv_mass_sum = inv_mass_a + inv_mass_b
    if inv_mass_sum <= 0.0:
        return  # both static (or degenerate) -- nothing to resolve

    if a is not None:
        wake(a)
    wake(b)

    vel_a = a.linear_velocity if a is not None else Vec3.zero()
    vel_b = b.linear_velocity
    rel_vel = vel_b - vel_a
    vel_along_normal = rel_vel.dot(normal)

    if vel_along_normal > 0:
        j = 0.0  # already separating
    else:
        effective_restitution = restitution if -vel_along_normal > REST_VELOCITY_THRESHOLD else 0.0
        j = -(1.0 + effective_restitution) * vel_along_normal / inv_mass_sum

    impulse = normal * j
    if a is not None:
        a.apply_impulse(-impulse)
    b.apply_impulse(impulse)

    # Coulomb friction along the single tangential relative-velocity direction.
    rel_vel = (b.linear_velocity) - (a.linear_velocity if a is not None else Vec3.zero())
    tangent_vel = rel_vel - normal * rel_vel.dot(normal)
    tangent_speed = tangent_vel.length()
    if tangent_speed > 1e-6 and j > 0.0:
        tangent = tangent_vel / tangent_speed
        jt = -rel_vel.dot(tangent) / inv_mass_sum
        max_friction = friction * j
        jt = max(-max_friction, min(max_friction, jt))
        friction_impulse = tangent * jt
        if a is not None:
            a.apply_impulse(-friction_impulse)
        b.apply_impulse(friction_impulse)

    # Positional correction to prevent sinking, applied directly (this is
    # a pseudo-velocity trick -- it does not affect reported velocities).
    correction_mag = max(penetration - PENETRATION_SLOP, 0.0) / inv_mass_sum * BAUMGARTE_FACTOR
    if correction_mag > 0.0:
        correction = normal * correction_mag
        if a is not None:
            a.position = a.position - correction * inv_mass_a
        b.position = b.position + correction * inv_mass_b
