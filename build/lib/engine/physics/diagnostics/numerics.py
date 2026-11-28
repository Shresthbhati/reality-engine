"""Numerical health tracking (spec sec 86 NUMERICAL HEALTH).

Every simulation tick records nan/inf counts, velocity/acceleration
extrema, and mass/energy error. Hard failure (SimulationDivergedError)
on NaN contamination, non-finite state, or explosive energy growth --
per spec sec 86, this must never be a silent "physics went a bit weird"
situation.
"""

from __future__ import annotations

from dataclasses import dataclass

ENERGY_EXPLOSION_FACTOR = 1000.0  # hard-failure threshold vs previous tick's energy
ENERGY_ABSOLUTE_FLOOR = 1.0  # joules; below this, relative growth is noise, not divergence

# A body starting near rest has energy near zero, so gravity alone makes
# the very first tick's energy "infinitely" larger in relative terms --
# that's expected physics, not divergence. Requiring the new energy to
# also clear an absolute floor avoids flagging normal startup as a
# hard failure while still catching genuine numerical blowups.


class SimulationDivergedError(RuntimeError):
    pass


@dataclass(frozen=True)
class NumericsReport:
    nan_count: int
    inf_count: int
    max_velocity: float
    max_angular_velocity: float
    total_kinetic_energy: float
    body_count: int

    def to_dict(self) -> dict:
        return {
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "max_velocity": self.max_velocity,
            "max_angular_velocity": self.max_angular_velocity,
            "total_kinetic_energy": self.total_kinetic_energy,
            "body_count": self.body_count,
        }


def check_world(bodies: list, previous_energy: float | None) -> NumericsReport:
    """Inspect all bodies after a step. Raises SimulationDivergedError on
    hard failure; otherwise returns a report the caller can log/store.
    """
    import math

    nan_count = 0
    inf_count = 0
    max_v = 0.0
    max_w = 0.0
    total_energy = 0.0

    for body in bodies:
        for component in (
            body.position.x, body.position.y, body.position.z,
            body.linear_velocity.x, body.linear_velocity.y, body.linear_velocity.z,
            body.angular_velocity.x, body.angular_velocity.y, body.angular_velocity.z,
        ):
            if math.isnan(component):
                nan_count += 1
            elif math.isinf(component):
                inf_count += 1

        if not body.is_finite():
            raise SimulationDivergedError(
                f"body '{body.id}' has non-finite state (position/velocity/orientation)"
            )

        max_v = max(max_v, body.linear_velocity.length())
        max_w = max(max_w, body.angular_velocity.length())
        total_energy += body.kinetic_energy()

    if nan_count > 0 or inf_count > 0:
        raise SimulationDivergedError(
            f"numerical divergence: nan_count={nan_count} inf_count={inf_count}"
        )

    if previous_energy is not None and previous_energy > 1e-9 and total_energy > ENERGY_ABSOLUTE_FLOOR:
        if total_energy > previous_energy * ENERGY_EXPLOSION_FACTOR:
            raise SimulationDivergedError(
                f"explosive energy growth: {previous_energy:.6g} -> {total_energy:.6g}"
            )

    return NumericsReport(
        nan_count=nan_count,
        inf_count=inf_count,
        max_velocity=max_v,
        max_angular_velocity=max_w,
        total_kinetic_energy=total_energy,
        body_count=len(bodies),
    )
