"""Material physics (spec sec 9), scoped to the fields rigid body contact
resolution actually needs right now: density, friction, restitution.

The full MaterialPhysics schema (sec 9) also lists YoungsModulus,
tensile/compressive/shear strength, fracture_energy, thermal and
acoustic properties -- those belong to solvers that don't exist yet
(fracture step 12, thermal, audio). Adding them here as unused fields
would be exactly the kind of half-finished scaffolding the project
conventions warn against; they get added when the solver that reads
them gets built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsMaterial:
    name: str
    density: float  # kg/m^3
    friction_static: float
    friction_dynamic: float
    restitution: float

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError(f"density must be positive, got {self.density}")
        if not (0.0 <= self.restitution <= 1.0):
            raise ValueError(f"restitution must be in [0, 1], got {self.restitution}")
        if self.friction_dynamic > self.friction_static + 1e-9:
            raise ValueError(
                f"dynamic friction ({self.friction_dynamic}) should not exceed "
                f"static friction ({self.friction_static})"
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "density": self.density,
            "friction_static": self.friction_static,
            "friction_dynamic": self.friction_dynamic,
            "restitution": self.restitution,
        }

    @staticmethod
    def from_dict(data: dict) -> "PhysicsMaterial":
        return PhysicsMaterial(
            name=data["name"],
            density=data["density"],
            friction_static=data["friction_static"],
            friction_dynamic=data["friction_dynamic"],
            restitution=data["restitution"],
        )


# Approximate, order-of-magnitude values -- placeholders for real
# measured/sourced data (spec sec 1.1: evidence first). Fine for
# gameplay-tier (P1) simulation; not for engineering-grade (P4) use.
CANONICAL_MATERIALS: dict[str, PhysicsMaterial] = {
    m.name: m
    for m in [
        PhysicsMaterial("concrete", density=2400.0, friction_static=0.7, friction_dynamic=0.6, restitution=0.1),
        PhysicsMaterial("steel", density=7850.0, friction_static=0.6, friction_dynamic=0.5, restitution=0.3),
        PhysicsMaterial("wood", density=600.0, friction_static=0.5, friction_dynamic=0.4, restitution=0.4),
        PhysicsMaterial("glass", density=2500.0, friction_static=0.4, friction_dynamic=0.3, restitution=0.6),
        PhysicsMaterial("brick", density=1900.0, friction_static=0.6, friction_dynamic=0.55, restitution=0.1),
        PhysicsMaterial("asphalt", density=2350.0, friction_static=0.8, friction_dynamic=0.7, restitution=0.05),
        PhysicsMaterial("rubber", density=1100.0, friction_static=0.9, friction_dynamic=0.8, restitution=0.85),
        PhysicsMaterial("ice", density=920.0, friction_static=0.05, friction_dynamic=0.03, restitution=0.1),
    ]
}
