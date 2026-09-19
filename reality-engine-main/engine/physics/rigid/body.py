"""Rigid body state (spec sec 8 RIGID BODY PHYSICS).

Scope note: contact resolution in this backend applies LINEAR impulses
only. Angular velocity still integrates correctly under directly
applied torque (so a free-spinning body behaves right), but contacts
don't yet generate angular impulses -- that needs orientation-aware
(OBB) collision to be correct, since our Box shape is axis-aligned and
doesn't rotate (see collision/shapes.py). Full rotational contact
response is deferred to when OBB/mesh collision (LOD_2+) lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from engine.physics.collision.shapes import Box, Shape, Sphere, shape_from_dict
from engine.physics.materials import PhysicsMaterial
from engine.physics.math3 import Mat3, Quat, Vec3


class SleepState(str, Enum):
    AWAKE = "AWAKE"
    SLEEPING = "SLEEPING"
    STATIC = "STATIC"  # infinite mass, never integrated


def _inertia_tensor(shape: Shape, mass: float) -> Mat3:
    """Body-frame inertia tensor for the shape's principal axes."""
    if isinstance(shape, Sphere):
        i = 0.4 * mass * shape.radius**2
        return Mat3.diagonal(i, i, i)
    if isinstance(shape, Box):
        he = shape.half_extents
        w, h, d = 2 * he.x, 2 * he.y, 2 * he.z
        ixx = (mass / 12.0) * (h**2 + d**2)
        iyy = (mass / 12.0) * (w**2 + d**2)
        izz = (mass / 12.0) * (w**2 + h**2)
        return Mat3.diagonal(ixx, iyy, izz)
    raise TypeError(f"no inertia tensor formula for shape {type(shape).__name__}")


@dataclass
class RigidBody:
    id: str
    shape: Shape
    material: PhysicsMaterial
    mass: float  # 0 => static (infinite mass, never moves)
    position: Vec3 = field(default_factory=Vec3.zero)
    orientation: Quat = field(default_factory=Quat.identity)
    linear_velocity: Vec3 = field(default_factory=Vec3.zero)
    angular_velocity: Vec3 = field(default_factory=Vec3.zero)
    linear_damping: float = 0.01
    angular_damping: float = 0.05
    gravity_scale: float = 1.0

    _force_accum: Vec3 = field(default_factory=Vec3.zero, repr=False)
    _torque_accum: Vec3 = field(default_factory=Vec3.zero, repr=False)
    sleep_state: SleepState = SleepState.AWAKE
    _sleep_timer: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if self.mass < 0:
            raise ValueError(f"mass must be >= 0, got {self.mass}")
        if self.mass == 0.0:
            self.sleep_state = SleepState.STATIC
            self._inertia = Mat3.diagonal(0.0, 0.0, 0.0)
        else:
            self._inertia = _inertia_tensor(self.shape, self.mass)

    @property
    def is_static(self) -> bool:
        return self.mass == 0.0

    @property
    def inv_mass(self) -> float:
        return 0.0 if self.is_static else 1.0 / self.mass

    @property
    def inertia(self) -> Mat3:
        return self._inertia

    @property
    def inv_inertia(self) -> Mat3:
        return self._inertia.inverse_diagonal()

    def apply_force(self, force: Vec3) -> None:
        self._force_accum = self._force_accum + force

    def apply_torque(self, torque: Vec3) -> None:
        self._torque_accum = self._torque_accum + torque

    def apply_impulse(self, impulse: Vec3) -> None:
        """Instantaneous linear velocity change (used by the contact solver)."""
        if self.is_static:
            return
        self.linear_velocity = self.linear_velocity + impulse * self.inv_mass

    def clear_accumulators(self) -> None:
        self._force_accum = Vec3.zero()
        self._torque_accum = Vec3.zero()

    @property
    def force(self) -> Vec3:
        return self._force_accum

    @property
    def torque(self) -> Vec3:
        return self._torque_accum

    def kinetic_energy(self) -> float:
        if self.is_static:
            return 0.0
        linear = 0.5 * self.mass * self.linear_velocity.length_sq()
        i = self._inertia
        w = self.angular_velocity
        angular = 0.5 * (i.rows[0][0] * w.x**2 + i.rows[1][1] * w.y**2 + i.rows[2][2] * w.z**2)
        return linear + angular

    def is_finite(self) -> bool:
        return (
            self.position.is_finite()
            and self.linear_velocity.is_finite()
            and self.angular_velocity.is_finite()
            and self.orientation.is_finite()
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "shape": self.shape.to_dict(),
            "material": self.material.to_dict(),
            "mass": self.mass,
            "position": self.position.to_dict(),
            "orientation": self.orientation.to_dict(),
            "linear_velocity": self.linear_velocity.to_dict(),
            "angular_velocity": self.angular_velocity.to_dict(),
            "linear_damping": self.linear_damping,
            "angular_damping": self.angular_damping,
            "gravity_scale": self.gravity_scale,
            "sleep_state": self.sleep_state.value,
        }

    @staticmethod
    def from_dict(data: dict) -> "RigidBody":
        return RigidBody(
            id=data["id"],
            shape=shape_from_dict(data["shape"]),
            material=PhysicsMaterial.from_dict(data["material"]),
            mass=data["mass"],
            position=Vec3.from_dict(data["position"]),
            orientation=Quat.from_dict(data["orientation"]),
            linear_velocity=Vec3.from_dict(data["linear_velocity"]),
            angular_velocity=Vec3.from_dict(data["angular_velocity"]),
            linear_damping=data.get("linear_damping", 0.01),
            angular_damping=data.get("angular_damping", 0.05),
            gravity_scale=data.get("gravity_scale", 1.0),
            sleep_state=SleepState(data.get("sleep_state", SleepState.AWAKE.value)),
        )
