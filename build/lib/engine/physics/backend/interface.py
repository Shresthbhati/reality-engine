"""Physics backend abstraction (spec sec 7.2).

Public technologies like NVIDIA PhysX are backend candidates, not a
reason to couple the canonical architecture to one vendor -- so every
concrete backend (this pass ships exactly one, SimpleRigidBodyBackend)
implements this interface and nothing outside engine/physics/backend/
is allowed to depend on backend internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from engine.physics.math3 import Vec3


@dataclass(frozen=True)
class PhysicsWorldConfig:
    gravity: Vec3 = field(default_factory=lambda: Vec3(0.0, -9.81, 0.0))
    seed: int = 0


@dataclass(frozen=True)
class RaycastHit:
    body_id: str
    point: Vec3
    normal: Vec3
    distance: float


class IPhysicsBackend(ABC):
    @abstractmethod
    def create_world(self, config: PhysicsWorldConfig): ...

    @abstractmethod
    def step(self, world, dt: float) -> None: ...

    @abstractmethod
    def query_contacts(self, world) -> list: ...

    @abstractmethod
    def query_raycast(
        self, world, origin: Vec3, direction: Vec3, max_distance: float
    ) -> Optional[RaycastHit]: ...

    @abstractmethod
    def serialize(self, world) -> dict: ...

    @abstractmethod
    def deserialize(self, data: dict):
        """Return a new world handle reconstructed from serialize()'s output."""
        ...
