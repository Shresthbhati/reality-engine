"""WorldIR top-level container (spec sec 6 WORLD IR).

Fields the spec lists that have no implemented subsystem yet (terrain,
environment, physics, materials, semantics, observations, sessions,
temporal_state, branches) are kept as opaque dicts -- present in the
schema and round-tripped faithfully through serialization, but not
interpreted here. `entities` and `coordinate_system` are the only
fields this foundation layer actually understands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .coordinates import Frame
from .entity import EntityRegistry


@dataclass
class WorldIR:
    id: str
    version: int = 1
    coordinate_system: Frame = Frame.WORLD
    entities: EntityRegistry = field(default_factory=EntityRegistry)

    # Opaque, not-yet-implemented sections -- preserved as-is.
    terrain: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    physics: dict = field(default_factory=dict)
    materials: dict = field(default_factory=dict)
    semantics: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)
    sessions: dict = field(default_factory=dict)
    temporal_state: dict = field(default_factory=dict)
    branches: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "coordinate_system": self.coordinate_system.value,
            "entities": self.entities.to_list(),
            "terrain": self.terrain,
            "environment": self.environment,
            "physics": self.physics,
            "materials": self.materials,
            "semantics": self.semantics,
            "observations": self.observations,
            "sessions": self.sessions,
            "temporal_state": self.temporal_state,
            "branches": self.branches,
        }

    @staticmethod
    def from_dict(data: dict) -> "WorldIR":
        return WorldIR(
            id=data["id"],
            version=data.get("version", 1),
            coordinate_system=Frame(data.get("coordinate_system", Frame.WORLD.value)),
            entities=EntityRegistry.from_list(data.get("entities", [])),
            terrain=data.get("terrain", {}),
            environment=data.get("environment", {}),
            physics=data.get("physics", {}),
            materials=data.get("materials", {}),
            semantics=data.get("semantics", {}),
            observations=data.get("observations", {}),
            sessions=data.get("sessions", {}),
            temporal_state=data.get("temporal_state", {}),
            branches=data.get("branches", {}),
        )
