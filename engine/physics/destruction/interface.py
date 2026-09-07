"""Destruction solver abstraction (spec §12).

All destruction solvers implement this interface. Concrete implementations
(fracture, debris) are backend-specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from engine.physics.math3 import Vec3


@dataclass
class FractureEvent:
    """An object has fractured."""
    entity_id: str
    impact_point: Vec3
    impact_velocity: float
    impact_normal: Vec3
    fragment_count: int
    energy_released_j: float
    tick: int
    timestamp: float


class IDestructionBackend(ABC):
    """Abstract interface for destruction (fracture, debris, damage)."""

    @abstractmethod
    def can_fracture(self, entity_id: str, mass: float, material_name: str) -> bool:
        """Check if an entity can be fractured."""
        pass

    @abstractmethod
    def fracture(
        self,
        entity_id: str,
        impact_point: Vec3,
        impact_velocity: float,
        impact_normal: Vec3,
        tick: int,
        timestamp: float,
    ) -> Optional[FractureEvent]:
        """Fracture an entity at impact point.

        Returns FractureEvent if fracture occurred, None if impact insufficient.
        """
        pass

    @abstractmethod
    def generate_fragments(
        self,
        entity_id: str,
        fracture_event: FractureEvent,
        geometry_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate fragment data from a fractured object.

        Returns list of fragment specifications (position, velocity, geometry, etc).
        """
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize destruction state."""
        pass

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore destruction state."""
        pass
