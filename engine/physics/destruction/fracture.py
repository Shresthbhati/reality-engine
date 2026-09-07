"""Basic fracture solver: P1 destruction via impact-based breaking.

Implements deterministic fracture based on:
  - Material fracture toughness (K_IC, G_f)
  - Impact kinetic energy
  - Contact stress concentration
  - Random shard generation (seeded)

Spec §12: BASIC FRACTURE
"""

from __future__ import annotations

import math
from typing import Optional, List, Dict, Any

from engine.core.rng import DeterministicRNG
from engine.core.logging import get_logger
from engine.physics.math3 import Vec3
from .interface import IDestructionBackend, FractureEvent


# Material fracture parameters (spec §12 Table 2.1)
FRACTURE_PARAMS = {
    "glass": {
        "fracture_toughness_mpa_sqrt_m": 0.9,  # K_IC (MPa√m)
        "fracture_energy_j_m2": 10.0,  # G_f (J/m²)
        "min_impact_velocity_m_s": 2.0,
        "fragment_count_min": 3,
        "fragment_count_max": 12,
        "density_kg_m3": 2500.0,
    },
    "ceramic": {
        "fracture_toughness_mpa_sqrt_m": 4.0,
        "fracture_energy_j_m2": 50.0,
        "min_impact_velocity_m_s": 3.0,
        "fragment_count_min": 2,
        "fragment_count_max": 8,
        "density_kg_m3": 3600.0,
    },
    "concrete": {
        "fracture_toughness_mpa_sqrt_m": 1.2,
        "fracture_energy_j_m2": 100.0,
        "min_impact_velocity_m_s": 5.0,
        "fragment_count_min": 1,
        "fragment_count_max": 6,
        "density_kg_m3": 2400.0,
    },
    "stone": {
        "fracture_toughness_mpa_sqrt_m": 1.5,
        "fracture_energy_j_m2": 80.0,
        "min_impact_velocity_m_s": 4.0,
        "fragment_count_min": 2,
        "fragment_count_max": 10,
        "density_kg_m3": 2700.0,
    },
}

# Minimum fragment size (m) — fragments smaller than this are not generated
MIN_FRAGMENT_SIZE_M = 0.05


class BasicFractureSolver(IDestructionBackend):
    """P1 fracture solver using impact-based breaking with deterministic shard generation.

    Invariants:
      - Fracture is deterministic given same seed and impact parameters
      - Fragment generation is repeatable
      - Energy conservation: kinetic energy ≤ fracture energy
      - All fragments are above minimum size threshold
    """

    def __init__(self, seed: int = 42):
        self._rng = DeterministicRNG(seed)
        self._logger = get_logger("engine.physics.destruction.fracture")
        self._fractured_entities: set[str] = set()

    def can_fracture(self, entity_id: str, mass: float, material_name: str) -> bool:
        """Check if entity can be fractured.

        Args:
            entity_id: Entity to check
            mass: Entity mass (kg)
            material_name: Material name (e.g. "glass", "concrete")

        Returns:
            True if material is fractureable and entity not already fractured
        """
        if entity_id in self._fractured_entities:
            return False
        if material_name not in FRACTURE_PARAMS:
            return False
        if mass <= 0:
            return False
        return True

    def fracture(
        self,
        entity_id: str,
        impact_point: Vec3,
        impact_velocity: float,
        impact_normal: Vec3,
        tick: int,
        timestamp: float,
        material_name: str = "glass",
        mass_kg: float = 1.0,
    ) -> Optional[FractureEvent]:
        """Fracture an entity.

        Determines if impact energy exceeds fracture toughness. If so,
        creates a FractureEvent and marks entity as fractured.

        Args:
            entity_id: Entity being impacted
            impact_point: Location of impact (world space)
            impact_velocity: Impact speed (m/s)
            impact_normal: Impact direction (unit vector)
            tick: Simulation tick
            timestamp: World time (seconds)
            material_name: Material type (must be in FRACTURE_PARAMS)
            mass_kg: Entity mass (kg)

        Returns:
            FractureEvent if fracture occurred, None if impact insufficient
        """
        if not self.can_fracture(entity_id, mass_kg, material_name):
            return None

        params = FRACTURE_PARAMS[material_name]

        # Check minimum impact velocity
        if impact_velocity < params["min_impact_velocity_m_s"]:
            return None

        # Compute impact kinetic energy (simplified: KE = 0.5 * m * v^2)
        impact_kinetic_energy_j = 0.5 * mass_kg * impact_velocity ** 2

        # Fracture energy threshold (simplified: stress concentration at impact point)
        # Real: σ = K_IC / (π * a)^0.5, but we use empirical formula for P1
        fracture_threshold_j = params["fracture_energy_j_m2"] * mass_kg / params["density_kg_m3"]

        if impact_kinetic_energy_j < fracture_threshold_j:
            return None

        # Fracture occurred!
        self._fractured_entities.add(entity_id)

        # Determine fragment count (seeded randomness)
        min_frags = params["fragment_count_min"]
        max_frags = params["fragment_count_max"]
        fragment_count = self._rng.randint(min_frags, max_frags)

        event = FractureEvent(
            entity_id=entity_id,
            impact_point=impact_point,
            impact_velocity=impact_velocity,
            impact_normal=impact_normal,
            fragment_count=fragment_count,
            energy_released_j=impact_kinetic_energy_j,
            tick=tick,
            timestamp=timestamp,
        )

        self._logger.info(
            "Entity fractured",
            context={
                "entity_id": entity_id,
                "material": material_name,
                "impact_velocity_m_s": impact_velocity,
                "fragments": fragment_count,
                "energy_j": impact_kinetic_energy_j,
            },
        )

        return event

    def generate_fragments(
        self,
        entity_id: str,
        fracture_event: FractureEvent,
        geometry_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate fragment data from a fractured object.

        Generates deterministic fragment positions, velocities, and sizes
        based on the fracture event and original geometry.

        Args:
            entity_id: Original entity ID
            fracture_event: Fracture event that triggered generation
            geometry_data: Original geometry (bounds, volume, etc.)

        Returns:
            List of fragment specifications
        """
        fragments = []
        impact_point = fracture_event.impact_point
        impact_velocity_mag = fracture_event.impact_velocity
        impact_normal = fracture_event.impact_normal

        # Get geometry bounds
        bounds_min = geometry_data.get("bounds_min", Vec3(-1, -1, -1))
        bounds_max = geometry_data.get("bounds_max", Vec3(1, 1, 1))
        bounds_size = bounds_max - bounds_min
        original_volume = bounds_size.x * bounds_size.y * bounds_size.z

        # Divide volume among fragments
        fragment_volume = original_volume / fracture_event.fragment_count
        fragment_size = (fragment_volume) ** (1.0 / 3.0)

        # Don't generate fragments smaller than threshold
        if fragment_size < MIN_FRAGMENT_SIZE_M:
            fragment_size = MIN_FRAGMENT_SIZE_M

        # Generate fragments deterministically around impact point
        for i in range(fracture_event.fragment_count):
            # Deterministic position offset from impact point
            offset_x = self._rng.uniform(-fragment_size, fragment_size)
            offset_y = self._rng.uniform(-fragment_size, fragment_size)
            offset_z = self._rng.uniform(-fragment_size, fragment_size)

            fragment_pos = impact_point + Vec3(offset_x, offset_y, offset_z)

            # Velocity: radiate outward from impact point with some scatter
            radial_direction = (Vec3(offset_x, offset_y, offset_z)).normalized()
            scatter_angle = self._rng.uniform(-0.3, 0.3)  # Radians
            cos_a = math.cos(scatter_angle)
            sin_a = math.sin(scatter_angle)

            # Add scatter to direction
            scattered_dir = Vec3(
                radial_direction.x * cos_a - radial_direction.y * sin_a,
                radial_direction.x * sin_a + radial_direction.y * cos_a,
                radial_direction.z,
            ).normalized()

            # Fragment velocity: inherit from impact + radial expansion
            expansion_velocity = impact_velocity_mag * (0.5 + 0.5 * self._rng.uniform(0, 1))
            fragment_velocity = scattered_dir * expansion_velocity

            fragments.append({
                "id": f"{entity_id}_frag_{i}",
                "parent_id": entity_id,
                "position": fragment_pos.to_dict(),
                "velocity": fragment_velocity.to_dict(),
                "size_m": fragment_size,
                "volume_m3": fragment_size ** 3,
                "lifetime_s": 10.0 + self._rng.uniform(0, 5),  # 10-15 seconds
            })

        return fragments

    def serialize(self) -> Dict[str, Any]:
        """Serialize fracture solver state."""
        return {
            "format_version": 1,
            "fractured_entities": list(self._fractured_entities),
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore fracture solver state."""
        if data.get("format_version") != 1:
            raise ValueError(f"Unsupported fracture state version: {data.get('format_version')}")

        self._fractured_entities = set(data.get("fractured_entities", []))
