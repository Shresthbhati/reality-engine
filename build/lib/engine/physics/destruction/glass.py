"""Glass physics solver: P2 destruction with pane-specific fracture patterns.

Implements deterministic glass fracture with:
  - Pane geometry and thickness
  - Frame attachment points
  - Radial/spider-web crack patterns
  - Mesh shard fragment generation
  - Energy-based temper classification (annealed/tempered)

Spec §13: GLASS PHYSICS
Extends BasicFractureSolver for window/panel-specific behavior.
"""

from __future__ import annotations

import math
from typing import Optional, List, Dict, Any

from engine.core.rng import DeterministicRNG
from engine.core.logging import get_logger
from engine.physics.math3 import Vec3
from .fracture import BasicFractureSolver, FractureEvent


class GlassTemper:
    """Glass tempering classification affects fracture behavior."""
    ANNEALED = "annealed"      # Shatters into large sharp shards
    TEMPERED = "tempered"      # Shatters into small granules
    LAMINATED = "laminated"    # Holds together after fracture


# Glass-specific fracture parameters (spec §13 Table 3.1)
GLASS_FRACTURE_PARAMS = {
    "annealed": {
        "fracture_toughness_mpa_sqrt_m": 0.9,
        "fracture_energy_j_m2": 10.0,
        "min_impact_velocity_m_s": 2.0,
        "shard_count_min": 6,
        "shard_count_max": 20,
        "shard_size_factor": 1.0,  # Larger shards
        "crack_pattern": "radial",
    },
    "tempered": {
        "fracture_toughness_mpa_sqrt_m": 2.5,  # Much tougher
        "fracture_energy_j_m2": 80.0,
        "min_impact_velocity_m_s": 5.0,
        "shard_count_min": 50,
        "shard_count_max": 200,
        "shard_size_factor": 0.3,  # Tiny granules
        "crack_pattern": "spiderweb",
    },
    "laminated": {
        "fracture_toughness_mpa_sqrt_m": 1.2,
        "fracture_energy_j_m2": 120.0,
        "min_impact_velocity_m_s": 8.0,
        "shard_count_min": 3,
        "shard_count_max": 8,
        "shard_size_factor": 2.0,  # Large pieces held by interlayer
        "crack_pattern": "laminated",
    },
}


class GlassPane:
    """Geometry of a glass pane (window, panel, etc.)."""

    def __init__(
        self,
        pane_id: str,
        width_m: float,
        height_m: float,
        thickness_m: float,
        temper: str = "annealed",
        normal: Vec3 = None,
    ):
        """Initialize glass pane.

        Args:
            pane_id: Unique pane identifier
            width_m: Pane width (meters)
            height_m: Pane height (meters)
            thickness_m: Pane thickness (millimeters, converted to meters)
            temper: Glass temper type (annealed/tempered/laminated)
            normal: Pane normal vector (default Z-up)
        """
        self.pane_id = pane_id
        self.width_m = width_m
        self.height_m = height_m
        self.thickness_m = thickness_m / 1000.0  # Convert mm to m
        self.temper = temper
        self.normal = normal or Vec3(0, 0, 1)
        self.frame_attachment_points: List[Vec3] = []
        self.is_fractured = False

    def add_frame_attachment(self, point: Vec3) -> None:
        """Add a frame attachment point (corner or edge)."""
        self.frame_attachment_points.append(point)

    def surface_area_m2(self) -> float:
        """Calculate pane surface area."""
        return self.width_m * self.height_m

    def volume_m3(self) -> float:
        """Calculate pane volume."""
        return self.width_m * self.height_m * self.thickness_m


class GlassPhysicsSolver(BasicFractureSolver):
    """Glass-specific fracture solver with pane geometry and patterns.

    Extends BasicFractureSolver with:
      - Pane-specific fracture thresholds
      - Radial/spider-web crack patterns
      - Mesh shard generation
      - Frame attachment constraint checking
    """

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._logger = get_logger("engine.physics.destruction.glass")
        self._panes: Dict[str, GlassPane] = {}

    def register_pane(self, pane: GlassPane) -> None:
        """Register a glass pane for fracture tracking.

        Args:
            pane: GlassPane object with geometry and tempering
        """
        self._panes[pane.pane_id] = pane
        self._logger.info(
            "Glass pane registered",
            context={
                "pane_id": pane.pane_id,
                "temper": pane.temper,
                "size_m": f"{pane.width_m:.2f}x{pane.height_m:.2f}",
                "attachments": len(pane.frame_attachment_points),
            },
        )

    def fracture_pane(
        self,
        pane_id: str,
        impact_point: Vec3,
        impact_velocity: float,
        impact_normal: Vec3,
        tick: int,
        timestamp: float,
    ) -> Optional[FractureEvent]:
        """Fracture a glass pane.

        Args:
            pane_id: ID of registered pane
            impact_point: Impact location
            impact_velocity: Impact speed (m/s)
            impact_normal: Impact direction
            tick: Simulation tick
            timestamp: World time

        Returns:
            FractureEvent if pane fractured, None otherwise
        """
        if pane_id not in self._panes:
            self._logger.warning(f"Unknown pane: {pane_id}")
            return None

        pane = self._panes[pane_id]
        if pane.is_fractured:
            return None

        # Get glass parameters based on temper
        params = GLASS_FRACTURE_PARAMS.get(pane.temper, GLASS_FRACTURE_PARAMS["annealed"])

        # Check velocity threshold
        if impact_velocity < params["min_impact_velocity_m_s"]:
            return None

        # Compute impact energy with pane mass
        pane_density = 2500.0  # kg/m³ for soda-lime glass
        pane_mass = pane.volume_m3() * pane_density
        impact_kinetic_energy_j = 0.5 * pane_mass * impact_velocity ** 2

        # Fracture energy threshold
        fracture_threshold_j = params["fracture_energy_j_m2"] * pane.surface_area_m2()

        if impact_kinetic_energy_j < fracture_threshold_j:
            return None

        # Pane fractured!
        pane.is_fractured = True

        # Determine shard count based on temper
        min_shards = params["shard_count_min"]
        max_shards = params["shard_count_max"]
        shard_count = self._rng.randint(min_shards, max_shards)

        event = FractureEvent(
            entity_id=pane_id,
            impact_point=impact_point,
            impact_velocity=impact_velocity,
            impact_normal=impact_normal,
            fragment_count=shard_count,
            energy_released_j=impact_kinetic_energy_j,
            tick=tick,
            timestamp=timestamp,
        )

        self._logger.info(
            "Glass pane fractured",
            context={
                "pane_id": pane_id,
                "temper": pane.temper,
                "impact_velocity_m_s": impact_velocity,
                "shards": shard_count,
                "pattern": params["crack_pattern"],
            },
        )

        return event

    def generate_shards(
        self,
        pane: GlassPane,
        impact_point: Vec3,
        fracture_event: FractureEvent,
    ) -> List[Dict[str, Any]]:
        """Generate mesh shards from fractured pane.

        Creates deterministic shard positions and velocities based on
        crack pattern type (radial/spider-web/laminated).

        Args:
            pane: GlassPane object
            impact_point: Impact location (center of fracture)
            fracture_event: Fracture event that created shards

        Returns:
            List of shard specifications with position, velocity, mesh data
        """
        shards = []
        params = GLASS_FRACTURE_PARAMS.get(pane.temper, GLASS_FRACTURE_PARAMS["annealed"])
        pattern = params["crack_pattern"]

        # Compute shard size
        pane_area = pane.surface_area_m2()
        avg_shard_area = pane_area / fracture_event.fragment_count
        shard_size = math.sqrt(avg_shard_area) * params["shard_size_factor"]

        if pattern == "radial":
            shards = self._generate_radial_shards(
                pane, impact_point, fracture_event, shard_size
            )
        elif pattern == "spiderweb":
            shards = self._generate_spiderweb_shards(
                pane, impact_point, fracture_event, shard_size
            )
        elif pattern == "laminated":
            shards = self._generate_laminated_shards(
                pane, impact_point, fracture_event, shard_size
            )

        return shards

    def _generate_radial_shards(
        self,
        pane: GlassPane,
        impact_point: Vec3,
        event: FractureEvent,
        shard_size: float,
    ) -> List[Dict[str, Any]]:
        """Generate shards in radial pattern (annealed glass)."""
        shards = []

        for i in range(event.fragment_count):
            # Radial direction from impact
            angle = (i / event.fragment_count) * 2 * math.pi
            radial_x = math.cos(angle)
            radial_y = math.sin(angle)
            radial_dir = Vec3(radial_x, radial_y, 0).normalized()

            # Position along radial line
            distance = (i % 5 + 1) * shard_size
            shard_pos = impact_point + radial_dir * distance

            # Velocity radiates outward
            velocity = event.impact_normal * event.impact_velocity * (0.5 + 0.5 * (i % 3) / 3)
            shard_velocity = radial_dir * (event.impact_velocity * 2) + velocity

            shards.append({
                "id": f"{pane.pane_id}_shard_{i}",
                "parent_pane": pane.pane_id,
                "position": shard_pos.to_dict(),
                "velocity": shard_velocity.to_dict(),
                "size_m": shard_size,
                "area_m2": shard_size ** 2,
                "thickness_m": pane.thickness_m,
                "sharpness": 0.9,  # Annealed = very sharp edges
                "lifetime_s": 5.0 + self._rng.uniform(0, 3),
                "mesh_type": "triangle_strip",
            })

        return shards

    def _generate_spiderweb_shards(
        self,
        pane: GlassPane,
        impact_point: Vec3,
        event: FractureEvent,
        shard_size: float,
    ) -> List[Dict[str, Any]]:
        """Generate shards in spider-web pattern (tempered glass)."""
        shards = []

        # Tempered glass creates fine granules in a web-like pattern
        for i in range(event.fragment_count):
            # Grid-based positioning with randomness
            grid_x = (i % 10) * (pane.width_m / 10)
            grid_y = (i // 10) * (pane.height_m / 10)

            offset_x = self._rng.uniform(-shard_size/2, shard_size/2)
            offset_y = self._rng.uniform(-shard_size/2, shard_size/2)

            shard_pos = Vec3(
                impact_point.x + grid_x + offset_x,
                impact_point.y + grid_y + offset_y,
                impact_point.z,
            )

            # Granules scatter in all directions with less energy
            scatter_angle = self._rng.uniform(0, 2 * math.pi)
            scatter_x = math.cos(scatter_angle)
            scatter_y = math.sin(scatter_angle)
            shard_velocity = Vec3(scatter_x, scatter_y, 0.1) * (event.impact_velocity * 0.5)

            shards.append({
                "id": f"{pane.pane_id}_granule_{i}",
                "parent_pane": pane.pane_id,
                "position": shard_pos.to_dict(),
                "velocity": shard_velocity.to_dict(),
                "size_m": shard_size * 0.1,  # Much smaller granules
                "area_m2": (shard_size * 0.1) ** 2,
                "thickness_m": pane.thickness_m * 0.5,
                "sharpness": 0.3,  # Granules are mostly dull
                "lifetime_s": 3.0 + self._rng.uniform(0, 2),
                "mesh_type": "granule",
            })

        return shards

    def _generate_laminated_shards(
        self,
        pane: GlassPane,
        impact_point: Vec3,
        event: FractureEvent,
        shard_size: float,
    ) -> List[Dict[str, Any]]:
        """Generate shards in laminated pattern (held by interlayer)."""
        shards = []

        # Laminated glass stays mostly intact but creates large chunks
        for i in range(event.fragment_count):
            # Mostly on pane surface
            angle = (i / event.fragment_count) * 2 * math.pi
            radial_x = math.cos(angle)
            radial_y = math.sin(angle)

            distance = (i + 1) * (pane.width_m / event.fragment_count) * 0.5
            shard_pos = impact_point + Vec3(radial_x, radial_y, 0) * distance

            # Large chunks move more slowly (held by interlayer)
            shard_velocity = Vec3(radial_x, radial_y, 0.05) * (event.impact_velocity * 0.3)

            shards.append({
                "id": f"{pane.pane_id}_chunk_{i}",
                "parent_pane": pane.pane_id,
                "position": shard_pos.to_dict(),
                "velocity": shard_velocity.to_dict(),
                "size_m": shard_size * 1.5,
                "area_m2": (shard_size * 1.5) ** 2,
                "thickness_m": pane.thickness_m,
                "sharpness": 0.5,
                "lifetime_s": 10.0 + self._rng.uniform(0, 5),
                "mesh_type": "chunk",
                "held_by_interlayer": True,
            })

        return shards

    def serialize(self) -> Dict[str, Any]:
        """Serialize glass solver state."""
        pane_states = {}
        for pane_id, pane in self._panes.items():
            pane_states[pane_id] = {
                "is_fractured": pane.is_fractured,
                "temper": pane.temper,
            }

        return {
            "format_version": 1,
            "pane_states": pane_states,
            "fractured_entities": list(self._fractured_entities),
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore glass solver state."""
        if data.get("format_version") != 1:
            raise ValueError(f"Unsupported glass solver state version: {data.get('format_version')}")

        self._fractured_entities = set(data.get("fractured_entities", []))

        # Restore pane fracture states
        for pane_id, state in data.get("pane_states", {}).items():
            if pane_id in self._panes:
                self._panes[pane_id].is_fractured = state.get("is_fractured", False)
