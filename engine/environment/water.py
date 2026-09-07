"""Water body simulation: depth, volume, buoyancy."""

from dataclasses import dataclass
from typing import Dict

from engine.core.logging import get_logger


@dataclass(frozen=True)
class WaterConfig:
    """Configuration for water physics simulation."""

    water_density_kg_m3: float = 1000.0
    """Density of fresh water in kg/m³."""

    gravity_m_s2: float = 9.81
    """Gravitational acceleration in m/s²."""

    seed: int = 42
    """Random seed for deterministic water behavior."""


class WaterBody:
    """A single water body (pond, lake, etc.) with depth and volume."""

    def __init__(self, body_id: str, surface_area_m2: float, depth_m: float = 0.0):
        """Initialize a water body.

        Args:
            body_id: Unique identifier for this water body.
            surface_area_m2: Surface area in square meters.
            depth_m: Current depth in meters (default 0.0).

        Raises:
            ValueError: If surface_area_m2 <= 0.
        """
        if surface_area_m2 <= 0:
            raise ValueError(
                f"surface_area_m2 must be > 0, got {surface_area_m2}"
            )
        self.body_id = body_id
        self.surface_area_m2 = surface_area_m2
        self.depth_m = depth_m

    @property
    def volume_m3(self) -> float:
        """Compute volume as surface_area_m2 * depth_m."""
        return self.surface_area_m2 * self.depth_m


class WaterState:
    """Manages multiple water bodies and computes water physics."""

    def __init__(self, config: WaterConfig):
        """Initialize water state with a config.

        Args:
            config: WaterConfig instance.
        """
        self.config = config
        self._bodies: Dict[str, WaterBody] = {}
        self.format_version = 1
        self._logger = get_logger("engine.environment.water")

    def register_body(self, body: WaterBody) -> None:
        """Register a water body.

        Args:
            body: WaterBody to register.

        Raises:
            ValueError: If a body with the same body_id is already registered.
        """
        if body.body_id in self._bodies:
            raise ValueError(
                f"Water body '{body.body_id}' is already registered"
            )
        self._bodies[body.body_id] = body
        self._logger.info(
            "Water body registered",
            context={
                "body_id": body.body_id,
                "surface_area_m2": body.surface_area_m2,
                "depth_m": body.depth_m,
            },
        )

    def get_body(self, body_id: str) -> WaterBody:
        """Retrieve a registered water body by ID.

        Args:
            body_id: ID of the water body.

        Returns:
            The WaterBody object.

        Raises:
            ValueError: If body_id is not registered.
        """
        if body_id not in self._bodies:
            raise ValueError(f"Unknown water body: '{body_id}'")
        return self._bodies[body_id]

    def buoyancy_force_n(
        self, body_id: str, submerged_volume_m3: float
    ) -> float:
        """Compute buoyant force on an object in water.

        Uses Archimedes' principle: F = ρ * g * V_submerged
        where ρ is water density, g is gravity, V_submerged is submerged volume.

        This is a P1 simplification: we assume the submerged volume is
        instantaneously available without flow dynamics or surface tension.
        A full model would include buoyancy over time as water level rises/falls.

        Args:
            body_id: ID of the water body containing the submerged object.
            submerged_volume_m3: Volume of the object submerged in cubic meters.

        Returns:
            Buoyant force in Newtons.

        Raises:
            ValueError: If submerged_volume_m3 < 0 or exceeds the body's volume.
        """
        if submerged_volume_m3 < 0:
            raise ValueError(
                f"submerged_volume_m3 must be >= 0, got {submerged_volume_m3}"
            )

        body = self.get_body(body_id)
        if submerged_volume_m3 > body.volume_m3:
            raise ValueError(
                f"submerged_volume_m3 ({submerged_volume_m3}) exceeds "
                f"body volume ({body.volume_m3}) for body '{body_id}'"
            )

        force = (
            self.config.water_density_kg_m3
            * self.config.gravity_m_s2
            * submerged_volume_m3
        )
        return force

    def serialize(self) -> dict:
        """Serialize all registered water bodies.

        Returns:
            Dictionary with format_version and bodies data.
        """
        bodies_data = {}
        for body_id, body in self._bodies.items():
            bodies_data[body_id] = {
                "surface_area_m2": body.surface_area_m2,
                "depth_m": body.depth_m,
            }

        return {
            "format_version": self.format_version,
            "bodies": bodies_data,
        }

    def deserialize(self, data: dict) -> None:
        """Deserialize water bodies from serialized data.

        Replaces any existing registered bodies.

        Args:
            data: Dictionary with format_version and bodies data.

        Raises:
            ValueError: If format_version does not match.
        """
        if data.get("format_version") != self.format_version:
            raise ValueError(
                f"Format version mismatch: expected {self.format_version}, "
                f"got {data.get('format_version')}"
            )

        self._bodies = {}
        for body_id, body_data in data.get("bodies", {}).items():
            body = WaterBody(
                body_id=body_id,
                surface_area_m2=body_data["surface_area_m2"],
                depth_m=body_data["depth_m"],
            )
            self.register_body(body)
