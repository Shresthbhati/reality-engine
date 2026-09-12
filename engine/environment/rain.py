"""Rain intensity model: meteorological band classification and visibility effects.

Implements deterministic rain simulation with:
  - Intensity band classification (NONE, LIGHT, MODERATE, HEAVY, EXTREME)
  - Visibility reduction based on rainfall intensity
  - Serializable state tracking with format versioning
  - Extension point for band-change event publishing (Task 2)

Spec §19: RAIN SYSTEM
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

from engine.core.rng import DeterministicRNG
from engine.core.logging import get_logger
from engine.world.events import EventBus


class RainIntensity(str, Enum):
    """Rain intensity classification bands (meteorological standard).

    Bands follow common meteorological conventions for rainfall rate.
    """
    NONE = "none"          # 0 mm/h
    LIGHT = "light"        # 0-2.5 mm/h
    MODERATE = "moderate"  # 2.5-7.6 mm/h
    HEAVY = "heavy"        # 7.6-50 mm/h
    EXTREME = "extreme"    # >50 mm/h


@dataclass(frozen=True)
class RainConfig:
    """Rain system configuration (immutable).

    Attributes:
        max_intensity_mm_h: Maximum possible rainfall intensity (mm/h)
        visibility_reduction_per_mm_h: Fraction of visibility lost per mm/h
            (clamped so visibility factor never goes below 0.1)
        seed: Random seed for DeterministicRNG
    """
    max_intensity_mm_h: float = 50.0
    visibility_reduction_per_mm_h: float = 0.015
    seed: int = 42


class RainSurface:
    """Surface that accumulates rainfall over time.

    Tracks rainfall accumulation on a specific surface with absorption modeling.
    Accumulation is deterministic and depends on rainfall intensity, time step,
    and absorption coefficient.
    """

    def __init__(
        self,
        surface_id: str,
        area_m2: float,
        absorption_coefficient: float = 0.3,
    ):
        """Initialize a rain surface.

        Args:
            surface_id: Unique identifier for this surface
            area_m2: Surface area in square meters
            absorption_coefficient: Fraction of rain that infiltrates (0-1).
                Default 0.3 means 30% is absorbed, 70% accumulates.

        Raises:
            ValueError: If absorption_coefficient is outside [0, 1]
        """
        if not (0 <= absorption_coefficient <= 1):
            raise ValueError(
                f"absorption_coefficient must be in [0, 1], got {absorption_coefficient}"
            )

        self.surface_id = surface_id
        self.area_m2 = area_m2
        self.absorption_coefficient = absorption_coefficient
        self.accumulated_depth_m = 0.0


class RainState:
    """Rain state manager with intensity tracking and visibility effects.

    Maintains current rainfall intensity, band classification, and visibility
    reduction factor. Manages registered surfaces and their accumulated rainfall.
    Provides serialization with format versioning and an extension point for
    band-change events (Task 2).
    """

    def __init__(self, config: RainConfig, event_bus: Optional[EventBus] = None):
        """Initialize rain state.

        Args:
            config: RainConfig with max intensity, visibility params, seed
            event_bus: Optional EventBus to publish "rain.intensity_changed"
                events to on band changes. Default None preserves prior
                (Task 1/2) behavior of no event publishing.
        """
        self._config = config
        self._rng = DeterministicRNG(config.seed)
        self._logger = get_logger("engine.environment.rain")
        self._event_bus = event_bus

        # Current state
        self._intensity_mm_h = 0.0
        self._previous_band = RainIntensity.NONE
        self._current_band = RainIntensity.NONE

        # Last update tracking
        self._last_tick = 0
        self._last_timestamp = 0.0

        # Surface accumulation (Task 2)
        self._surfaces: Dict[str, RainSurface] = {}

    def register_surface(self, surface: RainSurface) -> None:
        """Register a surface for rainfall accumulation.

        Args:
            surface: RainSurface object to register

        Raises:
            ValueError: If a surface with this ID is already registered
        """
        if surface.surface_id in self._surfaces:
            raise ValueError(
                f"Surface already registered: {surface.surface_id}"
            )

        self._surfaces[surface.surface_id] = surface
        self._logger.info(
            "Rain surface registered",
            context={
                "surface_id": surface.surface_id,
                "area_m2": surface.area_m2,
                "absorption_coefficient": surface.absorption_coefficient,
            },
        )

    def step(self, dt: float, tick: int) -> None:
        """Advance surface accumulation for one time step.

        For each registered surface, accumulates rainfall depth based on:
        - Current rainfall intensity
        - Time step duration
        - Surface absorption coefficient

        Formula: accumulated_depth_m += intensity_m_s * dt * (1 - absorption_coefficient)

        Args:
            dt: Time step duration in seconds
            tick: Simulation tick (for logging)
        """
        if not self._surfaces or self._intensity_mm_h == 0:
            return

        intensity_m_s = self.intensity_m_s
        for surface in self._surfaces.values():
            # Accumulate depth: rain that reaches surface and doesn't infiltrate
            depth_increment = intensity_m_s * dt * (1 - surface.absorption_coefficient)
            surface.accumulated_depth_m += depth_increment

    def get_accumulation(self, surface_id: str) -> float:
        """Get accumulated rainfall depth on a surface.

        Args:
            surface_id: ID of registered surface

        Returns:
            Accumulated depth in meters

        Raises:
            ValueError: If surface is not registered
        """
        if surface_id not in self._surfaces:
            raise ValueError(f"Unknown surface: {surface_id}")

        return self._surfaces[surface_id].accumulated_depth_m

    def set_intensity(self, mm_per_hour: float, tick: int, timestamp: float) -> None:
        """Set rainfall intensity with band-change detection.

        Clamps intensity to [0, config.max_intensity_mm_h]. Raises ValueError
        if given a negative value before clamping (indicates caller bug).

        Records previous intensity band for potential event publishing in Task 2.

        Args:
            mm_per_hour: Requested rainfall intensity (mm/h)
            tick: Simulation tick at this update
            timestamp: World time at this update

        Raises:
            ValueError: If mm_per_hour is negative (before clamping)
        """
        if mm_per_hour < 0:
            raise ValueError(f"Negative intensity is invalid: {mm_per_hour} mm/h")

        # Clamp to valid range
        self._intensity_mm_h = min(mm_per_hour, self._config.max_intensity_mm_h)

        # Update tracking
        self._last_tick = tick
        self._last_timestamp = timestamp

        # Detect band change
        self._previous_band = self._current_band
        self._current_band = self.intensity_band()

        if self._previous_band != self._current_band:
            self._logger.info(
                "Rain band changed",
                context={
                    "previous_band": self._previous_band.value,
                    "current_band": self._current_band.value,
                    "intensity_mm_h": self._intensity_mm_h,
                    "tick": tick,
                    "timestamp": timestamp,
                },
            )
            if self._event_bus is not None:
                self._event_bus.publish(
                    event_type="rain.intensity_changed",
                    tick=tick,
                    timestamp=timestamp,
                    data={
                        "from_band": self._previous_band.value,
                        "to_band": self._current_band.value,
                        "intensity_mm_h": self._intensity_mm_h,
                    },
                )

    def intensity_band(self) -> RainIntensity:
        """Classify current intensity into a meteorological band.

        Returns:
            RainIntensity enum matching current rainfall rate
        """
        if self._intensity_mm_h == 0:
            return RainIntensity.NONE
        elif self._intensity_mm_h < 2.5:
            return RainIntensity.LIGHT
        elif self._intensity_mm_h < 7.6:
            return RainIntensity.MODERATE
        elif self._intensity_mm_h < 50.0:
            return RainIntensity.HEAVY
        else:
            return RainIntensity.EXTREME

    def visibility_factor(self) -> float:
        """Compute visibility reduction factor based on rainfall intensity.

        Formula: 1.0 - min(0.9, intensity_mm_h * visibility_reduction_per_mm_h)

        Result is clamped to [0.1, 1.0] so visibility never goes below 10%.

        Returns:
            Visibility factor in range [0.1, 1.0]
                - 1.0 = no rain, full visibility
                - 0.1 = maximum rain, 10% visibility (floor)
        """
        reduction = self._intensity_mm_h * self._config.visibility_reduction_per_mm_h
        clamped_reduction = min(0.9, reduction)
        return 1.0 - clamped_reduction

    @property
    def intensity_mm_h(self) -> float:
        """Current rainfall intensity in mm/h."""
        return self._intensity_mm_h

    @property
    def intensity_m_s(self) -> float:
        """Current rainfall intensity converted to m/s.

        Conversion: 1 mm/h = 1/3600000 m/s
        """
        return self._intensity_mm_h / 3600000.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get current rain diagnostics for debugging/inspection.

        Returns:
            Dict with current intensity (mm/h), band, visibility factor,
            registered surface count, and total accumulated volume (m3)
            across all surfaces (sum of depth * area per surface).
        """
        total_volume_m3 = sum(
            surface.accumulated_depth_m * surface.area_m2
            for surface in self._surfaces.values()
        )
        return {
            "intensity_mm_h": self._intensity_mm_h,
            "band": self._current_band.value,
            "visibility_factor": self.visibility_factor(),
            "surface_count": len(self._surfaces),
            "total_accumulated_volume_m3": total_volume_m3,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rain state to dictionary.

        Returns:
            Dict with format_version, current intensity, last update info,
            and registered surfaces with their accumulated depths
        """
        # Serialize surface states
        surface_states = {}
        for surface_id, surface in self._surfaces.items():
            surface_states[surface_id] = {
                "area_m2": surface.area_m2,
                "absorption_coefficient": surface.absorption_coefficient,
                "accumulated_depth_m": surface.accumulated_depth_m,
            }

        return {
            "format_version": 1,
            "intensity_mm_h": self._intensity_mm_h,
            "intensity_band": self._current_band.value,
            "last_tick": self._last_tick,
            "last_timestamp": self._last_timestamp,
            "surfaces": surface_states,
        }

    def serialize(self) -> Dict[str, Any]:
        """Alias for to_dict() following established pattern.

        Returns:
            Serialized state dict
        """
        return self.to_dict()

    def deserialize(self, data: dict) -> None:
        """Restore rain state from dictionary into this instance.

        Modifies the current RainState object in place, restoring all fields
        from the serialized data, including registered surfaces and their
        accumulated depths. Raises ValueError on unsupported format_version.

        Args:
            data: Serialized state dict (must have format_version: 1)

        Raises:
            ValueError: If format_version is not 1
        """
        if data.get("format_version") != 1:
            raise ValueError(
                f"Unsupported rain state format_version: {data.get('format_version')}"
            )

        self._intensity_mm_h = data.get("intensity_mm_h", 0.0)
        self._last_tick = data.get("last_tick", 0)
        self._last_timestamp = data.get("last_timestamp", 0.0)
        self._current_band = RainIntensity(data.get("intensity_band", "none"))

        # Restore surface states
        self._surfaces.clear()
        for surface_id, surface_data in data.get("surfaces", {}).items():
            surface = RainSurface(
                surface_id=surface_id,
                area_m2=surface_data.get("area_m2", 0.0),
                absorption_coefficient=surface_data.get("absorption_coefficient", 0.3),
            )
            surface.accumulated_depth_m = surface_data.get("accumulated_depth_m", 0.0)
            self._surfaces[surface_id] = surface
