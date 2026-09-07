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


class RainState:
    """Rain state manager with intensity tracking and visibility effects.

    Maintains current rainfall intensity, band classification, and visibility
    reduction factor. Provides serialization with format versioning and an
    extension point for band-change events (Task 2).
    """

    def __init__(self, config: RainConfig):
        """Initialize rain state.

        Args:
            config: RainConfig with max intensity, visibility params, seed
        """
        self._config = config
        self._rng = DeterministicRNG(config.seed)
        self._logger = get_logger("engine.environment.rain")

        # Current state
        self._intensity_mm_h = 0.0
        self._previous_band = RainIntensity.NONE
        self._current_band = RainIntensity.NONE

        # Last update tracking
        self._last_tick = 0
        self._last_timestamp = 0.0

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
            # EXTENSION POINT: Task 2 will hook band-change event publishing here
            # self._publish_band_change_event()

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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rain state to dictionary.

        Returns:
            Dict with format_version, current intensity, and last update info
        """
        return {
            "format_version": 1,
            "intensity_mm_h": self._intensity_mm_h,
            "intensity_band": self._current_band.value,
            "last_tick": self._last_tick,
            "last_timestamp": self._last_timestamp,
        }

    def serialize(self) -> Dict[str, Any]:
        """Alias for to_dict() following established pattern.

        Returns:
            Serialized state dict
        """
        return self.to_dict()

    @classmethod
    def from_dict(cls, config: RainConfig, data: Dict[str, Any]) -> RainState:
        """Deserialize rain state from dictionary.

        Args:
            config: RainConfig to use for restored state
            data: Serialized state dict (must have format_version: 1)

        Returns:
            Restored RainState

        Raises:
            ValueError: If format_version is not 1
        """
        if data.get("format_version") != 1:
            raise ValueError(
                f"Unsupported rain state format_version: {data.get('format_version')}"
            )

        state = cls(config)
        state._intensity_mm_h = data.get("intensity_mm_h", 0.0)
        state._last_tick = data.get("last_tick", 0)
        state._last_timestamp = data.get("last_timestamp", 0.0)
        state._current_band = RainIntensity(data.get("intensity_band", "none"))

        return state

    @classmethod
    def deserialize(cls, config: RainConfig, data: Dict[str, Any]) -> RainState:
        """Alias for from_dict() following established pattern.

        Args:
            config: RainConfig to use for restored state
            data: Serialized state dict

        Returns:
            Restored RainState
        """
        return cls.from_dict(config, data)
