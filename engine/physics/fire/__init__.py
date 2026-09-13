"""Fire subsystem (simulation campaign sec 9): causal ignition/heat/spread/
suppression on physical material state."""

from .fire import (
    AMBIENT_TEMPERATURE_K,
    FireConfig,
    FireError,
    FireSolver,
    FuelCell,
    IgnitionSpec,
    IGNITION_TEMPERATURE_K_BY_CLASS,
    FUEL_LOAD_MJ_M2_BY_CLASS,
)

__all__ = [
    "AMBIENT_TEMPERATURE_K",
    "FireConfig",
    "FireError",
    "FireSolver",
    "FuelCell",
    "IgnitionSpec",
    "IGNITION_TEMPERATURE_K_BY_CLASS",
    "FUEL_LOAD_MJ_M2_BY_CLASS",
]
