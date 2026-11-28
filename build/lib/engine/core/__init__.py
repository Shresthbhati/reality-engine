"""
Engine core infrastructure (spec sec 82-83).

Provides: units, RNG, clock, jobs, math, configuration, logging, and error handling.
All foundational services that other modules depend on.
"""

from .units import Quantity, Unit
from .rng import DeterministicRNG
from .clock import SimulationContext
from .jobs import JobSystem
from .config import Config, PhysicsConfig, SimulationConfig, LoggingConfig, ConfigError, LogLevel as ConfigLogLevel, LogFormat
from .logging import Logger, get_logger, set_global_log_level, clear_loggers, Diagnostics, LogLevel
from .errors import (
    RealityEngineError, ConfigError, ArchitectureError, SerializationError,
    PhysicsError, NumericsError, CoordinateFrameError, EntityError, WorldIRError,
    ValidationError, CollisionError, ContactSolverError, IntegrationError, MaterialError,
    is_hard_failure, error_severity
)

__all__ = [
    # Units
    "Quantity", "Unit",
    # RNG
    "DeterministicRNG",
    # Clock
    "SimulationContext",
    # Jobs
    "JobSystem",
    # Config
    "Config", "PhysicsConfig", "SimulationConfig", "LoggingConfig",
    "ConfigLogLevel", "LogFormat",
    # Logging
    "Logger", "get_logger", "set_global_log_level", "clear_loggers",
    "Diagnostics", "LogLevel",
    # Errors
    "RealityEngineError", "ConfigError", "ArchitectureError",
    "SerializationError", "PhysicsError", "NumericsError",
    "CoordinateFrameError", "EntityError", "WorldIRError",
    "ValidationError", "CollisionError", "ContactSolverError",
    "IntegrationError", "MaterialError",
    "is_hard_failure", "error_severity",
]
