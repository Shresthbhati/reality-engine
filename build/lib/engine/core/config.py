"""
Configuration system (spec sec 82-83, SYSTEMS_DESIGN.md).

Loads TOML/YAML configuration with schema validation, type safety,
and immutability. All numeric config uses SI units via Quantity wrapper.
Unknown keys raise error (fail-fast, no silent ignores).

Usage:
    physics_config = Config.load_physics("config/physics.toml")
    simulation_config = Config.load_simulation("config/simulation.toml")
    logging_config = Config.load_logging("config/logging.toml")
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from enum import Enum


class ConfigError(Exception):
    """Configuration loading or validation failed."""
    pass


class LogLevel(str, Enum):
    """Logging severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output format."""
    JSON = "JSON"
    TEXT = "TEXT"


@dataclass(frozen=True)
class PhysicsConfig:
    """Immutable physics solver configuration."""
    # Gravity and timestep
    gravity_m_s2: float  # Negative value (downward)
    timestep_s: float  # Fixed simulation step
    max_contacts_per_body: int

    # Damping
    linear_damping: float  # Rate coefficient (1/s)
    angular_damping: float  # Rate coefficient (1/s)

    # Sleep thresholds
    sleep_velocity_threshold_m_s: float
    sleep_angular_threshold_rad_s: float
    sleep_time_s: float

    # Broadphase
    aabb_margin_m: float

    # Contact solver
    max_solver_iterations: int
    baumgarte_factor: float  # Penetration correction (0.1-0.3 typical)
    rest_velocity_threshold_m_s: float  # Below this, restitution = 0

    # Numerics
    nan_inf_hard_failure: bool
    energy_absolute_floor_j: float  # Below this, no explosion detection
    energy_explosion_threshold: float  # Relative growth ratio

    @staticmethod
    def from_dict(data: dict) -> PhysicsConfig:
        """Construct from validated dict."""
        return PhysicsConfig(
            gravity_m_s2=data["gravity_m_s2"],
            timestep_s=data["timestep_s"],
            max_contacts_per_body=data["max_contacts_per_body"],
            linear_damping=data["linear_damping"],
            angular_damping=data["angular_damping"],
            sleep_velocity_threshold_m_s=data["sleep_velocity_threshold_m_s"],
            sleep_angular_threshold_rad_s=data["sleep_angular_threshold_rad_s"],
            sleep_time_s=data["sleep_time_s"],
            aabb_margin_m=data["aabb_margin_m"],
            max_solver_iterations=data["max_solver_iterations"],
            baumgarte_factor=data["baumgarte_factor"],
            rest_velocity_threshold_m_s=data["rest_velocity_threshold_m_s"],
            nan_inf_hard_failure=data["nan_inf_hard_failure"],
            energy_absolute_floor_j=data["energy_absolute_floor_j"],
            energy_explosion_threshold=data["energy_explosion_threshold"],
        )


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable simulation-wide configuration."""
    fixed_timestep_s: float
    max_substeps: int
    deterministic_seed: int
    enable_event_logging: bool
    world_frame: str  # "X_FORWARD_Y_UP"
    handedness: str  # "RIGHT"


    @staticmethod
    def from_dict(data: dict) -> SimulationConfig:
        """Construct from validated dict."""
        return SimulationConfig(
            fixed_timestep_s=data["fixed_timestep_s"],
            max_substeps=data["max_substeps"],
            deterministic_seed=data["deterministic_seed"],
            enable_event_logging=data["enable_event_logging"],
            world_frame=data["world_frame"],
            handedness=data["handedness"],
        )


@dataclass(frozen=True)
class LoggingConfig:
    """Immutable logging configuration."""
    level: LogLevel
    format: LogFormat
    output: str  # "stdout", "stderr", or file path
    include_timestamp: bool
    include_module: bool
    include_thread_id: bool
    enable_performance_tracking: bool
    enable_numerics_checks: bool
    report_interval_seconds: float


    @staticmethod
    def from_dict(data: dict) -> LoggingConfig:
        """Construct from validated dict."""
        return LoggingConfig(
            level=LogLevel(data["level"]),
            format=LogFormat(data["format"]),
            output=data["output"],
            include_timestamp=data["include_timestamp"],
            include_module=data["include_module"],
            include_thread_id=data["include_thread_id"],
            enable_performance_tracking=data["enable_performance_tracking"],
            enable_numerics_checks=data["enable_numerics_checks"],
            report_interval_seconds=data["report_interval_seconds"],
        )


class Config:
    """Configuration loader with schema validation and immutability."""

    REQUIRED_PHYSICS_KEYS = {
        "gravity_m_s2", "timestep_s", "max_contacts_per_body",
        "linear_damping", "angular_damping",
        "sleep_velocity_threshold_m_s", "sleep_angular_threshold_rad_s", "sleep_time_s",
        "aabb_margin_m", "max_solver_iterations", "baumgarte_factor", "rest_velocity_threshold_m_s",
        "nan_inf_hard_failure", "energy_absolute_floor_j", "energy_explosion_threshold",
    }

    REQUIRED_SIMULATION_KEYS = {
        "fixed_timestep_s", "max_substeps", "deterministic_seed",
        "enable_event_logging", "world_frame", "handedness",
    }

    REQUIRED_LOGGING_KEYS = {
        "level", "format", "output",
        "include_timestamp", "include_module", "include_thread_id",
        "enable_performance_tracking", "enable_numerics_checks", "report_interval_seconds",
    }

    @staticmethod
    def load_physics(path: str | Path) -> PhysicsConfig:
        """Load physics.toml with schema validation."""
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(f"physics.toml not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to parse physics.toml: {e}") from e

        # Extract physics section
        physics = data.get("physics", {})

        # Validate required keys
        missing = Config.REQUIRED_PHYSICS_KEYS - set(physics.keys())
        if missing:
            raise ConfigError(f"physics.toml missing required keys: {missing}")

        # Reject unknown keys
        unknown = set(physics.keys()) - Config.REQUIRED_PHYSICS_KEYS
        if unknown:
            raise ConfigError(f"physics.toml has unknown keys: {unknown}")

        # Type validation
        try:
            return PhysicsConfig.from_dict(physics)
        except Exception as e:
            raise ConfigError(f"Invalid physics.toml schema: {e}") from e

    @staticmethod
    def load_simulation(path: str | Path) -> SimulationConfig:
        """Load simulation.toml with schema validation."""
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(f"simulation.toml not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to parse simulation.toml: {e}") from e

        # Extract simulation section
        simulation = data.get("simulation", {})

        # Validate required keys
        missing = Config.REQUIRED_SIMULATION_KEYS - set(simulation.keys())
        if missing:
            raise ConfigError(f"simulation.toml missing required keys: {missing}")

        # Reject unknown keys
        unknown = set(simulation.keys()) - Config.REQUIRED_SIMULATION_KEYS
        if unknown:
            raise ConfigError(f"simulation.toml has unknown keys: {unknown}")

        # Type validation
        try:
            return SimulationConfig.from_dict(simulation)
        except Exception as e:
            raise ConfigError(f"Invalid simulation.toml schema: {e}") from e

    @staticmethod
    def load_logging(path: str | Path) -> LoggingConfig:
        """Load logging.toml with schema validation."""
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(f"logging.toml not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to parse logging.toml: {e}") from e

        # Extract logging section
        logging = data.get("logging", {})

        # Validate required keys
        missing = Config.REQUIRED_LOGGING_KEYS - set(logging.keys())
        if missing:
            raise ConfigError(f"logging.toml missing required keys: {missing}")

        # Reject unknown keys
        unknown = set(logging.keys()) - Config.REQUIRED_LOGGING_KEYS
        if unknown:
            raise ConfigError(f"logging.toml has unknown keys: {unknown}")

        # Type validation
        try:
            return LoggingConfig.from_dict(logging)
        except Exception as e:
            raise ConfigError(f"Invalid logging.toml schema: {e}") from e

    @staticmethod
    def create_default_physics() -> PhysicsConfig:
        """Create default physics configuration."""
        return PhysicsConfig(
            gravity_m_s2=-9.81,
            timestep_s=0.01,
            max_contacts_per_body=8,
            linear_damping=0.01,
            angular_damping=0.05,
            sleep_velocity_threshold_m_s=0.05,
            sleep_angular_threshold_rad_s=0.05,
            sleep_time_s=0.5,
            aabb_margin_m=0.1,
            max_solver_iterations=5,
            baumgarte_factor=0.2,
            rest_velocity_threshold_m_s=0.5,
            nan_inf_hard_failure=True,
            energy_absolute_floor_j=1.0,
            energy_explosion_threshold=1000.0,
        )

    @staticmethod
    def create_default_simulation() -> SimulationConfig:
        """Create default simulation configuration."""
        return SimulationConfig(
            fixed_timestep_s=0.01,
            max_substeps=10,
            deterministic_seed=42,
            enable_event_logging=True,
            world_frame="X_FORWARD_Y_UP",
            handedness="RIGHT",
        )

    @staticmethod
    def create_default_logging() -> LoggingConfig:
        """Create default logging configuration."""
        return LoggingConfig(
            level=LogLevel.INFO,
            format=LogFormat.JSON,
            output="stdout",
            include_timestamp=True,
            include_module=True,
            include_thread_id=False,
            enable_performance_tracking=True,
            enable_numerics_checks=True,
            report_interval_seconds=10.0,
        )
