"""Tests for configuration system."""

import pytest
import tempfile
from pathlib import Path
from engine.core.config import (
    Config, PhysicsConfig, SimulationConfig, LoggingConfig,
    ConfigError, LogLevel, LogFormat
)


def test_default_physics_config():
    """Test default physics configuration is valid."""
    config = Config.create_default_physics()
    assert config.gravity_m_s2 == -9.81
    assert config.timestep_s == 0.01
    assert config.max_contacts_per_body == 8
    assert config.linear_damping == 0.01
    assert config.nan_inf_hard_failure is True


def test_default_simulation_config():
    """Test default simulation configuration is valid."""
    config = Config.create_default_simulation()
    assert config.fixed_timestep_s == 0.01
    assert config.max_substeps == 10
    assert config.deterministic_seed == 42
    assert config.enable_event_logging is True
    assert config.world_frame == "X_FORWARD_Y_UP"


def test_default_logging_config():
    """Test default logging configuration is valid."""
    config = Config.create_default_logging()
    assert config.level == LogLevel.INFO
    assert config.format == LogFormat.JSON
    assert config.output == "stdout"
    assert config.include_timestamp is True


def test_physics_config_immutability():
    """Test that physics config is immutable (frozen dataclass)."""
    config = Config.create_default_physics()
    # frozen=True prevents attribute assignment
    with pytest.raises((RuntimeError, AttributeError)):
        config.gravity_m_s2 = -10.0


def test_simulation_config_immutability():
    """Test that simulation config is immutable (frozen dataclass)."""
    config = Config.create_default_simulation()
    with pytest.raises((RuntimeError, AttributeError)):
        config.fixed_timestep_s = 0.02


def test_logging_config_immutability():
    """Test that logging config is immutable (frozen dataclass)."""
    config = Config.create_default_logging()
    with pytest.raises((RuntimeError, AttributeError)):
        config.level = LogLevel.DEBUG


def test_load_physics_missing_file():
    """Test that loading missing physics.toml raises error."""
    with pytest.raises(ConfigError, match="not found"):
        Config.load_physics("/nonexistent/physics.toml")


def test_load_physics_from_toml():
    """Test loading physics configuration from TOML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "physics.toml"
        config_path.write_text("""
[physics]
gravity_m_s2 = -9.81
timestep_s = 0.01
max_contacts_per_body = 8
linear_damping = 0.01
angular_damping = 0.05
sleep_velocity_threshold_m_s = 0.05
sleep_angular_threshold_rad_s = 0.05
sleep_time_s = 0.5
aabb_margin_m = 0.1
max_solver_iterations = 5
baumgarte_factor = 0.2
rest_velocity_threshold_m_s = 0.5
nan_inf_hard_failure = true
energy_absolute_floor_j = 1.0
energy_explosion_threshold = 1000.0
""")
        config = Config.load_physics(config_path)
        assert config.gravity_m_s2 == -9.81
        assert config.timestep_s == 0.01


def test_load_physics_missing_required_key():
    """Test that missing required keys raise error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "physics.toml"
        config_path.write_text("""
[physics]
gravity_m_s2 = -9.81
# Missing timestep_s and other required fields
""")
        with pytest.raises(ConfigError, match="missing required keys"):
            Config.load_physics(config_path)


def test_load_physics_unknown_key():
    """Test that unknown keys raise error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "physics.toml"
        config_path.write_text("""
[physics]
gravity_m_s2 = -9.81
timestep_s = 0.01
max_contacts_per_body = 8
linear_damping = 0.01
angular_damping = 0.05
sleep_velocity_threshold_m_s = 0.05
sleep_angular_threshold_rad_s = 0.05
sleep_time_s = 0.5
aabb_margin_m = 0.1
max_solver_iterations = 5
baumgarte_factor = 0.2
rest_velocity_threshold_m_s = 0.5
nan_inf_hard_failure = true
energy_absolute_floor_j = 1.0
energy_explosion_threshold = 1000.0
unknown_key = "should_fail"
""")
        with pytest.raises(ConfigError, match="unknown keys"):
            Config.load_physics(config_path)


def test_load_simulation_from_toml():
    """Test loading simulation configuration from TOML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "simulation.toml"
        config_path.write_text("""
[simulation]
fixed_timestep_s = 0.01
max_substeps = 10
deterministic_seed = 42
enable_event_logging = true
world_frame = "X_FORWARD_Y_UP"
handedness = "RIGHT"
""")
        config = Config.load_simulation(config_path)
        assert config.fixed_timestep_s == 0.01
        assert config.deterministic_seed == 42


def test_load_logging_from_toml():
    """Test loading logging configuration from TOML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "logging.toml"
        config_path.write_text("""
[logging]
level = "DEBUG"
format = "JSON"
output = "stdout"
include_timestamp = true
include_module = true
include_thread_id = false
enable_performance_tracking = true
enable_numerics_checks = true
report_interval_seconds = 10.0
""")
        config = Config.load_logging(config_path)
        assert config.level == LogLevel.DEBUG
        assert config.format == LogFormat.JSON
