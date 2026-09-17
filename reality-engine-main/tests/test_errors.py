"""Tests for error handling system."""

import pytest
from engine.core.errors import (
    RealityEngineError, ConfigError, ArchitectureError,
    SerializationError, PhysicsError, NumericsError,
    CoordinateFrameError, EntityError, WorldIRError,
    ValidationError, CollisionError, ContactSolverError,
    IntegrationError, MaterialError,
    is_hard_failure, error_severity
)


def test_exception_hierarchy():
    """Test that exception hierarchy is correct."""
    # All should be RealityEngineError
    assert issubclass(ConfigError, RealityEngineError)
    assert issubclass(ArchitectureError, RealityEngineError)
    assert issubclass(SerializationError, RealityEngineError)
    assert issubclass(PhysicsError, RealityEngineError)
    assert issubclass(CoordinateFrameError, RealityEngineError)
    assert issubclass(EntityError, RealityEngineError)

    # NumericsError is PhysicsError
    assert issubclass(NumericsError, PhysicsError)
    assert issubclass(CollisionError, PhysicsError)
    assert issubclass(ContactSolverError, PhysicsError)
    assert issubclass(IntegrationError, PhysicsError)
    assert issubclass(MaterialError, PhysicsError)


def test_config_error():
    """Test ConfigError."""
    with pytest.raises(ConfigError):
        raise ConfigError("Invalid config")


def test_architecture_error():
    """Test ArchitectureError."""
    with pytest.raises(ArchitectureError):
        raise ArchitectureError("Contract violation")


def test_serialization_error():
    """Test SerializationError."""
    with pytest.raises(SerializationError):
        raise SerializationError("Failed to deserialize")


def test_physics_error():
    """Test PhysicsError."""
    with pytest.raises(PhysicsError):
        raise PhysicsError("Solver diverged")


def test_numerics_error():
    """Test NumericsError."""
    with pytest.raises(NumericsError):
        raise NumericsError("NaN detected")


def test_coordinate_frame_error():
    """Test CoordinateFrameError."""
    with pytest.raises(CoordinateFrameError):
        raise CoordinateFrameError("Frame not found")


def test_entity_error():
    """Test EntityError."""
    with pytest.raises(EntityError):
        raise EntityError("Dangling reference")


def test_world_ir_error():
    """Test WorldIRError."""
    with pytest.raises(WorldIRError):
        raise WorldIRError("Invalid WorldIR")


def test_validation_error():
    """Test ValidationError."""
    with pytest.raises(ValidationError):
        raise ValidationError("Value out of bounds")


def test_collision_error():
    """Test CollisionError."""
    with pytest.raises(CollisionError):
        raise CollisionError("Collision detection failed")


def test_contact_solver_error():
    """Test ContactSolverError."""
    with pytest.raises(ContactSolverError):
        raise ContactSolverError("Solver instability")


def test_integration_error():
    """Test IntegrationError."""
    with pytest.raises(IntegrationError):
        raise IntegrationError("Step too large")


def test_material_error():
    """Test MaterialError."""
    with pytest.raises(MaterialError):
        raise MaterialError("Negative density")


def test_is_hard_failure_numerics():
    """Test is_hard_failure detects NumericsError."""
    error = NumericsError("NaN")
    assert is_hard_failure(error) is True


def test_is_hard_failure_architecture():
    """Test is_hard_failure detects ArchitectureError."""
    error = ArchitectureError("Contract violation")
    assert is_hard_failure(error) is True


def test_is_hard_failure_config():
    """Test is_hard_failure detects ConfigError."""
    error = ConfigError("Invalid config")
    assert is_hard_failure(error) is True


def test_is_hard_failure_physics_not_hard():
    """Test is_hard_failure returns False for PhysicsError."""
    error = PhysicsError("Solver warning")
    assert is_hard_failure(error) is False


def test_is_hard_failure_validation_not_hard():
    """Test is_hard_failure returns False for ValidationError."""
    error = ValidationError("Out of bounds")
    assert is_hard_failure(error) is False


def test_error_severity_critical_numerics():
    """Test error_severity classifies NumericsError as CRITICAL."""
    error = NumericsError("NaN")
    assert error_severity(error) == "CRITICAL"


def test_error_severity_critical_architecture():
    """Test error_severity classifies ArchitectureError as CRITICAL."""
    error = ArchitectureError("Contract violation")
    assert error_severity(error) == "CRITICAL"


def test_error_severity_critical_config():
    """Test error_severity classifies ConfigError as CRITICAL."""
    error = ConfigError("Invalid config")
    assert error_severity(error) == "CRITICAL"


def test_error_severity_error_physics():
    """Test error_severity classifies PhysicsError as ERROR."""
    error = PhysicsError("Solver issue")
    assert error_severity(error) == "ERROR"


def test_error_severity_error_entity():
    """Test error_severity classifies EntityError as ERROR."""
    error = EntityError("Dangling ref")
    assert error_severity(error) == "ERROR"


def test_error_severity_error_serialization():
    """Test error_severity classifies SerializationError as ERROR."""
    error = SerializationError("Deserialization failed")
    assert error_severity(error) == "ERROR"


def test_error_severity_warning_other():
    """Test error_severity classifies other errors as WARNING."""
    error = ValueError("Some value error")
    assert error_severity(error) == "WARNING"


def test_error_message_propagation():
    """Test that error messages are propagated."""
    msg = "Specific error message"
    error = NumericsError(msg)
    assert msg in str(error)
