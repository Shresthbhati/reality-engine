"""
Error model and exception hierarchy (spec sec 82-83, SYSTEMS_DESIGN.md).

All Reality Engine errors derive from RealityEngineError.
Specific exceptions are raised for configuration, architecture, physics,
coordinate, entity, and serialization failures.

Usage:
    try:
        value = risky_operation()
    except NumericsError as e:
        # NaN/Inf hard failure
        logger.critical(f"Physics diverged: {e}")
        raise
    except PhysicsError as e:
        # Non-critical solver issue
        logger.warning(f"Physics warning: {e}")
        continue  # Skip this contact, continue simulation
"""

from __future__ import annotations


class RealityEngineError(Exception):
    """Base exception for all Reality Engine errors."""
    pass


class ConfigError(RealityEngineError):
    """Configuration loading or validation failed."""
    pass


class ArchitectureError(RealityEngineError):
    """Module invariant or interface contract violated."""
    pass


class SerializationError(RealityEngineError):
    """Serialization or deserialization failed."""
    pass


class PhysicsError(RealityEngineError):
    """Physics solver encountered an error."""
    pass


class NumericsError(PhysicsError):
    """
    NaN, Inf, or numerical divergence detected.

    This is a HARD FAILURE per spec §86. Simulation must stop immediately.
    Unlike other PhysicsErrors, NumericsError indicates state corruption
    that cannot be recovered from within the same run.
    """
    pass


class CoordinateFrameError(RealityEngineError):
    """Frame resolution failed, transform invalid."""
    pass


class EntityError(RealityEngineError):
    """Entity registry invariant violated."""
    pass


class WorldIRError(RealityEngineError):
    """WorldIR schema or validation error."""
    pass


class ValidationError(RealityEngineError):
    """Data validation failed (wrong type, out of bounds, etc.)."""
    pass


# Specific physics errors for categorization

class CollisionError(PhysicsError):
    """Collision detection or resolution failed."""
    pass


class ContactSolverError(PhysicsError):
    """Contact solver divergence or instability."""
    pass


class IntegrationError(PhysicsError):
    """Numerical integration error (e.g., step too large)."""
    pass


class MaterialError(PhysicsError):
    """Material property error (negative density, etc.)."""
    pass


# Error handling utilities

def is_hard_failure(error: Exception) -> bool:
    """
    Determine if error should stop simulation immediately.

    Hard failures:
    - NumericsError (NaN/Inf)
    - ArchitectureError (broken contract)
    - ConfigError (invalid setup)

    Non-critical (can skip contact and continue):
    - PhysicsError (other solver issues)
    - ValidationError (recoverable issue)
    """
    return isinstance(error, (NumericsError, ArchitectureError, ConfigError))


def error_severity(error: Exception) -> str:
    """
    Classify error severity for logging.

    Returns: "CRITICAL", "ERROR", or "WARNING"
    """
    if isinstance(error, NumericsError):
        return "CRITICAL"  # NaN/Inf is always critical
    elif isinstance(error, (ArchitectureError, ConfigError)):
        return "CRITICAL"  # Setup errors prevent simulation
    elif isinstance(error, (PhysicsError, EntityError, SerializationError)):
        return "ERROR"  # Solver/data issues
    else:
        return "WARNING"  # Other errors
