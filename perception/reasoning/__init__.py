"""Spatial Reasoning and Physical Plausibility Validation (Wave 3, Blocks 20-22)."""

from perception.reasoning.physical_plausibility import (
    AutoFixProposal,
    CollisionRecord,
    ContainmentRecord,
    FloatingRecord,
    PhysicalPlausibilityReport,
    ViolationSeverity,
    validate_physical_plausibility,
)

__all__ = [
    "ViolationSeverity",
    "CollisionRecord",
    "FloatingRecord",
    "ContainmentRecord",
    "AutoFixProposal",
    "PhysicalPlausibilityReport",
    "validate_physical_plausibility",
]
