"""Spatial Reasoning and Physical Plausibility Validation (Wave 3, Blocks 20-22 & Directive Rule 8)."""

from perception.reasoning.physical_plausibility import (
    AutoFixProposal,
    CollisionRecord,
    ContainmentRecord,
    FloatingRecord,
    PhysicalPlausibilityReport,
    ViolationSeverity,
    validate_physical_plausibility,
)
from perception.reasoning.spatial_reasoning_engine import (
    ClearanceAccessibilityEngine,
    ClearanceReport,
    DoorwayClearanceRecord,
    LoadPathNode,
    LoadPathReport,
    SpatialReasoningEngine,
    SpatialReasoningReport,
    StructuralLoadPathEngine,
    UtilityNetworkEngine,
    UtilityNetworkReport,
    VolumetricVoidEngine,
    VolumetricVoidReport,
)

__all__ = [
    "ViolationSeverity",
    "CollisionRecord",
    "FloatingRecord",
    "ContainmentRecord",
    "AutoFixProposal",
    "PhysicalPlausibilityReport",
    "validate_physical_plausibility",
    "LoadPathNode",
    "LoadPathReport",
    "StructuralLoadPathEngine",
    "DoorwayClearanceRecord",
    "ClearanceReport",
    "ClearanceAccessibilityEngine",
    "VolumetricVoidReport",
    "VolumetricVoidEngine",
    "UtilityNetworkReport",
    "UtilityNetworkEngine",
    "SpatialReasoningReport",
    "SpatialReasoningEngine",
]
