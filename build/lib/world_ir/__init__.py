"""WorldIR: the canonical interchange representation (spec sec 6, V11 §6-§40).

Complete WorldIR implementation with schema v1:
- Entity system with full provenance tracking
- Geometry, materials, surfaces with observations
- Temporal events and causal relations
- Multiple coordinate frames and branching
- Deterministic serialization with validation
"""

# Legacy (foundation layer)
from .coordinates import Frame, Transform
from .entity import Entity as EntityLegacy, EntityRegistry
from .world import WorldIR as WorldIRLegacy
from .serialization import save_world, load_world, WORLD_SAVE_FORMAT_VERSION

# V1 (complete schema)
from .schema_v1 import (
    # Core types
    Vector3, Quaternion, Measurement, Observation,
    # Data structures
    PhysicalProperties, Material, Geometry, Surface,
    Component, Relationship, TemporalEvent, CausalRelation, Entity,
    # Enums
    GeometryType, SurfaceFinish, EntityType, RelationshipKind,
    TemporalEventType
)
from .world_v1 import WorldIR, Branch, Scenario, SimulationState, TemporalState

__all__ = [
    # Coordinates
    "Frame",
    "Transform",
    # Legacy (for backward compatibility)
    "EntityRegistry",
    "WorldIRLegacy",
    "EntityLegacy",
    # Serialization
    "save_world",
    "load_world",
    "WORLD_SAVE_FORMAT_VERSION",
    # V1 Schema
    "Vector3",
    "Quaternion",
    "Measurement",
    "Observation",
    "PhysicalProperties",
    "Material",
    "Geometry",
    "Surface",
    "Component",
    "Relationship",
    "TemporalEvent",
    "CausalRelation",
    "Entity",
    "GeometryType",
    "SurfaceFinish",
    "EntityType",
    "RelationshipKind",
    "TemporalEventType",
    # V1 World
    "WorldIR",
    "Branch",
    "Scenario",
    "SimulationState",
    "TemporalState",
]
