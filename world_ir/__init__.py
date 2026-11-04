"""WorldIR: the canonical interchange representation (spec sec 6, V11 §6-§40).

Complete WorldIR implementation with schema v1:
- Entity system with full provenance tracking
- Geometry, materials, surfaces with observations
- Temporal events and causal relations
- Multiple coordinate frames and branching
- Deterministic serialization with validation
- Competing hypothesis support and multi-engine reconciliation
"""

# Legacy (foundation layer)
from .coordinates import Frame, Transform
from .entity import Entity as EntityLegacy, EntityRegistry
from .serialization import (
    save_world,
    load_world,
    export_world_package_zip,
    import_world_package_zip,
    WORLD_SAVE_FORMAT_VERSION,
)

# V1 (complete schema)
from .schema_v1 import (
    # Core types
    Vector3, Quaternion, Measurement, Observation, ConfidenceVector,
    # Data structures
    PhysicalProperties, Material, Geometry, Surface,
    Component, Relationship, TemporalEvent, CausalRelation, Entity,
    # Enums
    GeometryType, SurfaceFinish, EntityType, RelationshipKind,
    TemporalEventType
)
from .world_v1 import WorldIR, Branch, Scenario, SimulationState, TemporalState
from .geo import GeoAnchor
from .diff import diff_worlds, WorldDiff, EntityDiff, GeometryDiff, ChangeKind
from .temporal import (
    TemporalChangeKind,
    TemporalEntityChange,
    TemporalEpoch,
    TemporalAlignmentResult,
    TemporalDiffReport,
    align_temporal_epochs,
    diff_temporal_epochs,
)
from .incremental import (
    affected_closure, apply_incremental_update,
    DEFAULT_PROPAGATING_KINDS, IncrementalUpdateResult,
)
from .hypotheses import (
    CompetingHypothesis,
    HypothesisSet,
    ReconciliationStrategy,
    attach_competing_hypothesis,
    get_active_hypothesis,
    get_competing_hypotheses,
    reconcile_hypotheses,
)

__all__ = [
    # Coordinates
    "Frame",
    "Transform",
    # Legacy (for backward compatibility)
    "EntityRegistry",
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
    "ConfidenceVector",
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
    "GeoAnchor",
    "Branch",
    "Scenario",
    "SimulationState",
    "TemporalState",
    # Diff
    "diff_worlds",
    "WorldDiff",
    "EntityDiff",
    "GeometryDiff",
    "ChangeKind",
    # Temporal (Epic E07-1)
    "TemporalChangeKind",
    "TemporalEntityChange",
    "TemporalEpoch",
    "TemporalAlignmentResult",
    "TemporalDiffReport",
    "align_temporal_epochs",
    "diff_temporal_epochs",
    # Incremental
    "affected_closure",
    "DEFAULT_PROPAGATING_KINDS",
    "apply_incremental_update",
    "IncrementalUpdateResult",
    # Hypotheses & Reconciliation (Tasks 7.3 & 7.6)
    "CompetingHypothesis",
    "HypothesisSet",
    "ReconciliationStrategy",
    "attach_competing_hypothesis",
    "get_active_hypothesis",
    "get_competing_hypotheses",
    "reconcile_hypotheses",
]
