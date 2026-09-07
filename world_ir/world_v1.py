"""
WorldIR V1 Complete Implementation (V11 spec §6-§40, §103).

The World is the authoritative container for all reconstructed reality:
- Entities and relationships
- Geometry and materials
- Temporal state and events
- Causal chains
- Provenance and uncertainty tracking
- Multiple coordinate frames and branches

INVARIANT: Every serialization must be bit-identical after deserialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Any
from uuid import uuid4

from provenance import Provenance, Uncertainty
from .schema_v1 import (
    Entity, Geometry, Material, Surface, Component, Relationship,
    TemporalEvent, CausalRelation, Observation
)
from .coordinates import Frame, Transform


@dataclass
class Branch:
    """Alternative simulation timeline (branching scenarios)."""
    id: str = field(default_factory=lambda: f"branch-{uuid4()}")
    name: str = ""
    parent_id: Optional[str] = None  # Reference to parent branch (None = main)
    created_at: float = 0.0
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "description": self.description,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict) -> "Branch":
        return Branch(
            id=data.get("id", f"branch-{uuid4()}"),
            name=data.get("name", ""),
            parent_id=data.get("parent_id"),
            created_at=data.get("created_at", 0.0),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Scenario:
    """Simulation scenario definition."""
    id: str = field(default_factory=lambda: f"scenario-{uuid4()}")
    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)  # Scenario-specific parameters
    branch_id: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "branch_id": self.branch_id,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict) -> "Scenario":
        return Scenario(
            id=data.get("id", f"scenario-{uuid4()}"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            branch_id=data.get("branch_id", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SimulationState:
    """Physics simulation state snapshot."""
    tick: int = 0
    timestamp: float = 0.0
    dt: float = 0.01
    body_states: dict = field(default_factory=dict)  # entity_id -> {position, velocity, rotation, angular_velocity}
    contact_count: int = 0
    total_energy: float = 0.0
    provenance: Provenance = Provenance.GENERATED

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "dt": self.dt,
            "body_states": self.body_states,
            "contact_count": self.contact_count,
            "total_energy": self.total_energy,
            "provenance": self.provenance.value,
        }

    @staticmethod
    def from_dict(data: dict) -> "SimulationState":
        return SimulationState(
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", 0.0),
            dt=data.get("dt", 0.01),
            body_states=data.get("body_states", {}),
            contact_count=data.get("contact_count", 0),
            total_energy=data.get("total_energy", 0.0),
            provenance=Provenance(data.get("provenance", Provenance.GENERATED.value)),
        )


@dataclass
class TemporalState:
    """Time-dependent world state (seasons, day/night, etc.)."""
    current_time: float = 0.0  # Seconds since epoch or scenario start
    date_time: Optional[str] = None  # ISO 8601 format
    season: str = "unknown"  # spring, summer, fall, winter
    weather: str = "clear"  # clear, cloudy, rain, snow, storm
    time_of_day: float = 0.5  # 0.0-1.0 (0=midnight, 0.5=noon)
    simulation_states: list[SimulationState] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current_time": self.current_time,
            "date_time": self.date_time,
            "season": self.season,
            "weather": self.weather,
            "time_of_day": self.time_of_day,
            "simulation_states": [s.to_dict() for s in self.simulation_states],
        }

    @staticmethod
    def from_dict(data: dict) -> "TemporalState":
        return TemporalState(
            current_time=data.get("current_time", 0.0),
            date_time=data.get("date_time"),
            season=data.get("season", "unknown"),
            weather=data.get("weather", "clear"),
            time_of_day=data.get("time_of_day", 0.5),
            simulation_states=[
                SimulationState.from_dict(s) for s in data.get("simulation_states", [])
            ],
        )


@dataclass
class WorldIR:
    """Complete canonical world representation."""

    # Identity and Versioning
    id: str = field(default_factory=lambda: f"world-{uuid4()}")
    name: str = ""
    version: int = 1  # Schema version
    # created_at/modified_at are wall-clock bookkeeping metadata, not
    # simulation state -- but the same object flows through deterministic
    # paths (hashing, golden-file comparison, replay) where an unrequested
    # datetime.now() default would silently make two otherwise-identical
    # worlds compare unequal depending on when they were constructed.
    # Default to 0.0 ("unset"), matching Branch.created_at's convention
    # (see Branch dataclass above) -- callers that want a real wall-clock
    # timestamp (e.g. actually persisting a newly-authored world) must
    # stamp it explicitly via `WorldIR(..., created_at=time.time())` or
    # equivalent, rather than relying on an implicit default.
    created_at: float = 0.0
    modified_at: float = 0.0

    # Core Data
    entities: dict[str, Entity] = field(default_factory=dict)
    geometries: dict[str, Geometry] = field(default_factory=dict)
    materials: dict[str, Material] = field(default_factory=dict)
    surfaces: dict[str, Surface] = field(default_factory=dict)
    components: dict[str, Component] = field(default_factory=dict)

    # Temporal and Events
    temporal_state: TemporalState = field(default_factory=TemporalState)
    temporal_events: dict[str, TemporalEvent] = field(default_factory=dict)
    causal_relations: list[CausalRelation] = field(default_factory=list)

    # Branching and Scenarios
    main_branch_id: str = field(default_factory=lambda: f"branch-main-{uuid4()}")
    branches: dict[str, Branch] = field(default_factory=dict)
    scenarios: dict[str, Scenario] = field(default_factory=dict)

    # Coordinate System and Transforms
    coordinate_frame: Frame = Frame.WORLD
    transforms: dict[str, Transform] = field(default_factory=dict)  # entity_id -> Transform

    # Observations and Provenance
    observations: dict[str, Observation] = field(default_factory=dict)
    global_provenance: Provenance = Provenance.UNKNOWN
    global_confidence: float = 0.5
    global_uncertainty: Uncertainty = field(default_factory=Uncertainty)

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to deterministically-ordered JSON-compatible dict."""
        return {
            "schema_version": 1,
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "entities": {
                k: v.to_dict() for k, v in sorted(self.entities.items())
            },
            "geometries": {
                k: v.to_dict() for k, v in sorted(self.geometries.items())
            },
            "materials": {
                k: v.to_dict() for k, v in sorted(self.materials.items())
            },
            "surfaces": {
                k: v.to_dict() for k, v in sorted(self.surfaces.items())
            },
            "components": {
                k: v.to_dict() for k, v in sorted(self.components.items())
            },
            "temporal_state": self.temporal_state.to_dict(),
            "temporal_events": {
                k: v.to_dict() for k, v in sorted(self.temporal_events.items())
            },
            "causal_relations": [c.to_dict() for c in sorted(self.causal_relations, key=lambda x: x.cause_id)],
            "main_branch_id": self.main_branch_id,
            "branches": {
                k: v.to_dict() for k, v in sorted(self.branches.items())
            },
            "scenarios": {
                k: v.to_dict() for k, v in sorted(self.scenarios.items())
            },
            "coordinate_frame": self.coordinate_frame.value,
            "transforms": {
                k: v.to_dict() for k, v in sorted(self.transforms.items())
            },
            "observations": {
                k: v.to_dict() for k, v in sorted(self.observations.items())
            },
            "global_provenance": self.global_provenance.value,
            "global_confidence": self.global_confidence,
            "global_uncertainty": self.global_uncertainty.to_dict(),
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict) -> "WorldIR":
        """Deserialize from JSON-compatible dict."""
        # Validate schema version
        schema_version = data.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"Unsupported WorldIR schema version: {schema_version}")

        return WorldIR(
            id=data.get("id", f"world-{uuid4()}"),
            name=data.get("name", ""),
            version=data.get("version", 1),
            created_at=data.get("created_at", 0.0),
            modified_at=data.get("modified_at", 0.0),
            entities={
                k: Entity.from_dict(v) for k, v in data.get("entities", {}).items()
            },
            geometries={
                k: Geometry.from_dict(v) for k, v in data.get("geometries", {}).items()
            },
            materials={
                k: Material.from_dict(v) for k, v in data.get("materials", {}).items()
            },
            surfaces={
                k: Surface.from_dict(v) for k, v in data.get("surfaces", {}).items()
            },
            components={
                k: Component.from_dict(v) for k, v in data.get("components", {}).items()
            },
            temporal_state=TemporalState.from_dict(data.get("temporal_state", {})),
            temporal_events={
                k: TemporalEvent.from_dict(v) for k, v in data.get("temporal_events", {}).items()
            },
            causal_relations=[
                CausalRelation.from_dict(c) for c in data.get("causal_relations", [])
            ],
            main_branch_id=data.get("main_branch_id", f"branch-main-{uuid4()}"),
            branches={
                k: Branch.from_dict(v) for k, v in data.get("branches", {}).items()
            },
            scenarios={
                k: Scenario.from_dict(v) for k, v in data.get("scenarios", {}).items()
            },
            coordinate_frame=Frame(data.get("coordinate_frame", Frame.WORLD.value)),
            transforms={
                k: Transform.from_dict(v) for k, v in data.get("transforms", {}).items()
            },
            observations={
                k: Observation.from_dict(v) for k, v in data.get("observations", {}).items()
            },
            global_provenance=Provenance(data.get("global_provenance", Provenance.UNKNOWN.value)),
            global_confidence=data.get("global_confidence", 0.5),
            global_uncertainty=Uncertainty.from_dict(data.get("global_uncertainty", {})),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string with deterministic ordering."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            separators=(',', ': '),
        )

    @staticmethod
    def from_json(json_str: str) -> "WorldIR":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return WorldIR.from_dict(data)

    def validate(self) -> list[str]:
        """Validate world consistency and return list of issues."""
        issues = []

        # Check entity relationships reference existing entities
        for entity_id, entity in self.entities.items():
            for rel in entity.relationships:
                if rel.target_id not in self.entities and rel.target_id != entity_id:
                    issues.append(
                        f"Entity {entity_id} has dangling relationship to {rel.target_id}"
                    )

            # Check geometry references
            for geom_id in entity.geometry_ids:
                if geom_id not in self.geometries:
                    issues.append(
                        f"Entity {entity_id} references unknown geometry {geom_id}"
                    )

            # Check material references
            for mat_id in entity.material_ids:
                if mat_id not in self.materials:
                    issues.append(
                        f"Entity {entity_id} references unknown material {mat_id}"
                    )

            # Check surface references
            for surf_id in entity.surface_ids:
                if surf_id not in self.surfaces:
                    issues.append(
                        f"Entity {entity_id} references unknown surface {surf_id}"
                    )

            # Check component references
            for comp_id in entity.component_ids:
                if comp_id not in self.components:
                    issues.append(
                        f"Entity {entity_id} references unknown component {comp_id}"
                    )

        # Check surface references valid geometries and materials
        for surface_id, surface in self.surfaces.items():
            if surface.geometry_id and surface.geometry_id not in self.geometries:
                issues.append(f"Surface {surface_id} references unknown geometry {surface.geometry_id}")
            if surface.material_id and surface.material_id not in self.materials:
                issues.append(f"Surface {surface_id} references unknown material {surface.material_id}")

        # Check temporal events reference existing entities
        for event_id, event in self.temporal_events.items():
            if event.entity_id and event.entity_id not in self.entities:
                issues.append(f"Temporal event {event_id} references unknown entity {event.entity_id}")

        # Check causal relations reference existing events
        for causal in self.causal_relations:
            if causal.cause_id and causal.cause_id not in self.temporal_events:
                issues.append(f"Causal relation references unknown cause event {causal.cause_id}")
            if causal.effect_id and causal.effect_id not in self.temporal_events:
                issues.append(f"Causal relation references unknown effect event {causal.effect_id}")

        # Check branch hierarchy
        for branch_id, branch in self.branches.items():
            if branch.parent_id and branch.parent_id not in self.branches and branch.parent_id != self.main_branch_id:
                issues.append(f"Branch {branch_id} has unknown parent {branch.parent_id}")

        # Check scenario branch references
        for scenario_id, scenario in self.scenarios.items():
            if scenario.branch_id and scenario.branch_id not in self.branches:
                issues.append(f"Scenario {scenario_id} references unknown branch {scenario.branch_id}")

        return issues
