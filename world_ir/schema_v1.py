"""
WorldIR V1 Schema: Complete canonical data model (V11 spec §6-§40, §103).

This module defines the authoritative schema for the Reality Engine's world
representation. Every type includes:
- Stable identity and versioning
- Serialization representation (JSON-compatible)
- Validation rules
- Provenance tracking (OBSERVED/RECONSTRUCTED/ESTIMATED/INFERRED/GENERATED/UNKNOWN/CONFLICT)
- Uncertainty representation
- Deterministic serialization (sorted keys, fixed-point floats where applicable)

CRITICAL INVARIANT: AI-generated hypotheses must NEVER silently become canonical
observed truth. Every field tracks its source via provenance enum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, Union, Literal
from uuid import uuid4
from datetime import datetime

from provenance import Provenance, Uncertainty


# ===== Basic Types & Enums =====

class GeometryType(str, Enum):
    """Supported geometry representations."""
    MESH = "mesh"  # Triangle mesh (vertices, faces, optional UVs, normals)
    POINTCLOUD = "pointcloud"  # Unorganized 3D points
    BOX = "box"  # Axis-aligned bounding box
    SPHERE = "sphere"  # Sphere (center, radius)
    CYLINDER = "cylinder"  # Cylinder (center, radius, height)
    CAPSULE = "capsule"  # Capsule (endpoints, radius)
    TERRAIN = "terrain"  # Height map or terrain mesh
    VOXEL = "voxel"  # Voxel grid
    IMPLICIT = "implicit"  # Implicit surface (SDF)
    UNKNOWN = "unknown"  # Geometry present but type not determined


class SurfaceFinish(str, Enum):
    """Material surface characteristics."""
    SMOOTH = "smooth"  # Polished, specular
    ROUGH = "rough"  # Matte, diffuse
    WEATHERED = "weathered"  # Oxidized, worn
    FRACTURED = "fractured"  # Broken, faceted
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """High-level entity classifications."""
    BUILDING = "building"
    STRUCTURE = "structure"
    VEHICLE = "vehicle"
    TERRAIN = "terrain"
    VEGETATION = "vegetation"
    WATER = "water"
    NATURAL_HAZARD = "natural_hazard"
    DEBRIS = "debris"
    SENSOR = "sensor"
    ROOM = "room"  # Enclosed interior space
    WALL = "wall"  # Vertical partition/boundary element
    FLOOR = "floor"  # Horizontal walking surface within a level
    CEILING = "ceiling"  # Upper interior boundary of a room
    ROOF = "roof"  # Exterior top covering of a building
    COLUMN = "column"  # Vertical load-bearing member
    BEAM = "beam"  # Horizontal load-bearing member
    DOOR = "door"  # Openable wall opening for passage
    WINDOW = "window"  # Wall opening for light/view
    STAIRS = "stairs"  # Vertical circulation element
    ROAD = "road"  # Vehicular travel surface
    CURB = "curb"  # Raised edge between road and sidewalk
    SIDEWALK = "sidewalk"  # Pedestrian walking surface adjacent to a road
    INFRASTRUCTURE = "infrastructure"  # General built infrastructure not otherwise classified
    UNKNOWN = "unknown"


class RelationshipKind(str, Enum):
    """Semantic relationships between entities."""
    PART_OF = "part_of"  # Structural containment (window part_of building)
    ATTACHED_TO = "attached_to"  # Mechanical attachment
    ADJACENT_TO = "adjacent_to"  # Spatial proximity
    COLLIDES_WITH = "collides_with"  # Contact (physics)
    CONTAINS = "contains"  # Conceptual containment
    OVERLAPS = "overlaps"  # Spatial overlap
    SUPPORTS = "supports"  # Load-bearing
    CROSSES = "crosses"  # Spatial intersection (without support)
    RESTS_ON = "rests_on"  # Gravity support
    UNKNOWN = "unknown"


class TemporalEventType(str, Enum):
    """Types of temporal changes."""
    CREATION = "creation"  # Entity first appeared
    OBSERVATION = "observation"  # Observed at this time
    MODIFICATION = "modification"  # State changed
    DESTRUCTION = "destruction"  # Entity ceased to exist
    MEASUREMENT = "measurement"  # Measurement taken
    SIMULATION_STATE = "simulation_state"  # Physics state snapshot


# ===== Core Data Structures =====

@dataclass
class Vector3:
    """3D point or vector."""
    x: float
    y: float
    z: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(data: dict) -> "Vector3":
        return Vector3(x=data["x"], y=data["y"], z=data["z"])


@dataclass
class Quaternion:
    """Rotation as quaternion (w, x, y, z)."""
    w: float
    x: float
    y: float
    z: float

    def to_dict(self) -> dict:
        return {"w": self.w, "x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(data: dict) -> "Quaternion":
        return Quaternion(w=data["w"], x=data["x"], y=data["y"], z=data["z"])


@dataclass
class Measurement:
    """A measured value with unit and precision."""
    value: float
    unit: str  # SI units only (meter, kg, second, etc.)
    precision: float = 0.01  # Standard deviation or margin of error
    timestamp: Optional[float] = None
    provenance: Provenance = Provenance.UNKNOWN
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "precision": self.precision,
            "timestamp": self.timestamp,
            "provenance": self.provenance.value,
            "confidence": self.confidence,
        }

    @staticmethod
    def from_dict(data: dict) -> "Measurement":
        return Measurement(
            value=data["value"],
            unit=data["unit"],
            precision=data.get("precision", 0.01),
            timestamp=data.get("timestamp"),
            provenance=Provenance(data.get("provenance", Provenance.UNKNOWN.value)),
            confidence=data.get("confidence", 0.5),
        )


@dataclass
class Observation:
    """Recorded sensory input (image, LiDAR, etc.)."""
    id: str = field(default_factory=lambda: f"obs-{uuid4()}")
    sensor_type: str = ""  # "camera", "lidar", "radar", "manual"
    timestamp: float = 0.0
    frame_id: str = ""  # Sensor frame in coordinate system
    data_uri: str = ""  # Reference to raw data (file:// or remote URL)
    data_hash: str = ""  # SHA256 of data for integrity
    metadata: dict = field(default_factory=dict)  # Sensor-specific (focal length, etc.)
    confidence: float = 1.0  # 0.0-1.0
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sensor_type": self.sensor_type,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "data_uri": self.data_uri,
            "data_hash": self.data_hash,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "Observation":
        return Observation(
            id=data.get("id", f"obs-{uuid4()}"),
            sensor_type=data.get("sensor_type", ""),
            timestamp=data.get("timestamp", 0.0),
            frame_id=data.get("frame_id", ""),
            data_uri=data.get("data_uri", ""),
            data_hash=data.get("data_hash", ""),
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 1.0),
            uncertainty=Uncertainty.from_dict(data.get("uncertainty", {})),
        )


@dataclass
class PhysicalProperties:
    """Material and mechanical properties."""
    density: Optional[Measurement] = None  # kg/m^3
    mass: Optional[Measurement] = None  # kg
    volume: Optional[Measurement] = None  # m^3
    friction_coefficient: Optional[float] = None  # Unitless
    restitution: Optional[float] = None  # Unitless (0-1)
    young_modulus: Optional[Measurement] = None  # Pa (elasticity)
    thermal_conductivity: Optional[Measurement] = None  # W/(m*K)
    specific_heat: Optional[Measurement] = None  # J/(kg*K)
    ignition_temperature: Optional[Measurement] = None  # Kelvin
    tensile_strength: Optional[Measurement] = None  # Pa
    shear_strength: Optional[Measurement] = None  # Pa
    custom_properties: dict = field(default_factory=dict)  # Domain-specific

    def to_dict(self) -> dict:
        return {
            "density": self.density.to_dict() if self.density else None,
            "mass": self.mass.to_dict() if self.mass else None,
            "volume": self.volume.to_dict() if self.volume else None,
            "friction_coefficient": self.friction_coefficient,
            "restitution": self.restitution,
            "young_modulus": self.young_modulus.to_dict() if self.young_modulus else None,
            "thermal_conductivity": self.thermal_conductivity.to_dict() if self.thermal_conductivity else None,
            "specific_heat": self.specific_heat.to_dict() if self.specific_heat else None,
            "ignition_temperature": self.ignition_temperature.to_dict() if self.ignition_temperature else None,
            "tensile_strength": self.tensile_strength.to_dict() if self.tensile_strength else None,
            "shear_strength": self.shear_strength.to_dict() if self.shear_strength else None,
            "custom_properties": self.custom_properties,
        }

    @staticmethod
    def from_dict(data: dict) -> "PhysicalProperties":
        return PhysicalProperties(
            density=Measurement.from_dict(data["density"]) if data.get("density") else None,
            mass=Measurement.from_dict(data["mass"]) if data.get("mass") else None,
            volume=Measurement.from_dict(data["volume"]) if data.get("volume") else None,
            friction_coefficient=data.get("friction_coefficient"),
            restitution=data.get("restitution"),
            young_modulus=Measurement.from_dict(data["young_modulus"]) if data.get("young_modulus") else None,
            thermal_conductivity=Measurement.from_dict(data["thermal_conductivity"]) if data.get("thermal_conductivity") else None,
            specific_heat=Measurement.from_dict(data["specific_heat"]) if data.get("specific_heat") else None,
            ignition_temperature=Measurement.from_dict(data["ignition_temperature"]) if data.get("ignition_temperature") else None,
            tensile_strength=Measurement.from_dict(data["tensile_strength"]) if data.get("tensile_strength") else None,
            shear_strength=Measurement.from_dict(data["shear_strength"]) if data.get("shear_strength") else None,
            custom_properties=data.get("custom_properties", {}),
        )


@dataclass
class Material:
    """Material definition with provenance."""
    id: str = field(default_factory=lambda: f"mat-{uuid4()}")
    name: str = ""
    class_name: str = "generic"  # wood, concrete, steel, glass, etc.
    surface_finish: SurfaceFinish = SurfaceFinish.UNKNOWN
    color_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)  # sRGB, 0-1
    properties: PhysicalProperties = field(default_factory=PhysicalProperties)
    provenance: Provenance = Provenance.UNKNOWN
    confidence: float = 0.5
    observations: list[Observation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "class_name": self.class_name,
            "surface_finish": self.surface_finish.value,
            "color_rgb": list(self.color_rgb),
            "properties": self.properties.to_dict(),
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "observations": [o.to_dict() for o in self.observations],
        }

    @staticmethod
    def from_dict(data: dict) -> "Material":
        return Material(
            id=data.get("id", f"mat-{uuid4()}"),
            name=data.get("name", ""),
            class_name=data.get("class_name", "generic"),
            surface_finish=SurfaceFinish(data.get("surface_finish", SurfaceFinish.UNKNOWN.value)),
            color_rgb=tuple(data.get("color_rgb", [0.5, 0.5, 0.5])),
            properties=PhysicalProperties.from_dict(data.get("properties", {})),
            provenance=Provenance(data.get("provenance", Provenance.UNKNOWN.value)),
            confidence=data.get("confidence", 0.5),
            observations=[Observation.from_dict(o) for o in data.get("observations", [])],
        )


@dataclass
class Geometry:
    """3D geometry representation."""
    id: str = field(default_factory=lambda: f"geom-{uuid4()}")
    type: GeometryType = GeometryType.UNKNOWN
    lod_level: int = 0  # Level of detail (0=coarse, higher=detail)
    vertex_count: Optional[int] = None
    triangle_count: Optional[int] = None
    data_uri: str = ""  # Reference to geometry file (OBJ, GLB, PLY, etc.)
    data_hash: str = ""  # SHA256 for integrity
    bounds_min: Optional[Vector3] = None  # Bounding box
    bounds_max: Optional[Vector3] = None
    provenance: Provenance = Provenance.UNKNOWN
    confidence: float = 0.5
    observations: list[Observation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "lod_level": self.lod_level,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "data_uri": self.data_uri,
            "data_hash": self.data_hash,
            "bounds_min": self.bounds_min.to_dict() if self.bounds_min else None,
            "bounds_max": self.bounds_max.to_dict() if self.bounds_max else None,
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "observations": [o.to_dict() for o in self.observations],
        }

    @staticmethod
    def from_dict(data: dict) -> "Geometry":
        return Geometry(
            id=data.get("id", f"geom-{uuid4()}"),
            type=GeometryType(data.get("type", GeometryType.UNKNOWN.value)),
            lod_level=data.get("lod_level", 0),
            vertex_count=data.get("vertex_count"),
            triangle_count=data.get("triangle_count"),
            data_uri=data.get("data_uri", ""),
            data_hash=data.get("data_hash", ""),
            bounds_min=Vector3.from_dict(data["bounds_min"]) if data.get("bounds_min") else None,
            bounds_max=Vector3.from_dict(data["bounds_max"]) if data.get("bounds_max") else None,
            provenance=Provenance(data.get("provenance", Provenance.UNKNOWN.value)),
            confidence=data.get("confidence", 0.5),
            observations=[Observation.from_dict(o) for o in data.get("observations", [])],
        )


@dataclass
class Surface:
    """Surface definition with material assignment."""
    id: str = field(default_factory=lambda: f"surf-{uuid4()}")
    name: str = ""
    geometry_id: str = ""  # Reference to Geometry
    material_id: str = ""  # Reference to Material
    area: Optional[Measurement] = None
    normal_direction: Optional[Vector3] = None
    custom_attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "geometry_id": self.geometry_id,
            "material_id": self.material_id,
            "area": self.area.to_dict() if self.area else None,
            "normal_direction": self.normal_direction.to_dict() if self.normal_direction else None,
            "custom_attributes": self.custom_attributes,
        }

    @staticmethod
    def from_dict(data: dict) -> "Surface":
        return Surface(
            id=data.get("id", f"surf-{uuid4()}"),
            name=data.get("name", ""),
            geometry_id=data.get("geometry_id", ""),
            material_id=data.get("material_id", ""),
            area=Measurement.from_dict(data["area"]) if data.get("area") else None,
            normal_direction=Vector3.from_dict(data["normal_direction"]) if data.get("normal_direction") else None,
            custom_attributes=data.get("custom_attributes", {}),
        )


@dataclass
class Relationship:
    """Entity relationship with provenance."""
    kind: RelationshipKind
    target_id: str
    confidence: float = 1.0
    provenance: Provenance = Provenance.OBSERVED
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "target_id": self.target_id,
            "confidence": self.confidence,
            "provenance": self.provenance.value,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict) -> "Relationship":
        return Relationship(
            kind=RelationshipKind(data.get("kind", RelationshipKind.UNKNOWN.value)),
            target_id=data["target_id"],
            confidence=data.get("confidence", 1.0),
            provenance=Provenance(data.get("provenance", Provenance.OBSERVED.value)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Component:
    """Generic component attached to entity (used by future subsystems)."""
    id: str = field(default_factory=lambda: f"comp-{uuid4()}")
    type: str = ""  # e.g., "physics", "destruction", "fire"
    data: dict = field(default_factory=dict)
    provenance: Provenance = Provenance.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "provenance": self.provenance.value,
        }

    @staticmethod
    def from_dict(data: dict) -> "Component":
        return Component(
            id=data.get("id", f"comp-{uuid4()}"),
            type=data.get("type", ""),
            data=data.get("data", {}),
            provenance=Provenance(data.get("provenance", Provenance.UNKNOWN.value)),
        )


@dataclass
class TemporalEvent:
    """Change in entity state over time."""
    id: str = field(default_factory=lambda: f"evt-{uuid4()}")
    type: TemporalEventType = TemporalEventType.OBSERVATION
    timestamp: float = 0.0
    entity_id: str = ""
    state_before: dict = field(default_factory=dict)
    state_after: dict = field(default_factory=dict)
    provenance: Provenance = Provenance.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "entity_id": self.entity_id,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "provenance": self.provenance.value,
        }

    @staticmethod
    def from_dict(data: dict) -> "TemporalEvent":
        return TemporalEvent(
            id=data.get("id", f"evt-{uuid4()}"),
            type=TemporalEventType(data.get("type", TemporalEventType.OBSERVATION.value)),
            timestamp=data.get("timestamp", 0.0),
            entity_id=data.get("entity_id", ""),
            state_before=data.get("state_before", {}),
            state_after=data.get("state_after", {}),
            provenance=Provenance(data.get("provenance", Provenance.UNKNOWN.value)),
        )


@dataclass
class CausalRelation:
    """Causal dependency between events."""
    cause_id: str = ""
    effect_id: str = ""
    relationship_type: str = ""  # e.g., "triggers", "enables", "prevents"
    confidence: float = 1.0
    provenance: Provenance = Provenance.INFERRED

    def to_dict(self) -> dict:
        return {
            "cause_id": self.cause_id,
            "effect_id": self.effect_id,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "provenance": self.provenance.value,
        }

    @staticmethod
    def from_dict(data: dict) -> "CausalRelation":
        return CausalRelation(
            cause_id=data.get("cause_id", ""),
            effect_id=data.get("effect_id", ""),
            relationship_type=data.get("relationship_type", ""),
            confidence=data.get("confidence", 1.0),
            provenance=Provenance(data.get("provenance", Provenance.INFERRED.value)),
        )


# ===== Entity =====

@dataclass
class Entity:
    """A world entity with full provenance tracking."""
    id: str = field(default_factory=lambda: f"ent-{uuid4()}")
    type: EntityType = EntityType.UNKNOWN
    name: str = ""
    transform: Optional[dict] = None  # Externalized transform (from coordinates system)
    geometry_ids: list[str] = field(default_factory=list)
    material_ids: list[str] = field(default_factory=list)
    surface_ids: list[str] = field(default_factory=list)
    component_ids: list[str] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    semantic_labels: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    temporal_events: list[TemporalEvent] = field(default_factory=list)
    provenance: Provenance = Provenance.UNKNOWN
    confidence: float = 0.5
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    custom_properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "transform": self.transform,
            "geometry_ids": list(self.geometry_ids),
            "material_ids": list(self.material_ids),
            "surface_ids": list(self.surface_ids),
            "component_ids": list(self.component_ids),
            "relationships": [r.to_dict() for r in self.relationships],
            "semantic_labels": list(self.semantic_labels),
            "observations": [o.to_dict() for o in self.observations],
            "temporal_events": [e.to_dict() for e in self.temporal_events],
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty.to_dict(),
            "custom_properties": self.custom_properties,
        }

    @staticmethod
    def from_dict(data: dict) -> "Entity":
        return Entity(
            id=data.get("id", f"ent-{uuid4()}"),
            type=EntityType(data.get("type", EntityType.UNKNOWN.value)),
            name=data.get("name", ""),
            transform=data.get("transform"),
            geometry_ids=list(data.get("geometry_ids", [])),
            material_ids=list(data.get("material_ids", [])),
            surface_ids=list(data.get("surface_ids", [])),
            component_ids=list(data.get("component_ids", [])),
            relationships=[Relationship.from_dict(r) for r in data.get("relationships", [])],
            semantic_labels=list(data.get("semantic_labels", [])),
            observations=[Observation.from_dict(o) for o in data.get("observations", [])],
            temporal_events=[TemporalEvent.from_dict(e) for e in data.get("temporal_events", [])],
            provenance=Provenance(data.get("provenance", Provenance.UNKNOWN.value)),
            confidence=data.get("confidence", 0.5),
            uncertainty=Uncertainty.from_dict(data.get("uncertainty", {})),
            custom_properties=data.get("custom_properties", {}),
        )
