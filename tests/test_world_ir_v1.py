"""
Tests for WorldIR V1 complete schema implementation.

Tests cover:
- Round-trip serialization (to_dict → from_dict → equality)
- JSON round-trip (to_json → from_json → equality)
- Schema validation
- Provenance tracking
- Uncertainty handling
- Deterministic serialization
- Invalid schema rejection
"""

import json
import pytest
from datetime import datetime

from world_ir.schema_v1 import (
    Vector3, Quaternion, Measurement, Observation, PhysicalProperties,
    Material, Geometry, Surface, Relationship, Component, TemporalEvent,
    CausalRelation, Entity, GeometryType, SurfaceFinish, EntityType,
    RelationshipKind, TemporalEventType
)
from world_ir.world_v1 import WorldIR, Branch, Scenario, SimulationState, TemporalState
from world_ir.coordinates import Frame, Transform, IDENTITY_MATRIX
from provenance import Provenance, Uncertainty


# ===== Vector3 Tests =====

def test_vector3_roundtrip():
    v = Vector3(x=1.5, y=2.5, z=3.5)
    d = v.to_dict()
    v2 = Vector3.from_dict(d)
    assert v.x == v2.x and v.y == v2.y and v.z == v2.z


def test_vector3_zero():
    v = Vector3(x=0.0, y=0.0, z=0.0)
    d = v.to_dict()
    v2 = Vector3.from_dict(d)
    assert v == v2


# ===== Measurement Tests =====

def test_measurement_roundtrip():
    m = Measurement(value=1.5, unit="meter", precision=0.01, timestamp=100.0, provenance=Provenance.OBSERVED, confidence=0.95)
    d = m.to_dict()
    m2 = Measurement.from_dict(d)
    assert m.value == m2.value
    assert m.unit == m2.unit
    assert m.precision == m2.precision
    assert m.timestamp == m2.timestamp
    assert m.provenance == m2.provenance
    assert m.confidence == m2.confidence


def test_measurement_defaults():
    m = Measurement(value=5.0, unit="kg")
    d = m.to_dict()
    m2 = Measurement.from_dict(d)
    assert m2.precision == 0.01
    assert m2.timestamp is None
    assert m2.provenance == Provenance.UNKNOWN
    assert m2.confidence == 0.5


# ===== Observation Tests =====

def test_observation_roundtrip():
    obs = Observation(
        sensor_type="lidar",
        timestamp=123.45,
        frame_id="sensor_frame_0",
        data_uri="file:///data/scan.ply",
        data_hash="abc123def456",
        metadata={"range": 100.0, "fov": 120.0},
        confidence=0.9
    )
    d = obs.to_dict()
    obs2 = Observation.from_dict(d)
    assert obs.sensor_type == obs2.sensor_type
    assert obs.timestamp == obs2.timestamp
    assert obs.data_uri == obs2.data_uri
    assert obs.metadata == obs2.metadata


# ===== PhysicalProperties Tests =====

def test_physical_properties_roundtrip():
    props = PhysicalProperties(
        density=Measurement(value=2400, unit="kg/m^3"),
        friction_coefficient=0.7,
        restitution=0.1,
        custom_properties={"color": "gray"}
    )
    d = props.to_dict()
    props2 = PhysicalProperties.from_dict(d)
    assert props2.density.value == 2400
    assert props2.friction_coefficient == 0.7
    assert props2.custom_properties == {"color": "gray"}


# ===== Material Tests =====

def test_material_roundtrip():
    mat = Material(
        name="concrete",
        class_name="concrete",
        surface_finish=SurfaceFinish.ROUGH,
        color_rgb=(0.5, 0.5, 0.5),
        provenance=Provenance.OBSERVED,
        confidence=0.95
    )
    d = mat.to_dict()
    mat2 = Material.from_dict(d)
    assert mat.name == mat2.name
    assert mat.class_name == mat2.class_name
    assert mat.surface_finish == mat2.surface_finish
    assert mat.color_rgb == mat2.color_rgb
    assert mat.provenance == mat2.provenance


# ===== Geometry Tests =====

def test_geometry_roundtrip():
    geom = Geometry(
        type=GeometryType.MESH,
        lod_level=2,
        vertex_count=10000,
        triangle_count=5000,
        data_uri="file:///geometry/model.obj",
        bounds_min=Vector3(x=-10, y=-10, z=-10),
        bounds_max=Vector3(x=10, y=10, z=10),
        provenance=Provenance.RECONSTRUCTED
    )
    d = geom.to_dict()
    geom2 = Geometry.from_dict(d)
    assert geom.type == geom2.type
    assert geom.vertex_count == geom2.vertex_count
    assert geom.bounds_min.x == geom2.bounds_min.x


# ===== Surface Tests =====

def test_surface_roundtrip():
    surf = Surface(
        name="exterior_wall",
        geometry_id="geom_123",
        material_id="mat_456",
        area=Measurement(value=100.0, unit="m^2"),
        custom_attributes={"orientation": "north"}
    )
    d = surf.to_dict()
    surf2 = Surface.from_dict(d)
    assert surf.name == surf2.name
    assert surf.geometry_id == surf2.geometry_id
    assert surf.area.value == surf2.area.value


# ===== Relationship Tests =====

def test_relationship_roundtrip():
    rel = Relationship(
        kind=RelationshipKind.PART_OF,
        target_id="building_123",
        confidence=0.95,
        provenance=Provenance.OBSERVED,
        metadata={"attachment": "structural"}
    )
    d = rel.to_dict()
    rel2 = Relationship.from_dict(d)
    assert rel.kind == rel2.kind
    assert rel.target_id == rel2.target_id
    assert rel.confidence == rel2.confidence
    assert rel.metadata == rel2.metadata


# ===== Component Tests =====

def test_component_roundtrip():
    comp = Component(
        type="physics",
        data={"mass": 1000.0, "velocity": [0, 0, 0]},
        provenance=Provenance.GENERATED
    )
    d = comp.to_dict()
    comp2 = Component.from_dict(d)
    assert comp.type == comp2.type
    assert comp.data == comp2.data
    assert comp.provenance == comp2.provenance


# ===== TemporalEvent Tests =====

def test_temporal_event_roundtrip():
    evt = TemporalEvent(
        type=TemporalEventType.MODIFICATION,
        timestamp=100.0,
        entity_id="ent_123",
        state_before={"pos": [0, 0, 0]},
        state_after={"pos": [1, 1, 1]},
        provenance=Provenance.RECONSTRUCTED
    )
    d = evt.to_dict()
    evt2 = TemporalEvent.from_dict(d)
    assert evt.type == evt2.type
    assert evt.timestamp == evt2.timestamp
    assert evt.state_after == evt2.state_after


# ===== CausalRelation Tests =====

def test_causal_relation_roundtrip():
    causal = CausalRelation(
        cause_id="evt_1",
        effect_id="evt_2",
        relationship_type="triggers",
        confidence=0.9,
        provenance=Provenance.INFERRED
    )
    d = causal.to_dict()
    causal2 = CausalRelation.from_dict(d)
    assert causal.cause_id == causal2.cause_id
    assert causal.effect_id == causal2.effect_id
    assert causal.relationship_type == causal2.relationship_type


# ===== Entity Tests =====

def test_entity_roundtrip():
    entity = Entity(
        type=EntityType.BUILDING,
        name="Tower A",
        geometry_ids=["geom_1"],
        material_ids=["mat_1"],
        semantic_labels=["office", "highrise"],
        relationships=[
            Relationship(kind=RelationshipKind.PART_OF, target_id="complex_1")
        ],
        provenance=Provenance.OBSERVED,
        confidence=0.95
    )
    d = entity.to_dict()
    entity2 = Entity.from_dict(d)
    assert entity.type == entity2.type
    assert entity.name == entity2.name
    assert len(entity2.relationships) == 1
    assert entity2.relationships[0].kind == RelationshipKind.PART_OF


# ===== Branch Tests =====

def test_branch_roundtrip():
    branch = Branch(
        name="scenario_1",
        parent_id="branch_main",
        created_at=100.0,
        description="Fire scenario",
        metadata={"intensity": "high"}
    )
    d = branch.to_dict()
    branch2 = Branch.from_dict(d)
    assert branch.name == branch2.name
    assert branch.parent_id == branch2.parent_id
    assert branch.metadata == branch2.metadata


# ===== Scenario Tests =====

def test_scenario_roundtrip():
    scenario = Scenario(
        name="earthquake_magnitude_7",
        description="7.0 magnitude earthquake simulation",
        parameters={"magnitude": 7.0, "depth_km": 10.0},
        branch_id="branch_scenario_1"
    )
    d = scenario.to_dict()
    scenario2 = Scenario.from_dict(d)
    assert scenario.name == scenario2.name
    assert scenario.parameters == scenario2.parameters


# ===== SimulationState Tests =====

def test_simulation_state_roundtrip():
    sim = SimulationState(
        tick=100,
        timestamp=1.0,
        dt=0.01,
        body_states={"ent_1": {"pos": [0, 0, 0], "vel": [1, 2, 3]}},
        contact_count=5,
        total_energy=1000.0
    )
    d = sim.to_dict()
    sim2 = SimulationState.from_dict(d)
    assert sim.tick == sim2.tick
    assert sim.body_states == sim2.body_states


# ===== TemporalState Tests =====

def test_temporal_state_roundtrip():
    temp = TemporalState(
        current_time=3600.0,
        date_time="2024-01-01T12:00:00Z",
        season="winter",
        weather="snow",
        time_of_day=0.5
    )
    d = temp.to_dict()
    temp2 = TemporalState.from_dict(d)
    assert temp.current_time == temp2.current_time
    assert temp.weather == temp2.weather


# ===== WorldIR Complete Tests =====

def test_empty_world_roundtrip():
    """Test empty world serialization."""
    world = WorldIR(name="empty_world")
    d = world.to_dict()
    world2 = WorldIR.from_dict(d)
    assert world.id == world2.id
    assert world.name == world2.name
    assert len(world2.entities) == 0
    assert len(world2.geometries) == 0


def test_world_with_entity_roundtrip():
    """Test world with single entity."""
    world = WorldIR(name="simple_world")
    entity = Entity(
        id="building_1",
        type=EntityType.BUILDING,
        name="Office Tower"
    )
    world.entities[entity.id] = entity

    d = world.to_dict()
    world2 = WorldIR.from_dict(d)
    assert "building_1" in world2.entities
    assert world2.entities["building_1"].name == "Office Tower"


def test_world_with_complex_relationships_roundtrip():
    """Test world with entities, geometries, materials, and relationships."""
    world = WorldIR(name="complex_world")

    # Add materials
    mat = Material(id="mat_concrete", name="concrete", class_name="concrete")
    world.materials[mat.id] = mat

    # Add geometries
    geom = Geometry(
        id="geom_building",
        type=GeometryType.MESH,
        bounds_min=Vector3(x=0, y=0, z=0),
        bounds_max=Vector3(x=100, y=100, z=50)
    )
    world.geometries[geom.id] = geom

    # Add surfaces
    surf = Surface(
        id="surf_facade",
        geometry_id=geom.id,
        material_id=mat.id
    )
    world.surfaces[surf.id] = surf

    # Add entities
    building = Entity(
        id="building_1",
        type=EntityType.BUILDING,
        name="Main Tower",
        geometry_ids=[geom.id],
        material_ids=[mat.id],
        surface_ids=[surf.id]
    )
    world.entities[building.id] = building

    # Add component
    comp = Component(id="comp_physics", type="physics")
    world.components[comp.id] = comp
    building.component_ids = [comp.id]

    # Round-trip
    d = world.to_dict()
    world2 = WorldIR.from_dict(d)

    # Verify
    assert world2.entities["building_1"].name == "Main Tower"
    assert world2.materials["mat_concrete"].class_name == "concrete"
    assert world2.geometries["geom_building"].type == GeometryType.MESH
    assert world2.surfaces["surf_facade"].material_id == "mat_concrete"
    assert len(world2.entities["building_1"].geometry_ids) == 1


def test_world_json_roundtrip():
    """Test JSON serialization (most strict)."""
    world = WorldIR(name="json_test")
    entity = Entity(
        id="ent_1",
        type=EntityType.VEHICLE,
        name="Car",
        provenance=Provenance.OBSERVED,
        confidence=0.95
    )
    world.entities[entity.id] = entity

    # JSON round-trip
    json_str = world.to_json()
    world2 = WorldIR.from_json(json_str)

    # Verify exact match
    assert world.id == world2.id
    assert world.name == world2.name
    assert world2.entities["ent_1"].name == "Car"
    assert world2.entities["ent_1"].provenance == Provenance.OBSERVED
    assert world2.entities["ent_1"].confidence == 0.95


def test_world_deterministic_serialization():
    """Test that same world serializes to identical JSON (determinism)."""
    world = WorldIR(name="determinism_test")

    # Add multiple entities in random order
    for i in [3, 1, 2]:
        world.entities[f"ent_{i}"] = Entity(
            id=f"ent_{i}",
            type=EntityType.UNKNOWN,
            name=f"Entity {i}"
        )

    # Serialize twice
    json1 = world.to_json()
    json2 = world.to_json()

    # Must be byte-identical (tests sort_keys=True)
    assert json1 == json2

    # Parse both and verify order is consistent
    data1 = json.loads(json1)
    data2 = json.loads(json2)
    assert data1 == data2
    assert list(data1["entities"].keys()) == ["ent_1", "ent_2", "ent_3"]  # Sorted


def test_world_validation_dangling_entity_ref():
    """Test validation catches dangling entity references."""
    world = WorldIR()
    entity = Entity(id="ent_1")
    world.entities[entity.id] = entity

    # Add relationship to non-existent entity
    entity.relationships.append(
        Relationship(kind=RelationshipKind.PART_OF, target_id="ent_nonexistent")
    )

    issues = world.validate()
    assert len(issues) > 0
    assert any("dangling relationship" in issue for issue in issues)


def test_world_validation_dangling_geometry_ref():
    """Test validation catches dangling geometry references."""
    world = WorldIR()
    entity = Entity(id="ent_1")
    entity.geometry_ids = ["geom_nonexistent"]
    world.entities[entity.id] = entity

    issues = world.validate()
    assert len(issues) > 0
    assert any("unknown geometry" in issue for issue in issues)


def test_world_validation_dangling_material_ref():
    """Test validation catches dangling material references."""
    world = WorldIR()
    entity = Entity(id="ent_1")
    entity.material_ids = ["mat_nonexistent"]
    world.entities[entity.id] = entity

    issues = world.validate()
    assert len(issues) > 0
    assert any("unknown material" in issue for issue in issues)


def test_world_validation_valid_world():
    """Test validation passes for valid world."""
    world = WorldIR()
    mat = Material(id="mat_1")
    geom = Geometry(id="geom_1")
    entity = Entity(
        id="ent_1",
        material_ids=["mat_1"],
        geometry_ids=["geom_1"]
    )
    world.materials["mat_1"] = mat
    world.geometries["geom_1"] = geom
    world.entities["ent_1"] = entity

    issues = world.validate()
    assert len(issues) == 0


def test_provenance_tracking_observed_vs_generated():
    """Test that provenance correctly distinguishes observed from generated."""
    # Observed data
    obs_entity = Entity(
        id="ent_obs",
        type=EntityType.BUILDING,
        provenance=Provenance.OBSERVED,
        confidence=0.95
    )
    assert obs_entity.provenance == Provenance.OBSERVED
    assert obs_entity.confidence == 0.95

    # Generated data
    gen_entity = Entity(
        id="ent_gen",
        type=EntityType.BUILDING,
        provenance=Provenance.GENERATED,
        confidence=0.5
    )
    assert gen_entity.provenance == Provenance.GENERATED
    assert gen_entity.confidence == 0.5

    # Ensure they're different
    assert obs_entity.provenance != gen_entity.provenance

    # Round-trip and verify
    d_obs = obs_entity.to_dict()
    d_gen = gen_entity.to_dict()
    obs2 = Entity.from_dict(d_obs)
    gen2 = Entity.from_dict(d_gen)
    assert obs2.provenance == Provenance.OBSERVED
    assert gen2.provenance == Provenance.GENERATED


def test_uncertainty_representation():
    """Test uncertainty is properly represented and round-trips."""
    obs = Observation(
        sensor_type="camera",
        uncertainty=Uncertainty(
            confidence=0.9,
            note="Sensor calibration offset +0.1m"
        )
    )

    d = obs.to_dict()
    obs2 = Observation.from_dict(d)
    assert obs2.uncertainty.confidence == 0.9
    assert obs2.uncertainty.note == "Sensor calibration offset +0.1m"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
