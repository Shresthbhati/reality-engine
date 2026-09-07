"""Tests for the Inspector query facade over WorldIR."""

import pytest

from provenance import Provenance
from world_ir import (
    Branch,
    CausalRelation,
    Entity,
    EntityType,
    Geometry,
    Material,
    Measurement,
    Observation,
    PhysicalProperties,
    Relationship,
    RelationshipKind,
    Scenario,
    TemporalEvent,
    TemporalEventType,
    WorldIR,
)
from engine.inspector import Inspector


def _build_world() -> WorldIR:
    world = WorldIR(id="w1", name="test-building")

    glass = Material(
        id="mat-glass",
        name="Window Glass",
        class_name="glass",
        properties=PhysicalProperties(
            density=Measurement(value=2500.0, unit="kg/m^3", provenance=Provenance.ESTIMATED, confidence=0.6),
            mass=Measurement(value=12.0, unit="kg", provenance=Provenance.ESTIMATED, confidence=0.6),
        ),
        provenance=Provenance.ESTIMATED,
        confidence=0.6,
        observations=[Observation(id="obs-1", sensor_type="camera", confidence=0.9)],
    )
    world.materials[glass.id] = glass

    geom = Geometry(id="geom-window", vertex_count=8)
    world.geometries[geom.id] = geom

    building = Entity(
        id="building_01",
        type=EntityType.BUILDING,
        name="Main Building",
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.9,
    )
    window = Entity(
        id="window_01",
        type=EntityType.UNKNOWN,
        name="North Window",
        transform={"position": {"x": 3.0, "y": 0.0, "z": 2.0}},
        material_ids=[glass.id],
        geometry_ids=[geom.id],
        relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="building_01")],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.8,
        observations=[Observation(id="obs-2", sensor_type="lidar", confidence=0.95)],
    )
    world.entities[building.id] = building
    world.entities[window.id] = window

    impact = TemporalEvent(id="evt-impact", type=TemporalEventType.MODIFICATION, timestamp=1.0, entity_id="window_01")
    fracture = TemporalEvent(id="evt-fracture", type=TemporalEventType.DESTRUCTION, timestamp=1.01, entity_id="window_01")
    world.temporal_events[impact.id] = impact
    world.temporal_events[fracture.id] = fracture
    world.causal_relations.append(
        CausalRelation(cause_id="evt-impact", effect_id="evt-fracture", relationship_type="triggers", confidence=0.95)
    )

    world.branches["branch-alt"] = Branch(id="branch-alt", name="storm", parent_id=world.main_branch_id)
    world.scenarios["scenario-1"] = Scenario(id="scenario-1", name="high wind", branch_id="branch-alt")

    return world


class TestInspectorEntities:
    def test_get_entity(self):
        inspector = Inspector(_build_world())
        entity = inspector.get_entity("window_01")
        assert entity is not None
        assert entity.name == "North Window"

    def test_get_unknown_entity(self):
        inspector = Inspector(_build_world())
        assert inspector.get_entity("nope") is None

    def test_list_entities(self):
        inspector = Inspector(_build_world())
        entities = inspector.list_entities()
        assert len(entities) == 2

    def test_inspect_entity_resolves_materials_and_geometry(self):
        inspector = Inspector(_build_world())
        summary = inspector.inspect_entity("window_01")
        assert summary is not None
        assert len(summary.materials) == 1
        assert summary.materials[0].name == "Window Glass"
        assert len(summary.geometries) == 1
        assert summary.geometries[0].vertex_count == 8

    def test_inspect_entity_resolves_relationship_target_name(self):
        inspector = Inspector(_build_world())
        summary = inspector.inspect_entity("window_01")
        assert summary.relationships == [
            {"kind": "part_of", "target_id": "building_01", "target_name": "Main Building"}
        ]

    def test_inspect_unknown_entity_returns_none(self):
        inspector = Inspector(_build_world())
        assert inspector.inspect_entity("nope") is None


class TestInspectorEvidenceAndMeasurements:
    def test_get_evidence(self):
        inspector = Inspector(_build_world())
        evidence = inspector.get_evidence("window_01")
        assert len(evidence) == 1
        assert evidence[0].sensor_type == "lidar"

    def test_get_evidence_unknown_entity(self):
        inspector = Inspector(_build_world())
        assert inspector.get_evidence("nope") == []

    def test_get_measurements(self):
        inspector = Inspector(_build_world())
        measurements = inspector.get_measurements("window_01")
        assert "mat-glass" in measurements
        assert measurements["mat-glass"]["density"].value == 2500.0
        assert measurements["mat-glass"]["mass"].value == 12.0

    def test_get_confidence(self):
        inspector = Inspector(_build_world())
        confidence = inspector.get_confidence("window_01")
        assert confidence["entity_provenance"] == "RECONSTRUCTED"
        assert confidence["entity_confidence"] == 0.8
        assert confidence["min_material_confidence"] == 0.6


class TestInspectorMaterialsAndRelationships:
    def test_get_materials(self):
        inspector = Inspector(_build_world())
        materials = inspector.get_materials("window_01")
        assert len(materials) == 1
        assert materials[0].class_name == "glass"

    def test_get_materials_none_for_entity_without_materials(self):
        inspector = Inspector(_build_world())
        assert inspector.get_materials("building_01") == []

    def test_get_relationships(self):
        inspector = Inspector(_build_world())
        rels = inspector.get_relationships("window_01")
        assert rels[0]["kind"] == "part_of"

    def test_find_supporting(self):
        world = _build_world()
        world.entities["beam_01"] = Entity(
            id="beam_01", name="Beam",
            relationships=[Relationship(kind=RelationshipKind.SUPPORTS, target_id="window_01")],
        )
        inspector = Inspector(world)
        assert inspector.find_supporting("window_01") == ["beam_01"]

    def test_find_supporting_none(self):
        inspector = Inspector(_build_world())
        assert inspector.find_supporting("window_01") == []


class TestInspectorDistanceAndQuality:
    def test_measure_distance(self):
        inspector = Inspector(_build_world())
        distance = inspector.measure_distance("building_01", "window_01")
        assert distance == pytest.approx(3.6055512755)

    def test_measure_distance_missing_transform(self):
        world = _build_world()
        world.entities["no_transform"] = Entity(id="no_transform", transform=None)
        inspector = Inspector(world)
        assert inspector.measure_distance("building_01", "no_transform") is None

    def test_measure_distance_unknown_entity(self):
        inspector = Inspector(_build_world())
        assert inspector.measure_distance("building_01", "nope") is None

    def test_query_by_provenance(self):
        inspector = Inspector(_build_world())
        reconstructed = inspector.query_by_provenance(Provenance.RECONSTRUCTED)
        assert len(reconstructed) == 2

    def test_low_confidence_entities(self):
        inspector = Inspector(_build_world())
        low = inspector.low_confidence_entities(threshold=0.85)
        ids = [e.id for e in low]
        assert "window_01" in ids
        assert "building_01" not in ids


class TestInspectorEventsAndCausality:
    def test_get_events_for_entity(self):
        inspector = Inspector(_build_world())
        events = inspector.get_events("window_01")
        assert [e.id for e in events] == ["evt-impact", "evt-fracture"]

    def test_get_all_events(self):
        inspector = Inspector(_build_world())
        assert len(inspector.get_events()) == 2

    def test_get_causal_chain(self):
        inspector = Inspector(_build_world())
        chain = inspector.get_causal_chain("evt-impact")
        assert chain == ["evt-impact", "evt-fracture"]

    def test_why(self):
        inspector = Inspector(_build_world())
        causes = inspector.why("evt-fracture")
        assert len(causes) == 1
        assert causes[0].cause_id == "evt-impact"

    def test_why_no_cause(self):
        inspector = Inspector(_build_world())
        assert inspector.why("evt-impact") == []


class TestInspectorBranchesAndSummary:
    def test_list_branches(self):
        inspector = Inspector(_build_world())
        branches = inspector.list_branches()
        assert any(b.name == "storm" for b in branches)

    def test_list_scenarios(self):
        inspector = Inspector(_build_world())
        scenarios = inspector.list_scenarios()
        assert any(s.name == "high wind" for s in scenarios)

    def test_summary(self):
        inspector = Inspector(_build_world())
        summary = inspector.summary()
        assert summary["entities"] == 2
        assert summary["materials"] == 1
        assert summary["geometries"] == 1
        assert summary["temporal_events"] == 2
        assert summary["causal_relations"] == 1
        assert summary["branches"] == 1
        assert summary["scenarios"] == 1
