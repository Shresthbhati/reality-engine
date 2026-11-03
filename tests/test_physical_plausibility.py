"""Tests for Physical Plausibility and Volumetric Void Validation Gate (Wave 3, Blocks 20 & 22)."""

from __future__ import annotations

import pytest

from perception.reasoning.physical_plausibility import (
    CollisionRecord,
    FloatingRecord,
    PhysicalPlausibilityReport,
    ViolationSeverity,
    validate_physical_plausibility,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Provenance, Vector3
from world_ir.world_v1 import WorldIR


def test_collision_detection_and_separation_proposal():
    world = WorldIR()

    # Structural Wall at [0, 0, 0] to [4.0, 3.0, 0.2]
    g_wall = Geometry(id="g-w", type=GeometryType.BOX, bounds_min=Vector3(x=0.0, y=0.0, z=0.0), bounds_max=Vector3(x=4.0, y=3.0, z=0.2))
    e_wall = Entity(id="ent-wall", name="Living Room Wall", type=EntityType.WALL, geometry_ids=["g-w"], provenance=Provenance.OBSERVED)
    world.geometries["g-w"] = g_wall
    world.entities["ent-wall"] = e_wall

    # Sofa penetrating wall by 10cm: [1.0, 0.0, 0.10] to [3.0, 0.85, 0.90] (overlaps in Z from 0.10 to 0.20)
    g_sofa = Geometry(id="g-s", type=GeometryType.BOX, bounds_min=Vector3(x=1.0, y=0.0, z=0.10), bounds_max=Vector3(x=3.0, y=0.85, z=0.90))
    e_sofa = Entity(id="ent-sofa", name="Living Sofa", type=EntityType.STRUCTURE, geometry_ids=["g-s"],
                    custom_properties={"entity_class": "furniture.sofa"}, provenance=Provenance.RECONSTRUCTED)
    world.geometries["g-s"] = g_sofa
    world.entities["ent-sofa"] = e_sofa

    report = validate_physical_plausibility(world)

    assert isinstance(report, PhysicalPlausibilityReport)
    assert report.is_plausible is False
    assert len(report.collisions) == 1

    col = report.collisions[0]
    assert col.entity_a_id == "ent-sofa"
    assert col.entity_b_id == "ent-wall"
    assert col.penetration_depth_m == pytest.approx(0.10, abs=0.01)

    # Check auto-fix proposal
    assert len(report.fix_proposals) >= 1
    proposal = [p for p in report.fix_proposals if p.entity_id == "ent-sofa"][0]
    assert proposal.fix_type == "separate_penetration"
    assert proposal.translation_delta[2] != 0.0


def test_floating_furniture_detection_and_snap_proposal():
    world = WorldIR()

    # Floor at y = 0.0: [-5.0, -0.15, -5.0] to [5.0, 0.0, 5.0]
    g_floor = Geometry(id="g-fl", type=GeometryType.BOX, bounds_min=Vector3(x=-5.0, y=-0.15, z=-5.0), bounds_max=Vector3(x=5.0, y=0.0, z=5.0))
    e_floor = Entity(id="ent-floor", name="Room Floor", type=EntityType.FLOOR, geometry_ids=["g-fl"], provenance=Provenance.OBSERVED)
    world.geometries["g-fl"] = g_floor
    world.entities["ent-floor"] = e_floor

    # Floating Chair 20cm above floor: [0.0, 0.20, 0.0] to [0.5, 1.10, 0.5]
    g_chair = Geometry(id="g-ch", type=GeometryType.BOX, bounds_min=Vector3(x=0.0, y=0.20, z=0.0), bounds_max=Vector3(x=0.5, y=1.10, z=0.5))
    e_chair = Entity(id="ent-chair", name="Dining Chair", type=EntityType.STRUCTURE, geometry_ids=["g-ch"],
                     custom_properties={"entity_class": "furniture.chair"}, provenance=Provenance.RECONSTRUCTED)
    world.geometries["g-ch"] = g_chair
    world.entities["ent-chair"] = e_chair

    report = validate_physical_plausibility(world)

    assert report.is_plausible is False
    assert len(report.floating_entities) == 1

    flt = report.floating_entities[0]
    assert flt.entity_id == "ent-chair"
    assert flt.elevation_above_floor_m == pytest.approx(0.20, abs=0.01)

    # Check proposal to snap down by 0.20m
    snap = [p for p in report.fix_proposals if p.fix_type == "snap_to_floor"][0]
    assert snap.translation_delta[1] == pytest.approx(-0.20, abs=0.01)


def test_perfectly_plausible_room():
    world = WorldIR()

    # Floor at y = 0.0
    g_floor = Geometry(id="g-fl", type=GeometryType.BOX, bounds_min=Vector3(x=-5.0, y=-0.15, z=-5.0), bounds_max=Vector3(x=5.0, y=0.0, z=5.0))
    e_floor = Entity(id="ent-floor", name="Room Floor", type=EntityType.FLOOR, geometry_ids=["g-fl"], provenance=Provenance.OBSERVED)
    world.geometries["g-fl"] = g_floor
    world.entities["ent-floor"] = e_floor

    # Chair resting firmly on floor at y = 0.0
    g_chair = Geometry(id="g-ch", type=GeometryType.BOX, bounds_min=Vector3(x=0.0, y=0.0, z=0.0), bounds_max=Vector3(x=0.5, y=0.9, z=0.5))
    e_chair = Entity(id="ent-chair", name="Dining Chair", type=EntityType.STRUCTURE, geometry_ids=["g-ch"],
                     custom_properties={"entity_class": "furniture.chair"}, provenance=Provenance.OBSERVED)
    world.geometries["g-ch"] = g_chair
    world.entities["ent-chair"] = e_chair

    report = validate_physical_plausibility(world)
    assert report.is_plausible is True
    assert report.plausibility_score == pytest.approx(1.0)
    assert len(report.collisions) == 0
    assert len(report.floating_entities) == 0
    assert report.directly_observed_count == 2
    assert report.unobserved_inferred_count == 0
