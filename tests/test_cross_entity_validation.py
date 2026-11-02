"""Tests for automated cross-entity consistency validation rules (Task 7.5).
"""

from __future__ import annotations

from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.validation import ValidationSeverity, validate_world_ir


def _make_geom(gid: str, min_pt: tuple, max_pt: tuple) -> Geometry:
    return Geometry(
        id=gid,
        type=GeometryType.BOX,
        bounds_min=Vector3(*min_pt),
        bounds_max=Vector3(*max_pt),
    )


def test_cross_entity_support_cycle_detection():
    world = WorldIR(id="w_cycle", name="Cycle Test")
    g1 = _make_geom("g1", (0, 0, 0), (1, 1, 1))
    g2 = _make_geom("g2", (0, 0, 1), (1, 1, 2))
    world.geometries["g1"] = g1
    world.geometries["g2"] = g2

    e1 = Entity(
        id="e1",
        name="Box 1",
        type=EntityType.STRUCTURE,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g1"],
        relationships=[Relationship(kind=RelationshipKind.SUPPORTS, target_id="e2")],
        observations=["obs1"],
    )
    e2 = Entity(
        id="e2",
        name="Box 2",
        type=EntityType.STRUCTURE,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g2"],
        relationships=[Relationship(kind=RelationshipKind.SUPPORTS, target_id="e1")],
        observations=["obs2"],
    )
    world.entities["e1"] = e1
    world.entities["e2"] = e2

    report = validate_world_ir(world)
    cycle_errors = [i for i in report.errors if i.code == "cross_entity_cyclic_support"]
    assert len(cycle_errors) >= 1
    assert "cyclic vertical support" in cycle_errors[0].message


def test_cross_entity_vertical_support_valid_and_invalid():
    world = WorldIR(id="w_support", name="Support Test")
    g_floor = _make_geom("g_floor", (0, 0, 0), (5, 5, 0.2))
    g_table = _make_geom("g_table", (1, 1, 0.2), (2, 2, 0.9))
    world.geometries["g_floor"] = g_floor
    world.geometries["g_table"] = g_table

    floor = Entity(
        id="floor1",
        name="Floor",
        type=EntityType.FLOOR,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g_floor"],
        relationships=[Relationship(kind=RelationshipKind.SUPPORTS, target_id="table1")],
        observations=["obs1"],
    )
    table = Entity(
        id="table1",
        name="Table",
        type=EntityType.STRUCTURE,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g_table"],
        relationships=[Relationship(kind=RelationshipKind.RESTS_ON, target_id="floor1")],
        observations=["obs2"],
    )
    world.entities["floor1"] = floor
    world.entities["table1"] = table

    report = validate_world_ir(world)
    support_warnings = [i for i in report.warnings if i.code == "cross_entity_vertical_support_inconsistent"]
    assert len(support_warnings) == 0

    # Now make table float 2.5m above floor top
    g_floating = _make_geom("g_floating", (1, 1, 2.7), (2, 2, 3.4))
    world.geometries["g_floating"] = g_floating
    table.geometry_ids = ["g_floating"]

    report2 = validate_world_ir(world)
    support_warnings2 = [i for i in report2.warnings if i.code == "cross_entity_vertical_support_inconsistent"]
    assert len(support_warnings2) >= 1
    assert "floats" in support_warnings2[0].message


def test_cross_entity_opening_bounds_check():
    world = WorldIR(id="w_open", name="Opening Test")
    g_wall = _make_geom("g_wall", (0, 0, 0), (4, 0.2, 2.6))
    g_door = _make_geom("g_door", (1, 0, 0), (2, 0.2, 3.2))  # top 3.2m exceeds wall top 2.6m
    world.geometries["g_wall"] = g_wall
    world.geometries["g_door"] = g_door

    wall = Entity(
        id="wall1",
        name="Host Wall",
        type=EntityType.WALL,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g_wall"],
        observations=["obs1"],
    )
    door = Entity(
        id="door1",
        name="Oversized Door",
        type=EntityType.DOOR,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g_door"],
        custom_properties={"host_wall_id": "wall1"},
        relationships=[Relationship(kind=RelationshipKind.ATTACHED_TO, target_id="wall1")],
        observations=["obs2"],
    )
    world.entities["wall1"] = wall
    world.entities["door1"] = door

    report = validate_world_ir(world)
    door_warnings = [i for i in report.warnings if i.code == "cross_entity_opening_exceeds_wall_height"]
    assert len(door_warnings) >= 1
    assert "exceeds host wall" in door_warnings[0].message


def test_cross_entity_containment_disjoint_check():
    world = WorldIR(id="w_cont", name="Containment Test")
    g_room = _make_geom("g_room", (0, 0, 0), (6, 6, 3))
    g_chair_outside = _make_geom("g_chair", (20, 20, 0), (21, 21, 1))
    world.geometries["g_room"] = g_room
    world.geometries["g_chair"] = g_chair_outside

    room = Entity(
        id="room1",
        name="Living Room",
        type=EntityType.ROOM,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g_room"],
        relationships=[Relationship(kind=RelationshipKind.CONTAINS, target_id="chair1")],
        observations=["obs1"],
    )
    chair = Entity(
        id="chair1",
        name="Chair Far Away",
        type=EntityType.STRUCTURE,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g_chair"],
        observations=["obs2"],
    )
    world.entities["room1"] = room
    world.entities["chair1"] = chair

    report = validate_world_ir(world)
    disjoint_warnings = [i for i in report.warnings if i.code == "cross_entity_containment_disjoint"]
    assert len(disjoint_warnings) >= 1
    assert "disjoint" in disjoint_warnings[0].message


def test_cross_entity_attachment_disconnected_check():
    world = WorldIR(id="w_attach", name="Attachment Test")
    g_wall = _make_geom("g_wall", (0, 0, 0), (0.2, 5, 2.8))
    g_sconce = _make_geom("g_sconce", (10, 2, 1.8), (10.2, 2.2, 2.0))  # 10m away
    world.geometries["g_wall"] = g_wall
    world.geometries["g_sconce"] = g_sconce

    wall = Entity(
        id="wall1",
        name="Wall",
        type=EntityType.WALL,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g_wall"],
        observations=["obs1"],
    )
    sconce = Entity(
        id="sconce1",
        name="Wall Lamp",
        type=EntityType.INFRASTRUCTURE,
        provenance=Provenance.OBSERVED,
        geometry_ids=["g_sconce"],
        relationships=[Relationship(kind=RelationshipKind.ATTACHED_TO, target_id="wall1")],
        observations=["obs2"],
    )
    world.entities["wall1"] = wall
    world.entities["sconce1"] = sconce

    report = validate_world_ir(world)
    attach_warnings = [i for i in report.warnings if i.code == "cross_entity_attachment_disconnected"]
    assert len(attach_warnings) >= 1
    assert "apart" in attach_warnings[0].message
