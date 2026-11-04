"""Unit tests for Blueprint Task 8.4: Geometric Spatial Relationships.

Verifies geometric computation of:
- supported_by (gravity support, contact plane)
- mounted_on (wall proximity and vertical overlap)
- part_of (structural containment and component membership)
- inside (room/space containment)
"""

import pytest

from engine.scene_graph import (
    GeometricRelationshipOptions,
    InferredRelationship,
    SceneGraph,
    apply_geometric_relationships,
    get_entity_bounds,
    infer_geometric_relationships,
)
from world_ir.schema_v1 import Entity, EntityType, Relationship, RelationshipKind
from world_ir.world_v1 import WorldIR


def test_supported_by_table_on_floor_and_vase_on_table():
    """Verify supported_by inferred for table on floor, and vase on table."""
    world = WorldIR("test_support_world")

    floor = Entity(
        id="floor_01",
        type=EntityType.FLOOR,
        name="Living Room Floor",
        custom_properties={"bounding_box": [(-5.0, -5.0, -0.2), (5.0, 5.0, 0.0)]},
    )
    table = Entity(
        id="table_01",
        type=EntityType.STRUCTURE,
        name="Coffee Table",
        semantic_labels=["furniture/table"],
        custom_properties={"bounding_box": [(0.0, 0.0, 0.0), (1.2, 0.8, 0.45)]},
    )
    vase = Entity(
        id="vase_01",
        type=EntityType.STRUCTURE,
        name="Ceramic Vase",
        semantic_labels=["decor/vase"],
        custom_properties={"bounding_box": [(0.4, 0.2, 0.45), (0.6, 0.4, 0.75)]},
    )

    world.entities = {
        floor.id: floor,
        table.id: table,
        vase.id: vase,
    }

    inferred = infer_geometric_relationships(world)
    pairs = {(r.source_id, r.kind, r.target_id) for r in inferred}

    # Table supported_by floor
    assert ("table_01", RelationshipKind.SUPPORTED_BY, "floor_01") in pairs
    assert ("floor_01", RelationshipKind.SUPPORTS, "table_01") in pairs

    # Vase supported_by table
    assert ("vase_01", RelationshipKind.SUPPORTED_BY, "table_01") in pairs
    assert ("table_01", RelationshipKind.SUPPORTS, "vase_01") in pairs


def test_mounted_on_tv_and_switch_on_wall():
    """Verify mounted_on inferred for TV and switch against wall surface."""
    world = WorldIR("test_mount_world")

    # North wall along X axis from (0, 4) to (6, 4), height 0 to 2.8m, thickness 0.2m
    wall = Entity(
        id="wall_north",
        type=EntityType.WALL,
        name="North Wall",
        semantic_labels=["architecture/wall"],
        custom_properties={"bounding_box": [(0.0, 3.9, 0.0), (6.0, 4.1, 2.8)]},
    )
    # TV mounted flush on wall at y=3.85, height 1.2 to 1.85m
    tv = Entity(
        id="tv_01",
        type=EntityType.STRUCTURE,
        name="OLED TV",
        semantic_labels=["electronics/tv"],
        custom_properties={"bounding_box": [(2.0, 3.82, 1.2), (3.4, 3.89, 2.0)]},
    )
    # Light switch mounted on same wall
    switch = Entity(
        id="switch_01",
        type=EntityType.STRUCTURE,
        name="Light Switch",
        semantic_labels=["fixture/switch"],
        custom_properties={"bounding_box": [(0.5, 3.85, 1.1), (0.6, 3.89, 1.2)]},
    )

    world.entities = {
        wall.id: wall,
        tv.id: tv,
        switch.id: switch,
    }

    inferred = infer_geometric_relationships(world)
    pairs = {(r.source_id, r.kind, r.target_id) for r in inferred}

    assert ("tv_01", RelationshipKind.MOUNTED_ON, "wall_north") in pairs
    assert ("switch_01", RelationshipKind.MOUNTED_ON, "wall_north") in pairs
    assert ("wall_north", RelationshipKind.ATTACHED_TO, "tv_01") in pairs


def test_part_of_opening_in_wall_and_component_in_sofa():
    """Verify part_of inferred for door opening in wall and subcomponents."""
    world = WorldIR("test_part_of_world")

    wall = Entity(
        id="wall_west",
        type=EntityType.WALL,
        name="West Wall",
        semantic_labels=["architecture/wall"],
        custom_properties={"bounding_box": [(0.0, 0.0, 0.0), (0.2, 5.0, 2.8)]},
    )
    door_opening = Entity(
        id="door_01",
        type=EntityType.DOOR,
        name="Bedroom Door",
        semantic_labels=["architecture/door"],
        custom_properties={"bounding_box": [(0.0, 1.0, 0.0), (0.2, 2.0, 2.1)]},
    )
    sofa = Entity(
        id="sofa_01",
        type=EntityType.STRUCTURE,
        name="3-Seat Sofa",
        component_ids=["sofa_cushion_01"],
        custom_properties={"bounding_box": [(1.0, 1.0, 0.0), (3.2, 2.0, 0.85)]},
    )
    cushion = Entity(
        id="sofa_cushion_01",
        type=EntityType.STRUCTURE,
        name="Sofa Cushion",
        custom_properties={"bounding_box": [(1.2, 1.2, 0.35), (1.8, 1.8, 0.5)]},
    )

    world.entities = {
        wall.id: wall,
        door_opening.id: door_opening,
        sofa.id: sofa,
        cushion.id: cushion,
    }

    inferred = infer_geometric_relationships(world)
    pairs = {(r.source_id, r.kind, r.target_id) for r in inferred}

    # Door opening part_of wall
    assert ("door_01", RelationshipKind.PART_OF, "wall_west") in pairs
    assert ("wall_west", RelationshipKind.CONTAINS, "door_01") in pairs

    # Cushion declared as component
    assert ("sofa_cushion_01", RelationshipKind.PART_OF, "sofa_01") in pairs
    assert ("sofa_01", RelationshipKind.CONTAINS, "sofa_cushion_01") in pairs


def test_inside_room_containment():
    """Verify inside relationship inferred when entity is contained within room volume."""
    world = WorldIR("test_room_world")

    room = Entity(
        id="room_living",
        type=EntityType.ROOM,
        name="Living Room",
        custom_properties={"bounding_box": [(0.0, 0.0, 0.0), (6.0, 5.0, 2.8)]},
    )
    bed = Entity(
        id="bed_01",
        type=EntityType.STRUCTURE,
        name="King Bed",
        semantic_labels=["furniture/bed"],
        custom_properties={"bounding_box": [(1.0, 1.0, 0.0), (3.0, 3.0, 0.9)]},
    )
    outside_tree = Entity(
        id="tree_01",
        type=EntityType.VEGETATION,
        name="Garden Tree",
        custom_properties={"bounding_box": [(12.0, 12.0, 0.0), (14.0, 14.0, 5.0)]},
    )

    world.entities = {
        room.id: room,
        bed.id: bed,
        outside_tree.id: outside_tree,
    }

    inferred = infer_geometric_relationships(world)
    pairs = {(r.source_id, r.kind, r.target_id) for r in inferred}

    assert ("bed_01", RelationshipKind.INSIDE, "room_living") in pairs
    assert ("room_living", RelationshipKind.CONTAINS, "bed_01") in pairs

    # Outside tree should not be inside room
    assert ("tree_01", RelationshipKind.INSIDE, "room_living") not in pairs


def test_apply_geometric_relationships_and_scenegraph():
    """Verify apply_geometric_relationships writes to WorldIR entities and SceneGraph queries work."""
    world = WorldIR("test_integration_world")

    floor = Entity(
        id="floor_01",
        type=EntityType.FLOOR,
        name="Floor",
        custom_properties={"bounding_box": [(0.0, 0.0, -0.2), (5.0, 5.0, 0.0)]},
    )
    desk = Entity(
        id="desk_01",
        type=EntityType.STRUCTURE,
        name="Office Desk",
        semantic_labels=["furniture/table"],
        custom_properties={"bounding_box": [(1.0, 1.0, 0.0), (2.2, 1.8, 0.75)]},
    )
    monitor = Entity(
        id="monitor_01",
        type=EntityType.STRUCTURE,
        name="Monitor",
        semantic_labels=["electronics/tv"],
        custom_properties={"bounding_box": [(1.4, 1.3, 0.75), (1.9, 1.5, 1.25)]},
    )
    room = Entity(
        id="office_room",
        type=EntityType.ROOM,
        name="Office",
        custom_properties={"bounding_box": [(0.0, 0.0, 0.0), (5.0, 5.0, 2.8)]},
    )

    world.entities = {
        floor.id: floor,
        desk.id: desk,
        monitor.id: monitor,
        room.id: room,
    }

    graph = SceneGraph(world)
    # Before applying, no relationships
    assert len(desk.relationships) == 0

    added = graph.compute_and_apply_relationships()
    assert added > 0

    # Test SceneGraph query methods
    desk_supporters = graph.supporters_of("desk_01")
    assert any(s.id == "floor_01" for s in desk_supporters)

    monitor_supporters = graph.supporters_of("monitor_01")
    assert any(s.id == "desk_01" for s in monitor_supporters)

    room_contents = graph.contents_of("office_room")
    room_content_ids = {c.id for c in room_contents}
    assert "desk_01" in room_content_ids
    assert "monitor_01" in room_content_ids

    # Second pass should be idempotent (no new edges added)
    added_second_pass = graph.compute_and_apply_relationships()
    assert added_second_pass == 0
