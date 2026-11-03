"""Unit tests for Multi-Room Topology Solver & Shared-Wall Deduplication (Wave 2, Block 9)."""

import pytest
from world_ir import Entity, EntityType, RelationshipKind, WorldIR
from perception.architecture.shared_walls import (
    SharedWallConsolidation,
    SharedWallDeduplicationReport,
    deduplicate_shared_walls,
)


def _create_wall_entity(
    wall_id: str,
    room_id: str,
    start: tuple,
    end: tuple,
    thickness: float = 0.12,
    height: float = 2.8,
    openings: list = None,
) -> Entity:
    return Entity(
        id=wall_id,
        type=EntityType.WALL,
        name=f"Wall {wall_id}",
        custom_properties={
            "room_id": room_id,
            "thickness_m": thickness,
            "height_m": height,
            "wall_slab": {
                "start_xyz_m": list(start),
                "end_xyz_m": list(end),
                "up": [0.0, 0.0, 1.0],
                "height_m": height,
                "thickness_m": thickness,
                "openings": openings or [],
            },
        },
    )


def test_solid_shared_wall_consolidation():
    world = WorldIR()
    # Room A: [0, 4] x [0, 4]
    world.entities["room-a"] = Entity(id="room-a", type=EntityType.ROOM, name="Room A")
    # Room B: [4, 8] x [0, 4]
    world.entities["room-b"] = Entity(id="room-b", type=EntityType.ROOM, name="Room B")

    # Wall A is at x = 3.95 (facing Room A)
    # Wall B is at x = 4.05 (facing Room B) - separation 0.10m
    w_a = _create_wall_entity("wall-a", "room-a", (3.95, 0.0, 0.0), (3.95, 4.0, 0.0), thickness=0.10)
    w_b = _create_wall_entity("wall-b", "room-b", (4.05, 4.0, 0.0), (4.05, 0.0, 0.0), thickness=0.10)
    world.entities[w_a.id] = w_a
    world.entities[w_b.id] = w_b

    report = deduplicate_shared_walls(world, remove_superseded_walls=True)

    assert isinstance(report, SharedWallDeduplicationReport)
    assert report.consolidated_wall_count == 1
    assert report.redundant_walls_removed == 2
    assert report.solid_room_pairs == [("room-a", "room-b")]
    assert report.permeable_room_pairs == []

    # Original walls removed
    assert "wall-a" not in world.entities
    assert "wall-b" not in world.entities

    # Shared wall exists
    shared_c = report.consolidations[0]
    assert shared_c.shared_wall_id in world.entities
    shared_ent = world.entities[shared_c.shared_wall_id]

    # Centerline is at x = 4.0
    slab = shared_ent.custom_properties["wall_slab"]
    assert slab["start_xyz_m"][0] == pytest.approx(4.0, abs=0.01)
    assert slab["end_xyz_m"][0] == pytest.approx(4.0, abs=0.01)
    # Thickness = 0.10 (separation) + 0.10 (half thicknesses) = 0.20m
    assert slab["thickness_m"] == pytest.approx(0.20, abs=0.01)

    # Both rooms connected to shared wall
    part_of_targets = [r.target_id for r in shared_ent.relationships if r.kind == RelationshipKind.PART_OF]
    assert set(part_of_targets) == {"room-a", "room-b"}

    # Topological adjacency between rooms established
    r_a = world.entities["room-a"]
    assert any(r.kind == RelationshipKind.ADJACENT_TO and r.target_id == "room-b" for r in r_a.relationships)
    assert "room-b" in r_a.custom_properties.get("adjacent_room_ids", [])


def test_permeable_shared_wall_with_door():
    world = WorldIR()
    world.entities["room-a"] = Entity(id="room-a", type=EntityType.ROOM, name="Room A")
    world.entities["room-b"] = Entity(id="room-b", type=EntityType.ROOM, name="Room B")

    # Door entity in world
    world.entities["door-1"] = Entity(id="door-1", type=EntityType.DOOR, name="Inter-room Door")

    door_op = {
        "entity_id": "door-1",
        "kind": "door",
        "offset_m": 1.2,
        "width_m": 0.9,
        "height_m": 2.1,
    }

    w_a = _create_wall_entity("wall-a", "room-a", (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), openings=[door_op])
    w_b = _create_wall_entity("wall-b", "room-b", (4.0, 4.0, 0.0), (4.0, 0.0, 0.0))
    world.entities[w_a.id] = w_a
    world.entities[w_b.id] = w_b

    report = deduplicate_shared_walls(world, remove_superseded_walls=True)

    assert report.consolidated_wall_count == 1
    assert report.permeable_room_pairs == [("room-a", "room-b")]

    shared_c = report.consolidations[0]
    assert shared_c.is_permeable is True
    assert shared_c.openings_count == 1

    # Check door entity updated
    door_ent = world.entities["door-1"]
    assert door_ent.custom_properties.get("shared_partition") is True
    assert set(door_ent.custom_properties.get("connects_room_ids", [])) == {"room-a", "room-b"}


def test_distant_walls_not_consolidated():
    world = WorldIR()
    world.entities["room-a"] = Entity(id="room-a", type=EntityType.ROOM)
    world.entities["room-b"] = Entity(id="room-b", type=EntityType.ROOM)

    # Walls 2.0 meters apart (separate buildings or courtyard)
    w_a = _create_wall_entity("wall-a", "room-a", (4.0, 0.0, 0.0), (4.0, 4.0, 0.0))
    w_b = _create_wall_entity("wall-b", "room-b", (6.0, 0.0, 0.0), (6.0, 4.0, 0.0))
    world.entities[w_a.id] = w_a
    world.entities[w_b.id] = w_b

    report = deduplicate_shared_walls(world)
    assert report.consolidated_wall_count == 0
    assert "wall-a" in world.entities
    assert "wall-b" in world.entities
