"""Tests for spatial reasoning and relationship inference (Tasks 6.6 & 7.2)."""

from __future__ import annotations

import pytest

from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType, RelationshipKind
from perception.spatial_reasoning import (
    infer_spatial_relationships,
    get_attached_entities,
    get_room_entities,
)


def _make_world() -> WorldIR:
    w = WorldIR(id="w1", name="Spatial Test World")

    # Room
    room = Entity(
        id="room-1",
        name="Living Room",
        type=EntityType.ROOM,
        transform={"position": {"x": 2.0, "y": 2.0, "z": 1.2}},
        custom_properties={"footprint_area_m2": 16.0},
    )
    w.entities["room-1"] = room

    # Wall
    wall = Entity(
        id="wall-1",
        name="North Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 2.0, "y": 4.0, "z": 1.2}},
        semantic_labels=["wall"],
    )
    w.entities["wall-1"] = wall

    # Floor
    floor = Entity(
        id="floor-1",
        name="Floor Slab",
        type=EntityType.FLOOR,
        transform={"position": {"x": 2.0, "y": 2.0, "z": 0.0}},
        semantic_labels=["floor"],
    )
    w.entities["floor-1"] = floor

    # Ceiling
    ceiling = Entity(
        id="ceiling-1",
        name="Ceiling Slab",
        type=EntityType.CEILING,
        transform={"position": {"x": 2.0, "y": 2.0, "z": 2.6}},
        semantic_labels=["ceiling"],
    )
    w.entities["ceiling-1"] = ceiling

    # Socket fitting (near north wall)
    socket = Entity(
        id="socket-1",
        name="Power Socket",
        type=EntityType.UNKNOWN,
        transform={"position": {"x": 2.0, "y": 3.95, "z": 0.4}},
        semantic_labels=["socket"],
        custom_properties={"entity_class": "electrical.switchboard.module.socket"},
    )
    w.entities["socket-1"] = socket

    # Chair (resting on floor)
    chair = Entity(
        id="chair-1",
        name="Armchair",
        type=EntityType.UNKNOWN,
        transform={"position": {"x": 2.0, "y": 2.0, "z": 0.4}},
        semantic_labels=["furniture_candidate"],
        custom_properties={"entity_class": "furniture.chair"},
    )
    w.entities["chair-1"] = chair

    # Pendant light (near ceiling)
    light = Entity(
        id="light-1",
        name="Ceiling Pendant",
        type=EntityType.UNKNOWN,
        transform={"position": {"x": 2.0, "y": 2.0, "z": 2.45}},
        semantic_labels=["downlight"],
        custom_properties={"entity_class": "electrical.light_fitting"},
    )
    w.entities["light-1"] = light

    return w


def test_spatial_attachment_and_support_inference():
    world = _make_world()
    result = infer_spatial_relationships(world)

    assert result.attachments_created >= 2  # socket on wall, light on ceiling
    assert result.resting_created >= 1      # chair on floor
    assert result.containments_created >= 3 # room contains objects

    socket = world.entities["socket-1"]
    wall = world.entities["wall-1"]
    # Socket attached to wall
    sock_rel = next((r for r in socket.relationships if r.target_id == "wall-1"), None)
    assert sock_rel is not None
    assert sock_rel.kind == RelationshipKind.ATTACHED_TO
    # Wall supports socket
    wall_rel = next((r for r in wall.relationships if r.target_id == "socket-1"), None)
    assert wall_rel is not None
    assert wall_rel.kind == RelationshipKind.SUPPORTS

    chair = world.entities["chair-1"]
    floor = world.entities["floor-1"]
    # Chair rests on floor
    chair_rel = next((r for r in chair.relationships if r.target_id == "floor-1"), None)
    assert chair_rel is not None
    assert chair_rel.kind == RelationshipKind.RESTS_ON

    light = world.entities["light-1"]
    ceil = world.entities["ceiling-1"]
    # Light attached to ceiling
    light_rel = next((r for r in light.relationships if r.target_id == "ceiling-1"), None)
    assert light_rel is not None
    assert light_rel.kind == RelationshipKind.ATTACHED_TO

    # Query helpers
    attached_to_wall = get_attached_entities(world, "wall-1")
    assert any(e.id == "socket-1" for e in attached_to_wall)

    room_items = get_room_entities(world, "room-1")
    assert len(room_items) >= 3


def test_spatial_inference_idempotence():
    world = _make_world()
    res1 = infer_spatial_relationships(world)
    assert res1.total_edges_created > 0

    edge_counts_1 = {eid: len(e.relationships) for eid, e in world.entities.items()}

    # Second pass must be idempotent and not create duplicate edges
    res2 = infer_spatial_relationships(world)
    assert res2.total_edges_created == 0

    edge_counts_2 = {eid: len(e.relationships) for eid, e in world.entities.items()}
    assert edge_counts_1 == edge_counts_2
