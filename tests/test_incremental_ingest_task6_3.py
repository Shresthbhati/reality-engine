"""Tests for Task 6.3: Incremental Update on Ingest.

Validates:
1. run_incremental_ingest computes regional affected closure without rebuilding the entire world.
2. Untouched entities preserve exact IDs, geometries, and attributes (Scenario A12).
3. Deterministic WorldDiff accurately reports added, modified, and untouched entities.
4. Ingest with photo candidates: registers valid poses, refuses invalid photos without world corruption.
5. Entity removal properly seeds affected closure and updates diff accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.pipeline.incremental_ingest import IncrementalIngestResult, run_incremental_ingest
from reconstruction.calibration.camera import CameraIntrinsics
from world_ir import Entity, EntityType, Geometry, GeometryType, Relationship, RelationshipKind, WorldIR


def test_incremental_ingest_adds_wall_and_preserves_untouched():
    world_v1 = WorldIR(id="wld-inc-01", name="Living Room V1")
    world_v1.version = 1

    wall_1 = Entity(id="wall-1", name="Wall 1", type=EntityType.WALL)
    wall_2 = Entity(id="wall-2", name="Wall 2", type=EntityType.WALL)
    wall_3 = Entity(id="wall-3", name="Wall 3", type=EntityType.WALL)
    floor = Entity(id="floor-1", name="Floor", type=EntityType.FLOOR)

    chair = Entity(
        id="chair-1",
        name="Armchair",
        type=EntityType.STRUCTURE,
        relationships=[Relationship(kind=RelationshipKind.RESTS_ON, target_id="floor-1")],
    )
    geom_chair = Geometry(id="geom-chair-1", type=GeometryType.BOX)
    chair.geometry_ids = [geom_chair.id]

    room = Entity(
        id="room-1",
        name="Living Room",
        type=EntityType.ROOM,
        custom_properties={"boundary_element_ids": ["wall-1", "wall-2", "wall-3"]},
        relationships=[
            Relationship(kind=RelationshipKind.CONTAINS, target_id="wall-1"),
            Relationship(kind=RelationshipKind.CONTAINS, target_id="wall-2"),
            Relationship(kind=RelationshipKind.CONTAINS, target_id="wall-3"),
        ],
    )

    world_v1.entities = {
        wall_1.id: wall_1,
        wall_2.id: wall_2,
        wall_3.id: wall_3,
        floor.id: floor,
        chair.id: chair,
        room.id: room,
    }
    world_v1.geometries = {geom_chair.id: geom_chair}

    # Pass 2: Add missing 4th wall
    wall_4 = Entity(
        id="wall-4",
        name="Wall 4",
        type=EntityType.WALL,
        relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-1")],
    )
    geom_wall_4 = Geometry(id="geom-wall-4", type=GeometryType.BOX)
    wall_4.geometry_ids = [geom_wall_4.id]

    res = run_incremental_ingest(
        base_world=world_v1,
        new_entities=[wall_4],
        new_geometries=[geom_wall_4],
    )

    assert isinstance(res, IncrementalIngestResult)
    assert res.new_world.version == 2
    assert "wall-4" in res.changed_entity_ids
    assert "wall-4" in res.diff.added_entity_ids

    # Room is affected due to PART_OF dependency
    assert "room-1" in res.affected_entity_ids

    # Untouched entities are completely preserved
    for eid in ["wall-1", "wall-2", "wall-3", "floor-1", "chair-1"]:
        assert eid in res.reused_entity_ids
        assert eid not in res.diff.modified_entity_ids
        assert res.new_world.entities[eid] is world_v1.entities[eid]


def test_incremental_ingest_with_photo_registration():
    world = WorldIR(id="wld-photo-ingest", name="Photo Ingest Test")
    world.version = 1

    # 3D points
    np.random.seed(123)
    pts_3d = np.random.uniform(-1.0, 1.0, size=(20, 3))
    pts_3d[:, 2] = np.random.uniform(2.0, 5.0, size=(20,))
    intrinsics = CameraIntrinsics(width=800, height=600, fx=600.0, fy=600.0, cx=400.0, cy=300.0)

    u = intrinsics.fx * (pts_3d[:, 0] / pts_3d[:, 2]) + intrinsics.cx
    v = intrinsics.fy * (pts_3d[:, 1] / pts_3d[:, 2]) + intrinsics.cy
    pts_2d = np.column_stack([u, v])

    photo_good = {
        "photo_id": "cam_valid.jpg",
        "points_2d": pts_2d,
        "points_3d": pts_3d,
        "intrinsics": intrinsics,
    }
    photo_bad = {
        "photo_id": "cam_refused.jpg",
        "points_2d": pts_2d[:3],  # too few points
        "points_3d": pts_3d[:3],
        "intrinsics": intrinsics,
    }

    res = run_incremental_ingest(
        base_world=world,
        new_photos=[photo_good, photo_bad],
        min_registration_inliers=10,
    )

    assert "cam_valid.jpg" in res.registered_photos
    assert len(res.refused_photos) == 1
    assert res.refused_photos[0]["photo_id"] == "cam_refused.jpg"
    assert res.refused_photos[0]["status"] == "refused"


def test_incremental_ingest_removes_entity_and_updates_closure():
    world = WorldIR(id="wld-remove-test", name="Removal Test")
    world.version = 1

    desk = Entity(id="desk-1", name="Old Desk", type=EntityType.STRUCTURE)
    lamp = Entity(
        id="lamp-1",
        name="Desk Lamp",
        type=EntityType.STRUCTURE,
        relationships=[Relationship(kind=RelationshipKind.RESTS_ON, target_id="desk-1")],
    )

    world.entities = {desk.id: desk, lamp.id: lamp}

    res = run_incremental_ingest(
        base_world=world,
        removed_entity_ids=["desk-1"],
    )

    assert "desk-1" in res.diff.removed_entity_ids
    assert "desk-1" not in res.new_world.entities
    # Lamp rested on desk-1, so it is in the affected closure
    assert "lamp-1" in res.affected_entity_ids
