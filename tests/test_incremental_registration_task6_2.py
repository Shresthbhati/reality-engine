"""Tests for Task 6.2: Incremental Photo Registration & Room Merge Refusal.

Validates:
1. register_photo_to_sparse_model registers valid photos with sufficient 2D-3D correspondences.
2. register_photo_to_sparse_model refuses non-matching or high-reprojection-error photos with explicit reason.
3. register_incremental_photos batch processes multiple candidates.
4. propose_room_merge refuses solid wall partition merge (Scenario A19).
5. propose_room_merge refuses merge for incompatible floor elevations (> 0.15m).
6. propose_room_merge accepts merge when dividing partition contains an opening.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.math import Quat, Vec3
from perception.architecture.room_join import propose_room_merge
from reconstruction.calibration.camera import CameraIntrinsics, PinholeCamera, CameraExtrinsics
from registration.photo_registration import (
    IncrementalRegistrationBatch,
    PhotoRegistrationResult,
    register_incremental_photos,
    register_photo_to_sparse_model,
)
from world_ir import Entity, EntityType, WorldIR


def _create_synthetic_correspondences():
    """Create a known 3D point cloud and project into a camera at (0, 0, -3)."""
    # 20 3D points in front of camera
    np.random.seed(42)
    pts_3d = np.random.uniform(-1.0, 1.0, size=(20, 3))
    pts_3d[:, 2] = np.random.uniform(2.0, 5.0, size=(20,))

    intrinsics = CameraIntrinsics(width=800, height=600, fx=600.0, fy=600.0, cx=400.0, cy=300.0)
    
    # Project with identity rotation, translation t = (0, 0, 0)
    u = intrinsics.fx * (pts_3d[:, 0] / pts_3d[:, 2]) + intrinsics.cx
    v = intrinsics.fy * (pts_3d[:, 1] / pts_3d[:, 2]) + intrinsics.cy
    pts_2d = np.column_stack([u, v])
    return pts_3d, pts_2d, intrinsics


def test_register_photo_success():
    pts_3d, pts_2d, intrinsics = _create_synthetic_correspondences()

    res = register_photo_to_sparse_model(
        image_id="photo_01.jpg",
        points_3d=pts_3d,
        points_2d=pts_2d,
        intrinsics=intrinsics,
        min_inliers=10,
        reprojection_error_threshold_px=4.0,
    )

    assert res.status == "registered"
    assert res.reprojection_error_px is not None
    assert res.reprojection_error_px < 2.0
    assert res.inlier_count >= 10
    assert res.position is not None
    assert res.rotation_quaternion is not None
    assert len(res.rotation_quaternion) == 4


def test_register_photo_refused_insufficient_points():
    intrinsics = CameraIntrinsics(width=800, height=600, fx=600.0, fy=600.0, cx=400.0, cy=300.0)
    pts_3d = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    pts_2d = np.array([[400.0, 300.0], [500.0, 300.0]])

    res = register_photo_to_sparse_model(
        image_id="photo_insufficient.jpg",
        points_3d=pts_3d,
        points_2d=pts_2d,
        intrinsics=intrinsics,
        min_inliers=6,
    )

    assert res.status == "refused"
    assert "too few 2d-3d correspondences" in res.refusal_reason.lower()
    assert res.inlier_count < 6


def test_register_photo_refused_high_reprojection_error():
    pts_3d, pts_2d, intrinsics = _create_synthetic_correspondences()
    # Corrupt 2D points with severe noise so RANSAC inliers fail threshold
    corrupted_2d = pts_2d + np.random.uniform(50.0, 100.0, size=pts_2d.shape)

    res = register_photo_to_sparse_model(
        image_id="photo_noisy.jpg",
        points_3d=pts_3d,
        points_2d=corrupted_2d,
        intrinsics=intrinsics,
        min_inliers=15,
        reprojection_error_threshold_px=2.0,
    )

    assert res.status == "refused"
    assert res.refusal_reason is not None


def test_register_incremental_photos_batch():
    pts_3d, pts_2d, intrinsics = _create_synthetic_correspondences()

    candidates = [
        {"image_id": "good_1.jpg", "points_3d": pts_3d, "points_2d": pts_2d, "intrinsics": intrinsics},
        {"image_id": "bad_too_few.jpg", "points_3d": pts_3d[:3], "points_2d": pts_2d[:3], "intrinsics": intrinsics},
    ]

    batch = register_incremental_photos(candidates)
    assert isinstance(batch, IncrementalRegistrationBatch)
    assert len(batch.results) == 2
    assert batch.registered_count == 1
    assert batch.refused_count == 1
    assert batch.results[0].status == "registered"
    assert batch.results[1].status == "refused"


def test_room_merge_refusal_and_acceptance():
    world = WorldIR(id="wld-merge-test", name="Merge Test")

    # Room A and Room B
    wall_shared = Entity(
        id="wall-partition",
        name="Partition Wall",
        type=EntityType.WALL,
        custom_properties={"wall_slab": {"openings": []}},
    )
    room_a = Entity(
        id="room-a",
        name="Room A",
        type=EntityType.ROOM,
        custom_properties={"boundary_element_ids": ["wall-partition"], "floor_elevation_m": 0.0},
    )
    room_b = Entity(
        id="room-b",
        name="Room B",
        type=EntityType.ROOM,
        custom_properties={"boundary_element_ids": ["wall-partition"], "floor_elevation_m": 0.0},
    )
    room_c = Entity(
        id="room-c",
        name="Room C (Split Level)",
        type=EntityType.ROOM,
        custom_properties={"boundary_element_ids": ["wall-partition"], "floor_elevation_m": 0.6},
    )

    world.entities[wall_shared.id] = wall_shared
    world.entities[room_a.id] = room_a
    world.entities[room_b.id] = room_b
    world.entities[room_c.id] = room_c

    # 1. Refuse merge between Room A and Room B because dividing wall has no opening
    res_solid = propose_room_merge(world, "room-a", "room-b")
    assert res_solid["status"] == "refused"
    assert "solid dividing partition with no opening" in res_solid["reason"]

    # 2. Refuse merge between Room A and Room C because of elevation step
    res_elev = propose_room_merge(world, "room-a", "room-c")
    assert res_elev["status"] == "refused"
    assert "incompatible floor elevation step" in res_elev["reason"]

    # 3. Add opening to wall-partition -> merge between A and B is accepted
    wall_shared.custom_properties["wall_slab"]["openings"] = [{"id": "door-01", "kind": "doorway"}]
    res_open = propose_room_merge(world, "room-a", "room-b")
    assert res_open["status"] == "accepted"
    assert res_open["has_opening"] is True
