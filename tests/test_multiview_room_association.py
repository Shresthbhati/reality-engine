"""Tests for Multi-View Room Association Engine (Programme 1: Multi-View Intelligence)."""

from __future__ import annotations

from perception.multiview.room_associator import (
    CameraPose3D,
    MultiViewRoomAssociator,
    RoomBoundary,
    point_in_polygon_2d,
)


def test_point_in_polygon_convex_and_l_shaped():
    # 1. Simple rectangle: [0, 0] to [5, 4]
    rect = [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)]
    assert point_in_polygon_2d((2.5, 2.0), rect) is True
    assert point_in_polygon_2d((6.0, 2.0), rect) is False
    assert point_in_polygon_2d((-0.5, 2.0), rect) is False

    # 2. L-shaped non-convex polygon (like Room 3 corridor office)
    # Bottom leg: [0,0] to [6,0] to [6,2] to [3,2] to [3,5] to [0,5]
    l_shape = [
        (0.0, 0.0),
        (6.0, 0.0),
        (6.0, 2.0),
        (3.0, 2.0),
        (3.0, 5.0),
        (0.0, 5.0),
    ]
    # Point inside horizontal leg
    assert point_in_polygon_2d((4.5, 1.0), l_shape) is True
    # Point inside vertical leg
    assert point_in_polygon_2d((1.5, 3.5), l_shape) is True
    # Point in the cut-out corner (outside!)
    assert point_in_polygon_2d((4.5, 3.5), l_shape) is False


def test_multiview_room_association_enclosure_and_visibility():
    associator = MultiViewRoomAssociator()

    room1 = RoomBoundary(
        room_id="room-studio",
        name="Living Studio",
        polygon_xy=[(0.0, 0.0), (5.2, 0.0), (5.2, 4.1), (0.0, 4.1)],
        floor_z=0.0,
        ceiling_z=2.65,
        boundary_walls=[
            {"id": "wall-south", "start": [0.0, 0.0], "end": [5.2, 0.0]},
            {"id": "wall-east", "start": [5.2, 0.0], "end": [5.2, 4.1]},
            {"id": "wall-north", "start": [5.2, 4.1], "end": [0.0, 4.1]},
            {"id": "wall-west", "start": [0.0, 4.1], "end": [0.0, 0.0]},
        ],
    )

    room2 = RoomBoundary(
        room_id="room-hall",
        name="Hallway",
        polygon_xy=[(0.0, -2.5), (3.0, -2.5), (3.0, 0.0), (0.0, 0.0)],
        floor_z=0.0,
        ceiling_z=2.65,
        boundary_walls=[
            {"id": "hall-south", "start": [0.0, -2.5], "end": [3.0, -2.5]},
            {"id": "hall-east", "start": [3.0, -2.5], "end": [3.0, 0.0]},
            {"id": "hall-north", "start": [3.0, 0.0], "end": [0.0, 0.0]},
            {"id": "hall-west", "start": [0.0, 0.0], "end": [0.0, -2.5]},
        ],
    )

    door_connecting = {
        "id": "door-studio-hall",
        "center": [1.2, 0.0, 1.05],
        "connects_rooms": ["room-studio", "room-hall"],
    }

    # Camera 1: Inside studio at (2.0, 2.0, 1.5), looking North towards wall-north
    cam_inside = CameraPose3D(
        evidence_id="photo_01",
        position=(2.0, 2.0, 1.5),
        look_dir=(0.0, 1.0, 0.0),
        hfov_deg=60.0,
    )

    # Camera 2: Inside studio at (4.0, 2.0, 1.5), also looking North towards wall-north
    cam_inside_parallel = CameraPose3D(
        evidence_id="photo_02",
        position=(4.0, 2.0, 1.5),
        look_dir=(0.0, 1.0, 0.0),
        hfov_deg=60.0,
    )

    # Camera 3: In hallway looking through door at Studio
    cam_in_hall = CameraPose3D(
        evidence_id="photo_03",
        position=(1.2, -1.5, 1.5),
        look_dir=(0.0, 1.0, 0.0),  # Looking North straight through the door
        hfov_deg=60.0,
    )

    assoc1 = associator.associate_camera(cam_inside, [room1, room2], [door_connecting])
    assert assoc1.inside_room_id == "room-studio"
    assert "wall-north" in assoc1.visible_wall_ids
    assert "room-studio" in assoc1.visible_room_ids

    assoc2 = associator.associate_camera(cam_inside_parallel, [room1, room2], [door_connecting])
    assert assoc2.inside_room_id == "room-studio"
    assert "wall-north" in assoc2.visible_wall_ids

    assoc3 = associator.associate_camera(cam_in_hall, [room1, room2], [door_connecting])
    assert assoc3.inside_room_id == "room-hall"
    # Looking through the portal, room-studio is visible too
    assert "room-studio" in assoc3.visible_room_ids
    assert "room-hall" in assoc3.visible_room_ids

    # Co-visibility between photos 1 and 2
    covis = associator.build_cross_view_covisibility([assoc1, assoc2, assoc3])
    pair_1_2 = ("photo_01", "photo_02")
    assert pair_1_2 in covis
    # Both in same room (0.40) + both view wall-north (0.60) = 1.0
    assert covis[pair_1_2] >= 0.80

    # Photos 1 and 3 reside in different rooms
    pair_1_3 = ("photo_01", "photo_03")
    assert pair_1_3 in covis
    assert covis[pair_1_3] < covis[pair_1_2]


def test_compiler_multiview_room_association_integration():
    """Verify that compile_reconstruction_to_world runs the MultiViewRoomAssociator pass."""
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from provenance import Uncertainty
    from reconstruction.backend.interface import (
        ReconstructedCameraPose,
        ReconstructedPoint,
        ReconstructionResult,
    )
    from world_ir.schema_v1 import EntityType

    def _pt(x, y, z, c=[0]):
        c[0] += 1
        return ReconstructedPoint(
            position=(x, y, z),
            track_id=f"pt-{c[0]:05d}",
            source_evidence_ids=["ev-1"],
            uncertainty=Uncertainty(1.0, "synthetic"),
        )

    pts = []
    # Floor at y=0.0 (3x3m)
    for xi in range(31):
        for zi in range(31):
            pts.append(_pt(xi * 0.1, 0.0, zi * 0.1))
    # Ceiling at y=2.5
    for xi in range(31):
        for zi in range(31):
            pts.append(_pt(xi * 0.1, 2.5, zi * 0.1))
    # 4 walls bounding the room
    for yi in range(13):
        for zi in range(31):
            pts.append(_pt(0.0, yi * 0.2, zi * 0.1))
            pts.append(_pt(3.0, yi * 0.2, zi * 0.1))
        for xi in range(31):
            pts.append(_pt(xi * 0.1, yi * 0.2, 0.0))
            pts.append(_pt(xi * 0.1, yi * 0.2, 3.0))

    cams = [
        ReconstructedCameraPose("ev-0", (1.5, 1.8, 1.5), (1.0, 0.0, 0.0, 0.0)),
        ReconstructedCameraPose("ev-1", (1.0, 1.8, 2.0), (1.0, 0.0, 0.0, 0.0)),
        ReconstructedCameraPose("ev-2", (2.0, 1.8, 1.0), (1.0, 0.0, 0.0, 0.0)),
    ]
    res = ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")
    world, diag = compile_reconstruction_to_world(res, CompileOptions(seed=42))

    assert diag.rooms_detected == 1
    # Check multiview_rooms in world metadata
    assert "multiview_rooms" in world.metadata
    mv = world.metadata["multiview_rooms"]
    assert mv["rooms_count"] == 1
    assert mv["cameras_count"] == 3
    assert len(mv["associations"]) == 3

    # Every camera was inside the 3x3 room
    room_ent = next(e for e in world.entities.values() if e.type == EntityType.ROOM)
    for assoc in mv["associations"]:
        assert assoc["inside_room_id"] == room_ent.id

    # The room entity recorded its camera stations
    props = getattr(room_ent, "custom_properties", {}) or getattr(room_ent, "metadata", {})
    assert sorted(props.get("camera_stations", [])) == ["ev-0", "ev-1", "ev-2"]
    # Co-visibility exists between the internal stations
    assert len(mv["covisibility"]) > 0
