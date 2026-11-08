"""Tests for Real-Time Multi-Room Semantic Navigation & Walkthrough Tours (Track B)."""

from __future__ import annotations

import math
import uuid
import pytest
from fastapi.testclient import TestClient

from apps.api.db import get_sessionmaker, init_db
from apps.api.main import app
from apps.api.models import World
from engine.pipeline.walkthrough_tour import (
    ArchitecturalTour,
    RoomWaypoint,
    SpatialBookmark,
    build_world_navigation_data,
    detect_room_at_position,
    extract_rooms_from_worldir,
    generate_architectural_tour,
    interpolate_tour_pose,
)
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR
from apps.api.worldstore_service import get_store


def _build_test_apartment_world() -> WorldIR:
    """Build a synthetic 3-room apartment WorldIR with connecting doors."""
    world = WorldIR(id=f"wld_apt_{uuid.uuid4().hex[:8]}")

    # Room 1: Entrance Foyer (-4 to 0 on X, -2 to 2 on Z)
    foyer = Entity(
        id="room_foyer",
        name="Entrance Foyer",
        type=EntityType.ROOM,
        transform={"position": {"x": -2.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width_m": 4.0, "depth_m": 4.0, "height_m": 2.7, "room_type": "foyer"},
    )
    world.entities[foyer.id] = foyer

    # Room 2: Living Room (0 to 6 on X, -3 to 3 on Z)
    living = Entity(
        id="room_living",
        name="Living Room",
        type=EntityType.ROOM,
        transform={"position": {"x": 3.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width_m": 6.0, "depth_m": 6.0, "height_m": 2.8, "room_type": "living_room"},
    )
    world.entities[living.id] = living

    # Room 3: Master Bedroom (6 to 11 on X, -2 to 3 on Z)
    bedroom = Entity(
        id="room_bedroom",
        name="Master Bedroom",
        type=EntityType.ROOM,
        transform={"position": {"x": 8.5, "y": 0.0, "z": 0.5}},
        custom_properties={"width_m": 5.0, "depth_m": 5.0, "height_m": 2.7, "room_type": "bedroom"},
    )
    world.entities[bedroom.id] = bedroom

    # Door 1: Between Foyer and Living Room at X=0.0, Z=0.0
    door1 = Entity(
        id="door_foyer_living",
        name="Foyer to Living Door",
        type=EntityType.DOOR,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width_m": 0.9, "connected_spaces": ["room_foyer", "room_living"]},
    )
    world.entities[door1.id] = door1

    # Door 2: Between Living Room and Bedroom at X=6.0, Z=0.5
    door2 = Entity(
        id="door_living_bedroom",
        name="Living to Bedroom Door",
        type=EntityType.DOOR,
        transform={"position": {"x": 6.0, "y": 0.0, "z": 0.5}},
        custom_properties={"width_m": 0.9, "connected_spaces": ["room_living", "room_bedroom"]},
    )
    world.entities[door2.id] = door2

    return world


def test_extract_rooms_from_worldir():
    """Verify room extraction identifies all rooms, dimensions, and doors."""
    world = _build_test_apartment_world()
    rooms, doors = extract_rooms_from_worldir(world)

    assert len(rooms) == 3
    room_names = [r.name for r in rooms]
    assert "Entrance Foyer" in room_names
    assert "Living Room" in room_names
    assert "Master Bedroom" in room_names

    assert len(doors) == 2
    # Verify foyer has door linked
    foyer = next(r for r in rooms if r.room_id == "room_foyer")
    assert foyer.area_sqm == 16.0
    assert foyer.ceiling_height_m == 2.7
    assert len(foyer.openings) >= 1


def test_detect_room_at_position():
    """Verify spatial containment maps coordinates to the correct room."""
    world = _build_test_apartment_world()
    rooms, _ = extract_rooms_from_worldir(world)

    # Point in Foyer: X=-2.0, Y=1.5, Z=0.0
    r_foyer = detect_room_at_position([-2.0, 1.5, 0.0], rooms)
    assert r_foyer is not None
    assert r_foyer.room_id == "room_foyer"

    # Point in Living Room: X=3.0, Y=1.5, Z=1.0
    r_living = detect_room_at_position([3.0, 1.5, 1.0], rooms)
    assert r_living is not None
    assert r_living.room_id == "room_living"

    # Point in Bedroom: X=8.5, Y=1.5, Z=0.5
    r_bed = detect_room_at_position([8.5, 1.5, 0.5], rooms)
    assert r_bed is not None
    assert r_bed.room_id == "room_bedroom"


def test_generate_architectural_tour():
    """Verify tour generation builds an intuitive sequential progression with keyframes."""
    world = _build_test_apartment_world()
    tour, bookmarks = generate_architectural_tour(world, project_name="Downtown Apartment")

    assert tour.title == "Downtown Apartment Walkthrough"
    assert tour.total_duration_seconds > 10.0
    assert len(tour.keyframes) == 6  # 2 keyframes per room (entry + pan) * 3 rooms
    assert len(tour.waypoint_path) > len(tour.keyframes)

    # Architectural progression: Foyer first, then Living Room, then Bedroom
    assert tour.rooms_visited[0] == "Entrance Foyer"
    assert tour.rooms_visited[1] == "Living Room"
    assert tour.rooms_visited[2] == "Master Bedroom"

    # Bookmarks should be created for each room center
    assert len(bookmarks) == 3
    bmk_names = [b.name for b in bookmarks]
    assert any("Foyer" in n for n in bmk_names)
    assert any("Living" in n for n in bmk_names)
    assert any("Bedroom" in n for n in bmk_names)


def test_interpolate_tour_pose():
    """Verify camera position and angle interpolation across time."""
    world = _build_test_apartment_world()
    tour, _ = generate_architectural_tour(world)

    # At start (t=0): should match first keyframe
    p0 = interpolate_tour_pose(tour, 0.0)
    assert p0["progress_pct"] == 0.0
    assert p0["current_room"] == tour.keyframes[0].room_name

    # At mid-tour: should be transitioning smoothly
    half_time = tour.total_duration_seconds / 2.0
    p_mid = interpolate_tour_pose(tour, half_time)
    assert 40.0 <= p_mid["progress_pct"] <= 60.0
    assert len(p_mid["position"]) == 3
    assert -math.pi <= p_mid["yaw_rad"] <= math.pi

    # At end (t >= duration)
    p_end = interpolate_tour_pose(tour, tour.total_duration_seconds + 5.0)
    assert p_end["progress_pct"] == 100.0
    assert p_end["current_room"] == tour.keyframes[-1].room_name


def test_empty_world_fallback():
    """Tour and navigation engine gracefully handles empty worlds."""
    empty_world = WorldIR(id=f"wld_empty_{uuid.uuid4().hex[:8]}")
    nav = build_world_navigation_data(empty_world)

    assert len(nav.rooms) >= 1
    assert nav.recommended_tour.total_duration_seconds > 0.0
    assert len(nav.suggested_bookmarks) >= 1


@pytest.mark.asyncio
async def test_navigation_api_endpoints():
    """Test REST endpoints: /api/worlds/{id}/navigation, /bookmarks, /navigation/path."""
    client = TestClient(app)

    # 1. Setup World in DB and WorldStore
    store = get_store()
    world_ir = _build_test_apartment_world()
    suffix = uuid.uuid4().hex[:8]
    world_ir.id = f"wld_nav_test_{suffix}"
    sv = store.save_version(world_ir, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        w = World(
            id=world_ir.id,
            name="Modern Flat 3B",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(w)
        await session.commit()

    # 2. Test GET /api/worlds/{id}/navigation
    res_nav = client.get(f"/api/worlds/{world_ir.id}/navigation")
    assert res_nav.status_code == 200
    nav_data = res_nav.json()
    assert nav_data["world_id"] == world_ir.id
    assert len(nav_data["rooms"]) == 3
    assert len(nav_data["recommended_tour"]["keyframes"]) == 6
    assert len(nav_data["suggested_bookmarks"]) == 3

    # 3. Test GET /api/worlds/{id}/bookmarks
    res_bmk = client.get(f"/api/worlds/{world_ir.id}/bookmarks")
    assert res_bmk.status_code == 200
    bmks = res_bmk.json()
    assert len(bmks) >= 3

    # 4. Test POST /api/worlds/{id}/bookmarks
    res_post_bmk = client.post(
        f"/api/worlds/{world_ir.id}/bookmarks",
        json={
            "name": "Balcony Sunset Angle",
            "room_name": "Living Room",
            "position": [4.5, 1.65, 2.5],
            "yaw_rad": 1.25,
            "pitch_rad": -0.15,
            "fov_deg": 65.0,
            "tags": ["balcony", "sunset", "custom"],
        },
    )
    assert res_post_bmk.status_code == 200
    created = res_post_bmk.json()
    assert created["name"] == "Balcony Sunset Angle"
    assert created["position"] == [4.5, 1.65, 2.5]

    # Verify bookmark is persisted on subsequent list
    res_bmk2 = client.get(f"/api/worlds/{world_ir.id}/bookmarks")
    assert res_bmk2.status_code == 200
    names = [b["name"] for b in res_bmk2.json()]
    assert "Balcony Sunset Angle" in names

    # 5. Test POST /api/worlds/{id}/navigation/path
    res_path = client.post(
        f"/api/worlds/{world_ir.id}/navigation/path",
        json={
            "start_pos": [-2.0, 1.65, 0.0],  # in Foyer
            "target_room_id": "room_bedroom",  # in Bedroom
        },
    )
    assert res_path.status_code == 200
    path_data = res_path.json()
    assert path_data["start_room"] == "Entrance Foyer"
    assert path_data["target_room"] == "Master Bedroom"
    assert path_data["distance_m"] > 0
    # Waypoints should route through connecting doors
    assert len(path_data["waypoints"]) >= 3
