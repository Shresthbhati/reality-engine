"""Tests for T-3.4: Sockets, switches, light fixtures, radiators placed on wall planes with measured height.

Verifies:
1. Classification of sockets, switches, light fixtures, radiators, and thermostats.
2. Orthogonal wall-plane segment projection to identify correct host wall.
3. Accurate height above floor measurement satisfying the +-5 cm tolerance.
4. Bilateral relationship generation (fixture ATTACHED_TO wall, wall SUPPORTS fixture).
5. Evaluation against ground truth benchmark with precision/recall/F1 metrics.
"""

from __future__ import annotations

import pytest

from perception.architecture.fixtures import (
    FIXTURE_HEIGHT_TOLERANCE_M,
    classify_fixture,
    evaluate_fixture_accuracy,
    find_host_wall,
    place_all_fixtures,
    place_fixture_on_wall,
)
from world_ir.schema_v1 import Entity, EntityType, RelationshipKind
from world_ir.world_v1 import WorldIR


def _make_measured_room_world() -> WorldIR:
    """Build a synthetic measured room: 5.0m x 4.0m with walls, floor, and fixtures."""
    world = WorldIR(id="wld_measured_room", name="Measured Room Scene")

    # 1. Floor slab at z = 0.0
    floor = Entity(
        id="floor_main",
        name="Oak Hardwood Floor",
        type=EntityType.FLOOR,
        transform={"position": {"x": 2.5, "y": 2.0, "z": 0.0}},
        custom_properties={"elevation_m": 0.0, "bounds": [(0.0, 0.0, -0.2), (5.0, 4.0, 0.0)]},
    )
    world.entities[floor.id] = floor

    # 2. Four Bounding Walls (Height 2.8m, Thickness 0.2m)
    # North Wall along y = 4.0 (x: 0 to 5)
    wall_north = Entity(
        id="wall_north",
        name="North Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 2.5, "y": 4.0, "z": 1.4}},
        custom_properties={
            "bounding_box": [(0.0, 3.9, 0.0), (5.0, 4.1, 2.8)],
            "start_point_m": [0.0, 4.0],
            "end_point_m": [5.0, 4.0],
            "normal": [0.0, -1.0, 0.0],
        },
    )
    world.entities[wall_north.id] = wall_north

    # South Wall along y = 0.0 (x: 0 to 5)
    wall_south = Entity(
        id="wall_south",
        name="South Wall (Window Wall)",
        type=EntityType.WALL,
        transform={"position": {"x": 2.5, "y": 0.0, "z": 1.4}},
        custom_properties={
            "bounding_box": [(0.0, -0.1, 0.0), (5.0, 0.1, 2.8)],
            "start_point_m": [0.0, 0.0],
            "end_point_m": [5.0, 0.0],
            "normal": [0.0, 1.0, 0.0],
        },
    )
    world.entities[wall_south.id] = wall_south

    # East Wall along x = 5.0 (y: 0 to 4)
    wall_east = Entity(
        id="wall_east",
        name="East Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 5.0, "y": 2.0, "z": 1.4}},
        custom_properties={
            "bounding_box": [(4.9, 0.0, 0.0), (5.1, 4.0, 2.8)],
            "start_point_m": [5.0, 0.0],
            "end_point_m": [5.0, 4.0],
            "normal": [-1.0, 0.0, 0.0],
        },
    )
    world.entities[wall_east.id] = wall_east

    # West Wall along x = 0.0 (y: 0 to 4)
    wall_west = Entity(
        id="wall_west",
        name="West Wall (Door Wall)",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 2.0, "z": 1.4}},
        custom_properties={
            "bounding_box": [(-0.1, 0.0, 0.0), (0.1, 4.0, 2.8)],
            "start_point_m": [0.0, 0.0],
            "end_point_m": [0.0, 4.0],
            "normal": [1.0, 0.0, 0.0],
        },
    )
    world.entities[wall_west.id] = wall_west

    # 3. Fixtures
    # A. Duplex Power Socket on North Wall (measured height = 0.30m)
    socket_01 = Entity(
        id="socket_01",
        name="Duplex Power Socket",
        type=EntityType.STRUCTURE,
        semantic_labels=["socket"],
        transform={"position": {"x": 2.1, "y": 3.93, "z": 0.30}},
        custom_properties={"entity_class": "electrical.switchboard.module.socket"},
    )
    world.entities[socket_01.id] = socket_01

    # B. Light Switch near West Door (measured height = 1.10m)
    switch_01 = Entity(
        id="switch_01",
        name="2-Gang Light Switch",
        type=EntityType.STRUCTURE,
        semantic_labels=["switch"],
        transform={"position": {"x": 0.07, "y": 1.2, "z": 1.10}},
        custom_properties={"entity_class": "electrical.switchboard.module.switch"},
    )
    world.entities[switch_01.id] = switch_01

    # C. Wall Sconce on East Wall (measured height = 1.85m)
    sconce_01 = Entity(
        id="sconce_01",
        name="Brass Wall Sconce",
        type=EntityType.STRUCTURE,
        semantic_labels=["wall_light", "sconce"],
        transform={"position": {"x": 4.93, "y": 2.0, "z": 1.85}},
        custom_properties={"entity_class": "electrical.light_fixture.sconce"},
    )
    world.entities[sconce_01.id] = sconce_01

    # D. Heating Radiator on South Wall under window (measured height = 0.20m)
    radiator_01 = Entity(
        id="radiator_01",
        name="Steel Panel Radiator",
        type=EntityType.STRUCTURE,
        semantic_labels=["radiator", "heater"],
        transform={"position": {"x": 2.5, "y": 0.09, "z": 0.20}},
        custom_properties={"entity_class": "hvac.radiator"},
    )
    world.entities[radiator_01.id] = radiator_01

    return world


def test_classify_architectural_fixtures():
    """Verify classification of sockets, switches, lights, and radiators."""
    e_sock = Entity(id="s1", name="Wall Outlet", type=EntityType.STRUCTURE, semantic_labels=["socket"])
    assert classify_fixture(e_sock) == "socket"

    e_sw = Entity(id="sw1", name="Rocker Switch", type=EntityType.STRUCTURE, semantic_labels=["switch"])
    assert classify_fixture(e_sw) == "switch"

    e_sc = Entity(id="sc1", name="Vanity Sconce", type=EntityType.STRUCTURE, semantic_labels=["sconce"])
    assert classify_fixture(e_sc) == "light_fixture"

    e_rad = Entity(id="rad1", name="Panel Convector", type=EntityType.STRUCTURE, semantic_labels=["radiator"])
    assert classify_fixture(e_rad) == "radiator"

    e_chair = Entity(id="c1", name="Dining Chair", type=EntityType.STRUCTURE, semantic_labels=["chair"])
    assert classify_fixture(e_chair) is None


def test_socket_placed_on_north_wall_with_measured_height():
    """Duplex power socket placed within +-5cm tolerance on North wall."""
    world = _make_measured_room_world()
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
    socket = world.entities["socket_01"]

    placement = place_fixture_on_wall(socket, walls=walls, floor_z=0.0)
    assert placement is not None
    assert placement.kind == "socket"
    assert placement.host_wall_id == "wall_north"
    assert placement.is_wall_mounted is True
    # Measured height exactly 0.30m above floor
    assert placement.height_above_floor_m == pytest.approx(0.30, abs=0.01)
    assert placement.distance_to_wall_m < 0.10

    # In-place custom_properties and relationships
    assert socket.custom_properties["host_wall_id"] == "wall_north"
    assert socket.custom_properties["height_above_floor_m"] == 0.30
    assert socket.custom_properties["elevation_measured"] is True

    # Check bilateral relationships
    attached_rel = next((r for r in socket.relationships if r.target_id == "wall_north"), None)
    assert attached_rel is not None
    assert attached_rel.kind == RelationshipKind.ATTACHED_TO

    wall_rel = next((r for r in world.entities["wall_north"].relationships if r.target_id == "socket_01"), None)
    assert wall_rel is not None
    assert wall_rel.kind == RelationshipKind.SUPPORTS


def test_light_switch_placed_on_west_wall():
    """Light switch mounted at 1.10m height near West door."""
    world = _make_measured_room_world()
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
    switch = world.entities["switch_01"]

    placement = place_fixture_on_wall(switch, walls=walls, floor_z=0.0)
    assert placement is not None
    assert placement.kind == "switch"
    assert placement.host_wall_id == "wall_west"
    assert placement.height_above_floor_m == pytest.approx(1.10, abs=0.01)


def test_radiator_placed_on_south_wall():
    """Radiator mounted on South exterior wall at 0.20m height."""
    world = _make_measured_room_world()
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
    radiator = world.entities["radiator_01"]

    placement = place_fixture_on_wall(radiator, walls=walls, floor_z=0.0)
    assert placement is not None
    assert placement.kind == "radiator"
    assert placement.host_wall_id == "wall_south"
    assert placement.height_above_floor_m == pytest.approx(0.20, abs=0.01)


def test_place_all_fixtures_in_world():
    """End-to-end placement of all fixtures in the measured room."""
    world = _make_measured_room_world()
    placements = place_all_fixtures(world)

    assert len(placements) == 4
    placement_kinds = {p.kind for p in placements}
    assert placement_kinds == {"socket", "switch", "light_fixture", "radiator"}

    # All fixtures have valid host walls
    for p in placements:
        assert p.host_wall_id is not None
        assert p.is_wall_mounted is True
        assert p.distance_to_wall_m <= 0.15


def test_evaluate_fixture_accuracy_against_benchmark():
    """Verify eval accuracy metrics against 8.6 ground truth set (meets 5cm tolerance)."""
    world = _make_measured_room_world()
    placements = place_all_fixtures(world)

    ground_truth = [
        {"entity_id": "gt_sock", "kind": "socket", "host_wall_id": "wall_north", "position": (2.1, 4.0, 0.30), "height_above_floor_m": 0.30},
        {"entity_id": "gt_sw", "kind": "switch", "host_wall_id": "wall_west", "position": (0.0, 1.2, 1.10), "height_above_floor_m": 1.10},
        {"entity_id": "gt_sc", "kind": "light_fixture", "host_wall_id": "wall_east", "position": (5.0, 2.0, 1.85), "height_above_floor_m": 1.85},
        {"entity_id": "gt_rad", "kind": "radiator", "host_wall_id": "wall_south", "position": (2.5, 0.0, 0.20), "height_above_floor_m": 0.20},
    ]

    report = evaluate_fixture_accuracy(placements, ground_truth=ground_truth, tolerance_m=FIXTURE_HEIGHT_TOLERANCE_M)

    assert report["recall"] == 1.0
    assert report["precision"] == 1.0
    assert report["f1"] == 1.0
    assert report["wall_accuracy"] == 1.0
    assert report["within_tolerance_ratio"] == 1.0
    assert report["meets_5cm_tolerance"] is True
    assert report["height_rmse_m"] <= 0.02


def test_segment_projection_avoids_wrong_wall_near_corners():
    """Corner disambiguation: An entity near a corner selects the correct host wall by planar segment."""
    # A switch located at x=0.08, y=3.85 (close to North-West corner)
    # Distance to North wall line (y=4.0) is 0.15m
    # Distance to West wall line (x=0.0) is 0.08m
    # Euclidean center of North wall is at (2.5, 4.0) -> dist = 2.42m
    # Euclidean center of West wall is at (0.0, 2.0) -> dist = 1.85m
    # Segment projection correctly associates with West wall (dist 0.08m < 0.15m)
    switch_corner = Entity(
        id="switch_corner",
        name="Corner Switch",
        type=EntityType.STRUCTURE,
        semantic_labels=["switch"],
        transform={"position": {"x": 0.08, "y": 3.85, "z": 1.10}},
    )

    world = _make_measured_room_world()
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL]

    best_wall, dist = find_host_wall((0.08, 3.85, 1.10), walls)
    assert best_wall is not None
    assert best_wall.id == "wall_west"
    assert dist == pytest.approx(0.08, abs=0.01)

    pl = place_fixture_on_wall(switch_corner, walls=walls, floor_z=0.0)
    assert pl is not None
    assert pl.host_wall_id == "wall_west"
