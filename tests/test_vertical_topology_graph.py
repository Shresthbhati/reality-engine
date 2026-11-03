"""Tests for Multi-Storey Staircase and Vertical Topology Graph (Epic E04-3)."""

from __future__ import annotations

import math
import pytest
import numpy as np

from engine.compiler.topology_graph import (
    StairTread,
    StairRiser,
    StairLanding,
    StaircaseAnatomy,
    StoreyLevel,
    VerticalPath,
    VerticalTopologyGraph,
    compile_vertical_topology,
)
from world_ir.coordinates import Frame, Transform
from world_ir.schema_v1 import Entity, EntityType, Vector3
from world_ir.world_v1 import WorldIR


def test_compile_vertical_topology_storeys_and_stairs():
    world = WorldIR(id="w-multi-storey")

    # Add 2 floors
    ground_floor = Entity(
        id="floor-0",
        name="Ground Floor Slab",
        type=EntityType.FLOOR,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}}
    )
    first_floor = Entity(
        id="floor-1",
        name="First Floor Slab",
        type=EntityType.FLOOR,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 3.0}}
    )
    # Add staircase linking ground to first floor
    stairs = Entity(
        id="stair-flight-1",
        name="Main Staircase Flight",
        type=EntityType.STAIRS,
        transform={"position": {"x": 2.0, "y": 3.0, "z": 0.0}},
        custom_properties={
            "total_rise_m": 3.0,
            "total_run_m": 3.6,
            "flight_width_m": 1.1,
            "num_risers": 16,
        }
    )
    # Add rooms
    room_ground = Entity(id="room-101", name="Living Room", type=EntityType.ROOM, transform={"position": {"x": 1.0, "y": 1.0, "z": 0.0}})
    room_upper = Entity(id="room-201", name="Master Bedroom", type=EntityType.ROOM, transform={"position": {"x": 1.0, "y": 1.0, "z": 3.0}})

    world.entities[ground_floor.id] = ground_floor
    world.entities[first_floor.id] = first_floor
    world.entities[stairs.id] = stairs
    world.entities[room_ground.id] = room_ground
    world.entities[room_upper.id] = room_upper

    graph = compile_vertical_topology(world)

    # Verify storeys
    assert len(graph.storeys) == 2
    assert "storey-ground" in graph.storeys
    assert "storey-1" in graph.storeys
    assert graph.storeys["storey-ground"].floor_elevation_m == 0.0
    assert graph.storeys["storey-1"].floor_elevation_m == 3.0

    # Verify staircase anatomy
    assert "stair-flight-1" in graph.staircases
    stair = graph.staircases["stair-flight-1"]
    assert stair.from_storey_id == "storey-ground"
    assert stair.to_storey_id == "storey-1"
    assert stair.total_rise_m == 3.0
    assert stair.total_run_m == 3.6
    assert stair.num_risers == 16
    assert stair.num_treads == 15
    assert math.isclose(stair.riser_height_m, 3.0 / 16, abs_tol=1e-4)
    assert math.isclose(stair.tread_depth_m, 3.6 / 15, abs_tol=1e-4)
    assert len(stair.treads) == 15
    assert len(stair.risers) == 16
    assert len(stair.landings) >= 1


def test_continuous_coordinate_transformation_across_storeys():
    """Verify continuous 4x4 coordinate transformation connecting ground floor to upper levels across stairs."""
    world = WorldIR(id="w-coords")
    f0 = Entity(id="f0", name="Floor 0", type=EntityType.FLOOR, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}})
    f1 = Entity(id="f1", name="Floor 1", type=EntityType.FLOOR, transform={"position": {"x": 0.0, "y": 0.0, "z": 2.80}})
    st = Entity(
        id="st1",
        name="Stairs",
        type=EntityType.STAIRS,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"total_rise_m": 2.80, "total_run_m": 3.50}
    )
    world.entities[f0.id] = f0
    world.entities[f1.id] = f1
    world.entities[st.id] = st

    graph = compile_vertical_topology(world)

    # Transform from ground floor to level 1
    t_g_to_1 = graph.get_transform_between_storeys("storey-ground", "storey-1")
    assert t_g_to_1 is not None

    # Test point on ground floor (e.g., origin)
    pt_ground = (1.5, 2.0, 0.0)
    pt_level1 = graph.transform_point_between_storeys(pt_ground, "storey-ground", "storey-1")

    # Z must increase by 2.80m, horizontal coordinates shift across stair flight
    assert math.isclose(pt_level1[2], 2.80, abs_tol=1e-3)
    assert math.isclose(pt_level1[0], 1.5 + 3.50, abs_tol=1e-3)
    assert math.isclose(pt_level1[1], 2.0, abs_tol=1e-3)

    # Inverse transformation must return exact ground point
    pt_back = graph.transform_point_between_storeys(pt_level1, "storey-1", "storey-ground")
    assert math.isclose(pt_back[0], pt_ground[0], abs_tol=1e-5)
    assert math.isclose(pt_back[1], pt_ground[1], abs_tol=1e-5)
    assert math.isclose(pt_back[2], pt_ground[2], abs_tol=1e-5)


def test_multi_storey_path_traversal():
    """Verify 3D path computation between rooms across multiple levels."""
    world = WorldIR(id="w-path")
    f0 = Entity(id="f0", name="F0", type=EntityType.FLOOR, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}})
    f1 = Entity(id="f1", name="F1", type=EntityType.FLOOR, transform={"position": {"x": 0.0, "y": 0.0, "z": 3.0}})
    st = Entity(
        id="st1",
        name="Stairs",
        type=EntityType.STAIRS,
        transform={"position": {"x": 10.0, "y": 5.0, "z": 0.0}},
        custom_properties={"total_rise_m": 3.0, "total_run_m": 4.0, "num_risers": 15}
    )
    r1 = Entity(id="room-entry", name="Entry", type=EntityType.ROOM, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}})
    r2 = Entity(id="room-bedroom", name="Bedroom", type=EntityType.ROOM, transform={"position": {"x": 15.0, "y": 5.0, "z": 3.0}})

    world.entities[f0.id] = f0
    world.entities[f1.id] = f1
    world.entities[st.id] = st
    world.entities[r1.id] = r1
    world.entities[r2.id] = r2

    graph = compile_vertical_topology(world)

    path = graph.find_shortest_vertical_path("room-entry", "room-bedroom")
    assert path is not None
    assert path.from_storey_id == "storey-ground"
    assert path.to_storey_id == "storey-1"
    assert "st1" in path.via_stair_ids
    assert path.total_vertical_elevation_m == 3.0
    assert path.total_steps == 15
    assert len(path.path_points) >= 15


def test_stair_architectural_proportions():
    """Verify architectural ergonomics (Blondel's rule: 2R + T ~ 60..65 cm)."""
    world = WorldIR(id="w-blondel")
    f0 = Entity(id="f0", type=EntityType.FLOOR, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}})
    f1 = Entity(id="f1", type=EntityType.FLOOR, transform={"position": {"x": 0.0, "y": 0.0, "z": 2.80}})
    st = Entity(
        id="st_ergonomic",
        type=EntityType.STAIRS,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"total_rise_m": 2.80, "total_run_m": 4.20, "num_risers": 16}
    )
    world.entities[f0.id] = f0
    world.entities[f1.id] = f1
    world.entities[st.id] = st

    graph = compile_vertical_topology(world)
    stair = graph.staircases["st_ergonomic"]

    # 2 * Riser + Tread
    stride = 2 * stair.riser_height_m + stair.tread_depth_m
    assert 0.58 <= stride <= 0.68  # Architectural ergonomics standards
