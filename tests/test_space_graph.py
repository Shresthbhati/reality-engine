"""Tests for InteriorSpaceGraph queries and building topology relationships."""

from __future__ import annotations

import pytest

from perception.architecture.corridor import CorridorGraph
from perception.architecture.room_graph import BuildingGraph, RoomGraph, RoomOpening, Storey
from perception.architecture.space_graph import InteriorSpaceGraph


def _sample_building_setup():
    # 2 Rooms on Level 0, connected by an opening (doorway)
    op1 = RoomOpening(
        wall_element_id="wall-shared",
        kind="doorway",
        width_m=0.9,
        height_m=2.1,
        sill_height_m=0.0,
        connected_space_ids=("room-2",),
        is_exterior=False,
    )
    room1 = RoomGraph(
        room_id="room-1",
        boundary_element_ids=("fl-1", "cl-1", "wall-shared", "w1", "w2", "w3"),
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(4.0, 4.0, 2.5),
        dimensions_m={"width": 4.0, "length": 4.0, "height": 2.5},
        floor_area_m2=16.0,
        openings=(op1,),
        adjacent_room_ids=("room-2", "corr-1"),
        corridor_ids=("corr-1",),
        status="detected",
        boundary_completeness=1.0,
        level_id="lvl-0",
    )
    room2 = RoomGraph(
        room_id="room-2",
        boundary_element_ids=("fl-2", "cl-2", "wall-shared", "w4", "w5", "w6"),
        bounds_min=(4.0, 0.0, 0.0),
        bounds_max=(8.0, 4.0, 2.5),
        dimensions_m={"width": 4.0, "length": 4.0, "height": 2.5},
        floor_area_m2=16.0,
        openings=(),
        adjacent_room_ids=("room-1", "corr-1"),
        corridor_ids=("corr-1",),
        status="detected",
        boundary_completeness=0.95,
        level_id="lvl-0",
    )

    corr = CorridorGraph(
        corridor_id="corr-1",
        boundary_element_ids=("fl-c1", "cl-c1", "w7", "w8"),
        bounds_min=(0.0, -2.0, 0.0),
        bounds_max=(1.2, 2.2, 2.4),
        length_m=4.2,
        width_m=1.2,
        height_m=2.4,
        floor_area_m2=5.04,
        longitudinal_axis=(0.0, 1.0, 0.0),
        connected_room_ids=("room-1", "room-2"),
    )

    # 1 Room on Level 1
    room3 = RoomGraph(
        room_id="room-3",
        boundary_element_ids=("fl-3", "cl-3", "w9", "w10", "w11", "w12"),
        bounds_min=(0.0, 0.0, 2.8),
        bounds_max=(4.0, 4.0, 5.3),
        dimensions_m={"width": 4.0, "length": 4.0, "height": 2.5},
        floor_area_m2=16.0,
        openings=(),
        adjacent_room_ids=(),
        status="detected",
        boundary_completeness=1.0,
        level_id="lvl-1",
    )

    # Stair connecting Level 0 to Level 1
    class DummyStair:
        def __init__(self):
            self.stair_id = "stair-01"
            self.step_count = 14
            self.total_rise_m = 2.8
            self.total_run_m = 3.5

        def to_dict(self):
            return {
                "stair_id": self.stair_id,
                "step_count": self.step_count,
                "total_rise_m": self.total_rise_m,
                "total_run_m": self.total_run_m,
            }

    stair = DummyStair()

    storey0 = Storey(
        storey_id="lvl-0",
        floor_height_m=0.0,
        room_ids=["room-1", "room-2"],
        corridor_ids=["corr-1"],
        stair_ids=["stair-01"],
    )
    storey1 = Storey(
        storey_id="lvl-1",
        floor_height_m=2.8,
        room_ids=["room-3"],
        corridor_ids=[],
        stair_ids=["stair-01"],
    )

    bld = BuildingGraph(
        building_id="bld-test",
        storeys=[storey0, storey1],
        envelope_bounds_min=(0.0, -2.0, 0.0),
        envelope_bounds_max=(8.0, 4.0, 5.3),
        corridors=[corr],
        stairs=[stair],
    )

    return bld, [room1, room2, room3], [corr], [stair]


def test_interior_space_graph_connectivity():
    bld, rooms, corridors, stairs = _sample_building_setup()
    graph = InteriorSpaceGraph.from_building(bld, rooms, corridors, stairs)

    # Test corridor reachability: corr-1 reaches room-1 and room-2
    reachable = graph.rooms_reachable_from_corridor("corr-1")
    assert "room-1" in reachable
    assert "room-2" in reachable

    # Test room adjacency: room-1 connects to room-2 and corr-1
    connected = graph.rooms_connected_to("room-1")
    assert "room-2" in connected
    assert "corr-1" in connected

    # Test openings connecting room-1 and room-2
    openings = graph.openings_connecting("room-1", "room-2")
    assert len(openings) == 1
    assert openings[0]["kind"] == "doorway"
    assert openings[0]["width_m"] == 0.9

    # Test level of space
    lvl0 = graph.level_of_space("room-1")
    assert lvl0 is not None
    assert lvl0["level_id"] == "lvl-0"

    lvl1 = graph.level_of_space("room-3")
    assert lvl1 is not None
    assert lvl1["level_id"] == "lvl-1"

    # Test stair connecting levels
    stair_conns = graph.stairs_connecting_levels("lvl-0", "lvl-1")
    assert len(stair_conns) == 1
    assert stair_conns[0]["stair_id"] == "stair-01"


def test_interior_space_graph_serialization_roundtrip():
    bld, rooms, corridors, stairs = _sample_building_setup()
    graph = InteriorSpaceGraph.from_building(bld, rooms, corridors, stairs)

    d = graph.to_dict()
    assert d["summary"]["level_count"] == 2
    assert d["summary"]["room_count"] == 3
    assert d["summary"]["corridor_count"] == 1
    assert d["summary"]["stair_count"] == 1

    rehydrated = InteriorSpaceGraph.from_dict(d)
    assert rehydrated.rooms_reachable_from_corridor("corr-1") == ["room-1", "room-2"]
    assert "room-2" in rehydrated.rooms_connected_to("room-1")
    assert rehydrated.level_of_space("room-3")["level_id"] == "lvl-1"
