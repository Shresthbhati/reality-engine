"""Vertical + horizontal interior connectivity (INTERIOR RECONSTRUCTION
MISSION, requirements 2/3/6/7).

The topology layer (PR #106) links stairs to storeys they span and
associates windows to rooms. This suite pins the remaining connectivity
facts:

  connected_rooms_through_openings(rooms, openings, world)
      - two rooms are CONNECTED_TO iff an opening's host wall bounds
        both rooms (the measured shared-wall fact) and the opening's
        kind is a passage (door or generic -- a WINDOW connects light,
        not people, and must NOT create a passage edge);

  connected_corridors / corridor room links
      - a corridor inferred by corridors.detect_corridor records
        CONNECTED_TO the rooms whose doors open onto it;

  storey_connectivity(building, stair_links)
      - Level n CONNECTED_TO Level n+1 through the stair that spans
        both floors (topology, not geometry);
"""

from __future__ import annotations

import pytest

from perception.architecture.room_graph import RoomGraph, RoomOpening

UP = (0.0, 0.0, 1.0)


def _room(rid, x0, y0, walls=(), openings=()):
    return RoomGraph(
        room_id=rid,
        boundary_element_ids=tuple(walls),
        bounds_min=(x0, y0, 0.0),
        bounds_max=(x0 + 3.0, y0 + 2.5, 2.4),
        dimensions_m={"x": 3.0, "y": 2.5, "z": 2.4},
        floor_area_m2=7.5,
        openings=tuple(openings),
        adjacent_room_ids=(),
    )


class TestRoomConnectivityThroughOpenings:
    def test_shared_wall_with_door_connects_rooms(self):
        from perception.architecture.topology import (
            connected_rooms_through_openings,
        )

        rooms = [
            _room("room-a", 0.0, 0.0, walls=("wall-x0", "wall-x3", "wall-y0", "wall-y2p5")),
            _room("room-b", 3.0, 0.0, walls=("wall-x3", "wall-x6", "wall-y0", "wall-y2p5")),
        ]
        # The shared wall x=3 carries a door (opening).
        openings = [(
            "wall-x3",
            RoomOpening(wall_element_id="wall-x3", kind="doorway",
                        width_m=0.9, height_m=2.0),
        )]
        links = connected_rooms_through_openings(rooms, openings)
        assert ("room-a", "room-b") in {tuple(sorted(l)) for l in links} or \
            any(set(l) == {"room-a", "room-b"} for l in links)

    def test_shared_wall_with_only_window_does_not_connect(self):
        from perception.architecture.topology import (
            connected_rooms_through_openings,
        )

        rooms = [
            _room("room-a", 0.0, 0.0, walls=("wall-x3",)),
            _room("room-b", 3.0, 0.0, walls=("wall-x3",)),
        ]
        openings = [(
            "wall-x3",
            RoomOpening(wall_element_id="wall-x3", kind="window",
                        width_m=1.2, height_m=1.0),
        )]
        links = connected_rooms_through_openings(rooms, openings)
        assert links == []

    def test_opening_on_unshared_wall_connects_nothing(self):
        from perception.architecture.topology import (
            connected_rooms_through_openings,
        )

        rooms = [
            _room("room-a", 0.0, 0.0, walls=("wall-x0",)),
            _room("room-b", 3.0, 0.0, walls=("wall-x6",)),
        ]
        openings = [(
            "wall-x0",
            RoomOpening(wall_element_id="wall-x0", kind="doorway",
                        width_m=0.9, height_m=2.0),
        )]
        assert connected_rooms_through_openings(rooms, openings) == []


class TestCorridorRoomLinks:
    def test_corridor_links_rooms_with_doors_on_it(self):
        from perception.architecture.topology import (
            link_corridor_to_rooms,
        )

        # Corridor is a room-like cell too; rooms open onto it through
        # doors in its long walls.
        corridor = _room(
            "room-corridor", 0.0, 10.0,
            walls=("wall-c-a", "wall-c-b"),
            openings=(
                RoomOpening(wall_element_id="wall-c-a", kind="doorway",
                            width_m=0.9, height_m=2.0),
            ),
        )
        # room-r1's door is IN wall-c-a (the corridor's long wall).
        r1 = _room(
            "room-r1", 0.0, 0.0, walls=("wall-c-a",),
            openings=(
                RoomOpening(wall_element_id="wall-c-a", kind="doorway",
                            width_m=0.9, height_m=2.0),
            ),
        )
        # room-r2 has a window only onto the corridor: no passage link.
        r2 = _room(
            "room-r2", 5.0, 0.0, walls=("wall-c-b",),
            openings=(
                RoomOpening(wall_element_id="wall-c-b", kind="window",
                            width_m=1.0, height_m=1.0),
            ),
        )
        links = link_corridor_to_rooms(corridor, [r1, r2], "corridor-001")
        assert links.corridor_id == "corridor-001"
        assert links.room_ids == ("room-r1",)

    def test_corridor_with_no_doors_links_nothing(self):
        from perception.architecture.topology import link_corridor_to_rooms

        corridor = _room("room-corridor", 0.0, 10.0)
        r1 = _room("room-r1", 0.0, 0.0)
        links = link_corridor_to_rooms(corridor, [r1], "corridor-001")
        assert links.room_ids == ()


class TestStoreyConnectivity:
    def test_stair_links_two_storeys(self):
        from perception.architecture.topology import storey_connectivity

        class _Storey:
            def __init__(self, sid, h):
                self.storey_id = sid
                self.floor_height_m = h

        class _Building:
            storeys = [_Storey("storey-01", 0.0), _Storey("storey-02", 2.8)]

        # One stair spanning z=0..2.8 (measured span), linking both.
        stair_links = [(
            "stairs-001",
            ("storey-01", "storey-02"),
        )]
        edges = storey_connectivity(_Building(), stair_links)
        assert ("storey-01", "storey-02") in edges

    def test_storey_without_stair_is_isolated(self):
        from perception.architecture.topology import storey_connectivity

        class _Storey:
            def __init__(self, sid, h):
                self.storey_id = sid
                self.floor_height_m = h

        class _Building:
            storeys = [_Storey("storey-01", 0.0), _Storey("storey-02", 2.8)]

        edges = storey_connectivity(_Building(), [])
        assert edges == []
