"""Tests for perception/architecture/room_graph.py (P8-01 room graph +
P8-02 building graph).

The room graph turns the geometry-only architectural classifier's
output (classify_planes) into an architectural ENTITY structure: a
room is NOT merely a mesh -- it has boundaries, openings, adjacency
topology, and measured dimensions. The building graph assembles room
graphs into storeys and a building envelope.

Everything is measured from the classifier's real plane evidence;
insufficient evidence refuses (no room, no guess). Deterministic
ordering throughout. Fixture conventions reused unchanged from
tests/test_architectural_perception.py (the hand-built synthetic room).
"""

import math

import pytest

from perception.architecture.classify import (
    PlaneInput,
    classify_planes,
)
from perception.architecture.room_graph import (
    build_room_graph,
    build_building_graph,
    RoomGraphError,
)

UP = (0.0, 0.0, 1.0)

X_MIN, X_MAX = 0.0, 3.0
Y_MIN, Y_MAX = 0.0, 2.5
Z_MIN, Z_MAX = 0.0, 2.4
DOOR_X = (1.0, 1.6)
DOOR_TOP_Z = 2.0
STEP = 0.1
WALL_STEP = 0.02


def _grid(a_range, b_range, step=STEP):
    a0, a1 = a_range
    b0, b1 = b_range
    n_a = int(round((a1 - a0) / step))
    n_b = int(round((b1 - b0) / step))
    for i in range(n_a + 1):
        for j in range(n_b + 1):
            yield a0 + i * step, b0 + j * step


def _floor_plane():
    pts = [(x, y, Z_MIN) for x, y in _grid((X_MIN, X_MAX), (Y_MIN, Y_MAX))]
    return PlaneInput(
        plane_id="floor", normal=(0.0, 0.0, 1.0), centroid=(1.5, 1.25, Z_MIN),
        bounds_min=(X_MIN, Y_MIN, Z_MIN), bounds_max=(X_MAX, Y_MAX, Z_MIN),
        inlier_positions=tuple(pts),
    )


def _ceiling_plane():
    pts = [(x, y, Z_MAX) for x, y in _grid((X_MIN, X_MAX), (Y_MIN, Y_MAX))]
    return PlaneInput(
        plane_id="ceiling", normal=(0.0, 0.0, 1.0), centroid=(1.5, 1.25, Z_MAX),
        bounds_min=(X_MIN, Y_MIN, Z_MAX), bounds_max=(X_MAX, Y_MAX, Z_MAX),
        inlier_positions=tuple(pts),
    )


def _front_wall_with_doorway():
    pts = []
    for x, z in _grid((X_MIN, X_MAX), (Z_MIN, Z_MAX), step=WALL_STEP):
        if DOOR_X[0] <= x <= DOOR_X[1] and z <= DOOR_TOP_Z:
            continue
        pts.append((x, Y_MIN, z))
    return PlaneInput(
        plane_id="wall-front", normal=(0.0, -1.0, 0.0), centroid=(1.5, Y_MIN, 1.2),
        bounds_min=(X_MIN, Y_MIN, Z_MIN), bounds_max=(X_MAX, Y_MIN, Z_MAX),
        inlier_positions=tuple(pts),
    )


def _solid_side_wall():
    pts = [(X_MIN, y, z) for y, z in _grid((Y_MIN, Y_MAX), (Z_MIN, Z_MAX), step=WALL_STEP)]
    return PlaneInput(
        plane_id="wall-side", normal=(-1.0, 0.0, 0.0), centroid=(X_MIN, 1.25, 1.2),
        bounds_min=(X_MIN, Y_MIN, Z_MIN), bounds_max=(X_MIN, Y_MAX, Z_MAX),
        inlier_positions=tuple(pts),
    )


def _back_wall():
    pts = [(x, Y_MAX, z) for x, z in _grid((X_MIN, X_MAX), (Z_MIN, Z_MAX))]
    return PlaneInput(
        plane_id="wall-back", normal=(0.0, 1.0, 0.0), centroid=(1.5, Y_MAX, 1.2),
        bounds_min=(X_MIN, Y_MAX, Z_MIN), bounds_max=(X_MAX, Y_MAX, Z_MAX),
        inlier_positions=tuple(pts),
    )


def _other_side_wall():
    pts = [(X_MAX, y, z) for y, z in _grid((Y_MIN, Y_MAX), (Z_MIN, Z_MAX))]
    return PlaneInput(
        plane_id="wall-side2", normal=(1.0, 0.0, 0.0), centroid=(X_MAX, 1.25, 1.2),
        bounds_min=(X_MAX, Y_MIN, Z_MIN), bounds_max=(X_MAX, Y_MAX, Z_MAX),
        inlier_positions=tuple(pts),
    )


def _room_planes():
    return [
        _floor_plane(), _ceiling_plane(), _front_wall_with_doorway(),
        _back_wall(), _solid_side_wall(), _other_side_wall(),
    ]


# ------------------------------------------------------------------
# P8-01: room graph
# ------------------------------------------------------------------


class TestRoomGraph:
    def test_single_room_from_complete_planes(self):
        elements = classify_planes(_room_planes(), UP)
        rooms = build_room_graph(elements, up=UP, planes=_room_planes())
        # One enclosing cell of walls+floor+ceiling -> exactly one room.
        assert len(rooms) == 1
        room = rooms[0]
        assert room.room_id == "room-001"
        # Boundaries: every wall/floor/ceiling element participates
        # (element ids carry the classifier's role prefix).
        assert room.boundary_element_ids == (
            "ceiling-ceiling", "floor-floor", "wall-wall-back",
            "wall-wall-front", "wall-wall-side", "wall-wall-side2",
        )
        # The doorway is an opening ON the front wall, measured.
        assert room.openings and len(room.openings) == 1
        opening = room.openings[0]
        assert opening.wall_element_id == "wall-wall-front"
        assert opening.kind == "doorway"
        assert opening.width_m == pytest.approx(DOOR_X[1] - DOOR_X[0], abs=0.05)
        assert opening.height_m == pytest.approx(DOOR_TOP_Z - Z_MIN, abs=0.05)

    def test_room_dimensions_measured_from_planes(self):
        elements = classify_planes(_room_planes(), UP)
        room = build_room_graph(elements, up=UP, planes=_room_planes())[0]
        d = room.dimensions_m
        assert d["x"] == pytest.approx(X_MAX - X_MIN, abs=0.05)
        assert d["y"] == pytest.approx(Y_MAX - Y_MIN, abs=0.05)
        assert d["z"] == pytest.approx(Z_MAX - Z_MIN, abs=0.05)
        assert room.floor_area_m2 == pytest.approx(
            (X_MAX - X_MIN) * (Y_MAX - Y_MIN), rel=0.02
        )

    def test_no_room_without_full_enclosure(self):
        # Floor + two walls: not a room -- refuse, never guess. (No
        # planes supplied either: openings honestly undetected.)
        elements = classify_planes(
            [_floor_plane(), _back_wall(), _solid_side_wall()], UP
        )
        assert build_room_graph(elements, up=UP, planes=_room_planes()) == []

    def test_two_enclosures_two_rooms_with_shared_wall_adjacency(self):
        # Two rooms sharing the x=X_MAX wall: the SECOND room's planes
        # are the first room's planes shifted by +3 in x, minus its own
        # left wall (the shared wall-side2 plane serves both rooms).
        # Elements are constructed directly (the classifier labels
        # floor/ceiling globally, which is single-room scope; a
        # multi-room pipeline feeds per-room plane sets and merges).
        shift = X_MAX - X_MIN
        els1 = classify_planes(_room_planes(), UP)
        by_src1 = {e.source_plane_id: e for e in els1}

        def shifted(p):
            bmin = tuple(c + (shift if i == 0 else 0.0) for i, c in enumerate(p.bounds_min))
            bmax = tuple(c + (shift if i == 0 else 0.0) for i, c in enumerate(p.bounds_max))
            cen = tuple(c + (shift if i == 0 else 0.0) for i, c in enumerate(p.centroid))
            return PlaneInput(
                plane_id=p.plane_id + "-2", normal=p.normal, centroid=cen,
                bounds_min=bmin, bounds_max=bmax,
                inlier_positions=tuple((x + shift, y, z) for x, y, z in p.inlier_positions),
            )

        els2 = []
        for p in _room_planes():
            if p.plane_id == "wall-side2":
                continue  # shared wall: room 1's element serves both
            e = shifted(p)
            role = "wall" if p.plane_id.startswith("wall") else (
                "floor" if p.plane_id == "floor" else "ceiling"
            )
            from perception.architecture.classify import ArchitecturalElement
            els2.append(ArchitecturalElement(
                element_id=f"{role}-{e.plane_id}",
                element_type=role,
                source_plane_id=e.plane_id,
                reason="constructed fixture (shifted second room)",
                bounds_min=e.bounds_min, bounds_max=e.bounds_max,
            ))
        all_elements = els1 + els2
        rooms = build_room_graph(all_elements, up=UP)
        assert len(rooms) == 2
        assert [r.room_id for r in rooms] == ["room-001", "room-002"]
        r1, r2 = rooms
        assert "wall-wall-side2" in r1.boundary_element_ids
        assert "wall-wall-side2" in r2.boundary_element_ids
        assert r2.room_id in r1.adjacent_room_ids
        assert r1.room_id in r2.adjacent_room_ids

    def test_determinism(self):
        elements = classify_planes(_room_planes(), UP)
        rooms1 = build_room_graph(elements, up=UP)
        rooms2 = build_room_graph(list(reversed(elements)), up=UP)
        assert [r.to_dict() for r in rooms1] == [r.to_dict() for r in rooms2]


# ------------------------------------------------------------------
# P8-02: building graph
# ------------------------------------------------------------------


class TestBuildingGraph:
    def test_single_room_building(self):
        elements = classify_planes(_room_planes(), UP)
        rooms = build_room_graph(elements, up=UP)
        b = build_building_graph(rooms, up=UP)
        assert len(b.storeys) == 1
        assert b.storeys[0].room_ids == (rooms[0].room_id,)
        # Envelope bounds = the union of the rooms' measured bounds.
        assert b.envelope_bounds_min == pytest.approx(
            tuple(rooms[0].bounds_min), abs=0.01
        )
        assert b.envelope_bounds_max == pytest.approx(
            tuple(rooms[0].bounds_max), abs=0.01
        )

    def test_two_storeys_stacked_rooms(self):
        elements = classify_planes(_room_planes(), UP)
        rooms = build_room_graph(elements, up=UP)
        # Duplicate the room shifted up by its own height (a second
        # storey): shift every plane by +2.4 in z.
        shift = Z_MAX - Z_MIN
        upper = []
        for p in _room_planes():
            bmin = tuple(c + (shift if i == 2 else 0.0) for i, c in enumerate(p.bounds_min))
            bmax = tuple(c + (shift if i == 2 else 0.0) for i, c in enumerate(p.bounds_max))
            cEN = tuple(c + (shift if i == 2 else 0.0) for i, c in enumerate(p.centroid))
            upper.append(PlaneInput(
                plane_id=p.plane_id + "-L2", normal=p.normal, centroid=cEN,
                bounds_min=bmin, bounds_max=bmax,
                inlier_positions=tuple((x, y, z + shift) for x, y, z in p.inlier_positions),
            ))
        # The classifier labels floor/ceiling GLOBALLY per call
        # (lowest/highest horizontal plane), which is single-room
        # scope: a two-storey pipeline classifies each storey's planes
        # separately and merges the element lists -- the graph builder
        # consumes classified elements, per-storey provenance kept.
        els_low = classify_planes(_room_planes(), UP)
        els_up = classify_planes(upper, UP)
        rooms2 = build_room_graph(
            els_low + els_up, up=UP, planes=_room_planes() + upper
        )
        assert len(rooms2) == 2, f"expected 2 rooms, got {len(rooms2)}"
        b = build_building_graph(rooms2, up=UP)
        assert len(b.storeys) == 2
        # Storeys ordered bottom-up by measured floor height.
        z0 = b.storeys[0].floor_height_m
        z1 = b.storeys[1].floor_height_m
        assert z1 > z0
        assert b.storeys[0].room_ids != b.storeys[1].room_ids

    def test_empty_rooms_no_building(self):
        assert build_building_graph([], up=UP) is None

    def test_unsorted_input_deterministic(self):
        elements = classify_planes(_room_planes(), UP)
        rooms = build_room_graph(elements, up=UP)
        b1 = build_building_graph(rooms, up=UP)
        b2 = build_building_graph(list(reversed(rooms)), up=UP)
        assert b1.to_dict() == b2.to_dict()
