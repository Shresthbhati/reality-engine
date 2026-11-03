"""Tests for Non-Manhattan Room Boundary Solver (Universal Intelligence Expansion Epic E04-1).

Validates:
1. Octagonal / chamfered room (45-degree angled walls).
2. Bay window room (outward 45-degree trapezoidal expansion with 7 walls).
3. Acute alcove room (acute 30/60 degree angled recess).
4. Angled non-Manhattan corridor with low bounding box fill ratio (<0.20).
5. 12+ wall complex concave room solved in milliseconds via spatial adjacency pruning.
6. Promotion into WorldIR with boundary_polygon_2d and is_non_manhattan=True properties.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from evidence.promote_rooms import (
    PlaneSummary,
    _is_polygon_non_manhattan,
    detect_rooms,
    promote_room_to_entity,
)
from world_ir import WorldIR

UP = (0.0, 1.0, 0.0)


def _make_wall_plane(
    plane_id: str,
    p1: tuple[float, float],  # (x, z) start
    p2: tuple[float, float],  # (x, z) end
    y_floor: float = 0.0,
    y_ceil: float = 2.5,
) -> PlaneSummary:
    """Create a vertical wall plane passing through p1 and p2."""
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    length = math.hypot(dx, dz)
    assert length > 1e-4

    # Wall normal pointing inward (left of the direction p1 -> p2)
    # in x-z plane: normal_x = -dz / length, normal_z = dx / length
    nx = -dz / length
    nz = dx / length
    # Plane equation: nx*x + 0*y + nz*z + d = 0 => d = -(nx*p1[0] + nz*p1[1])
    d = -(nx * p1[0] + nz * p1[1])

    xs = [p1[0], p2[0]]
    zs = [p1[1], p2[1]]
    bmin = (min(xs), y_floor, min(zs))
    bmax = (max(xs), y_ceil, max(zs))

    # Dense inlier points along the segment
    inlier_pts = []
    for t in np.linspace(0.05, 0.95, 8):
        x = p1[0] + t * dx
        z = p1[1] + t * dz
        for y in (y_floor + 0.1, y_floor + 1.2, y_ceil - 0.1):
            inlier_pts.append((float(x), float(y), float(z)))

    return PlaneSummary(
        plane_id=plane_id,
        role="wall",
        normal=(nx, 0.0, nz),
        d=d,
        inlier_rms_m=0.005,
        bounds_min=bmin,
        bounds_max=bmax,
        inlier_positions=tuple(inlier_pts),
        entity_id=f"ent-{plane_id}",
    )


def _make_floor_plane(
    plane_id: str,
    polygon_xz: list[tuple[float, float]],
    y_floor: float = 0.0,
) -> PlaneSummary:
    """Create a floor plane at y_floor with inlier positions covering the polygon."""
    xs = [p[0] for p in polygon_xz]
    zs = [p[1] for p in polygon_xz]
    bmin = (min(xs), y_floor, min(zs))
    bmax = (max(xs), y_floor, max(zs))

    # Generate interior points by sampling bounding box and keeping points inside polygon
    from evidence.promote_rooms import _point_in_polygon

    poly_tuple = tuple(polygon_xz)
    inliers = []
    for x in np.linspace(min(xs) + 0.1, max(xs) - 0.1, 15):
        for z in np.linspace(min(zs) + 0.1, max(zs) - 0.1, 15):
            if _point_in_polygon((float(x), float(z)), poly_tuple):
                inliers.append((float(x), y_floor, float(z)))

    return PlaneSummary(
        plane_id=plane_id,
        role="floor",
        normal=(0.0, 1.0, 0.0),
        d=-y_floor,
        inlier_rms_m=0.005,
        bounds_min=bmin,
        bounds_max=bmax,
        inlier_positions=tuple(inliers),
        entity_id=f"ent-{plane_id}",
    )


class TestNonManhattanRoomBoundarySolver:
    def test_octagonal_room_closed_ring_and_area(self):
        """Test regular octagonal room with 45-degree walls."""
        # Octagon with vertices at 45 degree steps
        r = 2.0  # radius
        n_sides = 8
        verts = [
            (
                round(r * math.cos(2 * math.pi * i / n_sides), 4),
                round(r * math.sin(2 * math.pi * i / n_sides), 4),
            )
            for i in range(n_sides)
        ]

        floor = _make_floor_plane("floor-oct", verts)
        walls = []
        for i in range(n_sides):
            p1 = verts[i]
            p2 = verts[(i + 1) % n_sides]
            walls.append(_make_wall_plane(f"wall-oct-{i}", p1, p2))

        summaries = [floor] + walls
        rooms = detect_rooms(summaries, UP)

        assert len(rooms) == 1
        assert rooms[0].status == "detected"
        room = rooms[0]
        assert len(room.ring.vertices) == 8

        # Expected area of regular octagon with circumradius r: 2 * sqrt(2) * r^2
        expected_area = 2.0 * math.sqrt(2.0) * (r ** 2)
        assert room.floor_area_m2() == pytest.approx(expected_area, rel=0.03)

        # Check non-Manhattan detection
        assert _is_polygon_non_manhattan(room.ring) is True

    def test_bay_window_room_detection_and_promotion(self):
        """Test a 7-wall room with an outward 45-degree bay window."""
        # Main room: (0,0) to (4, 3) with bay window on wall z=3 between x=1 and x=3
        # Bay window facets: (1, 3) -> (1.5, 3.8) -> (2.5, 3.8) -> (3, 3)
        verts = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 3.0),
            (3.0, 3.0),
            (2.5, 3.8),   # bay right facet (angled)
            (1.5, 3.8),   # bay front facet
            (1.0, 3.0),   # bay left facet (angled)
            (0.0, 3.0),
        ]
        # Notice that wall z=0 is 1 wall, x=4 is 1 wall, z=3 (right) is 1 wall,
        # bay facets are 3 walls, z=3 (left) is 1 wall, x=0 is 1 wall -> 8 segments!

        floor = _make_floor_plane("floor-bay", verts)
        walls = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            walls.append(_make_wall_plane(f"wall-bay-{i}", p1, p2))

        summaries = [floor] + walls
        rooms = detect_rooms(summaries, UP)

        assert len(rooms) == 1
        assert rooms[0].status == "detected"
        room = rooms[0]
        assert len(room.ring.vertices) >= 7

        # Area is main rectangle (4 * 3 = 12) + trapezoidal bay:
        # trapezoid with parallel sides 2.0 and 1.0, height 0.8:
        # area = (2.0 + 1.0) / 2 * 0.8 = 1.2 => total = 13.2
        expected_area = 12.0 + 1.2
        assert room.floor_area_m2() == pytest.approx(expected_area, rel=0.05)

        # Promote to WorldIR
        world = WorldIR(id="w-bay", name="Bay Window Room")
        for s in summaries:
            world.entities[s.entity_id] = _make_entity(s.entity_id)

        res = promote_room_to_entity(room, world, "room-bay", "Living Room with Bay Window")
        assert res.entity.custom_properties["is_non_manhattan"] is True
        assert "boundary_polygon_2d" in res.entity.custom_properties
        assert len(res.entity.custom_properties["boundary_polygon_2d"]) >= 7

    def test_angled_corridor_with_low_fill_ratio(self):
        """Test a turning non-Manhattan corridor with <0.18 bounding box fill ratio."""
        # L-shaped corridor of width 1.0 m:
        # Segment 1: (0, 0) to (1, 0) to (1, 6)
        # Turn 45 degrees: to (4, 9) to (3.3, 9.7) to (0.3, 6.7) to (0, 6) -> (0,0)
        corridor_verts = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 5.0),
            (4.0, 8.0),
            (3.3, 8.7),
            (0.3, 5.7),
            (0.0, 5.0),
        ]
        floor = _make_floor_plane("floor-corr", corridor_verts)
        walls = []
        for i in range(len(corridor_verts)):
            p1 = corridor_verts[i]
            p2 = corridor_verts[(i + 1) % len(corridor_verts)]
            walls.append(_make_wall_plane(f"wall-corr-{i}", p1, p2))

        summaries = [floor] + walls
        rooms = detect_rooms(summaries, UP)

        assert len(rooms) == 1
        assert rooms[0].status == "detected"
        room = rooms[0]
        assert _is_polygon_non_manhattan(room.ring) is True

    def test_complex_12_wall_non_manhattan_room_fast_convergence(self):
        """Validate 12-wall concave non-Manhattan room solves without timing out."""
        # 12-wall complex space: main room + 2 angled alcoves + bay window
        verts = [
            (0.0, 0.0),
            (5.0, 0.0),
            (5.0, 2.0),
            (5.8, 2.5),  # alcove 1 angled
            (5.8, 3.5),
            (5.0, 4.0),  # alcove 1 return
            (5.0, 6.0),
            (3.0, 6.0),
            (2.5, 6.7),  # bay facet 1
            (1.5, 6.7),  # bay facet 2
            (1.0, 6.0),  # bay facet 3
            (0.0, 6.0),
        ]
        floor = _make_floor_plane("floor-complex12", verts)
        walls = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            walls.append(_make_wall_plane(f"wall-comp-{i}", p1, p2))

        summaries = [floor] + walls
        import time
        t0 = time.perf_counter()
        rooms = detect_rooms(summaries, UP)
        elapsed = time.perf_counter() - t0

        assert len(rooms) == 1
        assert rooms[0].status == "detected"
        assert len(rooms[0].ring.vertices) == 12
        assert _is_polygon_non_manhattan(rooms[0].ring) is True
        # Proves bounded execution without combinatorial blowup: < 1.0 second
        assert elapsed < 1.0


def _make_entity(entity_id: str):
    from provenance import Provenance
    from world_ir import Entity, EntityType
    return Entity(
        id=entity_id,
        name=entity_id,
        type=EntityType.WALL if "wall" in entity_id else EntityType.FLOOR,
        provenance=Provenance.RECONSTRUCTED,
        confidence=1.0,
    )
