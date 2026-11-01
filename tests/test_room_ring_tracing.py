"""Ring tracing: convex rooms unchanged; simple non-convex (L/U) rooms are
traced; invalid/unsupported soups fail honestly (None)."""

from __future__ import annotations

import math

from evidence.promote_rooms import (
    _WallLine,
    _point_in_polygon,
    _trace_ring_from_lines,
)


def _lines_from_polygon(poly, ids=None, shrink=0.0):
    """One wall line per polygon edge (direction = edge direction, support
    = edge extent), in a deliberately scrambled order."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        d = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        out.append(_WallLine(
            point=a, direction=d,
            support_lo=shrink, support_hi=length - shrink,
            wall_plane_ids=(f"w{i}",),
        ))
    return out[::2] + out[1::2]


def _area(poly):
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2.0


L_SHAPE = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 6), (0, 6)]
U_SHAPE = [(0, 0), (9, 0), (9, 6), (6, 6), (6, 2), (3, 2), (3, 6), (0, 6)]
RECT = [(0, 0), (5, 0), (5, 4), (0, 4)]


def _same_vertices(ring, poly):
    return {tuple(round(c, 6) for c in v) for v in ring.vertices} == {
        (float(x), float(y)) for x, y in poly
    }


def test_convex_rectangle_unchanged():
    ring = _trace_ring_from_lines(_lines_from_polygon(RECT), (2.5, 2.0))
    assert ring is not None and _same_vertices(ring, RECT)
    assert math.isclose(ring.shoelace_area(), 20.0)


def test_convex_triangle_unchanged():
    tri = [(0, 0), (6, 0), (0, 6)]
    ring = _trace_ring_from_lines(_lines_from_polygon(tri), (2.0, 2.0))
    assert ring is not None and math.isclose(ring.shoelace_area(), 18.0)


def test_l_shape_traced():
    ring = _trace_ring_from_lines(_lines_from_polygon(L_SHAPE), (1.5, 1.5))
    assert ring is not None and _same_vertices(ring, L_SHAPE)
    assert math.isclose(ring.shoelace_area(), _area(L_SHAPE))
    assert len(ring.segment_wall_ids) == 6
    # The notch is outside the ring.
    assert not _point_in_polygon((4.5, 4.5), ring.vertices)
    assert _point_in_polygon((1.0, 5.0), ring.vertices)


def test_l_shape_with_centroid_in_notch_side_still_traced():
    # interior_point only seeds the convex walk; concave trace must not
    # depend on it.
    ring = _trace_ring_from_lines(_lines_from_polygon(L_SHAPE), (4.5, 4.5))
    assert ring is not None and _same_vertices(ring, L_SHAPE)


def test_u_shape_traced():
    ring = _trace_ring_from_lines(_lines_from_polygon(U_SHAPE), (1.0, 3.0))
    assert ring is not None and _same_vertices(ring, U_SHAPE)
    assert math.isclose(ring.shoelace_area(), _area(U_SHAPE))
    assert not _point_in_polygon((4.5, 4.0), ring.vertices)  # inside the slot


def test_l_shape_with_ragged_support_traced():
    # Edge-column loss shrinks support intervals; corners stay reachable.
    ring = _trace_ring_from_lines(_lines_from_polygon(L_SHAPE, shrink=0.4), (1.5, 1.5))
    assert ring is not None and _same_vertices(ring, L_SHAPE)


def test_l_shape_deterministic():
    a = _trace_ring_from_lines(_lines_from_polygon(L_SHAPE), (1.5, 1.5))
    b = _trace_ring_from_lines(list(reversed(_lines_from_polygon(L_SHAPE))), (1.5, 1.5))
    assert a is not None and b is not None
    assert set(a.vertices) == set(b.vertices)


def test_missing_wall_fails_honestly():
    lines = _lines_from_polygon(L_SHAPE)[:-1]  # one wall never observed
    assert _trace_ring_from_lines(lines, (1.5, 1.5)) is None


def test_two_walls_fail():
    assert _trace_ring_from_lines(_lines_from_polygon(RECT)[:2], (2.0, 2.0)) is None


def test_unsupported_corner_fails():
    # Walls far too short to reach their corners: nothing closes.
    lines = _lines_from_polygon(L_SHAPE, shrink=1.4)
    assert _trace_ring_from_lines(lines, (1.5, 1.5)) is None
