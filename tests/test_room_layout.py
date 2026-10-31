"""S2: the room layout solver -- regularised, closed, labelled. Pure geometry; no reconstruction needed."""

import math

import numpy as np
import pytest

from perception.architecture.room_layout import (
    DEFAULT_THICKNESS_M, box_mesh, polygon_slab_mesh, solve_room_layout, wall_pieces, wall_slab_mesh,
)


def _edge_pts(poly, skip=(), step=0.05):
    """Dense wall points along every footprint edge, except the edges (indices) in ``skip``."""
    pts = []
    for i, (p, q) in enumerate(zip(poly, poly[1:] + poly[:1])):
        if i in skip:
            continue
        n = max(2, int(math.dist(p, q) / step))
        pts += [(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t) for t in np.linspace(0, 1, n)]
    return np.array(pts)


def _lengths(layout):
    return sorted(round(w.length_m, 3) for w in layout.walls)


def test_rectangle_noisy_vertices_snap_to_an_exact_closed_rectangle():
    noisy = [(0.07, 0.04), (5.03, -0.05), (5.06, 4.02), (-0.04, 3.96)]
    lay = solve_room_layout(noisy, [2.6] * 4, _edge_pts(noisy))
    assert lay.closed and len(lay.walls) == 4
    assert all(w.snapped for w in lay.walls) and lay.manhattan_fraction == 1.0
    assert all(w.provenance == "OBSERVED" and w.height_m == 2.6 and w.height_basis == "measured" for w in lay.walls)
    # every wall lies exactly on a multiple of 90 degrees from the fitted dominant direction
    for w in lay.walls:
        ang = math.degrees(math.atan2(w.b[1] - w.a[1], w.b[0] - w.a[0])) - lay.dominant_angle_deg
        assert abs(ang / 90 - round(ang / 90)) < 1e-4  # dominant_angle_deg is rounded to 1e-3 deg
    # L/W within 2 % of the (noisy) measured rectangle's mean size
    assert _lengths(lay)[0] == pytest.approx(4.0, rel=0.02) and _lengths(lay)[-1] == pytest.approx(5.0, rel=0.02)
    assert all(w.thickness_m == DEFAULT_THICKNESS_M and w.thickness_basis == "default_unmeasured" for w in lay.walls)


def test_L_shape_with_redundant_collinear_vertices_and_a_skewed_edge():
    # true L (6x3 arm + 3x2 arm); extra collinear vertices on the long edge, the 3.05->2.95 edge is slightly skewed
    poly = [(0, 0), (3, 0.01), (6, 0), (6, 3), (3.05, 3.0), (2.95, 5), (0, 5)]
    heights = [2.5, 2.5, 2.5, 2.5, 2.8, 2.8, 2.8]
    lay = solve_room_layout(poly, heights, _edge_pts(poly))
    assert lay.closed and len(lay.walls) == 6, "the two collinear south edges merge into one wall"
    assert lay.manhattan_fraction == 1.0
    south = max(lay.walls, key=lambda w: w.length_m)
    assert south.length_m == pytest.approx(6.0, abs=0.05)
    area = 0.5 * abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(lay.polygon, lay.polygon[1:] + lay.polygon[:1])))
    assert area == pytest.approx(24.0, rel=0.02)
    assert sorted({w.height_m for w in lay.walls}) == [2.5, 2.8], "a measured step in wall height survives"


def test_walls_with_different_heights_are_not_merged_even_if_collinear():
    poly = [(0, 0), (3, 0), (6, 0), (6, 4), (0, 4)]
    lay = solve_room_layout(poly, [2.5, 2.9, 2.5, 2.5, 2.5], _edge_pts(poly))
    assert len(lay.walls) == 5


def test_non_manhattan_edge_is_kept_not_forced():
    # a rectangle with one 45-degree chamfered corner
    poly = [(0, 0), (5, 0), (5, 2.5), (3.5, 4), (0, 4)]
    lay = solve_room_layout(poly, [2.6] * 5, _edge_pts(poly))
    diag = [w for w in lay.walls if not w.snapped]
    assert len(diag) == 1
    assert math.degrees(math.atan2(diag[0].b[1] - diag[0].a[1], diag[0].b[0] - diag[0].a[0])) == pytest.approx(135.0, abs=1.0)
    assert 0 < lay.manhattan_fraction < 1


def test_rotated_room_snaps_to_its_own_dominant_axes():
    th = math.radians(33)
    rect = [(0, 0), (5, 0), (5, 4), (0, 4)]
    poly = [(x * math.cos(th) - y * math.sin(th) + 0.02 * (i % 2), x * math.sin(th) + y * math.cos(th)) for i, (x, y) in enumerate(rect)]
    lay = solve_room_layout(poly, [2.6] * 4, _edge_pts(poly))
    assert lay.closed and all(w.snapped for w in lay.walls)
    assert lay.dominant_angle_deg == pytest.approx(33.0, abs=1.5)
    assert _lengths(lay)[-1] == pytest.approx(5.0, rel=0.02)


def test_partially_observed_room_closes_with_labelled_inferred_walls():
    poly = [(0, 0), (5, 0), (5, 4), (0, 4)]
    lay = solve_room_layout(poly, [2.6] * 4, _edge_pts(poly, skip={2}))       # nobody saw the north wall
    assert lay.closed and len(lay.walls) == 4
    north = next(w for w in lay.walls if abs(w.a[1] - 4) < 1e-6 and abs(w.b[1] - 4) < 1e-6)
    assert north.provenance == "INFERRED" and north.observed_fraction < 0.2   # only the corner bins touch the neighbours
    assert len(north.unobserved_spans_m) == 1 and north.unobserved_spans_m[0][1] - north.unobserved_spans_m[0][0] > 4.0
    seen = [w for w in lay.walls if w is not north]
    assert all(w.provenance == "OBSERVED" for w in seen)
    assert north.confidence < min(w.confidence for w in seen)


def test_half_seen_wall_reports_the_unseen_span():
    poly = [(0, 0), (6, 0), (6, 4), (0, 4)]
    pts = _edge_pts(poly)
    pts = pts[~((np.abs(pts[:, 1]) < 1e-6) & (pts[:, 0] > 3.0))]                 # south wall seen only for x < 3
    south = next(w for w in solve_room_layout(poly, [2.6] * 4, pts).walls if abs(w.a[1]) < 1e-6 and abs(w.b[1]) < 1e-6)
    assert south.provenance == "OBSERVED"                                        # half the wall is still evidence
    assert len(south.unobserved_spans_m) == 1
    s0, s1 = south.unobserved_spans_m[0]
    assert s0 == pytest.approx(3.0, abs=0.3) and s1 == pytest.approx(6.0, abs=0.3)


def test_no_points_means_no_claim_of_observation():
    lay = solve_room_layout([(0, 0), (5, 0), (5, 4), (0, 4)], [2.6] * 4, None)
    assert all(w.provenance == "INFERRED" and w.observed_fraction is None for w in lay.walls)


def test_missing_heights_are_none_not_invented():
    lay = solve_room_layout([(0, 0), (5, 0), (5, 4), (0, 4)], None, None)
    assert all(w.height_m is None and w.height_basis == "none" for w in lay.walls)


# ---- slab geometry --------------------------------------------------------------------------------------------

def _volume(verts, faces):
    v = np.array(verts)
    return sum(float(np.dot(v[a], np.cross(v[b], v[c]))) for a, b, c in faces) / 6.0


def test_box_mesh_is_outward_wound():
    v, f = box_mesh([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0), (0, 0, 3), (2, 0, 3), (2, 1, 3), (0, 1, 3)])
    assert _volume(v, f) == pytest.approx(6.0)


def test_wall_pieces_tile_the_wall_minus_openings():
    openings = [{"offset_m": 1.0, "width_m": 0.9, "sill_m": 0.0, "height_m": 2.05},
                {"offset_m": 3.0, "width_m": 1.2, "sill_m": 0.9, "height_m": 1.2}]
    pieces = wall_pieces(5.0, 2.6, openings)
    area = sum((s1 - s0) * (v1 - v0) for s0, s1, v0, v1 in pieces)
    assert area == pytest.approx(5.0 * 2.6 - 0.9 * 2.05 - 1.2 * 1.2)
    for s0, s1, v0, v1 in pieces:                                                # nothing inside a hole
        for o in openings:
            ox0, ox1, oy0, oy1 = o["offset_m"], o["offset_m"] + o["width_m"], o["sill_m"], o["sill_m"] + o["height_m"]
            assert not (s0 < ox1 - 1e-9 and s1 > ox0 + 1e-9 and v0 < oy1 - 1e-9 and v1 > oy0 + 1e-9)


def test_wall_slab_mesh_volume_is_length_height_thickness_minus_the_holes():
    openings = [{"offset_m": 1.0, "width_m": 0.9, "sill_m": 0.0, "height_m": 2.05}]
    v, f = wall_slab_mesh((0, 0, 0), (1, 0, 0), (0, -1, 0), (0, 0, 1), 5.0, 2.6, 0.1, openings)
    assert _volume(v, f) == pytest.approx((5.0 * 2.6 - 0.9 * 2.05) * 0.1)
    ys = [p[1] for p in v]
    assert min(ys) == pytest.approx(-0.1) and max(ys) == pytest.approx(0.0), "slab extends outward from the inner face"


@pytest.mark.parametrize("direction", [-1, 1])
def test_polygon_slab_mesh_volume_matches_the_L_footprint(direction):
    ab = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 5), (0, 5)]
    top = [(a, b, 0.0) for a, b in ab]
    v, f = polygon_slab_mesh(top, (0, 0, 1), 0.15, direction, ab)
    assert _volume(v, f) == pytest.approx(24.0 * 0.15)
    zs = [p[2] for p in v]
    assert (min(zs), max(zs)) == pytest.approx((-0.15, 0.0) if direction < 0 else (0.0, 0.15))
