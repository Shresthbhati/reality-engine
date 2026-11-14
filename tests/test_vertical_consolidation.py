"""Fragments of one wall are one wall; opposite, perpendicular and far-apart walls are not merged."""

from __future__ import annotations

from perception.geometry.orientation import OrientedPlane, consolidate_vertical_roles
from perception.geometry.planes import DetectedPlane
from provenance import Uncertainty

UP = (0.0, 1.0, 0.0)


def _wall(pid, normal, origin, along, n_inliers, positions, length=4.0, offset_noise=0.0):
    """A vertical rectangle of points: ``along`` is its horizontal run, 2 high, pushed ``offset_noise`` along the normal."""
    ids = [f"{pid}-{k:04d}" for k in range(n_inliers)]
    for k, tid in enumerate(ids):
        s, h = (k % 10) / 9.0 * length, (k // 10 % 10) / 9.0 * 2.0
        wob = offset_noise * ((k % 3) - 1)
        positions[tid] = tuple(origin[a] + along[a] * s + normal[a] * wob + (h if a == 1 else 0.0) for a in range(3))
    dp = DetectedPlane(plane_id=pid, normal=normal, d=0.0, inlier_ids=ids, inlier_rms_distance_m=0.01,
                       uncertainty=Uncertainty(confidence=0.9, note="x"))
    return OrientedPlane(plane=dp, role="wall", normal=normal, d=0.0, tilt_from_role_rad=0.0,
                         uncertainty=Uncertainty(confidence=0.9, note="x"))


def _roles(planes):
    return {p.plane.plane_id: p.role for p in planes}


def test_touching_fragments_on_one_line_merge_but_opposite_perpendicular_and_distant_walls_stay():
    pos = {}
    planes = [
        _wall("main", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 300, pos),
        _wall("frag", (0.0, 0.0, 1.0), (3.5, 0.0, 0.02), (1.0, 0.0, 0.0), 120, pos, length=3.0),   # touches main's end
        _wall("opposite", (0.0, 0.0, -1.0), (0.0, 0.0, 5.0), (1.0, 0.0, 0.0), 200, pos),
        _wall("side", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 180, pos, length=5.0),
        _wall("far", (0.0, 0.0, 1.0), (30.0, 0.0, 0.0), (1.0, 0.0, 0.0), 90, pos, length=3.0),     # same line, 24 m away
    ]
    out, stats = consolidate_vertical_roles(planes, UP, pos)
    assert _roles(out) == {"main": "wall", "frag": "unknown", "opposite": "wall", "side": "wall", "far": "wall"}
    assert stats == {"merged": 1} and len(out[0].plane.inlier_ids) == 420
    assert "merged into it" in out[1].uncertainty.note and "main" in out[1].uncertainty.note


def test_a_parallel_wall_offset_by_more_than_the_walls_thickness_is_a_different_wall():
    pos = {}
    planes = [_wall("a", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 300, pos),
              _wall("b", (0.0, 0.0, 1.0), (0.0, 0.0, 1.5), (1.0, 0.0, 0.0), 200, pos)]       # a pilaster/step 1.5 proud
    out, stats = consolidate_vertical_roles(planes, UP, pos)
    assert _roles(out) == {"a": "wall", "b": "wall"} and stats == {"merged": 0}


def test_floors_unknowns_and_tiny_input_pass_through():
    pos = {}
    w = _wall("w", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 50, pos)
    assert consolidate_vertical_roles([w], UP, pos) == ([w], {"merged": 0})
    assert consolidate_vertical_roles([], UP, {}) == ([], {"merged": 0})
