"""Tests for perception/architecture/classify.py (P7-03).

Uses a small hand-constructed synthetic room with KNOWN ground truth
(floor, ceiling, 4 walls, one wall with a doorway-sized gap, plus one
deliberately ambiguous sloped plane) rather than running the full
scripts/render_room_dataset.py renderer -- same geometric conventions
(axis-aligned box, a doorway gap starting at the floor) but fast as a
unit test.
"""

import math

import pytest

from perception.architecture.classify import (
    PlaneInput,
    classify_planes,
    detect_wall_opening,
)

UP = (0.0, 0.0, 1.0)

# Hand-computable room: x in [0,3], y in [0,2.5] (depth), z in [0,2.4] (height).
X_MIN, X_MAX = 0.0, 3.0
Y_MIN, Y_MAX = 0.0, 2.5
Z_MIN, Z_MAX = 0.0, 2.4
DOOR_X = (1.0, 1.6)  # gap in the y=0 (front) wall, floor to head height
DOOR_TOP_Z = 2.0
STEP = 0.1
# Finer than classify.SCAN_BUCKET_M (0.05) so the doorway-scan buckets
# never alias against grid spacing and read a solid wall as gappy.
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
    """y=0 wall, with a doorway gap in x=[1.0,1.6] from floor to 2.0 m."""
    pts = []
    for x, z in _grid((X_MIN, X_MAX), (Z_MIN, Z_MAX), step=WALL_STEP):
        if DOOR_X[0] <= x <= DOOR_X[1] and z <= DOOR_TOP_Z:
            continue  # the doorway hole: no wall material here
        pts.append((x, Y_MIN, z))
    return PlaneInput(
        plane_id="wall-front", normal=(0.0, -1.0, 0.0), centroid=(1.5, Y_MIN, 1.2),
        bounds_min=(X_MIN, Y_MIN, Z_MIN), bounds_max=(X_MAX, Y_MIN, Z_MAX),
        inlier_positions=tuple(pts),
    )


def _solid_side_wall():
    """x=0 wall, no gap at all."""
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


def _sloped_ambiguous_plane():
    """Neither wall-steep nor floor/ceiling-flat -- a roof-pitch-like
    plane this repo has no detector for. Must classify UNKNOWN, never
    guessed into wall/floor/ceiling."""
    normal = (0.0, 0.6, 0.8)  # |n.up| = 0.8: between the two thresholds
    return PlaneInput(
        plane_id="sloped", normal=normal, centroid=(1.5, 1.25, 2.0),
        bounds_min=(0.0, 0.0, 1.8), bounds_max=(3.0, 2.5, 2.4),
    )


def test_classifies_walls_floor_ceiling_correctly():
    planes = [
        _floor_plane(), _ceiling_plane(), _front_wall_with_doorway(),
        _back_wall(), _solid_side_wall(), _other_side_wall(),
    ]
    results = {el.source_plane_id: el for el in classify_planes(planes, UP)}

    assert results["floor"].element_type == "floor"
    assert results["ceiling"].element_type == "ceiling"
    assert results["wall-front"].element_type == "wall"
    assert results["wall-back"].element_type == "wall"
    assert results["wall-side"].element_type == "wall"
    assert results["wall-side2"].element_type == "wall"


def test_ambiguous_sloped_plane_is_unknown_not_guessed():
    """The sharpest scope-discipline check: a plane that matches
    neither the wall nor floor/ceiling orientation range must never be
    silently classified as one anyway."""
    planes = [_floor_plane(), _ceiling_plane(), _sloped_ambiguous_plane()]
    results = {el.source_plane_id: el for el in classify_planes(planes, UP)}
    assert results["sloped"].element_type == "unknown"
    assert "ambiguous" in results["sloped"].reason


def test_single_horizontal_plane_without_walls_is_unknown():
    """No walls to resolve floor-vs-ceiling against -> honest UNKNOWN,
    not a guess."""
    planes = [_floor_plane()]
    results = classify_planes(planes, UP)
    assert len(results) == 1
    assert results[0].element_type == "unknown"


def test_detects_doorway_gap_in_front_wall():
    wall = _front_wall_with_doorway()
    opening = detect_wall_opening(wall, UP, floor_height=Z_MIN)
    assert opening is not None
    assert opening.element_type == "door"
    assert opening.source_plane_id == "wall-front"
    # Detected lateral span should land inside the true doorway range,
    # with correct width (within one scan bucket).
    lat_lo, lat_hi = opening.bounds_min[0], opening.bounds_max[0]
    assert DOOR_X[0] - 0.15 <= lat_lo <= DOOR_X[0] + 0.15
    assert DOOR_X[1] - 0.15 <= lat_hi <= DOOR_X[1] + 0.15
    width = lat_hi - lat_lo
    assert abs(width - (DOOR_X[1] - DOOR_X[0])) < 0.15


def test_solid_wall_has_no_false_doorway():
    """A wall with full coverage and no real gap must never report a
    fabricated doorway."""
    wall = _solid_side_wall()
    assert detect_wall_opening(wall, UP, floor_height=Z_MIN) is None


def test_no_inliers_means_no_opening_detected():
    """No evidence -> no classification, never a guess."""
    wall = PlaneInput(
        plane_id="wall-empty", normal=(0.0, -1.0, 0.0), centroid=(1.5, 0.0, 1.2),
        bounds_min=(X_MIN, Y_MIN, Z_MIN), bounds_max=(X_MAX, Y_MIN, Z_MAX),
        inlier_positions=(),
    )
    assert detect_wall_opening(wall, UP, floor_height=Z_MIN) is None
