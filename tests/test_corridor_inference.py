"""Tests for automatic corridor reconstruction from architectural geometry and topology."""

from __future__ import annotations

import pytest

from perception.architecture.classify import ArchitecturalElement, PlaneInput
from perception.architecture.corridor import (
    CORRIDOR_MAX_WIDTH_M,
    CORRIDOR_MIN_ASPECT_RATIO,
    CORRIDOR_MIN_LENGTH_M,
    CORRIDOR_MIN_WIDTH_M,
    CorridorGraph,
    CorridorIntersection,
    detect_corridors,
)


def _make_plane_input(
    plane_id: str,
    normal: tuple[float, float, float],
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
) -> PlaneInput:
    centroid = (
        (bmin[0] + bmax[0]) / 2.0,
        (bmin[1] + bmax[1]) / 2.0,
        (bmin[2] + bmax[2]) / 2.0,
    )
    return PlaneInput(
        plane_id=plane_id,
        normal=normal,
        centroid=centroid,
        bounds_min=bmin,
        bounds_max=bmax,
    )


def test_corridor_detected_with_valid_aspect_ratio_and_enclosing_walls():
    """A space with aspect ratio >= 2.0, width 1.2m, length 5.0m, floor, ceiling, and side walls is detected as corridor."""
    up = (0.0, 0.0, 1.0)

    # Floor at Z=0.0: X from 0.0 to 1.2 (width 1.2), Y from 0.0 to 5.0 (length 5.0)
    floor_pi = _make_plane_input("floor-1", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.2, 5.0, 0.0))
    floor_el = ArchitecturalElement(
        element_id="el-floor-1",
        element_type="floor",
        source_plane_id="floor-1",
        reason="horizontal upward normal",
        bounds_min=floor_pi.bounds_min,
        bounds_max=floor_pi.bounds_max,
    )

    # Ceiling at Z=2.5
    ceil_pi = _make_plane_input("ceil-1", (0.0, 0.0, -1.0), (0.0, 0.0, 2.5), (1.2, 5.0, 2.5))
    ceil_el = ArchitecturalElement(
        element_id="el-ceil-1",
        element_type="ceiling",
        source_plane_id="ceil-1",
        reason="horizontal downward normal",
        bounds_min=ceil_pi.bounds_min,
        bounds_max=ceil_pi.bounds_max,
    )

    # Left Wall at X=0.0, Y from 0.0 to 5.0, Z from 0.0 to 2.5
    w1_pi = _make_plane_input("w-1", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 5.0, 2.5))
    w1_el = ArchitecturalElement(
        element_id="el-w1",
        element_type="wall",
        source_plane_id="w-1",
        reason="vertical normal",
        bounds_min=w1_pi.bounds_min,
        bounds_max=w1_pi.bounds_max,
    )

    # Right Wall at X=1.2, Y from 0.0 to 5.0, Z from 0.0 to 2.5
    w2_pi = _make_plane_input("w-2", (-1.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.2, 5.0, 2.5))
    w2_el = ArchitecturalElement(
        element_id="el-w2",
        element_type="wall",
        source_plane_id="w-2",
        reason="vertical normal",
        bounds_min=w2_pi.bounds_min,
        bounds_max=w2_pi.bounds_max,
    )

    elements = [floor_el, ceil_el, w1_el, w2_el]
    plane_inputs = {
        "floor-1": floor_pi,
        "ceil-1": ceil_pi,
        "w-1": w1_pi,
        "w-2": w2_pi,
    }

    corridors = detect_corridors(elements, up=up, plane_inputs=plane_inputs)
    assert len(corridors) == 1
    c = corridors[0]
    assert c.aspect_ratio == pytest.approx(5.0 / 1.2, rel=0.05)
    assert c.width_m == pytest.approx(1.2, rel=0.05)
    assert c.length_m == pytest.approx(5.0, rel=0.05)
    assert c.height_m == pytest.approx(2.5, rel=0.05)
    assert c.floor_area_m2 == pytest.approx(6.0, rel=0.05)
    # Longitudinal axis along Y
    assert abs(c.longitudinal_axis[1]) > 0.9


def test_corridor_refused_when_aspect_ratio_too_low():
    """A square room (aspect ratio ~ 1.0) is refused as a corridor."""
    up = (0.0, 0.0, 1.0)

    # Floor: 4.0m wide along X, 4.0m long along Y, Z=0.0
    floor_pi = _make_plane_input("floor-sq", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))
    floor_el = ArchitecturalElement(
        element_id="el-floor-sq",
        element_type="floor",
        source_plane_id="floor-sq",
        reason="floor",
        bounds_min=floor_pi.bounds_min,
        bounds_max=floor_pi.bounds_max,
    )
    elements = [floor_el]
    plane_inputs = {"floor-sq": floor_pi}

    corridors = detect_corridors(elements, up=up, plane_inputs=plane_inputs)
    assert len(corridors) == 0


def test_corridor_intersections_detected():
    """Two perpendicular corridors intersecting form a corridor junction."""
    up = (0.0, 0.0, 1.0)

    # Corridor 1 along Y: X: 0 to 1.0, Y: 0 to 5.0, Z: 0 to 2.4
    f1_pi = _make_plane_input("f-c1", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 5.0, 0.0))
    f1_el = ArchitecturalElement("el-fc1", "floor", "f-c1", "floor", bounds_min=f1_pi.bounds_min, bounds_max=f1_pi.bounds_max)
    cl1_pi = _make_plane_input("cl-c1", (0.0, 0.0, -1.0), (0.0, 0.0, 2.4), (1.0, 5.0, 2.4))
    cl1_el = ArchitecturalElement("el-clc1", "ceiling", "cl-c1", "ceiling", bounds_min=cl1_pi.bounds_min, bounds_max=cl1_pi.bounds_max)
    w1_pi = _make_plane_input("w-c1a", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 5.0, 2.4))
    w1_el = ArchitecturalElement("el-wc1a", "wall", "w-c1a", "wall", bounds_min=w1_pi.bounds_min, bounds_max=w1_pi.bounds_max)
    w1b_pi = _make_plane_input("w-c1b", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 5.0, 2.4))
    w1b_el = ArchitecturalElement("el-wc1b", "wall", "w-c1b", "wall", bounds_min=w1b_pi.bounds_min, bounds_max=w1b_pi.bounds_max)

    # Corridor 2 along X: X: -2.0 to 3.0, Y: 2.0 to 3.0, Z: 0 to 2.4 (intersects Corridor 1 at X in [0.0, 1.0], Y in [2.0, 3.0])
    f2_pi = _make_plane_input("f-c2", (0.0, 0.0, 1.0), (-2.0, 2.0, 0.0), (3.0, 3.0, 0.0))
    f2_el = ArchitecturalElement("el-fc2", "floor", "f-c2", "floor", bounds_min=f2_pi.bounds_min, bounds_max=f2_pi.bounds_max)
    cl2_pi = _make_plane_input("cl-c2", (0.0, 0.0, -1.0), (-2.0, 2.0, 2.4), (3.0, 3.0, 2.4))
    cl2_el = ArchitecturalElement("el-clc2", "ceiling", "cl-c2", "ceiling", bounds_min=cl2_pi.bounds_min, bounds_max=cl2_pi.bounds_max)
    w2_pi = _make_plane_input("w-c2a", (0.0, 1.0, 0.0), (-2.0, 2.0, 0.0), (3.0, 2.0, 2.4))
    w2_el = ArchitecturalElement("el-wc2a", "wall", "w-c2a", "wall", bounds_min=w2_pi.bounds_min, bounds_max=w2_pi.bounds_max)
    w2b_pi = _make_plane_input("w-c2b", (0.0, -1.0, 0.0), (-2.0, 3.0, 0.0), (3.0, 3.0, 2.4))
    w2b_el = ArchitecturalElement("el-wc2b", "wall", "w-c2b", "wall", bounds_min=w2b_pi.bounds_min, bounds_max=w2b_pi.bounds_max)

    elements = [f1_el, cl1_el, w1_el, w1b_el, f2_el, cl2_el, w2_el, w2b_el]
    plane_inputs = {
        "f-c1": f1_pi, "cl-c1": cl1_pi, "w-c1a": w1_pi, "w-c1b": w1b_pi,
        "f-c2": f2_pi, "cl-c2": cl2_pi, "w-c2a": w2_pi, "w-c2b": w2b_pi,
    }

    corridors = detect_corridors(elements, up=up, plane_inputs=plane_inputs)
    assert len(corridors) == 2
    # At least one corridor has an intersection with the other
    assert any(len(c.intersections) > 0 for c in corridors)
    ix = [i for c in corridors for i in c.intersections][0]
    assert ix.kind in ("T-junction", "cross-junction")

