"""Tests for non-convex room prismatic 3D mesh generation and volume calculation."""

from __future__ import annotations

import math

import pytest

from reconstruction.meshing.room_mesh import (
    generate_prismatic_room_mesh,
    is_non_convex,
    room_mesh_surface_area,
    room_mesh_volume,
)


def test_rectangular_room_mesh_generation():
    # 4m x 5m room, 2.8m ceiling height
    rect = [(0.0, 0.0), (4.0, 0.0), (4.0, 5.0), (0.0, 5.0)]
    h = 2.8

    assert is_non_convex(rect) is False
    vol = room_mesh_volume(rect, h)
    assert math.isclose(vol, 4.0 * 5.0 * 2.8, rel_tol=1e-5)

    sa = room_mesh_surface_area(rect, h)
    expected_sa = 2 * (4.0 * 5.0) + (2 * 4.0 + 2 * 5.0) * 2.8
    assert math.isclose(sa, expected_sa, rel_tol=1e-5)

    mesh = generate_prismatic_room_mesh(rect, height_m=h, ground_z_m=0.0)
    assert len(mesh.vertices) == 8  # 4 floor + 4 ceiling
    # 2 floor triangles + 2 ceiling triangles + 4 * 2 wall triangles = 12 triangles (standard cube/box)
    assert len(mesh.faces) == 12
    assert mesh.is_watertight_ish() is True


def test_l_shaped_non_convex_room_mesh():
    # L-shaped room: 6 vertices, 1 reflex corner at (3.0, 2.0)
    l_shape = [
        (0.0, 0.0),
        (5.0, 0.0),
        (5.0, 2.0),
        (3.0, 2.0),
        (3.0, 6.0),
        (0.0, 6.0),
    ]
    h = 3.0

    assert is_non_convex(l_shape) is True

    # Area = (3 * 6) + (2 * 2) = 18 + 4 = 22 m2
    expected_vol = 22.0 * 3.0  # 66 m3
    vol = room_mesh_volume(l_shape, h)
    assert math.isclose(vol, expected_vol, rel_tol=1e-5)

    mesh = generate_prismatic_room_mesh(l_shape, height_m=h, ground_z_m=0.0)
    assert len(mesh.vertices) == 12  # 6 floor + 6 ceiling
    # ear_clip triangulates 6-gon into (6-2) = 4 triangles
    # 4 floor + 4 ceiling + 6 * 2 walls = 20 triangles
    assert len(mesh.faces) == 20
    assert mesh.is_watertight_ish() is True


def test_t_shaped_non_convex_room_mesh():
    # T-shaped room: 8 vertices, 2 reflex corners
    t_shape = [
        (2.0, 0.0),
        (4.0, 0.0),
        (4.0, 4.0),
        (6.0, 4.0),
        (6.0, 6.0),
        (0.0, 6.0),
        (0.0, 4.0),
        (2.0, 4.0),
    ]
    h = 2.5

    assert is_non_convex(t_shape) is True
    # Stem: 2 * 4 = 8. Cap: 6 * 2 = 12. Total area = 20 m2
    vol = room_mesh_volume(t_shape, h)
    assert math.isclose(vol, 20.0 * 2.5, rel_tol=1e-5)

    mesh = generate_prismatic_room_mesh(t_shape, height_m=h)
    assert len(mesh.vertices) == 16  # 8 floor + 8 ceiling
    # 8-gon: 6 floor + 6 ceiling + 8 * 2 walls = 28 triangles
    assert len(mesh.faces) == 28
    assert mesh.is_watertight_ish() is True


def test_invalid_room_inputs_raise():
    with pytest.raises(ValueError):
        generate_prismatic_room_mesh([(0.0, 0.0), (1.0, 0.0)], height_m=2.5)
    with pytest.raises(ValueError):
        generate_prismatic_room_mesh([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], height_m=-1.0)
