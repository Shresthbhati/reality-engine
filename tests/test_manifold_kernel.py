"""Unit tests for the Manifold Mesh Processing Kernel."""

import numpy as np
import pytest

from geometry.kernels.manifold_kernel import ManifoldMeshKernel


def test_manifold_box_creation_and_topology():
    """Verify standard box mesh is 100% 2-manifold, watertight, with Euler characteristic 2."""
    box = ManifoldMeshKernel.create_box_mesh((0, 0, 0), (2, 3, 4))
    assert box.is_watertight is True
    assert box.is_manifold is True
    assert box.euler_characteristic == 2
    assert len(box.vertices) == 8
    assert len(box.triangles) == 12

    is_m, issues = ManifoldMeshKernel.check_manifold(box.vertices, box.triangles)
    assert is_m is True
    assert len(issues) == 0


def test_manifold_carve_window_opening():
    """Verify carving a window opening through a wall generates 4 watertight sub-solids."""
    wall_min = (0.0, 0.0, 0.0)
    wall_max = (5.0, 0.3, 3.0)  # 5m long, 30cm thick, 3m high wall

    window_min = (2.0, 0.0, 1.0)
    window_max = (3.5, 0.3, 2.2)  # 1.5m wide window starting at 1m sill height

    carved = ManifoldMeshKernel.carve_opening_csg(wall_min, wall_max, window_min, window_max)

    assert carved.is_watertight is True
    assert carved.is_manifold is True
    # 4 sub-solids: left, right, sill, lintel
    assert len(carved.vertices) == 4 * 8  # 32 vertices
    assert len(carved.triangles) == 4 * 12  # 48 triangles

    is_m, issues = ManifoldMeshKernel.check_manifold(carved.vertices, carved.triangles)
    assert is_m is True
    assert len(issues) == 0


def test_manifold_carve_door_opening():
    """Verify carving a door opening (no sill) generates 3 watertight sub-solids."""
    wall_min = (0.0, 0.0, 0.0)
    wall_max = (4.0, 0.2, 2.8)

    door_min = (1.5, 0.0, 0.0)
    door_max = (2.5, 0.2, 2.1)  # 1m wide, 2.1m high door from floor

    carved = ManifoldMeshKernel.carve_opening_csg(wall_min, wall_max, door_min, door_max)

    assert carved.is_watertight is True
    assert carved.is_manifold is True
    # 3 sub-solids: left, right, lintel (no bottom sill)
    assert len(carved.vertices) == 3 * 8  # 24 vertices
    assert len(carved.triangles) == 3 * 12  # 36 triangles
