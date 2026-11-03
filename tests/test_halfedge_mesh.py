"""Unit tests for OpenMesh topological halfedge data structure."""

import numpy as np
import pytest

from geometry.halfedge_mesh import HalfedgeMesh


def create_tetrahedron():
    """Create a closed, watertight 4-vertex tetrahedron (chi = 2)."""
    verts = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ], dtype=np.float64)
    # 4 faces
    tris = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ], dtype=np.int64)
    return verts, tris


def create_open_quad():
    """Create an open 4-vertex quad made of two triangles with open boundary."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    tris = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int64)
    return verts, tris


def test_tetrahedron_is_watertight_and_euler_2():
    """Verify closed tetrahedron has Euler characteristic chi = 2 and zero boundary edges."""
    verts, tris = create_tetrahedron()
    mesh = HalfedgeMesh.from_indexed_triangles(verts, tris)

    assert len(mesh.vertices) == 4
    assert len(mesh.faces) == 4
    assert mesh.num_edges == 6  # 6 distinct edges
    assert mesh.is_watertight() is True
    assert len(mesh.get_boundary_halfedges()) == 0
    # Euler characteristic: V - E + F = 4 - 6 + 4 = 2
    assert mesh.compute_euler_characteristic() == 2


def test_open_quad_boundary_detection():
    """Verify open quad mesh correctly identifies 4 boundary halfedges."""
    verts, tris = create_open_quad()
    mesh = HalfedgeMesh.from_indexed_triangles(verts, tris)

    assert mesh.is_watertight() is False
    # Diagonal edge (0, 2) is shared; the 4 perimeter edges are open boundaries
    boundaries = mesh.get_boundary_halfedges()
    assert len(boundaries) == 4
    # Total edges: 1 shared paired edge + 4 boundary edges = 5
    # Euler characteristic for disk topology with 1 boundary loop: V - E + F = 4 - 5 + 2 = 1
    assert mesh.compute_euler_characteristic() == 1


def test_vertex_1ring_neighborhood_traversal():
    """Verify 1-ring neighbors around apex of tetrahedron includes all other 3 vertices."""
    verts, tris = create_tetrahedron()
    mesh = HalfedgeMesh.from_indexed_triangles(verts, tris)

    ring0 = mesh.get_vertex_1ring(0)
    assert len(ring0) == 3
    assert set(ring0) == {1, 2, 3}
