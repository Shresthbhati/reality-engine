"""Unit tests for meshoptimizer algorithmic extraction."""

import numpy as np
import pytest

from reconstruction.meshing.meshopt_simplifier import (
    QuadricMeshSimplifier,
    compute_acmr,
    optimize_mesh,
    optimize_vertex_cache,
    optimize_vertex_fetch,
)


def create_plane_mesh(subdivisions: int = 4):
    """Create a subdivided planar triangle grid."""
    xs = np.linspace(0.0, 1.0, subdivisions + 1)
    ys = np.linspace(0.0, 1.0, subdivisions + 1)
    verts = []
    for y in ys:
        for x in xs:
            verts.append([x, y, 0.0])
    verts = np.array(verts, dtype=np.float64)

    tris = []
    stride = subdivisions + 1
    for r in range(subdivisions):
        for c in range(subdivisions):
            i0 = r * stride + c
            i1 = i0 + 1
            i2 = (r + 1) * stride + c
            i3 = i2 + 1
            tris.append([i0, i2, i1])
            tris.append([i1, i2, i3])
    tris = np.array(tris, dtype=np.int64)
    return verts, tris


def test_vertex_cache_optimization_reduces_or_maintains_acmr():
    """Verify that vertex cache optimization improves cache hit locality (ACMR)."""
    verts, tris = create_plane_mesh(subdivisions=6)
    # Shuffle triangles to simulate poor index locality
    rng = np.random.default_rng(42)
    shuffled_tris = rng.permutation(tris)

    acmr_unoptimized = compute_acmr(shuffled_tris, cache_size=16)
    reordered_tris = optimize_vertex_cache(shuffled_tris, cache_size=16)
    acmr_optimized = compute_acmr(reordered_tris, cache_size=16)

    assert len(reordered_tris) == len(shuffled_tris)
    assert acmr_optimized <= acmr_unoptimized
    assert acmr_optimized < 1.4  # Well optimized planar grid


def test_quadric_edge_collapse_simplifier():
    """Verify Garland-Heckbert quadric edge collapses reduce face count without degeneration."""
    verts, tris = create_plane_mesh(subdivisions=6)
    orig_face_count = len(tris)
    target_face_count = orig_face_count // 2

    simplifier = QuadricMeshSimplifier(verts, tris)
    simp_verts, simp_tris = simplifier.simplify(target_face_count)

    assert len(simp_tris) <= orig_face_count
    assert len(simp_tris) <= target_face_count + 6  # Approximation within tolerance
    assert len(simp_verts) < len(verts)

    # Check non-degenerate triangles
    for tri in simp_tris:
        assert len(set(tri)) == 3


def test_full_meshopt_pipeline():
    """Verify complete optimize_mesh helper executes decimation, cache, and fetch reordering."""
    verts, tris = create_plane_mesh(subdivisions=5)
    out_verts, out_tris, stats = optimize_mesh(
        verts,
        tris,
        target_reduction_ratio=0.6,
        optimize_cache=True,
        optimize_fetch=True,
    )

    assert stats.optimized_triangles < stats.original_triangles
    assert stats.optimized_vertices <= stats.original_vertices
    assert stats.reduction_ratio <= 0.7
    assert len(out_verts) == stats.optimized_vertices
    assert len(out_tris) == stats.optimized_triangles
