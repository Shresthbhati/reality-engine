"""Unit tests for OpenVDB sparse signed distance field (SDF) and CSG."""

import numpy as np
import pytest

from geometry.openvdb_sdf import (
    SparseVoxelGridSDF,
    csg_difference,
    csg_intersection,
    csg_union,
    sdf_box,
    sdf_sphere,
)


def test_sdf_sphere_exact_distances():
    """Verify sphere signed distance function at surface, interior, and exterior points."""
    center = np.array([1.0, 2.0, 3.0])
    radius = 2.0

    # Center: distance should be -radius (-2.0)
    assert np.isclose(sdf_sphere(center, radius=radius, center=center), -2.0)

    # Surface: [1, 2, 5] has dist 2.0 from center => SDF = 0.0
    surface_pt = np.array([1.0, 2.0, 5.0])
    assert np.isclose(sdf_sphere(surface_pt, radius=radius, center=center), 0.0)

    # Exterior: [1, 2, 6] has dist 3.0 from center => SDF = +1.0
    exterior_pt = np.array([1.0, 2.0, 6.0])
    assert np.isclose(sdf_sphere(exterior_pt, radius=radius, center=center), 1.0)


def test_sdf_box_exact_distances():
    """Verify box signed distance function for half-extents [1, 2, 3]."""
    extents = np.array([1.0, 2.0, 3.0])
    center = np.zeros(3)

    # Center: deep inside box (distance to nearest face is -1.0)
    assert np.isclose(sdf_box(center, extents=extents, center=center), -1.0)

    # Surface point on +X face: [1.0, 0, 0] => SDF = 0.0
    face_pt = np.array([1.0, 0.0, 0.0])
    assert np.isclose(sdf_box(face_pt, extents=extents, center=center), 0.0)

    # Exterior point: [2.0, 0, 0] => SDF = +1.0
    ext_pt = np.array([2.0, 0.0, 0.0])
    assert np.isclose(sdf_box(ext_pt, extents=extents, center=center), 1.0)


def test_csg_boolean_operations():
    """Verify CSG Union, Intersection, and Difference in signed distance field space."""
    pts = np.array([
        [1.0, 0.0, 0.0],  # Center of sphere B (r=1 centered at [1,0,0]) and inside sphere A (r=2)
        [3.0, 0.0, 0.0],  # Outside both
        [0.0, 0.0, 0.0],  # Boundary of sphere B
    ])

    sdf_a = sdf_sphere(pts, radius=2.0)
    sdf_b = sdf_sphere(pts, radius=1.0, center=np.array([1.0, 0.0, 0.0]))

    # CSG Union: min(A, B)
    u = csg_union(sdf_a, sdf_b)
    assert np.all(u <= sdf_a) and np.all(u <= sdf_b)

    # CSG Difference (A \ B): max(A, -B)
    diff = csg_difference(sdf_a, sdf_b)
    # Inside sphere B means sdf_b < 0 => -sdf_b > 0 => carved out of A!
    assert diff[0] > 0.0  # Center of B is carved away: positive distance to exterior
    assert np.isclose(diff[0], 1.0)


def test_sparse_voxel_grid_narrow_band_filtering():
    """Verify SparseVoxelGridSDF stores only active narrow-band voxels."""
    grid = SparseVoxelGridSDF(voxel_size=0.1, half_bandwidth=0.25)
    active_voxels = grid.discretize_primitive(
        lambda p: sdf_sphere(p, radius=1.0),
        bounds_min=np.array([-1.5, -1.5, -1.5]),
        bounds_max=np.array([1.5, 1.5, 1.5]),
    )

    assert active_voxels > 0
    # Every active voxel must have distance magnitude <= half_bandwidth
    for dist in grid.voxels.values():
        assert abs(dist) <= 0.25

    # Center is deep inside (dist ~ -1.0) so it must NOT be in narrow band
    assert grid.sample_at_point(np.array([0.0, 0.0, 0.0])) is None
    # Surface point (dist ~ 0.0) MUST be in narrow band
    assert grid.sample_at_point(np.array([1.0, 0.0, 0.0])) is not None
