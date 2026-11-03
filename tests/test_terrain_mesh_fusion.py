"""Tests for Large-Scale Geospatial Terrain Elevation & Surface Mesh Fusion."""

from __future__ import annotations

import math
import numpy as np
import pytest

from geo.terrain import (
    TerrainElevationGrid,
    TerrainMesh,
    compute_slope_and_aspect,
    compute_vertex_normals,
    fuse_terrain_into_world,
    generate_terrain_mesh_from_grid,
    generate_terrain_mesh_from_point_cloud,
)
from world_ir.schema_v1 import EntityType, GeometryType
from world_ir.world_v1 import WorldIR


def test_terrain_grid_mesh_generation_with_watertight_skirt():
    # 5x5 elevation grid with a gentle hill
    x = np.linspace(0, 40, 5)
    z = np.linspace(0, 40, 5)
    grid_x, grid_z = np.meshgrid(x, z)
    elevation = 10.0 + 5.0 * np.sin(grid_x / 10.0) * np.cos(grid_z / 10.0)

    grid = TerrainElevationGrid(
        elevation=elevation,
        bounds=(0.0, 0.0, 40.0, 40.0),
        resolution_m=10.0,
        crs="EPSG:4326",
    )

    mesh = generate_terrain_mesh_from_grid(grid, add_skirt=True, skirt_depth_m=10.0)

    assert isinstance(mesh, TerrainMesh)
    assert mesh.is_watertight is True
    # Top vertices: 5*5 = 25. Perimeter vertices: 4*(5-1) = 16. Center base vertex: 1. Total: 25+16+1 = 42
    assert len(mesh.vertices) == 42
    # Faces: Top faces = 4*4*2 = 32. Skirt quad faces = 16*2 = 32. Base fan faces = 16. Total = 80
    assert len(mesh.faces) == 80
    assert mesh.normals.shape == (42, 3)
    assert len(mesh.slope_degrees) == 42
    assert len(mesh.aspect_degrees) == 42

    # Verify bounding box
    min_pt, max_pt = mesh.bounding_box
    assert min_pt[0] == pytest.approx(0.0)
    assert min_pt[2] == pytest.approx(0.0)
    assert max_pt[0] == pytest.approx(40.0)
    assert max_pt[2] == pytest.approx(40.0)
    # Base elevation is lower than min elevation by 10m skirt
    assert min_pt[1] < np.min(elevation)


def test_terrain_grid_mesh_without_skirt():
    elevation = np.array([
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ])
    grid = TerrainElevationGrid(
        elevation=elevation,
        bounds=(-10.0, -10.0, 10.0, 10.0),
        resolution_m=10.0,
    )
    mesh = generate_terrain_mesh_from_grid(grid, add_skirt=False)
    assert mesh.is_watertight is False
    assert len(mesh.vertices) == 9
    assert len(mesh.faces) == 2 * 2 * 2  # 8 triangles


def test_slope_and_aspect_calculation():
    # Surface sloping upwards towards East (x increases with height)
    # Normal tilted towards West: nx = -1, ny = 1, nz = 0
    norm = np.array([[-1.0, 1.0, 0.0]])
    norm = norm / np.linalg.norm(norm)
    slope, aspect = compute_slope_and_aspect(norm)

    assert slope[0] == pytest.approx(45.0, abs=1.0)
    # Tilted towards West: aspect should be ~270 degrees
    assert aspect[0] == pytest.approx(270.0, abs=2.0)


def test_terrain_from_point_cloud():
    # Generate 200 random points covering 20m x 20m with elevation
    np.random.seed(42)
    x = np.random.uniform(0.0, 20.0, 200)
    z = np.random.uniform(0.0, 20.0, 200)
    y = 5.0 + 0.1 * x - 0.05 * z + np.random.normal(0, 0.1, 200)
    points = np.stack([x, y, z], axis=-1)

    mesh = generate_terrain_mesh_from_point_cloud(points, cell_size_m=5.0, add_skirt=True)
    assert mesh.is_watertight is True
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0

    mesh_data = mesh.to_mesh_data()
    assert mesh_data.vertices.shape[0] == len(mesh.vertices)
    assert mesh_data.faces.shape[0] == len(mesh.faces)
    assert mesh_data.units == "meter"


def test_fuse_terrain_into_world():
    world = WorldIR(id="test-outdoor-site")
    elevation = np.array([
        [100.0, 101.0],
        [100.5, 102.0],
    ])
    grid = TerrainElevationGrid(
        elevation=elevation,
        bounds=(10.0, 20.0, 30.0, 40.0),
        resolution_m=20.0,
    )
    mesh = generate_terrain_mesh_from_grid(grid, add_skirt=True)

    entity = fuse_terrain_into_world(
        world=world,
        terrain_mesh=mesh,
        entity_id="site-terrain-01",
        name="Campus Terrain Surface",
        geo_anchor=(37.7749, -122.4194),
    )

    assert entity.id == "site-terrain-01"
    assert entity.type == EntityType.TERRAIN
    assert len(entity.geometry_ids) == 1
    geom_id = entity.geometry_ids[0]
    assert geom_id in world.geometries
    geom = world.geometries[geom_id]
    assert geom.type == GeometryType.MESH
    assert geom.vertex_count == len(mesh.vertices)
    assert geom.triangle_count == len(mesh.faces)
    assert entity.custom_properties["is_terrain"] is True
    assert entity.custom_properties["is_watertight"] is True
    assert "site-terrain-01" in world.entities

    # Verify world metadata
    assert "terrain" in world.metadata
    assert world.metadata["terrain"]["entity_id"] == "site-terrain-01"
    assert world.metadata["geo"]["anchor"]["lat"] == pytest.approx(37.7749)
    assert world.metadata["geo"]["anchor"]["lon"] == pytest.approx(-122.4194)
