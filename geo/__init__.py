"""Geographic context for worlds: real map data (OpenStreetMap) and terrain elevation mesh surfaces."""

from geo.terrain import (
    TerrainElevationGrid,
    TerrainMesh,
    compute_slope_and_aspect,
    compute_vertex_normals,
    fuse_terrain_into_world,
    generate_terrain_mesh_from_grid,
    generate_terrain_mesh_from_point_cloud,
)

__all__ = [
    "TerrainElevationGrid",
    "TerrainMesh",
    "compute_vertex_normals",
    "compute_slope_and_aspect",
    "generate_terrain_mesh_from_grid",
    "generate_terrain_mesh_from_point_cloud",
    "fuse_terrain_into_world",
]
