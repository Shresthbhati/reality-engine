"""Surface reconstruction and mesh processing."""

from .lod import DEFAULT_LOD_TIERS, LODTier, attach_lods_to_world, generate_mesh_lods
from .mesh import MeshData, mesh_summary
from .repair import clean_and_repair_mesh, mesh_to_trimesh, trimesh_to_mesh
from .room_mesh import (
    generate_prismatic_room_mesh,
    is_non_convex,
    room_mesh_surface_area,
    room_mesh_volume,
)

__all__ = [
    "MeshData",
    "mesh_summary",
    "clean_and_repair_mesh",
    "mesh_to_trimesh",
    "trimesh_to_mesh",
    "LODTier",
    "DEFAULT_LOD_TIERS",
    "generate_mesh_lods",
    "attach_lods_to_world",
    "generate_prismatic_room_mesh",
    "is_non_convex",
    "room_mesh_surface_area",
    "room_mesh_volume",
]
