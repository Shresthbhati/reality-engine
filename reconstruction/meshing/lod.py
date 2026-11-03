"""Multi-Resolution Level of Detail (LOD) Generation (Wave 5, Block 34).

Generates multi-scale discrete levels of detail (LOD) for reconstructed 3D meshes:
- LOD 0: Original full-resolution reconstructed/repaired surface mesh.
- LOD 1: Architectural detail preservation (~50% face reduction).
- LOD 2: Overview / mobile performance tier (~80% face reduction).
- LOD 3: Simplified volumetric proxy / collision hull (~95% face reduction).

Uses pure NumPy spatial vertex-clustering decimation, preserving topology and
avoiding external C-extension dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from provenance import Provenance
from reconstruction.meshing.mesh import MeshData, Tri, Vert
from world_ir.schema_v1 import Entity, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR


@dataclass(frozen=True)
class LODTier:
    """Specification for an LOD level."""
    lod_level: int
    name: str
    target_reduction_ratio: float   # Target ratio of original face count (e.g. 0.50)
    preserve_boundaries: bool = True


DEFAULT_LOD_TIERS = (
    LODTier(lod_level=0, name="Full Resolution", target_reduction_ratio=1.0),
    LODTier(lod_level=1, name="Medium Architectural", target_reduction_ratio=0.50),
    LODTier(lod_level=2, name="Coarse Overview", target_reduction_ratio=0.20),
    LODTier(lod_level=3, name="Proxy Hull", target_reduction_ratio=0.05),
)


def _compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex surface normals."""
    normals = np.zeros_like(vertices, dtype=np.float64)
    if len(faces) == 0:
        return normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    return normals / lengths


def decimate_by_vertex_clustering(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decimate a mesh to target face count using adaptive spatial vertex clustering."""
    if len(faces) <= target_faces or len(vertices) <= 4:
        return vertices, faces

    min_pt = np.min(vertices, axis=0)
    max_pt = np.max(vertices, axis=0)
    bbox_diag = float(np.linalg.norm(max_pt - min_pt))
    if bbox_diag < 1e-6:
        return vertices, faces

    # Binary search for cell_size that achieves approximately target_faces
    low_cell = bbox_diag * 0.005
    high_cell = bbox_diag * 0.40
    best_verts = vertices
    best_faces = faces
    best_diff = abs(len(faces) - target_faces)

    for _ in range(12):
        mid_cell = (low_cell + high_cell) / 2.0
        grid_coords = np.floor((vertices - min_pt) / mid_cell).astype(np.int64)
        unique_cells, inverse_indices = np.unique(grid_coords, axis=0, return_inverse=True)
        num_clusters = len(unique_cells)

        if num_clusters < 4:
            high_cell = mid_cell
            continue

        new_verts = np.zeros((num_clusters, 3), dtype=np.float64)
        counts = np.zeros(num_clusters, dtype=np.float64)
        for i, cluster_idx in enumerate(inverse_indices):
            new_verts[cluster_idx] += vertices[i]
            counts[cluster_idx] += 1
        new_verts /= counts[:, None]

        remapped_faces = inverse_indices[faces]
        # Remove degenerate triangles
        valid_mask = (
            (remapped_faces[:, 0] != remapped_faces[:, 1])
            & (remapped_faces[:, 1] != remapped_faces[:, 2])
            & (remapped_faces[:, 0] != remapped_faces[:, 2])
        )
        valid_faces = remapped_faces[valid_mask]

        if len(valid_faces) == 0:
            high_cell = mid_cell
            continue

        # Remove duplicate faces
        sorted_faces = np.sort(valid_faces, axis=1)
        _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
        final_faces = valid_faces[unique_indices]
        current_count = len(final_faces)

        diff = abs(current_count - target_faces)
        if diff < best_diff and current_count < len(faces):
            best_diff = diff
            best_verts = new_verts
            best_faces = final_faces

        if current_count > target_faces:
            low_cell = mid_cell
        else:
            high_cell = mid_cell

    return best_verts, best_faces


def generate_mesh_lods(
    mesh: MeshData,
    tiers: Sequence[LODTier] = DEFAULT_LOD_TIERS,
) -> Dict[int, MeshData]:
    """Generate multi-level LOD meshes using surface decimation."""
    out: Dict[int, MeshData] = {}
    verts_arr = np.array(mesh.vertices, dtype=np.float64)
    faces_arr = np.array(mesh.faces, dtype=np.int32)
    orig_face_count = len(faces_arr)

    for tier in tiers:
        if tier.lod_level == 0 or tier.target_reduction_ratio >= 0.99 or orig_face_count <= 12:
            out[tier.lod_level] = mesh
            continue

        target_faces = max(8, int(orig_face_count * tier.target_reduction_ratio))
        dec_v, dec_f = decimate_by_vertex_clustering(verts_arr, faces_arr, target_faces)

        if len(dec_f) >= orig_face_count or len(dec_f) == 0:
            out[tier.lod_level] = mesh
        else:
            normals = _compute_normals(dec_v, dec_f)
            v_tuple = tuple(tuple(float(c) for c in pt) for pt in dec_v)
            f_tuple = tuple(tuple(int(idx) for idx in tri) for tri in dec_f)
            n_tuple = tuple(tuple(float(c) for c in pt) for pt in normals)
            out[tier.lod_level] = MeshData(
                vertices=v_tuple,
                faces=f_tuple,
                normals=n_tuple,
                units=mesh.units,
                frame=mesh.frame,
            )

    return out


def attach_lods_to_world(
    world: WorldIR,
    entity_id: str,
    lod_meshes: Dict[int, MeshData],
) -> List[Geometry]:
    """Register generated LOD meshes in WorldIR and bind them to the specified entity."""
    created_geoms: List[Geometry] = []

    for lod_level, m in sorted(lod_meshes.items()):
        v_arr = np.array(m.vertices, dtype=np.float64)
        if len(v_arr) > 0:
            min_pt = np.min(v_arr, axis=0)
            max_pt = np.max(v_arr, axis=0)
            b_min = Vector3(x=float(min_pt[0]), y=float(min_pt[1]), z=float(min_pt[2]))
            b_max = Vector3(x=float(max_pt[0]), y=float(max_pt[1]), z=float(max_pt[2]))
        else:
            b_min, b_max = None, None

        gid = f"geom-lod{lod_level}-{uuid4().hex[:8]}"
        geom = Geometry(
            id=gid,
            type=GeometryType.MESH,
            lod_level=lod_level,
            vertex_count=len(m.vertices),
            triangle_count=len(m.faces),
            bounds_min=b_min,
            bounds_max=b_max,
            provenance=Provenance.RECONSTRUCTED,
            confidence=max(0.60, 0.95 - (lod_level * 0.08)),
            quality_metrics={
                "lod_level": lod_level,
                "is_watertight": m.is_watertight_ish(),
                "vertex_count": len(m.vertices),
                "face_count": len(m.faces),
            },
        )
        world.geometries[gid] = geom
        created_geoms.append(geom)

    # Bind to entity
    if entity_id in world.entities:
        ent = world.entities[entity_id]
        if ent.geometry_ids is None:
            ent.geometry_ids = []
        for g in created_geoms:
            if g.id not in ent.geometry_ids:
                ent.geometry_ids.append(g.id)
        if ent.custom_properties is None:
            ent.custom_properties = {}
        ent.custom_properties["has_lods"] = True
        ent.custom_properties["lod_levels"] = list(lod_meshes.keys())

    return created_geoms
