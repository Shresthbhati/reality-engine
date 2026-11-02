"""Mesh cleaning, repair, and simplification (Master Development Plan Phase 4, Task 4.7).

Uses trimesh (MIT licensed) to clean, repair, and optionally simplify reconstructed
meshes (e.g. from Poisson or dense depth surface reconstruction):
- Merges duplicate vertices
- Removes unreferenced/isolated vertices
- Removes degenerate (zero-area or collinear) faces
- Fixes face winding and normal consistency
- Optionally fills boundary holes
- Optionally simplifies quadric decimation to target face budget
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh

from reconstruction.meshing.mesh import MeshData, Tri, Vert


def mesh_to_trimesh(mesh: MeshData) -> trimesh.Trimesh:
    """Convert MeshData into a trimesh.Trimesh object."""
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64) if mesh.faces else np.empty((0, 3), dtype=np.int64)
    normals = np.array(mesh.normals, dtype=np.float64) if mesh.normals else None
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)


def trimesh_to_mesh(
    tm: trimesh.Trimesh,
    units: str = "meter",
    frame: str = "world",
) -> MeshData:
    """Convert a trimesh.Trimesh back to MeshData."""
    vertices: Tuple[Vert, ...] = tuple(
        (float(v[0]), float(v[1]), float(v[2])) for v in tm.vertices
    )
    faces: Tuple[Tri, ...] = tuple(
        (int(f[0]), int(f[1]), int(f[2])) for f in tm.faces
    )
    normals: Optional[Tuple[Vert, ...]] = None
    if tm.vertex_normals is not None and len(tm.vertex_normals) == len(tm.vertices):
        normals = tuple(
            (float(n[0]), float(n[1]), float(n[2])) for n in tm.vertex_normals
        )
    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        units=units,
        frame=frame,
    )


def clean_and_repair_mesh(
    mesh: MeshData,
    fill_holes: bool = False,
    max_faces: Optional[int] = None,
) -> Tuple[MeshData, Dict[str, Any]]:
    """Clean, repair, and optionally simplify a MeshData.

    Returns (cleaned_mesh, stats).
    Preserves units and coordinate frame.
    """
    initial_vertices = len(mesh.vertices)
    initial_faces = len(mesh.faces)

    if initial_vertices == 0 or initial_faces == 0:
        return mesh, {
            "initial_vertices": initial_vertices,
            "initial_faces": initial_faces,
            "final_vertices": initial_vertices,
            "final_faces": initial_faces,
            "watertight": False,
            "holes_filled": 0,
            "simplified": False,
        }

    tm = mesh_to_trimesh(mesh)

    # 1. Clean topology: merge vertices, remove degenerate faces, unreferenced vertices
    tm.process(validate=True)
    trimesh.repair.fix_normals(tm)
    trimesh.repair.fix_winding(tm)

    holes_filled = 0
    if fill_holes:
        try:
            holes_filled = int(trimesh.repair.fill_holes(tm) or 0)
        except Exception:
            pass

    simplified = False
    if max_faces is not None and len(tm.faces) > max_faces:
        try:
            # trimesh simplify via quadric decimation or face sampling
            simplified_mesh = tm.simplify_quadric_decimation(face_count=max_faces)
            if len(simplified_mesh.faces) > 0:
                tm = simplified_mesh
                simplified = True
        except Exception:
            pass

    final_mesh = trimesh_to_mesh(tm, units=mesh.units, frame=mesh.frame)

    stats: Dict[str, Any] = {
        "initial_vertices": initial_vertices,
        "initial_faces": initial_faces,
        "final_vertices": len(final_mesh.vertices),
        "final_faces": len(final_mesh.faces),
        "watertight": tm.is_watertight,
        "holes_filled": holes_filled,
        "simplified": simplified,
    }

    return final_mesh, stats
