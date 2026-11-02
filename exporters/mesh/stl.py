"""Stereolithography (STL) mesh exporter for WorldIR (Task 9.3).

Exports reconstructed 3D meshes and bounding volumes in binary or ASCII STL format.
"""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, List, Optional, Tuple

from exporters.boxes import _EXPORTABLE, MIN_DIMENSION_M
from exporters.report import ExportReport, content_hash
from world_ir.artifact_store import ArtifactNotFoundError, ArtifactStore
from world_ir.schema_v1 import GeometryType

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR

_CUBE_FACES = (
    (0, 2, 1), (0, 3, 2),  # bottom (-Z)
    (4, 5, 6), (4, 6, 7),  # top (+Z)
    (0, 1, 5), (0, 5, 4),  # front (-Y)
    (2, 3, 7), (2, 7, 6),  # back (+Y)
    (0, 4, 7), (0, 7, 3),  # left (-X)
    (1, 2, 6), (1, 6, 5),  # right (+X)
)


def _box_vertices(bmin: Tuple[float, float, float], bmax: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    return [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]


def _resolve_real_mesh(artifact_store: Optional[ArtifactStore], geometry) -> Optional[Tuple[list, list]]:
    if artifact_store is None or not getattr(geometry, "data_uri", None):
        return None
    if getattr(geometry, "type", None) is not GeometryType.MESH:
        return None
    try:
        payload = artifact_store.get(geometry.data_uri)
    except ArtifactNotFoundError:
        return None
    from reconstruction.meshing.mesh import MeshData
    try:
        mesh = MeshData.from_bytes(payload)
        return list(mesh.vertices), list(mesh.faces)
    except Exception:
        return None


def _compute_normal(v0: Tuple[float, float, float], v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
    bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-9:
        return (0.0, 0.0, 1.0)
    return (nx / norm, ny / norm, nz / norm)


def export_to_stl(
    world: WorldIR,
    artifact_store: Optional[ArtifactStore] = None,
    binary: bool = True,
) -> bytes:
    """Export world to binary or ASCII STL."""
    triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = []

    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        pos = (entity.transform or {}).get("position") if isinstance(entity.transform, dict) else None
        if not pos:
            continue
        px, py, pz = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)

        # 1. Resolve mesh
        verts = None
        faces = None
        for gid in entity.geometry_ids:
            g = world.geometries.get(gid)
            if g and g.type is GeometryType.MESH:
                real = _resolve_real_mesh(artifact_store, g)
                if real:
                    verts, faces = real
                    break

        # 2. Fall back to bounds
        if verts is None or faces is None:
            geom = next((world.geometries[gid] for gid in entity.geometry_ids
                         if gid in world.geometries and world.geometries[gid].type in _EXPORTABLE), None)
            if geom and geom.bounds_min is not None and geom.bounds_max is not None:
                bmin = (
                    px + geom.bounds_min.x,
                    py + geom.bounds_min.y,
                    pz + geom.bounds_min.z,
                )
                bmax = (
                    px + max(geom.bounds_min.x + MIN_DIMENSION_M, geom.bounds_max.x),
                    py + max(geom.bounds_min.y + MIN_DIMENSION_M, geom.bounds_max.y),
                    pz + max(geom.bounds_min.z + MIN_DIMENSION_M, geom.bounds_max.z),
                )
                verts = _box_vertices(bmin, bmax)
                faces = list(_CUBE_FACES)

        if not verts or not faces:
            continue

        for f in faces:
            v0 = verts[f[0]]
            v1 = verts[f[1]]
            v2 = verts[f[2]]
            normal = _compute_normal(v0, v1, v2)
            triangles.append((normal, v0, v1, v2))

    if binary:
        header_text = f"Reality Engine STL Export - World {world.id}"[:80].encode("ascii")
        header = header_text + b"\x00" * (80 - len(header_text))
        count = struct.pack("<I", len(triangles))
        parts = [header, count]
        for normal, v0, v1, v2 in triangles:
            parts.append(struct.pack(
                "<3f3f3f3fH",
                normal[0], normal[1], normal[2],
                v0[0], v0[1], v0[2],
                v1[0], v1[1], v1[2],
                v2[0], v2[1], v2[2],
                0,
            ))
        return b"".join(parts)
    else:
        lines = [f"solid {world.name or world.id or 'reality_engine_world'}\n"]
        for normal, v0, v1, v2 in triangles:
            lines.append(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            lines.append("    outer loop\n")
            lines.append(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            lines.append(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            lines.append(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            lines.append("    endloop\n")
            lines.append("  endfacet\n")
        lines.append(f"endsolid {world.name or world.id or 'reality_engine_world'}\n")
        return "".join(lines).encode("utf-8")


def export_to_stl_with_report(
    world: WorldIR, artifact_store: Optional[ArtifactStore] = None
) -> Tuple[bytes, ExportReport]:
    """Export to STL bytes and return structured report."""
    stl_bytes = export_to_stl(world, artifact_store, binary=True)
    exported = []
    skipped = []
    reasons = []

    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        pos = (entity.transform or {}).get("position") if isinstance(entity.transform, dict) else None
        if not pos:
            skipped.append(entity_id)
            reasons.append("no transform position")
            continue

        has_geom = any(world.geometries.get(gid) and (
            world.geometries[gid].type is GeometryType.MESH or (
                world.geometries[gid].type in _EXPORTABLE and
                world.geometries[gid].bounds_min and
                world.geometries[gid].bounds_max
            )
        ) for gid in entity.geometry_ids)

        if has_geom:
            exported.append(entity_id)
        else:
            skipped.append(entity_id)
            reasons.append("no exportable MESH, BOX, or PLANE geometry")

    report = ExportReport(
        format="stl",
        world_id=world.id,
        world_version=world.version,
        entities_exported=tuple(exported),
        entities_skipped=tuple(skipped),
        skip_reasons=tuple(reasons),
        content_hash=content_hash(stl_bytes),
    )
    return stl_bytes, report
