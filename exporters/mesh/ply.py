"""Polygon File Format (PLY) mesh exporter for WorldIR (Task 9.3).

Exports vertices, faces, and vertex colors in binary or ASCII PLY format.
"""

from __future__ import annotations

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
    from world_ir.geometry_data import MeshData
    try:
        mesh = MeshData.from_bytes(payload)
        return list(mesh.vertices), list(mesh.faces)
    except Exception:
        return None


def export_to_ply(
    world: WorldIR,
    artifact_store: Optional[ArtifactStore] = None,
    binary: bool = True,
) -> bytes:
    """Export world to PLY format (binary or ASCII)."""
    all_vertices: List[Tuple[float, float, float]] = []
    all_colors: List[Tuple[int, int, int]] = []
    all_faces: List[Tuple[int, int, int]] = []

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

        # Color
        color = (200, 200, 200)
        mat_id = getattr(entity, "material_id", None)
        if not mat_id and hasattr(entity, "custom_properties") and isinstance(entity.custom_properties, dict):
            mat_id = entity.custom_properties.get("material_id") or entity.custom_properties.get("material")

        if mat_id and hasattr(world, "materials") and mat_id in world.materials:
            m = world.materials[mat_id]
            if m.color_rgb:
                rgb = m.color_rgb
                if all(c <= 1.0 for c in rgb):
                    color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                else:
                    color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        elif isinstance(entity.custom_properties, dict):
            app = entity.custom_properties.get("appearance")
            if isinstance(app, dict) and "color_rgb" in app:
                rgb = app["color_rgb"]
                if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
                    if all(c <= 1.0 for c in rgb[:3]):
                        color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                    else:
                        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

        base_idx = len(all_vertices)
        all_vertices.extend(verts)
        all_colors.extend([color] * len(verts))
        for f in faces:
            all_faces.append((f[0] + base_idx, f[1] + base_idx, f[2] + base_idx))

    if binary:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"comment Reality Engine World {world.id}\n"
            f"element vertex {len(all_vertices)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            f"element face {len(all_faces)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")

        body_parts = []
        for v, c in zip(all_vertices, all_colors):
            body_parts.append(struct.pack("<fffBBB", v[0], v[1], v[2], c[0], c[1], c[2]))
        for f in all_faces:
            body_parts.append(struct.pack("<Biii", 3, f[0], f[1], f[2]))

        return header + b"".join(body_parts)
    else:
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"comment Reality Engine World {world.id}\n"
            f"element vertex {len(all_vertices)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            f"element face {len(all_faces)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        lines = [header]
        for v, c in zip(all_vertices, all_colors):
            lines.append(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for f in all_faces:
            lines.append(f"3 {f[0]} {f[1]} {f[2]}\n")
        return "".join(lines).encode("utf-8")


def export_to_ply_with_report(
    world: WorldIR, artifact_store: Optional[ArtifactStore] = None
) -> Tuple[bytes, ExportReport]:
    """Export to PLY bytes and return structured report."""
    ply_bytes = export_to_ply(world, artifact_store, binary=True)
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
        format="ply",
        world_id=world.id,
        world_version=world.version,
        entities_exported=tuple(exported),
        entities_skipped=tuple(skipped),
        skip_reasons=tuple(reasons),
        content_hash=content_hash(ply_bytes),
    )
    return ply_bytes, report
