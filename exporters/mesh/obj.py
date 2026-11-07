"""Wavefront OBJ and MTL mesh exporter for WorldIR (Task 9.3).

Exports real reconstructed meshes (MeshData) or bounding geometries (BOX/PLANE)
with materials, preserving entity hierarchy and naming.
"""

from __future__ import annotations

import math
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


def export_to_obj(
    world: WorldIR,
    artifact_store: Optional[ArtifactStore] = None,
    mtl_filename: str = "world.mtl",
) -> Tuple[str, str]:
    """Export world to Wavefront (obj_string, mtl_string)."""
    obj_lines = [
        "# Reality Engine Wavefront OBJ Export",
        f"# World: {world.name or world.id} (version {world.version})",
        f"mtllib {mtl_filename}",
        "",
    ]
    mtl_lines = [
        "# Reality Engine Material Library",
        f"# World: {world.name or world.id}",
        "",
    ]

    materials_defined = set()
    current_vert_offset = 1

    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        pos = (entity.transform or {}).get("position") if isinstance(entity.transform, dict) else None
        if not pos:
            continue
        px, py, pz = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)

        # 1. Check for real MESH
        verts = None
        faces = None
        for gid in entity.geometry_ids:
            g = world.geometries.get(gid)
            if g and g.type is GeometryType.MESH:
                real = _resolve_real_mesh(artifact_store, g)
                if real:
                    verts, faces = real
                    break

        # 2. Fall back to BOX / PLANE bounds
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

        if verts is None or faces is None or not verts or not faces:
            continue

        # Material resolution
        mat_id = getattr(entity, "material_id", None)
        if not mat_id and hasattr(entity, "custom_properties") and isinstance(entity.custom_properties, dict):
            mat_id = entity.custom_properties.get("material_id") or entity.custom_properties.get("material")

        mat_name = "default_material"
        color = (0.8, 0.8, 0.8)
        if mat_id and hasattr(world, "materials") and mat_id in world.materials:
            m = world.materials[mat_id]
            mat_name = m.name or m.id
            if m.color_rgb:
                rgb = m.color_rgb
                if any(c > 1.0 for c in rgb):
                    rgb = [c / 255.0 for c in rgb]
                color = tuple(rgb[:3])
        elif isinstance(entity.custom_properties, dict):
            app = entity.custom_properties.get("appearance")
            if isinstance(app, dict) and "color_rgb" in app:
                rgb = app["color_rgb"]
                if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
                    mat_name = f"mat_{entity.id}"
                    c_rgb = [float(c) for c in rgb[:3]]
                    if any(c > 1.0 for c in c_rgb):
                        c_rgb = [c / 255.0 for c in c_rgb]
                    color = tuple(c_rgb)

        if mat_name == "default_material":  # no authored material: name it by surface type (same defaults as the glTF export)
            from exporters.gltf.exporter import _SURFACE_DEFAULTS

            kind = getattr(entity.type, "value", None)
            if kind in _SURFACE_DEFAULTS:
                mat_name, color = f"surface_{kind}", tuple(_SURFACE_DEFAULTS[kind][0])

        if mat_name not in materials_defined:
            materials_defined.add(mat_name)
            mtl_lines.extend([
                f"newmtl {mat_name}",
                f"Ka {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}",
                f"Kd {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}",
                "Ks 0.2000 0.2000 0.2000",
                "Ns 32.0",
                "d 1.0",
                "illum 2",
                "",
            ])

        obj_lines.append(f"o {entity.name or entity.id}")
        obj_lines.append(f"g {entity.type.value if hasattr(entity.type, 'value') else str(entity.type)}")
        obj_lines.append(f"usemtl {mat_name}")

        for v in verts:
            obj_lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")

        for f in faces:
            obj_lines.append(f"f {f[0] + current_vert_offset} {f[1] + current_vert_offset} {f[2] + current_vert_offset}")

        current_vert_offset += len(verts)
        obj_lines.append("")

    return "\n".join(obj_lines), "\n".join(mtl_lines)


def export_to_obj_with_report(
    world: WorldIR, artifact_store: Optional[ArtifactStore] = None
) -> Tuple[str, ExportReport]:
    """Export to OBJ text and return structured report."""
    obj_str, _ = export_to_obj(world, artifact_store)
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

        has_geom = False
        for gid in entity.geometry_ids:
            g = world.geometries.get(gid)
            if g and (g.type is GeometryType.MESH or (g.type in _EXPORTABLE and g.bounds_min and g.bounds_max)):
                has_geom = True
                break

        if has_geom:
            exported.append(entity_id)
        else:
            skipped.append(entity_id)
            reasons.append("no exportable MESH, BOX, or PLANE geometry")

    report = ExportReport(
        format="obj",
        world_id=world.id,
        world_version=world.version,
        entities_exported=tuple(exported),
        entities_skipped=tuple(skipped),
        skip_reasons=tuple(reasons),
        content_hash=content_hash(obj_str.encode("utf-8")),
    )
    return obj_str, report
