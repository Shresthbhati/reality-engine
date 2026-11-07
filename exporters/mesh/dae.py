"""COLLADA (.dae) mesh exporter for WorldIR (Task T-2.4).

Plain mesh export for SketchUp, 3ds Max, Blender, and Revit workflows:
- Metric scale (<unit meter="1.0" name="meter"/>, up-axis Z_UP)
- Per-object nodes (<node name="..."> for every entity)
- Materials named by surface (surface_wall, surface_floor, etc. or authored materials)
- Real reconstructed meshes (MeshData) or bounding geometries (BOX/PLANE)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

from exporters.boxes import _EXPORTABLE, MIN_DIMENSION_M
from exporters.gltf.exporter import _SURFACE_DEFAULTS
from exporters.mesh.obj import _CUBE_FACES, _box_vertices, _resolve_real_mesh
from exporters.report import ExportReport, content_hash
from world_ir.artifact_store import ArtifactStore
from world_ir.schema_v1 import GeometryType

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR


def export_to_dae(
    world: "WorldIR",
    artifact_store: Optional[ArtifactStore] = None,
) -> str:
    """Export world to COLLADA 1.4.1 XML string."""
    root = ET.Element(
        "COLLADA",
        {
            "xmlns": "http://www.collada.org/2005/11/COLLADASchema",
            "version": "1.4.1",
        },
    )

    # 1. Asset block with metric scale and Z_UP
    asset = ET.SubElement(root, "asset")
    contrib = ET.SubElement(asset, "contributor")
    ET.SubElement(contrib, "authoring_tool").text = "Reality Engine COLLADA Exporter"
    ET.SubElement(asset, "unit", {"name": "meter", "meter": "1.0"})
    ET.SubElement(asset, "up_axis").text = "Z_UP"

    lib_effects = ET.SubElement(root, "library_effects")
    lib_materials = ET.SubElement(root, "library_materials")
    lib_geometries = ET.SubElement(root, "library_geometries")
    lib_scenes = ET.SubElement(root, "library_visual_scenes")
    visual_scene = ET.SubElement(lib_scenes, "visual_scene", {"id": "Scene", "name": "Scene"})

    defined_materials: Dict[str, Tuple[float, float, float, float]] = {}
    exported_entities: List[Tuple[str, str, str, List[Tuple[float, float, float]], List[Tuple[int, int, int]]]] = []

    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        pos = (entity.transform or {}).get("position") if isinstance(entity.transform, dict) else None
        if not pos:
            continue
        px, py, pz = float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))

        # Resolve geometry: real MESH or bounding box
        verts = None
        faces = None
        for gid in entity.geometry_ids:
            g = world.geometries.get(gid)
            if g and g.type is GeometryType.MESH:
                real = _resolve_real_mesh(artifact_store, g)
                if real:
                    verts, faces = real
                    break

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

        # Material resolution (authored or surface named)
        mat_id = getattr(entity, "material_id", None)
        if not mat_id and hasattr(entity, "custom_properties") and isinstance(entity.custom_properties, dict):
            mat_id = entity.custom_properties.get("material_id") or entity.custom_properties.get("material")

        mat_name = "default_material"
        color = (0.8, 0.8, 0.8, 1.0)
        if mat_id and hasattr(world, "materials") and mat_id in world.materials:
            m = world.materials[mat_id]
            mat_name = m.name or m.id
            if m.color_rgb:
                rgb = m.color_rgb
                if any(c > 1.0 for c in rgb):
                    rgb = [c / 255.0 for c in rgb]
                color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        elif isinstance(entity.custom_properties, dict):
            app = entity.custom_properties.get("appearance")
            if isinstance(app, dict) and "color_rgb" in app:
                rgb = app["color_rgb"]
                if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
                    mat_name = f"mat_{entity.id}"
                    c_rgb = [float(c) for c in rgb[:3]]
                    if any(c > 1.0 for c in c_rgb):
                        c_rgb = [c / 255.0 for c in c_rgb]
                    color = (c_rgb[0], c_rgb[1], c_rgb[2], 1.0)

        if mat_name == "default_material":
            kind = getattr(entity.type, "value", None)
            if kind in _SURFACE_DEFAULTS:
                rgb_def, _, alpha_def = _SURFACE_DEFAULTS[kind]
                mat_name = f"surface_{kind}"
                color = (float(rgb_def[0]), float(rgb_def[1]), float(rgb_def[2]), float(alpha_def))

        defined_materials[mat_name] = color
        exported_entities.append((entity.id, entity.name or entity.id, mat_name, verts, faces))

    # Build <library_effects> and <library_materials>
    for m_name, col in sorted(defined_materials.items()):
        eff_id = f"effect-{m_name}"
        eff = ET.SubElement(lib_effects, "effect", {"id": eff_id, "name": m_name})
        prof = ET.SubElement(eff, "profile_COMMON")
        tech = ET.SubElement(prof, "technique", {"sid": "common"})
        phong = ET.SubElement(tech, "phong")
        diffuse = ET.SubElement(phong, "diffuse")
        ET.SubElement(diffuse, "color").text = f"{col[0]:.4f} {col[1]:.4f} {col[2]:.4f} {col[3]:.4f}"

        mat_elem = ET.SubElement(lib_materials, "material", {"id": f"mat-{m_name}", "name": m_name})
        ET.SubElement(mat_elem, "instance_effect", {"url": f"#{eff_id}"})

    # Build <library_geometries> and per-object <node> in scene
    for ent_id, ent_name, mat_name, verts, faces in exported_entities:
        geom_id = f"geom-{ent_id}"
        geom = ET.SubElement(lib_geometries, "geometry", {"id": geom_id, "name": ent_name})
        mesh = ET.SubElement(geom, "mesh")

        # Position source
        pos_id = f"{geom_id}-positions"
        pos_array_id = f"{pos_id}-array"
        source = ET.SubElement(mesh, "source", {"id": pos_id})
        flat_verts = [f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" for v in verts]
        float_array = ET.SubElement(
            source,
            "float_array",
            {"id": pos_array_id, "count": str(len(verts) * 3)},
        )
        float_array.text = " ".join(flat_verts)

        tech_common = ET.SubElement(source, "technique_common")
        accessor = ET.SubElement(
            tech_common,
            "accessor",
            {"source": f"#{pos_array_id}", "count": str(len(verts)), "stride": "3"},
        )
        for axis in ("X", "Y", "Z"):
            ET.SubElement(accessor, "param", {"name": axis, "type": "float"})

        # Vertices element
        vertices_elem = ET.SubElement(mesh, "vertices", {"id": f"{geom_id}-vertices"})
        ET.SubElement(vertices_elem, "input", {"semantic": "POSITION", "source": f"#{pos_id}"})

        # Triangles
        triangles = ET.SubElement(
            mesh,
            "triangles",
            {"count": str(len(faces)), "material": mat_name},
        )
        ET.SubElement(triangles, "input", {"semantic": "VERTEX", "source": f"#{geom_id}-vertices", "offset": "0"})
        flat_indices = []
        for f in faces:
            flat_indices.extend([str(f[0]), str(f[1]), str(f[2])])
        p_elem = ET.SubElement(triangles, "p")
        p_elem.text = " ".join(flat_indices)

        # Per-object node in scene (T-2.4)
        node = ET.SubElement(visual_scene, "node", {"id": f"node-{ent_id}", "name": ent_name})
        inst_geom = ET.SubElement(node, "instance_geometry", {"url": f"#{geom_id}"})
        bind_mat = ET.SubElement(inst_geom, "bind_material")
        t_common = ET.SubElement(bind_mat, "technique_common")
        ET.SubElement(t_common, "instance_material", {"symbol": mat_name, "target": f"#mat-{mat_name}"})

    # Scene link
    scene = ET.SubElement(root, "scene")
    ET.SubElement(scene, "instance_visual_scene", {"url": "#Scene"})

    # Pretty-print indentation
    ET.indent(root, space="  ")
    xml_header = '<?xml version="1.0" encoding="utf-8"?>\n'
    return xml_header + ET.tostring(root, encoding="unicode")


def export_to_dae_with_report(
    world: "WorldIR",
    artifact_store: Optional[ArtifactStore] = None,
) -> Tuple[str, ExportReport]:
    """Export to COLLADA .dae string and return structured ExportReport."""
    dae_str = export_to_dae(world, artifact_store)
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
            reasons.append("no exportable mesh or box geometry")

    report = ExportReport(
        format="dae",
        world_id=world.id,
        world_version=world.version,
        entities_exported=tuple(exported),
        entities_skipped=tuple(skipped),
        skip_reasons=tuple(reasons),
        content_hash=content_hash(dae_str),
    )
    return dae_str, report
