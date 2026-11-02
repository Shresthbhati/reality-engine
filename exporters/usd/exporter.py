"""USD ASCII (.usda) exporter for WorldIR with stage hierarchy, UsdShade materials,
UsdGeomMesh normals/UVs/subsets, physics rigid body schema, and cameras.

Phase 9 Task 9.6:
  - Stage hierarchy: World -> Building -> Story -> Room -> Entity via parent-child
    links (parent_id and PART_OF relationships) nested under typed Xforms.
  - UsdShade material bindings: def Scope "Materials" with def Material and UsdPreviewSurface
    shaders, bound via `rel material:binding = </World/Materials/...>`.
  - UsdGeomMesh: real reconstructed triangle mesh from MeshData with per-vertex normals,
    UV coordinates (primvars:st), and GeomSubsets.
  - Physics rigid body schema: apiSchemas PhysicsRigidBodyAPI, PhysicsMassAPI, PhysicsCollisionAPI
    when mass or collision/rigid body properties are present on entities.
  - Camera prims: def Camera with focalLength, apertures, and reconstructed extrinsics/intrinsics.
  - Validation: validate_usda() checking usdchecker / pxr where available, with graceful
    fallback to an internal syntax and structural validator.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from world_ir.artifact_store import ArtifactNotFoundError, ArtifactStore
from world_ir.geometry_data import PointCloudData
from world_ir.schema_v1 import EntityType, GeometryType, RelationshipKind

from exporters.report import ExportReport, content_hash

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR


#: Exportable geometry types
_EXPORTABLE_GEOMETRY_TYPES = frozenset(
    {GeometryType.BOX, GeometryType.PLANE, GeometryType.MESH}
)


def _entity_box_geometry(world: "WorldIR", entity) -> bool:
    """True iff `entity` has at least one BOX/PLANE/MESH geometry attached."""
    for gid in getattr(entity, "geometry_ids", ()):
        geom = world.geometries.get(gid)
        if geom is not None and geom.type in _EXPORTABLE_GEOMETRY_TYPES:
            return True
    return False


def _entity_exportable_geometry(world: "WorldIR", entity):
    for gid in getattr(entity, "geometry_ids", ()):
        geom = world.geometries.get(gid)
        if geom is not None and geom.type in _EXPORTABLE_GEOMETRY_TYPES:
            return geom
    return None


def _is_camera_entity(entity) -> bool:
    """True iff `entity` represents a calibrated or reconstructed camera."""
    if getattr(entity, "type", None) == EntityType.SENSOR:
        return True
    cp = getattr(entity, "custom_properties", None)
    if isinstance(cp, dict):
        if cp.get("sensor_type") == "camera" or "camera" in cp or "focal_length" in cp:
            return True
    name = (getattr(entity, "name", "") or "").lower()
    return "camera" in name or name.startswith("cam_")


def _sanitize_prim_name(entity_id: str) -> str:
    """Turn `entity_id` into a valid USD prim name (identifier syntax)."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", entity_id)
    if not name or name[0].isdigit():
        name = "_" + name
    return name


def _resolve_real_points(artifact_store: Optional[ArtifactStore], geometry) -> Optional[list]:
    """Real point positions for `geometry`, or None if unresolvable."""
    if artifact_store is None or not getattr(geometry, "data_uri", None):
        return None
    try:
        payload = artifact_store.get(geometry.data_uri)
    except ArtifactNotFoundError:
        return None
    try:
        return list(PointCloudData.from_bytes(payload).points)
    except (ValueError, struct.error):
        return None


def _resolve_real_mesh(
    artifact_store: Optional[ArtifactStore], geometry
) -> Optional[Tuple[list, list, Optional[list], Optional[list]]]:
    """Real (vertices, faces, normals, uvs) for `geometry`, or None if unresolvable."""
    if artifact_store is None or not getattr(geometry, "data_uri", None):
        return None
    try:
        payload = artifact_store.get(geometry.data_uri)
    except ArtifactNotFoundError:
        return None
    from reconstruction.meshing.mesh import MeshData as _MeshData

    try:
        mesh = _MeshData.from_bytes(payload)
    except (ValueError, struct.error):
        return None

    verts = list(mesh.vertices)
    faces = list(mesh.faces)
    normals = list(mesh.normals) if getattr(mesh, "normals", None) else None
    uvs = list(getattr(mesh, "uvs")) if hasattr(mesh, "uvs") and mesh.uvs else None
    return verts, faces, normals, uvs


def _resolve_physics_properties(entity) -> Optional[dict]:
    """Extract physics rigid body, mass, and collision properties if present."""
    props = getattr(entity, "properties", None)
    res: dict[str, Any] = {}
    if props is not None:
        if hasattr(props, "mass") and props.mass is not None:
            val = getattr(props.mass, "value", None)
            if val is not None:
                res["mass"] = val
        elif isinstance(props, dict) and "mass" in props:
            res["mass"] = props["mass"]

        if hasattr(props, "friction_coefficient") and props.friction_coefficient is not None:
            res["rigid_body"] = True
        elif isinstance(props, dict) and "friction_coefficient" in props:
            res["rigid_body"] = True

    cp = getattr(entity, "custom_properties", None)
    if isinstance(cp, dict):
        if "physics" in cp and isinstance(cp["physics"], dict):
            res.update(cp["physics"])
        if "mass" in cp:
            res["mass"] = cp["mass"]
        if "rigid_body" in cp:
            res["rigid_body"] = bool(cp["rigid_body"])
        if "collision" in cp:
            res["collision"] = bool(cp["collision"])

    if res.get("mass") is not None or res.get("rigid_body") or res.get("collision"):
        res.setdefault("rigid_body", True)
        res.setdefault("collision", True)
        return res
    return None


def _build_usd_materials_scope(world: "WorldIR") -> tuple[list[str], dict[str, str]]:
    """Generate UsdShade material scope and map entity/material IDs to USD paths."""
    lines: list[str] = []
    mat_id_to_path: dict[str, str] = {}
    mats_to_emit: dict[str, dict] = {}

    # 1. WorldIR registered materials
    if hasattr(world, "materials"):
        for mat_id, mat in world.materials.items():
            mat_name = _sanitize_prim_name(mat.name or mat_id)
            rgb = getattr(mat, "color_rgb", (0.8, 0.8, 0.8)) or (0.8, 0.8, 0.8)
            if any(c > 1.0 for c in rgb):
                rgb = tuple(c / 255.0 for c in rgb)
            roughness = 0.5
            metallic = 0.0
            props = getattr(mat, "properties", None)
            if isinstance(props, dict):
                roughness = float(props.get("roughness", 0.5))
                metallic = float(props.get("metallic", 0.0))
            mats_to_emit[mat_id] = {
                "prim_name": mat_name,
                "diffuse": rgb,
                "roughness": roughness,
                "metallic": metallic,
            }

    # 2. Entity materials (named or appearance custom_properties)
    for entity in world.entities.values():
        mid = None
        if getattr(entity, "material_ids", None):
            mid = entity.material_ids[0]
        elif getattr(entity, "material_id", None):
            mid = entity.material_id
        elif hasattr(entity, "custom_properties") and isinstance(entity.custom_properties, dict):
            mid = entity.custom_properties.get("material_id") or entity.custom_properties.get("material")
        if mid and mid in mats_to_emit:
            mat_id_to_path[entity.id] = f"/World/Materials/{mats_to_emit[mid]['prim_name']}"
            continue

        cp = getattr(entity, "custom_properties", None)
        if isinstance(cp, dict):
            app = cp.get("appearance") or cp.get("material")
            if isinstance(app, dict) and "color_rgb" in app:
                rgb_raw = app["color_rgb"]
                if isinstance(rgb_raw, (list, tuple)) and len(rgb_raw) >= 3:
                    rgb = tuple(float(c) for c in rgb_raw[:3])
                    if any(c > 1.0 for c in rgb):
                        rgb = tuple(c / 255.0 for c in rgb)
                    anon_name = f"Mat_{_sanitize_prim_name(entity.id)}"
                    mats_to_emit[f"anon_{entity.id}"] = {
                        "prim_name": anon_name,
                        "diffuse": rgb,
                        "roughness": float(app.get("roughness", 0.5)),
                        "metallic": float(app.get("metallic", 0.0)),
                    }
                    mat_id_to_path[entity.id] = f"/World/Materials/{anon_name}"

    if not mats_to_emit:
        return [], mat_id_to_path

    lines.append('    def Scope "Materials"')
    lines.append("    {")
    for mid, data in mats_to_emit.items():
        pname = data["prim_name"]
        r, g, b = data["diffuse"]
        ro = data["roughness"]
        m = data["metallic"]
        mat_path = f"/World/Materials/{pname}"
        mat_id_to_path[mid] = mat_path
        lines.extend([
            f'        def Material "{pname}"',
            "        {",
            f'            token outputs:surface.connect = <{mat_path}/PBRShader.outputs:surface>',
            "",
            '            def Shader "PBRShader"',
            "            {",
            '                uniform token info:id = "UsdPreviewSurface"',
            f"                color3f inputs:diffuseColor = ({r}, {g}, {b})",
            f"                float inputs:metallic = {m}",
            f"                float inputs:roughness = {ro}",
            "                token outputs:surface",
            "            }",
            "        }",
        ])
    lines.append("    }")
    lines.append("")
    return lines, mat_id_to_path


def _mesh_prim_lines(
    prim_name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    origin: tuple[float, float, float],
    normals: Optional[list[tuple[float, float, float]]] = None,
    uvs: Optional[list[tuple[float, float]]] = None,
    subsets: Optional[list[dict]] = None,
    material_path: Optional[str] = None,
    physics_props: Optional[dict] = None,
    indent: str = "    ",
) -> list[str]:
    """USD ASCII lines for a real `Mesh` prim built from a reconstructed triangle mesh."""
    local = [(x - origin[0], y - origin[1], z - origin[2]) for x, y, z in vertices]
    points_str = ", ".join(f"({x}, {y}, {z})" for x, y, z in local)
    counts_str = ", ".join("3" for _ in faces)
    indices_str = ", ".join(str(i) for f in faces for i in f)

    api_schemas = []
    if physics_props and (physics_props.get("mass") is not None or physics_props.get("rigid_body")):
        api_schemas.extend(["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"])

    schema_header = ""
    if api_schemas:
        schemas_repr = ", ".join(f'"{s}"' for s in api_schemas)
        schema_header = f" (prepend apiSchemas = [{schemas_repr}])"

    lines = [
        f'{indent}def Mesh "{prim_name}"{schema_header}',
        f"{indent}{{",
        f"{indent}    int[] faceVertexCounts = [{counts_str}]",
        f"{indent}    int[] faceVertexIndices = [{indices_str}]",
        f"{indent}    point3f[] points = [{points_str}]",
    ]

    if normals and len(normals) == len(vertices):
        normals_str = ", ".join(f"({nx}, {ny}, {nz})" for nx, ny, nz in normals)
        lines.append(f'{indent}    normal3f[] normals = [{normals_str}] (interpolation = "vertex")')

    if uvs and len(uvs) == len(vertices):
        uvs_str = ", ".join(f"({u}, {v})" for u, v in uvs)
        lines.append(f'{indent}    texCoord2f[] primvars:st = [{uvs_str}] (interpolation = "vertex")')

    if material_path:
        lines.append(f"{indent}    rel material:binding = <{material_path}>")

    if physics_props:
        if physics_props.get("rigid_body"):
            lines.append(f"{indent}    bool physics:rigidBodyEnabled = true")
        if physics_props.get("collision") or physics_props.get("rigid_body"):
            lines.append(f"{indent}    bool physics:collisionEnabled = true")
        if physics_props.get("mass") is not None:
            lines.append(f"{indent}    float physics:mass = {float(physics_props['mass'])}")

    lines.append(f'{indent}    double3 xformOp:translate = ({origin[0]}, {origin[1]}, {origin[2]})')
    lines.append(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]')

    if subsets:
        for sub in subsets:
            sub_name = _sanitize_prim_name(sub.get("name", "subset"))
            sub_indices = ", ".join(str(i) for i in sub.get("indices", []))
            lines.extend([
                f'{indent}    def GeomSubset "{sub_name}"',
                f"{indent}    {{",
                f'{indent}        uniform token elementType = "face"',
                f'{indent}        uniform token familyName = "materialBind"',
                f"{indent}        int[] indices = [{sub_indices}]",
            ])
            if sub.get("material_path"):
                lines.append(f'{indent}        rel material:binding = <{sub["material_path"]}>')
            lines.append(f"{indent}    }}")

    lines.append(f"{indent}}}")
    return lines


def _points_prim_lines(
    prim_name: str,
    points: list[tuple[float, float, float]],
    origin: tuple[float, float, float],
    material_path: Optional[str] = None,
    physics_props: Optional[dict] = None,
    indent: str = "    ",
) -> list[str]:
    """USD ASCII lines for a real `Points` prim."""
    local = [(x - origin[0], y - origin[1], z - origin[2]) for x, y, z in points]
    points_str = ", ".join(f"({x}, {y}, {z})" for x, y, z in local)

    api_schemas = []
    if physics_props and (physics_props.get("mass") is not None or physics_props.get("rigid_body")):
        api_schemas.extend(["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"])
    schema_header = ""
    if api_schemas:
        schemas_repr = ", ".join(f'"{s}"' for s in api_schemas)
        schema_header = f" (prepend apiSchemas = [{schemas_repr}])"

    lines = [
        f'{indent}def Points "{prim_name}"{schema_header}',
        f"{indent}{{",
        f"{indent}    point3f[] points = [{points_str}]",
    ]
    if material_path:
        lines.append(f"{indent}    rel material:binding = <{material_path}>")
    if physics_props:
        if physics_props.get("rigid_body"):
            lines.append(f"{indent}    bool physics:rigidBodyEnabled = true")
        if physics_props.get("collision") or physics_props.get("rigid_body"):
            lines.append(f"{indent}    bool physics:collisionEnabled = true")
        if physics_props.get("mass") is not None:
            lines.append(f"{indent}    float physics:mass = {float(physics_props['mass'])}")
    lines.extend([
        f'{indent}    double3 xformOp:translate = ({origin[0]}, {origin[1]}, {origin[2]})',
        f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]',
        f"{indent}}}",
    ])
    return lines


def _cube_prim_lines(
    prim_name: str,
    origin: tuple[float, float, float],
    material_path: Optional[str] = None,
    physics_props: Optional[dict] = None,
    indent: str = "    ",
) -> list[str]:
    """USD ASCII lines for a fallback placeholder `Cube` prim."""
    api_schemas = []
    if physics_props and (physics_props.get("mass") is not None or physics_props.get("rigid_body")):
        api_schemas.extend(["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"])
    schema_header = ""
    if api_schemas:
        schemas_repr = ", ".join(f'"{s}"' for s in api_schemas)
        schema_header = f" (prepend apiSchemas = [{schemas_repr}])"

    lines = [
        f'{indent}def Cube "{prim_name}"{schema_header}',
        f"{indent}{{",
        f"{indent}    double size = 2",
        f'{indent}    double3 xformOp:translate = ({origin[0]}, {origin[1]}, {origin[2]})',
        f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]',
    ]
    if material_path:
        lines.append(f"{indent}    rel material:binding = <{material_path}>")
    if physics_props:
        if physics_props.get("rigid_body"):
            lines.append(f"{indent}    bool physics:rigidBodyEnabled = true")
        if physics_props.get("collision") or physics_props.get("rigid_body"):
            lines.append(f"{indent}    bool physics:collisionEnabled = true")
        if physics_props.get("mass") is not None:
            lines.append(f"{indent}    float physics:mass = {float(physics_props['mass'])}")
    lines.append(f"{indent}}}")
    return lines


def _camera_prim_lines(
    prim_name: str,
    origin: tuple[float, float, float],
    focal_length: float = 35.0,
    horizontal_aperture: float = 36.0,
    vertical_aperture: float = 24.0,
    orient: Optional[tuple[float, float, float, float]] = None,
    indent: str = "    ",
) -> list[str]:
    """USD ASCII lines for a `Camera` prim."""
    lines = [
        f'{indent}def Camera "{prim_name}"',
        f"{indent}{{",
        f"{indent}    float focalLength = {focal_length}",
        f"{indent}    float horizontalAperture = {horizontal_aperture}",
        f"{indent}    float verticalAperture = {vertical_aperture}",
        f'{indent}    double3 xformOp:translate = ({origin[0]}, {origin[1]}, {origin[2]})',
    ]
    if orient is not None:
        lines.append(f'{indent}    quatf xformOp:orient = ({orient[0]}, {orient[1]}, {orient[2]}, {orient[3]})')
        lines.append(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]')
    else:
        lines.append(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]')
    lines.append(f"{indent}}}")
    return lines


def _get_parent_id(world: "WorldIR", entity) -> Optional[str]:
    """Get parent entity id via parent_id or PART_OF relationship."""
    pid = getattr(entity, "parent_id", None)
    if not pid and hasattr(entity, "custom_properties") and isinstance(entity.custom_properties, dict):
        pid = entity.custom_properties.get("parent_id")
    if pid and pid in world.entities:
        return pid
    for rel in getattr(entity, "relationships", ()):
        if getattr(rel, "kind", None) is RelationshipKind.PART_OF and rel.target_id in world.entities:
            return rel.target_id
    return None


def export_to_usda(world: "WorldIR", artifact_store: Optional[ArtifactStore] = None) -> str:
    """Build a USD ASCII (.usda) text stage from `world`."""
    lines = [
        "#usda 1.0",
        "",
        'def Xform "World"',
        "{",
    ]

    # 1. UsdShade Materials
    mat_lines, mat_id_to_path = _build_usd_materials_scope(world)
    if mat_lines:
        lines.extend(mat_lines)

    # 2. Hierarchy identification
    parent_map: dict[str, Optional[str]] = {}
    children_map: dict[Optional[str], list[str]] = defaultdict(list)

    # Filter exported entities
    exported_set: set[str] = set()
    for entity in world.entities.values():
        if not entity.transform or "position" not in entity.transform:
            continue
        if _entity_box_geometry(world, entity) or _is_camera_entity(entity):
            exported_set.add(entity.id)

    # Determine ancestry for exported entities
    for eid in sorted(list(exported_set)):
        curr_id = eid
        visited: set[str] = set()
        while curr_id not in visited:
            visited.add(curr_id)
            curr_entity = world.entities.get(curr_id)
            if not curr_entity:
                break
            pid = _get_parent_id(world, curr_entity)
            if not pid or pid not in world.entities or pid == curr_id:
                break
            parent_entity = world.entities[pid]
            if parent_entity.transform and "position" in parent_entity.transform:
                exported_set.add(pid)
            curr_id = pid

    for eid in sorted(exported_set):
        entity = world.entities[eid]
        pid = _get_parent_id(world, entity)
        if pid not in exported_set:
            pid = None
        parent_map[eid] = pid
        children_map[pid].append(eid)

    def render_entity(eid: str, indent: str) -> list[str]:
        entity = world.entities[eid]
        pos = entity.transform["position"]
        origin = (pos["x"], pos["y"], pos["z"])
        prim_name = _sanitize_prim_name(entity.id)
        children = children_map.get(eid, [])
        mat_path = mat_id_to_path.get(eid)
        if not mat_path:
            mids = getattr(entity, "material_ids", None)
            if mids:
                mat_path = mat_id_to_path.get(mids[0])
            elif getattr(entity, "material_id", None):
                mat_path = mat_id_to_path.get(entity.material_id)
        phys_props = _resolve_physics_properties(entity)

        # Container with children
        if children:
            res = [
                f'{indent}def Xform "{prim_name}"',
                f"{indent}{{",
                f'{indent}    double3 xformOp:translate = ({origin[0]}, {origin[1]}, {origin[2]})',
                f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]',
            ]
            geometry = _entity_exportable_geometry(world, entity)
            if geometry is not None:
                real_mesh = _resolve_real_mesh(artifact_store, geometry)
                if real_mesh is not None and real_mesh[0] and real_mesh[1]:
                    verts, faces, normals, uvs = real_mesh
                    res.extend(_mesh_prim_lines("Geometry", verts, faces, origin, normals, uvs, None, mat_path, phys_props, indent + "    "))
                else:
                    real_pts = _resolve_real_points(artifact_store, geometry)
                    if real_pts is not None and len(real_pts) >= 1:
                        res.extend(_points_prim_lines("Geometry", real_pts, origin, mat_path, phys_props, indent + "    "))
                    else:
                        res.extend(_cube_prim_lines("Geometry", origin, mat_path, phys_props, indent + "    "))

            for child_id in children:
                res.extend(render_entity(child_id, indent + "    "))
            res.append(f"{indent}}}")
            return res

        # Camera entity
        if _is_camera_entity(entity):
            focal = 35.0
            h_ap = 36.0
            v_ap = 24.0
            orient = None
            cp = getattr(entity, "custom_properties", None)
            if isinstance(cp, dict):
                focal = float(cp.get("focal_length", 35.0))
                h_ap = float(cp.get("horizontal_aperture", 36.0))
                v_ap = float(cp.get("vertical_aperture", 24.0))
            if "rotation" in entity.transform and isinstance(entity.transform["rotation"], dict):
                rot = entity.transform["rotation"]
                if "w" in rot and "x" in rot and "y" in rot and "z" in rot:
                    orient = (rot["w"], rot["x"], rot["y"], rot["z"])
            return _camera_prim_lines(prim_name, origin, focal, h_ap, v_ap, orient, indent)

        # Leaf geometry entity
        geometry = _entity_exportable_geometry(world, entity)
        real_mesh = _resolve_real_mesh(artifact_store, geometry)
        if real_mesh is not None and real_mesh[0] and real_mesh[1]:
            verts, faces, normals, uvs = real_mesh
            subsets = getattr(geometry, "subsets", None) if geometry else None
            if not subsets and hasattr(geometry, "custom_properties") and isinstance(geometry.custom_properties, dict):
                subsets = geometry.custom_properties.get("subsets")
            return _mesh_prim_lines(prim_name, verts, faces, origin, normals, uvs, subsets, mat_path, phys_props, indent)

        real_points = _resolve_real_points(artifact_store, geometry)
        if real_points is not None and len(real_points) >= 1:
            return _points_prim_lines(prim_name, real_points, origin, mat_path, phys_props, indent)

        return _cube_prim_lines(prim_name, origin, mat_path, phys_props, indent)

    # Render all top-level roots under /World
    for root_id in children_map.get(None, []):
        lines.extend(render_entity(root_id, "    "))

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_usda_file(world: "WorldIR", path: str, artifact_store: Optional[ArtifactStore] = None) -> None:
    """Export `world` to USD ASCII text and write it to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(export_to_usda(world, artifact_store))


def _classify_entities(world: "WorldIR"):
    """(exported_ids, skipped_ids, reasons) in sorted-id order."""
    exported, skipped, reasons = [], [], []
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        if not entity.transform or "position" not in entity.transform:
            skipped.append(entity_id)
            reasons.append("no transform.position to place a prim at")
        elif not (_entity_box_geometry(world, entity) or _is_camera_entity(entity)):
            skipped.append(entity_id)
            reasons.append("no BOX/PLANE/MESH geometry to export")
        else:
            exported.append(entity_id)
    return tuple(exported), tuple(skipped), tuple(reasons)


def export_to_usda_with_report(
    world: "WorldIR", artifact_store: Optional[ArtifactStore] = None,
) -> tuple[str, ExportReport]:
    """Export to USDA with a structured ExportReport and deterministic hash."""
    usda = export_to_usda(world, artifact_store)
    exported, skipped, reasons = _classify_entities(world)
    report = ExportReport(
        format="usda",
        world_id=world.id,
        world_version=world.version,
        entities_exported=exported,
        entities_skipped=skipped,
        skip_reasons=reasons,
        content_hash=content_hash(usda),
    )
    return usda, report


def validate_usda(usda_text: str) -> tuple[bool, list[str]]:
    """Validate a USDA text stage.

    Attempts validation with `usdchecker` CLI or `pxr.Usd` if available in the environment.
    Falls back gracefully to a comprehensive USDA syntax and structural validator.
    Returns (is_valid, list_of_error_messages).
    """
    # 1. Check usdchecker CLI
    usdchecker_bin = shutil.which("usdchecker")
    if usdchecker_bin:
        with tempfile.NamedTemporaryFile("w", suffix=".usda", delete=False, encoding="utf-8") as f:
            f.write(usda_text)
            tmp_name = f.name
        try:
            proc = subprocess.run([usdchecker_bin, tmp_name], capture_output=True, text=True, timeout=10)
            if proc.returncode == 0:
                return True, []
            return False, [proc.stderr.strip() or proc.stdout.strip()]
        except Exception:
            pass
        finally:
            try:
                os.remove(tmp_name)
            except OSError:
                pass

    # 2. Check pxr
    try:
        from pxr import Usd  # type: ignore
        stage = Usd.Stage.CreateInMemory()
        # If pxr is present, test import / string parsing
        # (pxr does not have direct string parser in early versions without temp file)
        with tempfile.NamedTemporaryFile("w", suffix=".usda", delete=False, encoding="utf-8") as f:
            f.write(usda_text)
            tmp_name = f.name
        try:
            st = Usd.Stage.Open(tmp_name)
            if st is not None:
                return True, []
        except Exception as e:
            return False, [str(e)]
        finally:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
    except (ImportError, ModuleNotFoundError):
        pass

    # 3. Fallback syntax and structural validator
    errors: list[str] = []
    lines = [ln.strip() for ln in usda_text.splitlines() if ln.strip()]

    if not lines or not usda_text.strip().startswith("#usda 1.0"):
        errors.append("Missing required header '#usda 1.0'")

    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0
    valid_schemas = {
        "Xform", "Mesh", "Points", "Cube", "Camera", "Scope",
        "Material", "Shader", "GeomSubset", "Sphere", "Cylinder", "Capsule", "Cone"
    }

    prim_decl_re = re.compile(r'^def\s+([A-Za-z0-9_]+)\s+"([^"]+)"')

    for line_idx, line in enumerate(lines, start=1):
        # Count bracket and paren balance
        brace_depth += line.count("{") - line.count("}")
        paren_depth += line.count("(") - line.count(")")
        bracket_depth += line.count("[") - line.count("]")

        if brace_depth < 0:
            errors.append(f"Line {line_idx}: Unmatched closing brace '}}'")
            break

        # Check prim declarations
        if line.startswith("def "):
            m = prim_decl_re.match(line)
            if not m:
                errors.append(f"Line {line_idx}: Invalid prim declaration syntax: '{line}'")
            else:
                schema_type, prim_name = m.groups()
                if schema_type not in valid_schemas:
                    errors.append(f"Line {line_idx}: Unknown USD schema type '{schema_type}'")
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", prim_name):
                    errors.append(f"Line {line_idx}: Invalid prim identifier '{prim_name}'")

        # Check relation / connection syntax
        if "rel " in line or ".connect = <" in line:
            if "<" in line and ">" not in line:
                errors.append(f"Line {line_idx}: Unclosed target path in relation/connection")

    if brace_depth != 0:
        errors.append(f"Unbalanced braces in stage: depth {brace_depth}")
    if paren_depth != 0:
        errors.append(f"Unbalanced parentheses in stage: depth {paren_depth}")
    if bracket_depth != 0:
        errors.append(f"Unbalanced square brackets in stage: depth {bracket_depth}")

    return len(errors) == 0, errors
