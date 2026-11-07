"""Minimal glTF 2.0 exporter for WorldIR.

Scope (deliberately narrow — see docs/REALITY_ENGINE_AUDIT.md item 34):
  Exports one glTF node + unit-cube mesh primitive for every WorldIR Entity
  that has BOTH a transform with a position AND at least one Geometry of
  type BOX or PLANE (PLANE added 2026-09-13: evidence/promote_planes.py
  always sets bounds_min/bounds_max from real inlier points, same kind of
  data a BOX geometry would carry). The entity's transform position
  becomes the node's `translation`.

Real geometry (2026-09-13, P0.10/P0.11): pass an `artifact_store`
(world_ir/artifact_store.py) and any entity whose geometry has a
`data_uri` this store can resolve (a `PointCloudData` payload, written
by evidence/promote_planes.py when it was given the same store) gets
its own real mesh, built from actual reconstructed point positions, as
a POINTS-mode primitive — not the placeholder cube. Nothing is
fabricated: an entity with no resolvable real data (no store passed,
no data_uri set, or the store doesn't have that artifact) falls back to
the cube exactly as before.

What is intentionally NOT exported, and why:
  - Entities with no transform: glTF nodes need a placement; there is
    nothing to place them at, so they are skipped rather than guessed.
  - Entities whose geometry is POINTCLOUD (or anything else without a
    resolvable real payload): still skipped in this pass — real-data
    support covers BOX/PLANE (points artifact) and now MESH (real
    triangle mesh artifact, `_EXPORTABLE_GEOMETRY_TYPES`).
  - Without an artifact_store (or for a geometry with no real payload),
    BOX/PLANE entities still get the placeholder unit cube; their real
    AABB size (`bounds_min`/`bounds_max`) is not yet used to scale it
    here (see exporters/blender/exporter.py, which does use it).
  - Materials, textures, skinning, animation, and node hierarchy/parenting:
    out of scope for this pass; WorldIR's Entity/Geometry model does not
    yet carry the data these would need either.

Output is a single-file .gltf: all buffer data (the shared unit cube,
plus any real per-entity point-cloud data) is embedded as one base64
data URI in `buffers[0].uri`, so no separate .bin file is produced.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import TYPE_CHECKING, Optional

from world_ir.artifact_store import ArtifactNotFoundError, ArtifactStore
from world_ir.geometry_data import PointCloudData
from world_ir.schema_v1 import GeometryType, RelationshipKind

from exporters.report import ExportReport, content_hash

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR

# Unit cube (centered on origin, side length 1) — the only geometry this
# exporter knows how to emit, since it is the only shape WorldIR's BOX
# GeometryType requires no extra data to represent.
_CUBE_POSITIONS: list[tuple[float, float, float]] = [
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
]

_CUBE_INDICES: list[int] = [
    0, 1, 2, 0, 2, 3,  # back
    4, 6, 5, 4, 7, 6,  # front
    0, 3, 7, 0, 7, 4,  # left
    1, 5, 6, 1, 6, 2,  # right
    0, 4, 5, 0, 5, 1,  # bottom
    3, 2, 6, 3, 6, 7,  # top
]

_COMPONENT_TYPE_FLOAT = 5126
_COMPONENT_TYPE_USHORT = 5123
_COMPONENT_TYPE_UINT = 5125


#: PLANE carries a real inlier AABB (evidence/promote_planes.py always
#: sets bounds_min/bounds_max), same as BOX -- both get the placeholder
#: unit cube since neither stores real vertex data, but skipping PLANE
#: would silently drop every promoted wall/floor/ceiling entity.
_EXPORTABLE_GEOMETRY_TYPES = frozenset(
    {GeometryType.BOX, GeometryType.PLANE, GeometryType.MESH}
)


def _entity_box_geometry(world: "WorldIR", entity) -> bool:
    """True iff `entity` has at least one BOX/PLANE/MESH geometry attached."""
    return _entity_exportable_geometry(world, entity) is not None


def _entity_exportable_geometry(world: "WorldIR", entity):
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.type in _EXPORTABLE_GEOMETRY_TYPES:
            return geom
    return None


def _real_points_mesh(
    points: "list[tuple[float, float, float]]", origin: "tuple[float, float, float]",
) -> dict:
    """A real glTF mesh primitive (POINTS mode, no indices) from actual
    geometry-artifact positions, translated into the entity's local
    space (the node's own `translation` already places the origin)."""
    local = [(x - origin[0], y - origin[1], z - origin[2]) for x, y, z in points]
    position_bytes = b"".join(struct.pack("<fff", *v) for v in local)
    xs, ys, zs = [v[0] for v in local], [v[1] for v in local], [v[2] for v in local]
    return {
        "buffer_bytes": position_bytes,
        "accessor": {
            "componentType": _COMPONENT_TYPE_FLOAT,
            "count": len(local),
            "type": "VEC3",
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        },
        "mode": 0,  # POINTS
    }


#: neutral defaults by surface type: (rgb 0-1, roughness, alpha). Authored materials and measured colours always win.
_SURFACE_DEFAULTS = {
    "floor": ((0.62, 0.55, 0.47), 0.8, 1.0),
    "ceiling": ((0.93, 0.93, 0.92), 0.9, 1.0),
    "wall": ((0.88, 0.87, 0.84), 0.9, 1.0),
    "window": ((0.62, 0.78, 0.9), 0.1, 0.35),
    "door": ((0.55, 0.42, 0.3), 0.6, 1.0),
    "furniture": ((0.7, 0.66, 0.6), 0.7, 1.0),
}


def _resolve_material(
    world: "WorldIR",
    entity,
    gltf: dict,
    material_map: dict[str, int],
    artifact_store: Optional[ArtifactStore] = None,
    tail_buffer: Optional[bytearray] = None,
    base_buffer_len: int = 0,
) -> Optional[int]:
    """Resolve or create a glTF PBR material for an entity (Task 9.1)."""
    mat_id = getattr(entity, "material_id", None)
    if not mat_id and getattr(entity, "material_ids", None):
        mat_id = entity.material_ids[0]
    if not mat_id and hasattr(entity, "custom_properties") and isinstance(entity.custom_properties, dict):
        mat_id = entity.custom_properties.get("material_id") or entity.custom_properties.get("material")

    mat = world.materials.get(mat_id) if (mat_id and hasattr(world, "materials") and mat_id in world.materials) else None

    if mat is not None:
        if mat.id in material_map:
            return material_map[mat.id]
        rgb = mat.color_rgb or [0.8, 0.8, 0.8]
        if any(c > 1.0 for c in rgb):
            rgb = [c / 255.0 for c in rgb]
        props = mat.properties if isinstance(mat.properties, dict) else {}
        metallic = float(props.get("metallic", 0.0))
        roughness = float(props.get("roughness", 0.7))
        mat_def = {
            "name": mat.name or mat.id,
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0],
                "metallicFactor": metallic,
                "roughnessFactor": roughness,
            },
        }
        # Embed texture if available
        if tail_buffer is not None:
            img_bytes: Optional[bytes] = None
            if "texture_bytes" in props and isinstance(props["texture_bytes"], (bytes, bytearray)):
                img_bytes = bytes(props["texture_bytes"])
            elif "texture_base64" in props and isinstance(props["texture_base64"], str):
                try:
                    img_bytes = base64.b64decode(props["texture_base64"])
                except Exception:
                    pass
            elif "texture_uri" in props and artifact_store:
                try:
                    img_bytes = artifact_store.get(props["texture_uri"])
                except Exception:
                    pass

            if img_bytes:
                # 4-byte pad
                pad = (-len(tail_buffer)) % 4
                tail_buffer.extend(b"\x00" * pad)
                img_offset = base_buffer_len + len(tail_buffer)
                tail_buffer.extend(img_bytes)
                img_bv_idx = len(gltf["bufferViews"])
                gltf["bufferViews"].append({
                    "buffer": 0,
                    "byteOffset": img_offset,
                    "byteLength": len(img_bytes),
                })
                img_idx = len(gltf.setdefault("images", []))
                gltf["images"].append({
                    "bufferView": img_bv_idx,
                    "mimeType": props.get("texture_mime", "image/png"),
                })
                if not gltf.get("samplers"):
                    gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
                tex_idx = len(gltf.setdefault("textures", []))
                gltf["textures"].append({"sampler": 0, "source": img_idx})
                mat_def["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_idx}

        gltf.setdefault("materials", []).append(mat_def)
        mat_idx = len(gltf["materials"]) - 1
        material_map[mat.id] = mat_idx
        return mat_idx

    # Check appearance in custom_properties
    cp = getattr(entity, "custom_properties", None)
    if isinstance(cp, dict):
        app = cp.get("appearance")
        if isinstance(app, dict) and "color_rgb" in app:
            rgb_raw = app["color_rgb"]
            if isinstance(rgb_raw, (list, tuple)) and len(rgb_raw) >= 3:
                key = f"anon_{entity.id}"
                if key in material_map:
                    return material_map[key]
                rgb = [float(c) for c in rgb_raw[:3]]
                if any(c > 1.0 for c in rgb):
                    rgb = [c / 255.0 for c in rgb]
                mat_def = {
                    "name": f"Material_{entity.id}",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [rgb[0], rgb[1], rgb[2], 1.0],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.7,
                    },
                }
                gltf.setdefault("materials", []).append(mat_def)
                mat_idx = len(gltf["materials"]) - 1
                material_map[key] = mat_idx
                return mat_idx

    # No authored material: a default named by surface type, shared by every element of that type, so downstream tools
    # (SketchUp, 3ds Max, Blender) get floor / wall / window ... materials instead of one anonymous grey.
    kind = getattr(getattr(entity, "type", None), "value", None)
    default = _SURFACE_DEFAULTS.get(kind)
    if default is not None:
        key = f"surface:{kind}"
        if key in material_map:
            return material_map[key]
        rgb, rough, alpha = default
        mat_def = {
            "name": f"surface_{kind}",
            "pbrMetallicRoughness": {"baseColorFactor": [rgb[0], rgb[1], rgb[2], alpha], "metallicFactor": 0.0, "roughnessFactor": rough},
        }
        if alpha < 1.0:
            mat_def["alphaMode"] = "BLEND"
        gltf.setdefault("materials", []).append(mat_def)
        material_map[key] = len(gltf["materials"]) - 1
        return material_map[key]

    return None


def export_to_gltf(world: "WorldIR", artifact_store: Optional[ArtifactStore] = None) -> dict:
    """Build a glTF 2.0 JSON structure (as a plain dict) from `world`.

    Only entities with a transform position AND a BOX/PLANE/MESH
    geometry produce a node — see module docstring for exactly what is
    skipped and why. When `artifact_store` is given and an entity's geometry
    carries a `data_uri` this store can resolve (world_ir/geometry_data.py's
    PointCloudData, written by evidence/promote_planes.py when it was
    given a store), that entity gets its OWN mesh built from real,
    reconstructed point positions (a POINTS-mode primitive) instead of
    the shared placeholder unit cube. Entities with no resolvable real
    data keep using the cube exactly as before — this is additive, never
    a behavior change for existing callers that omit `artifact_store`.
    """
    # Shared unit-cube buffer data: positions (float32 xyz) then indices
    # (uint16), both already 4-byte aligned (24 floats = 96 bytes,
    # 36 ushorts = 72 bytes).
    position_bytes = b"".join(struct.pack("<fff", *v) for v in _CUBE_POSITIONS)
    index_bytes = b"".join(struct.pack("<H", i) for i in _CUBE_INDICES)
    buffer_bytes = position_bytes + index_bytes
    buffer_uri = "data:application/octet-stream;base64," + base64.b64encode(buffer_bytes).decode("ascii")

    xs = [v[0] for v in _CUBE_POSITIONS]
    ys = [v[1] for v in _CUBE_POSITIONS]
    zs = [v[2] for v in _CUBE_POSITIONS]

    gltf: dict = {
        "asset": {"version": "2.0", "generator": "reality-engine gltf exporter"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": _COMPONENT_TYPE_FLOAT,
                "count": len(_CUBE_POSITIONS),
                "type": "VEC3",
                "min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": _COMPONENT_TYPE_USHORT,
                "count": len(_CUBE_INDICES),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(index_bytes), "target": 34963},
        ],
        "buffers": [{"byteLength": len(buffer_bytes), "uri": buffer_uri}],
    }

    tail_buffer = bytearray()  # additional real-mesh bytes, appended after the cube buffer
    material_map: dict[str, int] = {}
    entity_id_to_node_idx: dict[str, int] = {}
    cube_mesh_by_mat: dict[Optional[int], int] = {}

    for entity in world.entities.values():
        if not entity.transform or "position" not in entity.transform:
            continue  # no placement to export the node with
        geometry = _entity_exportable_geometry(world, entity)
        if geometry is None:
            continue  # nothing real to export (see module docstring)

        pos = entity.transform["position"]
        mesh_index = 0  # default: shared placeholder cube

        real_mesh = _resolve_real_mesh(artifact_store, geometry)
        if real_mesh is not None and real_mesh[0] and real_mesh[1]:
            vertices, faces = real_mesh[0], real_mesh[1]
            uvs = real_mesh[2] if len(real_mesh) > 2 else None
            tri = _real_triangles_mesh(vertices, faces, (pos["x"], pos["y"], pos["z"]), uvs=uvs)
            base_offset = len(buffer_bytes) + len(tail_buffer)
            pv_index = len(gltf["bufferViews"])
            tail_buffer.extend(tri["position_bytes"])
            gltf["bufferViews"].append({
                "buffer": 0, "byteOffset": base_offset,
                "byteLength": len(tri["position_bytes"]), "target": 34962,
            })
            pos_acc = len(gltf["accessors"])
            gltf["accessors"].append({"bufferView": pv_index, "byteOffset": 0, **tri["accessor"]})
            base_offset = len(buffer_bytes) + len(tail_buffer)
            iv_index = len(gltf["bufferViews"])
            tail_buffer.extend(tri["index_bytes"])
            gltf["bufferViews"].append({
                "buffer": 0, "byteOffset": base_offset,
                "byteLength": len(tri["index_bytes"]), "target": 34963,
            })
            idx_acc = len(gltf["accessors"])
            gltf["accessors"].append({"bufferView": iv_index, "byteOffset": 0, **tri["index_accessor"]})

            prim_attrs: dict[str, int] = {"POSITION": pos_acc}
            if tri.get("uv_bytes") and tri.get("uv_accessor"):
                base_offset = len(buffer_bytes) + len(tail_buffer)
                uv_index = len(gltf["bufferViews"])
                tail_buffer.extend(tri["uv_bytes"])
                gltf["bufferViews"].append({
                    "buffer": 0, "byteOffset": base_offset,
                    "byteLength": len(tri["uv_bytes"]), "target": 34962,
                })
                uv_acc = len(gltf["accessors"])
                gltf["accessors"].append({"bufferView": uv_index, "byteOffset": 0, **tri["uv_accessor"]})
                prim_attrs["TEXCOORD_0"] = uv_acc

            mesh_index = len(gltf["meshes"])
            gltf["meshes"].append({
                "primitives": [{
                    "attributes": prim_attrs,
                    "indices": idx_acc, "mode": 4,
                }]
            })
            # keep the next accessor 4-byte aligned
            tail_buffer.extend(b"\x00" * ((-len(tail_buffer)) % 4))
        else:
            real_points = _resolve_real_points(artifact_store, geometry)
            if real_points is not None and len(real_points) >= 1:
                real = _real_points_mesh(real_points, (pos["x"], pos["y"], pos["z"]))
                buffer_view_index = len(gltf["bufferViews"])
                byte_offset = len(buffer_bytes) + len(tail_buffer)
                tail_buffer.extend(real["buffer_bytes"])
                gltf["bufferViews"].append({
                    "buffer": 0, "byteOffset": byte_offset,
                    "byteLength": len(real["buffer_bytes"]), "target": 34962,
                })
                accessor_index = len(gltf["accessors"])
                gltf["accessors"].append({"bufferView": buffer_view_index, "byteOffset": 0, **real["accessor"]})
                mesh_index = len(gltf["meshes"])
                gltf["meshes"].append({
                    "primitives": [{"attributes": {"POSITION": accessor_index}, "mode": real["mode"]}]
                })
                # keep the next accessor 4-byte aligned
                tail_buffer.extend(b"\x00" * ((-len(tail_buffer)) % 4))

        # Material binding (Task 9.1)
        mat_idx = _resolve_material(world, entity, gltf, material_map, artifact_store=artifact_store, tail_buffer=tail_buffer, base_buffer_len=len(buffer_bytes))
        if mesh_index == 0:
            if mat_idx in cube_mesh_by_mat:
                mesh_index = cube_mesh_by_mat[mat_idx]
            else:
                if not cube_mesh_by_mat:
                    # Claim the initial placeholder cube mesh 0 so it is not an orphan
                    if mat_idx is not None:
                        gltf["meshes"][0]["primitives"][0]["material"] = mat_idx
                    cube_mesh_by_mat[mat_idx] = 0
                    mesh_index = 0
                else:
                    new_idx = len(gltf["meshes"])
                    cube_prim = {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}
                    if mat_idx is not None:
                        cube_prim["material"] = mat_idx
                    gltf["meshes"].append({"primitives": [cube_prim]})
                    cube_mesh_by_mat[mat_idx] = new_idx
                    mesh_index = new_idx
        elif mat_idx is not None:
            for prim in gltf["meshes"][mesh_index]["primitives"]:
                prim["material"] = mat_idx

        node_entry = {
            "name": entity.name or entity.id,
            "mesh": mesh_index,
            "translation": [float(pos["x"]), float(pos["y"]), float(pos["z"])],
        }
        if isinstance(entity.transform, dict):
            rot = entity.transform.get("rotation")
            if isinstance(rot, dict) and all(k in rot for k in ("x", "y", "z", "w")):
                node_entry["rotation"] = [float(rot["x"]), float(rot["y"]), float(rot["z"]), float(rot["w"])]
            scale = entity.transform.get("scale")
            if isinstance(scale, dict) and all(k in scale for k in ("x", "y", "z")):
                node_entry["scale"] = [float(scale["x"]), float(scale["y"]), float(scale["z"])]

        node_index = len(gltf["nodes"])
        gltf["nodes"].append(node_entry)
        entity_id_to_node_idx[entity.id] = node_index

    # Build hierarchy: parent-child links (Task 9.1)
    child_node_indices: set[int] = set()
    for eid, n_idx in entity_id_to_node_idx.items():
        e = world.entities[eid]
        pid = getattr(e, "parent_id", None)
        if not pid:
            for rel in getattr(e, "relationships", ()):
                if getattr(rel, "kind", None) is RelationshipKind.PART_OF and rel.target_id in entity_id_to_node_idx:
                    pid = rel.target_id
                    break
        if pid and pid in entity_id_to_node_idx:
            parent_idx = entity_id_to_node_idx[pid]
            if parent_idx != n_idx:
                parent_node = gltf["nodes"][parent_idx]
                parent_node.setdefault("children", []).append(n_idx)
                child_node_indices.add(n_idx)

    # Top-level root nodes
    root_nodes = [idx for idx in range(len(gltf["nodes"])) if idx not in child_node_indices]
    gltf["scenes"][0]["nodes"] = root_nodes

    if tail_buffer:
        full_bytes = buffer_bytes + bytes(tail_buffer)
        gltf["buffers"][0] = {
            "byteLength": len(full_bytes),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(full_bytes).decode("ascii"),
        }

    return gltf


def _resolve_real_points(artifact_store: Optional[ArtifactStore], geometry) -> Optional[list]:
    """Real point positions for `geometry`, or None if there is no
    resolvable real data (no store, no data_uri, or the store doesn't
    have this artifact) -- never fabricated, the caller falls back to
    the placeholder cube."""
    if artifact_store is None or not geometry.data_uri:
        return None
    try:
        payload = artifact_store.get(geometry.data_uri)
    except ArtifactNotFoundError:
        return None
    try:
        return list(PointCloudData.from_bytes(payload).points)
    except (ValueError, struct.error):
        return None  # not a PointCloudData payload this exporter understands


def _resolve_real_mesh(artifact_store: Optional[ArtifactStore], geometry) -> "tuple | None":
    """Real (vertices, faces) for a MESH-type geometry whose data_uri
    resolves to a reconstruction/meshing MeshData payload, or None --
    never fabricated, the caller falls back to the placeholder cube."""
    if artifact_store is None or not geometry.data_uri:
        return None
    if geometry.type is not GeometryType.MESH:
        return None
    try:
        payload = artifact_store.get(geometry.data_uri)
    except ArtifactNotFoundError:
        return None
    from world_ir import MeshData as _MeshData

    try:
        mesh = _MeshData.from_bytes(payload)
    except (ValueError, struct.error):
        return None  # not a MeshData payload this exporter understands
    return mesh.vertices, mesh.faces, getattr(mesh, "uvs", None)


def _real_triangles_mesh(
    vertices: "list[tuple[float, float, float]]",
    faces: "list[tuple[int, int, int]]",
    origin: "tuple[float, float, float]",
    uvs: "list[tuple[float, float]] | tuple | None" = None,
) -> dict:
    """A real glTF TRIANGLES primitive from a reconstructed mesh,
    translated into the entity's local space. Index width follows the
    vertex count: uint16 up to 65535 vertices, uint32 beyond (a real
    indoor Poisson mesh reached 240k vertices -- uint16 overflowed)."""
    local = [(x - origin[0], y - origin[1], z - origin[2]) for x, y, z in vertices]
    position_bytes = b"".join(struct.pack("<fff", *v) for v in local)
    use_uint32 = len(local) > 0xFFFF
    index_fmt = "<I" if use_uint32 else "<H"
    index_bytes = b"".join(
        struct.pack(index_fmt, i) for f in faces for i in f
    )
    xs, ys, zs = [v[0] for v in local], [v[1] for v in local], [v[2] for v in local]
    uv_bytes = b""
    uv_accessor = None
    if uvs is not None and len(uvs) == len(local):
        uv_bytes = b"".join(struct.pack("<2f", float(u), float(v)) for u, v in uvs)
        u_vals = [float(u) for u, v in uvs]
        v_vals = [float(v) for u, v in uvs]
        uv_accessor = {
            "componentType": _COMPONENT_TYPE_FLOAT,
            "count": len(uvs),
            "type": "VEC2",
            "min": [min(u_vals), min(v_vals)],
            "max": [max(u_vals), max(v_vals)],
        }

    return {
        "position_bytes": position_bytes,
        "index_bytes": index_bytes,
        "uv_bytes": uv_bytes,
        "accessor": {
            "componentType": _COMPONENT_TYPE_FLOAT,
            "count": len(local),
            "type": "VEC3",
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        },
        "index_accessor": {
            "componentType": (
                _COMPONENT_TYPE_UINT if use_uint32 else _COMPONENT_TYPE_USHORT
            ),
            "count": len(faces) * 3,
            "type": "SCALAR",
        },
        "uv_accessor": uv_accessor,
    }


def write_gltf_file(world: "WorldIR", path: str, artifact_store: Optional[ArtifactStore] = None) -> None:
    """Export `world` to glTF 2.0 JSON and write it to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_to_gltf(world, artifact_store), f)


def _classify_entities(world: "WorldIR"):
    """(exported_ids, skipped_ids, reasons) in sorted-id order, matching
    export_to_gltf()'s own per-entity skip conditions exactly."""
    exported, skipped, reasons = [], [], []
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        if not entity.transform or "position" not in entity.transform:
            skipped.append(entity_id)
            reasons.append("no transform.position to place a node at")
        elif not _entity_box_geometry(world, entity):
            skipped.append(entity_id)
            reasons.append("no BOX/PLANE/MESH geometry to export")
        else:
            exported.append(entity_id)
    return tuple(exported), tuple(skipped), tuple(reasons)


def export_to_gltf_with_report(
    world: "WorldIR", artifact_store: Optional[ArtifactStore] = None,
) -> tuple[dict, ExportReport]:
    """Same as export_to_gltf(), plus a structured ExportReport naming
    exactly which entities were exported/skipped and why, and a
    deterministic content hash of the resulting JSON."""
    gltf = export_to_gltf(world, artifact_store)
    exported, skipped, reasons = _classify_entities(world)
    report = ExportReport(
        format="gltf",
        world_id=world.id,
        world_version=world.version,
        entities_exported=exported,
        entities_skipped=skipped,
        skip_reasons=reasons,
        content_hash=content_hash(json.dumps(gltf, sort_keys=True)),
    )
    return gltf, report


def export_to_glb(world: "WorldIR", artifact_store: Optional[ArtifactStore] = None) -> bytes:
    """Export world to binary glTF 2.0 (GLB) container with embedded JSON and BIN chunks (Task 9.1)."""
    gltf = export_to_gltf(world, artifact_store)
    uri = gltf["buffers"][0]["uri"]
    raw_bin = base64.b64decode(uri.split(",", 1)[1])
    gltf_glb = json.loads(json.dumps(gltf))
    del gltf_glb["buffers"][0]["uri"]
    json_bytes = json.dumps(gltf_glb, separators=(",", ":")).encode("utf-8")

    json_pad = (-len(json_bytes)) % 4
    json_bytes += b" " * json_pad

    bin_pad = (-len(raw_bin)) % 4
    bin_bytes = raw_bin + b"\x00" * bin_pad

    total_len = 12 + (8 + len(json_bytes)) + (8 + len(bin_bytes))

    header = struct.pack("<4sII", b"glTF", 2, total_len)
    chunk0_hdr = struct.pack("<II", len(json_bytes), 0x4E4F534A)
    chunk1_hdr = struct.pack("<II", len(bin_bytes), 0x004E4942)

    return header + chunk0_hdr + json_bytes + chunk1_hdr + bin_bytes


def write_glb_file(world: "WorldIR", path: str, artifact_store: Optional[ArtifactStore] = None) -> None:
    """Export `world` to binary GLB 2.0 and write to `path`."""
    with open(path, "wb") as f:
        f.write(export_to_glb(world, artifact_store))
