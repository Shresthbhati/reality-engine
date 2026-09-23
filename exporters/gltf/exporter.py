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
from world_ir.schema_v1 import GeometryType

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
            vertices, faces = real_mesh
            tri = _real_triangles_mesh(vertices, faces, (pos["x"], pos["y"], pos["z"]))
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
            mesh_index = len(gltf["meshes"])
            gltf["meshes"].append({
                "primitives": [{
                    "attributes": {"POSITION": pos_acc},
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

        node_index = len(gltf["nodes"])
        gltf["nodes"].append(
            {
                "name": entity.name or entity.id,
                "mesh": mesh_index,
                "translation": [pos["x"], pos["y"], pos["z"]],
            }
        )
        gltf["scenes"][0]["nodes"].append(node_index)

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
    from reconstruction.meshing.mesh import MeshData as _MeshData

    try:
        mesh = _MeshData.from_bytes(payload)
    except (ValueError, struct.error):
        return None  # not a MeshData payload this exporter understands
    return mesh.vertices, mesh.faces


def _real_triangles_mesh(
    vertices: "list[tuple[float, float, float]]",
    faces: "list[tuple[int, int, int]]",
    origin: "tuple[float, float, float]",
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
    return {
        "position_bytes": position_bytes,
        "index_bytes": index_bytes,
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
