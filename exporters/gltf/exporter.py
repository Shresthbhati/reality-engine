"""Minimal glTF 2.0 exporter for WorldIR.

Scope (deliberately narrow — see docs/REALITY_ENGINE_AUDIT.md item 34):
  Exports one glTF node + unit-cube mesh primitive for every WorldIR Entity
  that has BOTH a transform with a position AND at least one Geometry of
  type BOX. The entity's transform position becomes the node's
  `translation`.

What is intentionally NOT exported, and why:
  - Entities with no transform: glTF nodes need a placement; there is
    nothing to place them at, so they are skipped rather than guessed.
  - Entities whose geometry is MESH, POINTCLOUD, or anything other than
    BOX: WorldIR's `Geometry` dataclass (world_ir/schema_v1.py) stores only
    a `vertex_count` metadata int — it does not store actual vertex
    positions, faces, or point data anywhere. There is genuinely nothing
    real to export for those types yet, so they are skipped rather than
    fabricating a mesh. Exporting real mesh/point-cloud geometry requires
    WorldIR to gain actual vertex-buffer storage first (separate, larger
    work item).
  - Materials, textures, skinning, animation, and node hierarchy/parenting:
    out of scope for this pass; WorldIR's Entity/Geometry model does not
    yet carry the data these would need either.

Output is a single-file .gltf: the unit-cube vertex/index buffer is
embedded as a base64 data URI in the one `buffers[0].uri`, so no separate
.bin file is produced.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import TYPE_CHECKING

from world_ir.schema_v1 import GeometryType

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


def _entity_box_geometry(world: "WorldIR", entity) -> bool:
    """True iff `entity` has at least one BOX geometry attached."""
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.type == GeometryType.BOX:
            return True
    return False


def export_to_gltf(world: "WorldIR") -> dict:
    """Build a glTF 2.0 JSON structure (as a plain dict) from `world`.

    Only entities with a transform position AND a BOX geometry produce a
    node — see module docstring for exactly what is skipped and why.
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

    for entity in world.entities.values():
        if not entity.transform or "position" not in entity.transform:
            continue  # no placement to export the node with
        if not _entity_box_geometry(world, entity):
            continue  # nothing real to export (see module docstring)

        pos = entity.transform["position"]
        node_index = len(gltf["nodes"])
        gltf["nodes"].append(
            {
                "name": entity.name or entity.id,
                "mesh": 0,
                "translation": [pos["x"], pos["y"], pos["z"]],
            }
        )
        gltf["scenes"][0]["nodes"].append(node_index)

    return gltf


def write_gltf_file(world: "WorldIR", path: str) -> None:
    """Export `world` to glTF 2.0 JSON and write it to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_to_gltf(world), f)
