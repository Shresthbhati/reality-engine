"""Minimal USD ASCII (.usda) exporter for WorldIR.

Scope (deliberately narrow — matches exporters/gltf/exporter.py exactly):
  Exports one USD `Cube` prim, nested under a single flat "World" Xform, for
  every WorldIR Entity that has BOTH a transform with a position AND at least
  one Geometry of type BOX. The entity's transform position becomes the
  cube's `xformOp:translate`.

What is intentionally NOT exported, and why:
  - Entities with no transform: a USD prim needs a placement; there is
    nothing to place it at, so it is skipped rather than guessed.
  - Entities whose geometry is MESH, POINTCLOUD, or anything other than
    BOX: WorldIR's `Geometry` dataclass (world_ir/schema_v1.py) stores only
    a `vertex_count` metadata int — it does not store actual vertex
    positions, faces, or point data anywhere. There is genuinely nothing
    real to export for those types yet, so they are skipped rather than
    fabricating a mesh.
  - Real dimensions: `Geometry` for BOX has no stored width/height/depth,
    so every cube is emitted with USD's default unit `size = 2` (the same
    "no real shape data" honesty as the gltf exporter's unit-cube choice).
  - Materials, textures, real hierarchy beyond one flat parent Xform, and
    the real binary USD Crate format (.usd/.usdc): out of scope. This
    exporter produces hand-written .usda text conforming to the documented
    ASCII grammar — it has not been validated against the real `pxr`
    Python library or any real USD viewer.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from world_ir.schema_v1 import GeometryType

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR


def _entity_box_geometry(world: "WorldIR", entity) -> bool:
    """True iff `entity` has at least one BOX geometry attached."""
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.type == GeometryType.BOX:
            return True
    return False


def _sanitize_prim_name(entity_id: str) -> str:
    """Turn `entity_id` into a valid USD prim name (identifier syntax).

    USD prim names must consist of letters, digits, and underscores, and
    must not start with a digit. Any other character is replaced with an
    underscore; a leading digit gets an underscore prefix.
    """
    name = re.sub(r"[^A-Za-z0-9_]", "_", entity_id)
    if not name or name[0].isdigit():
        name = "_" + name
    return name


def export_to_usda(world: "WorldIR") -> str:
    """Build a USD ASCII (.usda) text stage from `world`.

    Only entities with a transform position AND a BOX geometry produce a
    Cube prim — see module docstring for exactly what is skipped and why.
    """
    lines = ["#usda 1.0", "", 'def Xform "World"', "{"]

    for entity in world.entities.values():
        if not entity.transform or "position" not in entity.transform:
            continue  # no placement to export the prim with
        if not _entity_box_geometry(world, entity):
            continue  # nothing real to export (see module docstring)

        pos = entity.transform["position"]
        prim_name = _sanitize_prim_name(entity.id)
        lines.append(f'    def Cube "{prim_name}"')
        lines.append("    {")
        lines.append("        double size = 2")
        lines.append(f'        double3 xformOp:translate = ({pos["x"]}, {pos["y"]}, {pos["z"]})')
        lines.append('        uniform token[] xformOpOrder = ["xformOp:translate"]')
        lines.append("    }")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_usda_file(world: "WorldIR", path: str) -> None:
    """Export `world` to USD ASCII text and write it to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(export_to_usda(world))
