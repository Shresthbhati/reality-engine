"""Blender import-script exporter for WorldIR.

Scope (matches exporters/gltf/exporter.py and exporters/usd/exporter.py):
  Produces a standalone Python script that, when run inside Blender
  (`blender --background --python scene.py`), recreates one mesh object
  for every WorldIR Entity that has BOTH a transform with a position AND
  at least one Geometry of type BOX or PLANE.

Why BOX and PLANE (unlike the gltf/usd exporters, which are BOX-only):
  `Geometry.bounds_min`/`bounds_max` (world_ir/schema_v1.py) is a real,
  non-fabricated axis-aligned bounding box whenever it is set --
  `evidence/promote_planes.py` always sets it from actual inlier points
  for PLANE geometry. When bounds are present this exporter emits a cube
  scaled to that real size (not a fixed unit cube); when they are absent
  (e.g. a BOX geometry created without bounds) it falls back to a plain
  1x1x1 cube, same honesty rule as the other two exporters: never invent
  a dimension that isn't backed by data.

Real geometry (2026-09-13, matches exporters/gltf/exporter.py): pass an
`artifact_store` (world_ir/artifact_store.py) and any entity whose
geometry has a `data_uri` this store can resolve (a `PointCloudData`
payload, written by evidence/promote_planes.py or
evidence/promote_objects.py when given the same store) gets a real
`bpy.data.meshes.new(...).from_pydata(...)` point-cloud mesh built from
actual reconstructed point positions (translated into the entity's own
local space, since the object's `location` already places the origin)
instead of the placeholder/AABB-scaled cube. Nothing is fabricated: an
entity with no resolvable real data (no store passed, no data_uri set,
or the store doesn't have that artifact) falls back to the cube exactly
as before.

What is intentionally NOT exported, and why:
  - Entities with no transform: nothing to place the object at.
  - Entities whose only geometry is MESH, POINTCLOUD, or anything other
    than BOX/PLANE: no real vertex/point data is stored in WorldIR yet
    (see the gltf/usd exporter docstrings) -- skipped rather than
    fabricated.
  - Real plane orientation (the PLANE Geometry does not retain the
    normal/d used at detection time, only the inlier AABB) -- the cube
    is placed and sized from that AABB, not rotated to match a wall's
    face; this is the same limitation the AABB-only Geometry schema
    imposes on every consumer today.
  - Materials, textures, and a real node hierarchy beyond one flat
    "RealityEngine" collection: out of scope for this pass.

Output is plain, dependency-free Python text: it references `bpy` (only
available inside Blender's own interpreter) but this module itself never
imports bpy, so it can be exercised and tested without Blender installed.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Optional

from world_ir.artifact_store import ArtifactNotFoundError, ArtifactStore
from world_ir.geometry_data import PointCloudData
from world_ir.schema_v1 import GeometryType

from exporters.report import ExportReport, content_hash

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR

#: Geometry types whose bounds_min/bounds_max (when set) is real,
#: non-fabricated data suitable for sizing an exported cube.
_EXPORTABLE_GEOMETRY_TYPES = frozenset({GeometryType.BOX, GeometryType.PLANE})

#: Minimum dimension (meters) for any exported axis -- a degenerate
#: (zero-thickness) plane AABB would otherwise produce an invisible,
#: unusable Blender mesh. This floors visibility without fabricating a
#: fake size for the other two axes.
_MIN_DIMENSION_M = 0.01


def _entity_geometry(world: "WorldIR", entity):
    """The first BOX/PLANE Geometry attached to `entity`, or None."""
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.type in _EXPORTABLE_GEOMETRY_TYPES:
            return geom
    return None


def _dimensions(geom) -> tuple[float, float, float]:
    """Real size from bounds_min/bounds_max, floored, or the (2, 2, 2)
    default Blender gives a cube primitive when no bounds are recorded."""
    if geom.bounds_min is None or geom.bounds_max is None:
        return (2.0, 2.0, 2.0)
    dx = max(_MIN_DIMENSION_M, geom.bounds_max.x - geom.bounds_min.x)
    dy = max(_MIN_DIMENSION_M, geom.bounds_max.y - geom.bounds_min.y)
    dz = max(_MIN_DIMENSION_M, geom.bounds_max.z - geom.bounds_min.z)
    return (dx, dy, dz)


def _py_str(value: str) -> str:
    """A Python string literal safe to splice into generated source."""
    return repr(value)


def _resolve_real_points(artifact_store: Optional[ArtifactStore], geometry) -> Optional[list]:
    """Real point positions for `geometry`, or None if there is no
    resolvable real data (no store, no data_uri, or the store doesn't
    have this artifact) -- never fabricated, the caller falls back to
    the placeholder/AABB cube. Mirrors exporters/gltf/exporter.py."""
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


def export_to_blender_script(world: "WorldIR", artifact_store: Optional[ArtifactStore] = None) -> str:
    """Build a standalone Blender Python script that reconstructs `world`.

    Only entities with a transform position AND a BOX/PLANE geometry
    produce an object -- see module docstring for exactly what is
    skipped and why. When `artifact_store` is given and an entity's
    geometry carries a resolvable `data_uri`, that entity gets a real
    point-cloud mesh (built via `from_pydata`) instead of the shared
    placeholder/AABB-scaled cube -- see module docstring. Omitting
    `artifact_store` reproduces the exact prior (cube-only) output.
    """
    lines: list[str] = [
        '"""Generated by exporters/blender/exporter.py -- do not edit by hand.',
        "",
        "Run with: blender --background --python <this file>",
        '"""',
        "",
        "import bpy",
        "",
        "# Remove Blender's default scene objects so re-running this script",
        "# on a fresh .blend produces exactly the entities below, nothing more.",
        "for obj in list(bpy.data.objects):",
        "    bpy.data.objects.remove(obj, do_unlink=True)",
        "",
        "collection = bpy.data.collections.new('RealityEngine')",
        "bpy.context.scene.collection.children.link(collection)",
        "",
    ]

    exported = 0
    for entity in world.entities.values():
        if not entity.transform or "position" not in entity.transform:
            continue  # no placement to export the object with
        geom = _entity_geometry(world, entity)
        if geom is None:
            continue  # nothing real to export (see module docstring)

        pos = entity.transform["position"]
        obj_name = entity.name or entity.id

        real_points = _resolve_real_points(artifact_store, geom)
        if real_points is not None and len(real_points) >= 1:
            local = [
                (x - pos["x"], y - pos["y"], z - pos["z"]) for x, y, z in real_points
            ]
            lines.append(f"mesh = bpy.data.meshes.new({_py_str(obj_name)})")
            lines.append(f"mesh.from_pydata({local!r}, [], [])")
            lines.append("mesh.update()")
            lines.append(f"obj = bpy.data.objects.new({_py_str(obj_name)}, mesh)")
            lines.append(f"obj.location = ({pos['x']}, {pos['y']}, {pos['z']})")
        else:
            dims = _dimensions(geom)
            lines.append(f"bpy.ops.mesh.primitive_cube_add(size=1.0, location=({pos['x']}, {pos['y']}, {pos['z']}))")
            lines.append("obj = bpy.context.active_object")
            lines.append(f"obj.name = {_py_str(obj_name)}")
            lines.append(f"obj.dimensions = ({dims[0]}, {dims[1]}, {dims[2]})")
        lines.append(f"obj['entity_id'] = {_py_str(entity.id)}")
        lines.append(f"obj['entity_type'] = {_py_str(entity.type.value)}")
        lines.append(f"obj['provenance'] = {_py_str(entity.provenance.value)}")
        lines.append(f"obj['confidence'] = {entity.confidence}")
        lines.append("for c in list(obj.users_collection):")
        lines.append("    c.objects.unlink(obj)")
        lines.append("collection.objects.link(obj)")
        lines.append("")
        exported += 1

    lines.append(f"# {exported} entity object(s) exported.")
    lines.append("")
    return "\n".join(lines)


def write_blender_script(world: "WorldIR", path: str, artifact_store: Optional[ArtifactStore] = None) -> None:
    """Export `world` to a Blender Python script and write it to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(export_to_blender_script(world, artifact_store))


def _classify_entities(world: "WorldIR"):
    """(exported_ids, skipped_ids, reasons) in sorted-id order, matching
    export_to_blender_script()'s own per-entity skip conditions exactly."""
    exported, skipped, reasons = [], [], []
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        if not entity.transform or "position" not in entity.transform:
            skipped.append(entity_id)
            reasons.append("no transform.position to place an object at")
        elif _entity_geometry(world, entity) is None:
            skipped.append(entity_id)
            reasons.append("no BOX/PLANE geometry to export")
        else:
            exported.append(entity_id)
    return tuple(exported), tuple(skipped), tuple(reasons)


def export_to_blender_script_with_report(
    world: "WorldIR", artifact_store: Optional[ArtifactStore] = None,
) -> tuple[str, ExportReport]:
    """Same as export_to_blender_script(), plus a structured ExportReport
    naming exactly which entities were exported/skipped and why, and a
    deterministic content hash of the generated script."""
    script = export_to_blender_script(world, artifact_store)
    exported, skipped, reasons = _classify_entities(world)
    report = ExportReport(
        format="blender-script",
        world_id=world.id,
        world_version=world.version,
        entities_exported=exported,
        entities_skipped=skipped,
        skip_reasons=reasons,
        content_hash=content_hash(script),
    )
    return script, report
