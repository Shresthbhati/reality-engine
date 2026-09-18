"""CityJSON 1.1 exporter for WorldIR (P16-01: downstream world
compilers -- the CityJSON/CityGML row).

Scope and honesty rules, mirroring the repo's established exporter
discipline (exporters/gltf, exporters/blender, exporters/usd):

  - Real CityJSON 1.1 structure: "type": "CityJSON", "version": "1.1",
    integer "vertices" under a "transform" quantization, "CityObjects"
    keyed by id. Parseable by official CityJSON tooling, not a
    lookalike.
  - Semantic mapping is honest: EntityType.BUILDING maps to the
    first-class "Building"; types without a CityJSON counterpart
    export as "GenericCityObject" with the WorldIR type recorded in
    attributes.worldir_type -- visible, traceable, never renamed into
    a fake Building.
  - Geometry is REAL: the entity's BOX/PLANE geometry bounds become a
    MultiSurface whose vertices are the actual AABB corners offset by
    the entity transform position (the entity's world placement).
    Degenerate axes floor to a minimum dimension (same rule and value
    as exporters/blender) rather than emitting invisible geometry.
    Entities without bounds are skipped with a reason, never given a
    fabricated size.
  - Every vertex lives in the top-level integer "vertices" array,
    deduplicated by exact quantized coordinates, referenced by index
    from the object boundaries (the CityJSON convention).
  - Deterministic: sorted object keys, fixed iteration order, stable
    vertex ordering; same world -> byte-identical JSON.

Not implemented in this pass (recorded, not hidden): semantic
surfaces (WallSurface/RoofSurface sub-objects), textures, CityJSON
Sequences, geometry templates, appearance blocks. WorldIR does not yet
carry per-face semantic surface labels.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from exporters.report import ExportReport, content_hash
from world_ir.schema_v1 import EntityType, GeometryType

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR

#: Geometry types whose bounds_min/bounds_max are real, non-fabricated
#: data suitable for emitting a MultiSurface (same set as the glTF and
#: Blender exporters: PLANE bounds are always set by
#: evidence/promote_planes.py from real inliers).
_EXPORTABLE_GEOMETRY_TYPES = frozenset({GeometryType.BOX, GeometryType.PLANE})

#: Minimum dimension (meters) for any exported axis -- identical rule
#: and value to exporters/blender/exporter.py: a zero-thickness plane
#: AABB would produce an invisible, unusable surface.
_MIN_DIMENSION_M = 0.01

#: Quantization: 1e-3 m (millimeter) vertices -- the CityJSON default
#: recommendation (0.001) for meter-scale scenes.
_VERTEX_SCALE = 0.001

#: WorldIR EntityType -> CityJSON first-class object type. Everything
#: else becomes GenericCityObject with attributes.worldir_type.
_ENTITY_TYPE_MAP = {
    EntityType.BUILDING.value: "Building",
    EntityType.STRUCTURE.value: "GenericCityObject",
    EntityType.TERRAIN.value: "LandUse",
    EntityType.VEGETATION.value: "PlantCover",
    EntityType.WATER.value: "WaterBody",
    EntityType.ROOM.value: "GenericCityObject",
}

CITYJSON_VERSION = "1.1"


def _entity_geometry(world: "WorldIR", entity):
    """The first BOX/PLANE Geometry attached to `entity`, or None."""
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.type in _EXPORTABLE_GEOMETRY_TYPES:
            return geom
    return None


def _cityjson_object_type(entity) -> str:
    return _ENTITY_TYPE_MAP.get(entity.type, "GenericCityObject")


def _world_corners(geom, position: dict):
    """The 8 world-space AABB corners of `geom` placed at `position`.

    Degenerate axes floor to _MIN_DIMENSION_M (same rule as the Blender
    exporter); geometry with no recorded bounds returns None -- the
    caller skips the entity rather than inventing a size.
    """
    if geom.bounds_min is None or geom.bounds_max is None:
        return None
    dx = max(_MIN_DIMENSION_M, geom.bounds_max.x - geom.bounds_min.x)
    dy = max(_MIN_DIMENSION_M, geom.bounds_max.y - geom.bounds_min.y)
    dz = max(_MIN_DIMENSION_M, geom.bounds_max.z - geom.bounds_min.z)
    cx, cy, cz = position["x"], position["y"], position["z"]
    corners = []
    for sx in (0.0, 1.0):
        for sy in (0.0, 1.0):
            for sz in (0.0, 1.0):
                corners.append((
                    cx + geom.bounds_min.x + sx * dx,
                    cy + geom.bounds_min.y + sy * dy,
                    cz + geom.bounds_min.z + sz * dz,
                ))
    return corners


def _quantize(corner, translate):
    return (
        round((corner[0] - translate[0]) / _VERTEX_SCALE),
        round((corner[1] - translate[1]) / _VERTEX_SCALE),
        round((corner[2] - translate[2]) / _VERTEX_SCALE),
    )


def export_to_cityjson(world: "WorldIR") -> dict:
    """Build a real CityJSON 1.1 document (a plain JSON-serializable
    dict) from `world`. See the module docstring for exactly what is
    exported, what is skipped and why."""
    translate = (0.0, 0.0, 0.0)
    vertices: list = []
    vertex_index: dict = {}
    city_objects: dict = {}

    for entity in world.entities.values():
        if not entity.transform or "position" not in entity.transform:
            continue  # no placement; nothing to place the object at
        geom = _entity_geometry(world, entity)
        if geom is None:
            continue  # no BOX/PLANE geometry with real bounds

        corners = _world_corners(geom, entity.transform["position"])
        if corners is None:
            continue  # geometry carries no real bounds

        # AABB corner order above enumerates x, then y, then z fastest;
        # the 6 quad faces of the box reference corners by index.
        faces = [
            [0, 2, 3, 1],  # y = 0  (corners 0-3)
            [4, 5, 7, 6],  # y = 1  (corners 4-7)
            [0, 1, 5, 4],  # x = 0
            [2, 6, 7, 3],  # x = 1
            [0, 4, 6, 2],  # z = 0
            [1, 3, 7, 5],  # z = 1
        ]
        boundaries = []
        for face in faces:
            ring = []
            for ci in face:
                q = _quantize(corners[ci], translate)
                idx = vertex_index.get(q)
                if idx is None:
                    idx = len(vertices)
                    vertex_index[q] = idx
                    vertices.append(list(q))
                ring.append(idx)
            boundaries.append(ring)

        position = entity.transform["position"]
        obj = {
            "type": _cityjson_object_type(entity),
            "attributes": {
                "worldir_type": entity.type,
                "worldir_provenance": entity.provenance.value,
                "worldir_confidence": entity.confidence,
            },
            "geometry": [{
                "type": "MultiSurface",
                "lod": "1",
                "boundaries": boundaries,
            }],
        }
        if entity.name:
            obj["attributes"]["name"] = entity.name
        city_objects[entity.id] = obj

    doc = {
        "type": "CityJSON",
        "version": CITYJSON_VERSION,
        "transform": {
            "scale": [_VERTEX_SCALE, _VERTEX_SCALE, _VERTEX_SCALE],
            "translate": list(translate),
        },
        "metadata": {
            "geographicalExtent": None,  # optional; CRS unknown here
        },
        "CityObjects": city_objects,
        "vertices": vertices,
    }
    # geographicalExtent None is not valid CityJSON content; drop it
    # rather than emitting a fake extent (honesty rule).
    if doc["metadata"]["geographicalExtent"] is None:
        del doc["metadata"]["geographicalExtent"]
    if not doc["metadata"]:
        del doc["metadata"]
    return doc


def export_to_cityjson_with_report(world: "WorldIR") -> tuple:
    """Same document as export_to_cityjson(), plus a structured
    ExportReport with the same skip-reason discipline as the glTF,
    USD and Blender exporters."""
    translate = (0.0, 0.0, 0.0)
    vertices: list = []
    vertex_index: dict = {}
    city_objects: dict = {}
    exported: list = []
    skipped: list = []
    reasons: list = []

    for entity in world.entities.values():
        if not entity.transform or "position" not in entity.transform:
            skipped.append(entity.id)
            reasons.append("no transform position")
            continue
        geom = _entity_geometry(world, entity)
        if geom is None:
            skipped.append(entity.id)
            reasons.append(
                "no BOX/PLANE geometry with real bounds attached")
            continue
        corners = _world_corners(geom, entity.transform["position"])
        if corners is None:
            skipped.append(entity.id)
            reasons.append("geometry has no real bounds_min/bounds_max")
            continue

        faces = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ]
        boundaries = []
        for face in faces:
            ring = []
            for ci in face:
                q = _quantize(corners[ci], translate)
                idx = vertex_index.get(q)
                if idx is None:
                    idx = len(vertices)
                    vertex_index[q] = idx
                    vertices.append(list(q))
                ring.append(idx)
            boundaries.append(ring)

        position = entity.transform["position"]
        obj = {
            "type": _cityjson_object_type(entity),
            "attributes": {
                "worldir_type": entity.type,
                "worldir_provenance": entity.provenance.value,
                "worldir_confidence": entity.confidence,
            },
            "geometry": [{
                "type": "MultiSurface",
                "lod": "1",
                "boundaries": boundaries,
            }],
        }
        if entity.name:
            obj["attributes"]["name"] = entity.name
        city_objects[entity.id] = obj
        exported.append(entity.id)

    doc = {
        "type": "CityJSON",
        "version": CITYJSON_VERSION,
        "transform": {
            "scale": [_VERTEX_SCALE, _VERTEX_SCALE, _VERTEX_SCALE],
            "translate": list(translate),
        },
        "CityObjects": city_objects,
        "vertices": vertices,
    }

    canonical = json.dumps(doc, sort_keys=True)
    report = ExportReport(
        format="cityjson",
        world_id=world.id,
        world_version=world.version,
        entities_exported=tuple(exported),
        entities_skipped=tuple(skipped),
        skip_reasons=tuple(reasons),
        content_hash=content_hash(canonical),
    )
    return doc, report
