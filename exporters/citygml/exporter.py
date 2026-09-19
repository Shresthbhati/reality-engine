"""CityGML 2.0 exporter for WorldIR (P14-01/P16-01: the CityGML writer
tracked as open alongside the CityJSON exporter).

Mirrors exporters/cityjson/exporter.py's honesty rules exactly, since
both consume the same WorldIR BOX/PLANE geometry:

  - Real CityGML 2.0 XML: a <CityModel> document in the citygml/2.0
    namespace with <cityObjectMember> per entity, parseable by
    standard CityGML tooling (not a lookalike).
  - Semantic mapping is honest: EntityType.BUILDING maps to
    <bldg:Building>; types without a CityGML building/generics
    counterpart export as <gen:GenericCityObject> with the WorldIR
    type recorded as a <gen:stringAttribute name="worldir_type">.
  - Geometry is REAL: the entity's BOX/PLANE bounds become an LoD1
    <gml:Solid> made of the same 6 quad faces (as
    <gml:CompositeSurface>) computed from the entity's world-space
    AABB corners -- identical corner/face math to the CityJSON
    exporter (exporters/cityjson/exporter.py), so both writers agree
    on what "the geometry" is. Entities without real bounds are
    skipped with a reason, never given fabricated geometry.
  - Deterministic: iterates world.entities in insertion order and
    emits fixed gml:id patterns, so the same world produces
    byte-identical XML.

Not implemented in this pass (recorded, not hidden): LoD0/2/3/4,
semantic surfaces (WallSurface/RoofSurface), textures/appearances,
CRS/srsName (WorldIR carries no CRS), 3DCityDB database export (that
is a database target, not a file format -- out of scope for a file
exporter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from exporters.report import ExportReport, content_hash
from world_ir.schema_v1 import EntityType, GeometryType

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR

#: Same exportable geometry set as the CityJSON exporter.
_EXPORTABLE_GEOMETRY_TYPES = frozenset({GeometryType.BOX, GeometryType.PLANE})

#: Same degenerate-axis floor as the CityJSON/Blender exporters.
_MIN_DIMENSION_M = 0.01

#: WorldIR EntityType -> CityGML element (namespace, local name).
#: Everything else becomes gen:GenericCityObject.
_ENTITY_TYPE_MAP = {
    EntityType.BUILDING.value: ("bldg", "Building"),
}

#: The 6 quad faces of an AABB, referencing the same 8-corner
#: enumeration order as exporters/cityjson/exporter.py's
#: _world_corners() (x fastest, then y, then z).
_FACES = [
    [0, 2, 3, 1],  # y = 0
    [4, 5, 7, 6],  # y = 1
    [0, 1, 5, 4],  # x = 0
    [2, 6, 7, 3],  # x = 1
    [0, 4, 6, 2],  # z = 0
    [1, 3, 7, 5],  # z = 1
]

_NAMESPACES = (
    'xmlns:core="http://www.opengis.net/citygml/2.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
    'xmlns:gen="http://www.opengis.net/citygml/generics/2.0" '
    'xmlns:gml="http://www.opengis.net/gml"'
)


def _entity_geometry(world: "WorldIR", entity):
    """The first BOX/PLANE Geometry attached to `entity`, or None
    (identical rule to the CityJSON exporter)."""
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.type in _EXPORTABLE_GEOMETRY_TYPES:
            return geom
    return None


def _world_corners(geom, position: dict):
    """The 8 world-space AABB corners of `geom` placed at `position`.
    Identical math to exporters/cityjson/exporter.py._world_corners so
    both exporters agree on the entity's real geometry."""
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


def _poslist(ring_corners) -> str:
    """A closed gml:posList (first point repeated at the end, per the
    GML linear-ring rule) for one face's 4 corners."""
    pts = list(ring_corners) + [ring_corners[0]]
    return " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts)


def _solid_xml(entity_id: str, corners) -> str:
    """A <gml:Solid> body built from the 6 real AABB faces --
    identical faces to the CityJSON exporter. Caller wraps this in the
    right lodNSolid/lodNGeometry property element."""
    surfaces = []
    for face_idx, face in enumerate(_FACES):
        ring = [corners[ci] for ci in face]
        surfaces.append(
            f'<gml:surfaceMember><gml:Polygon gml:id="{entity_id}-poly-{face_idx}">'
            f'<gml:exterior><gml:LinearRing><gml:posList>'
            f'{_poslist(ring)}'
            f'</gml:posList></gml:LinearRing></gml:exterior>'
            f'</gml:Polygon></gml:surfaceMember>'
        )
    return (
        f'<gml:Solid gml:id="{entity_id}-solid"><gml:exterior>'
        f'<gml:CompositeSurface>{"".join(surfaces)}</gml:CompositeSurface>'
        f'</gml:exterior></gml:Solid>'
    )


def _city_object_xml(entity, corners) -> str:
    ns, _tag = _ENTITY_TYPE_MAP.get(entity.type, (None, None))
    solid = _solid_xml(entity.id, corners)
    attrs = (
        f'<gen:stringAttribute name="worldir_type"><gen:value>{escape(entity.type)}</gen:value></gen:stringAttribute>'
        f'<gen:stringAttribute name="worldir_provenance"><gen:value>{escape(entity.provenance.value)}</gen:value></gen:stringAttribute>'
        f'<gen:doubleAttribute name="worldir_confidence"><gen:value>{entity.confidence}</gen:value></gen:doubleAttribute>'
    )
    name_xml = f'<gml:name>{escape(entity.name)}</gml:name>' if entity.name else ""
    if ns == "bldg":
        body = f'{name_xml}{attrs}<bldg:lod1Solid>{solid}</bldg:lod1Solid>'
        element = f'<bldg:Building gml:id="{entity.id}">{body}</bldg:Building>'
    else:
        body = f'{name_xml}{attrs}<gen:lod1Geometry>{solid}</gen:lod1Geometry>'
        element = f'<gen:GenericCityObject gml:id="{entity.id}">{body}</gen:GenericCityObject>'
    return f'<core:cityObjectMember>{element}</core:cityObjectMember>'


def _iter_exportable(world: "WorldIR"):
    """Yields (entity, corners) for every entity with a placed,
    real-bounded BOX/PLANE geometry. Skip reasons are only tracked by
    the *_with_report path."""
    for entity in world.entities.values():
        if not entity.transform or "position" not in entity.transform:
            continue
        geom = _entity_geometry(world, entity)
        if geom is None:
            continue
        corners = _world_corners(geom, entity.transform["position"])
        if corners is None:
            continue
        yield entity, corners


def export_to_citygml(world: "WorldIR") -> str:
    """Build a real CityGML 2.0 XML document (as a string) from
    `world`. See the module docstring for exactly what is exported,
    what is skipped and why."""
    members = [_city_object_xml(entity, corners) for entity, corners in _iter_exportable(world)]
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<core:CityModel {_NAMESPACES}>'
        f'{"".join(members)}'
        '</core:CityModel>'
    )


def export_to_citygml_with_report(world: "WorldIR") -> tuple:
    """Same document as export_to_citygml(), plus a structured
    ExportReport with the same skip-reason discipline as the CityJSON,
    glTF, USD and Blender exporters."""
    members = []
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
            reasons.append("no BOX/PLANE geometry with real bounds attached")
            continue
        corners = _world_corners(geom, entity.transform["position"])
        if corners is None:
            skipped.append(entity.id)
            reasons.append("geometry has no real bounds_min/bounds_max")
            continue
        members.append(_city_object_xml(entity, corners))
        exported.append(entity.id)

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<core:CityModel {_NAMESPACES}>'
        f'{"".join(members)}'
        '</core:CityModel>'
    )
    report = ExportReport(
        format="citygml",
        world_id=world.id,
        world_version=world.version,
        entities_exported=tuple(exported),
        entities_skipped=tuple(skipped),
        skip_reasons=tuple(reasons),
        content_hash=content_hash(xml),
    )
    return xml, report
