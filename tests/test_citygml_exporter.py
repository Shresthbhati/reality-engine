"""Tests for the CityGML exporter (exporters/citygml/, P14-01/P16-01:
the CityGML writer tracked alongside the CityJSON exporter).

Mirrors tests/test_cityjson_exporter.py's contract for the same
WorldIR geometry, since both exporters consume identical box/plane
bounds.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR
from exporters.citygml.exporter import (
    export_to_citygml,
    export_to_citygml_with_report,
)

_NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gen": "http://www.opengis.net/citygml/generics/2.0",
    "gml": "http://www.opengis.net/gml",
}


def _world() -> WorldIR:
    world = WorldIR()
    wall = Geometry(
        id="geom-wall",
        type=GeometryType.PLANE,
        bounds_min=Vector3(-2.0, 0.0, 0.0),
        bounds_max=Vector3(2.0, 0.2, 3.0),
        provenance=Provenance.OBSERVED,
    )
    world.geometries["geom-wall"] = wall
    world.entities["wall-north"] = Entity(
        id="wall-north",
        name="Wall North",
        type=EntityType.WALL,
        transform={"position": {"x": 10.0, "y": 5.0, "z": 0.0}},
        geometry_ids=["geom-wall"],
        provenance=Provenance.OBSERVED,
    )
    building = Geometry(
        id="geom-bld",
        type=GeometryType.BOX,
        bounds_min=Vector3(-1.0, -1.0, 0.0),
        bounds_max=Vector3(1.0, 1.0, 4.0),
    )
    world.geometries["geom-bld"] = building
    world.entities["bld-1"] = Entity(
        id="bld-1",
        name="Block A",
        type=EntityType.BUILDING,
        transform={"position": {"x": 50.0, "y": 50.0, "z": 0.0}},
        geometry_ids=["geom-bld"],
    )
    # No transform -> skipped, with a reason (checked via the report).
    world.geometries["geom-orphan"] = Geometry(
        id="geom-orphan", type=GeometryType.BOX)
    world.entities["orphan"] = Entity(
        id="orphan", type=EntityType.DEBRIS,
        geometry_ids=["geom-orphan"])
    return world


class TestCityGmlStructure:
    def test_document_is_valid_xml_with_citygml_namespaces(self):
        xml = export_to_citygml(_world())
        root = ET.fromstring(xml)
        assert root.tag == f"{{{_NS['core']}}}CityModel"

    def test_city_object_members_keyed_by_entity_id(self):
        root = ET.fromstring(export_to_citygml(_world()))
        members = root.findall(f"{{{_NS['core']}}}cityObjectMember")
        ids = set()
        for member in members:
            child = list(member)[0]
            ids.add(child.get(f"{{{_NS['gml']}}}id"))
        # The transform-less orphan is skipped, not emitted as an
        # empty object.
        assert ids == {"wall-north", "bld-1"}

    def test_semantic_type_mapping(self):
        root = ET.fromstring(export_to_citygml(_world()))
        tags = {}
        for member in root.findall(f"{{{_NS['core']}}}cityObjectMember"):
            child = list(member)[0]
            tags[child.get(f"{{{_NS['gml']}}}id")] = child.tag
        assert tags["bld-1"] == f"{{{_NS['bldg']}}}Building"
        # WALL has no first-class CityGML building type -> GenericCityObject.
        assert tags["wall-north"] == f"{{{_NS['gen']}}}GenericCityObject"

    def test_geometry_is_real_lod1_solid_with_six_faces(self):
        root = ET.fromstring(export_to_citygml(_world()))
        for member in root.findall(f"{{{_NS['core']}}}cityObjectMember"):
            child = list(member)[0]
            if child.get(f"{{{_NS['gml']}}}id") != "bld-1":
                continue
            solid = child.find(f".//{{{_NS['gml']}}}Solid")
            assert solid is not None
            surfaces = solid.findall(f".//{{{_NS['gml']}}}surfaceMember")
            assert len(surfaces) == 6  # a real AABB has 6 quad faces

    def test_worldir_attributes_are_traceable(self):
        root = ET.fromstring(export_to_citygml(_world()))
        for member in root.findall(f"{{{_NS['core']}}}cityObjectMember"):
            child = list(member)[0]
            if child.get(f"{{{_NS['gml']}}}id") != "wall-north":
                continue
            attrs = {
                a.get("name"): a.findtext(f"{{{_NS['gen']}}}value")
                for a in child.findall(f"{{{_NS['gen']}}}stringAttribute")
            }
            assert attrs["worldir_type"] == "wall"
            assert attrs["worldir_provenance"] == Provenance.OBSERVED.value

    def test_determinism(self):
        world = _world()
        assert export_to_citygml(world) == export_to_citygml(world)


class TestCityGmlReport:
    def test_report_lists_exported_and_skipped(self):
        _xml, report = export_to_citygml_with_report(_world())
        assert set(report.entities_exported) == {"wall-north", "bld-1"}
        assert report.entities_skipped == ("orphan",)
        assert "no transform" in report.skip_reasons[0]

    def test_report_hash_matches_document(self):
        xml, report = export_to_citygml_with_report(_world())
        from exporters.report import content_hash
        assert report.content_hash == content_hash(xml)

    def test_report_format_field(self):
        _xml, report = export_to_citygml_with_report(_world())
        assert report.format == "citygml"

    def test_geometryless_entity_is_skipped_with_reason(self):
        world = _world()
        world.entities["no-geom"] = Entity(
            id="no-geom",
            type=EntityType.STRUCTURE,
            transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        )
        _xml, report = export_to_citygml_with_report(world)
        idx = report.entities_skipped.index("no-geom")
        assert "no BOX/PLANE geometry" in report.skip_reasons[idx]
