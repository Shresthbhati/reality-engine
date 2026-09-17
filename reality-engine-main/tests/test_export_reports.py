"""Tests for the shared structured export report (exporters/report.py)
and its wiring into all three exporters."""

from __future__ import annotations

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR

from exporters.report import content_hash
from exporters.gltf.exporter import export_to_gltf_with_report
from exporters.usd.exporter import export_to_usda_with_report
from exporters.blender.exporter import export_to_blender_script_with_report


def _mixed_world() -> WorldIR:
    world = WorldIR(id="w-report-test", version=3)

    box_geom = Geometry(id="geom-box", type=GeometryType.BOX)
    exportable = Entity(
        id="ent-exportable", name="Crate", type=EntityType.DEBRIS,
        transform={"position": {"x": 1.0, "y": 0.0, "z": 0.0}}, geometry_ids=[box_geom.id],
    )

    pc_geom = Geometry(id="geom-pc", type=GeometryType.POINTCLOUD, vertex_count=100)
    no_real_geometry = Entity(
        id="ent-no-geometry", name="Scan",
        transform={"position": {"x": 2.0, "y": 0.0, "z": 0.0}}, geometry_ids=[pc_geom.id],
    )

    no_transform = Entity(id="ent-no-transform", name="Untethered", geometry_ids=[box_geom.id])

    world.geometries[box_geom.id] = box_geom
    world.geometries[pc_geom.id] = pc_geom
    world.entities[exportable.id] = exportable
    world.entities[no_real_geometry.id] = no_real_geometry
    world.entities[no_transform.id] = no_transform
    return world


def test_content_hash_deterministic_and_sensitive_to_content():
    assert content_hash("same") == content_hash("same")
    assert content_hash("a") != content_hash("b")
    assert content_hash(b"bytes") == content_hash(b"bytes")


def test_gltf_report_classifies_exported_and_skipped_entities():
    world = _mixed_world()
    gltf, report = export_to_gltf_with_report(world)

    assert report.format == "gltf"
    assert report.world_id == "w-report-test"
    assert report.world_version == 3
    assert report.entities_exported == ("ent-exportable",)
    assert report.entities_skipped == ("ent-no-geometry", "ent-no-transform")
    assert report.skip_reasons == (
        "no BOX/PLANE geometry to export",
        "no transform.position to place a node at",
    )
    assert len(gltf["nodes"]) == 1


def test_usd_report_classifies_exported_and_skipped_entities():
    world = _mixed_world()
    usda, report = export_to_usda_with_report(world)

    assert report.format == "usda"
    assert report.entities_exported == ("ent-exportable",)
    assert len(report.entities_skipped) == 2
    assert usda.count("def Cube") == 1


def test_blender_report_classifies_exported_and_skipped_entities():
    world = _mixed_world()
    script, report = export_to_blender_script_with_report(world)

    assert report.format == "blender-script"
    assert report.entities_exported == ("ent-exportable",)
    assert len(report.entities_skipped) == 2
    assert script.count("primitive_cube_add") == 1


def test_report_hash_matches_actual_content_and_is_reproducible():
    world = _mixed_world()
    usda_1, report_1 = export_to_usda_with_report(world)
    usda_2, report_2 = export_to_usda_with_report(world)

    assert report_1.content_hash == report_2.content_hash
    assert report_1.content_hash == content_hash(usda_1)
    assert usda_1 == usda_2


def test_report_hash_changes_when_content_changes():
    world_a = _mixed_world()
    world_b = _mixed_world()
    world_b.entities["ent-exportable"].transform = {"position": {"x": 99.0, "y": 0.0, "z": 0.0}}

    _, report_a = export_to_usda_with_report(world_a)
    _, report_b = export_to_usda_with_report(world_b)
    assert report_a.content_hash != report_b.content_hash


def test_report_to_dict_is_plain_data_and_lengths_match():
    world = _mixed_world()
    _, report = export_to_gltf_with_report(world)
    payload = report.to_dict()
    assert len(payload["entities_skipped"]) == len(payload["skip_reasons"])
    assert payload["format"] == "gltf"


def test_report_rejects_mismatched_skip_lengths():
    from exporters.report import ExportReport
    try:
        ExportReport(
            format="gltf", world_id="w", world_version=1,
            entities_exported=(), entities_skipped=("a", "b"), skip_reasons=("only one",),
            content_hash="x",
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_empty_world_report_has_no_exports_no_skips():
    world = WorldIR()
    _, report = export_to_gltf_with_report(world)
    assert report.entities_exported == ()
    assert report.entities_skipped == ()
