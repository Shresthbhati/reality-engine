"""T-2.3: the room schedule as CSV and PDF. Every number must equal the room report's (one source of truth)."""
import csv
import io

import pytest

from exporters.floorplan import build_room_report, export_to_room_schedule_csv_with_report, report_to_pdf
from tests.test_floorplan_export import _world
from world_ir.schema_v1 import EntityType


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_one_row_per_room_with_numbers_equal_to_the_room_report():
    w = _world()
    text, report = export_to_room_schedule_csv_with_report(w)
    rows = _rows(text)
    rep = build_room_report(w)["rooms"]
    assert len(rows) == len(rep) >= 1
    for row, r in zip(rows, rep):
        assert row["room"] == r["name"] and row["room_id"] == r["id"]
        assert float(row["area_m2"]) == pytest.approx(r["area_m2"])
        assert float(row["perimeter_m"]) == pytest.approx(r["perimeter_m"])
        assert float(row["wall_area_net_m2"]) == pytest.approx(r["wall_area_net_m2"])
        assert int(row["walls"]) == r["walls"]
        assert int(row["fixtures"]) == r["fixtures"]
        assert row["floor_finish"] == r["floor_finish"]
        assert row["wall_finish"] == r["wall_finish"]
        assert row["ceiling_finish"] == r["ceiling_finish"]
    assert report.format == "room_schedule_csv" and report.content_hash


def test_units_and_scale_state_are_stated_not_assumed():
    text, _ = export_to_room_schedule_csv_with_report(_world())
    row = _rows(text)[0]
    assert row["units"] and row["scale_state"]


def test_openings_and_finishes_and_fixtures_are_in_their_own_columns():
    text, _ = export_to_room_schedule_csv_with_report(_world())
    cols = _rows(text)[0].keys()
    assert {
        "windows", "doors", "opening_area_m2", "ceiling_height_m", "confidence_mean", "inferred_walls",
        "fixtures", "floor_finish", "wall_finish", "ceiling_finish",
    } <= set(cols)


def test_room_fixtures_and_finishes_roundtrip_accurately():
    w = _world()
    room = next(e for e in w.entities.values() if e.type == EntityType.ROOM)
    room.custom_properties["floor_finish"] = "hardwood_oak"
    room.custom_properties["wall_finish"] = "matte_paint_white"
    room.custom_properties["ceiling_finish"] = "smooth_plaster"

    # Add a fixture assigned to this room
    from world_ir.schema_v1 import Entity, Geometry, GeometryType, Vector3
    fixture = Entity(
        id="fixture_sink_01",
        name="Bathroom Sink",
        type=EntityType.UNKNOWN,
        geometry_ids=["geom_sink"],
        transform={"position": {"x": 2.0, "y": 2.0, "z": 0.8}},
        custom_properties={"room_id": room.id, "entity_class": "plumbing.sink"},
    )
    w.entities[fixture.id] = fixture
    w.geometries["geom_sink"] = Geometry(
        id="geom_sink",
        type=GeometryType.BOX,
        bounds_min=Vector3(x=-0.25, y=-0.25, z=0.0),
        bounds_max=Vector3(x=0.25, y=0.25, z=0.5),
    )

    rep = build_room_report(w)
    r_dict = next(r for r in rep["rooms"] if r["id"] == room.id)
    assert r_dict["fixtures"] >= 1
    assert "plumbing.sink" in r_dict["fixture_counts"]
    assert r_dict["floor_finish"] == "hardwood_oak"
    assert r_dict["wall_finish"] == "matte_paint_white"
    assert r_dict["ceiling_finish"] == "smooth_plaster"

    text, _ = export_to_room_schedule_csv_with_report(w)
    row = next(r for r in _rows(text) if r["room_id"] == room.id)
    assert int(row["fixtures"]) >= 1
    assert row["floor_finish"] == "hardwood_oak"
    assert row["wall_finish"] == "matte_paint_white"
    assert row["ceiling_finish"] == "smooth_plaster"

    # Test PDF generation includes fixtures and finishes
    pdf_bytes = report_to_pdf(rep)
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"Fixtures:" in pdf_bytes
    assert b"Finishes" in pdf_bytes


def test_a_world_without_a_room_is_refused_like_the_other_floor_plan_exports():
    from world_ir import WorldIR

    with pytest.raises(ValueError):
        export_to_room_schedule_csv_with_report(WorldIR())
