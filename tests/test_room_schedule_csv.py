"""T-2.3: the room schedule as CSV. Every number must equal the room report's (one source of truth)."""
import csv
import io

import pytest

from exporters.floorplan import build_room_report, export_to_room_schedule_csv_with_report
from tests.test_floorplan_export import _world


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
    assert report.format == "room_schedule_csv" and report.content_hash


def test_units_and_scale_state_are_stated_not_assumed():
    text, _ = export_to_room_schedule_csv_with_report(_world())
    row = _rows(text)[0]
    assert row["units"] and row["scale_state"]


def test_openings_are_counted_by_kind_in_their_own_columns():
    text, _ = export_to_room_schedule_csv_with_report(_world())
    cols = _rows(text)[0].keys()
    assert {"windows", "doors", "opening_area_m2", "ceiling_height_m", "confidence_mean", "inferred_walls"} <= set(cols)


def test_a_world_without_a_room_is_refused_like_the_other_floor_plan_exports():
    from world_ir import WorldIR

    with pytest.raises(ValueError):
        export_to_room_schedule_csv_with_report(WorldIR())
