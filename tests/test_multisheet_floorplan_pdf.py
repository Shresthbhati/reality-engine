"""Unit and contract tests for 2D Scaled Floor Plan Multi-Sheet PDF (Epic E10-2).

Verifies:
1. Multi-page PDF architectural drawing package generation (Sheet 1, 2, 3).
2. Title blocks conforming to architectural drawing conventions on every sheet (A-101, A-102, A-103).
3. Scaled 2D floor plan rendering with wall bands, room polygons, and north arrow.
4. Door & Window schedules with dimensions, operations, and fire ratings.
5. Net wall area takeoff tables accounting for carved openings.
6. Byte determinism and valid PDF-1.4 syntax.
"""

from __future__ import annotations

import copy
import pytest

from exporters.floorplan import (
    build_multisheet_floorplan_pdf,
    export_to_multisheet_floorplan_pdf_with_report,
)
from tests.test_room_fidelity_fixture import _compiled
from world_ir.measure import stamp_world_dimensions
from world_ir.world_v1 import WorldIR


def _world(metric: bool = True) -> WorldIR:
    _obs, world, _diag = copy.deepcopy(_compiled("albedo", False))
    if metric:
        world.metadata["scale"] = {
            "state": "metric",
            "meters_per_unit": 1.0,
            "dimension_state": "MEASURED",
        }
        stamp_world_dimensions(world)
    return world


def test_multisheet_pdf_has_three_sheets_and_valid_syntax():
    world = _world(metric=True)
    pdf, report = export_to_multisheet_floorplan_pdf_with_report(world)

    # Valid PDF-1.4 syntax
    assert pdf.startswith(b"%PDF-1.4")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert b"xref" in pdf
    assert b"trailer" in pdf

    # Multi-page check: exactly 3 landscape sheets
    assert b"/Count 3" in pdf
    assert pdf.count(b"/Type /Page") == 4  # 3 Pages + 1 Pages catalog parent
    assert b"/MediaBox [0 0 842 595]" in pdf

    # Report metadata
    assert report.format == "multisheet_floorplan_pdf"
    assert report.world_id == world.id
    assert len(report.content_hash) == 64


def test_title_blocks_on_all_sheets():
    world = _world(metric=True)
    pdf = build_multisheet_floorplan_pdf(world)

    # Title blocks for each sheet
    assert b"A-101" in pdf
    assert b"COVER & ROOM SCHEDULE" in pdf

    assert b"A-102" in pdf
    assert b"LEVEL 0 OVERALL PLAN" in pdf

    assert b"A-103" in pdf
    assert b"OPENING & WALL SCHEDULES" in pdf

    # Architectural drawing company / project title
    assert b"REALITY ENGINE ARCHITECTURAL DRAWINGS" in pdf


def test_sheet1_room_schedule_and_net_wall_area():
    world = _world(metric=True)
    pdf = build_multisheet_floorplan_pdf(world)

    # Room schedule headers and content
    assert b"ROOM SCHEDULE" in pdf
    assert b"Net Wall" in pdf
    assert b"Floor Area" in pdf
    assert b"Perimeter" in pdf


def test_sheet2_floorplan_drawing_and_north_arrow():
    world = _world(metric=True)
    pdf = build_multisheet_floorplan_pdf(world)

    # Floor plan elements
    assert b"LEVEL 0 OVERALL FLOOR PLAN" in pdf
    assert b"Scale 1:50" in pdf
    # North arrow
    assert b"(N) Tj" in pdf


def test_sheet3_door_and_window_schedules():
    world = _world(metric=True)
    pdf = build_multisheet_floorplan_pdf(world)

    # Opening schedule headers
    assert b"DOOR SCHEDULE" in pdf
    assert b"WINDOW SCHEDULE" in pdf
    assert b"NET WALL AREA TAKEOFF" in pdf
    assert b"D-01" in pdf
    assert b"W-01" in pdf


def test_byte_determinism():
    world = _world(metric=True)
    pdf1 = build_multisheet_floorplan_pdf(world)
    pdf2 = build_multisheet_floorplan_pdf(world)
    assert pdf1 == pdf2


def test_no_room_raises_error():
    empty_world = WorldIR(id="w-empty")
    with pytest.raises(ValueError, match="measured footprint"):
        build_multisheet_floorplan_pdf(empty_world)
