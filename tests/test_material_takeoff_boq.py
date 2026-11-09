"""Tests for Automated Architectural Bill of Quantities (BoQ) & Material Takeoff (MTO).

Verifies:
1. Exact geometric net surface area calculations with opening deductions.
2. Baseboard trim perimeter with door threshold deductions.
3. CSI MasterFormat 2020 division categorizations and summaries.
4. Custom rate and waste factor overrides.
5. Zero-dependency PDF-1.4 certificate and tabular CSV exports.
6. FastAPI REST API endpoints for takeoff retrieval, estimation, and downloads.
"""

from __future__ import annotations

import io
import uuid
import pytest

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR
from engine.pipeline.material_takeoff import (
    CsiDivision,
    DEFAULT_UNIT_RATES,
    TakeoffReport,
    compute_world_material_takeoff,
    export_takeoff_to_csv,
    export_takeoff_to_pdf,
)


def _build_takeoff_test_world() -> WorldIR:
    """Create a sample world with rooms, walls, openings, and fixtures."""
    world = WorldIR(id="world-takeoff-test")

    # Room 1: Office (5m x 4m = 20 m2, perimeter = 18m, height = 2.7m)
    # Gross wall = 18 * 2.7 = 48.6 m2
    # Openings: 1 door (0.9 x 2.1 = 1.89 m2), 1 window (1.2 x 1.5 = 1.8 m2) -> Total 3.69 m2
    # Net wall = 48.6 - 3.69 = 44.91 m2
    # Baseboard trim = 18 - 0.9 = 17.1 m
    room1 = Entity(
        id="room-office-1",
        name="Executive Office 101",
        type=EntityType.ROOM,
        custom_properties={
            "footprint_polygon_m": [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]],
            "floor_area_m2": 20.0,
            "height_m": 2.7,
            "floor_finish": "hardwood_oak",
            "wall_finish": "matte_paint",
            "ceiling_finish": "acoustic_tile",
            "walls": [
                {
                    "length_m": 5.0,
                    "height_m": 2.7,
                    "openings": [{"kind": "door", "width_m": 0.9, "height_m": 2.1}],
                },
                {
                    "length_m": 4.0,
                    "height_m": 2.7,
                    "openings": [{"kind": "window", "width_m": 1.2, "height_m": 1.5}],
                },
                {"length_m": 5.0, "height_m": 2.7, "openings": []},
                {"length_m": 4.0, "height_m": 2.7, "openings": []},
            ],
        },
    )
    world.entities[room1.id] = room1

    # Room 2: Restroom (3m x 2m = 6 m2, perimeter = 10m, height = 2.7m)
    # Gross wall = 10 * 2.7 = 27.0 m2
    # Openings: 1 door (0.9 x 2.1 = 1.89 m2)
    # Net wall = 27.0 - 1.89 = 25.11 m2
    room2 = Entity(
        id="room-wc-1",
        name="Restroom 102",
        type=EntityType.ROOM,
        custom_properties={
            "footprint_polygon_m": [[5.0, 0.0], [8.0, 0.0], [8.0, 2.0], [5.0, 2.0]],
            "floor_area_m2": 6.0,
            "height_m": 2.7,
            "floor_finish": "ceramic_tile",
            "wall_finish": "ceramic_tile",
            "ceiling_finish": "gypsum_board",
            "walls": [
                {
                    "length_m": 3.0,
                    "height_m": 2.7,
                    "openings": [{"kind": "door", "width_m": 0.9, "height_m": 2.1}],
                },
                {"length_m": 2.0, "height_m": 2.7, "openings": []},
                {"length_m": 3.0, "height_m": 2.7, "openings": []},
                {"length_m": 2.0, "height_m": 2.7, "openings": []},
            ],
        },
    )
    world.entities[room2.id] = room2

    # Add canonical EntityType.WALL entities for room 1
    w1_specs = [
        ("wall-off-1", [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], 5.0, [{"kind": "door", "width_m": 0.9, "height_m": 2.1, "offset_m": 1.0}]),
        ("wall-off-2", [5.0, 0.0, 0.0], [5.0, 4.0, 0.0], 4.0, [{"kind": "window", "width_m": 1.2, "height_m": 1.5, "offset_m": 1.0}]),
        ("wall-off-3", [5.0, 4.0, 0.0], [0.0, 4.0, 0.0], 5.0, []),
        ("wall-off-4", [0.0, 4.0, 0.0], [0.0, 0.0, 0.0], 4.0, []),
    ]
    for wid, start, end, length, ops in w1_specs:
        world.entities[wid] = Entity(
            id=wid,
            name=f"Wall {wid}",
            type=EntityType.WALL,
            custom_properties={
                "room_id": room1.id,
                "length_m": length,
                "height_m": 2.7,
                "wall_slab": {
                    "start_xyz_m": start,
                    "end_xyz_m": end,
                    "openings": ops,
                },
            },
        )

    # Add canonical EntityType.WALL entities for room 2
    w2_specs = [
        ("wall-wc-1", [5.0, 0.0, 0.0], [8.0, 0.0, 0.0], 3.0, [{"kind": "door", "width_m": 0.9, "height_m": 2.1, "offset_m": 0.5}]),
        ("wall-wc-2", [8.0, 0.0, 0.0], [8.0, 2.0, 0.0], 2.0, []),
        ("wall-wc-3", [8.0, 2.0, 0.0], [5.0, 2.0, 0.0], 3.0, []),
        ("wall-wc-4", [5.0, 2.0, 0.0], [5.0, 0.0, 0.0], 2.0, []),
    ]
    for wid, start, end, length, ops in w2_specs:
        world.entities[wid] = Entity(
            id=wid,
            name=f"Wall {wid}",
            type=EntityType.WALL,
            custom_properties={
                "room_id": room2.id,
                "length_m": length,
                "height_m": 2.7,
                "wall_slab": {
                    "start_xyz_m": start,
                    "end_xyz_m": end,
                    "openings": ops,
                },
            },
        )

    return world


def test_material_takeoff_calculation():
    """Verify net area deductions, trim perimeters, and quantity calculations."""
    w = _build_takeoff_test_world()
    report = compute_world_material_takeoff(w, project_name="Downtown Office Suite")

    assert report.world_id == "world-takeoff-test"
    assert report.project_name == "Downtown Office Suite"
    assert len(report.rooms) == 2

    # Verify Office (Room 1)
    office = next(r for r in report.rooms if r.room_id == "room-office-1")
    assert office.floor_area_m2 == pytest.approx(20.0, 0.01)
    assert office.perimeter_m == pytest.approx(18.0, 0.01)
    assert office.wall_gross_area_m2 == pytest.approx(48.6, 0.01)
    assert office.wall_opening_area_m2 == pytest.approx(3.69, 0.01)
    assert office.wall_net_area_m2 == pytest.approx(44.91, 0.01)
    assert office.baseboard_length_m == pytest.approx(17.1, 0.01)  # 18.0 - 0.9m door
    assert office.doors_count == 1
    assert office.windows_count == 1

    # Verify Restroom (Room 2)
    wc = next(r for r in report.rooms if r.room_id == "room-wc-1")
    assert wc.floor_area_m2 == pytest.approx(6.0, 0.01)
    assert wc.perimeter_m == pytest.approx(10.0, 0.01)
    assert wc.wall_net_area_m2 == pytest.approx(25.11, 0.01)
    assert wc.baseboard_length_m == pytest.approx(9.1, 0.01)  # 10.0 - 0.9m door
    assert wc.doors_count == 1
    assert wc.windows_count == 0

    # Total area metrics
    assert report.total_floor_area_m2 == pytest.approx(26.0, 0.01)
    assert report.total_net_wall_area_m2 == pytest.approx(70.02, 0.01)
    assert report.total_openings_count == 3  # 2 doors + 1 window


def test_csi_masterformat_divisions_and_totals():
    """Verify CSI MasterFormat 2020 division mappings and cost totals."""
    w = _build_takeoff_test_world()
    report = compute_world_material_takeoff(w)

    assert len(report.divisions) >= 4
    div_names = [d.division_name for d in report.divisions]
    assert any("06 - Wood" in d for d in div_names)
    assert any("08 - Openings" in d for d in div_names)
    assert any("09 - Finishes" in d for d in div_names)
    assert any("26 - Electrical" in d for d in div_names)

    # Verify percentages sum to 100%
    total_pct = sum(d.percentage_of_total for d in report.divisions)
    assert total_pct == pytest.approx(100.0, 0.1)

    # Cost consistency: material + labor == grand total
    assert report.grand_total_cost == pytest.approx(
        report.grand_total_material + report.grand_total_labor, 0.01
    )
    assert report.grand_total_cost > 1000.0
    assert report.cost_per_m2 > 50.0


def test_custom_unit_rates_and_waste_overrides():
    """Verify that user-defined rate overrides correctly adjust financial totals."""
    w = _build_takeoff_test_world()

    # Base report
    base_report = compute_world_material_takeoff(w)

    # Override drywall (09 29 00) rate and waste
    overrides = {
        "09 29 00": {"material_cost": 30.0, "labor_cost": 40.0, "waste_percent": 15.0}
    }
    custom_report = compute_world_material_takeoff(w, rate_overrides=overrides)

    assert custom_report.grand_total_cost > base_report.grand_total_cost

    # Find drywall item in custom report
    gyp_items = [i for i in custom_report.all_items if i.csi_code == "09 29 00"]
    assert len(gyp_items) >= 1
    assert gyp_items[0].unit_cost_material == 30.0
    assert gyp_items[0].unit_cost_labor == 40.0
    assert gyp_items[0].waste_percent == 15.0


def test_csv_and_pdf_exports():
    """Verify CSV and zero-dependency PDF-1.4 file output."""
    w = _build_takeoff_test_world()
    report = compute_world_material_takeoff(w, project_name="Metropolitan Tower Suite")

    # 1. CSV Export
    csv_text = export_takeoff_to_csv(report)
    assert "Reality Engine - Architectural Bill of Quantities" in csv_text
    assert "Metropolitan Tower Suite" in csv_text
    assert "09 29 00" in csv_text
    assert "08 14 16" in csv_text
    lines = csv_text.strip().split("\n")
    assert len(lines) >= 15

    # 2. PDF Export
    pdf_bytes = export_takeoff_to_pdf(report, project_name="Metropolitan Tower Suite")
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-1.4\n")
    assert b"%%EOF" in pdf_bytes
    assert b"xref\n" in pdf_bytes
    assert b"Metropolitan Tower Suite" in pdf_bytes
    assert b"CSI MasterFormat 2020" in pdf_bytes


@pytest.mark.asyncio
async def test_takeoff_api_endpoints():
    """Verify HTTP REST API endpoints for takeoff retrieval, custom estimate, and export."""
    from starlette.testclient import TestClient
    from apps.api.main import app
    from apps.api.db import get_sessionmaker, init_db
    from apps.api.models import World, WorldVersion
    from apps.api.worldstore_service import get_store

    client = TestClient(app)

    world_id = f"wld_takeoff_{uuid.uuid4().hex[:8]}"
    w = _build_takeoff_test_world()
    w.id = world_id

    store = get_store()
    sv = store.save_version(w, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        world_row = World(
            id=w.id,
            name="Commercial Office Level 4",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(world_row)

        ver_row = WorldVersion(
            id=sv.version_id,
            world_id=w.id,
            label="Base Version for Takeoff",
            report={},
        )
        session.add(ver_row)
        await session.commit()

    # 1. GET /api/worlds/{id}/takeoff
    res_get = client.get(f"/api/worlds/{w.id}/takeoff")
    assert res_get.status_code == 200, res_get.text
    data = res_get.json()
    assert data["world_id"] == w.id
    assert "summary" in data
    assert data["summary"]["total_floor_area_m2"] == pytest.approx(26.0, 0.1)
    assert len(data["divisions"]) >= 4
    assert len(data["rooms"]) == 2

    # 2. POST /api/worlds/{id}/takeoff/estimate
    res_est = client.post(
        f"/api/worlds/{w.id}/takeoff/estimate",
        json={
            "project_name": "Custom Cost Scenario",
            "rate_overrides": {
                "09 64 00": {"material_cost": 80.0, "labor_cost": 50.0}
            },
        },
    )
    assert res_est.status_code == 200, res_est.text
    est_data = res_est.json()
    assert est_data["project_name"] == "Custom Cost Scenario"
    assert est_data["summary"]["grand_total_cost"] > data["summary"]["grand_total_cost"]

    # 3. GET /api/worlds/{id}/takeoff/export/csv
    res_csv = client.get(f"/api/worlds/{w.id}/takeoff/export/csv")
    assert res_csv.status_code == 200, res_csv.text
    assert "CSI Division,CSI Code,Description" in res_csv.text

    # 4. GET /api/worlds/{id}/takeoff/export/pdf
    res_pdf = client.get(f"/api/worlds/{w.id}/takeoff/export/pdf")
    assert res_pdf.status_code == 200, res_pdf.text
    assert res_pdf.content.startswith(b"%PDF-1.4\n")
    assert b"%%EOF" in res_pdf.content
