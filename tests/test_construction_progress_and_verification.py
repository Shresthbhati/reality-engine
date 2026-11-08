"""Comprehensive tests for Construction Progress & As-Built vs. Plan Verification (Option A)."""

from __future__ import annotations

import io
from datetime import datetime, timezone
import pytest
from starlette.testclient import TestClient

from apps.api.main import app
from apps.api.models import World, WorldVersion, utcnow
from apps.api.worldstore_service import get_store
from engine.pipeline.construction_progress import (
    ComplianceStatus,
    SiteComplianceReport,
    Trade,
    compute_construction_timeline,
    export_construction_progress_pdf,
    infer_trade,
    verify_as_built_against_plan,
)
from world_ir.schema_v1 import Entity, EntityType
from world_ir.temporal import TemporalEpoch
from world_ir.world_v1 import WorldIR


def _make_sample_design_world() -> WorldIR:
    """Build a synthetic planned BIM/design world with structural, architectural, and MEP elements."""
    world = WorldIR(id="wld_design_plan")

    # Structure
    col1 = Entity(
        id="col_01",
        name="Structural Column C1",
        type=EntityType.COLUMN,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 1.5}},
        custom_properties={"trade": "structure", "tolerance_m": 0.03},
    )
    col2 = Entity(
        id="col_02",
        name="Structural Column C2",
        type=EntityType.COLUMN,
        transform={"position": {"x": 4.0, "y": 0.0, "z": 1.5}},
        custom_properties={"trade": "structure", "tolerance_m": 0.03},
    )
    slab = Entity(
        id="slab_01",
        name="Ground Concrete Slab",
        type=EntityType.FLOOR,
        transform={"position": {"x": 2.0, "y": 2.0, "z": 0.0}},
        custom_properties={"trade": "structure"},
    )

    # Architecture
    wall_north = Entity(
        id="wall_n",
        name="North Exterior Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 2.0, "y": 4.0, "z": 1.5}},
        custom_properties={"trade": "architecture", "tolerance_m": 0.05},
    )
    door_main = Entity(
        id="door_01",
        name="Main Entry Door",
        type=EntityType.DOOR,
        transform={"position": {"x": 1.0, "y": 4.0, "z": 1.0}},
        custom_properties={"trade": "architecture", "tolerance_m": 0.02},
    )

    # MEP
    socket_01 = Entity(
        id="socket_01",
        name="Power Socket North",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 3.0, "y": 3.95, "z": 0.4}},
        custom_properties={"trade": "mep", "tolerance_m": 0.05},
    )

    world.entities[col1.id] = col1
    world.entities[col2.id] = col2
    world.entities[slab.id] = slab
    world.entities[wall_north.id] = wall_north
    world.entities[door_main.id] = door_main
    world.entities[socket_01.id] = socket_01
    return world


def test_infer_trade():
    """Verify accurate construction trade inference across structural, architectural, and MEP entities."""
    col = Entity(id="1", name="Column", type=EntityType.COLUMN)
    assert infer_trade(col) == Trade.STRUCTURE

    wall = Entity(id="2", name="Drywall Partition", type=EntityType.WALL)
    assert infer_trade(wall) == Trade.ARCHITECTURE

    door = Entity(id="3", name="Entry Door", type=EntityType.DOOR)
    assert infer_trade(door) == Trade.ARCHITECTURE

    socket = Entity(id="4", name="Duplex Wall Socket", type=EntityType.STRUCTURE)
    assert infer_trade(socket) == Trade.MEP

    switch = Entity(id="5", name="Light Switch Double", type=EntityType.STRUCTURE, custom_properties={"trade": "mep"})
    assert infer_trade(switch) == Trade.MEP

    finish = Entity(id="6", name="Oak Hardwood Floor Finish", type=EntityType.STRUCTURE)
    assert infer_trade(finish) == Trade.FINISHES


def test_verify_plan_identical_worlds():
    """When design plan matches as-built capture identically, 100% compliance is reported."""
    design = _make_sample_design_world()
    site = _make_sample_design_world()

    report = verify_as_built_against_plan(design, site, default_tolerance_m=0.05)

    assert report.overall_completion_pct == 100.0
    assert len(report.completed_elements) == 6
    assert len(report.misaligned_elements) == 0
    assert len(report.missing_elements) == 0
    assert len(report.unplanned_elements) == 0

    trades = report.trade_summaries()
    assert trades[Trade.STRUCTURE].completion_pct == 100.0
    assert trades[Trade.ARCHITECTURE].completion_pct == 100.0
    assert trades[Trade.MEP].completion_pct == 100.0


def test_verify_plan_misaligned_element():
    """Elements shifted beyond acceptable tolerance are flagged as MISALIGNED with deviation metrics."""
    design = _make_sample_design_world()
    site = _make_sample_design_world()

    # Displace North wall by 9 cm (tolerance is 5 cm)
    shifted_wall = Entity(
        id="wall_n",
        name="North Exterior Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 2.0, "y": 4.09, "z": 1.5}},
        custom_properties={"trade": "architecture"},
    )
    site.entities[shifted_wall.id] = shifted_wall

    report = verify_as_built_against_plan(design, site, default_tolerance_m=0.05, auto_align=False)

    misaligned = [e for e in report.elements if e.status == ComplianceStatus.MISALIGNED]
    assert len(misaligned) == 1
    assert misaligned[0].element_id == "wall_n"
    assert pytest.approx(misaligned[0].deviation_m, abs=0.005) == 0.09
    assert misaligned[0].trade == Trade.ARCHITECTURE
    assert "Tolerance exceeded" in misaligned[0].notes

    # Overall completion percentage should reflect only 5/6 = 83.33%
    assert pytest.approx(report.overall_completion_pct, abs=0.1) == 83.33


def test_verify_plan_missing_and_unplanned():
    """Missing design elements and unplanned site elements are accurately categorized."""
    design = _make_sample_design_world()
    site = WorldIR(id="wld_site_partial")

    # Site only has columns and slab, missing door and socket, plus an unplanned dumpster/generator
    for eid in ("col_01", "col_02", "slab_01"):
        site.entities[eid] = design.entities[eid]

    unplanned_box = Entity(
        id="temp_generator",
        name="Temporary Site Generator",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 10.0, "y": 10.0, "z": 0.5}},
    )
    site.entities[unplanned_box.id] = unplanned_box

    report = verify_as_built_against_plan(design, site, default_tolerance_m=0.05)

    # 3 completed, 3 missing (wall_n, door_01, socket_01), 1 unplanned
    assert len(report.completed_elements) == 3
    assert len(report.missing_elements) == 3
    assert len(report.unplanned_elements) == 1
    assert report.unplanned_elements[0].element_id == "temp_generator"

    # Structure should be 100%, Architecture 0%, MEP 0%
    trades = report.trade_summaries()
    assert trades[Trade.STRUCTURE].completion_pct == 100.0
    assert trades[Trade.ARCHITECTURE].completion_pct == 0.0
    assert trades[Trade.MEP].completion_pct == 0.0


def test_verify_plan_in_progress_rough_framing():
    """Elements in rough framing or incomplete states are categorized as IN_PROGRESS."""
    design = _make_sample_design_world()
    site = _make_sample_design_world()

    rough_wall = Entity(
        id="wall_n",
        name="North Exterior Wall (Rough Framing)",
        type=EntityType.WALL,
        transform={"position": {"x": 2.0, "y": 4.0, "z": 1.5}},
        custom_properties={"trade": "architecture", "rough": True},
    )
    site.entities[rough_wall.id] = rough_wall

    report = verify_as_built_against_plan(design, site, default_tolerance_m=0.05)
    in_prog = [e for e in report.elements if e.status == ComplianceStatus.IN_PROGRESS]
    assert len(in_prog) == 1
    assert in_prog[0].element_id == "wall_n"
    assert "In progress" in in_prog[0].notes


def test_compute_construction_timeline_and_rework():
    """Multi-epoch timeline computes progress velocity and detects rework/demolition."""
    design = _make_sample_design_world()

    # Epoch 0: Foundation only (columns + slab)
    e0_world = WorldIR(id="wld_e0")
    for eid in ("col_01", "col_02", "slab_01"):
        e0_world.entities[eid] = design.entities[eid]
    e0 = TemporalEpoch("ep_0", "2026-01-01T08:00:00Z", e0_world)

    # Epoch 1: Framing complete (walls and door added)
    e1_world = WorldIR(id="wld_e1")
    for eid in ("col_01", "col_02", "slab_01", "wall_n", "door_01"):
        e1_world.entities[eid] = design.entities[eid]
    e1 = TemporalEpoch("ep_1", "2026-01-15T08:00:00Z", e1_world)

    # Epoch 2: MEP complete, but door was removed for rework
    e2_world = WorldIR(id="wld_e2")
    for eid in ("col_01", "col_02", "slab_01", "wall_n", "socket_01"):
        e2_world.entities[eid] = design.entities[eid]
    e2 = TemporalEpoch("ep_2", "2026-01-30T08:00:00Z", e2_world)

    timeline = compute_construction_timeline([e0, e1, e2], design, tolerance_m=0.05)

    assert len(timeline.milestones) == 3
    # Epoch 0: 3 of 6 = 50%
    assert timeline.milestones[0].overall_completion_pct == 50.0
    # Epoch 1: 5 of 6 = 83.33%
    assert pytest.approx(timeline.milestones[1].overall_completion_pct, abs=0.1) == 83.33
    assert timeline.milestones[1].net_velocity_elements_per_day > 0

    # Epoch 2: Door was completed in e1, now missing in e2 -> rework detected!
    assert timeline.rework_count >= 1
    assert any("door_01" in detail for detail in timeline.rework_details)


def test_export_construction_progress_pdf():
    """Pure-Python zero-dependency PDF generation produces valid PDF-1.4 binary data."""
    design = _make_sample_design_world()
    site = _make_sample_design_world()
    # Introduce one misalignment
    shifted = Entity(
        id="wall_n",
        name="North Wall Shifted",
        type=EntityType.WALL,
        transform={"position": {"x": 2.0, "y": 4.12, "z": 1.5}},
    )
    site.entities[shifted.id] = shifted

    report = verify_as_built_against_plan(design, site, default_tolerance_m=0.05)
    pdf_bytes = export_construction_progress_pdf(report, project_name="Metro Commercial Tower 4")

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-1.4\n")
    assert b"%%EOF\n" in pdf_bytes
    assert b"Metro Commercial Tower 4" in pdf_bytes
    assert b"Overall Progress" in pdf_bytes
    assert b"STRUCTURE" in pdf_bytes
    assert b"ARCHITECTURE" in pdf_bytes


@pytest.mark.asyncio
async def test_construction_api_endpoints():
    """Test FastAPI REST endpoints: /api/construction/verify, /timeline, /report/pdf."""
    client = TestClient(app)

    # 1. Create design and site worlds in SQLite + WorldStore
    store = get_store()
    design_wir = _make_sample_design_world()
    site_wir = _make_sample_design_world()
    site_wir.id = "wld_site_epoch_1" 

    # Save to WorldStore
    sv_design = store.save_version(design_wir, parent=None)
    sv_site = store.save_version(site_wir, parent=None)

    # Register World rows in SQLite
    from apps.api.db import get_sessionmaker, init_db
    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        w_design = World(
            id=design_wir.id,
            name="Design Model A",
            current_version_id=sv_design.version_id,
            status="active",
        )
        w_site = World(
            id=site_wir.id,
            name="Site Scan Epoch 1",
            current_version_id=sv_site.version_id,
            status="active",
        )
        session.add(w_design)
        session.add(w_site)
        await session.commit()

    # 2. Test POST /api/construction/verify
    res_verify = client.post(
        "/api/construction/verify",
        json={
            "design_world_id": design_wir.id,
            "site_world_id": site_wir.id,
            "tolerance_m": 0.05,
            "auto_align": True,
        },
    )
    assert res_verify.status_code == 200, res_verify.text
    data = res_verify.json()
    assert "summary" in data
    assert data["summary"]["overall_completion_pct"] == 100.0
    assert data["summary"]["completed_count"] == 6

    # 3. Test POST /api/construction/timeline
    res_timeline = client.post(
        "/api/construction/timeline",
        json={
            "design_world_id": design_wir.id,
            "epoch_world_ids": [site_wir.id],
            "tolerance_m": 0.05,
        },
    )
    assert res_timeline.status_code == 200, res_timeline.text
    t_data = res_timeline.json()
    assert "milestones" in t_data
    assert len(t_data["milestones"]) == 1
    assert t_data["milestones"][0]["overall_completion_pct"] == 100.0

    # 4. Test POST /api/construction/report/pdf
    res_pdf = client.post(
        "/api/construction/report/pdf",
        json={
            "design_world_id": design_wir.id,
            "site_world_id": site_wir.id,
            "project_name": "Hospital Wing B As-Built Inspection",
            "tolerance_m": 0.05,
        },
    )
    assert res_pdf.status_code == 200, res_pdf.text
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert res_pdf.content.startswith(b"%PDF-1.4\n")
    assert len(res_pdf.content) > 500
