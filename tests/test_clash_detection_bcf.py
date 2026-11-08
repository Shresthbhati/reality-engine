"""Tests for Automated Issue & Clash Detection with BCF (BIM Collaboration Format) Engine."""

from __future__ import annotations

import io
import uuid
import zipfile
import pytest
from fastapi.testclient import TestClient

from apps.api.db import get_sessionmaker, init_db
from apps.api.main import app
from apps.api.models import World, WorldVersion
from apps.api.worldstore_service import get_store
from engine.pipeline.clash_detection import (
    ClashDetectionOptions,
    ClashDetectionReport,
    ClashItem,
    ClashSeverity,
    ClashStatus,
    ClashType,
    detect_world_clashes,
    generate_bcf_zip,
    parse_bcf_zip,
)
from engine.pipeline.temporal_delta import TradeCategory
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def test_hard_clash_detection():
    """Verify detection of a physical geometric penetration between MEP and Structure."""
    w = WorldIR(id=f"wld_clash_{uuid.uuid4().hex[:8]}")

    # Concrete Structural Column at [2.0, 1.5, 2.0]
    col = Entity(
        id="col_c1",
        name="Structural Concrete Column C1",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 2.0, "y": 1.5, "z": 2.0}},
        custom_properties={"width_m": 0.5, "depth_m": 0.5, "height_m": 3.0, "trade": "structure"},
    )

    # HVAC Duct passing through the column: center [2.1, 2.5, 2.0], penetrating by 0.4m
    duct = Entity(
        id="duct_m1",
        name="Main Supply Air Duct",
        type=EntityType.INFRASTRUCTURE,
        transform={"position": {"x": 2.1, "y": 2.5, "z": 2.0}},
        custom_properties={"width_m": 0.6, "depth_m": 0.4, "height_m": 0.4, "trade": "mep"},
    )

    w.entities[col.id] = col
    w.entities[duct.id] = duct

    report = detect_world_clashes(w, world_id=w.id)
    assert report.total_clashes == 1
    assert report.hard_clashes == 1

    clash = report.items[0]
    assert clash.clash_type == ClashType.HARD
    assert clash.severity in (ClashSeverity.CRITICAL, ClashSeverity.MAJOR)
    assert clash.trade_a == TradeCategory.STRUCTURE or clash.trade_b == TradeCategory.STRUCTURE
    assert clash.trade_b == TradeCategory.MEP or clash.trade_a == TradeCategory.MEP
    assert clash.penetration_depth_m > 0.05
    assert clash.intersection_volume_m3 > 0.001
    assert clash.viewpoint is not None
    assert len(clash.viewpoint.camera_position) == 3


def test_clearance_clash_detection():
    """Verify clearance buffer violation when separation distance is below tolerance."""
    w = WorldIR(id=f"wld_clearance_{uuid.uuid4().hex[:8]}")

    # Wall at [0, 1.5, 0], bounds [-2.0..2.0, 0..3.0, -0.1..0.1]
    wall = Entity(
        id="wall_w1",
        name="Interior Partition Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 1.5, "z": 0.0}},
        custom_properties={"width_m": 4.0, "depth_m": 0.2, "height_m": 3.0, "trade": "structure"},
    )

    # Electrical Panel placed 0.02m away (clearance buffer is 0.05m)
    # Wall max Z is 0.1, panel center Z is 0.22 with depth 0.2 (min Z = 0.12), gap = 0.02m
    panel = Entity(
        id="panel_p1",
        name="Sub-Distribution Board",
        type=EntityType.INFRASTRUCTURE,
        transform={"position": {"x": 0.0, "y": 1.5, "z": 0.22}},
        custom_properties={"width_m": 0.8, "depth_m": 0.2, "height_m": 1.2, "trade": "mep"},
    )

    w.entities[wall.id] = wall
    w.entities[panel.id] = panel

    opts = ClashDetectionOptions(hard_clash_tolerance_m=0.01, clearance_buffer_m=0.05)
    report = detect_world_clashes(w, world_id=w.id, options=opts)

    assert report.clearance_clashes == 1
    clash = report.items[0]
    assert clash.clash_type == ClashType.CLEARANCE
    assert 0.015 < clash.clearance_distance_m < 0.025


def test_duplicate_element_detection():
    """Verify detection of duplicate modeled elements occupying identical space."""
    w = WorldIR(id=f"wld_dup_{uuid.uuid4().hex[:8]}")

    col1 = Entity(
        id="col_orig",
        name="Structural Column 101",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 5.0, "y": 1.5, "z": 5.0}},
        custom_properties={"width_m": 0.4, "depth_m": 0.4, "height_m": 3.0, "trade": "structure"},
    )

    col2 = Entity(
        id="col_copy",
        name="Structural Column 101 Copy",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 5.0, "y": 1.5, "z": 5.0}},
        custom_properties={"width_m": 0.4, "depth_m": 0.4, "height_m": 3.0, "trade": "structure"},
    )

    w.entities[col1.id] = col1
    w.entities[col2.id] = col2

    report = detect_world_clashes(w, world_id=w.id)
    assert report.duplicate_clashes == 1
    assert report.items[0].clash_type == ClashType.DUPLICATE


def test_bcf_zip_package_generation_and_parsing():
    """Verify compliance of generated BCF 2.1 zip package and roundtrip parsing."""
    report = ClashDetectionReport(
        world_id="world_bcf_test_123",
        timestamp_iso="2026-11-09T00:00:00Z",
        total_clashes=1,
        hard_clashes=1,
        clearance_clashes=0,
        duplicate_clashes=0,
        by_severity={"critical": 1, "major": 0, "minor": 0, "info": 0},
        by_trade_pair={"structure-mep": 1},
        items=[
            ClashItem(
                id=str(uuid.uuid4()),
                clash_type=ClashType.HARD,
                severity=ClashSeverity.CRITICAL,
                status=ClashStatus.ACTIVE,
                entity_a_id="col_01",
                entity_b_id="duct_01",
                entity_a_name="Column A",
                entity_b_name="Duct B",
                trade_a=TradeCategory.STRUCTURE,
                trade_b=TradeCategory.MEP,
                penetration_depth_m=0.15,
                intersection_volume_m3=0.035,
                clash_point=[2.0, 1.5, 3.0],
                bounds_min=[1.8, 1.3, 2.8],
                bounds_max=[2.2, 1.7, 3.2],
                viewpoint=None,
                title="Critical Clash: Column A & Duct B",
                description="HVAC duct penetrates structural column without designated sleeve opening.",
                assigned_to="HVAC Contractor",
                created_at_iso="2026-11-09T00:00:00Z",
            )
        ],
    )

    # Generate BCF zip
    bcf_bytes = generate_bcf_zip(report, author="Reality Engine Clash Perceive")
    assert len(bcf_bytes) > 100

    # Inspect zip structure
    with zipfile.ZipFile(io.BytesIO(bcf_bytes), mode="r") as zf:
        names = zf.namelist()
        assert "bcf.version" in names
        topic_id = report.items[0].id
        assert f"{topic_id}/markup.bcf" in names
        assert f"{topic_id}/viewpoint.bcfv" in names
        assert f"{topic_id}/snapshot.png" in names

        # Verify bcf.version content
        ver_xml = zf.read("bcf.version").decode("utf-8")
        assert 'VersionId="2.1"' in ver_xml

        # Verify markup.bcf content
        markup_xml = zf.read(f"{topic_id}/markup.bcf").decode("utf-8")
        assert 'TopicType="Hard"' in markup_xml
        assert "Critical Clash: Column A &amp; Duct B" in markup_xml

    # Test roundtrip parse
    topics = parse_bcf_zip(bcf_bytes)
    assert len(topics) == 1
    assert topics[0]["title"] == "Critical Clash: Column A & Duct B"
    assert topics[0]["priority"] == "Critical"
    assert "col_01" in topics[0]["components"]
    assert "duct_01" in topics[0]["components"]


@pytest.mark.asyncio
async def test_clash_and_bcf_api_endpoints():
    """Verify HTTP endpoints for clash detection and BCF export/import."""
    client = TestClient(app)

    # 1. Create a World with clashing entities in WorldStore & SQLite DB
    w = WorldIR(id=f"wld_api_clash_{uuid.uuid4().hex[:8]}")
    beam = Entity(
        id="beam_b1",
        name="Reinforced Concrete Beam",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 0.0, "y": 3.0, "z": 0.0}},
        custom_properties={"width_m": 6.0, "depth_m": 0.4, "height_m": 0.6, "trade": "structure"},
    )
    pipe = Entity(
        id="pipe_p1",
        name="Hydronic Heating Pipe",
        type=EntityType.INFRASTRUCTURE,
        transform={"position": {"x": 1.0, "y": 3.0, "z": 0.0}},
        custom_properties={"width_m": 0.2, "depth_m": 2.0, "height_m": 0.2, "trade": "mep"},
    )
    w.entities[beam.id] = beam
    w.entities[pipe.id] = pipe

    store = get_store()
    sv = store.save_version(w, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        world_row = World(
            id=w.id,
            name="Clash Test Facility",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(world_row)

        ver_row = WorldVersion(
            id=sv.version_id,
            world_id=w.id,
            label="Version with Clash",
        )
        session.add(ver_row)
        await session.commit()

    # 2. Test GET /api/worlds/{id}/clashes
    res_clashes = client.get(f"/api/worlds/{w.id}/clashes")
    assert res_clashes.status_code == 200
    clash_data = res_clashes.json()
    assert clash_data["world_id"] == w.id
    assert clash_data["total_clashes"] == 1
    assert clash_data["hard_clashes"] == 1
    assert len(clash_data["items"]) == 1
    assert clash_data["items"][0]["entity_a_id"] == beam.id or clash_data["items"][0]["entity_b_id"] == beam.id

    # 3. Test POST /api/worlds/{id}/clashes/detect with custom tolerances
    res_detect = client.post(
        f"/api/worlds/{w.id}/clashes/detect",
        json={"hard_tolerance_m": 0.02, "clearance_buffer_m": 0.1, "check_duplicates": True},
    )
    assert res_detect.status_code == 200
    assert res_detect.json()["total_clashes"] == 1

    # 4. Test GET /api/worlds/{id}/bcf/export (Download BCF 2.1 zip)
    res_export = client.get(f"/api/worlds/{w.id}/bcf/export")
    assert res_export.status_code == 200
    assert res_export.headers["content-type"] == "application/zip"
    bcf_content = res_export.content
    assert len(bcf_content) > 100

    # 5. Test POST /api/worlds/{id}/bcf/import (Upload BCF zip)
    res_import = client.post(
        f"/api/worlds/{w.id}/bcf/import",
        files={"file": ("clashes.bcfzip", bcf_content, "application/zip")},
    )
    assert res_import.status_code == 200
    import_data = res_import.json()
    assert import_data["imported_topic_count"] == 1
    assert len(import_data["topics"]) == 1
