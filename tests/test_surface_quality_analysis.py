"""Tests for Surface Quality Analysis Engine: ASTM E1155 Floor Flatness & Beam Deflection."""

from __future__ import annotations

import math
from typing import List, Tuple
import uuid
import pytest
from starlette.testclient import TestClient

from apps.api.db import get_sessionmaker, init_db
from apps.api.main import app
from apps.api.models import World, WorldVersion
from apps.api.worldstore_service import get_store
from engine.pipeline.surface_quality import (
    ASTMClassification,
    DeflectionCode,
    FloorFlatnessReport,
    BeamDeflectionReport,
    calculate_astm_e1155_slab,
    calculate_beam_deflection,
    generate_surface_deviation_heatmap,
    analyze_world_surface_quality,
    export_surface_quality_certificate_pdf,
)
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def test_astm_e1155_flat_slab_calculation():
    """Verify ASTM E1155 calculation on a high-precision flat slab."""
    bounds = (0.0, 6.0, 0.0, 6.0)  # 6m x 6m slab
    points: List[Tuple[float, float, float]] = []

    # Dense points with sub-millimeter noise (< 0.5 mm)
    for x_step in range(21):
        for z_step in range(21):
            x = x_step * 0.3
            z = z_step * 0.3
            # Flat elevation around Y=0.0 with tiny micro-variation
            y = 0.0 + 0.0003 * math.sin(x) * math.cos(z)
            points.append((x, y, z))

    report = calculate_astm_e1155_slab(
        points=points,
        bounds=bounds,
        entity_id="slab-test-1",
        surface_name="Executive Office Slab",
        spacing_m=0.30,
        target_ff=25.0,
        target_fl=20.0,
    )

    assert report.entity_id == "slab-test-1"
    assert report.sample_count == len(points)
    assert report.area_sqm == 36.0
    assert report.overall_ff >= 35.0
    assert report.overall_fl >= 25.0
    assert report.classification in (ASTMClassification.FLAT, ASTMClassification.VERY_FLAT, ASTMClassification.SUPER_FLAT)
    assert report.compliance_passed is True
    assert len(report.runs) > 0
    assert abs(report.mean_elevation_m) < 0.01


def test_astm_e1155_wavy_non_compliant_slab():
    """Verify detection of wavy, out-of-tolerance concrete floor slab."""
    bounds = (0.0, 6.0, 0.0, 6.0)
    points: List[Tuple[float, float, float]] = []

    # Significant multi-axis wave with 25 mm ripple
    for x_step in range(21):
        for z_step in range(21):
            x = x_step * 0.3
            z = z_step * 0.3
            y = 0.0 + 0.025 * math.sin(x * 3.0) + 0.025 * math.cos(z * 3.0)
            points.append((x, y, z))

    report = calculate_astm_e1155_slab(
        points=points,
        bounds=bounds,
        target_ff=35.0,
        target_fl=25.0,
    )

    assert report.overall_ff < 25.0
    assert report.classification in (ASTMClassification.CONVENTIONAL, ASTMClassification.NON_COMPLIANT)
    assert report.compliance_passed is False
    assert report.max_high_spot_mm > 15.0


def test_beam_deflection_calculation():
    """Verify structural deflection and code compliance calculation on beams."""
    # 6.0 meter span
    start_pos = (0.0, 3.5, 0.0)
    end_pos = (6.0, 3.5, 0.0)
    span = 6.0

    # 1. Compliant beam with 10 mm deflection (Ratio = 6000 / 10 = 600 -> L/600, passes L/360)
    pts_compliant: List[Tuple[float, float, float]] = []
    for step in range(61):
        s = step * 0.1
        # Parabolic sag: max 10 mm at midspan
        sag = -0.010 * 4.0 * (s / span) * (1.0 - s / span)
        pts_compliant.append((s, 3.5 + sag, 0.0))

    rep_pass = calculate_beam_deflection(
        points=pts_compliant,
        start_pos=start_pos,
        end_pos=end_pos,
        entity_id="beam-pass",
        beam_name="Girder G-1",
        target_code=DeflectionCode.STANDARD_L360,
    )
    assert rep_pass.is_compliant is True
    assert rep_pass.deflection_ratio >= 360.0
    assert rep_pass.achieved_code in (DeflectionCode.STRICT_L480, DeflectionCode.STANDARD_L360)
    assert abs(rep_pass.max_deflection_mm - 10.0) < 1.0

    # 2. Failing beam with 35 mm deflection (Ratio = 6000 / 35 = 171 -> L/171 < L/240)
    pts_failing: List[Tuple[float, float, float]] = []
    for step in range(61):
        s = step * 0.1
        sag = -0.035 * 4.0 * (s / span) * (1.0 - s / span)
        pts_failing.append((s, 3.5 + sag, 0.0))

    rep_fail = calculate_beam_deflection(
        points=pts_failing,
        start_pos=start_pos,
        end_pos=end_pos,
        entity_id="beam-fail",
        beam_name="Sagging Beam B-4",
        target_code=DeflectionCode.STANDARD_L360,
    )
    assert rep_fail.is_compliant is False
    assert rep_fail.achieved_code == DeflectionCode.EXCESSIVE
    assert rep_fail.max_deflection_mm > 30.0


def test_surface_deviation_heatmap_generation():
    """Verify point-to-plane surface deviation heatmap with diverging color scale."""
    bounds = (-2.0, 2.0, -2.0, 2.0)
    ref_y = 0.0
    points: List[Tuple[float, float, float]] = []

    for x_i in range(11):
        for z_i in range(11):
            x = -2.0 + x_i * 0.4
            z = -2.0 + z_i * 0.4
            # Slope from -10 mm to +10 mm
            y = (x / 2.0) * 0.010
            points.append((x, y, z))

    heatmap = generate_surface_deviation_heatmap(
        points=points,
        reference_elevation_m=ref_y,
        bounds=bounds,
        grid_resolution_m=0.40,
        max_deviation_threshold_mm=25.0,
    )

    assert heatmap.point_count > 0
    assert len(heatmap.points) == heatmap.point_count
    assert heatmap.min_deviation_mm <= -5.0
    assert heatmap.max_deviation_mm >= 5.0

    # Verify color codes
    for pt in heatmap.points:
        assert pt.color_hex.startswith("#")
        assert len(pt.color_hex) == 7
        assert len(pt.color_rgb) == 3
        assert all(0.0 <= c <= 1.0 for c in pt.color_rgb)


def test_surface_quality_certificate_pdf():
    """Verify zero-dependency PDF compliance certificate generation."""
    world = WorldIR(id="w-cert-1")
    world.entities["f1"] = Entity(
        id="f1",
        type=EntityType.FLOOR,
        name="Main Concourse Slab",
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width": 10.0, "depth": 10.0},
    )
    world.entities["b1"] = Entity(
        id="b1",
        type=EntityType.BEAM,
        name="Transfer Girder G-101",
        transform={"position": {"x": 0.0, "y": 4.0, "z": 0.0}},
        custom_properties={"length": 8.0, "direction": "X"},
    )

    report = analyze_world_surface_quality(world=world)
    pdf_bytes = export_surface_quality_certificate_pdf(report, "Terminal 3 Surface Quality Audit")

    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf_bytes
    assert len(pdf_bytes) > 1000
    assert b"ASTM E1155" in pdf_bytes


@pytest.mark.asyncio
async def test_surface_quality_api_endpoints():
    """Verify REST API endpoints for surface quality analysis, heatmap, and PDF download."""
    client = TestClient(app)

    world_id = f"wld_sq_{uuid.uuid4().hex[:8]}"
    w = WorldIR(id=world_id)
    w.entities["flr-1"] = Entity(
        id="flr-1",
        type=EntityType.FLOOR,
        name="Lobby Slab Level 1",
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width": 8.0, "depth": 8.0},
    )
    w.entities["bm-1"] = Entity(
        id="bm-1",
        type=EntityType.BEAM,
        name="Primary Spine Beam B-1",
        transform={"position": {"x": 0.0, "y": 3.5, "z": 0.0}},
        custom_properties={"length": 6.0, "direction": "X"},
    )

    store = get_store()
    sv = store.save_version(w, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        world_row = World(
            id=w.id,
            name="Audit World Test",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(world_row)

        ver_row = WorldVersion(
            id=sv.version_id,
            world_id=w.id,
            label="Version for Surface Audit",
            report={},
        )
        session.add(ver_row)
        await session.commit()

    # 1. GET surface quality
    res_get = client.get(f"/api/worlds/{w.id}/surface-quality")
    assert res_get.status_code == 200, res_get.text
    data = res_get.json()
    assert data["world_id"] == w.id
    assert len(data["floors"]) >= 1
    assert len(data["beams"]) >= 1
    assert "overall_compliance" in data

    # 2. POST analyze
    res_post = client.post(
        f"/api/worlds/{w.id}/surface-quality/analyze",
        json={"target_ff": 30.0, "target_fl": 20.0, "target_deflection_code": "L/360"},
    )
    assert res_post.status_code == 200, res_post.text
    assert res_post.json()["total_surfaces_analyzed"] >= 2

    # 3. GET heatmap
    res_hm = client.get(f"/api/worlds/{w.id}/surface-quality/heatmap")
    assert res_hm.status_code == 200, res_hm.text
    hm_data = res_hm.json()
    assert "point_count" in hm_data
    assert "points" in hm_data
    assert "min_deviation_mm" in hm_data

    # 4. GET PDF certificate
    res_pdf = client.get(f"/api/worlds/{w.id}/surface-quality/certificate")
    assert res_pdf.status_code == 200, res_pdf.text
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert res_pdf.content.startswith(b"%PDF-1.4")
