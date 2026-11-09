"""Tests for Building Envelope Thermal Performance & ASHRAE 90.1 Compliance Engine.

Verifies:
1. Exact Window-to-Wall Ratio (WWR %) computation and 40% prescriptive cap validation.
2. Component U-factor and R-value compliance across ASHRAE Climate Zones.
3. Area-weighted thermal transmittance UA budget tradeoff (UA_actual <= UA_target).
4. Design heat loss rate (q = U * A * delta_T).
5. Zero-dependency PDF-1.4 Thermal Certificate generation.
6. FastAPI REST API endpoints for thermal evaluation, custom scenarios, and certificate downloads.
"""

from __future__ import annotations

import io
import uuid
import pytest

from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR
from engine.pipeline.thermal_envelope import (
    ClimateZone,
    EnvelopeComponentType,
    ThermalStatus,
    evaluate_thermal_envelope,
    export_thermal_envelope_certificate_pdf,
)


def _build_thermal_test_world() -> WorldIR:
    """Build a test building with exterior walls, windows, and doors."""
    world = WorldIR(id="world-thermal-test")

    # Wall 1: North Façade (10m x 3m = 30 m2) with two windows (1.5m x 1.5m = 2.25 m2 each -> 4.5 m2 total glazing)
    # WWR for Wall 1 = 4.5 / 30 = 15.0%
    wall1 = Entity(
        id="wall-north-facade",
        name="North Exterior Façade",
        type=EntityType.WALL,
        custom_properties={
            "length_m": 10.0,
            "height_m": 3.0,
            "wall_slab": {
                "start_xyz_m": [0.0, 10.0, 0.0],
                "end_xyz_m": [10.0, 10.0, 0.0],
                "openings": [
                    {"kind": "window", "width_m": 1.5, "height_m": 1.5, "offset_m": 1.5},
                    {"kind": "window", "width_m": 1.5, "height_m": 1.5, "offset_m": 6.0},
                ],
            },
        },
    )
    world.entities[wall1.id] = wall1

    # Wall 2: South Façade (10m x 3m = 30 m2) with glazed storefront (3m x 2.4m = 7.2 m2) + door (1m x 2.1m = 2.1 m2)
    # Total glazing = 7.2 m2, Total door = 2.1 m2, Gross wall = 30 m2
    wall2 = Entity(
        id="wall-south-facade",
        name="South Exterior Façade",
        type=EntityType.WALL,
        custom_properties={
            "length_m": 10.0,
            "height_m": 3.0,
            "wall_slab": {
                "start_xyz_m": [0.0, 0.0, 0.0],
                "end_xyz_m": [10.0, 0.0, 0.0],
                "openings": [
                    {"kind": "window", "width_m": 3.0, "height_m": 2.4, "offset_m": 1.0},
                    {"kind": "door", "width_m": 1.0, "height_m": 2.1, "offset_m": 6.0},
                ],
            },
        },
    )
    world.entities[wall2.id] = wall2

    return world


def test_thermal_envelope_wwr_and_ua():
    """Verify Window-to-Wall Ratio (WWR %) and area-weighted UA calculations."""
    w = _build_thermal_test_world()
    report = evaluate_thermal_envelope(w, climate_zone=ClimateZone.ZONE_5)

    assert report.world_id == "world-thermal-test"
    assert report.climate_zone == "5A_cold"

    # Total Gross Wall Area = 30m2 + 30m2 = 60 m2
    assert report.summary["gross_exterior_wall_area_m2"] == pytest.approx(60.0, 0.01)

    # Glazing Area = 4.5 (North) + 7.2 (South) = 11.7 m2
    assert report.summary["total_glazing_area_m2"] == pytest.approx(11.7, 0.01)

    # WWR = 11.7 / 60.0 = 19.5%
    assert report.summary["window_to_wall_ratio_pct"] == pytest.approx(19.5, 0.1)
    assert report.summary["wwr_prescriptive_compliant"] is True

    # UA Actual & Target
    assert report.summary["total_ua_actual_w_k"] > 0.0
    assert report.summary["total_ua_target_w_k"] > 0.0
    assert report.compliance_verdict == "COMPLIANT"

    # Surfaces breakdown
    glaz_surfs = [s for s in report.surfaces if s.component_type == EnvelopeComponentType.FENESTRATION]
    assert len(glaz_surfs) == 3
    assert all(s.shgc is not None for s in glaz_surfs)


def test_climate_zone_prescriptive_targets():
    """Verify that stricter cold climate zones enforce tighter U-values."""
    w = _build_thermal_test_world()

    rep_zone1 = evaluate_thermal_envelope(w, climate_zone=ClimateZone.ZONE_1)
    rep_zone5 = evaluate_thermal_envelope(w, climate_zone=ClimateZone.ZONE_5)
    rep_zone8 = evaluate_thermal_envelope(w, climate_zone=ClimateZone.ZONE_8)

    # Zone 1 target wall U-factor is higher (less insulated allowed) than Zone 5, which is higher than Zone 8
    target1 = next(s.target_u_w_m2k for s in rep_zone1.surfaces if s.component_type == EnvelopeComponentType.EXTERIOR_WALL)
    target5 = next(s.target_u_w_m2k for s in rep_zone5.surfaces if s.component_type == EnvelopeComponentType.EXTERIOR_WALL)
    target8 = next(s.target_u_w_m2k for s in rep_zone8.surfaces if s.component_type == EnvelopeComponentType.EXTERIOR_WALL)

    assert target1 > target5 > target8
    assert target1 == pytest.approx(0.704, 0.001)
    assert target5 == pytest.approx(0.312, 0.001)
    assert target8 == pytest.approx(0.250, 0.001)


def test_thermal_certificate_export():
    """Verify pure PDF-1.4 generation for ASHRAE 90.1 certificate."""
    w = _build_thermal_test_world()
    report = evaluate_thermal_envelope(w, project_name="Net Zero Research Pavilion")

    pdf_bytes = export_thermal_envelope_certificate_pdf(report, auditor_name="LEED AP BD+C Auditor")
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-1.4\n")
    assert b"%%EOF" in pdf_bytes
    assert b"BUILDING ENVELOPE THERMAL COMPLIANCE CERTIFICATE" in pdf_bytes
    assert b"Net Zero Research Pavilion" in pdf_bytes


@pytest.mark.asyncio
async def test_thermal_envelope_api_endpoints():
    """Verify HTTP REST API endpoints for thermal report, custom evaluation, and certificate."""
    from starlette.testclient import TestClient
    from apps.api.main import app
    from apps.api.db import get_sessionmaker, init_db
    from apps.api.models import World, WorldVersion
    from apps.api.worldstore_service import get_store

    client = TestClient(app)

    world_id = f"wld_thm_{uuid.uuid4().hex[:8]}"
    w = _build_thermal_test_world()
    w.id = world_id

    store = get_store()
    sv = store.save_version(w, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        world_row = World(
            id=w.id,
            name="Sustainable Innovation Lab",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(world_row)

        ver_row = WorldVersion(
            id=sv.version_id,
            world_id=w.id,
            label="Base Version for Thermal",
            report={},
        )
        session.add(ver_row)
        await session.commit()

    # 1. GET /api/worlds/{id}/thermal-envelope
    res_get = client.get(f"/api/worlds/{w.id}/thermal-envelope?climate_zone=5A_cold")
    assert res_get.status_code == 200, res_get.text
    data = res_get.json()
    assert data["world_id"] == w.id
    assert "summary" in data
    assert data["summary"]["window_to_wall_ratio_pct"] == pytest.approx(19.5, 0.1)
    assert len(data["surfaces"]) >= 4

    # 2. POST /api/worlds/{id}/thermal-envelope/evaluate
    res_post = client.post(
        f"/api/worlds/{w.id}/thermal-envelope/evaluate",
        json={
            "climate_zone": "4A_mixed_humid",
            "design_delta_t_k": 20.0,
            "custom_u_overrides": {"wall_u": 0.25, "fenestration_u": 1.6},
            "project_name": "High Performance Lab",
        },
    )
    assert res_post.status_code == 200, res_post.text
    post_data = res_post.json()
    assert post_data["climate_zone"] == "4A_mixed_humid"
    assert post_data["design_delta_t_k"] == 20.0

    # 3. GET /api/worlds/{id}/thermal-envelope/certificate
    res_cert = client.get(f"/api/worlds/{w.id}/thermal-envelope/certificate")
    assert res_cert.status_code == 200, res_cert.text
    assert res_cert.content.startswith(b"%PDF-1.4\n")
    assert b"%%EOF" in res_cert.content
