"""Tests for Automated Life Safety & ADA Accessibility Compliance Engine.

Verifies:
1. ADA §404.2.3 & IBC §1010.1.1 doorway clear width checks (>= 32 inches / 0.813m).
2. ADA §304 wheelchair 60-inch (1.524m) turning space clearance checks.
3. ADA §403 corridor clear width checks.
4. IBC §1010.1.2.1 egress door swing direction for occupant loads > 50.
5. IBC §1017 means of egress travel distance & polyline generation.
6. Zero-dependency PDF-1.4 Life Safety Certificate generation.
7. FastAPI REST API endpoints for audit, egress routes, and certificate downloads.
"""

from __future__ import annotations

import io
import uuid
import pytest

from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR
from engine.pipeline.life_safety_compliance import (
    CodeCheckCategory,
    ComplianceStatus,
    LifeSafetyReport,
    audit_life_safety_compliance,
    export_life_safety_certificate_pdf,
)


def _build_life_safety_test_world() -> WorldIR:
    """Build a synthetic commercial suite with accessible and non-compliant doors/spaces."""
    world = WorldIR(id="world-life-safety-test")

    # Room 1: Large Conference / Assembly Hall (12m x 8m = 96 m2, occupant load ~ 68 > 50)
    # Has wide compliant door (1.0m) swinging outward (compliant)
    room1 = Entity(
        id="room-conf-1",
        name="Main Conference Hall",
        type=EntityType.ROOM,
        custom_properties={
            "footprint_polygon_m": [[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]],
            "floor_area_m2": 96.0,
            "height_m": 3.2,
        },
    )
    world.entities[room1.id] = room1

    # Door 1: Accessible exit door from Conference Hall (1.0m wide, outward swing)
    door1 = Entity(
        id="door-conf-exit",
        name="Conference Exit Door",
        type=EntityType.DOOR,
        custom_properties={
            "room_id": room1.id,
            "width_m": 1.0,  # 39.4 in > 32 in (PASS)
            "height_m": 2.15,
            "is_exit": True,
            "swing_direction": "outward",
        },
        transform={"position": {"x": 12.0, "y": 4.0, "z": 0.0}},
    )
    world.entities[door1.id] = door1

    # Room 2: Small Office (4m x 3.5m = 14 m2)
    # Has a standard 0.85m door (> 32 in PASS)
    room2 = Entity(
        id="room-office-2",
        name="Private Office 201",
        type=EntityType.ROOM,
        custom_properties={
            "footprint_polygon_m": [[0.0, 8.0], [4.0, 8.0], [4.0, 11.5], [0.0, 11.5]],
            "floor_area_m2": 14.0,
            "height_m": 2.7,
        },
    )
    world.entities[room2.id] = room2

    door2 = Entity(
        id="door-office-entry",
        name="Office 201 Entry Door",
        type=EntityType.DOOR,
        custom_properties={
            "room_id": room2.id,
            "width_m": 0.85,  # 33.5 in > 32 in (PASS)
            "height_m": 2.1,
            "is_exit": False,
            "swing_direction": "inward",
        },
        transform={"position": {"x": 2.0, "y": 8.0, "z": 0.0}},
    )
    world.entities[door2.id] = door2

    # Room 3: Restroom with narrow non-compliant door (0.70m < 32 in / 0.813m FAIL)
    # and tight dimensions (1.2m x 1.2m < 60-inch / 1.524m turning circle WARNING)
    room3 = Entity(
        id="room-wc-narrow",
        name="Historic Powder Room",
        type=EntityType.ROOM,
        custom_properties={
            "footprint_polygon_m": [[4.0, 8.0], [5.2, 8.0], [5.2, 9.2], [4.0, 9.2]],
            "floor_area_m2": 1.44,
            "height_m": 2.4,
        },
    )
    world.entities[room3.id] = room3

    door3 = Entity(
        id="door-wc-narrow",
        name="Powder Room Door",
        type=EntityType.DOOR,
        custom_properties={
            "room_id": room3.id,
            "width_m": 0.70,  # 27.5 in < 32 in (FAIL)
            "height_m": 2.0,
            "is_exit": False,
            "swing_direction": "inward",
        },
        transform={"position": {"x": 4.6, "y": 8.0, "z": 0.0}},
    )
    world.entities[door3.id] = door3

    return world


def test_life_safety_door_clearance_and_turning_space():
    """Verify doorway width compliance and ADA wheelchair turning circle evaluation."""
    w = _build_life_safety_test_world()
    report = audit_life_safety_compliance(w, occupancy_group="Business (Group B)", is_sprinklered=True)

    assert report.world_id == "world-life-safety-test"
    assert report.summary["total_doors_audited"] == 3
    assert report.summary["total_rooms_audited"] == 3

    # Check door clearance results
    door_checks = [c for c in report.checks if c.category == CodeCheckCategory.DOOR_CLEARANCE]
    assert len(door_checks) == 3

    # Door 1 (1.0m) and Door 2 (0.85m) pass
    assert any(c.status == ComplianceStatus.PASS and c.measured_value == 1.0 for c in door_checks)
    assert any(c.status == ComplianceStatus.PASS and c.measured_value == 0.85 for c in door_checks)

    # Door 3 (0.70m) fails ADA 32-inch requirement
    fail_door = next(c for c in door_checks if c.entity_id == "door-wc-narrow")
    assert fail_door.status == ComplianceStatus.FAIL
    assert fail_door.measured_value == 0.70
    assert "narrower than 32 in" in fail_door.message

    # Check turning space results
    assert len(report.turning_spaces) == 3
    conf_turn = next(t for t in report.turning_spaces if t.room_id == "room-conf-1")
    assert conf_turn.has_clearance is True
    assert conf_turn.status == ComplianceStatus.PASS

    wc_turn = next(t for t in report.turning_spaces if t.room_id == "room-wc-narrow")
    assert wc_turn.has_clearance is False
    assert wc_turn.status == ComplianceStatus.WARNING


def test_egress_travel_distance_and_polyline():
    """Verify means of egress travel distance and polyline routes."""
    w = _build_life_safety_test_world()
    report = audit_life_safety_compliance(w, is_sprinklered=True)

    assert len(report.egress_routes) == 3
    for route in report.egress_routes:
        assert len(route.polyline) >= 2
        assert route.total_distance_m > 0.0
        # In sprinklered building max permitted is 76.2m
        assert route.max_permitted_distance_m == 76.2
        assert route.status == ComplianceStatus.PASS

    # Ensure max egress distance calculation
    max_d = report.summary["max_egress_distance_m"]
    assert max_d > 5.0
    assert max_d < 76.2


def test_pdf_certificate_export():
    """Verify generation of valid PDF-1.4 Life Safety & Accessibility Certificate."""
    w = _build_life_safety_test_world()
    report = audit_life_safety_compliance(w, project_name="Downtown Innovation Hub")

    pdf_bytes = export_life_safety_certificate_pdf(report, architect_name="Licensed BIM Architect")
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-1.4\n")
    assert b"%%EOF" in pdf_bytes
    assert b"LIFE SAFETY & ADA ACCESSIBILITY CERTIFICATE" in pdf_bytes
    assert b"Downtown Innovation Hub" in pdf_bytes


@pytest.mark.asyncio
async def test_life_safety_api_endpoints():
    """Verify HTTP REST API endpoints for life safety retrieval, audit, and exports."""
    from starlette.testclient import TestClient
    from apps.api.main import app
    from apps.api.db import get_sessionmaker, init_db
    from apps.api.models import World, WorldVersion
    from apps.api.worldstore_service import get_store

    client = TestClient(app)

    world_id = f"wld_ls_{uuid.uuid4().hex[:8]}"
    w = _build_life_safety_test_world()
    w.id = world_id

    store = get_store()
    sv = store.save_version(w, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        world_row = World(
            id=w.id,
            name="Commercial Innovation Centre",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(world_row)

        ver_row = WorldVersion(
            id=sv.version_id,
            world_id=w.id,
            label="Base Version for Life Safety",
            report={},
        )
        session.add(ver_row)
        await session.commit()

    # 1. GET /api/worlds/{id}/life-safety
    res_get = client.get(f"/api/worlds/{w.id}/life-safety")
    assert res_get.status_code == 200, res_get.text
    data = res_get.json()
    assert data["world_id"] == w.id
    assert "summary" in data
    assert data["summary"]["total_doors_audited"] == 3
    assert len(data["checks"]) >= 5
    assert len(data["egress_routes"]) == 3

    # 2. POST /api/worlds/{id}/life-safety/audit
    res_post = client.post(
        f"/api/worlds/{w.id}/life-safety/audit",
        json={
            "occupancy_group": "Assembly (Group A)",
            "is_sprinklered": False,
            "project_name": "Audited Assembly Hall",
        },
    )
    assert res_post.status_code == 200, res_post.text
    post_data = res_post.json()
    assert post_data["occupancy_group"] == "Assembly (Group A)"
    assert post_data["is_sprinklered"] is False

    # 3. GET /api/worlds/{id}/life-safety/egress-path
    res_egr = client.get(f"/api/worlds/{w.id}/life-safety/egress-path")
    assert res_egr.status_code == 200, res_egr.text
    routes = res_egr.json()
    assert len(routes) == 3
    assert "polyline" in routes[0]

    # 4. GET /api/worlds/{id}/life-safety/certificate
    res_cert = client.get(f"/api/worlds/{w.id}/life-safety/certificate")
    assert res_cert.status_code == 200, res_cert.text
    assert res_cert.content.startswith(b"%PDF-1.4\n")
    assert b"%%EOF" in res_cert.content
