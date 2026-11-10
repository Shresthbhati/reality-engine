"""T006: specialist analysis responses declare they are preliminary and list their assumptions.

Each test calls the router's main analysis endpoint through the same
seed-a-world fixture pattern the router's existing test file uses.
"""

from __future__ import annotations

import uuid

import pytest

from tests.test_life_safety_compliance import _build_life_safety_test_world
from tests.test_material_takeoff_boq import _build_takeoff_test_world
from tests.test_mep_topology import _build_test_world as _build_mep_test_world
from tests.test_thermal_envelope_ashrae import _build_thermal_test_world


async def _seed(builder, prefix: str):
    from starlette.testclient import TestClient

    from apps.api.db import get_sessionmaker, init_db
    from apps.api.main import app
    from apps.api.models import World, WorldVersion
    from apps.api.worldstore_service import get_store

    w = builder()
    w.id = f"wld_{prefix}_{uuid.uuid4().hex[:8]}"
    sv = get_store().save_version(w, parent=None)
    await init_db()
    async with get_sessionmaker()() as session:
        session.add(World(id=w.id, name="Truth Label Test", current_version_id=sv.version_id, status="active"))
        session.add(WorldVersion(id=sv.version_id, world_id=w.id, label="v1", report={}))
        await session.commit()
    return TestClient(app), w.id


def _assert_labelled(body: dict) -> None:
    assert body["validation_status"] == "preliminary"
    assert len(body["assumptions"]) >= 1
    assert all(isinstance(a, str) and a for a in body["assumptions"])


@pytest.mark.asyncio
async def test_takeoff_declares_preliminary():
    client, wid = await _seed(_build_takeoff_test_world, "tk")
    r = client.get(f"/api/worlds/{wid}/takeoff")
    assert r.status_code == 200, r.text
    _assert_labelled(r.json())
    assert "summary" in r.json()  # existing keys preserved
    r = client.post(f"/api/worlds/{wid}/takeoff/estimate", json={})
    assert r.status_code == 200, r.text
    _assert_labelled(r.json())


@pytest.mark.asyncio
async def test_thermal_envelope_declares_preliminary():
    client, wid = await _seed(_build_thermal_test_world, "th")
    r = client.get(f"/api/worlds/{wid}/thermal-envelope?climate_zone=5A_cold")
    assert r.status_code == 200, r.text
    _assert_labelled(r.json())
    assert "compliance_verdict" in r.json()
    r = client.post(f"/api/worlds/{wid}/thermal-envelope/evaluate", json={"climate_zone": "4A_mixed_humid"})
    assert r.status_code == 200, r.text
    _assert_labelled(r.json())


@pytest.mark.asyncio
async def test_mep_declares_preliminary_and_basis():
    client, wid = await _seed(_build_mep_test_world, "mep")
    for r in (
        client.get(f"/api/worlds/{wid}/mep/topology"),
        client.post(f"/api/worlds/{wid}/mep/detect", json={}),
    ):
        assert r.status_code == 200, r.text
        body = r.json()
        _assert_labelled(body)
        assert body["basis"] == "inferred_from_world_entities"
        assert "systems" in body
    r = client.post(
        f"/api/worlds/{wid}/mep/route",
        json={"start_point": [0.2, 0.5, 1.4], "end_point": [3.5, 0.1, 0.45]},
    )
    assert r.status_code == 200, r.text
    _assert_labelled(r.json())
    assert "waypoints" in r.json()


@pytest.mark.asyncio
async def test_life_safety_declares_preliminary():
    client, wid = await _seed(_build_life_safety_test_world, "ls")
    r = client.get(f"/api/worlds/{wid}/life-safety")
    assert r.status_code == 200, r.text
    _assert_labelled(r.json())
    assert len(r.json()["egress_routes"]) == 3
    r = client.post(f"/api/worlds/{wid}/life-safety/audit", json={"is_sprinklered": False})
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_labelled(body)
    assert any("not sprinklered" in a.lower() for a in body["assumptions"])
