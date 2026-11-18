"""RERES platform contract: POST /api/worlds/{id}/refine is bounded, versioned, honest about engines, and tenant-scoped."""
from __future__ import annotations

import uuid

import pytest
from starlette.testclient import TestClient

from engine.quality.autonomous_correction_loop import RefinementKind
from engine.quality.refinement_engines import register_engine, unregister_engine
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


async def _make_world(entities: dict[str, dict], team_id: str | None = "default") -> str:
    from apps.api.db import get_sessionmaker, init_db
    from apps.api.models import World, WorldVersion
    from apps.api.worldstore_service import get_store

    world_id = f"wld_refine_{uuid.uuid4().hex[:8]}"
    w = WorldIR(id=world_id, name="Refine me")
    for eid, props in entities.items():
        w.entities[eid] = Entity(id=eid, type=EntityType.UNKNOWN, name=eid, custom_properties=props)
    sv = get_store().save_version(w, parent=None)
    await init_db()
    async with get_sessionmaker()() as session:
        session.add(World(id=world_id, name="Refine me", current_version_id=sv.version_id, status="active", team_id=team_id))
        session.add(WorldVersion(id=sv.version_id, world_id=world_id, label="v1", report={}))
        await session.commit()
    return world_id


def _head(client: TestClient, world_id: str) -> str:
    return client.get(f"/api/worlds/{world_id}").json()["current_version_id"]


@pytest.mark.asyncio
async def test_refine_commits_a_new_version_only_when_something_changed():
    from apps.api.main import app

    client = TestClient(app)
    wid = await _make_world({"sofa": {"needs_more_views": True}})
    v1 = _head(client, wid)

    first = client.post(f"/api/worlds/{wid}/refine", json={})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["committed"] is True and body["parent_version_id"] == v1 and body["version_id"] != v1
    assert body["report"]["tasks_succeeded"] == 1 and body["report"]["stop_reason"] in ("empty_heap", "converged_state_unchanged")
    assert _head(client, wid) == body["version_id"]

    second = client.post(f"/api/worlds/{wid}/refine", json={})                 # nothing left to refine
    assert second.json()["committed"] is False and second.json()["version_id"] == body["version_id"]
    assert _head(client, wid) == body["version_id"]


@pytest.mark.asyncio
async def test_tasks_without_an_engine_are_reported_unavailable_and_commit_nothing():
    from apps.api.main import app

    client = TestClient(app)
    wid = await _make_world({"almirah": {"attributes": {"lock_type": {"value": None}}, "candidate_crops": {"lock_type": 128}}})
    v1 = _head(client, wid)

    res = client.post(f"/api/worlds/{wid}/refine", json={"max_iterations": 3})

    body = res.json()
    assert body["committed"] is False and body["version_id"] == v1 and _head(client, wid) == v1
    assert body["report"]["tasks_unavailable"] == 1 and body["report"]["stop_reason"] == "no_executor_available"
    assert body["report"]["engines_registered"] == []


@pytest.mark.asyncio
async def test_a_registered_engine_is_used_and_its_result_is_committed():
    from apps.api.main import app

    def detector(task, world):
        ent = world.entities[task.metadata["entity_id"]]
        ent.extra_fields["attributes"][task.metadata["attribute"]] = {"value": "numpad", "basis": "visual", "p": 0.8}
        return True

    register_engine(RefinementKind.REDETECT_CROP, detector)
    try:
        client = TestClient(app)
        wid = await _make_world({"almirah": {"attributes": {"lock_type": {"value": None}}, "candidate_crops": {"lock_type": 128}}})
        body = client.post(f"/api/worlds/{wid}/refine", json={}).json()
        assert body["committed"] is True and body["report"]["engines_registered"] == ["redetect_crop"]
        stored = client.get(f"/api/worlds/{wid}/worldir").json()
        assert stored["entities"]["almirah"]["custom_properties"]["attributes"]["lock_type"]["value"] == "numpad"
    finally:
        unregister_engine(RefinementKind.REDETECT_CROP)


@pytest.mark.asyncio
async def test_budget_is_bounded_unknown_and_foreign_worlds_are_404():
    from apps.api.main import app

    client = TestClient(app)
    wid = await _make_world({"x": {}})
    assert client.post(f"/api/worlds/{wid}/refine", json={"max_iterations": 999}).status_code == 422
    assert client.post(f"/api/worlds/{wid}/refine", json={"time_budget_s": 9999}).status_code == 422
    assert client.post("/api/worlds/wld_does_not_exist/refine", json={}).status_code == 404
    foreign = await _make_world({"x": {"needs_more_views": True}}, team_id="another-tenant")
    assert client.post(f"/api/worlds/{foreign}/refine", json={}).status_code == 404    # tenant isolation: not even confirmed to exist
