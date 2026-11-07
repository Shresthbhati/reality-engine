"""T-6.5: per-tenant usage (storage, compute time, exports): the numbers billing will be built on, never another tenant's.

Reuses the two-tenant fixture of test_tenant_isolation (tenant A and B are EDITORs, `root` is an ADMIN).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
from tests.test_commit_validation_contract import world  # noqa: E402,F401  (fixture reuse: a really reconstructed world)
from tests.test_tenant_isolation import api  # noqa: E402,F401  (fixture reuse)


def _db(fn):
    import apps.api.db as db_mod

    async def run():
        async with db_mod.get_sessionmaker()() as db:
            return await fn(db)

    return asyncio.run(run())


def _seed(api_):
    """Tenant A: a session with 8 bytes of evidence, a job that ran 90 s, another that failed after 30 s, two exports (glb, ifc)."""
    sid = api_.call("tenant-a", "POST", "/api/sessions", json={"name": "A"}).json()["id"]
    wid = api_.call("tenant-a", "POST", "/api/worlds", json={"name": "A world"}).json()["id"]
    up = api_.call("tenant-a", "POST", f"/api/uploads?session_id={sid}", files={"file": ("a.csv", b"t,v\n1,2\n", "text/csv")})
    assert up.status_code == 201, up.text
    now = datetime.now(timezone.utc)

    async def add(db):
        from apps.api.models import ActivityEvent, Job

        for jid, status, secs in (("job_meter_ok", "succeeded", 90), ("job_meter_bad", "failed", 30)):
            db.add(Job(id=jid, type="reconstruct", entity_type="session", entity_id=sid, status=status,
                       started_at=now - timedelta(seconds=secs), completed_at=now))
        for i, fmt in enumerate(("glb", "ifc")):
            db.add(ActivityEvent(id=f"act_meter_{i}", type="world.exported", entity_type="world", entity_id=wid,
                                 summary=f"Exported as {fmt}", metadata_json={"format": fmt}))
        await db.commit()

    _db(add)
    return sid, wid


def test_a_tenant_sees_its_own_storage_compute_and_exports(api):
    _seed(api)
    u = api.call("tenant-a", "GET", "/api/usage").json()
    assert u["tenant"] == "tenant-a"
    assert u["storage"]["bytes"] >= 8 and u["storage"]["evidence_items"] >= 1
    assert u["compute"]["seconds"] >= 120 and u["compute"]["by_status"]["failed"] == 1 and u["compute"]["by_status"]["succeeded"] >= 1
    assert "GPU share is not measured separately" in u["compute"]["note"]
    assert u["exports"]["count"] == 2 and u["exports"]["by_format"] == {"glb": 1, "ifc": 1}


def test_another_tenant_sees_none_of_it(api):
    _seed(api)
    u = api.call("tenant-b", "GET", "/api/usage").json()
    assert u["storage"]["bytes"] == 0 and u["compute"]["seconds"] == 0 and u["exports"]["count"] == 0


def test_only_an_admin_can_read_another_tenants_usage(api):
    _seed(api)
    assert api.call("tenant-b", "GET", "/api/usage", params={"tenant": "tenant-a"}).status_code == 403
    admin = api.call("root", "GET", "/api/usage", params={"tenant": "tenant-a"})
    assert admin.status_code == 200 and admin.json()["exports"]["count"] == 2


def test_since_limits_the_window(api):
    _seed(api)
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    u = api.call("tenant-a", "GET", "/api/usage", params={"since": future}).json()
    assert u["compute"]["seconds"] == 0 and u["exports"]["count"] == 0
    assert u["since"] is not None
    assert api.call("tenant-a", "GET", "/api/usage", params={"since": "not-a-date"}).status_code == 422


def test_a_real_export_through_the_api_is_counted_in_usage(world):
    """Not a seeded row: a reconstructed world is exported twice through the real route and the default tenant's usage shows it."""
    c, wid, _ = world
    for _ in range(2):
        r = c.post(f"/api/worlds/{wid}/export", json={"format": "glb"})
        assert r.status_code == 200, r.text
    u = c.get("/api/usage").json()
    assert u["tenant"] == "default" and u["exports"]["count"] == 2 and u["exports"]["by_format"] == {"glb": 2}
    assert u["storage"]["evidence_items"] >= 3 and u["compute"]["jobs"] >= 1
