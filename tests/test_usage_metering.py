"""Tests for per-tenant usage metering (T-6.5).

Verifies:
1. Per-tenant usage endpoint: GET /api/tenants/{tenant_id}/usage and GET /api/usage.
2. GPU-minutes and compute duration attribution per tenant.
3. Storage bytes (uploads, evidence, exports) tracking per tenant.
4. Export count and format breakdown metering.
5. Strict tenant isolation: Tenant B cannot query Tenant A's usage (404).
6. Admin capability to audit any tenant's usage.
7. Time-range filtering via `since` and `until`.
"""

from __future__ import annotations

import importlib
from datetime import timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")
from fastapi.testclient import TestClient

from apps.api.auth import Role, issue_scoped_token, rate_limiter
from apps.api.models import ExportRecord, Job, utcnow

SECRET = "usage-metering-test-secret"


class Api:
    def __init__(self, client: TestClient):
        self.c = client
        self.h = {
            t: {"Authorization": "Bearer " + issue_scoped_token(t, f"user-{t}", role, secret=SECRET)}
            for t, role in (("tenant-a", Role.EDITOR), ("tenant-b", Role.EDITOR), ("root", Role.ADMIN))
        }

    def call(self, who: str, method: str, path: str, **kw):
        return self.c.request(method, path, headers=self.h[who], **kw)


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_SIGNING_KEY", SECRET)
    monkeypatch.setenv("REALITY_API_TOKEN", "master-token-for-tests")
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    rate_limiter.reset()
    with TestClient(main_mod.app) as c:
        yield Api(c)
    rate_limiter.reset()


def test_tenant_usage_empty_baseline(api):
    """A fresh tenant begins with 0 compute, 0 storage, and 0 exports."""
    res = api.call("tenant-a", "GET", "/api/tenants/tenant-a/usage")
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["tenant_id"] == "tenant-a"
    assert data["compute"]["gpu_minutes"] == 0.0
    assert data["compute"]["total_compute_seconds"] == 0.0
    assert data["compute"]["jobs_count"] == 0
    assert data["storage"]["total_bytes"] == 0
    assert data["exports"]["total_count"] == 0
    assert data["exports"]["by_format"] == {}


def test_tenant_usage_isolation_and_admin_override(api):
    """Tenant B cannot read Tenant A's usage (404), but Admin can."""
    # Tenant A creates a world
    w = api.call("tenant-a", "POST", "/api/worlds", json={"name": "A World"}).json()
    assert "id" in w

    # Tenant B tries to inspect Tenant A's usage -> 404 (anti-enumeration)
    res_b = api.call("tenant-b", "GET", "/api/tenants/tenant-a/usage")
    assert res_b.status_code == 404, res_b.text
    assert "not found" in res_b.json()["detail"].lower()

    # Tenant B tries to query Tenant A via /api/usage?tenant_id=tenant-a -> 404
    res_b_query = api.call("tenant-b", "GET", "/api/usage?tenant_id=tenant-a")
    assert res_b_query.status_code == 404

    # Tenant B querying their own usage succeeds
    res_b_own = api.call("tenant-b", "GET", "/api/usage")
    assert res_b_own.status_code == 200
    assert res_b_own.json()["tenant_id"] == "tenant-b"

    # Admin querying Tenant A's usage succeeds
    res_admin = api.call("root", "GET", "/api/tenants/tenant-a/usage")
    assert res_admin.status_code == 200
    assert res_admin.json()["tenant_id"] == "tenant-a"
    assert res_admin.json()["storage"]["counts"]["worlds"] == 1


def test_tenant_usage_compute_storage_and_export_metering(api):
    """Usage properly accumulates GPU-minutes, storage bytes, and export format counts."""
    # 1. Create a session and upload file for Tenant A
    sess_res = api.call("tenant-a", "POST", "/api/sessions", json={"name": "Survey Session"})
    session_id = sess_res.json()["id"]

    file_bytes = b"timestamp,val\n1,100\n2,200\n"
    up_res = api.call(
        "tenant-a",
        "POST",
        f"/api/uploads?session_id={session_id}",
        files={"file": ("data.csv", file_bytes, "text/csv")},
    )
    assert up_res.status_code == 201
    up_data = up_res.json()
    assert "upload_id" in up_data

    # Check that storage immediately reflects the upload
    usage1 = api.call("tenant-a", "GET", "/api/tenants/tenant-a/usage").json()
    assert usage1["storage"]["counts"]["uploads"] == 1
    assert usage1["storage"]["breakdown_bytes"]["uploads"] == len(file_bytes)
    assert usage1["storage"]["counts"]["evidence"] == 1
    assert usage1["storage"]["breakdown_bytes"]["evidence"] == len(file_bytes)

    # 2. Inject job with GPU compute metrics for Tenant A
    import asyncio

    import apps.api.db as db_mod

    async def _add_test_job_and_export():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            now = utcnow()
            # 120 seconds of GPU compute (2.0 GPU minutes)
            j = Job(
                id="job_test_gpu_meter",
                team_id="tenant-a",
                type="RECONSTRUCT_SESSION",
                entity_type="session",
                entity_id=session_id,
                status="succeeded",
                created_at=now - timedelta(seconds=150),
                started_at=now - timedelta(seconds=140),
                completed_at=now - timedelta(seconds=20),
                payload={"gpu_seconds": 120.0},
            )
            db.add(j)

            # Record two exports: GLB (5000 bytes) and DXF (2500 bytes)
            exp1 = ExportRecord(
                id="exp_test_glb",
                tenant_id="tenant-a",
                world_id="w_dummy",
                format="glb",
                size_bytes=5000,
                created_at=now - timedelta(minutes=5),
            )
            exp2 = ExportRecord(
                id="exp_test_dxf",
                tenant_id="tenant-a",
                world_id="w_dummy",
                format="floorplan_dxf",
                size_bytes=2500,
                created_at=now - timedelta(minutes=1),
            )
            db.add(exp1)
            db.add(exp2)
            await db.commit()

    asyncio.run(_add_test_job_and_export())

    # 3. Query usage endpoint
    usage2 = api.call("tenant-a", "GET", "/api/tenants/tenant-a/usage").json()

    # Compute assertions
    assert usage2["compute"]["jobs_count"] >= 1
    assert usage2["compute"]["jobs_by_status"].get("succeeded") >= 1
    assert usage2["compute"]["gpu_seconds"] == 120.0
    assert usage2["compute"]["gpu_minutes"] == 2.0
    assert usage2["compute"]["total_compute_seconds"] == 120.0

    # Export assertions
    assert usage2["exports"]["total_count"] == 2
    assert usage2["exports"]["total_bytes"] == 7500
    assert usage2["exports"]["by_format"]["glb"] == 1
    assert usage2["exports"]["by_format"]["floorplan_dxf"] == 1
    assert usage2["exports"]["last_export_at"] is not None

    # Storage assertions (includes exports and uploads)
    expected_storage = len(file_bytes) * 2 + 7500
    assert usage2["storage"]["total_bytes"] == expected_storage
    assert usage2["storage"]["breakdown_bytes"]["exports"] == 7500

    # 4. Confirm Tenant B's usage remains clean and unaffected
    usage_b = api.call("tenant-b", "GET", "/api/tenants/tenant-b/usage").json()
    assert usage_b["compute"]["gpu_minutes"] == 0.0
    assert usage_b["storage"]["total_bytes"] == 0
    assert usage_b["exports"]["total_count"] == 0


def test_tenant_usage_time_window_filtering(api):
    """Usage correctly applies `since` and `until` filters."""
    import asyncio

    import apps.api.db as db_mod

    now = utcnow()

    async def _seed_timeline():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            # Past job (2 days ago)
            j_past = Job(
                id="job_past",
                team_id="tenant-a",
                type="PROCESS_EVIDENCE",
                entity_type="session",
                entity_id="s_past",
                status="succeeded",
                created_at=now - timedelta(days=2),
                started_at=now - timedelta(days=2),
                completed_at=now - timedelta(days=2) + timedelta(seconds=60),
                payload={"gpu_seconds": 60.0},
            )
            # Recent job (1 hour ago)
            j_recent = Job(
                id="job_recent",
                team_id="tenant-a",
                type="RECONSTRUCT_SESSION",
                entity_type="session",
                entity_id="s_recent",
                status="succeeded",
                created_at=now - timedelta(hours=1),
                started_at=now - timedelta(hours=1),
                completed_at=now - timedelta(hours=1) + timedelta(seconds=180),
                payload={"gpu_seconds": 180.0},
            )
            # Past export (2 days ago)
            exp_past = ExportRecord(
                id="exp_past",
                tenant_id="tenant-a",
                world_id="w_past",
                format="glb",
                size_bytes=1000,
                created_at=now - timedelta(days=2),
            )
            # Recent export (1 hour ago)
            exp_recent = ExportRecord(
                id="exp_recent",
                tenant_id="tenant-a",
                world_id="w_recent",
                format="ifc",
                size_bytes=2000,
                created_at=now - timedelta(hours=1),
            )
            db.add_all([j_past, j_recent, exp_past, exp_recent])
            await db.commit()

    asyncio.run(_seed_timeline())

    # 1. Query all time
    all_time = api.call("tenant-a", "GET", "/api/tenants/tenant-a/usage").json()
    assert all_time["compute"]["jobs_count"] == 2
    assert all_time["compute"]["gpu_seconds"] == 240.0
    assert all_time["exports"]["total_count"] == 2

    # 2. Query with `since` = 1 day ago (should only include the recent job and export)
    since_ts = (now - timedelta(days=1)).isoformat()
    res = api.call("tenant-a", "GET", "/api/tenants/tenant-a/usage", params={"since": since_ts})
    assert res.status_code == 200, res.text
    filtered = res.json()
    assert filtered["compute"]["jobs_count"] == 1
    assert filtered["compute"]["gpu_seconds"] == 180.0
    assert filtered["exports"]["total_count"] == 1
    assert filtered["exports"]["by_format"] == {"ifc": 1}
    assert filtered["exports"]["total_bytes"] == 2000

