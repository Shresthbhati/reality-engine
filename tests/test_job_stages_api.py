"""P105: GET /api/jobs/{job_id}/stages lists a job's JobStageRun rows ordered by started_at."""

from __future__ import annotations

import asyncio
import importlib
from datetime import timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")
from fastapi.testclient import TestClient

from apps.api.auth import Role, issue_scoped_token, rate_limiter
from apps.api.models import JobStageRun, new_id, utcnow

SECRET = "job-stages-test-secret"


class Api:
    def __init__(self, client: TestClient):
        self.c = client
        self.h = {
            t: {"Authorization": "Bearer " + issue_scoped_token(t, f"user-{t}", role, secret=SECRET)}
            for t, role in (("tenant-a", Role.EDITOR), ("tenant-b", Role.EDITOR))
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


def _new_job(api: Api, stages: list[tuple[str, int]] | None = None) -> str:
    """Enqueue a tenant-a job; optionally insert (name, start-offset-seconds) stage rows in the given order."""
    session_id = api.call("tenant-a", "POST", "/api/sessions", json={"name": "Stages"}).json()["id"]
    import apps.api.db as db_mod
    from apps.api.jobs import enqueue_job

    async def _setup() -> str:
        async with db_mod.get_sessionmaker()() as db:
            j = enqueue_job(db, "PROCESS_EVIDENCE", "session", session_id, team_id="tenant-a", payload={})
            base = utcnow()
            for name, off in stages or []:
                db.add(
                    JobStageRun(
                        id=new_id("jsr"),
                        job_id=j.id,
                        team_id="tenant-a",
                        name=name,
                        version="1",
                        inputs_hash="h-" + name,
                        status="succeeded",
                        attempt=1,
                        metrics={"n": off},
                        warnings=[],
                        started_at=base + timedelta(seconds=off),
                        ended_at=base + timedelta(seconds=off + 2),
                    )
                )
            await db.commit()
            return j.id

    return asyncio.run(_setup())


def test_stages_empty_list_for_new_job(api):
    job_id = _new_job(api)
    r = api.call("tenant-a", "GET", f"/api/jobs/{job_id}/stages")
    assert r.status_code == 200, r.text
    assert r.json() == {"items": []}


def test_stages_returns_rows_in_order(api):
    job_id = _new_job(api, [("c", 20), ("a", 0), ("b", 10)])  # inserted out of order
    r = api.call("tenant-a", "GET", f"/api/jobs/{job_id}/stages")
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert [i["name"] for i in items] == ["a", "b", "c"]
    first = items[0]
    assert set(first) == {
        "name", "version", "status", "attempt", "metrics", "warnings", "error", "started_at", "ended_at", "duration_s",
    }
    assert first["status"] == "succeeded" and first["attempt"] == 1 and first["metrics"] == {"n": 0}
    assert first["duration_s"] == pytest.approx(2.0)


def test_unknown_job_404(api):
    assert api.call("tenant-a", "GET", "/api/jobs/does-not-exist/stages").status_code == 404


def test_other_tenant_denied(api):
    job_id = _new_job(api, [("a", 0)])
    logs = api.call("tenant-b", "GET", f"/api/jobs/{job_id}/logs")
    stages = api.call("tenant-b", "GET", f"/api/jobs/{job_id}/stages")
    assert stages.status_code == logs.status_code == 404
