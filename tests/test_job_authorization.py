"""Job authorization across tenants: tenant B must not read, cancel or list tenant A's jobs. Real application, real job queue
(deterministic test backend); no model runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

A = {"x-tenant-id": "tenant-a"}
B = {"x-tenant-id": "tenant-b"}
ROOT = {"x-tenant-id": "root"}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    import importlib

    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _job_of_tenant_a(client) -> str:
    sid = client.post("/api/sessions", json={"name": "A"}, headers=A).json()["id"]
    wid = client.post("/api/worlds", json={"name": "A world"}, headers=A).json()["id"]
    assert client.post(f"/api/worlds/{wid}/attach/{sid}", headers=A).status_code == 200
    for i in range(3):
        r = client.post(f"/api/uploads?session_id={sid}", headers=A,
                        files={"file": (f"f{i}.jpg", b"jpeg-" + str(i).encode() * 8, "image/jpeg")})
        assert r.status_code == 201, r.text
    r = client.post(f"/api/sessions/{sid}/reconstruct", headers=A)
    assert r.status_code in (200, 201, 202), r.text
    return r.json()["job_id"]


def test_another_tenant_cannot_read_or_cancel_a_job(client):
    job_id = _job_of_tenant_a(client)
    assert client.get(f"/api/jobs/{job_id}", headers=A).status_code == 200  # the owner can
    got = client.get(f"/api/jobs/{job_id}", headers=B)
    assert got.status_code in (403, 404), got.text  # fail closed
    cancelled = client.post(f"/api/jobs/{job_id}/cancel", headers=B)
    assert cancelled.status_code in (403, 404), cancelled.text
    assert not client.get(f"/api/jobs/{job_id}", headers=A).json().get("cancel_requested")  # B's attempt did nothing


def test_job_listing_is_tenant_scoped(client):
    job_id = _job_of_tenant_a(client)
    body_a, body_b = client.get("/api/jobs", headers=A).json(), client.get("/api/jobs", headers=B).json()
    ids = lambda b: {j["id"] for j in (b["items"] if isinstance(b, dict) else b)}  # noqa: E731
    assert job_id in ids(body_a) and job_id not in ids(body_b)


def test_a_global_administrator_can_still_see_a_job(client):
    job_id = _job_of_tenant_a(client)
    assert client.get(f"/api/jobs/{job_id}", headers=ROOT).status_code == 200
