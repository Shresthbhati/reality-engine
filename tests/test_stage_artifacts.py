"""A real production job publishes the compiled world as a stored artifact: the hash in the stage record resolves to the exact bytes."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from tests.test_reconstruction_reliability import _chain_setup, _wait_job  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def test_a_stage_deadline_fails_the_real_job_with_the_reason_and_adopts_nothing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("RE_STAGE_TIMEOUTS_S", "reconstruct=0.01")  # failure injection: the reconstruction stage cannot finish in 10 ms
    sid, wid = _chain_setup(client, tmp_path, monkeypatch)
    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=120.0)
    assert job["status"] == "failed" and "deadline" in str(job.get("error", "")).lower(), job
    stages = client.get(f"/api/jobs/{job['id']}/stages").json()["items"]
    rec = next(s for s in stages if s["name"] == "reconstruct")
    assert rec["status"] == "failed" and rec["metrics"].get("timed_out") == 1.0
    assert not any(s["name"] in ("compile_validate", "changeset_commit") for s in stages)  # nothing downstream ran
    assert client.get(f"/api/worlds/{wid}/status").json()["has_model"] is False  # no false successful world


def test_generous_deadlines_leave_the_real_job_succeeding_through_the_threaded_stage_path(client, tmp_path, monkeypatch):
    monkeypatch.setenv("RE_STAGE_TIMEOUTS_S", "evidence_resolve=600,reconstruct=600,compile_validate=600,changeset_commit=600")
    sid, _ = _chain_setup(client, tmp_path, monkeypatch)
    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=120.0)
    assert job["status"] == "succeeded", job
    names = {s["name"]: s["status"] for s in client.get(f"/api/jobs/{job['id']}/stages").json()["items"]}
    assert names["compile_validate"] == "succeeded" and names["changeset_commit"] in ("succeeded", "degraded")


def test_compile_validate_publishes_the_world_as_a_resolvable_artifact_with_matching_bytes(client, tmp_path, monkeypatch):
    sid, _ = _chain_setup(client, tmp_path, monkeypatch)
    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=120.0)
    assert job["status"] == "succeeded"
    items = client.get(f"/api/jobs/{job['id']}/stages").json()["items"]
    stage = next(i for i in items if i["name"] == "compile_validate")
    digest = stage["outputs"]["world_ir"]
    from engine.pipeline.stages.content_store import resolve_artifact

    path = resolve_artifact(f"sha256://{digest}")
    assert path is not None, "an output reference must point at an artifact that exists"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest  # reload verification: the stored bytes ARE what the hash names
