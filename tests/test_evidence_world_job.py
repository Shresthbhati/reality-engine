"""Evidence-to-world stage inside the PRODUCTION reconstruction job. Detectors are SYNTHETIC test doubles patched in place of the
real engines; the job, queue, cancellation, persistence and grading are the real production code (deterministic test backend).
These tests prove integration safety, not model accuracy."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from tests.test_multiengine_evidence_flow import _setup  # noqa: E402
from tests.test_reconstruction_reliability import _chain_setup, _wait_job  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import importlib

    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _fake_inputs(monkeypatch, *, raise_exc=None, on_first_infer=None):
    from apps.api import evidence_world_stage as ew

    def inputs():
        if raise_exc is not None:
            raise raise_exc
        reg, adapters = _setup()
        if on_first_infer is not None:
            first = adapters["det_a"]
            orig = first.infer

            def infer(request):
                on_first_infer()
                return orig(request)

            first.infer = infer
        inputs.adapters = adapters
        return reg, adapters, None

    monkeypatch.setattr(ew, "_default_flow_inputs", inputs)
    return inputs


def _evidence_dirs(tmp_path):
    base = tmp_path / "ws" / "evidence-world"
    return sorted(p for p in base.glob("*/*/*") if p.is_dir()) if base.exists() else []


def test_flag_off_leaves_the_production_job_exactly_as_before(client, tmp_path, monkeypatch):
    monkeypatch.delenv("RE_EVIDENCE_WORLD", raising=False)
    sid, _ = _chain_setup(client, tmp_path, monkeypatch)
    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=90.0)
    assert job["status"] == "succeeded", job.get("error")
    assert "evidence_world" not in job["payload"] and _evidence_dirs(tmp_path) == []


def test_flag_on_records_the_stage_without_changing_the_job_grade(client, tmp_path, monkeypatch):
    sid, _ = _chain_setup(client, tmp_path, monkeypatch)
    baseline = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=90.0)
    monkeypatch.setenv("RE_EVIDENCE_WORLD", "1")
    _fake_inputs(monkeypatch)
    sid2, wid2 = _chain_setup(client, tmp_path, monkeypatch)
    job = _wait_job(client, client.post(f"/api/sessions/{sid2}/reconstruct").json()["job_id"], timeout=90.0)
    assert job["status"] == baseline["status"] == "succeeded"  # the evidence stage did not move the grade
    assert job["payload"]["outcome"] == baseline["payload"]["outcome"]
    ew = job["payload"]["evidence_world"]
    assert ew["status"] == "complete" and ew["counts"]["entities"] == 6 and ew["counts"]["observations"] == 6
    assert any("2D" in x for x in ew["limitations"])
    (d,) = _evidence_dirs(tmp_path)
    assert d.name == job["id"] and d.parent.name == wid2 and (d / "snapshot.json").is_file()  # tenant/world/job layout


def test_a_failing_flow_never_fails_the_job_and_the_failure_is_recorded(client, tmp_path, monkeypatch):
    monkeypatch.setenv("RE_EVIDENCE_WORLD", "1")
    _fake_inputs(monkeypatch, raise_exc=RuntimeError("model zoo exploded"))
    sid, _ = _chain_setup(client, tmp_path, monkeypatch)
    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=90.0)
    assert job["status"] == "succeeded"
    ew = job["payload"]["evidence_world"]
    assert ew["status"] == "failed" and "model zoo exploded" in ew["error"]


def test_operator_cancel_during_the_stage_stops_the_job_before_adoption_and_keeps_finished_evidence(
    client, tmp_path, monkeypatch
):
    monkeypatch.setenv("RE_EVIDENCE_WORLD", "1")
    state: dict = {}

    def cancel_via_api():
        deadline = time.time() + 10
        while "job_id" not in state and time.time() < deadline:
            time.sleep(0.05)
        out: dict = {}
        t = threading.Thread(target=lambda: out.setdefault("r", client.post(f"/api/jobs/{state['job_id']}/cancel")))
        t.start()
        t.join(10)

    inputs = _fake_inputs(monkeypatch, on_first_infer=cancel_via_api)
    sid, wid = _chain_setup(client, tmp_path, monkeypatch)
    state["job_id"] = client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
    job = _wait_job(client, state["job_id"], timeout=90.0)
    assert job["status"] == "cancelled", job
    assert inputs.adapters["det_a"].loaded == 1 and inputs.adapters["det_b"].loaded == 0  # stopped between engines
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] is None  # cancelled BEFORE a version was adopted
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    (d,) = _evidence_dirs(tmp_path)
    runs = {r.engine_id: r.status for r in EvidenceMatrix(d).runs()}
    assert runs == {"det_a": "succeeded", "det_b": "cancelled"}  # what finished is durable


def test_evidence_paths_are_tenant_scoped_and_traversal_safe(tmp_path):
    from apps.api.evidence_world_stage import EvidenceWorldError, evidence_root, world_root

    p = evidence_root("team-a", "world-1", "job-1", base=tmp_path)
    assert p == tmp_path / "evidence-world" / "team-a" / "world-1" / "job-1"
    assert world_root("team-a", "world-1", base=tmp_path) == p.parent
    assert evidence_root(None, "w", "j", base=tmp_path).parts[-3] == "default"
    assert evidence_root("team-b", "world-1", "job-1", base=tmp_path) != p  # another tenant never shares a directory
    assert evidence_root("", "w", "j", base=tmp_path).parts[-3] == "default"  # same empty-team convention as world creation
    for bad in ("../x", "a/b", "a\\b", "..", ".", "", "x" * 200, "a..b", "/abs"):
        if bad:  # an empty TEAM id means "default"; an empty world/job id is invalid (checked below)
            with pytest.raises(EvidenceWorldError):
                evidence_root(bad, "w", "j", base=tmp_path)
        with pytest.raises(EvidenceWorldError):
            evidence_root("t", bad, "j", base=tmp_path)
        with pytest.raises(EvidenceWorldError):
            evidence_root("t", "w", bad, base=tmp_path)


def test_virtual_and_missing_sources_are_never_run(tmp_path):
    from apps.api.evidence_world_stage import run_for_job

    out = run_for_job(team_id="t", world_id="w", job_id="j", sources=["virtual:a", str(tmp_path / "nope.jpg")], base=tmp_path)
    assert out["status"] == "skipped" and out["skipped_sources"] == 2
    assert not (tmp_path / "evidence-world").exists()  # no model was started, nothing was written
