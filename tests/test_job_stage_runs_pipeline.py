"""P106: the reconstruct job runs through the stage runner and persists one ``job_stage_runs`` row per coarse stage."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from tests.test_application_api import _attach, _upload_photo, _wait_job, client  # noqa: F401  (fixture)

FINE = ["pose", "scale", "depth", "layout", "perception", "mesh", "dense", "detail", "finalize"]
STAGES = ["evidence_resolve", "evidence_quality", "reconstruct", *FINE, "quality_report", "compile_validate",
          "changeset_commit"]


def _first_run(client, tmp_path, monkeypatch, expect="succeeded"):
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None
    sid = client.post("/api/sessions", json={"name": "Stages"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Stages world"}).json()["id"]
    _attach(client, wid, sid)
    for i in range(3):
        _upload_photo(client, sid, f"frame_{i:02d}.jpg", b"jpeg-" + f"{i}".encode() * 8)

    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=90.0)
    assert job["status"] == expect, job.get("error")
    return sid, job


def _stages(client, job):
    return client.get(f"/api/jobs/{job['id']}/stages").json()["items"]


def test_successful_reconstruct_records_the_fine_stages_in_order(client, tmp_path, monkeypatch):
    _, job = _first_run(client, tmp_path, monkeypatch)
    items = _stages(client, job)
    assert [i["name"] for i in items] == STAGES
    status = {i["name"]: i["status"] for i in items}
    # backend-dependent optional stages are skipped (never "succeeded") on the deterministic test backend
    skipped = {"depth", "perception", "mesh", "dense", "detail", "quality_report"}  # no mesh -> NOT_EVALUABLE, skipped
    assert {n for n, st in status.items() if st == "skipped"} == skipped
    assert {st for n, st in status.items() if n not in skipped} == {"succeeded"}
    assert all(i["version"] == "1" for i in items)
    assert all(i["error"] is None for i in items)


def test_unchanged_run_keeps_its_payload_and_grades_succeeded(client, tmp_path, monkeypatch):
    """The early exit when the evidence is already reflected must not fall through to the adopt/commit tail."""
    sid, _ = _first_run(client, tmp_path, monkeypatch)
    import apps.api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod.world_delta, "is_unchanged", lambda *a, **k: True)
    again = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=90.0)
    assert again["status"] == "succeeded", again.get("error")
    assert again["payload"]["unchanged"] is True and again["payload"]["adopted"] is False
    assert {i["status"] for i in _stages(client, again)} == {"succeeded", "skipped"}  # skipped: optional backends off


def test_rejected_candidate_grades_partial_through_the_manifest(client, tmp_path, monkeypatch):
    sid, first = _first_run(client, tmp_path, monkeypatch)
    import apps.api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod.world_delta, "decide", lambda delta, conflicts: {
        "verdict": jobs_mod.world_delta.REJECT, "reasons": ["forced for test"], "uncertainties": []})
    again = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=90.0)
    assert again["status"] == "partial", again.get("error")
    assert again["payload"]["adopted"] is False and again["payload"]["kept_version_id"] == first["payload"]["version_id"]
    assert {i["name"]: i["status"] for i in _stages(client, again)}["changeset_commit"] == "degraded"


def test_failing_required_stage_fails_the_job_with_the_stage_error(client):
    """A .ply-only session has no photo evidence: evidence_resolve fails, nothing after it runs, and the job
    carries the original (deterministic, non-retried) error."""
    sid = client.post("/api/sessions", json={"name": "Nothing usable"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Nothing usable world"}).json()["id"]
    _attach(client, wid, sid)
    client.post(f"/api/uploads?session_id={sid}", files={"file": ("scan.ply", b"ply-bytes", "application/octet-stream")})

    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=90.0)
    assert job["status"] == "failed" and job["attempts"] == 1, job
    assert "0 usable photo evidence" in job["error"]

    items = _stages(client, job)
    assert [(i["name"], i["status"]) for i in items] == [("evidence_resolve", "failed")]
    assert "0 usable photo evidence" in items[0]["error"]


def test_single_image_fallback_skips_the_fine_stages_and_grades_partial_not_failed(client, tmp_path, monkeypatch):
    """The ladder is kept: when the multi-view rung is not the one used, the job still completes (partial) and the
    fine stages say why they did not contribute."""
    import dataclasses

    import apps.api.jobs as jobs_mod

    real = jobs_mod.run_progressive

    def fell_back(*args, **kwargs):
        prog = real(*args, **kwargs)
        attempts = [*prog.attempts, {"level": 2, "name": "x", "outcome": "failed", "detail": "VerticalSliceError: simulated"}]
        return dataclasses.replace(prog, vs_result=None, attempts=attempts, degraded=[*prog.degraded, "single-view fallback"])

    monkeypatch.setattr(jobs_mod, "run_progressive", fell_back)
    _, job = _first_run(client, tmp_path, monkeypatch, expect="partial")
    by = {i["name"]: i for i in _stages(client, job)}
    assert by["reconstruct"]["status"] == "degraded"
    for name in FINE:
        assert by[name]["status"] == "skipped" and "simulated" in by[name]["warnings"][0], name
    assert by["compile_validate"]["status"] == "succeeded"
