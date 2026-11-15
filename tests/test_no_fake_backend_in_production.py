"""Test doubles must never be part of a production default backend chain."""

from pathlib import Path

import engine.orchestrator.universal_pipeline as up
from apps.cli.main import _default_backends
from reconstruction.backend.fake import FakeReconstructionBackend
from tests.test_application_api import _upload_photo, _wait_job, client  # noqa: F401  (fixture reuse)


def test_cli_default_backends_exclude_the_fake():
    assert not any(isinstance(b, FakeReconstructionBackend) for b in _default_backends("colmap", False))


def test_universal_pipeline_default_backends_exclude_the_fake(monkeypatch):
    captured = {}

    class _Stop(Exception):
        pass

    class _Capture:
        def __init__(self, backends):
            captured["backends"] = list(backends)
            raise _Stop

    monkeypatch.setattr(up, "ReconstructionOrchestrator", _Capture)
    opts = up.UniversalPipelineOptions()
    hw = type("HW", (), {"is_gpu": False})()
    try:
        up.UniversalPipeline._run_reconstruction([], opts, Path("."), hw)
    except _Stop:
        pass
    assert captured["backends"], "no backends were constructed"
    assert not any(isinstance(b, FakeReconstructionBackend) for b in captured["backends"])


def test_test_backend_refused_outside_test_env(client, tmp_path, monkeypatch):
    """REALITY_TEST_BACKEND must fail the job loudly unless REALITY_ENV=test."""
    monkeypatch.setenv("REALITY_ENV", "production")
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None
    sid = client.post("/api/sessions", json={"name": "Refuse fake"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Refuse world"}).json()["id"]
    assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
    for i in range(3):
        _upload_photo(client, sid, f"frame_{i:02d}.jpg", b"jpeg-" + f"{i}".encode() * 8)

    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"],
                    timeout=60.0)
    assert job["status"] == "failed"
    assert "refusing to run a test backend" in (job.get("error") or "")
