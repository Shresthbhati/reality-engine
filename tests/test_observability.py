"""T-6.6: a failed job is diagnosable from logs alone, and the API never shows a traceback or a server path to a user.

`job.error` is what the API returns; the full traceback goes to the structured log under the job id.
"""
import io
import json
import logging
from types import SimpleNamespace

import pytest

from apps.api.obs import JsonFormatter, clean_error, enter_stage, job_log


def _job(job_id="job_1"):
    return SimpleNamespace(id=job_id, stage=None, payload={})


def test_the_user_facing_error_has_no_traceback_and_no_server_path():
    try:
        raise FileNotFoundError(r"C:\Users\ops\reality-engine\data\artifacts\ab\cd.jpg not found in /srv/app/store")
    except FileNotFoundError as exc:
        msg = clean_error(exc)
    assert msg.startswith("FileNotFoundError")
    assert "Traceback" not in msg and "\n" not in msg
    assert "C:\\Users" not in msg and "/srv/app" not in msg and "cd.jpg" in msg


def test_the_error_is_short_and_a_blank_one_still_names_the_class():
    assert len(clean_error(RuntimeError("x" * 5000))) <= 300
    assert clean_error(ValueError("")) == "ValueError"


def test_stage_timings_accumulate_per_stage_and_close_on_none(monkeypatch):
    import apps.api.obs as obs

    clock = iter([0.0, 2.0, 5.0, 9.0])
    monkeypatch.setattr(obs.time, "monotonic", lambda: next(clock))
    j = _job()
    enter_stage(j, "resolving_evidence")        # t=0
    enter_stage(j, "analyzing_evidence")        # t=2: resolving took 2 s
    enter_stage(j, "committing_version")        # t=5: analyzing took 3 s
    enter_stage(j, None)                        # t=9: committing took 4 s
    assert j.payload["stage_timings"] == {"resolving_evidence": 2.0, "analyzing_evidence": 3.0, "committing_version": 4.0}
    assert j.stage is None


def test_json_logs_carry_the_job_id_the_stage_and_the_traceback():
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setFormatter(JsonFormatter())
    lg = logging.getLogger("reality.test.obs")
    lg.handlers, lg.propagate = [h], False
    lg.setLevel(logging.INFO)
    try:
        raise RuntimeError("COLMAP exited 1")
    except RuntimeError:
        job_log(lg, "job.failed", job_id="job_9", stage="analyzing_evidence", exc_info=True)
    rec = json.loads(stream.getvalue())
    assert rec["event"] == "job.failed" and rec["job_id"] == "job_9" and rec["stage"] == "analyzing_evidence"
    assert "RuntimeError: COLMAP exited 1" in rec["exception"] and rec["level"] == "ERROR"


def test_a_log_record_without_extras_is_still_valid_json():
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setFormatter(JsonFormatter())
    lg = logging.getLogger("reality.test.obs2")
    lg.handlers, lg.propagate = [h], False
    lg.warning("plain message")
    assert json.loads(stream.getvalue())["message"] == "plain message"


def test_a_failing_job_returns_a_clean_error_and_logs_the_traceback_under_its_id(tmp_path, monkeypatch, caplog):
    """Through the real job runner: the pipeline raises with a server path in its message; the API error is one clean line, the log has
    the traceback under the job id, and stage changes are logged."""
    import importlib
    import time

    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.jobs as jobs_mod
    import apps.api.worldstore_service as ws

    class Boom(RuntimeError):
        retryable = False                                  # fail now, do not loop the queue

    def explode(*a, **k):
        raise Boom(r"C:\Users\ops\reality-engine\data\artifacts\ab\cd.jpg could not be decoded")

    monkeypatch.setattr(jobs_mod, "run_progressive", explode)
    ws._store = None
    caplog.set_level(logging.INFO)
    with TestClient(main_mod.app) as c:
        sid = c.post("/api/sessions", json={"name": "C"}).json()["id"]
        wid = c.post("/api/worlds", json={"name": "W"}).json()["id"]
        c.post(f"/api/worlds/{wid}/attach/{sid}")
        for i in range(3):
            c.post(f"/api/uploads?session_id={sid}", files={"file": (f"f{i}.jpg", b"\xff\xd8\xff\xe0" + str(i).encode() * 8, "image/jpeg")})
        jid = c.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
        for _ in range(200):
            job = c.get(f"/api/jobs/{jid}").json()
            if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
                break
            time.sleep(0.25)
    assert job["status"] == "failed", job
    assert job["error"].startswith("Boom") and "Traceback" not in job["error"] and "\n" not in job["error"]
    assert "C:" not in job["error"] and "cd.jpg" in job["error"]
    failed = [r for r in caplog.records if getattr(r, "event", None) == "job.failed" and getattr(r, "job_id", None) == jid]
    assert failed and failed[0].exc_info, "the traceback is in the structured log under the job id"
    stages = [getattr(r, "stage", None) for r in caplog.records if getattr(r, "event", None) == "job.stage" and getattr(r, "job_id", None) == jid]
    assert stages, "stage changes are logged under the job id"
