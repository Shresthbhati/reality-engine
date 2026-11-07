"""Tests for structured job logs, error diagnostics, and per-stage timings (T-6.6).

Verifies:
1. GET /api/jobs/{job_id}/logs returns structured JSON entries with stage, level, message, context.
2. Filtering logs by level (?level=ERROR).
3. GET /api/jobs/{job_id}/diagnostics provides root-cause tracking, failure classification,
   and actionable remediation recommendations (a failed job is diagnosable from logs alone).
4. GET /api/jobs/{job_id}/timings reports per-stage durations, percentages, and bottleneck stage.
5. Strict tenant isolation: Tenant B receives 404 when querying Tenant A's job observability endpoints.
"""

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
from apps.api.models import Job, JobLog, Session, new_id, utcnow
from apps.api.observability_service import record_job_log

SECRET = "observability-test-secret"


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


def test_job_structured_logs_lifecycle_and_filtering(api):
    """Enqueuing and running a job writes structured logs that can be retrieved and filtered."""
    # 1. Create session and enqueue job for tenant A
    sess_res = api.call("tenant-a", "POST", "/api/sessions", json={"name": "Observability Session"})
    session_id = sess_res.json()["id"]

    import apps.api.db as db_mod
    from apps.api.jobs import enqueue_job

    async def _setup_job():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            j = enqueue_job(
                db,
                "PROCESS_EVIDENCE",
                "session",
                session_id,
                team_id="tenant-a",
                payload={"images_count": 5},
            )
            # Add some intermediate stage logs
            await record_job_log(
                db,
                j.id,
                "Extracting EXIF metadata from 5 images",
                stage="exif_extraction",
                level="INFO",
                context={"extracted": 5},
            )
            await record_job_log(
                db,
                j.id,
                "Image 3 has low lighting and motion blur",
                stage="quality_check",
                level="WARNING",
                context={"image": "img_003.jpg", "blur_score": 0.28},
            )
            await record_job_log(
                db,
                j.id,
                "Feature matching failed for image 4: zero keypoint matches",
                stage="feature_matching",
                level="ERROR",
                context={"image": "img_004.jpg", "matches": 0},
            )
            await db.commit()
            return j.id

    job_id = asyncio.run(_setup_job())

    # 2. Query all logs for this job
    res_logs = api.call("tenant-a", "GET", f"/api/jobs/{job_id}/logs")
    assert res_logs.status_code == 200, res_logs.text
    data = res_logs.json()
    assert data["job_id"] == job_id
    assert data["total"] >= 4
    stages = [log["stage"] for log in data["logs"]]
    assert "queued" in stages
    assert "exif_extraction" in stages
    assert "quality_check" in stages
    assert "feature_matching" in stages

    # Check structure of a log entry
    err_log = next(log for log in data["logs"] if log["level"] == "ERROR")
    assert err_log["stage"] == "feature_matching"
    assert "zero keypoint matches" in err_log["message"]
    assert err_log["context"]["matches"] == 0

    # 3. Filter logs by level=ERROR
    res_errs = api.call("tenant-a", "GET", f"/api/jobs/{job_id}/logs?level=ERROR")
    assert res_errs.status_code == 200
    err_data = res_errs.json()
    assert err_data["total"] == 1
    assert err_data["logs"][0]["level"] == "ERROR"


def test_failed_job_diagnostics_and_root_cause_analysis(api):
    """A failed job must be diagnosable from logs and diagnostics alone without host inspection."""
    import apps.api.db as db_mod

    sess_id = "s_fail_diagnose"
    job_id = "job_failed_insufficient_evidence"

    async def _setup_failed_job():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            s = Session(id=sess_id, name="Failed Session", team_id="tenant-a")
            db.add(s)

            now = utcnow()
            j = Job(
                id=job_id,
                team_id="tenant-a",
                type="RECONSTRUCT_SESSION",
                entity_type="session",
                entity_id=sess_id,
                status="failed",
                stage="reconstructing_room",
                error="InsufficientEvidence: only 1 photo provided; multi-view stereo requires >= 3 views",
                created_at=now - timedelta(minutes=2),
                started_at=now - timedelta(minutes=2),
                completed_at=now - timedelta(minutes=1),
                payload={"failure_kind": "needs_more_evidence", "traceback_logged": True},
            )
            db.add(j)

            # Record diagnostic logs leading up to failure
            log1 = JobLog(
                id=new_id("log"),
                job_id=job_id,
                stage="resolving_evidence",
                level="INFO",
                message="Loaded 1 evidence item for session",
                context_json={"photos": 1},
            )
            log2 = JobLog(
                id=new_id("log"),
                job_id=job_id,
                stage="reconstructing_room",
                level="ERROR",
                message="Cannot compute epipolar geometry: insufficient camera poses (needed >= 3, got 1)",
                context_json={"photos_available": 1, "photos_required": 3},
            )
            db.add_all([log1, log2])
            await db.commit()

    asyncio.run(_setup_failed_job())

    # Query diagnostics endpoint
    diag_res = api.call("tenant-a", "GET", f"/api/jobs/{job_id}/diagnostics")
    assert diag_res.status_code == 200, diag_res.text
    diag = diag_res.json()

    # CORE ASSERTION: a failed job is diagnosable from logs alone
    assert diag["diagnosable"] is True
    assert diag["status"] == "failed"
    assert diag["stage"] == "reconstructing_room"
    assert diag["classification"] == "INSUFFICIENT_EVIDENCE"

    # Root cause details
    root_cause = diag["root_cause"]
    assert root_cause is not None
    assert root_cause["stage_at_failure"] == "reconstructing_room"
    assert "InsufficientEvidence" in root_cause["error_summary"]
    assert root_cause["primary_error_log"] is not None
    assert "epipolar geometry" in root_cause["primary_error_log"]["message"]

    # Actionable remediation advice present for the user/operator
    assert diag["remediation_suggestion"] is not None
    assert "overlapping photographs" in diag["remediation_suggestion"]


def test_job_stage_timings_dashboard(api):
    """Timing dashboard breaks down duration across pipeline stages and flags the slowest stage."""
    import apps.api.db as db_mod

    job_id = "job_timing_dashboard"
    sess_id = "s_timing"

    async def _setup_job_with_timings():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            s = Session(id=sess_id, name="Timing Session", team_id="tenant-a")
            db.add(s)

            now = utcnow()
            j = Job(
                id=job_id,
                team_id="tenant-a",
                type="RECONSTRUCT_SESSION",
                entity_type="session",
                entity_id=sess_id,
                status="succeeded",
                stage="completed",
                created_at=now - timedelta(seconds=60),
                started_at=now - timedelta(seconds=55),
                completed_at=now - timedelta(seconds=5),
                payload={
                    "stages": {
                        "timings_s": {
                            "bootstrap": 2.5,
                            "colmap_sfm": 25.0,
                            "depth_fusion": 15.0,
                            "mesh_generation": 7.5,
                        }
                    }
                },
            )
            db.add(j)
            await db.commit()

    asyncio.run(_setup_job_with_timings())

    res = api.call("tenant-a", "GET", f"/api/jobs/{job_id}/timings")
    assert res.status_code == 200, res.text
    timings = res.json()

    assert timings["job_id"] == job_id
    assert timings["total_duration_s"] == 50.0
    stage_map = timings["stage_timings_s"]
    assert stage_map["bootstrap"] == 2.5
    assert stage_map["colmap_sfm"] == 25.0
    assert stage_map["depth_fusion"] == 15.0

    percentages = timings["stage_percentages"]
    assert percentages["colmap_sfm"] == 50.0  # 25s / 50s = 50%
    assert percentages["depth_fusion"] == 30.0  # 15s / 50s = 30%

    slowest = timings["slowest_stage"]
    assert slowest["stage"] == "colmap_sfm"
    assert slowest["duration_s"] == 25.0
    assert slowest["percentage"] == 50.0


def test_observability_tenant_isolation(api):
    """Tenant B cannot access Tenant A's job logs, diagnostics, or timings."""
    import apps.api.db as db_mod

    job_id = "job_tenant_a_secret"
    sess_id = "s_tenant_a_secret"

    async def _setup_tenant_a_job():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            s = Session(id=sess_id, name="Private Session", team_id="tenant-a")
            j = Job(
                id=job_id,
                team_id="tenant-a",
                type="PROCESS_EVIDENCE",
                entity_type="session",
                entity_id=sess_id,
                status="succeeded",
            )
            log_entry = JobLog(
                id=new_id("log"),
                job_id=job_id,
                stage="queued",
                level="INFO",
                message="Private tenant log entry",
            )
            db.add_all([s, j, log_entry])
            await db.commit()

    asyncio.run(_setup_tenant_a_job())

    # Tenant B tries to reach Tenant A's logs, diagnostics, timings -> all 404
    assert api.call("tenant-b", "GET", f"/api/jobs/{job_id}/logs").status_code == 404
    assert api.call("tenant-b", "GET", f"/api/jobs/{job_id}/diagnostics").status_code == 404
    assert api.call("tenant-b", "GET", f"/api/jobs/{job_id}/timings").status_code == 404

    # Admin has access
    assert api.call("root", "GET", f"/api/jobs/{job_id}/logs").status_code == 200
    assert api.call("root", "GET", f"/api/jobs/{job_id}/diagnostics").status_code == 200
    assert api.call("root", "GET", f"/api/jobs/{job_id}/timings").status_code == 200
