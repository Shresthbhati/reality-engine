"""B-00 / B-01: the API stays responsive during a job, and job progress is truthful.

Pins: GET routes answer while a blocking backend runs (B-00: the 2026-10-13 probe on a real 10-photo job saw a worst
read of 0.06 s, so this is a regression guard, not a fix); progress is monotonic and 100 only on success/partial;
a settled job never reports a working stage (B-01 / O8); the SSE stream ends on the terminal event.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from tests.test_cli_compile import _TwoViewBackend  # noqa: E402
from tests.test_reconstruction_lifecycle_semantics import _setup, _wait, client  # noqa: E402,F401


@dataclass(frozen=True)
class _SlowBackend(_TwoViewBackend):
    """Blocks a worker thread for a while, like a real COLMAP subprocess wait, then registers like the two-view fake.

    It must register cameras: a backend that returns nothing falls through to the single-image fallback, which loads
    the real MiDaS depth model on any decodable photograph and makes a "fast offline" test slow and load-dependent."""

    def reconstruct(self, evidence):
        time.sleep(3)
        return super().reconstruct(evidence)


def test_reads_stay_fast_while_a_job_blocks(client, tmp_path, monkeypatch):
    sid, _ = _setup(client, tmp_path, monkeypatch, spec="tests.test_job_progress:_SlowBackend")
    job_id = client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
    worst = 0.0
    deadline = time.time() + 30
    seen_running = False
    while time.time() < deadline:
        t0 = time.time()
        assert client.get("/api/worlds").status_code == 200
        worst = max(worst, time.time() - t0)
        job = client.get(f"/api/jobs/{job_id}").json()
        seen_running = seen_running or job["status"] == "running"
        if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
            break
        time.sleep(0.1)
    assert seen_running, "the job never ran, so responsiveness was not exercised"
    assert worst < 0.5, f"a read took {worst:.2f}s while a job ran"


def test_progress_is_monotonic_and_settles_truthfully(client, tmp_path, monkeypatch):
    sid, _ = _setup(client, tmp_path, monkeypatch)
    job_id = client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
    seen: list[float] = []
    deadline = time.time() + 120
    job = {}
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        seen.append(job["progress"] or 0.0)
        if job["terminal"]:
            break
        time.sleep(0.05)
    assert seen == sorted(seen), f"progress went backwards: {seen}"
    assert job["status"] in ("succeeded", "partial")
    assert job["progress"] == 100.0
    assert job["stage"] in ("complete", "partial")
    assert job["stage"] == ("complete" if job["status"] == "succeeded" else "partial")
    assert job["photos_total"] == 3


def test_failed_job_is_not_100_percent_and_has_failed_stage(client, tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "1")
    sid, _ = _setup(client, tmp_path, monkeypatch,
                    spec="tests.test_reconstruction_lifecycle_semantics:_HangingBackend")
    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] == "failed"
    assert job["stage"] == "failed"
    assert job["progress"] is not None and job["progress"] < 100.0
    assert job["terminal"] is True


def test_events_stream_ends_on_terminal_event(client, tmp_path, monkeypatch):
    sid, _ = _setup(client, tmp_path, monkeypatch)
    job_id = client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events?interval=0.05") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert events, "no events"
    assert events[-1]["terminal"] is True
    assert [e["progress"] or 0 for e in events] == sorted(e["progress"] or 0 for e in events)


def test_events_for_unknown_job_is_404(client):
    assert client.get("/api/jobs/job_nope/events").status_code == 404
