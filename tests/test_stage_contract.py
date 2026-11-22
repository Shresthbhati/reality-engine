"""Stage contract: required outputs are enforced and only idempotent stages are retried."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.runner import run_stages


def _result(name, status="succeeded", outputs=None):
    now = datetime.now(timezone.utc)
    return StageResult(name=name, status=status, inputs_hash="", outputs=dict(outputs or {}), metrics={}, warnings=[], error=None,
                       started_at=now, ended_at=now, peak_rss_mb=None)


class Fn:
    def __init__(self, spec, fn):
        self.spec, self.fn, self.calls = spec, fn, 0

    def run(self, ctx):
        self.calls += 1
        return self.fn(self.calls)


def _ctx(tmp_path: Path):
    return StageContext(job_id="j", workdir=tmp_path, artifacts={}, params={})


def test_success_without_required_outputs_is_failed(tmp_path):
    st = Fn(StageSpec("a", "1", "cpu", True, requires_outputs=("world",)), lambda n: _result("a", "succeeded", {}))
    m = run_stages([st], _ctx(tmp_path))
    assert m.stages[0].status == "failed" and "world" in m.stages[0].error
    assert m.grade([st.spec]) == "failed"


def test_required_outputs_present_passes(tmp_path):
    st = Fn(StageSpec("a", "1", "cpu", True, requires_outputs=("world",)), lambda n: _result("a", "succeeded", {"world": "h"}))
    assert run_stages([st], _ctx(tmp_path)).stages[0].status == "succeeded"


def test_idempotent_stage_is_retried_and_records_attempts(tmp_path):
    def flaky(n):
        if n < 3:
            raise RuntimeError("transient")
        return _result("a")

    st = Fn(StageSpec("a", "1", "cpu", True, idempotent=True, retries=2), flaky)
    m = run_stages([st], _ctx(tmp_path))
    assert st.calls == 3 and m.stages[0].status == "succeeded" and m.stages[0].metrics["attempts"] == 3.0


def test_non_idempotent_stage_is_never_retried(tmp_path):
    def boom(n):
        raise RuntimeError("x")

    st = Fn(StageSpec("a", "1", "cpu", True, retries=5), boom)  # retries ignored without idempotent=True
    m = run_stages([st], _ctx(tmp_path))
    assert st.calls == 1 and m.stages[0].status == "failed"


def test_stage_deadline_fails_the_stage_and_is_never_retried(tmp_path):
    import threading

    release = threading.Event()

    def slow(n):
        release.wait(5)
        return _result("a")

    st = Fn(StageSpec("a", "1", "cpu", True, idempotent=True, retries=3), slow)
    try:
        m = run_stages([st], _ctx(tmp_path), stage_timeouts={"a": 0.1})
    finally:
        release.set()
    assert st.calls == 1  # abandoned, not retried, even though idempotent
    r = m.stages[0]
    assert r.status == "failed" and "deadline" in r.error and r.metrics["timed_out"] == 1.0


def test_stage_without_a_deadline_runs_inline(tmp_path):
    st = Fn(StageSpec("a", "1", "cpu", True), lambda n: _result("a"))
    assert run_stages([st], _ctx(tmp_path), stage_timeouts={"other": 0.1}).stages[0].status == "succeeded"


def test_stage_timeouts_env_is_parsed_and_malformed_entries_are_ignored(monkeypatch):
    from apps.api.jobs import _stage_timeouts

    monkeypatch.setenv("RE_STAGE_TIMEOUTS_S", "reconstruct=3600, compile_validate=300,bad,x=notnum,y=-1")
    assert _stage_timeouts() == {"reconstruct": 3600.0, "compile_validate": 300.0}
    monkeypatch.delenv("RE_STAGE_TIMEOUTS_S")
    assert _stage_timeouts() == {}


def test_retries_exhausted_is_failed(tmp_path):
    def boom(n):
        raise RuntimeError("x")

    st = Fn(StageSpec("a", "1", "cpu", True, idempotent=True, retries=1), boom)
    m = run_stages([st], _ctx(tmp_path))
    assert st.calls == 2 and m.stages[0].status == "failed"
