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


def test_retries_exhausted_is_failed(tmp_path):
    def boom(n):
        raise RuntimeError("x")

    st = Fn(StageSpec("a", "1", "cpu", True, idempotent=True, retries=1), boom)
    m = run_stages([st], _ctx(tmp_path))
    assert st.calls == 2 and m.stages[0].status == "failed"
