"""P101: stage contract and runner (pure, no database)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.pipeline.stages.base import (
    StageContext,
    StageResult,
    StageSpec,
    compute_inputs_hash,
)
from engine.pipeline.stages.runner import StageCancelled, run_stages


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FakeStage:
    def __init__(self, name, status="succeeded", required=True, raises=None):
        self.spec = StageSpec(name=name, version="1", resource_class="cpu", required=required)
        self._status = status
        self._raises = raises
        self.calls = 0

    def run(self, ctx):
        self.calls += 1
        if self._raises:
            raise self._raises
        t = _now()
        return StageResult(
            name=self.spec.name, status=self._status, inputs_hash="h", outputs={},
            metrics={}, warnings=[], error=None, started_at=t, ended_at=t, peak_rss_mb=None,
        )


def _ctx(tmp_path: Path) -> StageContext:
    return StageContext(job_id="j1", workdir=tmp_path, artifacts={}, params={}, state={}, log=lambda m: None)


def _grade(stages, tmp_path):
    m = run_stages(stages, _ctx(tmp_path))
    return m, m.grade([s.spec for s in stages])


def test_runs_in_order(tmp_path):
    stages = [FakeStage("a"), FakeStage("b"), FakeStage("c")]
    m, g = _grade(stages, tmp_path)
    assert [r.name for r in m.stages] == ["a", "b", "c"]
    assert g == "succeeded"


def test_required_failure_stops_and_grades_failed(tmp_path):
    stages = [FakeStage("a"), FakeStage("b", status="failed"), FakeStage("c")]
    m, g = _grade(stages, tmp_path)
    assert [r.name for r in m.stages] == ["a", "b"]
    assert stages[2].calls == 0
    assert g == "failed"


def test_optional_failure_continues_and_grades_partial(tmp_path):
    stages = [FakeStage("a"), FakeStage("b", status="failed", required=False), FakeStage("c")]
    m, g = _grade(stages, tmp_path)
    assert [r.name for r in m.stages] == ["a", "b", "c"]
    assert g == "partial"


def test_degraded_grades_partial(tmp_path):
    _, g = _grade([FakeStage("a", status="degraded")], tmp_path)
    assert g == "partial"


def test_skipped_does_not_change_grade(tmp_path):
    _, g = _grade([FakeStage("a"), FakeStage("b", status="skipped")], tmp_path)
    assert g == "succeeded"
    _, g = _grade([FakeStage("a", status="degraded"), FakeStage("b", status="skipped")], tmp_path)
    assert g == "partial"


def test_cancel_before_second_stage_raises(tmp_path):
    stages = [FakeStage("a"), FakeStage("b")]
    seen = []
    with pytest.raises(StageCancelled):
        run_stages(stages, _ctx(tmp_path), should_cancel=lambda: bool(seen) or seen.append(1) or False,
                   on_end=lambda r: None)
    assert stages[0].calls == 1
    assert stages[1].calls == 0


def test_checkpoint_hit_skips_run(tmp_path):
    t = _now()
    cached = StageResult(name="a", status="succeeded", inputs_hash="cached", outputs={"x": "d" * 64},
                         metrics={}, warnings=[], error=None, started_at=t, ended_at=t, peak_rss_mb=None)

    class Store:
        def lookup(self, job_id, name, inputs_hash):
            return cached

        def record(self, job_id, result):
            raise AssertionError("must not re-record a hit")

    stage = FakeStage("a")
    m = run_stages([stage], _ctx(tmp_path), checkpoints=Store())
    assert stage.calls == 0
    assert m.stages == [cached]


def test_exception_becomes_failed_result(tmp_path):
    stages = [FakeStage("a", raises=ValueError("boom"), required=False), FakeStage("b")]
    m, g = _grade(stages, tmp_path)
    assert m.stages[0].status == "failed"
    assert m.stages[0].error == repr(ValueError("boom"))
    assert m.stages[1].status == "succeeded"
    assert g == "partial"


def test_inputs_hash_is_order_independent_sha256():
    h = compute_inputs_hash("1", {"a": "x", "b": "y"}, {"p": 1})
    assert h == compute_inputs_hash("1", {"b": "y", "a": "x"}, {"p": 1})
    assert len(h) == 64 and h == h.lower()
