"""Workstream 4: memory-aware stage admission, resource release and per-stage RSS (fake readers; no models, no GPU)."""
from __future__ import annotations

from datetime import datetime, timezone

from engine.hardware import admission as adm
from engine.hardware.admission import MemorySnapshot, StageAdmission, StageNeed
from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.runner import run_stages


def _now():
    return datetime.now(timezone.utc)


class Stage:
    def __init__(self, name, *, required=True, raises=None, status="succeeded"):
        self.spec = StageSpec(name=name, version="1", resource_class="cpu", required=required)
        self.calls, self._raises, self._status = 0, raises, status

    def run(self, ctx):
        self.calls += 1
        if self._raises:
            raise self._raises
        t = _now()
        return StageResult(self.spec.name, self._status, "", {}, {}, [], None, t, t, None)


def _snap(ram=8000.0, vram=None, rss=500.0):
    return MemorySnapshot(avail_ram_mb=ram, rss_mb=rss, free_vram_mb=vram, vram_source="fake")


def _admission(snaps, needs, **kw):
    seq = list(snaps)
    released = []
    a = StageAdmission(needs=needs, reserve_mb=500, max_wait_s=kw.pop("max_wait_s", 4), poll_s=1,
                       reader=lambda: seq.pop(0) if len(seq) > 1 else seq[0], sleep=lambda s: None,
                       release=lambda: released.append(1), sampler_interval_s=0.01, **kw)
    return a, released


def _run(stages, a, tmp_path):
    ctx = StageContext(job_id="j", workdir=tmp_path, artifacts={}, params={}, state={})
    m = run_stages(stages, ctx, admission=a)
    return m, ctx, m.grade([s.spec for s in stages])


def test_admitted_stage_records_rss_and_releases_after_heavy_stages(tmp_path):
    a, released = _admission([_snap()], {"a": StageNeed(ram_mb=1000)})
    m, ctx, grade = _run([Stage("a"), Stage("b")], a, tmp_path)
    assert grade == "succeeded" and len(released) == 1, "released after the heavy stage only (gc is GIL-bound)"
    r = m.stages[0]
    assert r.metrics["rss_before_mb"] == 500.0 and r.metrics["avail_ram_before_mb"] == 8000.0
    assert r.metrics["rss_after_mb"] > 0 and r.peak_rss_mb and r.peak_rss_mb > 0
    assert ctx.state["admission"]["a"]["decision"] == "admit"


def test_insufficient_ram_fails_the_stage_with_numbers_and_never_runs_it(tmp_path):
    a, _ = _admission([_snap(ram=900)], {"heavy": StageNeed(ram_mb=2500)})
    stages = [Stage("heavy"), Stage("after")]
    m, _, grade = _run(stages, a, tmp_path)
    r = m.stages[0]
    assert r.status == "failed" and "admission refused" in r.error and "need 2500 MB" in r.error and "900 MB available" in r.error
    assert r.metrics["admission_refused"] == 1.0 and r.metrics["admission_waited_s"] == 4.0
    assert stages[0].calls == 0 and stages[1].calls == 0, "refused stage not run; required failure stops the run"
    assert grade == "failed"


def test_deferred_stage_runs_once_memory_frees_up(tmp_path):
    a, released = _admission([_snap(ram=900), _snap(ram=900), _snap(ram=6000)], {"heavy": StageNeed(ram_mb=2500)})
    stage = Stage("heavy")
    m, ctx, grade = _run([stage], a, tmp_path)
    assert grade == "succeeded" and stage.calls == 1
    assert m.stages[0].metrics["admission_waited_s"] == 2.0 and "after waiting" in ctx.state["admission"]["heavy"]["reason"]
    assert len(released) >= 3, "each deferral poll collects garbage before re-reading memory"


def test_degrade_records_fallback_and_quality_effect_and_grades_partial(tmp_path):
    need = StageNeed(ram_mb=2500, fallback="half_resolution", fallback_ram_mb=800, quality_effect="coarser depth, +-4 cm")
    a, _ = _admission([_snap(ram=1500)], {"depth": need})
    m, ctx, grade = _run([Stage("depth", required=False)], a, tmp_path)
    r = m.stages[0]
    assert r.status == "degraded" and grade == "partial"
    assert "half_resolution" in r.warnings[0] and "coarser depth" in r.warnings[0]
    assert ctx.state["admission"]["depth"]["fallback"] == "half_resolution"


def test_vram_need_without_a_gpu_reading_is_refused_not_assumed(tmp_path):
    a, _ = _admission([_snap(vram=None)], {"gpu": StageNeed(vram_mb=1700)}, max_wait_s=0)
    m, _, grade = _run([Stage("gpu")], a, tmp_path)
    assert m.stages[0].status == "failed" and "VRAM" in m.stages[0].error and grade == "failed"
    a2, _ = _admission([_snap(vram=1000)], {"gpu": StageNeed(vram_mb=1700)}, max_wait_s=0)
    assert _run([Stage("gpu")], a2, tmp_path)[2] == "failed"
    a3, _ = _admission([_snap(vram=5000)], {"gpu": StageNeed(vram_mb=1700)}, max_wait_s=0)
    assert _run([Stage("gpu")], a3, tmp_path)[2] == "succeeded"


def test_record_mode_measures_but_never_refuses(tmp_path):
    a, _ = _admission([_snap(ram=100)], {"heavy": StageNeed(ram_mb=2500)}, mode="record")
    m, _, grade = _run([Stage("heavy")], a, tmp_path)
    assert grade == "succeeded" and "would refuse" in a.admit("heavy").reason and m.stages[0].metrics["avail_ram_before_mb"] == 100.0


def test_unknown_ram_reading_does_not_block(tmp_path):
    a, _ = _admission([_snap(ram=None)], {"heavy": StageNeed(ram_mb=2500)})
    assert _run([Stage("heavy")], a, tmp_path)[2] == "succeeded"


def test_a_failing_stage_still_releases_and_records_after_metrics(tmp_path):
    a, released = _admission([_snap()], {})
    m, _, grade = _run([Stage("boom", raises=RuntimeError("oom"))], a, tmp_path)
    assert grade == "failed" and released and "rss_after_mb" in m.stages[0].metrics


def test_checkpoint_hits_are_not_admitted(tmp_path):
    class Hit:
        def lookup(self, job_id, name, h):
            t = _now()
            return StageResult(name, "succeeded", h, {}, {}, [], None, t, t, None)

        def record(self, job_id, result):
            raise AssertionError("a reused stage is not re-recorded")

    a, released = _admission([_snap(ram=1)], {"heavy": StageNeed(ram_mb=9999)})
    ctx = StageContext(job_id="j", workdir=tmp_path, artifacts={}, params={}, state={})
    m = run_stages([Stage("heavy")], ctx, checkpoints=Hit(), admission=a)
    assert m.stages[0].status == "succeeded" and not released


def test_real_readers_return_sane_numbers_and_env_modes(monkeypatch):
    assert adm.read_rss_mb() and adm.read_rss_mb() > 5
    assert adm.read_available_ram_mb() and adm.read_available_ram_mb() > 1
    adm.release_resources()  # must not raise and must not import torch
    monkeypatch.setenv("RE_MEMORY_ADMISSION", "off")
    assert StageAdmission.from_env() is None
    monkeypatch.setenv("RE_MEMORY_ADMISSION", "bogus")
    assert StageAdmission.from_env().mode == "enforce"
