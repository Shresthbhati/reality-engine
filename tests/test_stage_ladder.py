"""P108.6: the fine stages run INSIDE the progressive ladder's multi-view rung; the ladder (and its single-image
fallback) is kept. Fine stages are reported in the outer manifest: real results when the rung was used, ``skipped``
with a reason when the ladder fell back -- so a fallback grades ``partial``, never ``failed``."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.pipeline.dataset import load_capture_dataset
from engine.pipeline.stages.base import StageContext, StageResult
from engine.pipeline.stages.ladder import Ladder, run_fine_stages
from engine.pipeline.stages.registry import PIPELINE_PROFILES, PROFILE_NAME, SegmentReport, build_stages
from engine.pipeline.stages.runner import run_stages
from engine.pipeline.vertical_slice import VerticalSliceError, VerticalSliceOptions, vertical_slice
from tests.test_cli_compile import _TwoViewBackend
from tests.test_stage_pose_scale import GOOD, _capture

pytest.importorskip("PIL")

FINE = ["pose", "scale", "depth", "layout", "perception", "mesh", "dense", "detail", "finalize"]


def _setup(tmp_path: Path, backend=None):
    items, refs, _, _ = load_capture_dataset(_capture(tmp_path, GOOD))
    opts = VerticalSliceOptions(reconstruction_backend=backend or _TwoViewBackend(), measured_baselines=tuple(refs),
                                depth_model=None, perception_model=None, mesh_enabled=False, detail_enabled=False)
    return items, opts


def test_profile_lists_ladder_then_the_fine_stages_then_the_tail() -> None:
    names = PIPELINE_PROFILES[PROFILE_NAME]
    assert names == ["evidence_resolve", "evidence_quality", "reconstruct", *FINE, "quality_report", "compile_validate",
                     "changeset_commit"]
    assert [s.spec.name for s in build_stages(PROFILE_NAME)] == names
    assert "reconstruct_legacy" not in names


def test_staged_slice_matches_vertical_slice_and_fills_the_sink(tmp_path: Path) -> None:
    items, opts = _setup(tmp_path)
    sink: dict = {}
    got = run_fine_stages(items, opts, sink)
    ref = vertical_slice(items, opts)
    assert list(sink) == FINE
    assert [sink[n].status for n in FINE] == ["succeeded", "succeeded", "skipped", "succeeded", "skipped", "skipped",
                                              "skipped", "skipped", "succeeded"]
    assert sorted(got.world.entities) == sorted(ref.world.entities) and len(got.world.entities) > 0
    assert (got.points, got.camera_poses, got.scale_state, got.meters_per_unit, got.registration_status) == (
        ref.points, ref.camera_poses, ref.scale_state, ref.meters_per_unit, ref.registration_status)
    assert set(got.stage_facts) == set(ref.stage_facts)
    assert {"reconstruction", "scale", "frame", "depth", "compile", "perception", "mesh", "dense",
            "fusion_and_detail", "diagnostics_and_correction_plan"} <= set(got.stage_facts["stage_timings_s"])
    assert got.world.metadata["scale"]["state"] == ref.world.metadata["scale"]["state"]


def test_a_failing_stage_raises_the_original_exception_so_the_ladder_can_fall_back(tmp_path: Path, monkeypatch) -> None:
    import engine.pipeline.vertical_slice as vs

    monkeypatch.setattr(vs, "compile_reconstruction_to_world", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    items, opts = _setup(tmp_path)
    with pytest.raises(VerticalSliceError, match="world compile stage failed: x"):
        run_fine_stages(items, opts, {})


def _prog(vs_result, attempts):
    return SimpleNamespace(vs_result=vs_result, attempts=attempts)


def _ladder_run(ladder_for, warnings=()):
    """Reconstruct (host segment) followed by the fine-stage reporters, as in the job profile."""
    stages = [s for s in build_stages(PROFILE_NAME) if s.spec.name == "reconstruct" or s.spec.name in FINE]

    def seg(ctx):
        ctx.state["ladder"] = ladder_for()
        return SegmentReport(warnings=list(warnings))

    ctx = StageContext(job_id="j", workdir=Path("."), artifacts={}, params={}, state={"segments": {"reconstruct": seg}})
    return run_stages(stages, ctx), [s.spec for s in stages]


def test_single_image_fallback_reports_every_fine_stage_skipped_with_a_reason_and_grades_partial() -> None:
    attempts = [{"level": 2, "outcome": "failed", "detail": "VerticalSliceError: need >= 2 photos"}]
    manifest, specs = _ladder_run(lambda: Ladder(prog=_prog(None, attempts), fine={}), warnings=["single-view hypothesis"])
    by = {r.name: r for r in manifest.stages}
    assert by["reconstruct"].status == "degraded"
    for n in FINE:
        assert by[n].status == "skipped" and "need >= 2 photos" in by[n].warnings[0], n
    assert manifest.grade(specs) == "partial"


def test_a_rejected_multiview_rung_is_skipped_even_though_its_stages_ran() -> None:
    now = datetime.now(timezone.utc)
    done = {"pose": StageResult(name="pose", status="succeeded", inputs_hash="", outputs={}, metrics={}, warnings=[],
                                error=None, started_at=now, ended_at=now, peak_rss_mb=None)}
    attempts = [{"level": 2, "outcome": "rejected", "detail": "not credible (only 2 cameras registered)"}]
    manifest, specs = _ladder_run(lambda: Ladder(prog=_prog(None, attempts), fine=done), warnings=["single-view"])
    by = {r.name: r for r in manifest.stages}
    assert by["pose"].status == "skipped" and "rejected" in by["pose"].warnings[0]
    assert manifest.grade(specs) == "partial"


def test_no_multiview_attempt_is_explained() -> None:
    manifest, _ = _ladder_run(lambda: Ladder(prog=_prog(None, []), fine={}), warnings=["x"])
    assert "not attempted" in {r.name: r for r in manifest.stages}["pose"].warnings[0]


def test_used_rung_reports_the_real_stage_results_and_grades_succeeded(tmp_path: Path) -> None:
    items, opts = _setup(tmp_path)
    sink: dict = {}
    vs = run_fine_stages(items, opts, sink)
    manifest, specs = _ladder_run(lambda: Ladder(prog=_prog(vs, []), fine=sink))
    by = {r.name: r for r in manifest.stages}
    assert [by[n].status for n in FINE] == [sink[n].status for n in FINE]
    assert by["pose"].metrics == sink["pose"].metrics and by["pose"].started_at == sink["pose"].started_at
    assert manifest.grade(specs) == "succeeded"


@dataclass(frozen=True)
class _RaisingHook(_TwoViewBackend):
    def post_compile_hook(self, world, result, evidence_items):
        raise RuntimeError("hook exploded")


def test_a_raising_post_compile_hook_is_logged_not_a_nameerror(tmp_path: Path, caplog) -> None:
    """Regression: ``log`` was undefined in vertical_slice.py, so the except branch itself raised NameError."""
    items, opts = _setup(tmp_path, backend=_RaisingHook())
    with caplog.at_level("WARNING"):
        out = vertical_slice(items, opts)
    assert len(out.world.entities) > 0
    assert any("post_compile_hook failed: hook exploded" in r.getMessage() for r in caplog.records)
