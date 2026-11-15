"""P108.2: the dense stage (dense MVS + dense-structure pass + CPU plane sweep + fusion), run directly with the
deterministic backend of test_cli_compile. Layout/perception/mesh state is built by hand here (those stages are
extracted in later P108 commits); the Dense stage only reads their outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.compiler.world_compiler import CompileOptions, compile_reconstruction_to_world
from engine.pipeline.dataset import load_capture_dataset
from engine.pipeline.stages.runner import run_stages
from engine.pipeline.stages.s04_pose import Pose
from engine.pipeline.stages.s05_scale import Scale
from engine.pipeline.stages.s06_depth import Depth
from engine.pipeline.stages.s07_dense import Dense
from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice
from tests.test_cli_compile import _TwoViewBackend
from tests.test_stage_pose_scale import GOOD, _capture, _ctx

PIL = pytest.importorskip("PIL")


@dataclass(frozen=True)
class _Layout:
    """Stand-in for the layout stage output (the stage is extracted in a later P108 commit)."""

    world: object
    diagnostics: object
    compile_options: object


def _state_through_compile(tmp_path: Path, **opt_kw):
    items, refs, _, _ = load_capture_dataset(_capture(tmp_path, GOOD))
    opts = VerticalSliceOptions(reconstruction_backend=_TwoViewBackend(), measured_baselines=tuple(refs),
                                depth_model=None, **opt_kw)
    ctx = _ctx(items, opts)
    run_stages([Pose(), Scale(), Depth()], ctx)
    copts = CompileOptions(seed=opts.seed, up=opts.up, artifact_store=opts.artifact_store)
    world, diagnostics = compile_reconstruction_to_world(ctx.state["pose"].result, copts)
    ctx.state["layout"] = _Layout(world=world, diagnostics=diagnostics, compile_options=copts)
    ctx.state["perception"] = SimpleNamespace(facts={"status": "skipped", "objects": 0})
    ctx.state["mesh"] = SimpleNamespace(facts={"status": "skipped"})
    return items, opts, ctx


def _dense(ctx):
    return run_stages([Dense()], ctx).stages[0]


def test_spec() -> None:
    s = Dense.spec
    assert (s.name, s.version, s.resource_class, s.required) == ("dense", "1", "highmem", False)


def test_everything_off_is_skipped_with_every_reason(tmp_path: Path) -> None:
    items, opts, ctx = _state_through_compile(tmp_path)
    world_before = ctx.state["layout"].world
    result = _dense(ctx)
    assert result.status == "skipped"
    text = " | ".join(result.warnings)
    assert "dense_mvs: disabled" in text and "plane_sweep: disabled" in text and "fusion: skipped" in text
    assert ctx.state["layout"].world is world_before and ctx.state["dense"].structure_facts is None
    # the facts are the ones vertical_slice() reports for the same options
    ref = vertical_slice(items, opts)
    dn = ctx.state["dense"]
    assert dn.dense_mvs_facts == ref.stage_facts["dense_mvs"] and dn.plane_sweep_facts == ref.stage_facts["plane_sweep"]
    assert dn.dense_gate_facts == ref.stage_facts["dense_gate"] and ref.stage_facts["dense_structure"] is None


def test_fusion_that_runs_is_succeeded_and_attaches_the_pointcloud(tmp_path: Path) -> None:
    from reconstruction.backend.interface import ReconstructedPoint
    from world_ir.artifact_store import FileArtifactStore

    _, _, ctx = _state_through_compile(tmp_path, artifact_store=FileArtifactStore(tmp_path / "art"))
    ctx.state["pose"].result.points.extend(
        ReconstructedPoint(position=(0.1 * i, -1.0, 0.2 * i), track_id=f"depth-{i}", source_evidence_ids=["img_000"])
        for i in range(12))
    result = _dense(ctx)
    assert result.status == "succeeded", result.warnings
    assert ctx.state["dense"].fusion_facts["status"] == "ran"
    assert "geom-fused-pointcloud" in ctx.state["layout"].world.geometries
    assert result.metrics["fusion_associated"] >= 0.0


def test_failed_substep_is_degraded_never_success(tmp_path: Path, monkeypatch) -> None:
    import engine.pipeline.vertical_slice as vs

    monkeypatch.setattr(vs, "_dense_mvs_stage", lambda *a, **k: {"status": "failed", "note": "boom"})
    _, _, ctx = _state_through_compile(tmp_path)
    result = _dense(ctx)
    assert result.status == "degraded" and any("dense_mvs: failed (boom)" in w for w in result.warnings)


def test_judged_but_not_adopted_dense_candidate_still_ran(tmp_path: Path, monkeypatch) -> None:
    import engine.pipeline.vertical_slice as vs

    monkeypatch.setattr(vs, "_dense_mvs_stage", lambda *a, **k: {"status": "rejected", "note": "contradicts sparse"})
    _, _, ctx = _state_through_compile(tmp_path)
    result = _dense(ctx)
    assert result.status == "succeeded" and any("rejected" in w for w in result.warnings)
