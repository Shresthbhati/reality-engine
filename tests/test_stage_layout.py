"""P108.5: the layout stage (single-image layout evidence + the structure compile), and the whole stage chain against
``vertical_slice`` on the deterministic backend of test_cli_compile."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from engine.pipeline.dataset import load_capture_dataset
from engine.pipeline.stages.runner import run_stages
from engine.pipeline.stages.s04_pose import Pose
from engine.pipeline.stages.s05_scale import Scale
from engine.pipeline.stages.s06_depth import Depth
from engine.pipeline.stages.s07_dense import Dense
from engine.pipeline.stages.s08_mesh import Mesh
from engine.pipeline.stages.s09_perception import Perception
from engine.pipeline.stages.s10_layout import Layout
from engine.pipeline.vertical_slice import VerticalSliceError, VerticalSliceOptions, vertical_slice
from tests.test_cli_compile import _TwoViewBackend
from tests.test_stage_pose_scale import GOOD, _capture, _ctx

pytest.importorskip("PIL")

CHAIN = [Pose, Scale, Depth, Layout, Perception, Mesh, Dense]
SPECS = [c.spec for c in CHAIN]


def _setup(tmp_path: Path, backend=None, **opt_kw):
    items, refs, _, _ = load_capture_dataset(_capture(tmp_path, GOOD))
    opts = VerticalSliceOptions(reconstruction_backend=backend or _TwoViewBackend(), measured_baselines=tuple(refs),
                                depth_model=None, perception_model=None, mesh_enabled=False, detail_enabled=False,
                                **opt_kw)
    return items, opts, _ctx(items, opts)


def test_spec() -> None:
    s = Layout.spec
    assert (s.name, s.version, s.resource_class, s.required) == ("layout", "1", "cpu", True)


def test_layout_compiles_the_same_world_as_vertical_slice(tmp_path: Path) -> None:
    items, opts, ctx = _setup(tmp_path)
    manifest = run_stages([Pose(), Scale(), Depth(), Layout()], ctx)
    layout_result = manifest.stages[3]
    assert layout_result.name == "layout" and layout_result.status == "succeeded"
    ref = vertical_slice(items, opts)
    out = ctx.state["layout"]
    assert sorted(out.world.entities) == sorted(ref.world.entities) and len(out.world.entities) > 0
    assert out.diagnostics.rooms_detected == ref.compile.rooms_detected
    assert out.diagnostics.measurements_count == ref.compile.measurements_count
    assert out.layout_facts == ref.stage_facts["layout_evidence"]
    # no layout/<stem>.json sidecars and no learned estimator: reported, not graded as damage
    assert out.layout_facts["status"] == "unavailable" and layout_result.warnings


def test_whole_chain_matches_vertical_slice_and_every_optional_stage_is_skipped(tmp_path: Path) -> None:
    items, opts, ctx = _setup(tmp_path)
    manifest = run_stages([c() for c in CHAIN], ctx)
    assert [(r.name, r.status) for r in manifest.stages] == [
        ("pose", "succeeded"), ("scale", "succeeded"), ("depth", "skipped"), ("layout", "succeeded"),
        ("perception", "skipped"), ("mesh", "skipped"), ("dense", "skipped")]
    assert manifest.grade(SPECS) == "succeeded"
    ref = vertical_slice(items, opts)
    assert len(ctx.state["layout"].world.entities) == len(ref.world.entities)
    assert tuple(tuple(float(c) for c in p.position) for p in ctx.state["pose"].result.points) == ref.points


def test_compile_failure_fails_the_required_stage_and_stops_the_run(tmp_path: Path, monkeypatch) -> None:
    import engine.pipeline.vertical_slice as vs

    def boom(*a, **k):
        raise RuntimeError("compiler exploded")

    monkeypatch.setattr(vs, "compile_reconstruction_to_world", boom)
    _, _, ctx = _setup(tmp_path)
    manifest = run_stages([c() for c in CHAIN], ctx)
    assert [(r.name, r.status) for r in manifest.stages][-1] == ("layout", "failed")
    assert "world compile stage failed: compiler exploded" in manifest.stages[-1].error
    assert manifest.grade(SPECS) == "failed" and len(manifest.stages) == 4
    with pytest.raises(VerticalSliceError, match="world compile stage failed"):
        Layout().run(ctx)


@dataclass(frozen=True)
class _HookBackend(_TwoViewBackend):
    seen: list = None

    def post_compile_hook(self, world, result, evidence_items):
        self.seen.append((len(world.entities), len(result.points), len(evidence_items)))


def test_backend_post_compile_hook_is_called_once_with_the_compiled_world(tmp_path: Path) -> None:
    hook = _HookBackend(seen=[])
    items, _, ctx = _setup(tmp_path, backend=hook)
    run_stages([Pose(), Scale(), Depth(), Layout()], ctx)
    assert hook.seen == [(len(ctx.state["layout"].world.entities), len(ctx.state["pose"].result.points), 3)]
