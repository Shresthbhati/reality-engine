"""P108.1: the depth stage, run directly with the deterministic backend of test_cli_compile.

The Scale stage now also carries frame canonicalization (stage 2.5 of vertical_slice; plan 3.4 has no stage of its own
for it), so the depth stage always sees the canonical frame, exactly as ``vertical_slice`` does.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.pipeline.dataset import load_capture_dataset
from engine.pipeline.stages.runner import run_stages
from engine.pipeline.stages.s04_pose import Pose
from engine.pipeline.stages.s05_scale import Scale
from engine.pipeline.stages.s06_depth import Depth
from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice
from tests.test_cli_compile import _TwoViewBackend
from tests.test_stage_pose_scale import GOOD, _capture, _ctx

PIL = pytest.importorskip("PIL")

SPECS = [Pose.spec, Scale.spec, Depth.spec]


def _run(tmp_path: Path, **opt_kw):
    items, refs, _, _ = load_capture_dataset(_capture(tmp_path, GOOD))
    opts = VerticalSliceOptions(reconstruction_backend=_TwoViewBackend(), measured_baselines=tuple(refs), **opt_kw)
    ctx = _ctx(items, opts)
    return items, opts, ctx, run_stages([Pose(), Scale(), Depth()], ctx)


def test_spec() -> None:
    s = Depth.spec
    assert (s.name, s.version, s.resource_class, s.required) == ("depth", "1", "gpu", False)


def test_disabled_by_options_is_skipped_with_reason(tmp_path: Path) -> None:
    _, _, ctx, manifest = _run(tmp_path, depth_model=None)
    depth = manifest.stages[2]
    assert (depth.name, depth.status) == ("depth", "skipped")
    assert any("disabled (depth_model=None)" in w for w in depth.warnings)
    assert ctx.state["depth"].metric_maps == []
    assert manifest.grade(SPECS) == "succeeded"          # skipped never upgrades or downgrades


def test_missing_intrinsics_is_skipped_not_succeeded(tmp_path: Path) -> None:
    _, _, _, manifest = _run(tmp_path)                     # default depth_model, but no trusted intrinsics
    depth = manifest.stages[2]
    assert depth.status == "skipped" and any("no trusted intrinsics" in w for w in depth.warnings)


def test_scale_stage_carries_the_canonical_frame_like_vertical_slice(tmp_path: Path) -> None:
    items, opts, ctx, _ = _run(tmp_path, depth_model=None)
    ref = vertical_slice(items, opts)
    result = ctx.state["pose"].result
    assert ctx.state["frame"].frame_record is not None
    assert tuple(tuple(float(c) for c in p.position) for p in result.points) == ref.points
    assert [p.evidence_id for p in result.camera_poses] == [p[0] for p in ref.camera_poses]


def test_ran_is_succeeded_with_metrics_and_failed_is_degraded(tmp_path: Path, monkeypatch) -> None:
    import engine.pipeline.vertical_slice as vs

    ran = ({"status": "ran", "maps": 3, "metricized": 2, "dense_points": 40}, ["map-a", "map-b"])
    monkeypatch.setattr(vs, "_depth_stage", lambda *a, **k: ran)
    _, _, ctx, manifest = _run(tmp_path)
    depth = manifest.stages[2]
    assert depth.status == "succeeded" and depth.warnings == []
    assert depth.metrics == {"maps": 3.0, "metricized": 2.0, "dense_points": 40.0}
    assert ctx.state["depth"].metric_maps == ["map-a", "map-b"]

    monkeypatch.setattr(vs, "_depth_stage", lambda *a, **k: ({"status": "failed", "note": "boom"}, []))
    _, _, _, manifest = _run(tmp_path / "again")
    assert manifest.stages[2].status == "degraded" and "boom" in manifest.stages[2].warnings[0]
    assert manifest.grade(SPECS) == "partial"
