"""P108.4: the perception stage. No detector weights are loaded: every case here is decided before the model is built
(disabled / no intrinsics / no metric depth), or by a stubbed ``_perception_stage``."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.pipeline.stages.runner import run_stages
from engine.pipeline.stages.s09_perception import Perception
from engine.pipeline.vertical_slice import VerticalSliceError, vertical_slice
from tests.test_stage_dense import _state_through_compile

pytest.importorskip("PIL")


def _prepare(tmp_path: Path, maps=(), **opt_kw):
    items, opts, ctx = _state_through_compile(tmp_path, **opt_kw)
    ctx.state["depth"] = type("D", (), {"metric_maps": list(maps), "facts": None})()
    return items, opts, ctx


def _perception(ctx):
    return run_stages([Perception()], ctx).stages[0]


def test_spec() -> None:
    s = Perception.spec
    assert (s.name, s.version, s.resource_class, s.required) == ("perception", "1", "gpu", False)


def test_disabled_by_options_is_skipped_with_reason(tmp_path: Path) -> None:
    items, opts, ctx = _prepare(tmp_path, perception_model=None)
    result = _perception(ctx)
    assert result.status == "skipped" and any("disabled (perception_model=None)" in w for w in result.warnings)
    assert ctx.state["perception"].facts == vertical_slice(items, opts).stage_facts["perception"]


def test_no_intrinsics_is_skipped(tmp_path: Path) -> None:
    _, _, ctx = _prepare(tmp_path)
    result = _perception(ctx)
    assert result.status == "skipped" and any("no trusted intrinsics" in w for w in result.warnings)


def test_no_metric_depth_is_skipped_never_lifted_against_relative_depth(tmp_path: Path) -> None:
    _, _, ctx = _prepare(tmp_path, intrinsics=(800.0, 800.0, 32.0, 24.0), image_size=(64, 48))
    result = _perception(ctx)
    assert result.status == "skipped" and any("no metricized depth maps" in w for w in result.warnings)


def test_ran_is_succeeded_failed_is_degraded_and_invalid_world_fails(tmp_path: Path, monkeypatch) -> None:
    import engine.pipeline.vertical_slice as vs

    monkeypatch.setattr(vs, "_perception_stage", lambda *a, **k: {"status": "ran", "entities_promoted": 2})
    _, _, ctx = _prepare(tmp_path)
    ran = _perception(ctx)
    assert ran.status == "succeeded" and ran.metrics == {"entities_promoted": 2.0}

    monkeypatch.setattr(vs, "_perception_stage", lambda *a, **k: {"status": "failed", "note": "x"})
    _, _, ctx = _prepare(tmp_path / "b")
    assert _perception(ctx).status == "degraded"

    def invalid(*a, **k):
        raise VerticalSliceError("perception stage produced an invalid world: ['x']")

    monkeypatch.setattr(vs, "_perception_stage", invalid)
    _, _, ctx = _prepare(tmp_path / "c")
    failed = _perception(ctx)
    assert failed.status == "failed" and "invalid world" in failed.error
