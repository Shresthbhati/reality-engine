"""P107: the pose and scale stages, run directly with the deterministic backend of test_cli_compile."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from engine.pipeline.dataset import load_capture_dataset
from engine.pipeline.stages.base import StageContext
from engine.pipeline.stages.runner import run_stages
from engine.pipeline.stages.s04_pose import Pose
from engine.pipeline.stages.s05_scale import Scale
from engine.pipeline.vertical_slice import VerticalSliceError, VerticalSliceOptions, vertical_slice
from tests.test_cli_compile import _TwoViewBackend

PIL = pytest.importorskip("PIL")


def _capture(tmp_path: Path, baselines: list[dict]) -> Path:
    import json

    from PIL import Image

    d = tmp_path / "capture"
    (d / "images").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (64, 48), color=(i * 40, 100, 150)).save(d / "images" / f"img_{i:03d}.jpg")
    (d / "manifest.json").write_text(json.dumps({
        "dataset": "p107", "images": [{"file": f"img_{i:03d}.jpg"} for i in range(3)],
        "measured_baselines": baselines, "scale_reference": {"method": "test_baseline"},
    }), encoding="utf-8")
    return d


def _ctx(items, options) -> StageContext:
    return StageContext(job_id="j", workdir=Path("."), artifacts={}, params={},
                        state={"evidence_items": items, "vertical_slice_options": options})


GOOD = [{"evidence_id_a": "img_000", "evidence_id_b": "img_001", "distance_m": 1.0}]


def test_specs() -> None:
    assert (Pose.spec.name, Pose.spec.version, Pose.spec.resource_class, Pose.spec.required) == ("pose", "1", "gpu", True)
    assert (Scale.spec.name, Scale.spec.version, Scale.spec.resource_class, Scale.spec.required) == ("scale", "1", "cpu", True)


def test_pose_then_scale_matches_vertical_slice(tmp_path: Path) -> None:
    items, refs, _, _ = load_capture_dataset(_capture(tmp_path, GOOD))
    opts = VerticalSliceOptions(reconstruction_backend=_TwoViewBackend(), measured_baselines=tuple(refs))
    ctx = _ctx(items, opts)
    manifest = run_stages([Pose(), Scale()], ctx)
    assert [(r.name, r.status) for r in manifest.stages] == [("pose", "succeeded"), ("scale", "succeeded")]
    assert manifest.grade([Pose.spec, Scale.spec]) == "succeeded"
    ref = vertical_slice(items, opts)
    sc = ctx.state["scale"]
    assert sc.scale_state == ref.scale_state and sc.scale_state != "relative"
    assert sc.meters_per_unit == ref.meters_per_unit and sc.scale_note == ref.scale_note
    assert len(ctx.state["pose"].result.points) == ref.points_total
    assert manifest.stages[0].metrics["cameras_registered"] == ref.cameras_registered == 3


def test_no_baselines_stays_relative_and_is_not_degraded(tmp_path: Path) -> None:
    items, _, _, _ = load_capture_dataset(_capture(tmp_path, []))
    ctx = _ctx(items, VerticalSliceOptions(reconstruction_backend=_TwoViewBackend()))
    manifest = run_stages([Pose(), Scale()], ctx)
    assert [r.status for r in manifest.stages] == ["succeeded", "succeeded"]
    assert ctx.state["scale"].scale_state == "relative" and ctx.state["scale"].meters_per_unit is None


def test_unregistrable_baseline_is_degraded_not_success(tmp_path: Path) -> None:
    bad = [{"evidence_id_a": "img_000", "evidence_id_b": "img_001", "distance_m": 1.0}]
    items, refs, _, _ = load_capture_dataset(_capture(tmp_path, bad))
    # a baseline naming photographs the reconstruction never registered
    import dataclasses
    refs = [dataclasses.replace(r, evidence_id_b="never_registered") for r in refs]
    ctx = _ctx(items, VerticalSliceOptions(reconstruction_backend=_TwoViewBackend(), measured_baselines=tuple(refs)))
    manifest = run_stages([Pose(), Scale()], ctx)
    scale = manifest.stages[1]
    assert scale.status == "degraded" and "stays RELATIVE" in scale.warnings[0]
    assert manifest.grade([Pose.spec, Scale.spec]) == "partial"


@dataclass(frozen=True)
class _FailedBackend:
    def reconstruct(self, evidence):
        from reconstruction.interfaces import ReconstructionResult
        return ReconstructionResult(points=[], camera_poses=[], registration_status="failed")


def test_pose_failure_raises_and_runner_marks_failed(tmp_path: Path) -> None:
    items, _, _, _ = load_capture_dataset(_capture(tmp_path, []))
    opts = VerticalSliceOptions(reconstruction_backend=_FailedBackend())
    with pytest.raises(VerticalSliceError):
        Pose().run(_ctx(items, opts))
    manifest = run_stages([Pose(), Scale()], _ctx(items, opts))
    assert [(r.name, r.status) for r in manifest.stages] == [("pose", "failed")]  # required failure stops the run
    assert manifest.grade([Pose.spec, Scale.spec]) == "failed"


def test_fewer_than_two_items_refused(tmp_path: Path) -> None:
    items, _, _, _ = load_capture_dataset(_capture(tmp_path, []))
    with pytest.raises(VerticalSliceError, match=">= 2"):
        Pose().run(_ctx(items[:1], VerticalSliceOptions(reconstruction_backend=_TwoViewBackend())))
