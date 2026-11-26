"""Tests for the unified multi-engine orchestrator.

Each test pins one acceptance criterion of the orchestration contract:

* an engine whose availability probe fails is never executed and is reported
  ``unavailable`` with the probe's own reason;
* a CPU-only machine does not defer a VRAM-declaring engine (no GPU to gate on);
* an engine whose dependency did not succeed is skipped, not run on missing input;
* dependencies execute before their dependents;
* cancellation stops the run and is recorded;
* a timeout is recorded as ``timed_out``, never as success;
* a timed-out engine is DRAINED (waited for) before the next engine starts, so two
  engines never hold the device at once; an engine that outlives the drain wait keeps
  the GPU quarantined instead of sharing it;
* a malformed output is ``invalid_output``, never success;
* a transient failure is retried, a permanent one is not;
* content-identical observations are counted once and reported as duplicates;
* contradictions are only claimed between observations that overlap on one image,
  two different objects on one image are not a contradiction;
* relative and metric positions are never compared as metres.
"""

from __future__ import annotations

import time

import pytest

from perception import engine_orchestrator as eo
from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
    bbox_iou,
    compare_observations,
)
from perception.engine_orchestrator import (
    EngineSpec,
    MultiEngineOrchestrator,
    get_unavailable_engines,
    list_available_engines,
    merge_engine_outputs,
    register_engine,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Every test starts from an empty registry and leaves it empty."""
    saved_engines = dict(eo._ENGINES)
    saved_fns = dict(eo._ENGINE_FNS)
    eo._ENGINES.clear()
    eo._ENGINE_FNS.clear()
    yield
    eo._ENGINES.clear()
    eo._ENGINE_FNS.clear()
    eo._ENGINES.update(saved_engines)
    eo._ENGINE_FNS.update(saved_fns)


def _obs(label: str, *, image: str = "img1", bbox=(0.0, 0.0, 10.0, 10.0), arch: str = "sift",
         world_pos=None, scale: str = "image", units: str = "pixels", confidence=0.8,
         evidence_id: str | None = None, material=None) -> ObservationRecord:
    return ObservationRecord(
        evidence_id=evidence_id or f"{arch}:{image}:{label}:{bbox}:{world_pos}",
        source_image_id=image,
        coordinate_frame=CoordinateFrame(frame_id="world" if world_pos else "image",
                                         scale_state=scale, units=units),
        model_id=ModelIdentity(engine_id="matching", architecture_id=arch, checkpoint_id="stub"),
        label=label,
        category="object",
        bbox_px=bbox,
        world_pos=world_pos,
        confidence=confidence,
        material_hypothesis=material,
    )


def _ok(engine_id: str, obs=()) -> UnifiedEngineOutput:
    return UnifiedEngineOutput(engine_id=engine_id, run_id="run-1", observations=tuple(obs), status="succeeded")


# --------------------------------------------------------------------------- availability


def test_unavailable_engine_is_never_executed_and_is_reported_with_its_reason():
    calls = []

    def runner(image_paths, **ctx):
        calls.append(1)
        return _ok("blocked")

    register_engine(
        EngineSpec(engine_id="blocked", architecture_id="blocked", license="Apache-2.0",
                   run_condition=lambda: "checkpoint not cached"),
        runner,
    )

    assert list_available_engines() == []
    assert get_unavailable_engines() == [{"engine_id": "blocked", "reason": "checkpoint not cached"}]

    report = MultiEngineOrchestrator().run_all(["a.png"])
    assert calls == []  # never executed
    assert report.engines_run == {"blocked": "unavailable"}
    assert report.unavailable == [{"engine_id": "blocked", "reason": "checkpoint not cached"}]
    assert report.engine_executions[0].status == "unavailable"
    assert report.engine_executions[0].reason == "unavailable"


def test_a_broken_availability_probe_makes_the_engine_unavailable_not_a_crash():
    def probe():
        raise RuntimeError("driver exploded")

    register_engine(EngineSpec(engine_id="probe_broken", architecture_id="x", run_condition=probe),
                    lambda *a, **k: _ok("probe_broken"))
    report = MultiEngineOrchestrator().run_all(["a.png"])
    assert report.engines_run == {"probe_broken": "unavailable"}
    assert "driver exploded" in report.unavailable[0]["reason"]


def test_cpu_only_machine_does_not_defer_a_vram_declaring_engine(monkeypatch):
    monkeypatch.setattr(eo, "_cuda_available", lambda: False)
    monkeypatch.setattr(eo, "_get_free_vram_mb", lambda: 0.0)
    monkeypatch.setattr(eo, "_get_vram_peak_mb", lambda: None)
    monkeypatch.setattr(eo, "_torch_empty_cache", lambda: None)

    ran = []

    def runner(image_paths, **ctx):
        ran.append(1)
        return _ok("gpu_engine", [_obs("wall", arch="grounding_dino")])

    register_engine(EngineSpec(engine_id="gpu_engine", architecture_id="grounding_dino", vram_mb=4096), runner)
    report = MultiEngineOrchestrator().run_all(["a.png"])

    assert ran == [1], "a CPU-only machine must not defer a CPU-capable engine for '0 MB free VRAM'"
    assert report.engines_run == {"gpu_engine": "succeeded"}
    assert report.engine_executions[0].device == "cpu"
    assert report.unavailable == []


def test_gpu_machine_still_defers_an_engine_that_cannot_fit(monkeypatch):
    monkeypatch.setattr(eo, "_cuda_available", lambda: True)
    monkeypatch.setattr(eo, "_get_free_vram_mb", lambda: 100.0)
    monkeypatch.setattr(eo, "_get_vram_peak_mb", lambda: 12.0)
    monkeypatch.setattr(eo, "_torch_empty_cache", lambda: None)

    ran = []
    register_engine(EngineSpec(engine_id="too_big", architecture_id="too_big", vram_mb=6000),
                    lambda *a, **k: ran.append(1) or _ok("too_big"))
    report = MultiEngineOrchestrator(vram_headroom_mb=50).run_all(["a.png"])

    assert ran == []
    assert report.engines_run == {"too_big": "unavailable"}
    assert "VRAM" in report.unavailable[0]["reason"]


# --------------------------------------------------------------------------- dependencies


def test_dependencies_run_before_their_dependents():
    order = []

    def first(image_paths, **ctx):
        order.append("first")
        return _ok("first", [_obs("wall", arch="a1")])

    def second(image_paths, **ctx):
        order.append("second")
        return _ok("second")

    register_engine(EngineSpec(engine_id="second", architecture_id="a2", priority=99, dependencies=("first",)), second)
    register_engine(EngineSpec(engine_id="first", architecture_id="a1", priority=1), first)

    MultiEngineOrchestrator().run_all(["a.png"])
    assert order == ["first", "second"], "priority must never invert a dependency edge"


def test_an_engine_whose_dependency_failed_is_skipped_not_run():
    ran = []

    def boom(image_paths, **ctx):
        raise ValueError("model load failed")

    def dependent(image_paths, **ctx):
        ran.append(1)
        return _ok("dependent")

    register_engine(EngineSpec(engine_id="upstream", architecture_id="detector"), boom)
    register_engine(EngineSpec(engine_id="dependent", architecture_id="segmenter", dependencies=("upstream",)), dependent)

    report = MultiEngineOrchestrator().run_all(["a.png"])

    assert ran == []
    assert report.engines_run["upstream"] == "failed"
    assert report.engines_run["dependent"] == "skipped"
    skipped = next(e for e in report.engine_executions if e.engine_id == "dependent")
    assert skipped.reason == "dependency_unmet"
    assert "upstream" in skipped.error


def test_a_dependency_that_is_unavailable_also_skips_dependents():
    register_engine(EngineSpec(engine_id="upstream", architecture_id="d", run_condition=lambda: "no weights"),
                    lambda *a, **k: _ok("upstream"))
    register_engine(EngineSpec(engine_id="downstream", architecture_id="s", dependencies=("upstream",)),
                    lambda *a, **k: _ok("downstream"))
    report = MultiEngineOrchestrator().run_all(["a.png"])
    assert report.engines_run == {"upstream": "unavailable", "downstream": "skipped"}


def test_detection_boxes_reach_downstream_engines_by_category_not_by_engine_name():
    """The segmenter receives boxes from whatever produced object observations."""
    seen = {}

    def detector(image_paths, **ctx):
        return _ok("my_detector", [_obs("chair", arch="owlvit", bbox=(1.0, 2.0, 30.0, 40.0))])

    def segmenter(image_paths, **ctx):
        seen["bboxes"] = ctx.get("bboxes")
        return _ok("segmenter")

    register_engine(EngineSpec(engine_id="my_detector", architecture_id="owlvit", priority=90), detector)
    register_engine(EngineSpec(engine_id="segmenter", architecture_id="sam2", priority=10,
                               dependencies=("my_detector",)), segmenter)

    MultiEngineOrchestrator().run_all(["a.png"])
    assert seen["bboxes"] == {"img1": [(1.0, 2.0, 30.0, 40.0)]}


# --------------------------------------------------------------------------- cancellation / timeout / validation / retry


def test_cancellation_stops_the_run_and_names_what_did_not_run():
    register_engine(EngineSpec(engine_id="one", architecture_id="a", priority=90),
                    lambda *a, **k: _ok("one", [_obs("wall", arch="a")]))
    register_engine(EngineSpec(engine_id="two", architecture_id="b", priority=80),
                    lambda *a, **k: _ok("two"))

    calls = {"n": 0}

    def cancel_after_one():
        calls["n"] += 1
        return calls["n"] > 1  # still running, then cancelled

    report = MultiEngineOrchestrator().run_all(["a.png"], {"should_cancel": cancel_after_one})
    assert report.engines_run["one"] == "succeeded"
    assert report.engines_run["two"] == "cancelled"


def test_a_timeout_is_recorded_as_timed_out_and_never_as_success():
    def slow(image_paths, **ctx):
        time.sleep(0.4)
        return _ok("slow", [_obs("wall")])

    register_engine(EngineSpec(engine_id="slow", architecture_id="slow", timeout_s=0.05), slow)
    report = MultiEngineOrchestrator().run_all(["a.png"])

    assert report.engines_run == {"slow": "timed_out"}
    ex = report.engine_executions[0]
    assert ex.status == "timed_out"
    assert "timeout" in ex.reason or "timeout" in ex.error
    assert report.observations == []


def test_an_abandoned_engine_is_drained_before_the_next_engine_starts():
    """Python cannot kill the abandoned call, so the orchestrator waits for it.

    Starting the next engine while the abandoned one is still allocating would put two
    engines on one 6 GB device at once. The recorded wall time includes the wait.
    """
    order: list[str] = []

    def slow(image_paths, **ctx):
        time.sleep(0.3)
        order.append("slow-returned")
        return _ok("slow", [_obs("feature_match")])

    def after(image_paths, **ctx):
        order.append("next-started")
        return _ok("after", [_obs("feature_match", arch="other")])

    register_engine(EngineSpec(engine_id="slow", architecture_id="slow", timeout_s=0.05,
                               drain_after_timeout_s=5.0, priority=90), slow)
    register_engine(EngineSpec(engine_id="after", architecture_id="after", priority=10), after)

    report = MultiEngineOrchestrator().run_all(["a.png"])

    assert report.engines_run["slow"] == "timed_out"
    assert order == ["slow-returned", "next-started"], order
    ex = next(e for e in report.engine_executions if e.engine_id == "slow")
    assert ex.provenance["abandoned_thread_drained"] is True
    assert ex.provenance["drain_wait_s"] > 0.1
    assert ex.wall_time_s >= 0.2, "the drain wait is part of the time it held the device"
    assert "released" in ex.provenance["device_release"]


def test_an_engine_that_outlives_the_drain_wait_keeps_the_gpu_quarantined(monkeypatch):
    """When the wait fails, the next GPU engine is refused rather than sharing the device."""
    monkeypatch.setattr(eo, "_cuda_available", lambda: True)
    monkeypatch.setattr(eo, "_get_free_vram_mb", lambda: 6000.0)
    started: list[str] = []

    def stuck(image_paths, **ctx):
        time.sleep(0.5)
        return _ok("stuck", [_obs("feature_match")])

    def gpu_engine(image_paths, **ctx):
        started.append("gpu")
        return _ok("gpu", [_obs("feature_match", arch="gpu")])

    register_engine(EngineSpec(engine_id="stuck", architecture_id="stuck", timeout_s=0.05,
                               drain_after_timeout_s=0.01, vram_mb=1024, priority=90), stuck)
    register_engine(EngineSpec(engine_id="gpu", architecture_id="gpu", vram_mb=1024, priority=10),
                    gpu_engine)

    report = MultiEngineOrchestrator().run_all(["a.png"])

    assert started == [], "a GPU engine was started next to an abandoned call"
    assert report.engines_run["stuck"] == "timed_out"
    assert report.engines_run["gpu"] == "unavailable"
    gpu_row = next(e for e in report.engine_executions if e.engine_id == "gpu")
    assert gpu_row.reason == "gpu_quarantined"
    assert "stuck" in gpu_row.error
    stuck_row = next(e for e in report.engine_executions if e.engine_id == "stuck")
    assert stuck_row.provenance["abandoned_thread_drained"] is False
    assert "NOT released" in stuck_row.provenance["device_release"]
    assert any(u["engine_id"] == "gpu" for u in report.unavailable)


def test_malformed_output_is_invalid_not_success():
    register_engine(EngineSpec(engine_id="liar", architecture_id="liar"),
                    lambda *a, **k: _ok("liar", []) if False else "not an output")
    report = MultiEngineOrchestrator().run_all(["a.png"])
    assert report.engines_run == {"liar": "invalid_output"}
    assert "UnifiedEngineOutput" in report.engine_executions[0].error


def test_success_with_no_observations_is_downgraded():
    register_engine(EngineSpec(engine_id="empty", architecture_id="empty"),
                    lambda *a, **k: UnifiedEngineOutput(engine_id="empty", run_id="r", observations=(), status="succeeded"))
    report = MultiEngineOrchestrator().run_all(["a.png"])
    assert report.engines_run == {"empty": "invalid_output"}


def test_a_transient_failure_is_retried():
    attempts = {"n": 0}

    def flaky(image_paths, **ctx):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise OSError("device temporarily unavailable")
        return _ok("flaky", [_obs("wall", arch="flaky")])

    register_engine(EngineSpec(engine_id="flaky", architecture_id="flaky", max_attempts=3), flaky)
    report = MultiEngineOrchestrator().run_all(["a.png"])

    assert attempts["n"] == 3
    assert report.engines_run == {"flaky": "succeeded"}
    assert report.engine_executions[0].attempts == 3


def test_a_permanent_failure_is_not_retried():
    attempts = {"n": 0}

    def broken(image_paths, **ctx):
        attempts["n"] += 1
        raise ValueError("bad configuration")

    register_engine(EngineSpec(engine_id="broken", architecture_id="broken", max_attempts=3), broken)
    MultiEngineOrchestrator().run_all(["a.png"])
    assert attempts["n"] == 1


# --------------------------------------------------------------------------- fusion


def test_identical_observations_are_counted_once():
    a = _obs("wall", bbox=(0.0, 0.0, 10.0, 10.0), arch="a", evidence_id="e1")
    b = _obs("wall", bbox=(0.0, 0.0, 10.0, 10.0), arch="a", evidence_id="e2")
    report = merge_engine_outputs([
        UnifiedEngineOutput(engine_id="a", run_id="r", observations=(a,), status="succeeded"),
        UnifiedEngineOutput(engine_id="a2", run_id="r", observations=(b,), status="succeeded"),
    ])
    assert len(report.observations) == 1
    assert report.duplicate_observations == 1


def test_contradicting_labels_on_the_same_target_are_reported():
    a = _obs("door", arch="a", bbox=(0.0, 0.0, 10.0, 10.0))
    b = _obs("window", arch="b", bbox=(0.0, 0.0, 10.0, 10.0))
    report = merge_engine_outputs([
        UnifiedEngineOutput(engine_id="a", run_id="r", observations=(a,), status="succeeded"),
        UnifiedEngineOutput(engine_id="b", run_id="r", observations=(b,), status="succeeded"),
    ])
    assert len(report.contradictions) == 1
    assert report.contradictions[0]["issue"].startswith("label_mismatch")
    assert report.cross_engine_summary["n_contradictions"] == 1


def test_two_different_objects_on_one_image_are_not_a_contradiction():
    a = _obs("door", arch="a", bbox=(0.0, 0.0, 10.0, 10.0))
    b = _obs("window", arch="b", bbox=(200.0, 200.0, 240.0, 260.0))
    report = merge_engine_outputs([
        UnifiedEngineOutput(engine_id="a", run_id="r", observations=(a,), status="succeeded"),
        UnifiedEngineOutput(engine_id="b", run_id="r", observations=(b,), status="succeeded"),
    ])
    assert report.contradictions == []


def test_agreeing_engines_are_counted_as_agreement_not_contradiction():
    a = _obs("door", arch="a", bbox=(0.0, 0.0, 10.0, 10.0))
    b = _obs("door", arch="b", bbox=(0.0, 0.0, 10.0, 10.0))
    report = merge_engine_outputs([
        UnifiedEngineOutput(engine_id="a", run_id="r", observations=(a,), status="succeeded"),
        UnifiedEngineOutput(engine_id="b", run_id="r", observations=(b,), status="succeeded"),
    ])
    assert report.contradictions == []
    assert report.cross_engine_summary["n_agreements"] == 1


def test_the_same_engine_twice_is_not_cross_engine_evidence():
    a = _obs("door", arch="a", bbox=(0.0, 0.0, 10.0, 10.0))
    b = _obs("window", arch="a", bbox=(0.0, 0.0, 10.0, 10.0))
    report = merge_engine_outputs([
        UnifiedEngineOutput(engine_id="a", run_id="r", observations=(a,), status="succeeded"),
        UnifiedEngineOutput(engine_id="a", run_id="r", observations=(b,), status="succeeded"),
    ])
    assert report.contradictions == []


def test_metric_positions_disagreeing_beyond_tolerance_are_a_contradiction_in_metres():
    a = _obs("chair", arch="a", bbox=(0.0, 0.0, 10.0, 10.0), world_pos=(0.0, 0.0, 0.0),
             scale="metric", units="m")
    b = _obs("chair", arch="b", bbox=(0.0, 0.0, 10.0, 10.0), world_pos=(0.5, 0.0, 0.0),
             scale="metric", units="m")
    issues = compare_observations(a, b)
    assert any(i.startswith("position_mismatch") and i.endswith("m") for i in issues)


def test_relative_and_metric_positions_are_never_compared_as_metres():
    a = _obs("chair", arch="a", bbox=(0.0, 0.0, 10.0, 10.0), world_pos=(0.0, 0.0, 0.0),
             scale="relative", units="relative")
    b = _obs("chair", arch="b", bbox=(0.0, 0.0, 10.0, 10.0), world_pos=(0.5, 0.0, 0.0),
             scale="metric", units="m")
    issues = compare_observations(a, b)
    assert issues and all(i.startswith("scale_state_mismatch") for i in issues)
    assert not any(" m" in i and "mismatch" not in i for i in issues)


def test_a_scale_mismatch_is_a_note_not_a_contradiction():
    a = _obs("chair", arch="a", bbox=(0.0, 0.0, 10.0, 10.0), world_pos=(0.0, 0.0, 0.0),
             scale="sfm_gauge", units="sfm_units")
    b = _obs("chair", arch="b", bbox=(0.0, 0.0, 10.0, 10.0), world_pos=(5.0, 0.0, 0.0),
             scale="metric", units="m")
    report = merge_engine_outputs([
        UnifiedEngineOutput(engine_id="a", run_id="r", observations=(a,), status="succeeded"),
        UnifiedEngineOutput(engine_id="b", run_id="r", observations=(b,), status="succeeded"),
    ])
    assert report.contradictions == []
    assert report.cross_engine_summary["n_comparability_notes"] == 1


def test_bbox_iou_edges():
    assert bbox_iou(None, (0, 0, 1, 1)) == 0.0
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# --------------------------------------------------------------------------- the execution contract


def test_every_engine_reports_the_full_execution_contract():
    register_engine(
        EngineSpec(engine_id="sift", architecture_id="sift", checkpoint_id="opencv", version="4.8",
                   license="Apache-2.0"),
        lambda *a, **k: _ok("sift", [_obs("feature_match", arch="sift")]),
    )
    report = MultiEngineOrchestrator().run_all(["a.png"])
    row = report.to_dict()["engine_executions"][0]

    for key in ("engine_id", "architecture_id", "checkpoint_id", "version", "license", "status",
                "error", "device", "attempts", "wall_time_s", "peak_vram_mb", "n_observations",
                "frame_ids", "scale_states", "units", "provenance"):
        assert key in row, key
    assert row["architecture_id"] == "sift"
    assert row["checkpoint_id"] == "opencv"
    assert row["n_observations"] == 1
    assert row["frame_ids"] == ["image"]
    assert row["units"] == ["pixels"]
