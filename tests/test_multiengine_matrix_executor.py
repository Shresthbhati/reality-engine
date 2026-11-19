"""Executor -> Evidence Matrix persistence. Every adapter here is a SYNTHETIC TEST DOUBLE: these tests prove persistence, failure
preservation, cancellation and restart recovery, not any model's accuracy."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from engine.multiengine.contract import (
    Availability,
    AvailabilityStatus,
    CheckpointRequirement,
    EngineAdapter,
    EngineError,
    EngineErrorKind,
    EngineRequest,
    EngineSpec,
    HardwareRequirement,
    Task,
)
from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.evidence_store import EvidenceStore
from engine.multiengine.executor import (
    CONTRACT_VALIDATION,
    ExecutionCancelled,
    ExecutionManager,
    Policy,
    ReportStatus,
)
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.registry import EngineRegistry

OK = Availability(AvailabilityStatus.AVAILABLE, "verified in test", evidence=("test double",))
BLOCKED = Availability(AvailabilityStatus.BLOCKED, "weights missing: checkpoint not cached", "download it")


def _spec(eid, *, probe=None, family="", ckpt=None, version="v1"):
    return EngineSpec(
        engine_id=eid, name=eid, version=version, tasks=frozenset({Task.DEPTH}), input_schema="img", output_schema="d",
        probe=probe or (lambda: OK), family=family, commercial_ok=True, licence="Apache-2.0",
        hardware=HardwareRequirement(), checkpoints=ckpt or (),
    )


class FakeAdapter(EngineAdapter):
    """SYNTHETIC. ``fail`` raises a model error from infer."""

    def __init__(self, sp, value=1.0, fail=False):
        self.spec, self.value, self.fail = sp, value, fail
        self.loaded = self.unloaded = 0

    def validate_input(self, request):
        pass

    def load(self, device):
        self.loaded += 1

    def unload(self):
        self.unloaded += 1

    def infer(self, request):
        if self.fail:
            raise EngineError(EngineErrorKind.MODEL_ERROR, "synthetic model crash")
        return [
            ObservationDraft(task="depth", quantity="depth_value", source_id=s, coordinate_system=f"cam:{s}",
                             predicted={"v": self.value}, confidence=0.8)
            for s in request.sources
        ]


REQ = EngineRequest(sources=("img1", "img2"))


def _mgr(entries, matrix, **kw):
    reg = EngineRegistry()
    adapters = {}
    for sp, ad in entries:
        reg.register(sp)
        if ad is not None:
            adapters[sp.engine_id] = ad
    return ExecutionManager(reg, adapters, matrix=matrix, run_id="job-1", policy=Policy.MAXIMUM_FIDELITY, **kw)


def _three_engines():
    good = _spec("good", family="fam-a", ckpt=(CheckpointRequirement("good.pt", sha256="ab" * 32),))
    crash = _spec("crash", ckpt=(CheckpointRequirement("crash.pt"),))
    blocked = _spec("blocked", probe=lambda: BLOCKED)
    return [(good, FakeAdapter(good, 2.0)), (crash, FakeAdapter(crash, fail=True)), (blocked, FakeAdapter(blocked))]


def test_every_run_is_recorded_and_successful_observations_carry_full_context():
    m = EvidenceMatrix()
    report = _mgr(_three_engines(), m, camera_refs={"img1": "cam-A"}).run(Task.DEPTH, REQ)
    assert report.status is ReportStatus.PARTIAL
    by = {r.engine_id: r for r in m.runs("job-1")}
    assert set(by) == {"good", "crash", "blocked"}  # the failed and the unavailable engine are preserved, not dropped
    assert by["good"].status == "succeeded" and len(by["good"].observation_ids) == 2
    assert by["crash"].status == "failed" and by["crash"].error_kind == "model_error"
    assert "synthetic model crash" in by["crash"].error
    assert by["blocked"].status == "unavailable" and by["blocked"].observation_ids == ()
    assert by["good"].architecture_id == "fam-a" and by["good"].checkpoint_id == f"good.pt@sha256:{'ab' * 32}"
    assert by["crash"].checkpoint_id == "crash.pt (revision unpinned)"  # an unpinned checkpoint is labelled, not dressed up
    assert by["good"].model_version == "v1" and by["good"].wall_s is not None
    assert by["good"].sources == ("img1", "img2")
    rows = {r["observation"]["source_id"]: r for r in m.rows()}
    assert set(rows) == {"img1", "img2"} and len(m) == 2
    r1 = rows["img1"]
    assert r1["context"]["run_id"] == "job-1" and r1["context"]["camera_ref"] == "cam-A"
    assert r1["context"]["validation"] == CONTRACT_VALIDATION
    assert rows["img2"]["context"]["camera_ref"] is None  # an unknown camera stays unknown
    assert r1["raw_confidence"] == 0.8 and r1["calibrated_confidence"] is None  # nothing was calibrated, so nothing claims it
    assert m.verify() == []


def test_run_output_is_stored_byte_for_byte_as_one_content_addressed_artifact():
    m = EvidenceMatrix()
    _mgr(_three_engines(), m).run(Task.DEPTH, REQ)
    run = next(r for r in m.runs() if r.engine_id == "good")
    bundle = json.loads(m.artifacts.get(run.output_artifact))
    assert {o["observation_id"] for o in bundle} == set(run.observation_ids)
    for oid in run.observation_ids:
        assert m.row(oid)["context"]["artifact_refs"] == {"run_output": run.output_artifact}
        assert m.store.get(oid).to_dict() in bundle  # the stored observation IS the model output, unmodified


def test_model_is_unloaded_before_persistence_and_everything_survives_a_restart(tmp_path):
    entries = _three_engines()
    m = EvidenceMatrix(tmp_path)
    _mgr(entries, m).run(Task.DEPTH, REQ)
    good = entries[0][1]
    assert good.loaded == good.unloaded == 1  # the model was released; persistence does not depend on it being alive
    before_rows, before_runs = m.rows(), m.runs()
    del m, entries  # drop every in-memory reference: a restart sees only what is on disk
    again = EvidenceMatrix(tmp_path)
    assert again.rows() == before_rows and again.runs() == before_runs
    assert again.verify() == []
    assert len(again.runs("job-1")) == 3


def test_cancellation_between_engines_keeps_finished_work_and_records_the_rest_as_cancelled():
    m = EvidenceMatrix()
    a, b = _spec("a"), _spec("b")
    state = {"calls": 0}

    def cancel():
        state["calls"] += 1
        return state["calls"] > 1  # let the first engine start, cancel before the second

    ad_a, ad_b = FakeAdapter(a), FakeAdapter(b)
    mgr = _mgr([(a, ad_a), (b, ad_b)], m, should_cancel=cancel)
    with pytest.raises(ExecutionCancelled) as ei:
        mgr.run(Task.DEPTH, REQ)
    assert ad_a.loaded == 1 and ad_b.loaded == 0  # the second model never loaded
    by = {r.engine_id: r for r in m.runs()}
    assert by["a"].status == "succeeded" and len(m) == 2  # what finished is persisted
    assert by["b"].status == "cancelled" and "cancellation was requested" in by["b"].error
    assert any("cancelled before b" in w for w in ei.value.report.warnings)
    assert m.verify() == []


def test_one_source_of_raw_evidence_is_enforced():
    m = EvidenceMatrix()
    reg = EngineRegistry()
    with pytest.raises(ValueError, match="same EvidenceStore"):
        ExecutionManager(reg, {}, store=EvidenceStore(), matrix=m, run_id="r")
    with pytest.raises(ValueError, match="run_id is required"):
        ExecutionManager(reg, {}, matrix=m)
    mgr = ExecutionManager(reg, {}, matrix=m, run_id="r")
    assert mgr.store is m.store


def test_repeated_run_of_the_same_engine_is_logged_not_overwritten():
    m = EvidenceMatrix()
    sp = _spec("a")
    reg = EngineRegistry()
    reg.register(sp)
    mgr = ExecutionManager(reg, {"a": FakeAdapter(sp)}, matrix=m, run_id="job-1")
    mgr.run(Task.DEPTH, REQ)
    mgr.run(Task.DEPTH, REQ)  # e.g. targeted refinement re-runs it; processed_at differs so the observations are new rows
    assert len(m.runs()) == 2 and len(m) == 4


def test_store_only_path_is_unchanged_when_no_matrix_is_given():
    sp = _spec("a")
    reg = EngineRegistry()
    reg.register(sp)
    store = EvidenceStore()
    ExecutionManager(reg, {"a": FakeAdapter(sp)}, store=store).run(Task.DEPTH, replace(REQ))
    assert len(store) == 2
