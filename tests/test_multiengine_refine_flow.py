"""Refinement in the evidence-to-world flow: reprocessing, rollback, regression checks, explicit stop reasons. Engines are SYNTHETIC
scripted doubles; they prove the loop's control logic and the preservation of raw evidence, not any detector's accuracy."""

from __future__ import annotations

from engine.multiengine.contract import (
    Availability,
    AvailabilityStatus,
    EngineAdapter,
    EngineError,
    EngineErrorKind,
    EngineSpec,
    HardwareRequirement,
    Task,
)
from engine.multiengine.evidence_flow import StageSpec, run_evidence_flow
from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.registry import EngineRegistry

OK = Availability(AvailabilityStatus.AVAILABLE, "verified in test", evidence=("test double",))
IMG = "img1.jpg"
X, Y = (10, 10, 110, 110), (400, 300, 500, 400)  # two well-separated objects in one image
UNKNOWN = "zzzz-unlabeled"


class Scripted(EngineAdapter):
    """SYNTHETIC: returns the scripted (label, box) detections for image 1; optionally fails."""

    def __init__(self, sp, dets, fail=False):
        self.spec, self.dets, self.fail, self.loaded = sp, dets, fail, 0

    def validate_input(self, request):
        pass

    def load(self, device):
        self.loaded += 1

    def unload(self):
        pass

    def infer(self, request):
        if self.fail:
            raise EngineError(EngineErrorKind.MODEL_ERROR, "synthetic crash")
        return [ObservationDraft(task="detection", quantity="bbox", source_id=IMG, coordinate_system=f"image:{IMG}",
                                 image_region=box, predicted={"label": lab}, confidence=0.9) for lab, box in self.dets]


def _spec(eid, fam):
    return EngineSpec(engine_id=eid, name=eid, version="v1", tasks=frozenset({Task.DETECTION}), input_schema="img",
                      output_schema="bbox", probe=lambda: OK, family=fam, commercial_ok=True, licence="Apache-2.0",
                      hardware=HardwareRequirement())


def _run(tmp_path, a, b, *, only=("det_a",), b_fail=False, **kw):
    reg, adapters = EngineRegistry(), {}
    for eid, fam, dets, fail in (("det_a", "fa", a, False), ("det_b", "fb", b, b_fail)):
        sp = _spec(eid, fam)
        reg.register(sp)
        adapters[eid] = Scripted(sp, dets, fail)
    stages = [StageSpec(Task.DETECTION, only=only)] if only else [StageSpec(Task.DETECTION)]
    return run_evidence_flow(tmp_path, "run-1", [IMG], registry=reg, adapters=adapters, stages=stages, **kw), adapters


def _hist(res):
    return res.snapshot["refinement_history"][0]


def _actions(res):
    return [a for it in _hist(res)["iterations"] for a in it["actions"]]


def test_reprocessing_with_an_untried_engine_fixes_an_unclassified_object(tmp_path):
    res, ad = _run(tmp_path, [(UNKNOWN, X)], [("chair", X)])
    applied = [a for a in _actions(res) if a["outcome"] == "applied"]
    assert applied and applied[0]["engines"] == ["det_b"] and ad["det_b"].loaded == 1  # the untried engine really ran
    (ent,) = res.snapshot["entities"]
    assert ent["ontology_path"] == "furniture/seating/chair" and len(ent["members"]) == 2  # both engines' raw rows kept
    assert _hist(res)["stop_reason"] == "awaiting_evidence" and res.capture_requests  # what is left needs new photographs
    assert _hist(res)["rolled_back_observations"] == []
    assert {r.engine_id for r in EvidenceMatrix(tmp_path).runs()} == {"det_a", "det_b"}


def test_a_reprocessing_result_that_breaks_an_established_class_is_rolled_back_but_kept(tmp_path):
    # det_a: X is a chair (established), Y is unknown. det_b would fix Y but ALSO call X a sofa -> conflict on an established class.
    res, _ = _run(tmp_path, [("chair", X), (UNKNOWN, Y)], [("sofa", X), ("chair", Y)])
    outcomes = [a["outcome"] for a in _actions(res)]
    assert "applied" not in outcomes and any(o.startswith("rejected_") for o in outcomes)
    paths = sorted(e["ontology_path"] for e in res.snapshot["entities"])
    assert paths == ["fallback/unknown_object", "furniture/seating/chair"]  # the previous selection stands: X still a chair
    rolled = _hist(res)["rolled_back_observations"]
    assert len(rolled) == 2
    m = EvidenceMatrix(tmp_path)
    for oid in rolled:  # the raw predictions are preserved, marked rejected with the reason
        row = m.row(oid)
        assert row["observation"]["engine_id"] == "det_b" and row["status"] == "rejected"
        assert "not adopted" in row["status_history"][-1]["reason"]
    assert res.snapshot["counts"]["observations"] == 4  # nothing was deleted


def test_when_every_engine_already_ran_nothing_is_reprocessed_and_the_gap_becomes_a_request(tmp_path):
    res, ad = _run(tmp_path, [(UNKNOWN, X)], [(UNKNOWN, X)], only=None)
    assert all(a["outcome"] != "applied" for a in _actions(res)) and ad["det_a"].loaded == 1 and ad["det_b"].loaded == 1
    assert _hist(res)["stop_reason"] in ("awaiting_evidence", "no_engine_available") and res.capture_requests


def test_a_failing_reprocess_engine_leaves_the_state_unchanged_and_is_recorded(tmp_path):
    res, _ = _run(tmp_path, [(UNKNOWN, X)], [("chair", X)], b_fail=True)
    assert any(a["outcome"] == "engine_failed" and "synthetic crash" in a["detail"] for a in _actions(res))
    assert [e["ontology_path"] for e in res.snapshot["entities"]] == ["fallback/unknown_object"]
    runs = {r.engine_id: r for r in EvidenceMatrix(tmp_path).runs()}
    assert runs["det_b"].status == "failed" and "synthetic crash" in runs["det_b"].error


def test_cancellation_during_reprocessing_keeps_the_pre_refinement_state_and_finished_work(tmp_path):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2  # stage start + det_a start run; the reprocessing engine is stopped before it starts

    res, ad = _run(tmp_path, [(UNKNOWN, X)], [("chair", X)], should_cancel=cancel)
    assert res.status == "cancelled" and ad["det_b"].loaded == 0
    assert [e["ontology_path"] for e in res.snapshot["entities"]] == ["fallback/unknown_object"]  # reprocessing never adopted
    assert {r.engine_id: r.status for r in EvidenceMatrix(tmp_path).runs()} == {"det_a": "succeeded", "det_b": "cancelled"}


def test_refinement_can_be_switched_off_and_says_so(tmp_path):
    res, ad = _run(tmp_path, [(UNKNOWN, X)], [("chair", X)], refine=False)
    assert ad["det_b"].loaded == 0 and _hist(res)["stop_reason"] == "not_run"
    assert [e["ontology_path"] for e in res.snapshot["entities"]] == ["fallback/unknown_object"]
