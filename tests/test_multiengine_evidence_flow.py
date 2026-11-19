"""Evidence-to-world flow + evidence-driven capture requests. Engines are SYNTHETIC test doubles (no real model runs here); they
prove orchestration, persistence, cancellation, partial results and the capture-request rules, not model accuracy."""

from __future__ import annotations

import pytest

from engine.multiengine.association import PairScore
from engine.multiengine.capture_requests import ProblemEvidence, request_for_problem
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
from engine.multiengine.evidence_flow import (
    LIMITATIONS,
    StageSpec,
    load_snapshot,
    run_evidence_flow,
)
from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.registry import EngineRegistry
from engine.multiengine.worldir_bridge import trace

OK = Availability(AvailabilityStatus.AVAILABLE, "verified in test", evidence=("test double",))
SOURCES = ("img1.jpg", "img2.jpg")


def _spec(eid, family):
    return EngineSpec(
        engine_id=eid, name=eid, version="v1", tasks=frozenset({Task.DETECTION}), input_schema="img", output_schema="bbox",
        probe=lambda: OK, family=family, commercial_ok=True, licence="Apache-2.0", hardware=HardwareRequirement(),
    )


class FakeDetector(EngineAdapter):
    """SYNTHETIC: detects a chair (and a tiny socket) in image 1, and a chair in image 2."""

    def __init__(self, sp, fail=False):
        self.spec, self.fail, self.loaded = sp, fail, 0

    def validate_input(self, request):
        pass

    def load(self, device):
        self.loaded += 1

    def unload(self):
        pass

    def infer(self, request):
        if self.fail:
            raise EngineError(EngineErrorKind.MODEL_ERROR, "synthetic crash")

        def d(src, label, box):
            return ObservationDraft(task="detection", quantity="bbox", source_id=src, coordinate_system=f"image:{src}",
                                    image_region=box, predicted={"label": label, "score": 0.9}, confidence=0.9)

        return [d("img1.jpg", "chair", (10, 10, 200, 300)), d("img1.jpg", "power outlet", (400, 400, 406, 408)),
                d("img2.jpg", "chair", (50, 40, 260, 320))]


def _setup(fail=()):
    reg = EngineRegistry()
    adapters = {}
    for eid, fam in (("det_a", "fam-a"), ("det_b", "fam-b")):
        sp = _spec(eid, fam)
        reg.register(sp)
        adapters[eid] = FakeDetector(sp, fail=eid in fail)
    return reg, adapters


def _run(tmp_path, **kw):
    reg, adapters = _setup(kw.pop("fail", ()))
    res = run_evidence_flow(tmp_path, "run-1", SOURCES, registry=reg, adapters=adapters,
                            stages=[StageSpec(Task.DETECTION)], **kw)
    return res, adapters


def test_flow_persists_runs_builds_entities_and_states_its_limits(tmp_path):
    res, _ = _run(tmp_path)
    snap = res.snapshot
    assert res.status == "complete" and snap["counts"]["observations"] == 6 and snap["counts"]["runs"] == 2
    assert snap["counts"]["candidates"] == 6 and snap["counts"]["entities"] == 6  # no pair scorer: no cross-view fusion
    assert set(snap["limitations"]) >= set(LIMITATIONS) and any("pair scorer" in x for x in snap["limitations"])
    for ent in snap["entities"]:
        world = ent["world_entity"]
        assert world["geometry_ids"] == []  # 2D boxes only: no geometry is fabricated
        assert world["confidence"] == 0.5 and "evidence" in world["custom_properties"]
    assert any("no geometry created" in n for n in snap["bridge_notes"])
    assert snap["matrix_problems"] == []
    assert snap["hierarchy"]["city_scale_claimable"] is False


def test_every_entity_traces_back_to_images_and_matrix_rows(tmp_path):
    res, _ = _run(tmp_path)
    matrix = EvidenceMatrix(tmp_path)  # a fresh open: nothing in memory
    for ent in res.bridge.entities.values():
        assert trace(ent, matrix) == []
        assert {o.data_uri for o in ent.observations} <= set(SOURCES)


def test_snapshot_and_matrix_survive_a_restart(tmp_path):
    res, _ = _run(tmp_path)
    assert load_snapshot(tmp_path) == res.snapshot
    again = EvidenceMatrix(tmp_path)
    assert len(again) == 6 and len(again.runs("run-1")) == 2 and again.verify() == []


def test_capture_requests_rest_on_measured_facts(tmp_path):
    res, _ = _run(tmp_path)
    by_path = {}
    for r in res.capture_requests:
        by_path.setdefault(r.target_region["ontology_path"], []).append(r)
    chair_views = {v.kind for r in by_path["furniture/seating/chair"] for v in r.suggested_views}
    assert {"side_view", "reverse_view"} <= chair_views  # size unmeasured; single viewpoint -> rear UNOBSERVED
    socket_views = {v.kind for r in by_path["fixtures/electrical/socket"] for v in r.suggested_views}
    assert "close_up" in socket_views  # a 6x8 px box is below the ontology's min_pixels for a socket
    assert all(r.expected_gain is None for r in res.capture_requests)  # no invented benefit
    text = " ".join(w for r in res.capture_requests for w in r.why_insufficient)
    assert "does not mean it is absent" in text  # an unseen rear is never called absent


def test_cross_view_association_removes_the_reverse_view_request_but_not_the_missing_size(tmp_path):
    def scorer(cands, matrix):
        chairs = [c for c in cands if c.label == "chair"]
        a = next(c for c in chairs if c.source_id == "img1.jpg")
        b = next(c for c in chairs if c.source_id == "img2.jpg")
        return [PairScore(a.observation_id, b.observation_id, 0.9, {"note": "synthetic scorer"})]

    res, _ = _run(tmp_path, pair_scorer=scorer)
    merged = next(e for e in res.canonicals if len(e.members) == 2)
    reqs = [r for r in res.capture_requests if r.target_ids == merged.members]
    kinds = {v.kind for r in reqs for v in r.suggested_views}
    assert "side_view" in kinds and "reverse_view" not in kinds  # two viewpoints: rear not known to be unobserved
    assert all("pair scorer" not in x for x in res.snapshot["limitations"])
    merged_decisions = [d for d in res.snapshot["associations"] if d["outcome"] == "merged"]
    assert any(d["basis"] == {"note": "synthetic scorer"} for d in merged_decisions)


def test_partial_when_one_engine_fails_and_its_failure_is_preserved(tmp_path):
    res, _ = _run(tmp_path, fail=("det_b",))
    assert res.status == "partial" and res.snapshot["counts"]["observations"] == 3
    runs = {r.engine_id: r for r in EvidenceMatrix(tmp_path).runs()}
    assert runs["det_a"].status == "succeeded" and runs["det_b"].status == "failed"
    assert "synthetic crash" in runs["det_b"].error


def test_all_engines_failing_is_failed_not_complete_and_makes_no_entities(tmp_path):
    res, _ = _run(tmp_path, fail=("det_a", "det_b"))
    assert res.status == "failed" and res.snapshot["counts"]["entities"] == 0 and res.capture_requests == []
    assert len(EvidenceMatrix(tmp_path).runs()) == 2  # both failures recorded


def test_cancellation_keeps_what_finished_and_marks_the_rest_cancelled(tmp_path):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2  # stage start, first engine start: go; second engine: stop

    res, adapters = _run(tmp_path, should_cancel=cancel)
    assert res.cancelled and res.status == "cancelled"
    assert adapters["det_a"].loaded == 1 and adapters["det_b"].loaded == 0
    runs = {r.engine_id: r.status for r in EvidenceMatrix(tmp_path).runs()}
    assert runs == {"det_a": "succeeded", "det_b": "cancelled"}
    assert res.snapshot["counts"]["observations"] == 3  # partial work is a durable result


def test_a_successful_run_that_finds_nothing_is_complete_not_failed(tmp_path):
    """REGRESSION (found on a real run): OWL-ViT ran fine on outdoor photos and detected nothing; the flow called that 'failed'."""

    class Empty(FakeDetector):
        def infer(self, request):
            return []

    reg = EngineRegistry()
    sp = _spec("det_a", "fam-a")
    reg.register(sp)
    res = run_evidence_flow(tmp_path, "run-1", SOURCES, registry=reg, adapters={"det_a": Empty(sp)},
                            stages=[StageSpec(Task.DETECTION)])
    assert res.status == "complete" and res.snapshot["counts"]["entities"] == 0
    assert any("no detections" in x for x in res.snapshot["limitations"])  # said plainly; no entity is invented


def test_detector_adapters_call_the_prompter_with_its_real_signature():
    """REGRESSION (found on a real run): the adapters passed scale_tier= to get_prompts(), which only takes scale=; every real
    Grounding DINO / OWL-ViT run raised TypeError. The test doubles never reach this code, so guard it directly."""
    import inspect

    from engine.multiengine.adapters import perception_detectors
    from perception.detection.extensible_prompter import ExtensiblePrompter, ScaleFilter

    assert "get_prompts(scale_tier" not in inspect.getsource(perception_detectors)
    prompts = ExtensiblePrompter().get_prompts(scale=ScaleFilter.MESO)
    assert prompts and all(isinstance(p, str) for p in prompts)
    with pytest.raises(TypeError):
        ExtensiblePrompter().get_prompts(scale_tier="meso")  # type: ignore[call-arg]


def _real_owlvit_ready():
    from pathlib import Path

    from perception.availability import real_models_by_name

    try:
        ok = real_models_by_name()["owlvit"].status == "available"
    except Exception:  # noqa: BLE001
        ok = False
    return ok and Path("datasets/south_building/images/P1180141.JPG").is_file()


@pytest.mark.skipif(not _real_owlvit_ready(), reason="real OWL-ViT checkpoint and the South Building photos are not present")
def test_REAL_model_owlvit_runs_through_the_flow_and_is_persisted(tmp_path):
    """REAL model execution (skipped where the checkpoint is absent): OWL-ViT on a real photo, through the real adapter."""
    from engine.multiengine.adapters import create_adapters_for_registry
    from engine.multiengine.executor import TorchResourceProbe
    from engine.multiengine.registry import maximum_fidelity_registry

    reg = maximum_fidelity_registry()
    res = run_evidence_flow(
        tmp_path, "real-1", ["datasets/south_building/images/P1180141.JPG"], registry=reg,
        adapters=create_adapters_for_registry(reg), stages=[StageSpec(Task.DETECTION, only=("owlvit",))],
        probe=TorchResourceProbe(),
    )
    run = EvidenceMatrix(tmp_path).runs()[0]
    assert run.engine_id == "owlvit" and run.status == "succeeded" and run.device is not None and run.error == ""
    assert res.status == "complete" and res.snapshot["matrix_problems"] == []  # zero detections on outdoor photos is not a failure
    # no claim about accuracy or about furnished rooms: only that the real adapter executed and its evidence persisted


# ------------------------------------------------------------------------------------------------ capture-request rules
def _req(kind, **ev):
    return request_for_problem(problem_kind=kind, location={}, target_ids=["x"], evidence=ProblemEvidence(**ev))


def test_side_reverse_and_architectural_requests_each_need_their_fact():
    assert _req("incorrect_dimension") is None
    side = _req("incorrect_dimension", unmeasured_dimension_axes=("d", "h"))
    assert [v.kind for v in side.suggested_views] == ["side_view"] and "d, h" in side.why_insufficient[0]
    rear = _req("incorrect_dimension", rear_observed=False)
    assert [v.kind for v in rear.suggested_views] == ["reverse_view"]
    assert _req("incorrect_dimension", rear_observed=True) is None  # observed rear: nothing to ask for
    wall = _req("incomplete_wall", inconsistent_estimates=True)
    assert [v.kind for v in wall.suggested_views] == ["architectural_view"]
    # the architectural rule is for architectural problems only: an inconsistent chair does not trigger it
    assert _req("incorrect_dimension", inconsistent_estimates=True) is None


@pytest.mark.parametrize("kind", ["incomplete_wall", "missing_corner", "missing_architectural_feature"])
def test_architectural_kinds_all_trigger_the_architectural_view(kind):
    r = _req(kind, inconsistent_estimates=True)
    assert r is not None and r.suggested_views[0].kind == "architectural_view"
