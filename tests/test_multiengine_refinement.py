"""Quality gates, capture requests, the refinement loop and the /api/engines view.

Engines are TEST DOUBLES and the "reconstruction" is a synthetic dict. These tests prove the orchestration rules (no false success,
targeted reprocessing, rollback on inconsistency or regression, evidence requests instead of futile reruns). They say nothing about
how any real engine performs.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.multiengine.capture_requests import ProblemEvidence, request_for_problem
from engine.multiengine.contract import Availability, AvailabilityStatus, EngineAdapter, EngineSpec, HardwareRequirement, Task
from engine.multiengine.executor import ExecutionManager, Policy
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.quality_gates import (
    GateSpec, GateVerdict, Measurement, Overall, check_regressions, default_gate_specs, evaluate,
)
from engine.multiengine.refinement import Problem, ProblemCause, ProblemKind, RefinementLoop, StopReason
from engine.multiengine.registry import EngineRegistry

OK = Availability(AvailabilityStatus.AVAILABLE, "test double", evidence=("test double",))


def m(v):
    return Measurement(v, "test measurement")


# ------------------------------------------------------------------------------------------------ quality gates

def test_a_gate_without_a_benchmark_threshold_can_never_pass():
    specs = default_gate_specs()
    assert len(specs) == 15 and all(s.threshold is None for s in specs)
    q = evaluate(specs, {s.name: m(0.0) for s in specs}, stages_ok=True)
    assert q.overall is Overall.UNVERIFIED and all(g.verdict is GateVerdict.UNVERIFIED for g in q.gates)
    assert any("does not make the reconstruction complete" in n for n in q.notes)


def test_a_threshold_must_cite_the_benchmark_it_came_from():
    with pytest.raises(ValueError):
        GateSpec("reprojection_error", "lower_is_better", threshold=1.0)
    with pytest.raises(ValueError):
        default_gate_specs({"made_up_gate": (1.0, "x")})
    assert GateSpec("reprojection_error", "lower_is_better", 1.0, "benchmarks/x.json").threshold == 1.0


def specs2():
    return [GateSpec("reprojection_error", "lower_is_better", 1.0, "benchmarks/a.json"),
            GateSpec("surface_coverage", "higher_is_better", 0.9, "benchmarks/a.json")]


def test_overall_is_complete_only_when_every_required_gate_was_measured_and_passed():
    assert evaluate(specs2(), {"reprojection_error": m(0.5), "surface_coverage": m(0.95)}).overall is Overall.COMPLETE
    partial = evaluate(specs2(), {"reprojection_error": m(0.5)})
    assert partial.overall is Overall.PARTIAL and "surface_coverage" in partial.notes[0]
    assert evaluate(specs2(), {}).overall is Overall.UNVERIFIED
    failed = evaluate(specs2(), {"reprojection_error": m(3.0), "surface_coverage": m(0.95)})
    assert failed.overall is Overall.FAILED


def test_a_failed_non_required_gate_does_not_fail_the_result_but_a_failed_stage_does():
    specs = [GateSpec("reprojection_error", "lower_is_better", 1.0, "b"),
             GateSpec("surface_coverage", "higher_is_better", 0.9, "b", required=False)]
    assert evaluate(specs, {"reprojection_error": m(0.5), "surface_coverage": m(0.1)}).overall is Overall.COMPLETE
    assert evaluate(specs, {"reprojection_error": m(0.5)}, stages_ok=False).overall is Overall.FAILED


def test_regressions_of_previously_validated_features_fail_the_result():
    prev = {"wall_1.length": 4.00, "door_1": "present", "floor.area": 20.0}
    cur = {"wall_1.length": 4.02, "door_1": "present"}
    regs = check_regressions(prev, cur, tolerance={"wall_1.length": 0.05})
    assert [(r.feature_id, r.kind) for r in regs] == [("floor.area", "missing")]
    assert check_regressions({"a": 1.0}, {"a": 1.0001}, tolerance={})[0].kind == "changed"     # no tolerance stated -> exact match
    q = evaluate(specs2(), {"reprojection_error": m(0.5), "surface_coverage": m(0.95)}, regressions=regs)
    assert q.overall is Overall.FAILED and "regressed" in " ".join(q.notes)


# ------------------------------------------------------------------------------------------------ capture requests

def test_a_capture_request_states_only_what_was_measured_and_never_invents_a_gain():
    ev = ProblemEvidence(n_registered_views=1, view_ids=("img_3",), truncated_at_frame_edge=True)
    req = request_for_problem(problem_kind="incomplete_wall", location={"frame": "world", "wall": "w2"}, target_ids=["w2"],
                              evidence=ev, basis=["obs_1"])
    assert len(req.why_insufficient) == 2 and "only 1 registered view" in req.why_insufficient[0]
    assert {v.kind for v in req.suggested_views} == {"new_viewpoint", "wider_context"}
    assert req.overlap_with == ("img_3",) and "img_3" in req.instruction and req.expected_gain is None
    row = req.to_row(world_id="w1", request_id="cr_1", created_version="v2")
    assert row["status"] == "open" and row["expected_gain"] is None and row["target_ids"] == ["w2"] and len(row["reason"]) <= 300


def test_no_measured_fact_means_no_request_and_a_scale_reference_is_asked_for_only_for_dimensions():
    assert request_for_problem(problem_kind="incomplete_wall", location={}, target_ids=[], evidence=ProblemEvidence()) is None
    ev = ProblemEvidence(scale_reference_present=False)
    assert request_for_problem(problem_kind="incomplete_wall", location={}, target_ids=[], evidence=ev) is None
    dim = request_for_problem(problem_kind="incorrect_dimension", location={}, target_ids=[], evidence=ev)
    assert dim.scale_reference_needed and dim.suggested_views[0].kind == "scale_reference"
    close = request_for_problem(problem_kind="missed_small_object", location={}, target_ids=["o1"],
                                evidence=ProblemEvidence(pixels_on_target=12, pixels_on_target_needed=48))
    assert close.suggested_views[0].kind == "close_up" and "12 px" in close.why_insufficient[0]


# ------------------------------------------------------------------------------------------------ refinement loop

class FakeAdapter(EngineAdapter):
    """TEST DOUBLE: returns one draft, or raises the scripted error."""

    def __init__(self, sp, fail=None):
        self.spec = sp
        self.fail = fail
        self.loaded = 0

    def validate_input(self, request):
        pass

    def load(self, device):
        self.loaded += 1

    def unload(self):
        pass

    def infer(self, request):
        if self.fail:
            raise self.fail
        return [ObservationDraft("depth", "depth_value", "img1", "cam:img1", {"v": 1.0})]


def spec(eid, probe=None, priority=100):
    return EngineSpec(engine_id=eid, name=eid, version="1", tasks=frozenset({Task.DEPTH}), input_schema="i", output_schema="o",
                      probe=probe or (lambda: OK), commercial_ok=True, licence="Apache-2.0", priority=priority,
                      hardware=HardwareRequirement())


GATES = [GateSpec("reprojection_error", "lower_is_better", 1.0, "benchmarks/a.json")]


def build(engines, *, apply_fn, consistency=lambda s: [], features=lambda s: {}, max_iterations=5, patience=1, problem=None,
          tolerance=None, policy=Policy.MAXIMUM_FIDELITY):
    reg = EngineRegistry()
    adapters = {}
    for sp, ad in engines:
        reg.register(sp)
        adapters[sp.engine_id] = ad
    mgr = ExecutionManager(reg, adapters, policy=policy)
    prob = problem or Problem("p1", ProblemKind.INCONSISTENT_DEPTH, ProblemCause.ENGINE_DISAGREEMENT, {"frame": "world"},
                              tasks=(Task.DEPTH,), target_ids=("img1",))

    def measure(state):
        return {"reprojection_error": m(state["err"])}, True

    def detect(state, q):
        return [prob] if state["err"] > 1.0 else []

    loop = RefinementLoop(manager=mgr, registry=reg, specs=GATES, measure=measure, detect=detect, apply=apply_fn,
                          consistency=consistency, features=features, regression_tolerance=tolerance,
                          max_iterations=max_iterations, patience=patience)
    return loop, adapters


def improve(by):
    return lambda state, problem, report: {**state, "err": state["err"] - by}


def test_a_problem_an_available_engine_can_address_is_reprocessed_and_the_loop_stops_when_the_target_is_met():
    sp = spec("alt")
    loop, ad = build([(sp, FakeAdapter(sp))], apply_fn=improve(4.5))
    res = loop.run({"err": 5.0})
    assert res.stop_reason is StopReason.TARGET_MET and res.final_quality.overall is Overall.COMPLETE and res.state["err"] == 0.5
    assert [a.outcome for a in res.iterations[0].actions] == ["applied"] and res.iterations[0].improved
    assert ad["alt"].loaded == 1


def test_an_update_that_breaks_global_consistency_is_rejected_and_the_previous_state_is_kept():
    sp = spec("alt")
    loop, _ = build([(sp, FakeAdapter(sp))], apply_fn=improve(4.5), consistency=lambda s: ["wall w2 now intersects floor"])
    res = loop.run({"err": 5.0})
    assert res.state["err"] == 5.0                                     # nothing was applied
    assert res.iterations[0].actions[0].outcome == "rejected_inconsistent" and "intersects" in res.iterations[0].actions[0].detail
    assert res.stop_reason is not StopReason.TARGET_MET


def test_an_update_that_regresses_a_validated_feature_is_rejected():
    sp = spec("alt")
    loop, _ = build([(sp, FakeAdapter(sp))], apply_fn=lambda s, p, r: {**s, "err": 0.5, "wall": 3.0},
                    features=lambda s: {"wall": s.get("wall", 4.0)}, tolerance={"wall": 0.05})
    res = loop.run({"err": 5.0, "wall": 4.0})
    assert res.state == {"err": 5.0, "wall": 4.0} and res.iterations[0].actions[0].outcome == "rejected_regression"


def test_insufficient_evidence_becomes_a_capture_request_and_no_engine_is_rerun():
    sp = spec("alt")
    prob = Problem("p9", ProblemKind.INCOMPLETE_WALL, ProblemCause.INSUFFICIENT_EVIDENCE, {"wall": "w2"}, tasks=(Task.DEPTH,),
                   evidence=ProblemEvidence(n_registered_views=1, view_ids=("img_3",)), target_ids=("w2",))
    loop, ad = build([(sp, FakeAdapter(sp))], apply_fn=improve(1.0), problem=prob)
    res = loop.run({"err": 5.0})
    assert res.stop_reason is StopReason.AWAITING_EVIDENCE and len(res.capture_requests) == 1 and ad["alt"].loaded == 0
    assert res.capture_requests[0].overlap_with == ("img_3",) and res.iterations[0].actions[0].outcome == "needs_evidence"


def test_a_problem_with_no_measured_reason_to_ask_for_photos_stays_unresolved_instead_of_a_futile_request():
    sp = spec("alt")
    prob = Problem("p9", ProblemKind.MISSING_CORNER, ProblemCause.UNKNOWN, {}, tasks=(Task.DEPTH,))
    loop, ad = build([(sp, FakeAdapter(sp))], apply_fn=improve(1.0), problem=prob)
    res = loop.run({"err": 5.0})
    assert res.iterations[0].actions[0].outcome == "unresolved" and not res.capture_requests and ad["alt"].loaded == 0
    assert res.stop_reason is StopReason.NO_ENGINE_AVAILABLE and res.unresolved == ("p9",)


def test_an_unavailable_engine_is_not_run_and_the_problem_is_reported_exhausted():
    sp = spec("alt", probe=lambda: Availability(AvailabilityStatus.BLOCKED, "checkpoint missing"))
    loop, ad = build([(sp, FakeAdapter(sp))], apply_fn=improve(4.5))
    res = loop.run({"err": 5.0})
    assert res.iterations[0].actions[0].outcome == "exhausted" and ad["alt"].loaded == 0
    assert res.stop_reason is StopReason.NO_ENGINE_AVAILABLE and res.final_quality.overall is not Overall.COMPLETE


def test_a_failed_engine_is_not_retried_for_the_same_problem_and_the_failure_is_reported():
    sp = spec("alt")
    loop, ad = build([(sp, FakeAdapter(sp, fail=ValueError("shape mismatch")))], apply_fn=improve(4.5), max_iterations=4)
    res = loop.run({"err": 5.0})
    first = res.iterations[0].actions[0]
    assert first.outcome == "engine_failed" and "shape mismatch" in first.detail
    assert ad["alt"].loaded == 1                                        # tried once, never again for this problem
    assert res.state["err"] == 5.0


def test_only_untried_engines_are_used_so_a_second_engine_gets_its_turn():
    a, b = spec("a", priority=1), spec("b", priority=2)
    loop, ad = build([(a, FakeAdapter(a, fail=ValueError("x"))), (b, FakeAdapter(b))], apply_fn=improve(4.5), patience=2)
    res = loop.run({"err": 5.0})
    assert res.stop_reason is StopReason.TARGET_MET and ad["a"].loaded == 1 and ad["b"].loaded == 1


def test_stagnation_stops_the_loop_with_no_improvement():
    a, b = spec("a", priority=1), spec("b", priority=2)
    loop, _ = build([(a, FakeAdapter(a)), (b, FakeAdapter(b))], apply_fn=lambda s, p, r: {**s, "touched": True}, patience=1)
    res = loop.run({"err": 5.0})
    assert res.stop_reason is StopReason.NO_IMPROVEMENT and not res.iterations[0].improved


def test_the_iteration_budget_is_a_hard_stop_that_is_reported_as_such():
    built = [(spec(f"e{i}", priority=i), None) for i in range(4)]
    built = [(sp, FakeAdapter(sp)) for sp, _ in built]
    loop, ad = build(built, apply_fn=improve(0.5), max_iterations=2, patience=3, policy=Policy.ACCURATE)   # one engine per step
    res = loop.run({"err": 9.0})
    assert res.stop_reason is StopReason.MAX_ITERATIONS and len(res.iterations) == 2 and res.final_quality.overall is not Overall.COMPLETE
    assert sorted(a.loaded for a in ad.values()) == [0, 0, 1, 1]                  # two iterations, two different engines
    assert res.state["err"] == 8.0


def test_an_unchanged_reconciliation_is_recorded_as_unchanged_not_as_an_improvement():
    sp = spec("alt")
    loop, _ = build([(sp, FakeAdapter(sp))], apply_fn=lambda s, p, r: s)
    res = loop.run({"err": 5.0})
    assert res.iterations[0].actions[0].outcome == "unchanged" and not res.iterations[0].improved


def test_nothing_to_refine_but_unverified_is_never_reported_as_target_met():
    sp = spec("alt")
    reg = EngineRegistry()
    reg.register(sp)
    mgr = ExecutionManager(reg, {"alt": FakeAdapter(sp)})
    loop = RefinementLoop(manager=mgr, registry=reg, specs=default_gate_specs(),
                          measure=lambda s: ({"reprojection_error": m(0.1)}, True), detect=lambda s, q: [],
                          apply=lambda s, p, r: s, consistency=lambda s: [])
    res = loop.run({"anything": 1})
    assert res.stop_reason is StopReason.NOTHING_TO_REFINE and res.final_quality.overall is Overall.UNVERIFIED


def test_the_result_serialises_for_the_api_without_losing_what_happened():
    sp = spec("alt")
    loop, _ = build([(sp, FakeAdapter(sp))], apply_fn=improve(4.5))
    d = loop.run({"err": 5.0}).to_dict()
    assert d["stop_reason"] == "target_met" and d["iterations"][0]["actions"][0]["engines"] == ["alt"]


# ------------------------------------------------------------------------------------------------ API

def client():
    from apps.api.routes_engines import engines_router

    app = FastAPI()
    app.include_router(engines_router)
    return TestClient(app)


def test_engines_endpoint_reports_real_availability_integration_level_and_policies():
    from perception.availability import real_models_by_name

    res = client().get("/api/engines")
    assert res.status_code == 200
    body = res.json()
    real = real_models_by_name()
    assert {e["engine_id"] for e in body["engines"]} == set(real)
    for e in body["engines"]:
        assert e["availability"]["status"] == real[e["engine_id"]].status
        assert e["hardware"]["min_vram_mb"] is None
    assert body["summary"]["total"] == 6 and body["summary"]["adapters_implemented"] == 0
    assert {p["policy"] for p in body["policies"]} == {"fast", "accurate", "maximum_fidelity"}


def test_single_engine_and_quality_gate_endpoints():
    c = client()
    assert c.get("/api/engines/sam").json()["integration"] == "registry-only"
    assert c.get("/api/engines/nope").status_code == 404
    gates = c.get("/api/engines/quality-gates").json()
    assert gates["count"] == 15 and all(g["threshold"] is None for g in gates["gates"])


def test_the_main_application_mounts_the_engines_router():
    from apps.api.main import app

    # The OpenAPI schema is the stable route table: on newer FastAPI, ``app.routes`` nests included routers without a ``.path``.
    assert any(p.rstrip("/") == "/api/engines" for p in app.openapi()["paths"])
