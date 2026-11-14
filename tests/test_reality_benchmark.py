"""The benchmark registry is coherent and the scorer cannot be talked into a pass: unmeasured is never scored, a failed
critical gate beats a high weighted score, and small objects must be measured on their own."""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from benchmarks.reality_benchmark.manifests import manifest_template
from benchmarks.reality_benchmark.registry import (
    CATEGORIES, ENTITY_GROUPS, GATES, METRIC_BY_ID, METRICS, SCENARIO_BY_ID, SCENARIOS, SMALL_OBJECT_GATE_ID,
    applicable_metrics,
)
from benchmarks.reality_benchmark.scorer import (
    aggregate, blank_results, evaluate_metric, score_scenario, validate_results, wilson_ci,
)

SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "reality_benchmark" / "result.schema.json"


def _passing(scenario_id, skip=()):
    """Every applicable metric measured and passing (scene-specific ones get an explicit threshold)."""
    out = []
    for m in applicable_metrics(SCENARIO_BY_ID[scenario_id]):
        if m.id in skip:
            continue
        if m.op is None:
            out.append(evaluate_metric(m, 0.5, 100, threshold={"op": "<=", "value": 1.0}))
            continue
        value = {">=": m.target + 1, "<=": max(m.target - 1, 0), "==": m.target}[m.op]
        out.append(evaluate_metric(m, value, 100))
    return out


def test_registry_is_coherent():
    assert sum(c.weight for c in CATEGORIES) == pytest.approx(100.0)
    assert len(SCENARIOS) == 24 and len({s.id for s in SCENARIOS}) == 24 and len({s.slug for s in SCENARIOS}) == 24
    assert len({m.id for m in METRICS}) == len(METRICS)
    cats = {c.id for c in CATEGORIES}
    assert all(m.category in cats for m in METRICS)
    assert all(set(s.categories) <= cats and set(s.entity_groups) <= set(ENTITY_GROUPS) for s in SCENARIOS)
    assert all(mid in METRIC_BY_ID and METRIC_BY_ID[mid].critical for g in GATES for mid in g.metric_ids)
    assert all(m.small_object for m in METRICS if m.category == "E")
    assert all((m.op is None) == (m.target is None) for m in METRICS)


def test_nothing_measured_means_incomplete_never_a_score():
    card = score_scenario("001", blank_results(SCENARIO_BY_ID["001"]))
    assert card["verdict"] == "INCOMPLETE" and card["weighted_score"] is None and card["coverage"] == 0.0
    statuses = {r["status"] for r in card["metric_results"]}
    assert statuses <= {"NOT_RUN", "BLOCKED"} and "BLOCKED" in statuses                # geometry etc. need ground truth
    assert {r["status"] for r in card["metric_results"] if not METRIC_BY_ID[r["metric_id"]].needs_ground_truth} == {"NOT_RUN"}
    assert all(c["score"] is None for c in card["categories"].values())


def test_metric_statuses():
    acc = METRIC_BY_ID["A.precision"]
    assert evaluate_metric(acc, 97.0, 200)["status"] == "PASS"
    fail = evaluate_metric(acc, 80.0, 200)
    assert fail["status"] == "FAIL" and "95" in fail["reason"]
    prov = evaluate_metric(acc, 99.0, 5)                                                 # too few samples to claim a pass
    assert prov["status"] == "PROVISIONAL" and "would PASS" in prov["reason"]
    chamfer = METRIC_BY_ID["B.chamfer_distance"]
    assert evaluate_metric(chamfer, 0.02, 500)["status"] == "PROVISIONAL"                # scene-specific, no threshold supplied
    assert evaluate_metric(chamfer, 0.02, 500, threshold={"op": "<=", "value": 0.05})["status"] == "PASS"
    assert evaluate_metric(acc, None)["status"] == "NOT_RUN"
    assert evaluate_metric(acc, None, blocked_reason="no ground truth")["status"] == "BLOCKED"
    with pytest.raises(ValueError):
        evaluate_metric(acc, math.nan, 100)


def test_a_failed_critical_gate_beats_a_high_weighted_score():
    results = _passing("001")
    bad = METRIC_BY_ID["G.invalid_commits_reported_success"]
    results = [r for r in results if r["metric_id"] != bad.id] + [evaluate_metric(bad, 1, 10)]   # one invalid commit reported OK
    card = score_scenario("001", results, ground_truth_available=True)
    assert card["weighted_score"] > 0.95
    gate = next(g for g in card["gates"] if g["gate_id"] == "worldir_commit_integrity")
    assert gate["status"] == "FAIL" and card["verdict"] == "FAIL"


def test_every_applicable_metric_passing_is_a_pass():
    card = score_scenario("018", _passing("018"), ground_truth_available=True)
    assert card["verdict"] == "PASS" and card["coverage"] == 1.0 and card["weighted_score"] == 1.0
    assert all(g["status"] in ("PASS", "NOT_APPLICABLE") for g in card["gates"])


def test_small_objects_must_be_measured_on_their_own():
    skip = {m.id for m in applicable_metrics(SCENARIO_BY_ID["010"]) if m.small_object}
    card = score_scenario("010", _passing("010", skip=skip), ground_truth_available=True)
    gate = next(g for g in card["gates"] if g["gate_id"] == SMALL_OBJECT_GATE_ID)
    assert gate["status"] == "NOT_RUN" and card["verdict"] == "INCOMPLETE"               # excellent everything-else is not a pass
    # scenario 001 only scores the entity groups it contains
    ids = {m.id for m in applicable_metrics(SCENARIO_BY_ID["001"]) if m.category == "E"}
    assert "E.electrical.recall" in ids and "E.street.recall" not in ids


def test_results_are_validated_for_honesty():
    spec = METRIC_BY_ID["A.recall"]
    ok = evaluate_metric(spec, 96.0, 100)
    assert validate_results([ok]) == []
    assert validate_results([{**ok, "sample_count": 0}])                                  # PASS with no samples
    assert validate_results([{**blank_results(SCENARIO_BY_ID["001"])[0], "value": 5.0}])  # an unmeasured metric carrying a value
    assert validate_results([{**ok, "metric_id": "A.nope"}])
    assert validate_results([ok, ok])                                                     # reported twice
    assert validate_results([{**ok, "confidence_interval_95": {"lower": 0.9, "upper": 0.1}}])
    with pytest.raises(ValueError):
        score_scenario("001", [{**ok, "status": "MAYBE"}])


def test_wilson_interval():
    lo, hi = wilson_ci(95, 100)
    assert 0.88 < lo < 0.95 < hi < 0.99
    assert wilson_ci(0, 0) == (None, None)


def test_manifests_start_empty_and_demand_thresholds_for_scene_specific_metrics():
    m = manifest_template(SCENARIO_BY_ID["001"])
    assert m["ground_truth_status"] == "missing" and m["camera_poses"]["verified"] is False
    assert m["datasets"]["A_synthetic"]["images"] == [] and m["datasets"]["B_real"]["images"] == []
    assert "B.chamfer_distance" in m["tolerances"] and m["tolerances"]["B.chamfer_distance"] is None
    assert SMALL_OBJECT_GATE_ID in m["pass_fail"]["critical_gates"]
    assert SMALL_OBJECT_GATE_ID not in manifest_template(SCENARIO_BY_ID["018"])["pass_fail"]["critical_gates"]


def test_aggregate_takes_the_worst_gate_and_pools_categories():
    good = score_scenario("018", _passing("018"), ground_truth_available=True)
    blank = score_scenario("001", blank_results(SCENARIO_BY_ID["001"]))
    agg = aggregate([good, blank])
    assert agg["verdict"] == "INCOMPLETE" and agg["scenarios_by_verdict"] == {"PASS": 1, "FAIL": 0, "INCOMPLETE": 1}
    assert agg["gates"]["worldir_commit_integrity"] == "NOT_RUN"
    assert agg["categories"]["C"]["measured"] > 0 and agg["categories"]["A"]["score"] is None


def test_cli_dashboard_says_nothing_was_measured(capsys):
    from scripts.run_reality_benchmark import main

    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Awaiting execution" in out and "INCOMPLETE" in out


def test_cli_emits_one_manifest_per_scenario(tmp_path, capsys):
    from scripts.run_reality_benchmark import main

    assert main(["--emit-manifests", str(tmp_path)]) == 0
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 24 and json.loads(files[0].read_text())["scenario_id"] == "001"


def test_scorecard_matches_the_published_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    for sid in ("001", "018"):
        card = score_scenario(sid, blank_results(SCENARIO_BY_ID[sid]))
        jsonschema.validate(card, schema)
    jsonschema.validate(score_scenario("018", _passing("018"), True), schema)
