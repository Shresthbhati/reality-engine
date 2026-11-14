"""Benchmark scorer: turns measured metric values into honest metric results, gates and a weighted scorecard.

Rules that make the score trustworthy:
* A metric without a measurement is NOT_RUN (or BLOCKED when it needs ground truth that does not exist). It is never
  scored, never defaulted, and never counted as a pass.
* A measurement with too few samples, or without a threshold (scene-specific and none supplied), is PROVISIONAL.
* Critical gates are evaluated independently: a failed gate fails the scenario whatever the weighted score is.
* The weighted score covers only measured categories, and ``coverage`` says how much of the weight that is.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

from benchmarks.reality_benchmark.registry import (
    BENCHMARK_VERSION, CATEGORY_BY_ID, GATES, METRIC_BY_ID, SCENARIO_BY_ID,
    SMALL_OBJECT_GATE_ID, STATUSES, UNMEASURED, MetricSpec, Scenario, applicable_categories, applicable_metrics,
)

MEASURED = ("PASS", "FAIL")
_OPS = {">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t, "==": lambda v, t: v == t}


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """95% Wilson interval for a proportion, as (lower, upper) fractions; (None, None) when there are no samples."""
    if n <= 0:
        return None, None
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def threshold_for(spec: MetricSpec, override: Optional[dict] = None) -> Optional[dict]:
    """The threshold in force: a scenario-manifest override wins; else the registry target; else None (scene-specific)."""
    if override:
        return {"op": override["op"], "value": float(override["value"])}
    if spec.op is not None and spec.target is not None:
        return {"op": spec.op, "value": float(spec.target)}
    return None


def _result(spec: MetricSpec, value, status: str, n: int, threshold, reason, ci=None, breakdown=None) -> dict:
    lo, hi = (ci if ci else (None, None))
    return {"metric_id": spec.id, "value": value, "unit": spec.unit, "threshold": threshold, "status": status,
            "sample_count": int(n), "confidence_interval_95": {"lower": lo, "upper": hi},
            "breakdown": dict(breakdown or {}), "reason": reason}


def evaluate_metric(spec: MetricSpec, value: Optional[float], sample_count: int = 0, *, threshold: Optional[dict] = None,
                    ci=None, breakdown: Optional[dict] = None, blocked_reason: Optional[str] = None) -> dict:
    """One metric result (schema shape). ``value=None`` means it was not measured."""
    thr = threshold_for(spec, threshold)
    if blocked_reason:
        return _result(spec, None, "BLOCKED", 0, thr, blocked_reason)
    if value is None:
        return _result(spec, None, "NOT_RUN", 0, thr, "not measured")
    if not math.isfinite(float(value)):
        raise ValueError(f"{spec.id}: value must be finite, got {value!r}")
    value = float(value)
    if thr is None:
        return _result(spec, value, "PROVISIONAL", sample_count, None,
                       "scene-specific metric and the scenario manifest supplied no threshold", ci, breakdown)
    meets = _OPS[thr["op"]](value, thr["value"])
    if sample_count < spec.min_samples:
        return _result(spec, value, "PROVISIONAL", sample_count, thr,
                       f"n={sample_count} is below the minimum {spec.min_samples}; the value would {'PASS' if meets else 'FAIL'}",
                       ci, breakdown)
    return _result(spec, value, "PASS" if meets else "FAIL", sample_count, thr,
                   None if meets else f"{value} does not satisfy {thr['op']} {thr['value']}", ci, breakdown)


def blank_results(scenario: Scenario, ground_truth_available: bool = False) -> List[dict]:
    """The 'awaiting execution' results of a scenario: every applicable metric NOT_RUN (BLOCKED when it needs ground
    truth that does not exist), every other metric NOT_APPLICABLE."""
    applicable = {m.id for m in applicable_metrics(scenario)}
    out = []
    for spec in METRIC_BY_ID.values():
        if spec.id not in applicable:
            out.append(_result(spec, None, "NOT_APPLICABLE", 0, threshold_for(spec), "not part of this scenario"))
        elif spec.needs_ground_truth and not ground_truth_available:
            out.append(evaluate_metric(spec, None, blocked_reason="ground truth (geometry, poses, identities) is not available"))
        else:
            out.append(evaluate_metric(spec, None))
    return out


def validate_results(results: Sequence[dict]) -> List[str]:
    """Structural and honesty errors in a list of metric results (empty = valid)."""
    errs: List[str] = []
    seen = set()
    for i, r in enumerate(results):
        mid = r.get("metric_id")
        if mid not in METRIC_BY_ID:
            errs.append(f"[{i}] unknown metric_id {mid!r}")
            continue
        if mid in seen:
            errs.append(f"{mid}: reported twice")
        seen.add(mid)
        for key in ("value", "unit", "threshold", "status", "sample_count", "confidence_interval_95", "breakdown", "reason"):
            if key not in r:
                errs.append(f"{mid}: missing {key!r}")
        st = r.get("status")
        if st not in STATUSES:
            errs.append(f"{mid}: invalid status {st!r}")
            continue
        if st in MEASURED + ("PROVISIONAL",) and r.get("value") is None:
            errs.append(f"{mid}: {st} without a value")
        if st in MEASURED and int(r.get("sample_count", 0)) <= 0:
            errs.append(f"{mid}: {st} with no samples")
        if st in UNMEASURED and r.get("value") is not None:
            errs.append(f"{mid}: {st} must not carry a value")
        ci = r.get("confidence_interval_95") or {}
        if ci.get("lower") is not None and ci.get("upper") is not None and ci["lower"] > ci["upper"]:
            errs.append(f"{mid}: confidence interval lower > upper")
    return errs


def _gate_result(gate_id: str, name: str, status: str, reasons: List[str]) -> dict:
    return {"gate_id": gate_id, "name": name, "status": status, "reasons": reasons}


def evaluate_gates(scenario: Scenario, by_id: Dict[str, dict]) -> List[dict]:
    cats = set(applicable_categories(scenario))
    out = []
    for g in GATES:
        sts, reasons = [], []
        for mid in g.metric_ids:
            if METRIC_BY_ID[mid].category not in cats:
                sts.append("NOT_APPLICABLE")
                continue
            r = by_id.get(mid)
            st = r["status"] if r else "NOT_RUN"
            sts.append(st)
            if st != "PASS":
                reasons.append(f"{mid}: {st}" + (f" ({r['reason']})" if r and r.get("reason") else ""))
        if "FAIL" in sts:
            status = "FAIL"
        elif all(s == "NOT_APPLICABLE" for s in sts):
            status, reasons = "NOT_APPLICABLE", []
        elif all(s in ("PASS", "NOT_APPLICABLE") for s in sts):
            status, reasons = "PASS", []
        else:
            status = "NOT_RUN"
        out.append(_gate_result(g.id, g.name, status, reasons))
    if scenario.entity_groups:                      # small objects must be measured and reported on their own
        small = [m.id for m in applicable_metrics(scenario) if m.small_object]
        missing = [m for m in small if (by_id.get(m) or {"status": "NOT_RUN"})["status"] in UNMEASURED]
        out.append(_gate_result(SMALL_OBJECT_GATE_ID, "Small-object metrics are measured and reported per entity group",
                                "PASS" if not missing else "NOT_RUN",
                                [f"{len(missing)} of {len(small)} small-object metrics not measured"] if missing else []))
    return out


def score_scenario(scenario_id: str, metric_results: Sequence[dict], ground_truth_available: bool = False) -> dict:
    """Scorecard of one scenario: per-category scores, gates, weighted score, coverage and a verdict.

    Verdict: FAIL when any gate or any measured metric fails; PASS only when every applicable metric PASSed;
    otherwise INCOMPLETE (something was not measured, blocked or provisional)."""
    scenario = SCENARIO_BY_ID[scenario_id]
    errors = validate_results(metric_results)
    if errors:
        raise ValueError("invalid metric results: " + "; ".join(errors[:5]))
    by_id = {r["metric_id"]: r for r in metric_results}
    for r in blank_results(scenario, ground_truth_available):   # anything not reported is NOT_RUN / BLOCKED / NOT_APPLICABLE
        by_id.setdefault(r["metric_id"], r)
    applicable = applicable_metrics(scenario)
    cats: Dict[str, dict] = {}
    for cid in applicable_categories(scenario):
        mine = [by_id[m.id] for m in applicable if m.category == cid]
        n_pass = sum(r["status"] == "PASS" for r in mine)
        n_meas = sum(r["status"] in MEASURED for r in mine)
        cats[cid] = {"name": CATEGORY_BY_ID[cid].name, "weight": CATEGORY_BY_ID[cid].weight, "metrics": len(mine),
                     "measured": n_meas, "passed": n_pass, "coverage": round(n_meas / len(mine), 4) if mine else 0.0,
                     "score": round(n_pass / n_meas, 4) if n_meas else None}
    weighted = [c for c in cats.values() if c["weight"] > 0]
    wsum = sum(c["weight"] for c in weighted) or 1.0
    scored = [c for c in weighted if c["score"] is not None]
    weighted_score = (round(sum(c["weight"] * c["score"] for c in scored) / sum(c["weight"] for c in scored), 4)
                      if scored else None)
    coverage = round(sum(c["weight"] * c["coverage"] for c in weighted) / wsum, 4)
    gates = evaluate_gates(scenario, by_id)
    statuses = [by_id[m.id]["status"] for m in applicable]
    if any(g["status"] == "FAIL" for g in gates) or "FAIL" in statuses:
        verdict = "FAIL"
    elif all(s == "PASS" for s in statuses) and all(g["status"] in ("PASS", "NOT_APPLICABLE") for g in gates):
        verdict = "PASS"
    else:
        verdict = "INCOMPLETE"
    return {"benchmark_version": BENCHMARK_VERSION, "scenario_id": scenario_id, "ground_truth_available": ground_truth_available,
            "verdict": verdict, "weighted_score": weighted_score, "coverage": coverage, "categories": cats, "gates": gates,
            "metric_results": [by_id[m.id] for m in applicable]}


def aggregate(scorecards: Iterable[dict]) -> dict:
    """Roll scenario scorecards up: worst gate status wins, category scores are pooled over measured metrics."""
    cards = list(scorecards)
    gate_status: Dict[str, str] = {}
    order = {"NOT_APPLICABLE": 0, "PASS": 1, "NOT_RUN": 2, "FAIL": 3}
    for c in cards:
        for g in c["gates"]:
            if order[g["status"]] >= order[gate_status.get(g["gate_id"], "NOT_APPLICABLE")]:
                gate_status[g["gate_id"]] = g["status"]
    cats: Dict[str, dict] = {}
    for c in cards:
        for cid, v in c["categories"].items():
            a = cats.setdefault(cid, {"name": v["name"], "weight": v["weight"], "metrics": 0, "measured": 0, "passed": 0})
            a["metrics"] += v["metrics"]
            a["measured"] += v["measured"]
            a["passed"] += v["passed"]
    for a in cats.values():
        a["coverage"] = round(a["measured"] / a["metrics"], 4) if a["metrics"] else 0.0
        a["score"] = round(a["passed"] / a["measured"], 4) if a["measured"] else None
    verdicts = [c["verdict"] for c in cards]
    verdict = "FAIL" if "FAIL" in verdicts else "PASS" if verdicts and all(v == "PASS" for v in verdicts) else "INCOMPLETE"
    return {"scenarios": len(cards), "verdict": verdict, "gates": gate_status, "categories": cats,
            "scenarios_by_verdict": {v: verdicts.count(v) for v in ("PASS", "FAIL", "INCOMPLETE")}}
