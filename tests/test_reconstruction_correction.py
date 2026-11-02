"""Bounded correction loop: baseline preservation, acceptance, rejection, safety, budgets, failures, real data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult
from reconstruction.correction import ACCEPTED, FAILED, REJECTED, SKIPPED, Budget, correct
from reconstruction.diagnostics import WEAK_OBSERVATIONS
from world_ir.schema_v1 import Uncertainty

ROOT = Path(__file__).resolve().parents[1]


def _pose(eid, pos=(0.0, 0.0, 0.0)):
    return ReconstructedCameraPose(evidence_id=eid, position=pos, rotation=(1.0, 0.0, 0.0, 0.0))


def _pts(n, seen_by, start=0):
    return [ReconstructedPoint(position=(float(start + i), 0.0, 0.0), track_id=f"t{start + i}",
                               source_evidence_ids=list(seen_by), uncertainty=Uncertainty(confidence=0.9))
            for i in range(n)]


def _result():
    # a, b are well supported; w sees only 5 points (weak)
    points = _pts(200, ["a", "b"]) + _pts(5, ["b", "w"], start=200)
    return ReconstructionResult(points=points, camera_poses=[_pose("a"), _pose("b"), _pose("w")],
                                registration_status="partial")


IDS = ["a", "b", "w", "missing"]


def test_weak_camera_exclusion_is_accepted_and_baseline_is_untouched():
    base = _result()
    snapshot = (len(base.points), [p.evidence_id for p in base.camera_poses], [list(p.source_evidence_ids) for p in base.points])
    selected, rec = correct(base, IDS)

    assert rec["selected"] == "EXCLUDE_WEAK_CAMERAS" and selected is not base
    assert {p.evidence_id for p in selected.camera_poses} == {"a", "b"}
    # the baseline object is exactly as it was
    assert snapshot == (len(base.points), [p.evidence_id for p in base.camera_poses], [list(p.source_evidence_ids) for p in base.points])
    step = rec["attempts"][0]
    assert step["status"] == ACCEPTED and step["removed"] == ["w"]
    assert step["rationale"] and step["expected_effect"], "rationale and expected effect must be recorded"
    assert rec["baseline"]["summary"]["registered"] == 3 and step["candidate"]["summary"]["registered"] == 2
    assert json.dumps(rec), "the record must be JSON-serialisable"


def test_unregistered_photographs_become_an_evidence_request_not_a_fix():
    _, rec = correct(_result(), IDS)
    req = rec["evidence_requests"]
    assert len(req) == 1 and req[0]["images"] == ["missing"]
    assert all(a["diagnostic"] != "UNREGISTERED_IMAGE" for a in rec["attempts"])


def test_healthy_model_is_left_alone():
    base = ReconstructionResult(points=_pts(200, ["a", "b"]), camera_poses=[_pose("a"), _pose("b")], registration_status="success")
    selected, rec = correct(base, ["a", "b"])
    assert selected is base and rec["attempts"] == [] and rec["selected"] == "baseline"


def test_candidate_that_loses_too_many_points_is_rejected():
    # weak camera 'w' shares 20 of 120 points with 'a'; dropping w leaves them single-view -> 17% of points lost
    pts = _pts(20, ["a", "w"]) + _pts(100, ["a", "b"], start=20)
    base = ReconstructionResult(points=pts, camera_poses=[_pose("a"), _pose("b"), _pose("w")], registration_status="partial")
    selected, rec = correct(base, ["a", "b", "w"])
    step = rec["attempts"][0]
    assert selected is base and step["status"] == REJECTED
    assert any("points retained" in r for r in step["reasons"])


def test_candidate_that_regresses_the_reference_is_rejected(monkeypatch):
    calls = {"n": 0}

    def fake_eval(poses, gt_images, min_cameras=3):
        calls["n"] += 1
        worse = calls["n"] > 1  # first call = baseline, later calls = candidate
        return {"refused": False, "rotation_deg_median": 5.0 if worse else 0.2, "center_error_median": 1.0}

    import reconstruction.evaluation as ev
    monkeypatch.setattr(ev, "evaluate_absolute_poses", fake_eval)
    base = _result()
    selected, rec = correct(base, IDS, gt_images={"a": (1, 0, 0, 0, 0, 0, 0)})
    assert selected is base and rec["attempts"][0]["status"] == REJECTED
    assert any("rotation error regressed" in r for r in rec["attempts"][0]["reasons"])


def test_a_noop_candidate_is_not_accepted():
    _, rec = correct(_result(), IDS, actions={"EXCLUDE_WEAK_CAMERAS": lambda r: (r, [])})
    assert rec["attempts"][0]["status"] == REJECTED and rec["selected"] == "baseline"
    assert "no measured improvement" in " ".join(rec["attempts"][0]["reasons"])


def test_invalid_geometry_from_a_candidate_is_rejected():
    def corrupt(r):
        bad = ReconstructedPoint(position=(math.nan, 0, 0), track_id="x", source_evidence_ids=["a", "b"])
        return ReconstructionResult(points=r.points + [bad], camera_poses=[p for p in r.camera_poses if p.evidence_id != "w"],
                                    registration_status="partial"), ["w"]
    _, rec = correct(_result(), IDS, actions={"EXCLUDE_WEAK_CAMERAS": corrupt})
    step = rec["attempts"][0]
    assert step["status"] == REJECTED and any("non-finite" in r for r in step["reasons"])
    assert rec["selected"] == "baseline"


def test_action_that_raises_is_recorded_as_failed_and_never_propagates():
    def boom(_):
        raise RuntimeError("subprocess died")
    base = _result()
    selected, rec = correct(base, IDS, actions={"EXCLUDE_WEAK_CAMERAS": boom})
    assert selected is base and rec["attempts"][0]["status"] == FAILED
    assert "subprocess died" in rec["attempts"][0]["reasons"][0]


def test_unsafe_findings_are_never_corrected():
    # NON_FINITE_GEOMETRY is a backend fault: safe_to_auto_correct is False, so nothing may run
    bad = ReconstructedPoint(position=(math.nan, 0, 0), track_id="bad", source_evidence_ids=["a", "b"])
    base = ReconstructionResult(points=_pts(100, ["a", "b"]) + [bad], camera_poses=[_pose("a"), _pose("b")], registration_status="success")
    ran = []
    selected, rec = correct(base, ["a", "b"], actions={"EXCLUDE_WEAK_CAMERAS": lambda r: ran.append(1) or (r, [])})
    assert selected is base and not ran
    assert rec["attempts"] and all(a["status"] == SKIPPED for a in rec["attempts"])


def test_compute_budget_stops_before_the_action_runs():
    ticks = iter([0.0, 100.0, 100.0, 100.0])
    ran = []
    _, rec = correct(_result(), IDS, budget=Budget(max_attempts=2, max_seconds=10),
                     actions={"EXCLUDE_WEAK_CAMERAS": lambda r: ran.append(1) or (r, [])}, clock=lambda: next(ticks))
    assert not ran and rec["attempts"][0]["status"] == SKIPPED
    assert "compute budget exhausted" in rec["attempts"][0]["reasons"]


def test_attempt_budget_limits_candidates():
    ran = []
    _, rec = correct(_result(), IDS, budget=Budget(max_attempts=0),
                     actions={"EXCLUDE_WEAK_CAMERAS": lambda r: ran.append(1) or (r, [])})
    assert not ran and "attempt budget exhausted" in rec["attempts"][0]["reasons"]


def test_missing_dense_artifacts_are_not_invented():
    selected, _ = correct(_result(), IDS)
    assert selected.dense_points is None and selected.dense_report is None


def test_deterministic_output():
    a, b = correct(_result(), IDS)[1], correct(_result(), IDS)[1]
    for r in (a, b):
        for step in r["attempts"]:
            step.pop("seconds", None)
    assert a == b


SPARSE = ROOT / "datasets" / "south_building" / "sparse"


@pytest.mark.skipif(not (SPARSE / "images.txt").exists(), reason="committed South Building reference model not present")
def test_real_reference_model_is_not_degraded_by_the_loop():
    from reconstruction.backend.colmap_backend import _parse_images_txt, _parse_points3d_txt

    text = (SPARSE / "images.txt").read_text()
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    id_to_name = {lines[i].split()[0]: lines[i].split()[9] for i in range(0, len(lines), 2)}
    poses = _parse_images_txt(text, {})
    points = _parse_points3d_txt((SPARSE / "points3D.txt").read_text(), id_to_name)
    base = ReconstructionResult(points=points, camera_poses=poses, registration_status="success")

    selected, rec = correct(base, sorted(id_to_name.values()))
    # every camera in the reference is well supported (min > WEAK_OBSERVATIONS), so there is nothing safe to correct
    assert rec["baseline"]["summary"]["min_observations_per_camera"] > WEAK_OBSERVATIONS
    assert selected is base and rec["selected"] == "baseline"


FIXTURE = ROOT / "tests" / "fixtures" / "south_building_stride4_misregistered.json.gz"


def _real_misregistered():
    """The model the product path really produced on the committed photographs: 25 of 32 registered, of which three
    (P1180173/177/181) sit 104-129 deg wrong against the reference. Photograph names are the evidence ids."""
    import gzip

    from reconstruction.backend.colmap_backend import _parse_images_txt

    doc = json.loads(gzip.open(FIXTURE, "rt", encoding="utf8").read())
    rows = []
    for i, (name, *vals) in enumerate(doc["poses"], 1):
        rows += [f"{i} " + " ".join(repr(v) for v in vals) + f" 1 {name}", "0.0 0.0 -1"]  # second line = unused 2D points
    poses = _parse_images_txt("\n".join(rows), {})
    points = [ReconstructedPoint(position=tuple(xyz), track_id=f"s{i}", source_evidence_ids=list(names),
                                 uncertainty=Uncertainty(confidence=0.9))
              for i, (xyz, names) in enumerate(doc["tracks"])]
    return ReconstructionResult(points=points, camera_poses=poses, registration_status="partial"), doc["input_names"]


def _reference_poses():
    lines = [ln for ln in (SPARSE / "images.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return {lines[i].split()[9]: tuple(float(x) for x in lines[i].split()[1:8]) for i in range(0, len(lines), 2)}


@pytest.mark.skipif(not (FIXTURE.exists() and (SPARSE / "images.txt").exists()), reason="real-data fixture or reference missing")
def test_real_misregistration_is_detected_corrected_and_measurably_better():
    base, names = _real_misregistered()
    gt = _reference_poses()
    selected, rec = correct(base, names, gt_images=gt)

    wrong = {"P1180173.JPG", "P1180177.JPG", "P1180181.JPG"}
    d = rec["baseline"]["diagnostic_counts"]
    assert d.get("WEAKLY_BRIDGED_GROUP") == 1, d
    assert rec["baseline"]["summary"]["registered"] == 25 and len(rec["evidence_requests"][0]["images"]) == 7

    step = next(a for a in rec["attempts"] if a["action"] == "EXCLUDE_BRIDGED_GROUP")
    assert step["status"] == ACCEPTED and set(step["removed"]) == wrong
    assert {p.evidence_id for p in selected.camera_poses} == {p.evidence_id for p in base.camera_poses} - wrong

    before, after = rec["baseline"]["pose_evaluation"], step["candidate"]["pose_evaluation"]
    # measured on the real model, against the committed reference (units: degrees; scene units)
    assert before["rotation_deg_max"] > 100 and after["rotation_deg_max"] < 2.0, (before, after)
    assert after["center_error_max"] < before["center_error_max"] / 10
    assert after["common_cameras"] == 22


@pytest.mark.skipif(not FIXTURE.exists(), reason="real-data fixture missing")
def test_per_camera_support_alone_cannot_see_the_misregistered_group():
    from reconstruction.diagnostics import camera_support

    base, _ = _real_misregistered()
    sup = camera_support(base.points, base.camera_poses)
    for name in ("P1180173.JPG", "P1180177.JPG", "P1180181.JPG"):
        assert sup[name]["observations"] > WEAK_OBSERVATIONS, "if this changes, the bridge rule may be redundant"


def test_dense_points_do_not_inflate_camera_support():
    from reconstruction.diagnostics import analyze

    sparse = _pts(40, ["a", "b"])
    dense = [ReconstructedPoint(position=(0.0, 0.0, float(i)), track_id=f"dense_mvs:{i}", source_evidence_ids=["a", "b"])
             for i in range(1000)]
    rep = analyze(sparse + dense, [_pose("a"), _pose("b")], ["a", "b"])
    assert rep["summary"]["points"] == 40 and rep["summary"]["dense_points"] == 1000
    assert rep["summary"]["median_observations_per_camera"] == 40


def test_missing_or_error_severity_diagnostics_never_grade_as_clean():
    from apps.api.jobs import _stage_facts_degraded

    clean = {"reconstruction_diagnostics": {"summary": {}, "diagnostics": [{"type": "UNREGISTERED_IMAGE", "severity": "warning"}]}}
    assert _stage_facts_degraded(clean) == []
    assert _stage_facts_degraded({"reconstruction_diagnostics": None}) == []   # single-image fallback: nothing to analyse
    broken = _stage_facts_degraded({"reconstruction_diagnostics": {"error": "ValueError: boom"}})
    assert broken and "diagnostics unavailable" in broken[0]
    bridged = _stage_facts_degraded({"reconstruction_diagnostics": {"diagnostics": [{"type": "WEAKLY_BRIDGED_GROUP", "severity": "error"}]}})
    assert bridged and "WEAKLY_BRIDGED_GROUP" in bridged[0]
    nonfinite = _stage_facts_degraded({"reconstruction_diagnostics": {"diagnostics": [{"type": "NON_FINITE_GEOMETRY", "severity": "error"}]}})
    assert nonfinite and "NON_FINITE_GEOMETRY" in nonfinite[0]
    # a single thinly supported camera is error-severity in the diagnostics but has never changed the job grade
    weak = _stage_facts_degraded({"reconstruction_diagnostics": {"diagnostics": [{"type": "WEAKLY_SUPPORTED_CAMERA", "severity": "error"}]}})
    assert weak == []


def test_correction_failure_is_recorded_not_raised_by_the_pipeline_plan():
    # the plan wrapper in vertical_slice swallows everything; the loop itself must also never raise on odd input
    empty = ReconstructionResult(points=[], camera_poses=[], registration_status="failed")
    selected, rec = correct(empty, ["a", "b"])
    assert selected is empty and rec["selected"] == "baseline"
    assert [r["images"] for r in rec["evidence_requests"]] == [["a", "b"]]


def test_small_captures_do_not_produce_bridged_group_false_positives():
    """Found by the lifecycle tests: with every camera pair sharing fewer than BRIDGE_MIN_SHARED_POINTS, each lone
    camera used to be reported as its own 'group' and graded the job partial. A lone camera is not a group."""
    from reconstruction.diagnostics import analyze, bridged_groups

    points = _pts(5, ["a", "b"]) + _pts(5, ["b", "c"], start=5) + _pts(5, ["c", "d"], start=10)
    assert bridged_groups(points, ["a", "b", "c", "d"]) == []
    report = analyze(points, [_pose("a"), _pose("b"), _pose("c"), _pose("d")], ["a", "b", "c", "d"])
    assert not _types_of(report, "WEAKLY_BRIDGED_GROUP")

    # but a real pair that is consistent with each other and thinly tied to a larger body IS reported
    body = _pts(60, ["a", "b"]) + _pts(60, ["b", "c"], start=60) + _pts(60, ["a", "c"], start=120)
    pair = _pts(50, ["x", "y"], start=200) + _pts(3, ["c", "x"], start=300)
    groups = bridged_groups(body + pair, ["a", "b", "c", "x", "y"])
    assert [g["cameras"] for g in groups] == [["x", "y"]]
    assert groups[0]["internal_max_shared"] == 50 and groups[0]["to_rest_max_shared"] == 3


def _types_of(report, kind):
    return [d for d in report["diagnostics"] if d["type"] == kind]
