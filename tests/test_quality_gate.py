"""Quality gate: the verdict follows measured facts, never "the pipeline finished"."""
from __future__ import annotations

from engine.multiengine.quality_gate import assess


def _ok(**over):
    r = {"stage_status": "complete",
         "world": {"status": "persisted", "entities": 3, "entities_with_position": 3, "entities_with_geometry": 3, "withheld_entities": []},
         "detection": {"status": "ran", "segmentation": "ran"},
         "scale": {"status": "physically_anchored", "physical_metric": True, "records_by_scale_state": {"metric_anchored": 3}}}
    r.update(over)
    return r


def test_ready_only_when_every_gate_measured_and_passing():
    assert assess(_ok())["verdict"] == "READY"


def test_unanchored_scale_is_partial_not_ready():
    r = _ok(scale={"status": "unanchored", "physical_metric": False, "records_by_scale_state": {"sfm_gauge": 3}})
    v = assess(r)
    assert v["verdict"] == "PARTIAL" and any("physical_scale" in x for x in v["reasons"])


def test_unmeasured_gate_never_counts_as_pass():
    v = assess(_ok(detection={"status": "not_run"}))
    assert v["verdict"] == "PARTIAL" and any("not measured" in x for x in v["reasons"])


def test_unpositioned_entities_make_it_partial_and_none_positioned_blocks():
    w = {"status": "persisted", "entities": 3, "entities_with_position": 1, "entities_with_geometry": 1, "withheld_entities": []}
    assert assess(_ok(world=w))["verdict"] == "PARTIAL"
    assert assess(_ok(world={**w, "entities_with_position": 0}))["verdict"] == "BLOCKED"


def test_blocked_for_empty_or_unpersisted_world():
    assert assess(_ok(world={"status": "persisted", "entities": 0}))["verdict"] == "BLOCKED"
    assert assess(_ok(world={"status": "not_persisted"}))["verdict"] == "BLOCKED"


def test_failed_for_unreadable_run_or_bad_integrity():
    assert assess({"stage_status": "unreadable", "reason": "snapshot unavailable"})["verdict"] == "FAILED"
    assert assess(_ok(world={"status": "failed_verification", "verification": ["hash mismatch"]}))["verdict"] == "FAILED"


def test_partial_stage_is_never_ready():
    assert assess(_ok(stage_status="partial"))["verdict"] == "PARTIAL"
