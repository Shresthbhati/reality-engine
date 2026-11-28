"""Tests for engine/multiengine/model_selection.py (WS4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from engine.multiengine.model_selection import (
    SelectionRequest, SelectionStatus, select_model, select_models,
)


@dataclass(frozen=True)
class _Rec:
    model_id: str
    task: str
    dataset: str = "heldout-v1"
    commit_sha: str = "abc123"
    image_ids: tuple[str, ...] = ("i1", "i2")
    status: str = "succeeded"
    wall_time_s: float = 1.0
    peak_vram_mb: float | None = 1000.0
    license: str = "Apache-2.0"
    per_class_metrics: dict = field(default_factory=dict)


def _acc(value, key="small_high"):
    return {"per_class_metrics": {key: {"f1": value}}}


def test_selects_highest_measured_accuracy():
    recs = [_Rec("a", "detect", **_acc(0.5)), _Rec("b", "detect", **_acc(0.9))]
    r = select_model(SelectionRequest("detect", object_scale="small", image_quality="high"), recs)
    assert r.status is SelectionStatus.SELECTED
    assert r.winner == "b"
    assert r.candidates[1].measured["accuracy"] == 0.9


def test_no_data_returns_no_data():
    r = select_model(SelectionRequest("detect"), [])
    assert r.status is SelectionStatus.NO_DATA and r.winner is None


def test_failed_candidate_never_wins_or_falls_back():
    recs = [
        _Rec("a", "detect", status="failed", **_acc(1.0)),
        _Rec("b", "detect", **_acc(0.4)),
    ]
    r = select_model(SelectionRequest("detect", object_scale="small", image_quality="high"), recs)
    assert r.winner == "b"
    assert any(c.model_id == "a" and not c.eligible for c in r.candidates)
    assert "no silent fallback" in r.candidates[0].reason


def test_incomparable_evidence_is_blocked_not_averaged():
    recs = [
        _Rec("a", "detect", **_acc(0.5)),
        _Rec("b", "detect", commit_sha="different", **_acc(0.9)),
    ]
    r = select_model(SelectionRequest("detect"), recs)
    assert r.status is SelectionStatus.BLOCKED and r.winner is None
    assert any("not comparable" in x for x in r.reasons)


def test_vram_hard_gate():
    recs = [_Rec("a", "detect", peak_vram_mb=5000.0, **_acc(0.9)),
            _Rec("b", "detect", peak_vram_mb=1000.0, **_acc(0.5))]
    r = select_model(SelectionRequest("detect", object_scale="small", image_quality="high",
                                      available_vram_mb=2000.0), recs)
    assert r.winner == "b"
    assert any("VRAM" in c.reason for c in r.candidates if c.model_id == "a")


def test_cpu_only_blocks_gpu_records():
    recs = [_Rec("a", "detect", peak_vram_mb=1000.0, **_acc(0.9)),
            _Rec("b", "detect", peak_vram_mb=0.0, **_acc(0.5))]
    r = select_model(SelectionRequest("detect", object_scale="small", image_quality="high",
                                      cpu_only=True), recs)
    assert r.winner == "b"


def test_commercial_licence_gate():
    recs = [_Rec("a", "detect", license="AGPL-3.0 non-commercial", **_acc(0.9)),
            _Rec("b", "detect", license="Apache-2.0", **_acc(0.5))]
    r = select_model(SelectionRequest("detect", object_scale="small", image_quality="high",
                                      require_commercial=True), recs)
    assert r.winner == "b"


def test_no_measured_metric_is_not_eligible():
    recs = [_Rec("a", "detect")]  # no per_class_metrics
    r = select_model(SelectionRequest("detect", object_scale="small", image_quality="high"), recs)
    assert r.status is SelectionStatus.BLOCKED
    assert "no measured accuracy" in r.candidates[0].reason


def test_different_tasks_win_different_models():
    recs = [
        _Rec("a", "detect", **_acc(0.9)),
        _Rec("b", "detect", **_acc(0.5)),
        _Rec("b", "depth", **_acc(0.8)),
        _Rec("a", "depth", **_acc(0.3)),
    ]
    results = select_models([
        SelectionRequest("detect", object_scale="small", image_quality="high"),
        SelectionRequest("depth", object_scale="small", image_quality="high"),
    ], recs)
    assert results[0].winner == "a"
    assert results[1].winner == "b"  # different model wins a different task


def test_scale_quality_specific_metric_selection():
    # model a wins small objects, model b wins large objects on the SAME split.
    recs = [
        _Rec("a", "detect", per_class_metrics={"small_high": {"f1": 0.9}, "large_high": {"f1": 0.3}}),
        _Rec("b", "detect", per_class_metrics={"small_high": {"f1": 0.4}, "large_high": {"f1": 0.8}}),
    ]
    small = select_model(SelectionRequest("detect", object_scale="small", image_quality="high"), recs)
    large = select_model(SelectionRequest("detect", object_scale="large", image_quality="high"), recs)
    assert small.winner == "a" and large.winner == "b"


def test_wall_time_is_a_measured_tiebreaker():
    recs = [
        _Rec("a", "detect", wall_time_s=9.0, **_acc(0.6)),
        _Rec("b", "detect", wall_time_s=0.5, **_acc(0.6)),
    ]
    r = select_model(SelectionRequest("detect", object_scale="small", image_quality="high",
                                      weight_accuracy=0.5, weight_speed=0.5, weight_memory=0.0), recs)
    assert r.winner == "b"  # equal accuracy, faster measured wall time wins


def test_max_wall_time_hard_gate():
    recs = [_Rec("a", "detect", wall_time_s=50.0, **_acc(0.9)),
            _Rec("b", "detect", wall_time_s=1.0, **_acc(0.5))]
    r = select_model(SelectionRequest("detect", object_scale="small", image_quality="high",
                                      max_wall_time_s=10.0), recs)
    assert r.winner == "b"


def test_to_dict_is_machine_readable():
    r = select_model(SelectionRequest("detect", object_scale="small", image_quality="high"),
                     [_Rec("a", "detect", **_acc(0.5))])
    d = r.to_dict()
    assert d["status"] == "selected" and d["winner"] == "a"
    assert d["request"]["weights"]["accuracy"] == 0.7
    assert d["candidates"][0]["measured"]["accuracy"] == 0.5
