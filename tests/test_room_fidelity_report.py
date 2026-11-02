"""The room-fidelity scorecard (plan T0.2 / prepares F9): metrics, "not available" honesty, and the regression gate."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from scripts import room_fidelity_report as rfr

BASELINE = Path(__file__).resolve().parents[1] / "docs" / "reconstruction" / "room_fidelity_baseline.json"


@lru_cache(maxsize=None)
def _report(colors: str = "albedo"):
    return rfr.build_report(colors=colors, voxel_m=0.08)


def test_delta_e76_reference_values():
    assert rfr.delta_e76((10, 20, 30), (10, 20, 30)) == 0.0
    assert rfr.delta_e76((0, 0, 0), (255, 255, 255)) == pytest.approx(100.0, abs=0.5)  # L* 0 -> 100
    assert rfr.delta_e76((255, 0, 0), (0, 255, 0)) > 100  # red vs green: far apart


def test_report_matches_committed_baseline_and_gate_passes():
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    rep = _report()
    assert base["config"] == rep["config"], "baseline was written with a different config"
    assert rfr.compare_to_baseline(rep, base) == []


def test_baseline_pins_the_measured_good_and_bad_numbers():
    """Facts measured on 2026-10-31, not aspirations: when a defect is fixed this test must be updated with the new
    baseline (python scripts/room_fidelity_report.py --write-baseline ...)."""
    m = _report()["metrics"]
    assert m["room_count_error"]["value"] == 0 and m["duplicate_room_ids"]["value"] == 0
    assert m["wall_count"]["value"] == 4 and m["wall_perimeter_coverage"]["value"] == pytest.approx(1.0)
    for k in ("room_length_error_pct", "room_width_error_pct", "room_height_error_pct"):
        assert m[k]["value"] < 0.5, k
    assert m["window_recall"]["value"] == 1.0 and m["furniture_recall"]["value"] == 1.0
    assert all(m[k]["value"] < 3.0 for k in m if k.startswith("deltaE76_"))
    assert m["door_recall"]["value"] == 1.0 and m["door_precision"]["value"] == 1.0
    assert m["footprint_area_error_pct"]["value"] < 3.0
    assert m["room_wall_color_rgb_present"]["value"] == 1.0
    # D1 done: every dimension now carries the five-key shape. D3: layout L/W from the regularised loop (not the
    # axis-aligned box) and the opposite-wall consistency verdict. Scale stays honestly unavailable at compile time.
    assert m["dimensions_with_state_pct"]["value"] == 100.0
    assert m["dimensional_consistency"]["value"] == 1.0
    assert m["layout_length_error_pct"]["value"] < 2.0 and m["layout_width_error_pct"]["value"] < 2.0
    assert m["scale_state"]["value"] is None
    # known gaps (see ROOM_FIDELITY_BASELINE.md "Defects found"); the same fixture at voxel 0.04 loses the door and
    # dilates the footprint by 16% (docs sweep), so these numbers are a statement about THIS density only.


def test_unavailable_metrics_are_never_numbers():
    rep = _report()
    assert rep["metrics"]["scale_state"]["value"] is None and rep["metrics"]["scale_state"]["reason"]
    # a world with no detected door: recall is 0 and the size error is NOT a number
    from types import SimpleNamespace
    from synthetic.indoor import fidelity_ground_truth, fidelity_room
    m = rfr.opening_metrics(SimpleNamespace(entities={}), fidelity_ground_truth(fidelity_room()))
    assert m["door_recall"]["value"] == 0.0 and m["door_width_error_cm"]["value"] is None
    assert m["door_width_error_cm"]["status"] == "not available" and m["door_width_error_cm"]["reason"]
    bare = _report("none")["metrics"]
    for k in bare:
        if k.startswith("deltaE76_"):
            assert bare[k]["value"] is None and bare[k]["status"] == "not available", k


def test_real_room_fixture_is_not_fabricated():
    rep = rfr.real_room_report()
    assert rep["metrics"] and all(m["value"] is None and m["reason"] for m in rep["metrics"].values())


def test_fail_on_regression_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rfr, "build_report", lambda *a, **k: _report())
    assert rfr.main(["--baseline", str(BASELINE), "--fail-on-regression"]) == 0
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    base["metrics"]["footprint_area_error_pct"]["value"] = 1.0   # pretend it used to be good -> now a regression
    worse = tmp_path / "base.json"
    worse.write_text(json.dumps(base), encoding="utf-8")
    assert rfr.main(["--baseline", str(worse), "--fail-on-regression"]) == 1
    assert "REGRESSION footprint_area_error_pct" in capsys.readouterr().out
    assert rfr.main(["--fail-on-regression"]) == 2                 # no baseline given
    base["config"]["colors"] = "rendered"
    worse.write_text(json.dumps(base), encoding="utf-8")
    assert rfr.main(["--baseline", str(worse), "--fail-on-regression"]) == 2   # config mismatch is not a pass


def test_a_metric_that_becomes_unavailable_is_a_regression():
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    rep = json.loads(json.dumps(_report()))
    rep["metrics"]["window_recall"] = rfr._na("test")
    assert any("window_recall" in p for p in rfr.compare_to_baseline(rep, base))
