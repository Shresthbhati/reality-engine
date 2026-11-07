"""12.2: the measured-room accuracy report. It compares an engine world against tape/laser measurements, element by element,
against the plan's tolerances, and says plainly when the engine's own stated uncertainty was too optimistic."""
import pytest

from benchmarks.measured_rooms.report import TOLERANCES, evaluate, format_markdown

TRUTH = {
    "schema": "measured_room/1", "room": "living room", "method": "laser", "ceiling_height_m": 2.55, "floor_area_m2": 12.0,
    "walls": [{"id": "w1", "length_m": 4.00}, {"id": "w2", "length_m": 3.00}, {"id": "w3", "length_m": 4.00}, {"id": "w4", "length_m": 3.00}],
    "openings": [{"kind": "window", "width_m": 1.20, "height_m": 1.40, "sill_m": 0.90}, {"kind": "door", "width_m": 0.90, "height_m": 2.05, "sill_m": 0.0}],
}


def _dim(v, unc=None, state="ESTIMATED", unit="m"):
    return {"value": v, "unit": unit, "state": state, "basis": "test", "uncertainty_m": unc}


def _entity(eid, etype, dims, **cp):
    return {"id": eid, "type": etype, "name": eid, "custom_properties": {"dimensions": dims, **cp}}


def _world(walls=(4.01, 2.90), area=11.5, height=2.50, win_w=1.18, unc=0.05, unit="m"):
    ents = {f"wall{i}": _entity(f"wall{i}", "wall", {"length_m": _dim(L, unc, unit=unit)}) for i, L in enumerate(walls)}
    ents["room"] = _entity("room", "room", {"floor_area_m2": _dim(area, 0.5, unit="m2" if unit == "m" else "model_units2"),
                                           "height_m": _dim(height, unc, unit=unit)})
    ents["win"] = _entity("win", "window", {"width_m": _dim(win_w, unc, unit=unit), "height_m": _dim(1.35, unc, unit=unit),
                                           "sill_m": _dim(0.95, unc, unit=unit)})
    return {"id": "w", "entities": ents, "metadata": {"scale": {"state": "relative" if unit != "m" else "metric"}}}


def test_walls_are_matched_by_length_and_errors_are_signed():
    r = evaluate(TRUTH, _world())
    walls = {row["truth"]: row for row in r["rows"] if row["element"] == "wall length" and row["engine"] is not None}
    assert walls[4.0]["engine"] == pytest.approx(4.01) and walls[4.0]["error"] == pytest.approx(0.01)
    assert walls[3.0]["engine"] == pytest.approx(2.90) and walls[3.0]["error"] == pytest.approx(-0.10)


def test_pass_fail_uses_the_planned_tolerances():
    r = evaluate(TRUTH, _world())
    by = {(row["element"], row["truth"]): row for row in r["rows"] if row["engine"] is not None}
    assert by[("wall length", 4.0)]["pass"] is True  # 1 cm <= 2 cm
    assert by[("wall length", 3.0)]["pass"] is False  # 10 cm
    assert by[("room area", 12.0)]["pass"] is False  # 4 % > 1 %
    assert by[("ceiling height", 2.55)]["pass"] is False  # 5 cm > 2 cm
    assert TOLERANCES["wall_length_m"] == 0.02 and TOLERANCES["room_area_rel"] == 0.01


def test_walls_the_engine_never_saw_are_reported_as_missed_not_skipped():
    r = evaluate(TRUTH, _world())
    assert r["summary"]["walls_missed"] == 2 and r["summary"]["walls_extra"] == 0
    assert sum(1 for row in r["rows"] if row["element"] == "wall length" and row["engine"] is None) == 2


def test_the_engines_own_uncertainty_is_checked_against_the_truth():
    r = evaluate(TRUTH, _world(unc=0.05))
    by = {(row["element"], row["truth"]): row for row in r["rows"] if row["engine"] is not None}
    assert by[("wall length", 4.0)]["within_stated_uncertainty"] is True  # |1 cm| <= 5 cm
    assert by[("wall length", 3.0)]["within_stated_uncertainty"] is False  # |10 cm| > 5 cm: the engine was over-confident
    assert 0.0 < r["summary"]["uncertainty_honesty"] < 1.0


def test_model_units_are_never_silently_treated_as_metres():
    r = evaluate(TRUTH, _world(unit="model_units"))
    assert r["summary"]["scale_calibrated"] is False
    assert all("model units" in row["note"] for row in r["rows"] if row["engine"] is not None)


def test_calibration_separates_scale_error_from_shape_error():
    w = _world(walls=(4.4, 3.3), unit="model_units", area=14.5, height=2.805, win_w=1.32)  # everything 10 % too big, same shape
    raw = evaluate(TRUTH, w)
    cal = evaluate(TRUTH, w, calibrate=True)
    assert cal["summary"]["scale_factor"] == pytest.approx(1 / 1.1, abs=0.01)
    errs = [abs(row["error"]) for row in cal["rows"] if row["element"] == "wall length" and row["engine"] is not None]
    raw_errs = [abs(row["error"]) for row in raw["rows"] if row["element"] == "wall length" and row["engine"] is not None]
    assert max(errs) < 0.03 < max(raw_errs)  # after one scale factor the shape error is small


def test_the_markdown_table_states_pass_counts_and_the_scale_caveat():
    md = format_markdown(evaluate(TRUTH, _world(unit="model_units")))
    assert "wall length" in md and ("PASS" in md or "FAIL" in md)
    assert "model units" in md and "tolerance" in md.lower()
