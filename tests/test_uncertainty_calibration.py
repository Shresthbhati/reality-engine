"""Uncertainty calibration logic on SYNTHETIC errors with a known noise model: proves the procedure (scene-disjoint split, conformal coverage,
refusal on insufficient or non-independent data, per-class fallback), not the accuracy of any reconstruction."""

from __future__ import annotations

import numpy as np

from engine.multiengine.uncertainty_calibration import LEVELS, MIN_SCENES, ErrorRecord, _sig_of, calibrate, split_records

CLASSES = ("desk", "chair", "vent")


def _records(n_scenes, per_scene, seed=0, noise_depth=0.05, noise_extent=0.10, floor=0.02):
    """Reference errors whose true spread depends on depth and extent (heteroscedastic) and is much larger than the stated lower-bound sigma."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_scenes):
        for i in range(per_scene):
            depth = rng.uniform(1.0, 4.0)
            pred = rng.uniform(0.1, 2.0)
            true_sd = float(np.sqrt(floor ** 2 + (noise_depth * depth) ** 2 + (noise_extent * pred) ** 2))
            err = rng.standard_t(6) * true_sd * 0.8
            out.append(ErrorRecord(f"scene-{s}", f"scene-{s}/f{i % 6}", "rare" if i % 40 == 0 else CLASSES[i % 3], depth, pred, float(err), 0.005, bool(i % 5 == 0)))
    return out


def test_with_enough_independent_scenes_heldout_coverage_matches_nominal_and_the_result_is_calibrated():
    out = calibrate(_records(n_scenes=9, per_scene=120))
    assert out["independent"] and out["state"] == "calibrated" and out["supported"] and not out["reasons"]
    assert set(out["scenes_calibration"]).isdisjoint(out["scenes_held_out"])  # never fitted on the evaluation scenes
    for lv in LEVELS:  # measured on scenes the model never saw
        assert abs(out["held_out"][str(lv)]["coverage"] - lv) < 0.08, (lv, out["held_out"][str(lv)])
    # the stated lower-bound sigma alone is wildly over-confident on the same held-out data
    assert out["lower_bound_baseline_held_out"]["0.9"]["coverage"] < 0.4
    assert all(out["held_out"][str(lv)]["mean_width_m"] > 0 and "interval_score" in out["held_out"][str(lv)] for lv in LEVELS)  # sharpness reported
    curve = out["calibration_curve_held_out"]
    assert all(b["empirical"] >= a["empirical"] - 1e-9 for a, b in zip(curve, curve[1:])) and all(abs(c["nominal"] - c["empirical"]) < 0.12 for c in curve)


def test_the_model_is_conditional_not_a_single_multiplier():
    out = calibrate(_records(9, 120))
    coef = out["model"]["coef"]
    assert coef["c1_per_depth_m"] > 0.01 and coef["c2_per_extent"] > 0.03  # depth and extent both widen the interval
    c = np.array([coef["c0_floor_m"], coef["c1_per_depth_m"], coef["c2_per_extent"], coef["c3_truncated_per_extent"]])
    held = split_records(_records(9, 120))[1]
    near = next(r for r in held if r.depth_m < 1.3 and r.pred_m < 0.4)
    far = next(r for r in held if r.depth_m > 3.5 and r.pred_m > 1.5)
    assert _sig_of(c, far) > 3 * _sig_of(c, near)  # far / large detections get far wider intervals than near / small ones


def test_one_scene_gives_a_frame_split_that_is_reported_but_never_independent_and_stays_a_lower_bound():
    out = calibrate(_records(1, 600))
    assert out["independent"] is False and "NOT an independent" in out["split"]
    assert out["supported"] is False and out["state"] == "lower_bound_only"
    assert any("scene-independent" in r for r in out["reasons"]) and any("calibration scene" in r for r in out["reasons"])
    assert "held_out" in out  # the numbers are still reported, honestly labelled


def test_too_few_scenes_or_axes_are_refused_even_when_the_split_is_by_scene():
    out = calibrate(_records(MIN_SCENES, 30))
    assert out["supported"] is False and out["state"] == "lower_bound_only" and out["reasons"]
    tiny = calibrate(_records(2, 5))
    assert tiny["supported"] is False and tiny["state"] == "lower_bound_only"


def test_a_rare_class_falls_back_to_the_pooled_quantile_and_it_is_recorded():
    out = calibrate(_records(9, 120))
    assert set(out["model"]["class_specific"]) <= set(CLASSES)  # "rare" has < MIN_STRATUM calibration samples: no class-specific quantile
    assert "rare" in out["model"]["class_fallback_to_pooled"]
    assert all("coverage_90" in c and "n_calibration" in c for c in out["per_class_held_out"]) and out["failure_cases"]


def test_invalid_records_are_dropped_and_a_biased_heldout_scene_set_does_not_pass_as_calibrated():
    recs = _records(9, 120)
    bad = recs + [ErrorRecord("scene-x", "f", "desk", 2.0, 1.0, float("nan"), 0.01, False), ErrorRecord("scene-x", "f", "desk", 2.0, 1.0, 0.1, 0.0, False)]
    assert calibrate(bad)["n_valid"] == len(recs)
    cal_scenes = _records(8, 120, seed=1)
    held = [ErrorRecord(f"biased-{r.scene_id}", r.frame_id, r.cls, r.depth_m, r.pred_m, r.err_m + 0.6, r.sigma_lb_m, r.truncated) for r in _records(3, 120, seed=2)]
    out = calibrate(cal_scenes + held)
    held_biased = [s for s in out["scenes_held_out"] if s.startswith("biased-")]
    if held_biased and not [s for s in out["scenes_calibration"] if s.startswith("biased-")]:
        assert out["state"] == "lower_bound_only" and any("coverage" in r for r in out["reasons"])
