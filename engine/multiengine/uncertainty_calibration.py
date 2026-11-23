"""Empirical calibration of extent (dimension) uncertainty against independent reference measurements.

The lifter states a LOWER-BOUND sigma (sampling spread only). This module learns, from reference errors, how much larger the real error is and
under which conditions (depth range, extent, truncation, class), then states split-conformal intervals whose coverage is MEASURED on records the
model never saw. It never multiplies every sigma by one constant to hit a target coverage:

    sigma(x)^2 = sigma_lb^2 + c0^2 + (c1 * depth)^2 + (c2 * extent)^2 + (c3 * truncated * extent)^2          (c >= 0, fitted on the calibration split)

fitted by a robust (Student-t) likelihood on the FIT partition ONLY; then, for every level p, q_p is the finite-sample conformal quantile of
|error| / sigma(x) on a SEPARATE conformal partition (never the records the model was fitted on, whose residuals are optimistic), and the interval
is +- q_p * sigma(x). Coverage and sharpness are measured on a THIRD, untouched held-out partition. The three partitions are SCENE-disjoint. A class
gets its own q_p only with >= MIN_STRATUM samples in the conformal partition, otherwise the pooled q_p is used and the fallback is recorded.

``supported`` is True only when the evidence justifies calling the intervals calibrated: a SCENE-disjoint split (frames of one scene are
correlated, so a frame split is reported but never counts as independent), >= MIN_SCENES calibration scenes, enough axes on both sides, and held-out
coverage near nominal. Otherwise the stated sigma stays a lower bound and the quality gate stays NOT_MEASURED. Aleatoric and epistemic
uncertainty are NOT separated: that needs an ensemble or repeated measurements, which this data does not have.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

LEVELS = (0.5, 0.68, 0.9, 0.95)
MIN_SCENES_FIT = 3      # distinct scenes in each role before intervals may be called calibrated
MIN_SCENES_CONFORMAL = 2
MIN_SCENES_HELD = 2
MIN_SCENES = MIN_SCENES_FIT + MIN_SCENES_CONFORMAL + MIN_SCENES_HELD
MIN_AXES_FIT = 200
MIN_AXES_CONFORMAL = 100
MIN_AXES_HELD = 60
SEED = "dim-unc-v1"
MIN_STRATUM = 30
COVERAGE_TOL = 0.05     # |held-out coverage at 90% - 0.90| must be within this
NU = 4.0                # Student-t degrees of freedom for the robust fit


@dataclass(frozen=True)
class ErrorRecord:
    scene_id: str
    frame_id: str
    cls: str
    depth_m: float        # median depth of the detection's points
    pred_m: float         # the PREDICTED extent along this axis (available at inference time, unlike the reference)
    err_m: float          # predicted - reference
    sigma_lb_m: float     # the stated lower-bound sigma
    truncated: bool       # the box touches the image border (unseen extent)

    @staticmethod
    def from_row(r: Mapping[str, Any]) -> "ErrorRecord":
        return ErrorRecord(str(r["scene_id"]), str(r["frame_id"]), str(r.get("cls", "unknown")), float(r["depth_m"]),
                           float(r["extent_m"]) + float(r["err_m"]), float(r["err_m"]), float(r["sigma_m"]), bool(r.get("truncated", False)))


def _bucket(key: str, mod: int = 3) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % mod


def _digest(rs: Sequence[ErrorRecord]) -> str:
    keys = sorted(f"{r.scene_id}|{r.frame_id}|{r.cls}|{r.depth_m:.6f}|{r.pred_m:.6f}|{r.err_m:.6f}|{r.sigma_lb_m:.6f}|{int(r.truncated)}" for r in rs)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def split_records(records: Sequence[ErrorRecord], seed: str = SEED) -> tuple[list[ErrorRecord], list[ErrorRecord], list[ErrorRecord], bool, dict[str, Any]]:
    """Three-way split -> (fit, conformal, held_out, independent, manifest).

    By SCENE (independent) when there are enough scenes: scenes are ordered by sha256(seed + scene id); the last ~25 % become held-out, the next
    ~25 % conformal, the rest fit; each role needs at least its MIN_SCENES_* scenes, otherwise the split falls back to FRAMES of whatever scenes exist
    (reported, never independent). The manifest lists every scene's role, per-role counts and a sha256 of each role's records, plus its own hash, so
    the partition can be reproduced and audited."""
    scenes = sorted({r.scene_id for r in records}, key=lambda sc: hashlib.sha256((seed + "|" + sc).encode()).hexdigest())
    n = len(scenes)
    n_held = max(MIN_SCENES_HELD, round(0.25 * n))
    n_conf = max(MIN_SCENES_CONFORMAL, round(0.25 * n))
    n_fit = n - n_held - n_conf
    independent = n_fit >= MIN_SCENES_FIT
    role_of: dict[str, str] = {}
    if independent:
        role_of = {sc: ("fit" if k < n_fit else "conformal" if k < n_fit + n_conf else "held_out") for k, sc in enumerate(scenes)}

        def role(r: ErrorRecord) -> str:
            return role_of[r.scene_id]

        how = f"scene-disjoint: {n_fit} fit / {n_conf} conformal / {n_held} held-out scenes"
    else:
        def role(r: ErrorRecord) -> str:
            return ("fit", "fit", "conformal", "held_out")[_bucket(f"{seed}|{r.scene_id}|{r.frame_id}", 4)]

        how = f"frame split within {n} scene(s): the partitions share scenes, so this is NOT an independent evaluation"
    parts: dict[str, list[ErrorRecord]] = {"fit": [], "conformal": [], "held_out": []}
    for r in records:
        parts[role(r)].append(r)
    manifest: dict[str, Any] = {"seed": seed, "how": how, "independent": independent, "scene_roles": role_of,
                                "counts": {k: {"axes": len(v), "scenes": len({r.scene_id for r in v})} for k, v in parts.items()},
                                "record_sha256": {k: _digest(v) for k, v in parts.items()}}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return parts["fit"], parts["conformal"], parts["held_out"], independent, manifest


def _sigma(c: np.ndarray, sig_lb: np.ndarray, depth: np.ndarray, pred: np.ndarray, trunc: np.ndarray) -> np.ndarray:
    return np.sqrt(sig_lb ** 2 + c[0] ** 2 + (c[1] * depth) ** 2 + (c[2] * pred) ** 2 + (c[3] * trunc * pred) ** 2)


def _arrays(rs: Sequence[ErrorRecord]) -> tuple[np.ndarray, ...]:
    return (np.array([r.err_m for r in rs]), np.array([r.sigma_lb_m for r in rs]), np.array([r.depth_m for r in rs]),
            np.array([r.pred_m for r in rs]), np.array([1.0 if r.truncated else 0.0 for r in rs]))


def _sig_of(c: np.ndarray, r: ErrorRecord) -> float:
    return float(_sigma(c, np.array([r.sigma_lb_m]), np.array([r.depth_m]), np.array([r.pred_m]), np.array([1.0 if r.truncated else 0.0]))[0])


def fit_model(cal: Sequence[ErrorRecord]) -> np.ndarray:
    e, sl, d, p, t = _arrays(cal)

    def nll(c: np.ndarray) -> float:
        s = _sigma(np.abs(c), sl, d, p, t)
        return float(np.sum(np.log(s) + 0.5 * (NU + 1) * np.log1p(e ** 2 / (NU * s ** 2))))

    best = None
    for x0 in ([0.01, 0.01, 0.05, 0.0], [0.05, 0.02, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0]):
        r = minimize(nll, np.array(x0), method="L-BFGS-B", bounds=[(0, 1.0)] * 4)
        if best is None or r.fun < best.fun:
            best = r
    return np.abs(best.x)


def _conformal_q(z: np.ndarray, level: float) -> float:
    n = len(z)
    k = min(n, math.ceil((n + 1) * level))
    return float(np.sort(z)[k - 1])


@dataclass(frozen=True)
class DimensionUncertaintyModel:
    coef: tuple[float, float, float, float]
    q_pooled: dict[float, float]
    q_class: dict[str, dict[float, float]]
    n_calibration: int

    def half_width(self, r: ErrorRecord, level: float) -> float:
        q = self.q_class.get(r.cls, self.q_pooled).get(level, self.q_pooled[level])
        return q * _sig_of(np.array(self.coef), r)


def calibrate(records: Sequence[ErrorRecord]) -> dict[str, Any]:
    """Fit on FIT, conformal quantiles on CONFORMAL, evaluate on HELD-OUT; decide whether the result may be called calibrated."""
    recs = [r for r in records if math.isfinite(r.err_m) and math.isfinite(r.sigma_lb_m) and r.sigma_lb_m > 0]
    fit, conf, held, independent, manifest = split_records(recs)
    out: dict[str, Any] = {"split": manifest["how"], "independent": independent, "split_manifest": manifest, "n_total": len(records), "n_valid": len(recs),
                           "n_fit": len(fit), "n_conformal": len(conf), "n_held_out": len(held), "levels": list(LEVELS),
                           "scenes_fit": sorted({r.scene_id for r in fit}), "scenes_conformal": sorted({r.scene_id for r in conf}),
                           "scenes_held_out": sorted({r.scene_id for r in held}), "aleatoric_epistemic_separated": False}
    if len(fit) < 20 or len(conf) < 5 or not held:
        out.update(supported=False, state="lower_bound_only",
                   reasons=[f"only {len(fit)} fit / {len(conf)} conformal / {len(held)} held-out axes: too few to fit anything"])
        return out
    coef = fit_model(fit)                                                       # 1. model fit: FIT partition only
    zc = np.array([abs(r.err_m) / _sig_of(coef, r) for r in conf])              # 2. conformal scores: CONFORMAL partition only
    pooled = {lv: _conformal_q(zc, lv) for lv in LEVELS}
    by_class: dict[str, dict[float, float]] = {}
    for c in sorted({r.cls for r in conf}):
        zz = np.array([abs(r.err_m) / _sig_of(coef, r) for r in conf if r.cls == c])
        if len(zz) >= MIN_STRATUM:
            by_class[c] = {lv: _conformal_q(zz, lv) for lv in LEVELS}
    model = DimensionUncertaintyModel(tuple(float(x) for x in coef), pooled, by_class, len(fit))
    out["model"] = {"coef": {"c0_floor_m": float(coef[0]), "c1_per_depth_m": float(coef[1]), "c2_per_extent": float(coef[2]),
                             "c3_truncated_per_extent": float(coef[3])},
                    "pooled_q": {str(k): v for k, v in pooled.items()}, "class_specific": sorted(by_class),
                    "class_fallback_to_pooled": sorted({r.cls for r in conf} - set(by_class))}

    def evaluate(rs: Sequence[ErrorRecord], half) -> dict[str, Any]:
        ev: dict[str, Any] = {}
        ae = np.array([abs(r.err_m) for r in rs])
        for lv in LEVELS:
            hw = np.array([half(r, lv) for r in rs])
            ev[str(lv)] = {"coverage": float(np.mean(ae <= hw)), "mean_width_m": float(np.mean(2 * hw)),
                           "interval_score": float(np.mean(2 * hw + (2 / (1 - lv)) * np.maximum(ae - hw, 0)))}
        return ev

    z_of = {0.5: 0.6745, 0.68: 1.0, 0.9: 1.645, 0.95: 1.96}  # Gaussian multipliers: what the stated sigma WOULD mean if it were calibrated
    out["held_out"] = evaluate(held, model.half_width)                          # 3. evaluation: HELD-OUT partition only
    out["lower_bound_baseline_held_out"] = evaluate(held, lambda r, lv: z_of[lv] * r.sigma_lb_m)
    out["fit_partition_in_sample"] = evaluate(fit, model.half_width)           # reported only to show how optimistic in-sample numbers are
    curve = []
    ae_h = np.array([abs(r.err_m) for r in held])
    sg_h = np.array([_sig_of(coef, r) for r in held])
    for nominal in np.linspace(0.1, 0.95, 18):
        q = _conformal_q(zc, float(nominal))
        curve.append({"nominal": round(float(nominal), 3), "empirical": float(np.mean(ae_h <= q * sg_h))})
    out["calibration_curve_held_out"] = curve
    per_class = []
    for c in sorted({r.cls for r in held}):
        rs = [r for r in held if r.cls == c]
        ev = evaluate(rs, model.half_width)
        per_class.append({"class": c, "n_held_out": len(rs), "n_conformal": sum(1 for r in conf if r.cls == c), "class_specific_q": c in by_class,
                          "coverage_90": ev["0.9"]["coverage"], "mean_width_90_m": ev["0.9"]["mean_width_m"]})
    out["per_class_held_out"] = per_class
    worst = sorted(held, key=lambda r: -abs(r.err_m) / max(model.half_width(r, 0.9), 1e-12))[:5]
    out["failure_cases"] = [{"scene": r.scene_id, "frame": r.frame_id, "class": r.cls, "err_m": r.err_m, "half_width_90_m": model.half_width(r, 0.9),
                             "depth_m": r.depth_m, "truncated": r.truncated} for r in worst]
    reasons = []
    if not independent:
        reasons.append("the partitions are not scene-disjoint")
    for role, parts, need_s, need_a in (("fit", fit, MIN_SCENES_FIT, MIN_AXES_FIT), ("conformal", conf, MIN_SCENES_CONFORMAL, MIN_AXES_CONFORMAL),
                                        ("held-out", held, MIN_SCENES_HELD, MIN_AXES_HELD)):
        ns = len({r.scene_id for r in parts})
        if ns < need_s:
            reasons.append(f"{ns} {role} scene(s); at least {need_s} are needed")
        if len(parts) < need_a:
            reasons.append(f"{len(parts)} {role} axes; at least {need_a} are needed")
    cov90 = out["held_out"]["0.9"]["coverage"]
    if abs(cov90 - 0.90) > COVERAGE_TOL:
        reasons.append(f"held-out coverage at 90% is {cov90:.2f}, outside 0.90 +- {COVERAGE_TOL}")
    out["supported"] = not reasons
    out["reasons"] = reasons
    out["state"] = "calibrated" if not reasons else "lower_bound_only"
    return out
