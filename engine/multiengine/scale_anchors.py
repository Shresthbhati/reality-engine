"""Physical scale anchoring: independent measurements -> a validated similarity transform, or an explicit refusal.

SfM / relative depth live in an arbitrary gauge. A physical scale is only ever *estimated from independent measurements* and is only
trusted after it predicts measurements it was NOT fitted to (held-out anchors). Nothing here applies a default scale factor.

Two anchor kinds:
  * ``DistanceAnchor`` -- a measured length between two reconstruction points (laser meter, tape, a surveyed wall distance, a reference
                          object of independently established size). Determines SCALE only (a distance carries no orientation).
  * ``PointAnchor``    -- a reconstruction point with an independently measured physical position (survey / external pose). Determines the
                          full similarity (scale, rotation, translation); needs >= 4 fit points so the fit can be checked.

Outcome (``ScaleCalibration.status``):
  validated    -- fitted on consistent anchors AND every held-out anchor is predicted within tolerance
  unvalidated  -- fitted on >= 2 mutually consistent anchors, but nothing was held out to test it
  insufficient -- too few (or degenerate) anchors; no applicable transform is produced
  failed       -- the anchors contradict each other, or a held-out anchor is not predicted; no transform is produced
The original (unscaled) reconstruction is never modified here: this module only returns the transform and its evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from engine.multiengine.alignment import DegenerateGeometry, Sim3, estimate_similarity

_TO_METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001}
REJECT_Z = 3.5      # robust outlier threshold on the standardised anchor residual
HELD_OUT_Z = 3.0    # a held-out anchor must be predicted within this many sigma


class AnchorError(ValueError):
    """An anchor record is malformed. Raised, never repaired: a guessed value would be a fabricated measurement."""


def _stamp(ts: str) -> str:
    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception as exc:  # noqa: BLE001
        raise AnchorError(f"measured_at {ts!r} is not an ISO-8601 timestamp") from exc
    return ts


def _vec(v: Sequence[float], what: str) -> np.ndarray:
    a = np.asarray(v, float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise AnchorError(f"{what} must be 3 finite numbers")
    return a


@dataclass(frozen=True)
class DistanceAnchor:
    anchor_id: str
    point_a: tuple[float, float, float]       # reconstruction (gauge) coordinates
    point_b: tuple[float, float, float]
    measured: float                            # the independently measured length, in ``unit``
    sigma: float                               # 1-sigma measurement uncertainty, in ``unit``
    unit: str                                  # "m" | "cm" | "mm"
    source: str                                # e.g. "laser_distance_meter", "tape", "reference_object:A4 sheet"
    measured_at: str                           # ISO-8601
    provenance: str                            # who/what measured it, and how the endpoints were identified in the reconstruction
    endpoint_sigma_gauge: float = 0.0          # 1-sigma localisation uncertainty of EACH endpoint in the reconstruction (gauge units)
    held_out: bool = False                     # True: never used in the fit, only to test it

    def __post_init__(self) -> None:
        if self.unit not in _TO_METRES:
            raise AnchorError(f"unit {self.unit!r} unknown; use one of {sorted(_TO_METRES)}")
        if not (math.isfinite(self.measured) and self.measured > 0):
            raise AnchorError("measured length must be a positive finite number")
        if not (math.isfinite(self.sigma) and self.sigma > 0):
            raise AnchorError("sigma must be positive: a measurement with no stated uncertainty is not an anchor")
        if not (self.source and self.provenance):
            raise AnchorError("source and provenance are required")
        _stamp(self.measured_at)
        a, b = _vec(self.point_a, "point_a"), _vec(self.point_b, "point_b")
        if np.linalg.norm(a - b) <= 0 or self.endpoint_sigma_gauge < 0:
            raise AnchorError("anchor endpoints coincide (or negative endpoint sigma)")

    @property
    def gauge_length(self) -> float:
        return float(np.linalg.norm(_vec(self.point_a, "a") - _vec(self.point_b, "b")))

    @property
    def metres(self) -> float:
        return self.measured * _TO_METRES[self.unit]

    @property
    def sigma_m(self) -> float:
        return self.sigma * _TO_METRES[self.unit]


@dataclass(frozen=True)
class PointAnchor:
    anchor_id: str
    recon_xyz: tuple[float, float, float]
    physical_xyz: tuple[float, float, float]   # metres, in the physical frame
    sigma_m: float
    source: str
    measured_at: str
    provenance: str
    held_out: bool = False

    def __post_init__(self) -> None:
        _vec(self.recon_xyz, "recon_xyz"), _vec(self.physical_xyz, "physical_xyz")
        if not (math.isfinite(self.sigma_m) and self.sigma_m > 0):
            raise AnchorError("sigma_m must be positive")
        if not (self.source and self.provenance):
            raise AnchorError("source and provenance are required")
        _stamp(self.measured_at)


@dataclass
class ScaleCalibration:
    status: str                                 # validated | unvalidated | insufficient | failed
    reason: str
    kind: str                                   # "distance" | "points"
    scale_m_per_gauge: float | None = None
    scale_sigma: float | None = None
    transform: Sim3 | None = None               # points only: recon -> physical (scale, rotation, translation)
    fit_ids: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    residuals: list[dict[str, Any]] = field(default_factory=list)       # fit anchors, with location for spatial inspection
    held_out: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def applicable(self) -> bool:
        return self.status in ("validated", "unvalidated") and self.scale_m_per_gauge is not None

    def to_dict(self) -> dict[str, Any]:
        t = self.transform
        return {"status": self.status, "reason": self.reason, "kind": self.kind, "scale_m_per_gauge": self.scale_m_per_gauge,
                "scale_sigma": self.scale_sigma, "applicable": self.applicable,
                "transform": None if t is None else {"scale": t.scale, "rotation": [list(r) for r in t.rotation],
                                                     "translation": list(t.translation)},
                "fit_ids": self.fit_ids, "rejected": self.rejected, "residuals": self.residuals, "held_out": self.held_out,
                "warnings": self.warnings,
                "note": "estimated from the listed independent measurements; the unscaled reconstruction is preserved"}


# ----------------------------------------------------------------------------------------------------------------- distances
def _implied(a: DistanceAnchor) -> tuple[float, float]:
    """(implied scale m/gauge, its 1-sigma): measurement and endpoint-localisation uncertainty combined in quadrature."""
    d = a.gauge_length
    rel = math.hypot(a.sigma_m / a.metres, math.sqrt(2.0) * a.endpoint_sigma_gauge / d)
    s = a.metres / d
    return s, s * rel


def calibrate_distances(anchors: Sequence[DistanceAnchor]) -> ScaleCalibration:
    ids = [a.anchor_id for a in anchors]
    if len(set(ids)) != len(ids):
        raise AnchorError("anchor ids must be unique")
    fit = [a for a in anchors if not a.held_out]
    hold = [a for a in anchors if a.held_out]
    cal = ScaleCalibration(status="insufficient", reason="", kind="distance")
    rejected: list[dict[str, Any]] = []
    cur = list(fit)
    while len(cur) >= 2:
        imp = [_implied(a) for a in cur]
        s_i = np.array([i[0] for i in imp])
        sig_i = np.array([i[1] for i in imp])
        w = 1.0 / sig_i ** 2
        s = float((w * s_i).sum() / w.sum())
        z = (s_i - s) / sig_i
        worst = int(np.argmax(np.abs(z)))
        if abs(z[worst]) <= REJECT_Z:
            break
        if len(cur) == 2:  # two anchors that disagree: which one is wrong cannot be known
            cal.status, cal.reason = "failed", (
                f"the two anchors disagree ({cur[0].anchor_id}: {s_i[0]:.6g}, {cur[1].anchor_id}: {s_i[1]:.6g} m/gauge, "
                f"{abs(z[worst]):.1f} sigma from their mean) and there is no third to decide which is wrong")
            cal.fit_ids, cal.rejected = [a.anchor_id for a in cur], rejected
            return cal
        rejected.append({"anchor_id": cur[worst].anchor_id, "implied_scale": float(s_i[worst]), "z": float(z[worst]),
                         "reason": "inconsistent with the other anchors"})
        cur = [a for k, a in enumerate(cur) if k != worst]
    cal.rejected = rejected
    if len(cur) < 2:
        cal.reason = (f"{len(cur)} usable fit anchor(s) after consistency checking; at least 2 mutually consistent anchors "
                      "are required (a single measurement cannot be checked against anything)")
        if len(cur) == 1:  # reported as evidence only: status stays insufficient, so it is never applied
            cal.scale_m_per_gauge, cal.scale_sigma = _implied(cur[0])
            cal.fit_ids = [cur[0].anchor_id]
        return cal
    imp = [_implied(a) for a in cur]
    s_i = np.array([i[0] for i in imp])
    sig_i = np.array([i[1] for i in imp])
    w = 1.0 / sig_i ** 2
    s = float((w * s_i).sum() / w.sum())
    chi2 = float((w * (s_i - s) ** 2).sum())
    birge = max(1.0, math.sqrt(chi2 / (len(cur) - 1)))
    sigma_s = float(birge / math.sqrt(w.sum()))
    cal.scale_m_per_gauge, cal.scale_sigma, cal.fit_ids = s, sigma_s, [a.anchor_id for a in cur]
    cal.residuals = [{"anchor_id": a.anchor_id, "gauge_length": a.gauge_length, "measured_m": a.metres,
                      "predicted_m": s * a.gauge_length, "error_m": s * a.gauge_length - a.metres,
                      "location_gauge": ((_vec(a.point_a, "a") + _vec(a.point_b, "b")) / 2).tolist()} for a in cur]
    dirs = np.array([(_vec(a.point_b, "b") - _vec(a.point_a, "a")) / a.gauge_length for a in cur])
    spread = float(np.degrees(np.max(np.arccos(np.clip(np.abs(dirs @ dirs.T), 0, 1)))))
    if spread < 20.0:
        cal.warnings.append(f"all fit anchors lie within {spread:.0f} degrees of one direction: a direction-dependent scale error "
                            "(anisotropy) cannot be detected from them")
    if birge > 1.0:
        cal.warnings.append(f"anchors scatter more than their stated uncertainty (Birge ratio {birge:.2f}); scale sigma was inflated")
    bad = False
    for a in hold:
        d = a.gauge_length
        pred = s * d
        sig = math.sqrt(a.sigma_m ** 2 + (d * sigma_s) ** 2 + (s * math.sqrt(2.0) * a.endpoint_sigma_gauge) ** 2)
        err = pred - a.metres
        zz = err / sig
        cal.held_out.append({"anchor_id": a.anchor_id, "measured_m": a.metres, "predicted_m": pred, "error_m": err,
                             "relative_error": err / a.metres, "predicted_sigma_m": sig, "z": zz, "within_3_sigma": abs(zz) <= HELD_OUT_Z})
        bad |= abs(zz) > HELD_OUT_Z
    if bad:
        cal.status, cal.reason = "failed", "a held-out anchor is not predicted within 3 sigma: the fitted scale does not generalise"
        cal.scale_m_per_gauge = cal.scale_sigma = None
    elif hold:
        cal.status, cal.reason = "validated", f"{len(cur)} consistent fit anchors; {len(hold)} held-out anchor(s) predicted within 3 sigma"
    else:
        cal.status, cal.reason = "unvalidated", f"{len(cur)} consistent fit anchors but none held out: the scale is not independently tested"
    return cal


# ------------------------------------------------------------------------------------------------------------------- points
def calibrate_points(anchors: Sequence[PointAnchor]) -> ScaleCalibration:
    ids = [a.anchor_id for a in anchors]
    if len(set(ids)) != len(ids):
        raise AnchorError("anchor ids must be unique")
    fit = [a for a in anchors if not a.held_out]
    hold = [a for a in anchors if a.held_out]
    cal = ScaleCalibration(status="insufficient", reason="", kind="points")
    rejected: list[dict[str, Any]] = []
    cur = list(fit)

    def solve(sel: Sequence[PointAnchor]) -> Sim3:
        src = np.array([a.recon_xyz for a in sel], float)
        dst = np.array([a.physical_xyz for a in sel], float)
        return estimate_similarity(src, dst, weights=1.0 / np.array([a.sigma_m for a in sel]) ** 2)

    def err(t: Sim3, a: PointAnchor) -> float:
        return float(np.linalg.norm(t.apply(np.asarray(a.recon_xyz))[0] - np.asarray(a.physical_xyz)))

    while len(cur) >= 5:  # leave >= 4 after a rejection
        try:
            t = solve(cur)
        except DegenerateGeometry as exc:
            cal.reason = f"fit anchors are degenerate: {exc}"
            return cal
        res = np.array([err(t, a) for a in cur])
        med = float(np.median(res))
        floor = float(np.median([a.sigma_m for a in cur]))  # residuals at the stated measurement noise are not outliers
        mad = max(float(np.median(np.abs(res - med))) / 0.6745, floor)
        z = (res - med) / mad
        worst = int(np.argmax(z))
        if z[worst] <= REJECT_Z:
            break
        rejected.append({"anchor_id": cur[worst].anchor_id, "residual_m": float(res[worst]), "robust_z": float(z[worst]),
                         "reason": "residual far outside the other anchors'"})
        cur = [a for k, a in enumerate(cur) if k != worst]
    cal.rejected = rejected
    if len(cur) < 4:
        cal.reason = f"{len(cur)} usable point anchors; at least 4 are required to fit a similarity AND check it"
        return cal
    try:
        t = solve(cur)
    except DegenerateGeometry as exc:
        cal.reason = f"fit anchors are degenerate: {exc}"
        return cal
    res = [err(t, a) for a in cur]
    loo = []
    for k in range(len(cur)):  # leave-one-out scale: its spread is the scale's own uncertainty (jackknife)
        try:
            loo.append(solve([a for j, a in enumerate(cur) if j != k]).scale)
        except DegenerateGeometry:
            pass
    n = len(loo)
    sigma_s = float(math.sqrt((n - 1) / n * np.sum((np.array(loo) - np.mean(loo)) ** 2))) if n > 2 else None
    cal.transform, cal.scale_m_per_gauge, cal.scale_sigma, cal.fit_ids = t, float(t.scale), sigma_s, [a.anchor_id for a in cur]
    cal.residuals = [{"anchor_id": a.anchor_id, "residual_m": r, "location_gauge": list(a.recon_xyz)} for a, r in zip(cur, res)]
    fit_rms = float(math.sqrt(np.mean(np.square(res))))
    bad = False
    for a in hold:
        e = err(t, a)
        tol = HELD_OUT_Z * math.hypot(fit_rms, a.sigma_m)  # what the fit itself could not explain, plus the anchor's own sigma
        cal.held_out.append({"anchor_id": a.anchor_id, "error_m": e, "tolerance_m": tol, "within_tolerance": e <= tol})
        bad |= e > tol
    if bad:
        cal.status, cal.reason = "failed", "a held-out point is not predicted within the fit's own residual scale: the transform does not generalise"
        cal.scale_m_per_gauge = cal.scale_sigma = cal.transform = None
    elif hold:
        cal.status, cal.reason = "validated", f"{len(cur)} fit points (RMS {fit_rms:.4f} m); {len(hold)} held-out point(s) predicted"
    else:
        cal.status, cal.reason = "unvalidated", f"{len(cur)} fit points (RMS {fit_rms:.4f} m) but none held out"
    return cal
