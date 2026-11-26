"""Confidence Calibration & Reliability Diagram Engine (A4.11).

Computes reliability bins, Expected Calibration Error (ECE), Maximum Calibration
Error (MCE), Brier score decomposition, and temperature-scaling calibration ONLY
when human-reviewed real ground truth is supplied.

Strictly refuses to mark any calibration model ``is_calibrated=True`` when
trained on synthetic or unlabelled data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float | None
    empirical_accuracy: float | None
    calibration_gap: float | None


@dataclass(frozen=True)
class CalibrationReport:
    dataset_name: str
    is_human_reviewed_real: bool
    is_calibrated: bool
    sample_count: int
    ece: float
    mce: float
    brier_score: float
    fitted_temperature: float | None
    bins: tuple[ReliabilityBin, ...]
    note: str


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    pc = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(pc / (1.0 - pc))


def fit_temperature_scaling(
    raw_scores: Sequence[float],
    outcomes: Sequence[int],
    *,
    is_human_reviewed_real: bool,
) -> float | None:
    """Fit scalar temperature T > 0 on logits to minimize NLL when real human GT exists."""
    if not is_human_reviewed_real or len(raw_scores) < 5 or len(raw_scores) != len(outcomes):
        return None
    logits = np.array([_logit(s) for s in raw_scores], dtype=np.float64)
    y = np.array([1.0 if int(v) > 0 else 0.0 for v in outcomes], dtype=np.float64)

    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.4, 3.0, 53):
        probs = np.array([_sigmoid(float(l / t)) for l in logits], dtype=np.float64)
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        nll = -float(np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return round(best_t, 4)


def apply_temperature(raw_score: float, temperature: float) -> float:
    """Apply fitted temperature scaling to a raw [0, 1] detector score."""
    t = max(1e-3, float(temperature))
    return round(_sigmoid(_logit(raw_score) / t), 6)


def compute_calibration_report(
    raw_scores: Sequence[float],
    outcomes: Sequence[int],
    *,
    dataset_name: str,
    is_human_reviewed_real: bool,
    n_bins: int = 10,
    fit_temperature: bool = True,
) -> CalibrationReport:
    """Compute reliability bins, ECE, MCE, Brier score, and optional temperature fit."""
    if len(raw_scores) != len(outcomes):
        raise ValueError("raw_scores and outcomes must have equal length")
    n = len(raw_scores)
    if n == 0:
        return CalibrationReport(
            dataset_name=dataset_name,
            is_human_reviewed_real=is_human_reviewed_real,
            is_calibrated=False,
            sample_count=0,
            ece=0.0,
            mce=0.0,
            brier_score=0.0,
            fitted_temperature=None,
            bins=(),
            note="empty_sample_set",
        )

    scores = np.clip(np.asarray(raw_scores, dtype=np.float64), 0.0, 1.0)
    y = np.asarray([1.0 if int(v) > 0 else 0.0 for v in outcomes], dtype=np.float64)

    brier = float(np.mean((scores - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0

    for bi in range(n_bins):
        lo = float(edges[bi])
        hi = float(edges[bi + 1])
        if bi == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            bins.append(ReliabilityBin(bi, round(lo, 3), round(hi, 3), 0, None, None, None))
            continue
        m_conf = float(np.mean(scores[mask]))
        m_acc = float(np.mean(y[mask]))
        gap = abs(m_conf - m_acc)
        ece += (cnt / float(n)) * gap
        mce = max(mce, gap)
        bins.append(ReliabilityBin(
            bin_index=bi,
            lower_bound=round(lo, 3),
            upper_bound=round(hi, 3),
            count=cnt,
            mean_confidence=round(m_conf, 4),
            empirical_accuracy=round(m_acc, 4),
            calibration_gap=round(gap, 4),
        ))

    temp = (
        fit_temperature_scaling(raw_scores, outcomes, is_human_reviewed_real=is_human_reviewed_real)
        if fit_temperature
        else None
    )
    can_calibrate = bool(is_human_reviewed_real and temp is not None)
    note = (
        f"calibrated_on_human_reviewed_real(T={temp})"
        if can_calibrate
        else "uncalibrated: requires >=5 human_reviewed_real samples"
    )
    return CalibrationReport(
        dataset_name=dataset_name,
        is_human_reviewed_real=is_human_reviewed_real,
        is_calibrated=can_calibrate,
        sample_count=n,
        ece=round(ece, 4),
        mce=round(mce, 4),
        brier_score=round(brier, 4),
        fitted_temperature=temp,
        bins=tuple(bins),
        note=note,
    )


__all__ = [
    "CalibrationReport",
    "ReliabilityBin",
    "apply_temperature",
    "compute_calibration_report",
    "fit_temperature_scaling",
]
