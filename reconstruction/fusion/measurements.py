"""Fuse repeated measurements of one quantity (BUILD_SPEC B14).

Robust centre (median/MAD) -> reject outliers -> inverse-variance mean, where measurements that share a
``group`` (same model or same photo) are correlated and count together as ONE measurement.
Every shown/exported dimension should come from ``fuse`` so it carries what was used and rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FusedMeasurement:
    value: float
    sigma: float
    used: tuple[int, ...]       # indices into the input arrays
    rejected: tuple[int, ...]
    residuals: tuple[float, ...]  # value - fused for the used inputs


def fuse(values: Sequence[float], sigmas: Sequence[float], group: Optional[Sequence] = None) -> FusedMeasurement:
    x = np.asarray(values, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    if x.ndim != 1 or x.shape != s.shape or x.size == 0:
        raise ValueError("values and sigmas must be equal-length, non-empty 1-D sequences")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("values must be finite and sigmas finite and > 0 (unknown sigma is not a number to invent)")
    med = float(np.median(x))
    scale = 1.4826 * float(np.median(np.abs(x - med)))
    keep = np.abs(x - med) <= 3.0 * np.maximum(scale, s)
    idx = np.flatnonzero(keep)
    w = 1.0 / (s[idx] ** 2)
    if group is not None:
        g = np.asarray([str(v) for v in group])
        if g.shape != x.shape:
            raise ValueError("group must match values")
        _, inv, counts = np.unique(g[idx], return_inverse=True, return_counts=True)
        w = w / counts[inv]                      # a group of n counts as one measurement
    est = float(np.sum(w * x[idx]) / np.sum(w))
    sigma = max(float(np.sqrt(1.0 / np.sum(w))), float(np.min(s[idx])) / 2.0)
    return FusedMeasurement(
        value=est, sigma=sigma, used=tuple(int(i) for i in idx),
        rejected=tuple(int(i) for i in np.flatnonzero(~keep)),
        residuals=tuple(float(v) for v in (x[idx] - est)),
    )
