"""Attributes: which kind of socket, lock, handle (BUILD_SPEC B05).

Decision logic is pure numpy so it is testable without a detector. A value is asserted only with a clear margin and a
raw score above the detector's box threshold; otherwise the attribute stays unknown (``value=None``) and every
candidate is kept in the record. Across views, evidence is summed in log space but capped at 3 views' worth (views
come from one model and are correlated).
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np

TEMPERATURE = 0.07
MARGIN = 0.20
MAX_VIEWS = 3


def _softmax(scores: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    z = (scores - scores.max()) / temperature
    e = np.exp(z)
    return e / e.sum()


def _decide(values: Sequence[str], p: np.ndarray, top_raw: float, box_threshold: float) -> Optional[str]:
    if len(values) == 0:
        return None
    order = np.argsort(-p)
    second = p[order[1]] if len(p) > 1 else 0.0
    if p[order[0]] - second >= MARGIN and top_raw >= box_threshold:
        return values[int(order[0])]
    return None


def classify_attribute(values: Sequence[str], scores: Sequence[float], box_threshold: float, obs_id: str = "") -> dict:
    """``values`` are the allowed values minus 'unknown'; ``scores`` the best detector score per value's phrase."""
    s = np.asarray(scores, dtype=float)
    if len(s) != len(values):
        raise ValueError("one score per value")
    p = _softmax(s) if len(s) else s
    return {"value": _decide(values, p, float(s.max()) if len(s) else 0.0, box_threshold),
            "p": float(p.max()) if len(p) else 0.0, "basis": "visual", "obs": [obs_id] if obs_id else [],
            "candidates": {v: float(x) for v, x in zip(values, p)}}


def classify_attribute_from_crop(image_rgb, box_xyxy, values: Sequence[str], phrases: Mapping[str, str],
                                 detector: Callable[[object, str], float], box_threshold: float, obs_id: str = "",
                                 pad: float = 0.15) -> dict:
    """Crop (padded 15 %), ask ``detector(crop, phrase) -> best score`` once per value, then decide."""
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    crop = image_rgb[max(int(y0 - py), 0):min(int(y1 + py), h), max(int(x0 - px), 0):min(int(x1 + px), w)]
    return classify_attribute(values, [detector(crop, phrases[v]) for v in values], box_threshold, obs_id)


def combine_views(values: Sequence[str], per_view: Sequence[dict], box_threshold: float = 0.0) -> dict:
    """Sum per-view log-probabilities (at most 3 views' worth), renormalise, apply the same margin rule.

    Conflicting views that do not produce a clear winner give ``value=None`` with every candidate kept."""
    if not per_view:
        return {"value": None, "p": 0.0, "basis": "visual", "obs": [], "candidates": {}}
    logp = np.array([[np.log(max(v["candidates"].get(name, 0.0), 1e-9)) for name in values] for v in per_view])
    k = min(len(per_view), MAX_VIEWS)
    total = logp.sum(axis=0) * (k / len(per_view))        # cap evidence at k views' worth
    p = _softmax(total, temperature=1.0)
    return {"value": _decide(values, p, max(box_threshold, 0.0), box_threshold), "p": float(p.max()), "basis": "visual",
            "obs": [o for v in per_view for o in v.get("obs", [])],
            "candidates": {v: float(x) for v, x in zip(values, p)}}
