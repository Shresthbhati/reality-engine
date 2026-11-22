"""Lossless mask <-> JSON codec for evidence observations (observations are JSON-canonical and cannot hold arrays).

Row-major run-length encoding of a boolean mask: ``runs`` alternate from ``first_value`` (0 = background, 1 = object). Holes are
preserved exactly because nothing is simplified; ``decode_mask(encode_mask(m)) == m`` for every boolean mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ENCODING = "row-major run-length, alternating from first_value (0 background, 1 object)"


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.size == 0:
        raise ValueError(f"a mask must be a non-empty 2-D array, got shape {m.shape}")
    flat = m.ravel(order="C")
    bounds = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [flat.size]))
    return {"shape": [int(m.shape[0]), int(m.shape[1])], "first_value": int(flat[0]), "runs": np.diff(bounds).astype(int).tolist(),
            "encoding": ENCODING, "area_px": int(flat.sum())}


def decode_mask(enc: Mapping[str, Any]) -> np.ndarray:
    h, w = (int(v) for v in enc["shape"])
    runs = [int(r) for r in enc["runs"]]
    if sum(runs) != h * w or any(r <= 0 for r in runs):
        raise ValueError("corrupt mask: run lengths do not cover the image exactly")
    out = np.empty(h * w, dtype=bool)
    pos, val = 0, bool(enc["first_value"])
    for r in runs:
        out[pos:pos + r] = val
        pos += r
        val = not val
    return out.reshape(h, w)


def count_holes(mask: np.ndarray) -> int:
    """Number of interior holes (cv2 RETR_CCOMP child contours); -1 when OpenCV is unavailable (unknown, never reported as 0)."""
    try:
        import cv2
    except ImportError:
        return -1
    _, hier = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return 0 if hier is None else int(np.sum(hier[0, :, 3] >= 0))
