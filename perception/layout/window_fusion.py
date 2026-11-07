"""Second evidence source for shut windows and glass doors (task T-3.1): detector boxes fused into the segmenter's masks.

The surface segmenter sometimes labels a closed window (flat glass, blinds, a bright pane) as wall. A text-prompted detector box
around it is an independent opinion. Fusing never replaces the segmenter: a box where the segmenter already found the window is
left alone, only pixels currently labelled wall are converted, and a box that mostly covers other surfaces is refused.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

MIN_SCORE = 0.4
ALREADY_FOUND_SHARE = 0.25  # the segmenter's own mask covers this much of the box: it saw the window, so do nothing
MIN_WALL_SHARE = 0.6  # a box must be mostly wall-labelled to be a window the segmenter missed
INSET = 0.08  # shrink the box on every side: boxes include the frame and a margin of wall

Box = Tuple[float, float, float, float]


def fuse_window_boxes(masks: Dict[str, np.ndarray], boxes: Sequence[Tuple[str, float, Box]]
                      ) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """Returns ``(new masks, fused)``; ``fused`` lists ``{kind, score, box, pixels}`` for each box that changed the masks."""
    out = {k: v.copy() for k, v in masks.items()}
    h, w = out["wall"].shape
    fused: List[dict] = []
    for kind, score, (x0, y0, x1, y1) in sorted(boxes, key=lambda b: -b[1]):
        if kind not in ("window", "door") or score < MIN_SCORE:
            continue
        dx, dy = (x1 - x0) * INSET, (y1 - y0) * INSET
        a, b, c, d = (int(round(v)) for v in (max(0, x0 + dx), max(0, y0 + dy), min(w, x1 - dx), min(h, y1 - dy)))
        if c - a < 4 or d - b < 4:
            continue
        area = float((c - a) * (d - b))
        if out[kind][b:d, a:c].sum() / area >= ALREADY_FOUND_SHARE:
            continue
        region = out["wall"][b:d, a:c].copy()
        if region.sum() / area < MIN_WALL_SHARE:
            continue
        out[kind][b:d, a:c] |= region
        out["wall"][b:d, a:c] &= ~region
        fused.append({"kind": kind, "score": round(float(score), 3), "box": [a, b, c, d], "pixels": int(region.sum())})
    return out, fused
