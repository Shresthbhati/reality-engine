"""Scene-complete eval scorer (build-plan task 8.6).

Label and prediction files share one shape (one JSON per image in ``labels/`` and a
predictions dir of the same names)::

    {"image": "room01.jpg", "source": "ai" | "real" | "synthetic",
     "objects": [{"class": "sofa", "tier": "furniture",
                  "box": [x0, y0, x1, y1], "support": "floor"}]}

Matching is greedy by IoU; a prediction counts only if class and IoU >= ``iou`` agree.
Recall/precision are reported per tier and overall. Nothing is scored for an image with
no prediction file: it counts as zero detections (a missing run is a miss, not a skip).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TIERS = ("architecture", "infrastructure", "furniture", "appliance", "decor", "small")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")).get("objects", [])


def score(labels_dir: Path, preds_dir: Path, iou_min: float = 0.5) -> dict:
    tp, fn, fp = defaultdict(int), defaultdict(int), defaultdict(int)
    images = 0
    for lab in sorted(Path(labels_dir).glob("*.json")):
        images += 1
        gt = _load(lab)
        pf = Path(preds_dir) / lab.name
        preds = _load(pf) if pf.exists() else []
        used: set[int] = set()
        for g in gt:
            tier = g.get("tier", "furniture")
            best, best_i = 0.0, None
            for i, p in enumerate(preds):
                if i in used or p["class"] != g["class"]:
                    continue
                v = iou(g["box"], p["box"])
                if v > best:
                    best, best_i = v, i
            if best_i is not None and best >= iou_min:
                used.add(best_i)
                tp[tier] += 1
            else:
                fn[tier] += 1
        for i, p in enumerate(preds):
            if i not in used:
                fp[p.get("tier", "furniture")] += 1
    out = {"images": images, "iou": iou_min, "tiers": {}}
    for t in sorted(set(tp) | set(fn) | set(fp)):
        r = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else None
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else None
        out["tiers"][t] = {"tp": tp[t], "fn": fn[t], "fp": fp[t], "recall": r, "precision": p}
    T, N, P = sum(tp.values()), sum(fn.values()), sum(fp.values())
    out["overall"] = {"recall": T / (T + N) if T + N else None, "precision": T / (T + P) if T + P else None}
    return out


if __name__ == "__main__":
    import sys

    base = Path(__file__).parent
    print(json.dumps(score(base / "labels", Path(sys.argv[1] if len(sys.argv) > 1 else base / "predictions")), indent=2))
