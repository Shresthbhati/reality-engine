"""Hierarchical detection: whole -> part -> sub-part (BUILD_SPEC B04).

Breadth-first over a ``deque`` so every whole is found before any part. Each node's children come from the
ontology ``components``; the detector (any object with ``detect(image_rgb, queries)`` returning items with
``label``, ``score``, ``box``) runs on the parent's crop only. Boxes are always stored in full-image pixels.
Parts outside their parent are dropped; detector calls are bounded by ``budget_calls``.
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from perception.ontology import Ontology
from world_ir.observation_table import ObservationTable

NMS_IOU = 0.5
PROMPTS_PER_PASS = 2


def _area(b: np.ndarray) -> np.ndarray:
    return (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_min: float = NMS_IOU) -> list[int]:
    order = np.argsort(-scores, kind="stable"); keep: list[int] = []
    while len(order):
        i = int(order[0]); keep.append(i)
        r = order[1:]
        if len(r) == 0:
            break
        iw = np.minimum(boxes[i, 2], boxes[r, 2]) - np.maximum(boxes[i, 0], boxes[r, 0])
        ih = np.minimum(boxes[i, 3], boxes[r, 3]) - np.maximum(boxes[i, 1], boxes[r, 1])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        iou = inter / np.maximum(_area(boxes[i]) + _area(boxes[r]) - inter, 1e-9)
        order = r[iou < iou_min]
    return keep


def _child_queries(ontology: Ontology, path: str) -> dict[str, tuple[str, str]]:
    """query phrase -> (child path, label). A component with its own ontology node uses that node's prompts."""
    out: dict[str, tuple[str, str]] = {}
    node = ontology.get(path)
    for comp in (node.components if node else ()):
        child = ontology.get(f"{path}/{comp}") or ontology.get(comp)
        cpath = child.path if child else f"{path}/{comp}"
        for ph in (child.prompts if child and child.prompts else (comp,)):
            out[ph] = (cpath, comp)
    return out


def crop_sharpness(crop_rgb: np.ndarray) -> float:
    """Laplacian variance of grayscale crop as a sharpness metric (RE-PERC-000020).

    Higher = sharper. Flat or heavily blurred patches evaluate to near 0.0.
    """
    if crop_rgb.size < 16 or min(crop_rgb.shape[:2]) < 4:
        return 0.0
    gray = (
        crop_rgb[..., 0] * 0.299 +
        crop_rgb[..., 1] * 0.587 +
        crop_rgb[..., 2] * 0.114
    ).astype(np.float32)
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap)) if lap.size > 0 else 0.0


def detect_hierarchy(image_rgb: np.ndarray, image_id: str, detector, ontology: Ontology, roots: Sequence[str],
                     max_depth: int = 3, min_crop_px: int = 96, budget_calls: int = 40,
                     model: str = "detector@unknown",
                     min_crop_sharpness: float = 0.0,
                     low_score_threshold: float = 0.45) -> ObservationTable:
    h, w = image_rgb.shape[:2]
    rows: list[dict] = []
    calls = 0
    queue: deque = deque()
    for r in roots:
        if ontology.get(r):
            queue.append(("root", r, (0, 0, w, h), -1, 0))
    while queue:
        kind, path, crop, parent_row, depth = queue.popleft()
        x0, y0, x1, y1 = crop
        if kind == "root":
            node = ontology.get(path)
            queries = {ph: (path, node.name) for ph in (node.prompts or (node.name,))}
        elif min(x1 - x0, y1 - y0) < min_crop_px:
            continue
        else:
            queries = _child_queries(ontology, path)
        if not queries:
            continue
        phrases = list(queries)
        sub = image_rgb[y0:y1, x0:x1]
        found = []
        for k in range(0, len(phrases), PROMPTS_PER_PASS):
            if calls >= budget_calls:
                break
            calls += 1
            found += list(detector.detect(sub, phrases[k:k + PROMPTS_PER_PASS]))
        by_class: dict[str, list] = {}
        for d in found:
            if d.label in queries:
                by_class.setdefault(queries[d.label][0], []).append(d)
        for cpath, dets in by_class.items():
            b = np.array([d.box for d in dets], dtype=float) + np.array([x0, y0, x0, y0])   # crop -> full image
            s = np.array([d.score for d in dets], dtype=float)
            for i in _nms(b, s):
                score_val = float(s[i])
                # Blurry region gating for low-score detections (RE-PERC-000020)
                if min_crop_sharpness > 0.0 and score_val < low_score_threshold:
                    bx_crop = (int(max(b[i, 0], 0)), int(max(b[i, 1], 0)), int(min(b[i, 2], w)), int(min(b[i, 3], h)))
                    patch = image_rgb[bx_crop[1]:bx_crop[3], bx_crop[0]:bx_crop[2]]
                    if crop_sharpness(patch) < min_crop_sharpness:
                        continue
                if parent_row >= 0:                                     # a part must sit inside its parent
                    pb = rows[parent_row]["box"]
                    cx, cy = (b[i, 0] + b[i, 2]) / 2, (b[i, 1] + b[i, 3]) / 2
                    if not (pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]):
                        continue
                rows.append({"path": cpath, "box": b[i].tolist(), "score": float(s[i]), "parent": parent_row})
                if depth < max_depth:
                    bx = (int(max(b[i, 0], 0)), int(max(b[i, 1], 0)), int(min(b[i, 2], w)), int(min(b[i, 3], h)))
                    queue.append(("part", cpath, bx, len(rows) - 1, depth + 1))
    t = ObservationTable.empty(len(rows))
    for i, r in enumerate(rows):
        t.obs_id[i] = f"{image_id}#{i}"; t.image_id[i] = image_id; t.class_path[i] = r["path"]
        t.parent_obs[i] = r["parent"]; t.bbox_px[i] = r["box"]; t.score[i] = r["score"]; t.model[i] = model
    t.reindex()
    return t
