"""Furniture measured from the segmentation mask and metric depth, not from a detector box.

The scene segmenter (ADE20K) labels sofas, tables, cabinets ... per pixel. Inside a detector box, the pixels carrying a furniture
label are the piece's mask; back-projected with the metric depth map and dropped on the floor they give its footprint
(oriented rectangle), height and centre. The class is the segmenter's own label when it covers the box (it separates a coffee table
from a desk and a TV console from a bookshelf by what the pixels are, not by a box prior).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: ADE20K label -> the placer's furniture class (perception.instances.furniture_from_detections.SIZE_PRIORS)
ADE_CLASS: Dict[str, str] = {
    "sofa": "sofa", "couch": "sofa", "coffee table": "coffee_table", "table": "dining_table", "desk": "desk",
    "chair": "chair", "armchair": "armchair", "swivel chair": "chair", "seat": "chair", "bench": "chair",
    "bed": "bed", "cabinet": "cabinet", "chest of drawers": "dresser", "wardrobe": "wardrobe", "bookcase": "bookshelf",
    "shelf": "bookshelf", "stool": "stool", "ottoman": "stool", "pouf": "stool", "television receiver": "tv", "monitor": "tv",
    "buffet": "cabinet", "sideboard": "cabinet",
}
MIN_BOX_SHARE = 0.15   # furniture-labelled pixels must cover this much of the box for the segmenter's class to be trusted
HINT_KEEP = 0.6        # the detector's own class is kept when the segmenter finds at least this fraction as many of its pixels
MIN_MASK_PX = 60
TRIM_PCT = (2.0, 98.0)


def class_and_mask(label_map: np.ndarray, id2label: Dict[int, str], box: Sequence[float], hint: Optional[str] = None
                   ) -> Optional[Tuple[str, np.ndarray]]:
    """``(class, full-size bool mask)`` of the furniture inside ``box`` (x0, y0, x1, y1 px), or None when no furniture label is there."""
    h, w = label_map.shape
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    region = label_map[y0:y1, x0:x1]
    ids_of: Dict[str, list] = {}
    for lid, name in id2label.items():
        cls = ADE_CLASS.get(name)
        if cls:
            ids_of.setdefault(cls, []).append(int(lid))
    counts = {cls: int(np.isin(region, ids).sum()) for cls, ids in ids_of.items()}
    counts = {c: n for c, n in counts.items() if n > 0}
    if not counts:
        return None
    best = max(counts, key=counts.get)
    if counts[best] < MIN_BOX_SHARE * region.size:
        return None
    if hint in counts and counts[hint] >= HINT_KEEP * counts[best]:
        best = hint
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = np.isin(region, ids_of[best])
    return best, mask


def footprint_from_mask(mask: np.ndarray, depth_m: np.ndarray, f: float, cx: float, cy: float, *, up: Sequence[float],
                        camera_height_m: float) -> Optional[dict]:
    """Oriented plan rectangle, height and centre of the masked pixels; camera frame x right, y DOWN, z forward, ``up`` unit vector.

    Returns ``{centre_xz, length_m, width_m, height_m, yaw_rad, n_points}`` (plan axes: a = right, b = forward) or None.
    """
    import cv2

    vs, us = np.nonzero(mask)
    z = depth_m[vs, us]
    ok = np.isfinite(z) & (z > 0.2) & (z < 25.0)
    vs, us, z = vs[ok], us[ok], z[ok]
    if len(z) < MIN_MASK_PX:
        return None
    p = np.stack([(us + 0.5 - cx) / f * z, (vs + 0.5 - cy) / f * z, z], axis=1)
    upv = np.asarray(up, float)
    e1 = np.array([1.0, 0.0, 0.0]) - upv * upv[0]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(upv, e1)
    height = p @ upv + camera_height_m
    plan = np.stack([p @ e1, p @ e2], axis=1)
    keep = height > -0.15
    plan, height = plan[keep], height[keep]
    if len(plan) < MIN_MASK_PX:
        return None
    lo, hi = np.percentile(plan, TRIM_PCT, axis=0)
    inner = np.all((plan >= lo) & (plan <= hi), axis=1)
    (ca, cb), (rw, rh), angle = cv2.minAreaRect(plan[inner].astype(np.float32))
    long_w = rw >= rh
    axis_deg = angle if long_w else angle + 90.0
    floor_pt = ca * e1 + cb * e2 - camera_height_m * upv  # the footprint centre ON the floor, camera frame (y down)
    return {"floor_point_cam": tuple(float(c) for c in floor_pt), "centre_xz": (float(ca), float(cb)), "length_m": float(max(rw, rh)), "width_m": float(min(rw, rh)),
            "height_m": float(np.percentile(height, 95.0)), "yaw_rad": math.radians(axis_deg), "n_points": int(len(plan))}
