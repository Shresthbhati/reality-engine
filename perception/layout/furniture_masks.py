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
CELL_M = 0.1            # plan occupancy grid for splitting non-rectangular footprints
SINGLE_FILL = 0.72      # a footprint filling this much of its bounding rectangle is one rectangle
MAX_PARTS = 3
MIN_PART_SHARE = 0.12   # a further rectangle must carry this share of the footprint


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


def _max_rect(g: np.ndarray):
    """Largest all-ones axis-aligned rectangle of a 0/1 grid: ``(area, y0, y1, x0, x1)`` (half-open)."""
    rows, cols = g.shape
    run = np.zeros(cols, int)
    best = (0, 0, 0, 0, 0)
    for y in range(rows):
        run = np.where(g[y] > 0, run + 1, 0)
        stack: list = []
        for x in range(cols + 1):
            hgt = int(run[x]) if x < cols else 0
            start = x
            while stack and stack[-1][1] >= hgt:
                s0, h0 = stack.pop()
                if h0 * (x - s0) > best[0]:
                    best = (h0 * (x - s0), y - h0 + 1, y + 1, s0, x)
                start = s0
            stack.append((start, hgt))
    return best


def split_rectangles(plan: np.ndarray, height: np.ndarray) -> list:
    """Cover a plan footprint (N x 2 points, metres) with up to MAX_PARTS rectangles: an L-shaped sectional becomes two pieces.

    Each part: ``{centre (a, b), length, width, yaw_rad, height}``. A footprint that fills most of its bounding rectangle is one part.
    """
    import cv2

    (ca, cb), _, angle = cv2.minAreaRect(plan.astype(np.float32))
    ang = math.radians(angle)
    d1, d2 = np.array([math.cos(ang), math.sin(ang)]), np.array([-math.sin(ang), math.cos(ang)])
    c = np.array([ca, cb])
    uv = np.stack([(plan - c) @ d1, (plan - c) @ d2], axis=1)
    lo = uv.min(0)
    ij = np.floor((uv - lo) / CELL_M).astype(int)
    rows, cols = int(ij[:, 1].max()) + 1, int(ij[:, 0].max()) + 1
    cnt = np.zeros((rows, cols), np.int32)
    np.add.at(cnt, (ij[:, 1], ij[:, 0]), 1)
    occ = cv2.morphologyEx((cnt >= 2).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    total = int(occ.sum())

    def part(y0: int, y1: int, x0: int, x1: int) -> dict:
        centre = c + d1 * (lo[0] + (x0 + x1) / 2 * CELL_M) + d2 * (lo[1] + (y0 + y1) / 2 * CELL_M)
        lu, lv = (x1 - x0) * CELL_M, (y1 - y0) * CELL_M
        inside = (ij[:, 0] >= x0) & (ij[:, 0] < x1) & (ij[:, 1] >= y0) & (ij[:, 1] < y1)
        h = np.percentile(height[inside] if inside.sum() >= 5 else height, 95.0)
        return {"centre": (float(centre[0]), float(centre[1])), "length": float(max(lu, lv)), "width": float(min(lu, lv)),
                "yaw_rad": ang if lu >= lv else ang + math.pi / 2, "height": float(h)}

    if total == 0 or total >= SINGLE_FILL * rows * cols:
        return [part(0, rows, 0, cols)]
    work, parts = occ.copy(), []
    for _ in range(MAX_PARTS):
        area, y0, y1, x0, x1 = _max_rect(work)
        if area < max(6, MIN_PART_SHARE * total):
            break
        parts.append(part(y0, y1, x0, x1))
        work[y0:y1, x0:x1] = 0
    return parts or [part(0, rows, 0, cols)]


def footprint_from_mask(mask: np.ndarray, depth_m: np.ndarray, f: float, cx: float, cy: float, *, up: Sequence[float],
                        camera_height_m: float, depth_hint_m: float = 0.9) -> Optional[dict]:
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
    # The footprint is the mask's FLOOR-CONTACT line (its lowest pixel per column, on the floor plane), extended back by the class's
    # typical depth: the visible top/back surfaces of a sofa form a filled ramp that hides an L, the contact line does not.
    order = np.lexsort((vs, us))
    last = np.r_[np.nonzero(np.diff(us[order]))[0], len(order) - 1]
    cu, cv_ = us[order][last], vs[order][last]
    ray = np.stack([(cu + 0.5 - cx) / f, (cv_ + 1.0 - cy) / f, np.ones(len(cu))], axis=1)
    den = ray @ upv
    t = np.where(den < -0.02, -camera_height_m / np.minimum(den, -1e-9), np.inf)
    ok = t < 30.0
    contact = np.stack([(t[:, None] * ray)[ok] @ e1, (t[:, None] * ray)[ok] @ e2], axis=1)
    if len(contact) < 8:
        return None
    away = contact.mean(axis=0)
    away = away / max(float(np.linalg.norm(away)), 1e-9)  # one direction (the mean view direction): a radial fan would widen the piece
    occupied = np.vstack([contact + away * dd for dd in np.arange(0.0, depth_hint_m + 1e-9, 0.05)])
    lo, hi = np.percentile(occupied, TRIM_PCT, axis=0)
    inner = np.all((occupied >= lo) & (occupied <= hi), axis=1)
    (ca, cb), (rw, rh), angle = cv2.minAreaRect(occupied[inner].astype(np.float32))
    long_w = rw >= rh
    axis_deg = angle if long_w else angle + 90.0
    floor_pt = ca * e1 + cb * e2 - camera_height_m * upv  # the footprint centre ON the floor, camera frame (y down)
    parts = split_rectangles(occupied[inner], np.full(int(inner.sum()), float(np.percentile(height, 95.0))))
    for q in parts:
        # the part's own height: the mask points that fall inside its rectangle (else the whole piece's)
        ang = q["yaw_rad"]
        dl, dw = np.array([math.cos(ang), math.sin(ang)]), np.array([-math.sin(ang), math.cos(ang)])
        rel = plan - np.array(q["centre"])
        within = (np.abs(rel @ dl) <= q["length"] / 2 + 0.15) & (np.abs(rel @ dw) <= q["width"] / 2 + 0.15)
        if within.sum() >= 20:
            q["height"] = float(np.percentile(height[within], 95.0))
        q["floor_point_cam"] = tuple(float(c) for c in q["centre"][0] * e1 + q["centre"][1] * e2 - camera_height_m * upv)
    return {"parts": parts, "floor_point_cam": tuple(float(c) for c in floor_pt), "centre_xz": (float(ca), float(cb)), "length_m": float(max(rw, rh)), "width_m": float(min(rw, rh)),
            "height_m": float(np.percentile(height, 95.0)), "yaw_rad": math.radians(axis_deg), "n_points": int(len(plan))}
