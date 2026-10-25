"""Image-space opening CANDIDATES on a detected wall, from one photograph.

A window or door is a rectangular region that differs in appearance from the
wall around it. This module finds such rectangles inside the pixels that a
single-view RANSAC plane already claimed as a vertical surface, and reports
them as *candidates* with the measured evidence (contrast against the ring
around them, area share, rectangularity). It does not recognise objects:

  * kind is a heuristic on position/shape only (touches the bottom of the
    wall region and is taller than wide => door-like; otherwise window-like),
    and the basis is returned with every candidate;
  * nothing outside the wall region's pixels is considered;
  * a clean, uniform wall yields NO candidates -- absence is the normal result.

Method: gaussian blur -> Canny -> contours -> 4-vertex convex polygons that
sit inside the wall mask, fill most of their minimum-area rectangle, have a
plausible size/aspect and a real intensity contrast against a surrounding ring.

(perception/architecture/openings.py is a different tool: it finds voids in
3D point coverage of registered planes. Monocular depth is dense, so that
approach has nothing to find in a single view; this one works on pixels.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

MIN_AREA_FRAC = 0.006        # of the wall mask area; smaller fragments (panes, rails) are noise
MAX_AREA_FRAC = 0.25
ASPECT_RANGE = (0.25, 4.0)   # width / height of the minimum-area rectangle
MIN_RECTANGULARITY = 0.75    # polygon area / min-area-rect area
MIN_CONTRAST = 18.0          # |mean inside - mean ring|, 0..255 grey levels
BOTTOM_TOUCH_FRAC = 0.06     # of the wall bbox height
DOOR_MIN_ASPECT_H_OVER_W = 1.3
MAX_CANDIDATES = 12
EDGE_TOUCH_FRAC = 0.03       # of the wall bbox: a rectangle this close to the wall's top/side is cut off by it


@dataclass(frozen=True)
class OpeningCandidate:
    quad_px: Tuple[Tuple[float, float], ...]   # 4 corners, depth-map pixel space
    bbox_px: Tuple[int, int, int, int]         # x0, y0, x1, y1
    contrast: float
    area_frac: float
    rectangularity: float
    touches_bottom: bool
    kind: str                                  # "door" | "window" | "opening" (unresolved: not door- or window-shaped)
    kind_basis: str
    wall_frac_w: float = 0.0                   # width / height as a fraction of the host wall region's pixel
    wall_frac_h: float = 0.0                   # extent -- the only size a single view can state honestly


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detect_opening_candidates(
    gray, plane_pixels: Sequence[Tuple[int, int]], cell: int,
) -> List[OpeningCandidate]:
    """``gray``: uint8 HxW image already resized to the depth map's size.
    ``plane_pixels``: (u, v) grid samples that supported the wall plane;
    ``cell``: the grid step, used to turn samples back into a solid mask."""
    import cv2
    import numpy as np

    h0, w0 = gray.shape[:2]
    # A door usually runs to the frame edge, where Canny sees no border and the
    # contour stays open. Pad with the median grey so a dark region touching the
    # frame closes into a rectangle; coordinates are mapped back at the end.
    pad_px = 8
    gray = cv2.copyMakeBorder(gray, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT,
                              value=int(np.median(gray)))
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    half = max(1, cell // 2 + 1)
    for u, v in plane_pixels:
        u, v = u + pad_px, v + pad_px
        mask[max(0, v - half):min(h, v + half + 1), max(0, u - half):min(w, u + half + 1)] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cell * 2 + 1, cell * 2 + 1), np.uint8))
    area = float(cv2.countNonZero(mask))
    ys, xs = np.nonzero(mask)
    if area < 400 or xs.size == 0:
        return []
    wy0, wy1 = int(ys.min()), int(ys.max())
    wx0, wx1 = int(xs.min()), int(xs.max())
    wall_h = max(1, wy1 - wy0)
    wall_w = max(1, wx1 - wx0)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 40, 120), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    found: List[OpeningCandidate] = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < MIN_AREA_FRAC * area or a > MAX_AREA_FRAC * area:
            continue
        poly = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        (cx, cy), (rw, rh), _ang = cv2.minAreaRect(poly)
        if rw < 4 or rh < 4:
            continue
        pa = cv2.contourArea(poly)
        rect_ratio = pa / (rw * rh)
        aspect = rw / rh
        if rect_ratio < MIN_RECTANGULARITY or not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
            continue
        if not (0 <= int(cy) < h and 0 <= int(cx) < w) or mask[int(cy), int(cx)] == 0:
            continue                                    # must lie on the wall's own pixels
        x, y, bw, bh = cv2.boundingRect(poly)
        inside = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inside, [poly], 255)
        pad = max(3, int(0.25 * min(bw, bh)))
        ring = np.zeros((h, w), np.uint8)
        cv2.rectangle(ring, (max(0, x - pad), max(0, y - pad)),
                      (min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)), 255, -1)
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(cv2.dilate(inside, np.ones((3, 3), np.uint8))))
        ring = cv2.bitwise_and(ring, mask)
        if cv2.countNonZero(ring) < 30:
            continue
        contrast = abs(float(gray[inside > 0].mean()) - float(gray[ring > 0].mean()))
        if contrast < MIN_CONTRAST:
            continue
        touches = (y + bh) >= wy1 - BOTTOM_TOUCH_FRAC * wall_h
        tall = bh / max(1, bw) >= DOOR_MIN_ASPECT_H_OVER_W
        cut_top = y <= wy0 + EDGE_TOUCH_FRAC * wall_h
        cut_side = x <= wx0 + EDGE_TOUCH_FRAC * wall_w or (x + bw) >= wx1 - EDGE_TOUCH_FRAC * wall_w
        if touches and tall:
            kind = "door"
            basis = f"touches the bottom of the wall region and is {bh / max(1, bw):.1f}x taller than wide"
        elif cut_top or cut_side:
            # An edge-bounded opening needs wall on every side it claims. A rectangle running into the wall
            # region's top or side is cut off by missing data (the wall is incomplete there): no candidate.
            continue
        elif not touches:
            kind = "window"
            basis = "enclosed by the wall region on all four sides, with no contact with its bottom"
        else:
            kind = "opening"
            basis = ("touches the bottom of the wall region but is not door-shaped: an opening whose kind "
                     "cannot be determined from one view")
        def _back(px, py):
            return (float(min(w0 - 1, max(0, px - pad_px))), float(min(h0 - 1, max(0, py - pad_px))))

        found.append(OpeningCandidate(
            quad_px=tuple(_back(float(p[0][0]), float(p[0][1])) for p in poly),
            bbox_px=(max(0, x - pad_px), max(0, y - pad_px), min(w0, x + bw - pad_px), min(h0, y + bh - pad_px)), contrast=round(contrast, 1),
            area_frac=round(pa / area, 4), rectangularity=round(rect_ratio, 3),
            touches_bottom=bool(touches), kind=kind, kind_basis=basis,
            wall_frac_w=round(bw / wall_w, 3), wall_frac_h=round(bh / wall_h, 3),
        ))

    found.sort(key=lambda o: o.contrast * o.area_frac, reverse=True)
    kept: List[OpeningCandidate] = []
    for cand in found:
        if all(_iou(cand.bbox_px, k.bbox_px) < 0.5 for k in kept):
            kept.append(cand)
        if len(kept) >= MAX_CANDIDATES:
            break
    return kept
