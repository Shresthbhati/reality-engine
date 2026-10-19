"""Deterministic L-shaped room with a false ceiling, a window and furniture, rendered as REAL perspective images.

World frame: metres, Y up (the frame ``scripts/render_room_dataset.py`` and the vertical slice use). Plan (x, z):

    arm A  x in [0, 6], z in [0, 3]   false ceiling at y = 2.5 over this arm
    arm B  x in [0, 3], z in [3, 5]   bare structural slab at y = 2.8
    footprint area 24.0 m^2 (bounding box 30 m^2); window in the z = 0 wall (x 1.5..3.5, y 0.9..2.1) behind glass
    furniture: table 1.2 x 0.8 x 0.75 at x 3.6..4.8, z 0.8..1.6; sideboard 2.0 x 0.9 x 0.8 at x 0.5..2.5, z 4.0..4.9

Rendering follows render_room_dataset: quads are rasterised into per-pixel WORLD-POSITION buffers (exact geometry)
and shaded from world position with multi-octave value noise, so the same surface point looks identical from every
view -- the property SfM needs. Window glass is flat (low texture, so SfM yields no points there, like real glass).
Skirting / rail bands and frames put off-plane structure on every wall (planar-degeneracy guard, as in the original).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

FOOTPRINT = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 5), (0, 5)]
FOOTPRINT_AREA_M2 = 24.0
SLAB_Y, FALSE_Y = 2.8, 2.5
WINDOW = {"wall": "z=0", "x": (1.5, 3.5), "y": (0.9, 2.1), "recess": 0.15}
TABLE = ((3.6, 0.0, 0.8), (4.8, 0.75, 1.6))
SIDEBOARD = ((0.5, 0.0, 4.0), (2.5, 0.8, 4.9))
FURNITURE = {"table": TABLE, "sideboard": SIDEBOARD}

W, H = 960, 720
FX = FY = 760.0
CX, CY = W / 2, H / 2

FLOOR, CEIL_FALSE, CEIL_SLAB, WALL, GLASS, FRAME, TABLE_S, BOARD_S, STEP = range(9)
BASE = {FLOOR: (150, 120, 90), CEIL_FALSE: (238, 236, 230), CEIL_SLAB: (205, 200, 190), WALL: (185, 178, 168),
        GLASS: (170, 205, 245), FRAME: (235, 235, 235), TABLE_S: (128, 96, 60), BOARD_S: (96, 72, 52),
        STEP: (215, 210, 200)}


def _box(lo, hi, sid) -> List[Tuple[np.ndarray, int]]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    f = [[(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
         [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
         [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]]
    return [(np.array(q, float), sid) for q in f]


def build_scene() -> List[Tuple[np.ndarray, int]]:
    q: List[Tuple[np.ndarray, int]] = []
    A = lambda x0, x1, z0, z1, y: np.array([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], float)
    V = lambda a, b, y0, y1: np.array([(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y1, a[1])], float)
    # floor: two rectangles covering the L
    q += [(A(0, 6, 0, 3, 0), FLOOR), (A(0, 3, 3, 5, 0), FLOOR)]
    # ceilings: false ceiling over arm A, structural slab over arm B (the slab above arm A is hidden)
    q += [(A(0, 6, 0, 3, FALSE_Y), CEIL_FALSE), (A(0, 3, 3, 5, SLAB_Y), CEIL_SLAB)]
    # the drop face where the false ceiling stops (z = 3, x 0..3), facing arm B
    q.append((np.array([(0, FALSE_Y, 3), (3, FALSE_Y, 3), (3, SLAB_Y, 3), (0, SLAB_Y, 3)], float), STEP))
    # walls; the x=0 wall rises to the slab along arm B only, below the false ceiling elsewhere
    q += [(V((6, 0), (6, 3), 0, FALSE_Y), WALL), (V((6, 3), (3, 3), 0, FALSE_Y), WALL),
          (V((3, 3), (3, 5), 0, SLAB_Y), WALL), (V((3, 5), (0, 5), 0, SLAB_Y), WALL),
          (V((0, 5), (0, 3), 0, SLAB_Y), WALL), (V((0, 3), (0, 0), 0, FALSE_Y), WALL),
          (V((0, 3), (0, 3), 0, FALSE_Y), WALL)]
    wx0, wx1 = WINDOW["x"]
    wy0, wy1 = WINDOW["y"]
    r = WINDOW["recess"]
    # z = 0 wall in pieces around the window opening, glass recessed behind it
    q += [(V((0, 0), (wx0, 0), 0, FALSE_Y), WALL), (V((wx1, 0), (6, 0), 0, FALSE_Y), WALL),
          (V((wx0, 0), (wx1, 0), 0, wy0), WALL), (V((wx0, 0), (wx1, 0), wy1, FALSE_Y), WALL)]
    q += [(np.array([(wx0, wy0, r), (wx1, wy0, r), (wx1, wy1, r), (wx0, wy1, r)], float), GLASS)]
    q += [(np.array([(wx0, wy0, 0), (wx0, wy1, 0), (wx0, wy1, r), (wx0, wy0, r)], float), WALL),
          (np.array([(wx1, wy0, 0), (wx1, wy1, 0), (wx1, wy1, r), (wx1, wy0, r)], float), WALL),
          (np.array([(wx0, wy0, 0), (wx1, wy0, 0), (wx1, wy0, r), (wx0, wy0, r)], float), WALL),
          (np.array([(wx0, wy1, 0), (wx1, wy1, 0), (wx1, wy1, r), (wx0, wy1, r)], float), WALL)]
    # frame + mullion (protrude 4-5 cm into the room)
    q += _box((wx0 - 0.04, wy0 - 0.04, 0.0), (wx1 + 0.04, wy0, 0.04), FRAME)
    q += _box((wx0 - 0.04, wy1, 0.0), (wx1 + 0.04, wy1 + 0.04, 0.04), FRAME)
    q += _box(((wx0 + wx1) / 2 - 0.025, wy0, 0.0), ((wx0 + wx1) / 2 + 0.025, wy1, 0.05), FRAME)
    # furniture
    q += _box(TABLE[0], TABLE[1], TABLE_S) + _box(SIDEBOARD[0], SIDEBOARD[1], BOARD_S)
    # skirting + picture-rail bands protruding 2 cm along every wall segment (except across the window)
    for a, b in zip(FOOTPRINT, FOOTPRINT[1:] + FOOTPRINT[:1]):
        a, b = np.array(a, float), np.array(b, float)
        length = float(np.linalg.norm(b - a))
        d = (b - a) / length
        inward = np.array([-d[1], d[0]]) * 0.02
        for k in range(int(length / 0.5)):
            s0, s1 = a + d * (k * 0.5), a + d * min(length, (k + 1) * 0.5)
            for y0, y1 in ((0.0, 0.1), (1.55, 1.62)):
                on_window_wall = abs(s0[1]) < 1e-9 and abs(s1[1]) < 1e-9
                if on_window_wall and max(s0[0], s1[0]) > wx0 - 0.05 and min(s0[0], s1[0]) < wx1 + 0.05 and y1 > wy0 - 0.05 and y0 < wy1:
                    continue
                lo = (min(s0[0], s1[0], s0[0] + inward[0]), y0, min(s0[1], s1[1], s0[1] + inward[1]))
                hi = (max(s0[0], s1[0], s0[0] + inward[0]), y1, max(s0[1], s1[1], s0[1] + inward[1]))
                q += _box(lo, hi, WALL)
    # non-planar clutter on the floor (SfM initialisation aid)
    q += _box((5.2, 0.0, 2.3), (5.6, 0.35, 2.7), TABLE_S)
    q += _box((0.2, 0.0, 3.4), (0.5, 0.4, 3.7), BOARD_S)
    return q


def trajectory() -> List[dict]:
    """Walk positions through both arms, 8 yaws each (45 deg), pitched down ~15 deg; known poses."""
    # centreline walk (walls >= 1.2 m away) with a lateral offset on alternate spots for baseline diversity
    spots = [(1.2, 1.5), (2.1, 1.0), (3.0, 1.7), (3.9, 1.2), (4.8, 1.6), (1.4, 3.6), (1.6, 4.2), (2.3, 3.7), (2.2, 2.3)]
    cams = []
    for i, (x, z) in enumerate(spots):
        y = 1.2 if i % 2 else 1.5
        for k in range(8):
            yaw = k * math.pi / 4 + 0.2 * (i % 3)
            # alternate looking slightly down (floor/furniture) and up ~22 deg (ceiling layers), as a real capture should
            d = np.array([math.cos(yaw), -0.18 if k % 2 == 0 else 0.40, math.sin(yaw)])
            pos = np.array([x, y, z])
            cams.append({"id": f"L_{len(cams):04d}", "position": pos, "look_at": pos + d * 3.0})
    return cams


def look_at(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = target - pos
    f = f / np.linalg.norm(f)
    right = np.cross(f, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, f)
    return np.stack([right, -up, f], axis=0)          # world->camera rotation, +Z forward, +Y down (COLMAP convention)


def render(quads, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import render_room_dataset as base

    pbuf = np.zeros((H, W, 3))
    zbuf = np.full((H, W), np.inf)
    sbuf = np.full((H, W), -1, dtype=np.int32)
    origin = -R.T @ t
    cx, cy, cz = R[0], R[1], R[2]
    for p3d, sid in quads:
        pc = (R @ p3d.T).T + t
        if (pc[:, 2] <= 1e-3).all():
            continue
        pc[:, 2] = np.maximum(pc[:, 2], 1e-3)           # quads crossing the camera plane are approximated, not clipped
        uv = np.stack([FX * pc[:, 0] / pc[:, 2] + CX, FY * pc[:, 1] / pc[:, 2] + CY], axis=1)
        x0, x1 = int(max(0, np.floor(uv[:, 0].min()))), int(min(W - 1, np.ceil(uv[:, 0].max())))
        y0, y1 = int(max(0, np.floor(uv[:, 1].min()))), int(min(H - 1, np.ceil(uv[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        signs = np.empty((4,) + xs.shape)
        for i in range(4):
            j = (i + 1) % 4
            signs[i] = (uv[j, 0] - uv[i, 0]) * (ys - uv[i, 1]) - (uv[j, 1] - uv[i, 1]) * (xs - uv[i, 0])
        inside = np.all(signs >= 0, axis=0) | np.all(signs <= 0, axis=0)
        if not inside.any():
            continue
        n = np.cross(p3d[1] - p3d[0], p3d[2] - p3d[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        dw = ((xs - CX) / FX)[..., None] * cx + ((ys - CY) / FY)[..., None] * cy + cz
        den = dw @ n
        safe = np.abs(den) > 1e-12
        tau = np.where(safe, (float(n @ p3d[0]) - float(n @ origin)) / np.where(safe, den, 1.0), np.inf)
        zr, pr, sr = zbuf[y0:y1 + 1, x0:x1 + 1], pbuf[y0:y1 + 1, x0:x1 + 1], sbuf[y0:y1 + 1, x0:x1 + 1]
        upd = inside & safe & (tau > 1e-6) & (tau < zr)
        if upd.any():
            zr[upd], sr[upd], pr[upd] = tau[upd], sid, origin + tau[upd, None] * dw[upd]
    img = np.zeros((H, W, 3))
    cov = sbuf >= 0
    if cov.any():
        basec = np.array([BASE[i] for i in range(len(BASE))], float)
        sid = sbuf[cov]
        col = base._shade(pbuf[cov], basec[sid])
        col = col.reshape(-1, 3)                          # _shade returns (1, N, 3)
        col[sid == GLASS] = basec[GLASS]                # flat: no texture on glass
        img[cov] = col
    return img.astype(np.uint8)


def ground_truth() -> Dict:
    return {"footprint": FOOTPRINT, "area_m2": FOOTPRINT_AREA_M2, "false_ceiling_y": FALSE_Y, "slab_y": SLAB_Y,
            "window": WINDOW, "furniture": {k: {"min": v[0], "max": v[1]} for k, v in FURNITURE.items()}, "up": "y"}
