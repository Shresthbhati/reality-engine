"""Textured, ray-cast box room seen from ANY camera pose: ground truth for photo-to-photo registration and multi-view layout.

World frame: x right, y DOWN, z forward (the same convention as the layout solver). The room is 5 x 5 x 2.6 m: walls at x=-2.5,
x=+2.5, z=-2.0, z=+3.0; floor at y=+1.5 and ceiling at y=-1.1 (a camera at y=0 is 1.5 m above the floor). Every surface carries its
own blurred-noise texture so feature matching has something to find, and no two surfaces look alike.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

HFOV = 60.0
PLANES = {  # name: (axis, value)
    "floor": (1, 1.5), "ceiling": (1, -1.1),
    "wall_xm": (0, -2.5), "wall_xp": (0, 2.5), "wall_zm": (2, -2.0), "wall_zp": (2, 3.0),
}
_TEX = 512
_PX_PER_M = 40.0


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (_TEX, _TEX), dtype=np.uint8).astype(np.float32)
    fine = cv2.GaussianBlur(base, (0, 0), 1.2)
    coarse = cv2.GaussianBlur(rng.integers(0, 255, (_TEX, _TEX), dtype=np.uint8).astype(np.float32), (0, 0), 6.0)
    out = 0.65 * fine + 0.35 * coarse
    out = (out - out.min()) / (np.ptp(out) + 1e-6) * 255
    return out.astype(np.uint8)


_TEXTURES = {name: _texture(100 + i) for i, name in enumerate(PLANES)}


@dataclass
class View:
    depth: np.ndarray  # (H, W) metric z-depth in the camera frame
    rgb: np.ndarray  # (H, W, 3) uint8
    masks: Dict[str, np.ndarray]  # floor / ceiling / wall / window / door
    R_wc: np.ndarray  # camera -> world rotation
    t_wc: np.ndarray  # camera position in the world
    hfov: float


def pose_matrix(x: float, z: float, yaw_deg: float, pitch_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])  # positive pitch looks down
    return ry @ rx, np.array([x, 0.0, z])


def render(x: float, z: float, yaw_deg: float, pitch_deg: float = 0.0, size: Tuple[int, int] = (320, 240), hfov: float = HFOV,
           window: Optional[Tuple[float, float, float, float]] = None) -> View:
    """``window``: (x0, x1, h0, h1) on the +z wall, metres / height above floor; labelled 'window' and see-through (depth 12 m)."""
    w, h = size
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cam_rays = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u)], -1)
    R, t = pose_matrix(x, z, yaw_deg, pitch_deg)
    rays = cam_rays @ R.T
    best_t = np.full((h, w), np.inf)
    best = np.full((h, w), "", dtype=object)
    for name, (ax, val) in PLANES.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (val - t[ax]) / rays[..., ax]
        ok = (tt > 1e-6) & (tt < best_t)
        best_t, best = np.where(ok, tt, best_t), np.where(ok, name, best)
    hit = t + rays * best_t[..., None]
    depth = ((hit - t) @ R)[..., 2]
    gray = np.zeros((h, w), np.float32)
    for name, (ax, _) in PLANES.items():
        m = best == name
        if not m.any():
            continue
        a, b = {1: (0, 2), 0: (2, 1), 2: (0, 1)}[ax]
        uu = (hit[..., a][m] * _PX_PER_M).astype(int) % _TEX
        vv = (hit[..., b][m] * _PX_PER_M).astype(int) % _TEX
        gray[m] = _TEXTURES[name][vv, uu]
    rgb = np.repeat(gray[..., None], 3, -1).astype(np.uint8)
    masks = {"floor": best == "floor", "ceiling": best == "ceiling", "wall": np.char.startswith(best.astype(str), "wall"),
             "window": np.zeros((h, w), bool), "door": np.zeros((h, w), bool)}
    if window is not None:
        x0, x1, h0, h1 = window
        win = (best == "wall_zp") & (hit[..., 0] > x0) & (hit[..., 0] < x1) & (hit[..., 1] < 1.5 - h0) & (hit[..., 1] > 1.5 - h1)
        masks["window"], masks["wall"] = win, masks["wall"] & ~win
        depth = np.where(win, 12.0, depth)
    return View(depth=depth, rgb=rgb, masks=masks, R_wc=R, t_wc=t, hfov=hfov)


def relative_pose(ref: View, src: View) -> Tuple[np.ndarray, np.ndarray]:
    """Pose of ``src`` in ``ref``'s frame: p_ref = R @ p_src + t."""
    return ref.R_wc.T @ src.R_wc, ref.R_wc.T @ (src.t_wc - ref.t_wc)


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(a.T @ b) - 1) / 2
    return math.degrees(math.acos(float(np.clip(c, -1, 1))))
