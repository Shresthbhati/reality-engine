"""Photo-realistic-ENOUGH renders of a KNOWN box room, for measuring the photos -> world chain.

A textured axis-aligned box (width x depth x height, z up) is ray-cast with a pinhole camera along a known path. The
output is what a phone would give the pipeline -- JPEG images (and optionally an .mp4) -- plus the exact ground truth
(room dimensions, every camera pose, intrinsics). Everything is seeded and deterministic. numpy + cv2 only.

What this proves: the photos -> SfM -> world chain recovers a known shape. What it does NOT prove: anything about real
photographs (lens distortion, motion blur, rolling shutter, exposure change, people, clutter, glass). Real-photo
behaviour is a separate, still-unmeasured question.

Camera convention is OpenCV (x right, y down, z forward); ``R_wc`` maps camera -> world, ``C`` is the centre.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class RoomSpec:
    width: float = 5.0   # x extent, model units
    depth: float = 4.0   # y extent
    height: float = 2.6  # z extent


@dataclass(frozen=True)
class Camera:
    width: int = 960
    height: int = 720
    fx: float = 760.0  # ~64 deg horizontal fov

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.width / 2], [0, self.fx, self.height / 2], [0, 0, 1.0]])


def _texture(seed: int, size: int = 1536) -> np.ndarray:
    """Dense multi-scale colour texture with corner-rich detail (SIFT-friendly)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    for sigma, amp in ((60, 70), (22, 55), (8, 40), (3, 25)):
        n = rng.normal(0, 1, (size, size, 3)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        n /= max(float(n.std()), 1e-6)
        img += n * amp
    img += 128
    out = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(140):  # hard-edged shapes give repeatable corners
        c = tuple(int(v) for v in rng.integers(20, 235, 3))
        x, y = (int(v) for v in rng.integers(0, size, 2))
        if rng.random() < 0.5:
            w, h = (int(v) for v in rng.integers(12, 110, 2))
            cv2.rectangle(out, (x, y), (x + w, y + h), c, -1 if rng.random() < 0.7 else 2)
        else:
            cv2.circle(out, (x, y), int(rng.integers(6, 45)), c, -1 if rng.random() < 0.7 else 2)
    return out


def _planes(room: RoomSpec):
    """(origin, u_axis, v_axis, u_len, v_len, texture seed) for each of the six faces; normal is irrelevant."""
    W, D, H = room.width, room.depth, room.height
    o = np.array
    return [
        (o([0, 0, 0.]), o([1, 0, 0.]), o([0, 1, 0.]), W, D, 11),   # floor
        (o([0, 0, H]), o([1, 0, 0.]), o([0, 1, 0.]), W, D, 12),    # ceiling
        (o([0, 0, 0.]), o([1, 0, 0.]), o([0, 0, 1.]), W, H, 13),   # wall y=0
        (o([0, D, 0.]), o([1, 0, 0.]), o([0, 0, 1.]), W, H, 14),   # wall y=D
        (o([0, 0, 0.]), o([0, 1, 0.]), o([0, 0, 1.]), D, H, 15),   # wall x=0
        (o([W, 0, 0.]), o([0, 1, 0.]), o([0, 0, 1.]), D, H, 16),   # wall x=W
    ]


def look_at(C: Sequence[float], target: Sequence[float], up=(0, 0, 1.0)) -> np.ndarray:
    """R_wc (columns = camera x,y,z axes in world) for a camera at C looking at target, OpenCV convention."""
    C = np.asarray(C, float)
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # down
    return np.stack([x, y, z], axis=1)


def orbit_path(room: RoomSpec, n: int, *, turns: float = 1.0, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """A handheld-like walk around the room's interior: positions on an ellipse at ~eye height with slight jitter,
    each looking across the room (at the point on the far side, offset so every wall is seen over the loop)."""
    rng = np.random.default_rng(seed)
    W, D, H = room.width, room.depth, room.height
    cx, cy = W / 2, D / 2
    out = []
    for i in range(n):
        a = 2 * np.pi * turns * i / n
        pos = np.array([cx + 0.28 * W * np.cos(a) + rng.normal(0, 0.03),
                        cy + 0.26 * D * np.sin(a) + rng.normal(0, 0.03),
                        0.45 * H + 0.12 * np.sin(3 * a) + rng.normal(0, 0.02)])
        # look past the centre toward the wall on the opposite side, swinging so floor/ceiling edges appear
        tgt = np.array([cx - 0.45 * W * np.cos(a + 0.6), cy - 0.45 * D * np.sin(a + 0.6), 0.5 * H + 0.35 * np.sin(2 * a)])
        out.append((look_at(pos, tgt), pos))
    return out


def render(room: RoomSpec, cam: Camera, R_wc: np.ndarray, C: np.ndarray, textures) -> np.ndarray:
    """Ray-cast one view. Returns BGR uint8."""
    h, w = cam.height, cam.width
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d_cam = np.stack([(u - w / 2) / cam.fx, (v - h / 2) / cam.fx, np.ones_like(u)], -1).reshape(-1, 3)
    d = d_cam @ R_wc.T  # world-space ray directions
    best_t = np.full(d.shape[0], np.inf)
    out = np.zeros((d.shape[0], 3), np.float32)
    for (o, ua, va, ul, vl, seed), tex in zip(_planes(room), textures):
        n = np.cross(ua, va)
        denom = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((o - C) @ n) / denom
        P = C + t[:, None] * d
        s = (P - o) @ ua
        r = (P - o) @ va
        ok = (t > 1e-6) & (t < best_t) & (s >= -1e-9) & (s <= ul + 1e-9) & (r >= -1e-9) & (r <= vl + 1e-9)
        if not ok.any():
            continue
        th, tw = tex.shape[:2]
        mx = np.clip(s[ok] / ul, 0, 1) * (tw - 1)
        my = np.clip(r[ok] / vl, 0, 1) * (th - 1)
        x0 = np.minimum(mx.astype(int), tw - 2)
        y0 = np.minimum(my.astype(int), th - 2)
        fx_, fy_ = (mx - x0)[:, None], (my - y0)[:, None]
        t32 = tex.astype(np.float32)
        samp = ((t32[y0, x0] * (1 - fx_) + t32[y0, x0 + 1] * fx_) * (1 - fy_)
                + (t32[y0 + 1, x0] * (1 - fx_) + t32[y0 + 1, x0 + 1] * fx_) * fy_)
        shade = 0.82 + 0.18 * abs(float(n @ np.array([0.3, 0.2, 0.93])) / np.linalg.norm(n))
        out[ok] = samp * shade
        best_t[ok] = t[ok]
    img = np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3)
    return cv2.GaussianBlur(img, (0, 0), 0.6)  # soften aliasing like a lens would


def write_capture(out_dir: Path, room: RoomSpec = RoomSpec(), cam: Camera = Camera(), n_images: int = 40,
                  n_video_frames: int = 90, fps: int = 30, seed: int = 0) -> dict:
    """Write images/ (JPEG), room.mp4 (a SEPARATE denser pass along the same loop) and ground_truth.json.
    Returns the ground truth dict."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    textures = [_texture(p[5]) for p in _planes(room)]
    gt_cams = []
    for i, (R, C) in enumerate(orbit_path(room, n_images, seed=seed)):
        img = render(room, cam, R, C, textures)
        name = f"img_{i:03d}.jpg"
        cv2.imwrite(str(out_dir / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        gt_cams.append({"image": name, "C": C.tolist(), "R_wc": R.tolist()})
    vid_cams = []
    if n_video_frames:
        vw = cv2.VideoWriter(str(out_dir / "room.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cam.width, cam.height))
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter could not open an mp4v writer")
        for i, (R, C) in enumerate(orbit_path(room, n_video_frames, seed=seed + 1)):
            vw.write(render(room, cam, R, C, textures))
            vid_cams.append({"frame": i, "C": C.tolist(), "R_wc": R.tolist()})
        vw.release()
    gt = {"room": asdict(room), "camera": asdict(cam), "K": cam.K.tolist(), "images": gt_cams, "video_frames": vid_cams,
          "units": "model units (no metric claim)"}
    (out_dir / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    return gt
