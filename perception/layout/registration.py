"""Register two photographs of one room from metric depth + matched image features.

Structure-from-motion (COLMAP) is the usual way to place photos in one space, and it fails on few, low-overlap or generated
photos. Here every photo already has metric depth (``perception.layout.learned_models``), so a feature match lifts to a 3D-3D
correspondence and the rigid motion between the two cameras follows from a robust Kabsch fit: SIFT matches -> back-project with
each photo's depth -> RANSAC rigid transform -> refit on the inliers.

``register_pair`` returns the pose of the SOURCE camera in the REFERENCE camera's frame (``p_ref = R @ p_src + t``) or ``None``
when the photos cannot be placed together (too few consistent matches). It never returns a low-confidence pose: refusing is
the honest answer, and the caller then leaves that photo out and says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

#: matches deeper than this are dropped: indoor metric depth is unreliable there and far points dominate rotation errors
MAX_MATCH_DEPTH_M = 8.0
MIN_MATCH_DEPTH_M = 0.3
RATIO = 0.8
RANSAC_ITERS = 1500
#: inlier band: a base error plus a fraction of the range (monocular depth error grows with distance)
INLIER_BASE_M = 0.10
INLIER_DEPTH_FRACTION = 0.03
MIN_INLIERS = 25
MIN_INLIER_RATIO = 0.25
MAX_SIFT_SIDE = 1000
#: the inliers must spread over more than a line (second singular value, metres) or the rotation is undetermined
MIN_SPREAD_M = 0.3


@dataclass(frozen=True)
class Registration:
    R: np.ndarray  # source -> reference rotation
    t: np.ndarray  # source camera position in the reference frame
    inliers: int
    n_matches: int
    rms_m: float


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _features(gray: np.ndarray):
    h, w = gray.shape
    s = min(1.0, MAX_SIFT_SIDE / float(max(h, w)))
    small = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else gray
    kps, desc = cv2.SIFT_create(nfeatures=4000).detectAndCompute(small, None)
    pts = np.array([k.pt for k in kps], dtype=np.float64) / s if kps else np.zeros((0, 2))
    return pts, desc


def _lift(pts_px: np.ndarray, depth: np.ndarray, hfov_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates -> camera-frame 3D points using the depth map. Returns (points, valid mask)."""
    h, w = depth.shape
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u, v = pts_px[:, 0], pts_px[:, 1]
    ui, vi = np.clip(u.astype(int), 0, w - 1), np.clip(v.astype(int), 0, h - 1)
    z = depth[vi, ui]
    ok = np.isfinite(z) & (z > MIN_MATCH_DEPTH_M) & (z < MAX_MATCH_DEPTH_M)
    return np.stack([(u - w / 2.0) / f * z, (v - h / 2.0) / f * z, z], axis=-1), ok


def _kabsch(src: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid transform with ``ref ~ R @ src + t`` (proper rotation)."""
    cs, cr = src.mean(axis=0), ref.mean(axis=0)
    h = (src - cs).T @ (ref - cr)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, cr - rot @ cs


def register_pair(ref_rgb: np.ndarray, ref_depth: np.ndarray, src_rgb: np.ndarray, src_depth: np.ndarray, hfov_deg: float,
                  *, seed: int = 0, min_inliers: int = MIN_INLIERS) -> Optional[Registration]:
    pr, dr = _features(_gray(ref_rgb))
    ps, ds = _features(_gray(src_rgb))
    if dr is None or ds is None or len(pr) < 8 or len(ps) < 8:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    fwd = matcher.knnMatch(ds, dr, k=2)
    back = matcher.knnMatch(dr, ds, k=1)
    back_of = {m[0].queryIdx: m[0].trainIdx for m in back if m}
    pairs = []
    for m in fwd:
        if len(m) < 2 or m[0].distance > RATIO * m[1].distance:
            continue
        if back_of.get(m[0].trainIdx) != m[0].queryIdx:  # mutual nearest neighbours only
            continue
        pairs.append((m[0].queryIdx, m[0].trainIdx))
    if len(pairs) < min_inliers:
        return None
    si, ri = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    p_src, ok_s = _lift(ps[si], src_depth, hfov_deg)
    p_ref, ok_r = _lift(pr[ri], ref_depth, hfov_deg)
    ok = ok_s & ok_r
    p_src, p_ref = p_src[ok], p_ref[ok]
    n = len(p_src)
    if n < min_inliers:
        return None
    tol = INLIER_BASE_M + INLIER_DEPTH_FRACTION * p_ref[:, 2]
    rng = np.random.default_rng(seed)
    best_mask, best_count = None, 0
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(n, 3, replace=False)
        a = p_src[idx]
        if np.linalg.matrix_rank(a - a.mean(axis=0), tol=0.05) < 2:  # collinear sample: rotation undetermined
            continue
        rot, tr = _kabsch(a, p_ref[idx])
        mask = np.linalg.norm(p_src @ rot.T + tr - p_ref, axis=1) <= tol
        cnt = int(mask.sum())
        if cnt > best_count:
            best_mask, best_count = mask, cnt
    if best_mask is None or best_count < min_inliers or best_count < MIN_INLIER_RATIO * n:
        return None
    mask = best_mask
    rot = tr = None
    for _ in range(3):  # refit on the inliers, then re-collect
        rot, tr = _kabsch(p_src[mask], p_ref[mask])
        mask = np.linalg.norm(p_src @ rot.T + tr - p_ref, axis=1) <= tol
        if int(mask.sum()) < min_inliers:
            return None
    inl = p_ref[mask]
    if np.linalg.svd(inl - inl.mean(axis=0), compute_uv=False)[1] < MIN_SPREAD_M:
        return None
    resid = np.linalg.norm(p_src[mask] @ rot.T + tr - p_ref[mask], axis=1)
    return Registration(R=rot, t=tr, inliers=int(mask.sum()), n_matches=n, rms_m=float(np.sqrt(np.mean(resid ** 2))))
