"""Score a reconstructed box room against its ground truth, scale-free.

Photos give no metric scale, so the reconstruction lives in an arbitrary gauge (a similarity transform of the truth).
Everything here is either invariant to that gauge (angles, ratios) or measured AFTER aligning the estimated camera
centres to the true ones with a similarity transform (Umeyama) and expressed as a fraction of the true room diagonal.
No metres are ever claimed. numpy only; pure functions (unit-tested in tests/test_room_eval.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FACE_NAMES = ("floor", "ceiling", "wall_y0", "wall_yD", "wall_x0", "wall_xW")


def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var)
    return s, R, mu_d - s * R @ mu_s


def gt_faces(room: dict) -> List[Tuple[np.ndarray, float]]:
    """(unit normal, offset d) with n.x = d, in FACE_NAMES order."""
    W, D, H = room["width"], room["depth"], room["height"]
    e = np.eye(3)
    return [(e[2], 0.0), (e[2], H), (e[1], 0.0), (e[1], D), (e[0], 0.0), (e[0], W)]


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def quat_wxyz_to_R(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fit_plane(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """(unit normal, centroid, rms residual) by SVD."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, c, float(np.sqrt(np.mean(((pts - c) @ n) ** 2)))


def score_cameras(est_centers: np.ndarray, est_R_c2w: Optional[np.ndarray], gt_centers: np.ndarray,
                  gt_R_c2w: np.ndarray, diag: float):
    s, R, t = umeyama(est_centers, gt_centers)
    err = np.linalg.norm((s * (est_centers @ R.T) + t) - gt_centers, axis=1)
    out = {"n": int(len(err)), "similarity_scale_est_to_gt": s,
           "position_rmse_frac_of_room_diagonal": float(np.sqrt(np.mean(err ** 2)) / diag),
           "position_max_frac_of_room_diagonal": float(err.max() / diag)}
    if est_R_c2w is not None:
        flip = np.diag([1.0, -1.0, -1.0])  # OpenGL-style camera axes vs OpenCV
        best = None
        for name, F in (("as_reported", np.eye(3)), ("axes_y_z_flipped", flip)):
            degs = [_geodesic_deg(g, R @ e @ F) for g, e in zip(gt_R_c2w, est_R_c2w)]
            if best is None or np.median(degs) < best[1]:
                best = (name, float(np.median(degs)), float(np.max(degs)), float(np.mean(degs)))
        out.update(rotation_convention_best_fit=best[0], rotation_median_deg=best[1],
                   rotation_mean_deg=best[3], rotation_max_deg=best[2])
    return out, (s, R, t)


def score_points(pts_est: np.ndarray, sim: Tuple[float, np.ndarray, np.ndarray], room: dict) -> Dict:
    s, R, t = sim
    p = s * (pts_est @ R.T) + t
    W, D, H = room["width"], room["depth"], room["height"]
    diag = float(np.sqrt(W * W + D * D + H * H))
    # distance to the nearest face PLANE of the box (faces extend only over the box, so also report outside-ness)
    d = np.stack([np.abs(p[:, 2]), np.abs(p[:, 2] - H), np.abs(p[:, 1]), np.abs(p[:, 1] - D),
                  np.abs(p[:, 0]), np.abs(p[:, 0] - W)], axis=1)
    inside = ((p[:, 0] > -0.02 * diag) & (p[:, 0] < W + 0.02 * diag) & (p[:, 1] > -0.02 * diag)
              & (p[:, 1] < D + 0.02 * diag) & (p[:, 2] > -0.02 * diag) & (p[:, 2] < H + 0.02 * diag))
    dmin = d.min(1) / diag
    out = {"n_points": int(len(p)),
           "frac_inside_room_bbox_(+-2%)": float(inside.mean()),
           "median_dist_to_nearest_face_frac_diag": float(np.median(dmin[inside])) if inside.any() else None,
           "p90_dist_to_nearest_face_frac_diag": float(np.percentile(dmin[inside], 90)) if inside.any() else None,
           "frac_within_1pct_diag": float(((dmin < 0.01) & inside).sum() / len(p)),
           "frac_within_3pct_diag": float(((dmin < 0.03) & inside).sum() / len(p))}
    surf = p[(dmin < 0.05) & inside]
    if len(surf) > 100:  # extents of the points that actually lie on a face -> aspect ratios
        ext = np.percentile(surf, 99, axis=0) - np.percentile(surf, 1, axis=0)
        out["extent_ratio_W_over_D"] = float(ext[0] / ext[1])
        out["extent_ratio_H_over_D"] = float(ext[2] / ext[1])
    out["gt_ratio_W_over_D"] = W / D
    out["gt_ratio_H_over_D"] = H / D
    return out


def score_planes(planes_est: Sequence[np.ndarray], sim, room: dict, ang_tol: float = 5.0, off_tol: float = 0.05) -> Dict:
    """``planes_est``: one (N,3) inlier cloud per product plane entity (est frame)."""
    s, R, t = sim
    W, D, H = room["width"], room["depth"], room["height"]
    diag = float(np.sqrt(W * W + D * D + H * H))
    faces = gt_faces(room)
    rows = []
    for pts in planes_est:
        if len(pts) < 30:
            continue
        n, c, rms = fit_plane(pts)
        n_gt = R @ n
        c_gt = s * R @ c + t
        cand = []
        for fi, (fn, fd) in enumerate(faces):
            ang = float(np.degrees(np.arccos(np.clip(abs(float(n_gt @ fn)), 0, 1))))
            cand.append((fi, ang, abs(float(c_gt @ fn) - fd) / diag, n))
        # parallel faces (floor/ceiling, opposite walls) tie on angle: among faces within tolerance take the nearest
        near = [c for c in cand if c[1] <= ang_tol]
        best = min(near, key=lambda c: c[2]) if near else min(cand, key=lambda c: c[1])
        rows.append({"face": best[0], "angle_deg": best[1], "offset_frac_diag": best[2], "n_pts": int(len(pts)),
                     "normal_est": n.tolist(), "centroid_est": c.tolist(), "rms_frac_diag": rms * s / diag})
    matched = [r for r in rows if r["angle_deg"] <= ang_tol and r["offset_frac_diag"] <= off_tol]
    out = {"n_planes": len(rows), "n_matching_a_true_face_(<=5deg,<=5%diag)": len(matched),
           "n_spurious": len(rows) - len(matched),
           "median_angle_deg_all": float(np.median([r["angle_deg"] for r in rows])) if rows else None,
           "median_angle_deg_matched": float(np.median([r["angle_deg"] for r in matched])) if matched else None}
    best_per_face: Dict[int, dict] = {}
    for r in matched:
        if r["face"] not in best_per_face or r["n_pts"] > best_per_face[r["face"]]["n_pts"]:
            best_per_face[r["face"]] = r
    out["true_faces_recovered"] = sorted(FACE_NAMES[i] for i in best_per_face)
    out["n_true_faces_recovered_of_6"] = len(best_per_face)
    # gauge-free shape from the product's own planes (est frame): angles between recovered faces, ratios of spans
    ideal = {}
    ang_dev = []
    ids = sorted(best_per_face)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = best_per_face[ids[i]], best_per_face[ids[j]]
            ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(a["normal_est"], b["normal_est"])), 0, 1))))
            same_axis = (ids[i] // 2) == (ids[j] // 2)
            ang_dev.append(abs(ang - (0.0 if same_axis else 90.0)))
    if ang_dev:
        out["recovered_face_pair_angle_error_deg"] = {"mean": float(np.mean(ang_dev)), "max": float(np.max(ang_dev)),
                                                      "n_pairs": len(ang_dev)}

    def span(pair):
        a, b = best_per_face.get(pair[0]), best_per_face.get(pair[1])
        if a is None or b is None:
            return None
        n = np.asarray(a["normal_est"]) + np.sign(np.dot(a["normal_est"], b["normal_est"])) * np.asarray(b["normal_est"])
        n /= np.linalg.norm(n)
        return abs(float((np.asarray(a["centroid_est"]) - np.asarray(b["centroid_est"])) @ n))

    sH, sD, sW = span((0, 1)), span((2, 3)), span((4, 5))
    if sD and sW:
        out["plane_span_ratio_W_over_D"] = sW / sD
    if sD and sH:
        out["plane_span_ratio_H_over_D"] = sH / sD
    out["gt_ratio_W_over_D"], out["gt_ratio_H_over_D"] = W / D, H / D
    return out
