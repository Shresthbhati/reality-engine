"""Geometry metrics, all in METRES, all numerically tested on synthetic cases with known answers (tests/test_benchmark_geometry_metrics.py).

Conventions: point sets are (N, 3) float arrays; meshes are ``(vertices (V,3), faces (F,3) int)``; poses are camera-to-world 4x4 matrices.
Nothing here guesses: functions that need independent ground truth return a ``BLOCKED`` dict instead of a number when it is absent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

Mesh = tuple[np.ndarray, np.ndarray]


def _pts(a: Any, name: str) -> np.ndarray:
    p = np.asarray(a, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] == 0 or not np.isfinite(p).all():
        raise ValueError(f"{name}: need a non-empty finite (N, 3) array, got shape {p.shape}")
    return p


# ---------------------------------------------------------------- point clouds
def nn_distances(src: Any, dst: Any) -> np.ndarray:
    """Distance from each point of ``src`` to its nearest point in ``dst``."""
    return cKDTree(_pts(dst, "dst")).query(_pts(src, "src"))[0]


def chamfer(pred: Any, gt: Any) -> dict[str, float]:
    """accuracy = mean d(pred->gt); completeness = mean d(gt->pred); chamfer = their mean (the usual symmetric L1 Chamfer)."""
    acc, comp = float(nn_distances(pred, gt).mean()), float(nn_distances(gt, pred).mean())
    return {"accuracy_m": acc, "completeness_m": comp, "chamfer_m": 0.5 * (acc + comp)}


def fscore(pred: Any, gt: Any, thresholds_m: Sequence[float] = (0.02, 0.05, 0.10, 0.25)) -> dict[str, dict[str, float]]:
    """Precision (pred within t of gt), recall (gt within t of pred) and F-score at each threshold."""
    dp, dg = nn_distances(pred, gt), nn_distances(gt, pred)
    out: dict[str, dict[str, float]] = {}
    for t in thresholds_m:
        if not (t > 0 and math.isfinite(t)):
            raise ValueError("thresholds must be positive and finite")
        p, r = float((dp <= t).mean()), float((dg <= t).mean())
        out[f"{t:g}"] = {"precision": p, "recall": r, "fscore": 0.0 if p + r == 0 else 2 * p * r / (p + r)}
    return out


# --------------------------------------------------------------------- meshes
def _closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point on each triangle (a,b,c: (F,3)) to one point ``p`` (3,). Ericson, Real-Time Collision Detection 5.1.5, vectorised."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v_in, w_in = vb / denom, vc / denom
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    # later assignment wins; priority (Ericson's early-return order) is A > B > AB > C > AC > BC > interior
    res = a + ab * v_in[:, None] + ac * w_in[:, None]
    res = np.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[:, None], b + (c - b) * t_bc[:, None], res)  # edge BC
    res = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], a + ac * t_ac[:, None], res)         # edge AC
    res = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, res)                                         # vertex C
    res = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], a + ab * t_ab[:, None], res)         # edge AB
    res = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, res)                                         # vertex B
    res = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, res)                                          # vertex A
    return res


def point_to_surface_distance(points: Any, mesh: Mesh) -> np.ndarray:
    """Exact distance from each point to the triangle mesh surface.

    ponytail: brute force O(N*F) (vectorised over faces); fine for benchmark-sized samples. Upgrade to a BVH for millions of samples.
    """
    p = _pts(points, "points")
    v, f = np.asarray(mesh[0], float), np.asarray(mesh[1], int)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("mesh needs (V,3) vertices and a non-empty (F,3) face array")
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    d = np.array([np.linalg.norm(_closest_on_triangles(q, a, b, c) - q, axis=1) for q in p])
    return np.where(np.isnan(d), np.inf, d).min(axis=1)


def sample_surface(mesh: Mesh, n: int, seed: int = 0) -> np.ndarray:
    """Area-weighted uniform surface samples (deterministic for a given seed)."""
    v, f = np.asarray(mesh[0], float), np.asarray(mesh[1], int)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        raise ValueError("degenerate mesh: zero total area")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u, w = np.where(flip, 1 - u, u), np.where(flip, 1 - w, w)
    return a[idx] + (b - a)[idx] * u[:, None] + (c - a)[idx] * w[:, None]


def mesh_accuracy_completeness(pred: Mesh, gt: Mesh, n_samples: int = 2000, thresholds_m: Sequence[float] = (0.02, 0.05, 0.10),
                               seed: int = 0) -> dict[str, Any]:
    """accuracy: pred-surface samples -> GT surface; completeness: GT-surface samples -> pred surface (both exact point-to-SURFACE)."""
    d_acc = point_to_surface_distance(sample_surface(pred, n_samples, seed), gt)
    d_comp = point_to_surface_distance(sample_surface(gt, n_samples, seed + 1), pred)
    out: dict[str, Any] = {"accuracy_m": float(d_acc.mean()), "completeness_m": float(d_comp.mean()), "n_samples": n_samples, "fscore": {}}
    for t in thresholds_m:
        p, r = float((d_acc <= t).mean()), float((d_comp <= t).mean())
        out["fscore"][f"{t:g}"] = {"precision": p, "recall": r, "fscore": 0.0 if p + r == 0 else 2 * p * r / (p + r)}
    return out


# ------------------------------------------------------------- camera poses
def umeyama_sim3(src: Any, dst: Any, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform with ``dst ~ s * R @ src + t`` (Umeyama 1991). Needs >= 3 non-collinear points."""
    x, y = _pts(src, "src"), _pts(dst, "dst")
    if x.shape != y.shape or len(x) < 3:
        raise ValueError("umeyama needs matching point sets of at least 3 points")
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    cov = yc.T @ xc / len(x)
    if np.linalg.matrix_rank(cov, tol=1e-9) < 2:
        raise ValueError("points are collinear: similarity alignment is under-determined")
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1
    r = u @ s_mat @ vt
    var = (xc ** 2).sum() / len(x)
    s = float((d * np.diag(s_mat)).sum() / var) if with_scale else 1.0
    return s, r, my - s * r @ mx


def _rot_angle_deg(r: np.ndarray) -> float:
    # atan2 form: acos((tr-1)/2) loses ~1e-6 deg of precision near identity
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2.0
    return math.degrees(math.atan2(float(np.linalg.norm(v)), (float(np.trace(r)) - 1.0) / 2.0))


def camera_registration(est: Mapping[str, np.ndarray], gt: Mapping[str, np.ndarray], *, with_scale: bool = True) -> dict[str, Any]:
    """Register estimated camera-to-world 4x4 poses to GT after a Sim(3) alignment of camera centres.

    Cameras the pipeline did not register count against ``registered_fraction``; errors are over registered cameras only.
    Returns ATE (RMSE of centre error after alignment, in the GT frame's metres), rotation error, and the recovered scale.
    """
    ids = [k for k in gt if k in est]
    out: dict[str, Any] = {"n_gt": len(gt), "n_registered": len(ids), "registered_fraction": len(ids) / len(gt) if gt else 0.0}
    if len(ids) < 3:
        return {**out, "status": "BLOCKED", "reason": f"{len(ids)} registered camera(s) with GT; a similarity alignment needs >= 3"}
    ce = np.array([np.asarray(est[k], float)[:3, 3] for k in ids])
    cg = np.array([np.asarray(gt[k], float)[:3, 3] for k in ids])
    try:
        s, r, t = umeyama_sim3(ce, cg, with_scale)
    except ValueError as exc:
        return {**out, "status": "BLOCKED", "reason": str(exc)}
    pos_err = np.linalg.norm(s * ce @ r.T + t - cg, axis=1)
    rot_err = [_rot_angle_deg(r @ np.asarray(est[k], float)[:3, :3] @ np.asarray(gt[k], float)[:3, :3].T) for k in ids]
    return {**out, "status": "OK", "scale": s, "ate_rmse_m": float(math.sqrt((pos_err ** 2).mean())), "ate_mean_m": float(pos_err.mean()),
            "ate_max_m": float(pos_err.max()), "rotation_err_mean_deg": float(np.mean(rot_err)), "rotation_err_max_deg": float(np.max(rot_err)),
            "alignment": {"s": s, "R": r.tolist(), "t": t.tolist()}}


def trajectory_errors(est_seq: Sequence[np.ndarray], gt_seq: Sequence[np.ndarray], delta: int = 1, *, with_scale: bool = True) -> dict[str, Any]:
    """ATE (after Sim3 alignment) and RPE over ``delta`` frames for index-matched pose sequences (camera-to-world 4x4)."""
    if len(est_seq) != len(gt_seq) or len(gt_seq) <= delta:
        raise ValueError("sequences must be index-matched and longer than delta")
    reg = camera_registration({str(i): p for i, p in enumerate(est_seq)}, {str(i): p for i, p in enumerate(gt_seq)}, with_scale=with_scale)
    if reg["status"] != "OK":
        return reg
    s = reg["scale"]
    scaled = [np.asarray(p, float).copy() for p in est_seq]
    for p in scaled:
        p[:3, 3] *= s
    tr, rr = [], []
    for i in range(len(gt_seq) - delta):
        g = np.linalg.inv(np.asarray(gt_seq[i], float)) @ np.asarray(gt_seq[i + delta], float)
        e = np.linalg.inv(scaled[i]) @ scaled[i + delta]
        err = np.linalg.inv(g) @ e
        tr.append(float(np.linalg.norm(err[:3, 3])))
        rr.append(_rot_angle_deg(err[:3, :3]))
    return {"status": "OK", "delta": delta, "ate_rmse_m": reg["ate_rmse_m"], "scale": s,
            "rpe_trans_rmse_m": float(math.sqrt(np.mean(np.square(tr)))), "rpe_rot_rmse_deg": float(math.sqrt(np.mean(np.square(rr))))}


# --------------------------------------------------------- structural elements
def _rect_distance(p: np.ndarray, rect: Mapping[str, Any]) -> np.ndarray:
    """Distance from points (N,3) to a planar rectangle {center,u,v,half_u,half_v} (u,v orthonormal)."""
    c, u, v = (np.asarray(rect[k], float) for k in ("center", "u", "v"))
    d = p - c
    a = np.clip(d @ u, -rect["half_u"], rect["half_u"])
    b = np.clip(d @ v, -rect["half_v"], rect["half_v"])
    return np.linalg.norm(d - np.outer(a, u) - np.outer(b, v), axis=1)


def _normal(rect: Mapping[str, Any]) -> np.ndarray:
    n = np.cross(np.asarray(rect["u"], float), np.asarray(rect["v"], float))
    return n / np.linalg.norm(n)


def _grid(rect: Mapping[str, Any], step: float) -> np.ndarray:
    nu, nv = max(1, math.ceil(2 * rect["half_u"] / step)), max(1, math.ceil(2 * rect["half_v"] / step))
    a = (np.arange(nu) + 0.5) / nu * 2 - 1
    b = (np.arange(nv) + 0.5) / nv * 2 - 1
    aa, bb = np.meshgrid(a * rect["half_u"], b * rect["half_v"])
    return np.asarray(rect["center"], float) + np.outer(aa.ravel(), rect["u"]) + np.outer(bb.ravel(), rect["v"])


def _covered(samples: np.ndarray, normal: np.ndarray, others: Sequence[Mapping[str, Any]], dist_tol: float, cos_tol: float) -> np.ndarray:
    hit = np.zeros(len(samples), bool)
    for o in others:
        if abs(float(_normal(o) @ normal)) >= cos_tol:
            hit |= _rect_distance(samples, o) <= dist_tol
    return hit


def structural_completeness(pred: Sequence[Mapping[str, Any]], gt: Sequence[Mapping[str, Any]], *, dist_tol_m: float = 0.10,
                            angle_tol_deg: float = 10.0, step_m: float = 0.05,
                            kinds: Sequence[str] = ("wall", "floor", "ceiling")) -> dict[str, Any]:
    """Per-kind AREA completeness: fraction of each GT planar element (rectangle) covered by a predicted rectangle of the SAME kind that is
    within ``dist_tol_m`` and whose normal is within ``angle_tol_deg`` (sign-agnostic). Also reports spurious area (pred not near any GT).
    Elements: {"kind","center","u","v","half_u","half_v"} with orthonormal in-plane axes u, v."""
    cos_tol = math.cos(math.radians(angle_tol_deg))
    out: dict[str, Any] = {}
    for kind in kinds:
        g_el = [e for e in gt if e["kind"] == kind]
        p_el = [e for e in pred if e["kind"] == kind]
        if not g_el:
            out[kind] = {"status": "BLOCKED", "reason": f"no ground-truth {kind} elements"}
            continue
        area_g = cov_g = area_p = spur = 0.0
        for e in g_el:
            s = _grid(e, step_m)
            w = 4 * e["half_u"] * e["half_v"] / len(s)
            area_g += w * len(s)
            cov_g += w * _covered(s, _normal(e), p_el, dist_tol_m, cos_tol).sum()
        for e in p_el:
            s = _grid(e, step_m)
            w = 4 * e["half_u"] * e["half_v"] / len(s)
            area_p += w * len(s)
            spur += w * (~_covered(s, _normal(e), g_el, dist_tol_m, cos_tol)).sum()
        out[kind] = {"status": "OK", "completeness": cov_g / area_g, "gt_area_m2": area_g, "pred_area_m2": area_p,
                     "spurious_area_fraction": spur / area_p if area_p else 0.0, "n_gt": len(g_el), "n_pred": len(p_el)}
    return out


def opening_completeness(pred: Sequence[Mapping[str, Any]], gt: Sequence[Mapping[str, Any]], *, center_tol_m: float = 0.15,
                         size_tol_frac: float = 0.25) -> dict[str, Any]:
    """Doors/windows matched greedily one-to-one by centre distance AND width/height within ``size_tol_frac``. Recall = completeness.
    Elements: {"kind": "door"|"window", "center": xyz, "width_m", "height_m"}."""
    if not gt:
        return {"status": "BLOCKED", "reason": "no ground-truth openings"}
    free, matched, w_err, h_err = list(range(len(pred))), 0, [], []
    for g in gt:
        best, bd = None, None
        for i in free:
            p = pred[i]
            d = float(np.linalg.norm(np.asarray(p["center"], float) - np.asarray(g["center"], float)))
            if (p["kind"] == g["kind"] and d <= center_tol_m and abs(p["width_m"] - g["width_m"]) <= size_tol_frac * g["width_m"]
                    and abs(p["height_m"] - g["height_m"]) <= size_tol_frac * g["height_m"] and (bd is None or d < bd)):
                best, bd = i, d
        if best is not None:
            free.remove(best)
            matched += 1
            w_err.append(abs(pred[best]["width_m"] - g["width_m"]))
            h_err.append(abs(pred[best]["height_m"] - g["height_m"]))
    return {"status": "OK", "recall": matched / len(gt), "precision": matched / len(pred) if pred else 0.0, "n_gt": len(gt), "n_pred": len(pred),
            "matched": matched, "width_mae_m": float(np.mean(w_err)) if w_err else None, "height_mae_m": float(np.mean(h_err)) if h_err else None}


# ---------------------------------------------------------------- dimensions
def dimension_errors(pred_m: Mapping[str, float], gt_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Dimension error ONLY against independent measurements (``provenance_class == "measured"`` with a finite uncertainty).

    GT items from fitted/estimated/synthetic sources are excluded and listed -- a number derived from a model is not a measurement.
    Items: {"name", "value_m", "uncertainty_m", "provenance_class"}.
    """
    used, excluded = [], []
    for it in gt_items:
        ok = (it.get("provenance_class") == "measured" and isinstance(it.get("uncertainty_m"), (int, float))
              and math.isfinite(it["uncertainty_m"]) and it.get("value_m", 0) > 0)
        if not ok:
            excluded.append({"name": it.get("name"), "reason": f"not an independent measurement (provenance_class={it.get('provenance_class')!r}, "
                                                               f"uncertainty_m={it.get('uncertainty_m')!r})"})
        elif it["name"] not in pred_m:
            excluded.append({"name": it["name"], "reason": "no predicted value for this dimension"})
        else:
            err = pred_m[it["name"]] - it["value_m"]
            used.append({"name": it["name"], "pred_m": pred_m[it["name"]], "gt_m": it["value_m"], "error_m": err, "abs_error_m": abs(err),
                         "rel_error": abs(err) / it["value_m"], "within_2_sigma": abs(err) <= 2 * it["uncertainty_m"]})
    if not used:
        return {"status": "BLOCKED", "reason": "no independent measured dimension with a matching prediction", "excluded": excluded}
    return {"status": "OK", "n": len(used), "mae_m": float(np.mean([u["abs_error_m"] for u in used])),
            "rmse_m": float(math.sqrt(np.mean([u["error_m"] ** 2 for u in used]))), "mean_rel_error": float(np.mean([u["rel_error"] for u in used])),
            "items": used, "excluded": excluded}
