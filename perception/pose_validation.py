"""Camera-pose validation: robust ATE / RPE and reference-free sanity checks.

Contract (docs/engineering/camera_tracking_report.md)
  input   ``{name: (R, t)}``, R 3x3 world->camera, ``x_cam = R @ X + t`` (same convention as
          ``perception.camera_pose_graph``). Units are whatever the producer uses; the
          ``scale_state`` arguments say whether those units are metres ("metric") or arbitrary
          ("relative"). Nothing here assumes metric scale.
  output  plain dicts, every number measured; ``refused`` + ``reason`` instead of a made-up score.
  gauge   the world frame of an estimate is arbitrary, so error is measured AFTER a *robust*
          alignment. Scale is solved for unless BOTH sides are metric (then a rigid SE(3) fit is used
          and a scale mismatch shows up as error instead of being absorbed).

Why robust: ``reconstruction.evaluation.umeyama_sim3`` is plain least squares, so a single wrong camera
(e.g. the 113 deg outlier seen in a South Building run) drags the whole alignment and every *good*
camera then looks bad. Here the alignment is RANSAC + inlier refit and outliers are reported by name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

Pose = tuple[Any, Any]  # (R 3x3 world->camera, t 3)


def _arr(poses: Mapping[str, Pose], names: Sequence[str]):
    R = np.stack([np.asarray(poses[n][0], dtype=np.float64) for n in names])
    t = np.stack([np.asarray(poses[n][1], dtype=np.float64).reshape(3) for n in names])
    return R, t


def camera_centers(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-frame centres ``C = -R^T t`` for stacked poses."""
    return -np.einsum("nji,nj->ni", R, t)


def _angle_deg(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _fit(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Umeyama: dst ~ s * R @ src + t. Returns (s, R, t) or None when degenerate."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var = float((sc ** 2).sum() / len(src))
    if var < 1e-18:
        return None
    U, D, Vt = np.linalg.svd(dc.T @ sc / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(D @ np.diag(S)) / var if with_scale else 1.0
    if s <= 0:
        return None
    return s, R, mu_d - s * (R @ mu_s)


def robust_align(src: np.ndarray, dst: np.ndarray, *, with_scale: bool, inlier_frac: float = 0.05,
                 iters: int = 400, seed: int = 0):
    """RANSAC (3-point) + inlier refit. ``inlier_frac`` is relative to the reference spread
    (median distance of dst to its centroid), so the threshold carries no unit assumption.
    Returns dict(s, R, t, inliers, threshold) or None."""
    n = len(src)
    if n < 3:
        return None
    thr = inlier_frac * float(np.median(np.linalg.norm(dst - dst.mean(0), axis=1)))
    if thr <= 0:
        return None
    rng = np.random.default_rng(seed)
    best, best_in = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        m = _fit(src[idx], dst[idx], with_scale)
        if m is None:
            continue
        s, R, t = m
        inl = np.linalg.norm(s * src @ R.T + t - dst, axis=1) < thr
        if inl.sum() > best_in:
            best, best_in = inl, int(inl.sum())
    if best is None or best_in < 3:
        return None
    for _ in range(5):  # refit on inliers until stable
        m = _fit(src[best], dst[best], with_scale)
        if m is None:
            return None
        s, R, t = m
        new = np.linalg.norm(s * src @ R.T + t - dst, axis=1) < thr
        if new.sum() < 3 or (new == best).all():
            break
        best = new
    return {"s": s, "R": R, "t": t, "inliers": best, "threshold": thr}


def _stats(x: np.ndarray, m: np.ndarray):
    x = x[m]
    if len(x) == 0:
        return None
    return {"rmse": float(np.sqrt((x ** 2).mean())), "median": float(np.median(x)), "max": float(x.max())}


def evaluate_trajectory(est: Mapping[str, Pose], ref: Mapping[str, Pose], *, est_scale_state: str,
                        ref_scale_state: str, order: Sequence[str] | None = None, rpe_delta: int = 1,
                        outlier_rot_deg: float = 10.0, min_cameras: int = 4) -> dict:
    """ATE (centres) + rotation error after robust alignment, and alignment-free RPE.

    ``order``: frame sequence for RPE (video/trajectory). Omit for unordered photo sets (RPE skipped).
    A camera is an *outlier* when its aligned rotation error exceeds ``outlier_rot_deg`` or its centre
    is outside the alignment inlier threshold.
    """
    for label, v in (("est_scale_state", est_scale_state), ("ref_scale_state", ref_scale_state)):
        if v not in ("metric", "relative"):
            raise ValueError(f"{label} must be 'metric' or 'relative', got {v!r}")
    names = sorted(set(est) & set(ref))
    if len(names) < min_cameras:
        return {"refused": True, "reason": f"only {len(names)} common cameras (< {min_cameras})"}
    Re, te = _arr(est, names)
    Rr, tr = _arr(ref, names)
    Ce, Cr = camera_centers(Re, te), camera_centers(Rr, tr)
    with_scale = not (est_scale_state == "metric" and ref_scale_state == "metric")
    al = robust_align(Ce, Cr, with_scale=with_scale)
    if al is None:
        return {"refused": True, "reason": "degenerate geometry: robust alignment failed",
                "common_cameras": len(names)}
    s, Ra, ta, inl = al["s"], al["R"], al["t"], al["inliers"]
    c_err = np.linalg.norm(s * Ce @ Ra.T + ta - Cr, axis=1)
    # gauge maps est c2w (Re^T) to Ra @ Re^T; error rotation = (Ra Re^T)^T Rr^T = Re Ra^T Rr^T
    rot_err = _angle_deg(Re @ Ra.T @ np.swapaxes(Rr, 1, 2))
    out = (~inl) | (rot_err > outlier_rot_deg)
    everyone = np.ones_like(inl)
    res: dict[str, Any] = {
        "refused": False, "common_cameras": len(names), "scale_state": {"est": est_scale_state, "ref": ref_scale_state},
        "alignment": "sim3" if with_scale else "se3", "gauge_scale": s if with_scale else None,
        "alignment_inliers": int(inl.sum()), "alignment_threshold_ref_units": al["threshold"],
        "ate_translation_ref_units": {"inliers": _stats(c_err, inl), "all": _stats(c_err, everyone)},
        "rotation_error_deg": {"inliers": _stats(rot_err, ~out), "all": _stats(rot_err, everyone)},
        "outliers": [{"name": n, "center_err": float(c_err[i]), "rot_err_deg": float(rot_err[i])}
                     for i, n in enumerate(names) if out[i]],
        "notes": [],
    }
    if ref_scale_state == "relative":
        res["notes"].append("reference scale is arbitrary: ATE is in reference units, not metres")
    if order is not None:
        res["rpe"] = _rpe(est, ref, [n for n in order if n in est and n in ref], rpe_delta, s)
    return res


def _rpe(est, ref, order, delta, scale):
    if len(order) <= delta:
        return {"refused": True, "reason": "sequence shorter than delta"}
    Re, te = _arr(est, order)
    Rr, tr = _arr(ref, order)
    rr, tt = [], []
    for i in range(len(order) - delta):
        j = i + delta
        # relative motion cam_i -> cam_j : R_ij = R_j R_i^T, t_ij = t_j - R_ij t_i
        Rij_e, Rij_r = Re[j] @ Re[i].T, Rr[j] @ Rr[i].T
        tij_e, tij_r = te[j] - Rij_e @ te[i], tr[j] - Rij_r @ tr[i]
        rr.append(float(_angle_deg(Rij_e @ Rij_r.T)))
        tt.append(float(np.linalg.norm(scale * tij_e - tij_r)))
    m = np.ones(len(rr), bool)
    return {"refused": False, "delta": delta, "pairs": len(rr), "rot_deg": _stats(np.array(rr), m),
            "trans_ref_units": _stats(np.array(tt), m)}


def validate_poses(poses: Mapping[str, Pose], *, edges: Sequence[tuple[str, str]] | None = None,
                   order: Sequence[str] | None = None, step_mad_k: float = 6.0,
                   min_step_deg: float = 15.0) -> dict:
    """Reference-free checks. Reports, never repairs.

    * improper / non-finite rotations, coincident centres
    * connected components of the supplied covisibility/match ``edges`` (disconnected pieces cannot
      share a gauge: their relative scale and placement are undefined)
    * with ``order``: consecutive-frame rotation steps that are robust-MAD outliers. An interior frame
      with a big step on BOTH sides is a pose failure; a lone big step is a trajectory discontinuity.
    """
    names = sorted(poses)
    issues: list[dict] = []
    ok = []
    for n in names:
        R, t = np.asarray(poses[n][0], float), np.asarray(poses[n][1], float)
        if R.shape != (3, 3) or not np.isfinite(R).all() or not np.isfinite(t).all():
            issues.append({"name": n, "issue": "non_finite_pose"})
        elif abs(np.linalg.det(R) - 1) > 1e-3 or np.abs(R @ R.T - np.eye(3)).max() > 1e-3:
            issues.append({"name": n, "issue": "not_a_rotation"})
        else:
            ok.append(n)
    if len(ok) >= 2:
        R, t = _arr(poses, ok)
        C = camera_centers(R, t)
        d = np.linalg.norm(C[:, None] - C[None], axis=-1) + np.eye(len(ok)) * 1e9
        span = float(np.linalg.norm(C - C.mean(0), axis=1).max()) or 1.0
        for i, j in zip(*np.where(np.triu(d < 1e-6 * span, 1))):
            issues.append({"name": ok[i], "issue": "coincident_with", "other": ok[j]})
    comps = None
    if edges is not None:
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in edges:
            if a in parent and b in parent:
                parent[find(a)] = find(b)
        groups: dict[str, list[str]] = {}
        for n in names:
            groups.setdefault(find(n), []).append(n)
        comps = sorted(groups.values(), key=len, reverse=True)
        if len(comps) > 1:
            issues.append({"issue": "disconnected_components", "sizes": [len(c) for c in comps]})
    steps = None
    if order is not None:
        seq = [n for n in order if n in ok]
        if len(seq) >= 4:
            R, _ = _arr(poses, seq)
            ang = _angle_deg(R[1:] @ np.swapaxes(R[:-1], 1, 2))
            med = float(np.median(ang))
            mad = float(np.median(np.abs(ang - med))) * 1.4826
            # 3x median floor: uniform motion has MAD ~ 0, so MAD alone would flag ordinary steps
            thr = max(min_step_deg, 3.0 * med, med + step_mad_k * mad)
            steps = {"median_deg": med, "mad_deg": mad, "threshold_deg": thr}
            big = ang > thr
            explained = set()
            for k in range(1, len(seq) - 1):
                if big[k - 1] and big[k]:
                    issues.append({"name": seq[k], "issue": "trajectory_rotation_jump",
                                   "step_deg": float(max(ang[k - 1], ang[k]))})
                    explained.update((k - 1, k))
            for k in np.where(big)[0]:
                if int(k) not in explained:
                    issues.append({"issue": "trajectory_discontinuity", "between": [seq[k], seq[k + 1]],
                                   "step_deg": float(ang[k])})
    return {"cameras": len(names), "valid": len(ok), "issues": issues,
            "components": None if comps is None else [len(c) for c in comps], "rotation_steps": steps,
            "scale_state_note": "validation is unit-free; it never infers metric scale"}


def rotation_about_axis(deg: float, axis=(0, 1, 0)) -> np.ndarray:
    """Test/benchmark helper: Rodrigues rotation matrix."""
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = math.radians(deg)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
