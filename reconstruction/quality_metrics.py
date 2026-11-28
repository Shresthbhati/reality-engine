"""Reconstruction quality metrics: one registry, every metric with a formula, inputs, limitations and a test.

Design rules (see tests/test_reconstruction_quality_metrics.py, which enforces them):

* A metric that CANNOT be computed from its inputs returns ``{"value": None, "refused": <reason>}`` -- never a
  fabricated number and never a silently-defaulted one.
* A metric that needs a reference frame refuses unless the caller states the estimate is already in that frame.
* Nothing here converts relative / gauge coordinates to metres. ``scale_validity`` is the only place that decides
  whether a metres claim is allowed, and it is conservative.
* "More points" or "more triangles" is never a score: completeness is measured against a reference, mesh quality
  against structural invariants.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricSpec:
    name: str
    formula: str
    inputs: str
    limitations: str


METRICS: dict[str, MetricSpec] = {m.name: m for m in (
    MetricSpec("registration_rate", "n_registered / n_input",
               "count of images placed in the model, count of images supplied",
               "counts placed images only; a placed image can still have a wrong pose (see pose_graph_cycle_error)"),
    MetricSpec("reprojection_error", "median / mean / p95 / rmse of per-observation pixel residuals",
               "iterable of reprojection errors in pixels",
               "measures self-consistency of the optimisation, not accuracy against the world; low values coexist with gauge or scale error"),
    MetricSpec("pose_graph_cycle_error", "for each camera triple (a,b,c): angle(R_bc @ R_ab @ R_ac^T); report mean/p95/max",
               "pairwise relative rotations R_ab with x_b ~ R_ab x_a",
               "rotation only (translation cycles need scale); needs >= 1 fully connected triple, else refused"),
    MetricSpec("scale_validity", "metres claim allowed iff state == metric AND a validated calibration exists",
               "scale_state string, calibration status string",
               "does not verify the calibration itself, only whether its recorded status permits a metres claim"),
    MetricSpec("point_cloud_completeness", "recall = |{r in ref: min_e ||e-r|| <= tau}| / |ref|; precision symmetric; F1",
               "estimated points, reference points, tau (same units), same_frame=True",
               "meaningful only when estimate and reference share one frame and unit; says nothing about areas the reference misses"),
    MetricSpec("depth_consistency", "among non-occluded samples: fraction with |z_warp - d_b| / d_b <= rel_tol",
               "depth maps A,B (same units), intrinsics K, relative pose T_ab, rel_tol",
               "occlusion is detected only by z-buffer ordering; invalid (<=0 / non-finite) pixels are excluded, not counted"),
    MetricSpec("mesh_integrity", "counts: degenerate, duplicate, non-manifold, boundary edges, winding conflicts, components",
               "triangle mesh (vertices, faces)",
               "structural only: a clean mesh can still be geometrically wrong; self-intersection is a bbox proxy"),
    MetricSpec("surface_coverage", "fraction of reference points within tau of an area-weighted sample of the mesh surface",
               "mesh, reference points, tau, same_frame=True",
               "sampled (seeded, deterministic); resolution-limited by n_samples; reference must be in the mesh's frame/units"),
    MetricSpec("uncertainty_calibration", "coverage = fraction of |error| <= 1.96 * sigma (expected 0.95 if calibrated); rms z-score",
               "per-point error magnitudes along one axis, per-point predicted sigma (same units)",
               "assumes Gaussian errors; over/under-confidence is reported, not corrected"),
    MetricSpec("cross_view_consistency", "fraction of tracks whose worst per-view reprojection error <= tol_px",
               "per-track list of per-view pixel errors, tol_px",
               "tracks with a single observation are excluded (nothing to be consistent with)"),
)}


def _refuse(name: str, reason: str) -> dict[str, Any]:
    return {"metric": name, "value": None, "refused": reason}


# 1 ------------------------------------------------------------------------------------------------------------
def registration_rate(n_registered: int, n_input: int) -> dict[str, Any]:
    if n_input <= 0:
        return _refuse("registration_rate", "no input images")
    if not 0 <= n_registered <= n_input:
        return _refuse("registration_rate", f"n_registered={n_registered} outside [0, {n_input}]")
    return {"metric": "registration_rate", "value": n_registered / n_input, "n_registered": n_registered, "n_input": n_input}


# 2 ------------------------------------------------------------------------------------------------------------
def reprojection_error(errors_px: Iterable[float]) -> dict[str, Any]:
    e = np.asarray([float(x) for x in errors_px], dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return _refuse("reprojection_error", "no finite residuals")
    return {"metric": "reprojection_error", "value": float(np.median(e)), "median_px": float(np.median(e)),
            "mean_px": float(e.mean()), "p95_px": float(np.percentile(e, 95)),
            "rmse_px": float(math.sqrt(float((e ** 2).mean()))), "n": int(e.size)}


# 3 ------------------------------------------------------------------------------------------------------------
def pose_graph_cycle_error(nodes: Sequence[str], edges: Iterable[tuple[str, str, Any]]) -> dict[str, Any]:
    from perception.camera_pose_graph import cycle_residuals

    rows = cycle_residuals(nodes, edges)
    if not rows:
        return _refuse("pose_graph_cycle_error", "no fully connected camera triple")
    r = np.asarray([row["residual_deg"] for row in rows], dtype=np.float64)
    return {"metric": "pose_graph_cycle_error", "value": float(r.mean()), "mean_deg": float(r.mean()),
            "p95_deg": float(np.percentile(r, 95)), "max_deg": float(r.max()), "n_triples": int(r.size),
            "worst": rows[0]}


# 4 ------------------------------------------------------------------------------------------------------------
_METRIC_STATES = {"metric", "metric_anchored", "physically_anchored"}


def scale_validity(scale_state: str | None, calibration_status: str | None) -> dict[str, Any]:
    state = str(scale_state or "unknown").lower()
    status = str(calibration_status or "none").lower()
    if state in _METRIC_STATES and status == "validated":
        kind, allowed = "metric_validated", True
    elif state in _METRIC_STATES and status == "unvalidated":
        kind, allowed = "metric_unvalidated", False
    elif status == "estimated_prior":
        kind, allowed = "estimated", False
    elif state in ("unknown", "none", ""):
        kind, allowed = "unknown", False
    else:
        kind, allowed = "relative", False
    return {"metric": "scale_validity", "value": kind, "metres_claim_allowed": allowed,
            "scale_state": state, "calibration_status": status}


# 5 ------------------------------------------------------------------------------------------------------------
def point_cloud_completeness(est: Any, ref: Any, tau: float, same_frame: bool = False) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    if not same_frame:
        return _refuse("point_cloud_completeness", "estimate and reference not declared to share one frame/unit")
    e, r = np.asarray(est, dtype=np.float64).reshape(-1, 3), np.asarray(ref, dtype=np.float64).reshape(-1, 3)
    if len(e) == 0 or len(r) == 0 or tau <= 0:
        return _refuse("point_cloud_completeness", "empty point set or non-positive tau")
    recall = float((cKDTree(e).query(r, k=1)[0] <= tau).mean())
    precision = float((cKDTree(r).query(e, k=1)[0] <= tau).mean())
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    return {"metric": "point_cloud_completeness", "value": recall, "recall": recall, "precision": precision, "f1": f1,
            "tau": tau, "n_est": int(len(e)), "n_ref": int(len(r))}


# 6 ------------------------------------------------------------------------------------------------------------
def depth_consistency(depth_a: Any, depth_b: Any, K: Any, T_ab: Any, rel_tol: float = 0.05,
                      stride: int = 4) -> dict[str, Any]:
    """Warp A's depth into B (``T_ab`` maps A-camera coords to B-camera coords) and compare with B's depth."""
    da, db = np.asarray(depth_a, dtype=np.float64), np.asarray(depth_b, dtype=np.float64)
    K, T = np.asarray(K, dtype=np.float64), np.asarray(T_ab, dtype=np.float64)
    h, w = da.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = da[vs, us]
    ok = np.isfinite(z) & (z > 0)
    if not ok.any():
        return _refuse("depth_consistency", "no valid pixels in depth map A")
    u, v, z = us[ok].astype(np.float64), vs[ok].astype(np.float64), z[ok]
    pa = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    pb = pa @ T[:3, :3].T + T[:3, 3]
    zb = pb[:, 2]
    front = zb > 1e-9
    ub = np.round(K[0, 0] * pb[:, 0] / np.where(front, zb, 1) + K[0, 2]).astype(int)
    vb = np.round(K[1, 1] * pb[:, 1] / np.where(front, zb, 1) + K[1, 2]).astype(int)
    inside = front & (ub >= 0) & (ub < w) & (vb >= 0) & (vb < db.shape[0])
    d_obs = np.full(zb.shape, np.nan)
    d_obs[inside] = db[vb[inside], ub[inside]]
    valid = inside & np.isfinite(d_obs) & (d_obs > 0)
    if not valid.any():
        return _refuse("depth_consistency", "no overlapping valid pixels")
    rel = (zb[valid] - d_obs[valid]) / d_obs[valid]
    occluded = rel > rel_tol          # A's point is behind what B sees: B sees a nearer surface -> not a contradiction
    measured = ~occluded
    if not measured.any():
        return _refuse("depth_consistency", "every overlapping sample is occluded")
    consistent = np.abs(rel[measured]) <= rel_tol
    return {"metric": "depth_consistency", "value": float(consistent.mean()), "n_measured": int(measured.sum()),
            "n_occluded_excluded": int(occluded.sum()), "median_abs_rel_error": float(np.median(np.abs(rel[measured]))),
            "rel_tol": rel_tol}


# 7 ------------------------------------------------------------------------------------------------------------
def mesh_integrity(mesh: Any) -> dict[str, Any]:
    from reconstruction.meshing.validation import validate_mesh

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if len(v) == 0 or len(f) == 0:
        return _refuse("mesh_integrity", "empty mesh")
    if f.min() < 0 or f.max() >= len(v):
        return _refuse("mesh_integrity", "face index out of range")
    rep = validate_mesh(mesh)
    dup = int(len(f) - len({tuple(sorted(map(int, t))) for t in f}))
    # Winding: in a consistently oriented manifold every shared edge is traversed in OPPOSITE directions by its two faces.
    directed: dict[tuple[int, int], int] = {}
    for a, b, c in f:
        for e in ((a, b), (b, c), (c, a)):
            directed[(int(e[0]), int(e[1]))] = directed.get((int(e[0]), int(e[1])), 0) + 1
    # An edge used twice in the SAME direction means two faces disagree on orientation.
    conflicts = sum(1 for n in directed.values() if n > 1)
    return {"metric": "mesh_integrity", "value": rep.n_non_manifold_edges + conflicts + dup + rep.n_degenerate_triangles,
            "n_triangles": int(len(f)), "n_degenerate": rep.n_degenerate_triangles, "n_duplicate_triangles": dup,
            "n_non_manifold_edges": rep.n_non_manifold_edges, "n_boundary_edges": rep.n_boundary_edges,
            "n_winding_conflicts": int(conflicts), "n_components": rep.n_connected_components,
            "has_holes": rep.has_holes, "valid": bool(rep.n_non_manifold_edges + conflicts + dup + rep.n_degenerate_triangles == 0),
            "units": getattr(mesh, "units", None), "frame": getattr(mesh, "frame", None)}


# 8 ------------------------------------------------------------------------------------------------------------
def surface_coverage(mesh: Any, ref: Any, tau: float, same_frame: bool = False, n_samples: int = 20000,
                     seed: int = 0) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    if not same_frame:
        return _refuse("surface_coverage", "mesh and reference not declared to share one frame/unit")
    v, f = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    r = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0 or len(r) == 0 or tau <= 0:
        return _refuse("surface_coverage", "empty mesh/reference or non-positive tau")
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return _refuse("surface_coverage", "zero-area mesh")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), size=n_samples, p=area / area.sum())
    r1, r2 = rng.random(n_samples), rng.random(n_samples)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    t = tri[idx]
    samples = t[:, 0] + r1[:, None] * (t[:, 1] - t[:, 0]) + r2[:, None] * (t[:, 2] - t[:, 0])
    cov = float((cKDTree(samples).query(r, k=1)[0] <= tau).mean())
    return {"metric": "surface_coverage", "value": cov, "tau": tau, "n_samples": n_samples, "n_ref": int(len(r))}


# 9 ------------------------------------------------------------------------------------------------------------
def uncertainty_calibration(errors: Iterable[float], sigmas: Iterable[float]) -> dict[str, Any]:
    e, s = np.abs(np.asarray(list(errors), dtype=np.float64)), np.asarray(list(sigmas), dtype=np.float64)
    if e.shape != s.shape or e.size == 0:
        return _refuse("uncertainty_calibration", "errors and sigmas must be non-empty and equal length")
    ok = np.isfinite(e) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return _refuse("uncertainty_calibration", "no finite positive sigma")
    z = e[ok] / s[ok]
    cov = float((z <= 1.96).mean())
    return {"metric": "uncertainty_calibration", "value": cov, "coverage_95": cov, "expected": 0.95,
            "rms_z": float(math.sqrt(float((z ** 2).mean()))), "n": int(ok.sum()),
            "verdict": "overconfident" if cov < 0.90 else "underconfident" if cov > 0.99 else "calibrated"}


# 10 -----------------------------------------------------------------------------------------------------------
def cross_view_consistency(track_errors_px: Sequence[Sequence[float]], tol_px: float = 2.0) -> dict[str, Any]:
    multi = [np.asarray(t, dtype=np.float64) for t in track_errors_px if len(t) >= 2]
    if not multi:
        return _refuse("cross_view_consistency", "no track observed in >= 2 views")
    good = sum(1 for t in multi if np.isfinite(t).all() and t.max() <= tol_px)
    return {"metric": "cross_view_consistency", "value": good / len(multi), "n_tracks": len(multi), "tol_px": tol_px}


def all_metric_names() -> Mapping[str, MetricSpec]:
    return METRICS
