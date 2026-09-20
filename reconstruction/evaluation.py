"""Absolute-pose fidelity evaluation (P1 RECONSTRUCTION FIDELITY).

An incremental SfM run recovers the scene only up to an arbitrary
gauge: its global rotation, scale, and translation are chosen by the
mapper (and by where the first registered image happened to sit), not
by the world. Comparing reconstructed quaternions DIRECTLY against a
reference model therefore measures mostly the gauge difference, not
fidelity -- the recorded 8.94 deg "median rotation disagreement" in
datasets/south_building/runs is dominated by gauge, not per-camera
error.

This module is the canonical absolute-pose evaluator: it aligns the
estimated model to the reference with closed-form least-squares
transforms (Horn quaternion rotation on orientations; Umeyama Sim(3)
on camera centers) and only THEN measures per-camera disagreement.

Measured-property contract (enforced by tests/test_pose_evaluation.py):

  P1 gauge invariance  a known rigid Sim(3) transform between estimate
                       and GT evaluates to ~zero error.
  P2 noise honesty     a known injected perturbation is measured back
                       at the injected magnitude.
  P3 refusal           fewer than 3 common cameras cannot define a
                       frame -> REFUSED, never guessed.
  P4 determinism       identical inputs -> identical numbers.

Conventions (must match reconstruction.backend.interface):
  - estimated poses carry CAMERA->WORLD quaternions (w,x,y,z) and
    camera centers (position).
  - reference (COLMAP images.txt) poses are WORLD->CAMERA
    (qw,qx,qy,qz,tx,ty,tz); they are conjugated + converted to centers
    here so both sides live in one convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from reconstruction.backend.interface import ReconstructedCameraPose

Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Quaternion / matrix primitives (deterministic, closed-form)
# ---------------------------------------------------------------------------


def quat_to_matrix(q: Sequence[float]) -> List[List[float]]:
    """(w,x,y,z) -> 3x3 rotation matrix (list of row lists)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero-length quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _matrix_to_quat(R: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """3x3 rotation matrix -> (w,x,y,z); Shepperd's method."""
    trace = R[0][0] + R[1][1] + R[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2][1] - R[1][2]) * s
        y = (R[0][2] - R[2][0]) * s
        z = (R[1][0] - R[0][1]) * s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def quat_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Geodesic angle between two (w,x,y,z) quaternions, degrees."""
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def horn_rotation(src: Sequence[Vec3], dst: Sequence[Vec3]) -> List[List[float]]:
    """Closed-form least-squares rotation aligning src onto dst
    (Horn 1987 quaternion method, eigen decomposition of the
    correlation matrix). Exact when a true rotation exists."""
    if len(src) != len(dst):
        raise ValueError("paired point counts differ")
    if len(src) < 2:
        raise ValueError("rotation needs at least 2 points")
    src_m = np.asarray(src, dtype=np.float64)
    dst_m = np.asarray(dst, dtype=np.float64)
    cs = src_m.mean(axis=0)
    cd = dst_m.mean(axis=0)
    # Horn's correlation: S_ab = sum(src_a * dst_b) (l_a r_b in the paper).
    S = (src_m - cs).T @ (dst_m - cd)  # 3x3
    Sxx, Sxy, Sxz = S[0]
    Syx, Syy, Syz = S[1]
    Szx, Szy, Szz = S[2]
    N = np.array(
        [
            [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
            [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
            [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
            [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
        ]
    )
    evals, evecs = np.linalg.eigh(N)
    q = evecs[:, int(np.argmax(evals))]  # (w,x,y,z): Horn's q0 is the scalar part
    qw, qx, qy, qz = (float(v) for v in q)
    return quat_to_matrix((qw, qx, qy, qz))


def umeyama_sim3(
    src: Sequence[Vec3], dst: Sequence[Vec3]
) -> Tuple[List[List[float]], Vec3, float]:
    """Closed-form least-squares similarity (R, t, s) mapping src onto
    dst (Umeyama 1991). Returns (R, t, scale)."""
    if len(src) != len(dst):
        raise ValueError("paired point counts differ")
    if len(src) < 2:
        raise ValueError("similarity needs at least 2 points")
    src_m = np.asarray(src, dtype=np.float64)
    dst_m = np.asarray(dst, dtype=np.float64)
    mu_s = src_m.mean(axis=0)
    mu_d = dst_m.mean(axis=0)
    src_c = src_m - mu_s
    dst_c = dst_m - mu_d
    var_s = float((src_c ** 2).sum() / len(src))
    if var_s <= 0.0:
        raise ValueError("degenerate source configuration (zero variance)")
    H = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    scale = float(D @ np.diag(S)) / var_s
    if scale <= 0.0:
        raise ValueError("degenerate correspondence (non-positive scale)")
    t = mu_d - scale * (R @ mu_s)
    return [list(row) for row in R], (float(t[0]), float(t[1]), float(t[2])), scale


# ---------------------------------------------------------------------------
# Absolute-pose evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseEvaluation:
    """Measured disagreement between an estimated model and a reference
    model, AFTER least-squares gauge alignment. Every number is
    measured; none is a guess."""

    common_cameras: int
    rotation_deg_median: float
    rotation_deg_max: float
    rotation_deg_mean: float
    center_error_median: float  # meters, in GT units, after Sim(3) alignment
    center_error_max: float
    center_error_mean: float
    scale_ratio: float  # estimated_scene / reference_scene (>1: est too large)
    aligned: bool  # True when >= 3 common cameras defined the alignment
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "common_cameras": self.common_cameras,
            "rotation_deg_median": round(self.rotation_deg_median, 4),
            "rotation_deg_max": round(self.rotation_deg_max, 4),
            "rotation_deg_mean": round(self.rotation_deg_mean, 4),
            "center_error_median": round(self.center_error_median, 6),
            "center_error_max": round(self.center_error_max, 6),
            "center_error_mean": round(self.center_error_mean, 6),
            "scale_ratio": round(self.scale_ratio, 6),
            "aligned": self.aligned,
            "notes": list(self.notes),
        }


def _gt_to_c2w(
    gt_images: Dict[str, Sequence[float]],
) -> Dict[str, Tuple[Tuple[float, float, float, float], Vec3]]:
    """COLMAP images.txt rows (world->cam) -> (camera->world quat,
    camera center)."""
    out: Dict[str, Tuple[Tuple[float, float, float, float], Vec3]] = {}
    for name, (qw, qx, qy, qz, tx, ty, tz) in gt_images.items():
        q_c2w = (qw, -qx, -qy, -qz)  # conjugate
        R = quat_to_matrix(q_c2w)
        t = (tx, ty, tz)
        center = tuple(-sum(R[r][c] * t[c] for c in range(3)) for r in range(3))
        out[name] = (q_c2w, center)
    return out


def evaluate_point_cloud(
    points,
    gt_points: Dict[int, Vec3],
    transform: Tuple[List[List[float]], Vec3, float],
    max_points: int = 20000,
) -> dict:
    """Geometric fidelity: mean/median one-way distance from estimated
    points to the reference surface, measured AFTER applying the
    gauge transform recovered by evaluate_absolute_poses (one rigid
    gauge explains orientations, centers AND points -- re-deriving a
    separate transform here would silently measure a second alignment
    error, not fidelity).

    `points` are reconstruction.backend.interface.ReconstructedPoint
    (position, track_id); `gt_points` maps COLMAP point3D id -> xyz.
    Distances via scipy cKDTree (the repo's established kNN tool).

    Honesty rules:
      - one-way est->GT distance only (no symmetric claim: a sparse
        estimate over dense GT says nothing about GT coverage);
      - the estimated model's density is recorded next to the distance
        so the number is interpretable;
      - refuses (no fabricated distance) when either side is empty.
    """
    from scipy.spatial import cKDTree

    if not points or not gt_points:
        return {"refused": True, "reason": "no points on one side"}
    R, t, s = transform
    R_arr = np.asarray(R, dtype=np.float64)
    est = np.asarray([p.position for p in points], dtype=np.float64)
    if max_points and len(est) > max_points:
        # Deterministic stride subsample -- bounded work, same result.
        stride = int(math.ceil(len(est) / max_points))
        est = est[::stride]
    mapped = (s * (est @ R_arr.T)) + np.asarray(t, dtype=np.float64)
    gt_arr = np.asarray([gt_points[k] for k in sorted(gt_points)], dtype=np.float64)
    d, _ = cKDTree(gt_arr).query(mapped, k=1)
    d_sorted = np.sort(d)
    n = len(d)
    return {
        "refused": False,
        "measured_points": int(n),
        "reference_points": len(gt_arr),
        "dist_mean": float(d.mean()),
        "dist_median": float(d_sorted[n // 2]),
        "dist_p90": float(d_sorted[int(0.9 * (n - 1))]),
        "dist_max": float(d[-1]),
        "est_points_total": len(points),
    }


def evaluate_absolute_poses(
    poses: List[ReconstructedCameraPose],
    gt_images: Dict[str, Sequence[float]],
    min_cameras: int = 3,
) -> dict:
    """Evaluate estimated absolute poses against a reference model.

    Returns a dict with `refused` set when evaluation is not possible
    (too few common cameras, degenerate geometry); the measured report
    otherwise. Never fabricates a score for an unalignable model.
    """
    if not gt_images or not poses:
        return {"refused": True, "reason": "no poses or no reference poses"}

    gt = _gt_to_c2w(gt_images)
    est = {p.evidence_id: p for p in poses}
    common = sorted(set(est) & set(gt))
    if len(common) < min_cameras:
        return {
            "refused": True,
            "reason": f"only {len(common)} common cameras; >= {min_cameras} required to define alignment",
            "common_cameras": len(common),
        }

    est_centers = [est[n].position for n in common]
    gt_centers = [gt[n][1] for n in common]

    # Degeneracy checks on BOTH center sets. A coincident configuration
    # cannot define scale or translation; a collinear one is flagged.
    def _scales(pts):
        a = np.asarray(pts, dtype=np.float64)
        c = a - a.mean(axis=0)
        return np.linalg.svd(c, compute_uv=False), float((c ** 2).sum())

    gt_sv, _ = _scales(gt_centers)
    est_sv, est_var = _scales(est_centers)
    notes: List[str] = []
    for label, sv in (("reference", gt_sv), ("estimated", est_sv)):
        if sv[0] <= 1e-9 * max(1.0, float(sv[0])):
            return {
                "refused": True,
                "reason": f"degenerate {label} geometry (coincident camera centers)",
                "common_cameras": len(common),
            }
    if gt_sv[1] <= 1e-3 * gt_sv[0] or est_sv[1] <= 1e-3 * est_sv[0]:
        notes.append("camera centers nearly planar/collinear; alignment is weakly constrained")

    # --- gauge alignment -------------------------------------------------
    # ONE rigid gauge (R_align, s, t) explains the estimated world:
    # orientations AND centers share it. The rotation comes from the
    # paired camera->world orientation frames (Horn, chordal-optimal);
    # the scale from center distance ratios (well-defined for planar
    # and collinear configurations, unlike a rank-deficient Umeyama
    # call); the translation from the center centroids.
    R_align = _horn_rotation_from_quats([est[n].rotation for n in common], [gt[n][0] for n in common])

    est_d = gt_d = 0.0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            est_d += math.dist(est_centers[i], est_centers[j])
            gt_d += math.dist(gt_centers[i], gt_centers[j])
    if est_d <= 1e-12:
        return {
            "refused": True,
            "reason": "degenerate estimated geometry (zero center spread)",
            "common_cameras": len(common),
        }
    s = gt_d / est_d  # scale mapping estimated scene onto reference
    scale_ratio = 1.0 / s

    est_mean = (sum(c[0] for c in est_centers) / len(common),
                sum(c[1] for c in est_centers) / len(common),
                sum(c[2] for c in est_centers) / len(common))
    gt_mean = (sum(c[0] for c in gt_centers) / len(common),
               sum(c[1] for c in gt_centers) / len(common),
               sum(c[2] for c in gt_centers) / len(common))
    rot_est_mean = [sum(R_align[r][k] * est_mean[k] for k in range(3)) for r in range(3)]
    t = tuple(gt_mean[r] - s * rot_est_mean[r] for r in range(3))

    rot_errors: List[float] = []
    cen_errors: List[float] = []
    for idx, n in enumerate(common):
        R_est = quat_to_matrix(est[n].rotation)
        R_est_aligned = [[sum(R_align[r][k] * R_est[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
        q_est_aligned = _matrix_to_quat(R_est_aligned)
        rot_errors.append(quat_angle_deg(q_est_aligned, gt[n][0]))
        mapped = tuple(s * sum(R_align[r][k] * est_centers[idx][k] for k in range(3)) + t[r] for r in range(3))
        cen_errors.append(math.dist(mapped, gt[n][1]))

    rot_errors.sort()
    cen_errors.sort()
    n = len(rot_errors)

    return {
        "refused": False,
        "common_cameras": n,
        "rotation_deg_median": rot_errors[n // 2],
        "rotation_deg_max": rot_errors[-1],
        "rotation_deg_mean": sum(rot_errors) / n,
        "center_error_median": cen_errors[n // 2],
        "center_error_max": cen_errors[-1],
        "center_error_mean": sum(cen_errors) / n,
        "scale_ratio": scale_ratio,
        "notes": notes,
        "gauge_transform": (R_align, t, s),
    }


def _horn_rotation_from_quats(
    est_quats: List[Sequence[float]], gt_quats: List[Sequence[float]]
) -> List[List[float]]:
    """Least-squares global rotation R_align maximizing alignment of
    paired camera->world orientations: for each pair, R_align @ R_est
    ~= R_gt. Built on the quaternion relative-rotation trick: the
    optimal R_align is the chordal mean of R_gt @ R_est^T across
    pairs, computed with Horn's method on virtual paired centers
    (e_i and R_rel_i @ e_i)."""
    if len(est_quats) != len(gt_quats):
        raise ValueError("paired quaternion counts differ")
    # Build virtual paired point sets whose Horn solution IS the
    # chordal-mean rotation: use the rotation matrices' columns as
    # paired vectors (columns of R_gt vs R_align @ R_est).
    src: List[Vec3] = []
    dst: List[Vec3] = []
    for q_est, q_gt in zip(est_quats, gt_quats):
        R_est = quat_to_matrix(q_est)
        R_gt = quat_to_matrix(q_gt)
        for c in range(3):
            col_est = tuple(R_est[r][c] for r in range(3))
            col_gt = tuple(R_gt[r][c] for r in range(3))
            src.append(col_est)
            dst.append(col_gt)
    return horn_rotation(src, dst)
