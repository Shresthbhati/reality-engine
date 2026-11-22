"""Independent validation of a production run on 7-Scenes "office" photographs against the dataset's OWN ground truth.

Ground truth used (never given to the pipeline): per-frame camera-to-world poses (KinectFusion) and 16-bit depth in millimetres.
Both are estimates with centimetre-level error of their own, so every number below is "agreement with KinectFusion", not a laser
survey. Nothing here is a physical-accuracy claim about objects: the dataset has no labelled object dimensions.

    python benchmarks/measured_rooms/validate_office7.py <run_dir> <frames_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform ``dst ~ s * R @ src + t`` minimising squared error (Umeyama 1991). Needs >= 3 non-collinear points."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("need matching Nx3 arrays with N >= 3")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs ** 2).sum() / len(src)
    if var_s < 1e-12:
        raise ValueError("source points are degenerate")
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s)
    return s, R, mu_d - s * R @ mu_s


def gt_centre(pose_file: Path) -> np.ndarray:
    return np.loadtxt(pose_file).reshape(4, 4)[:3, 3]


def validate_cameras(sfm_centres: dict[str, list[float]], gt_centres: dict[str, np.ndarray]) -> dict:
    names = sorted(set(sfm_centres) & set(gt_centres))
    if len(names) < 4:
        return {"status": "insufficient", "matched": len(names)}
    a = np.array([sfm_centres[n] for n in names], float)
    b = np.array([gt_centres[n] for n in names], float)
    s, R, t = umeyama(a, b)
    err = np.linalg.norm(s * (a @ R.T) + t - b, axis=1)
    return {"status": "ok", "matched": len(names), "scale_metres_per_sfm_unit": s, "ate_rmse_m": float(np.sqrt((err ** 2).mean())),
            "ate_max_m": float(err.max()), "per_camera_error_m": {n: float(e) for n, e in zip(names, err)},
            "gt_trajectory_extent_m": float(np.linalg.norm(b.max(0) - b.min(0))),
            "note": "similarity-aligned to KinectFusion poses (itself an estimate); scale is recovered FROM the ground truth, "
                    "so this validates pose consistency and yields a scale estimate, it does not make the reconstruction metric"}


def main(run_dir: Path, frames: Path) -> int:
    run = json.loads((run_dir / "run.json").read_text())
    payload = run["job"]["payload"]
    cams = payload.get("cameras") or {}
    recs = (payload.get("evidence") or {}).get("records") or []
    by_id = {}
    for r in recs:  # evidence id -> original file name (the upload keeps the file name)
        name = r.get("filename") or r.get("name") or r.get("original_name")
        if name:
            by_id[r.get("evidence_id") or r.get("id")] = name
    sfm = {by_id[k].replace(".color.png", ""): v for k, v in cams.items() if k in by_id}
    gt = {p.name.replace(".pose.txt", ""): gt_centre(p) for p in frames.glob("frame-*.pose.txt")}
    out = {"registered_cameras": len(cams), "named": len(sfm), "record_keys": sorted(recs[0]) if recs else [],
           "cameras": validate_cameras(sfm, gt)}
    (run_dir / "validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: v for k, v in out["cameras"].items() if k != "per_camera_error_m"}
    print(json.dumps({**{k: v for k, v in out.items() if k != "cameras"}, "cameras": brief}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
