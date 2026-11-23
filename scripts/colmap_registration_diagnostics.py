"""Per-image and per-pair COLMAP registration diagnostics, run with the EXACT configuration Reality Engine's backend uses.

    python scripts/colmap_registration_diagnostics.py --images datasets/south_building/images --n 6 --out <empty dir>

Runs feature_extractor -> exhaustive_matcher -> mapper (every attempt in MAPPER_ATTEMPTS that the backend would run) with the backend's own flag
builders, keeps the COMPLETE stdout/stderr of every step (``logs/``), then reads COLMAP's own database and the mapper's sub-models and writes
``report.json``:

* per image: size, camera model/focal, EXIF orientation/focal, keypoint count, 8x8 spatial coverage, blur (variance of Laplacian), registered?
* per pair: raw matches, verified inliers, two-view geometry type, inlier ratio, median inlier displacement (px, and as a share of the width)
* per mapper attempt: sub-models, registered images, points, observations, and the final COLMAP log lines (initial-pair search, failures)

Nothing here estimates or invents: every number is read from COLMAP's database/logs or the image file.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction.backend.colmap_backend import (  # noqa: E402
    MAPPER_ATTEMPTS, _sift_extraction_args, _sift_matching_args, _subprocess_env,
)

#: COLMAP two_view_geometries.config codes
GEOMETRY = {0: "undefined", 1: "degenerate", 2: "calibrated", 3: "uncalibrated", 4: "planar", 5: "panoramic", 6: "planar_or_panoramic",
            7: "watermark", 8: "multiple"}
PAIR_BASE = 2147483647


def run_step(colmap: str, step: str, args: list[str], env: dict, logdir: Path, tag: str) -> dict:
    t0 = time.time()
    proc = subprocess.run([colmap, step, *args], capture_output=True, text=True, env=env)
    (logdir / f"{tag}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (logdir / f"{tag}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    return {"step": step, "tag": tag, "exit": proc.returncode, "seconds": round(time.time() - t0, 1),
            "command": [colmap, step, *args], "stdout_bytes": len(proc.stdout or ""), "stderr_bytes": len(proc.stderr or "")}


def _blob(b, dtype, cols):
    return np.frombuffer(b, dtype=dtype).reshape(-1, cols) if b else np.zeros((0, cols), dtype=dtype)


def read_database(db: Path) -> dict:
    con = sqlite3.connect(str(db))
    cams = {cid: dict(model=m, width=w, height=h, params=_blob(p, np.float64, 1).ravel().tolist())
            for cid, m, w, h, p in con.execute("select camera_id,model,width,height,params from cameras")}
    images = {iid: dict(name=n, camera_id=c) for iid, n, c in con.execute("select image_id,name,camera_id from images")}
    kps = {iid: _blob(d, np.float32, cols) for iid, r, cols, d in con.execute("select image_id,rows,cols,data from keypoints")}
    tvg = {}
    for pid, rows, cols, data, cfg in con.execute("select pair_id,rows,cols,data,config from two_view_geometries"):
        tvg[pid] = (_blob(data, np.uint32, 2) if rows else np.zeros((0, 2), np.uint32), int(cfg))
    raw = {pid: int(rows) for pid, rows in con.execute("select pair_id,rows from matches")}
    con.close()
    return {"cameras": cams, "images": images, "keypoints": kps, "two_view": tvg, "raw_matches": raw}


def image_facts(path: Path) -> dict:
    out: dict = {}
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out["blur_laplacian_var"] = round(float(cv2.Laplacian(img, cv2.CV_64F).var()), 1)
            out["pixels"] = [int(img.shape[1]), int(img.shape[0])]
    except Exception as exc:  # noqa: BLE001
        out["blur_error"] = repr(exc)
    try:
        from PIL import Image

        ex = Image.open(path).getexif()
        sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
        out["exif_orientation"] = ex.get(274)
        focal = sub.get(37386) or ex.get(37386)
        out["exif_focal_mm"] = float(focal) if focal else None
        out["exif_focal_35mm"] = sub.get(41989)
    except Exception as exc:  # noqa: BLE001
        out["exif_error"] = repr(exc)
    return out


def coverage(xy: np.ndarray, w: int, h: int, grid: int = 8) -> float:
    """Share of the grid x grid image cells that contain at least one keypoint."""
    if len(xy) == 0 or not w or not h:
        return 0.0
    cx = np.clip((xy[:, 0] / w * grid).astype(int), 0, grid - 1)
    cy = np.clip((xy[:, 1] / h * grid).astype(int), 0, grid - 1)
    return round(len({(int(a), int(b)) for a, b in zip(cx, cy)}) / (grid * grid), 3)


def parse_model(model_dir: Path) -> dict:
    imgs = (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
    names = [ln.split()[9] for ln in imgs if ln and not ln.startswith("#") and len(ln.split()) >= 10]
    pts = [ln.split() for ln in (model_dir / "points3D.txt").read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    tracks = [(len(p) - 8) // 2 for p in pts if len(p) >= 9]
    err = [float(p[7]) for p in pts if len(p) >= 9]
    return {"model": model_dir.name, "registered_images": sorted(names), "points": len(pts), "observations": int(sum(tracks)),
            "mean_track_length": round(float(np.mean(tracks)), 2) if tracks else None,
            "mean_reprojection_error_px": round(float(np.mean(err)), 3) if err else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=0, help="first N sorted images (0 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain", action="store_true", help="plain SIFT on GPU (NOT the production configuration)")
    ap.add_argument("--colmap", default="colmap")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir()
    names = sorted(p.name for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    names = names[: a.n] if a.n else names
    for n in names:
        shutil.copy(Path(a.images) / n, out / "images" / n)
    colmap = shutil.which(a.colmap) or a.colmap
    env = _subprocess_env(colmap)
    robust = not a.plain
    gpu = "1" if a.plain else "0"  # backend: affine-shape SIFT is CPU-only, so robust SIFT runs extraction/matching on the CPU
    db = out / "database.db"
    steps = [run_step(colmap, "feature_extractor", ["--database_path", str(db), "--image_path", str(out / "images"),
                                                    *_sift_extraction_args(robust=robust), "--FeatureExtraction.use_gpu", gpu], env, out / "logs", "01_extract")]
    steps.append(run_step(colmap, "exhaustive_matcher", ["--database_path", str(db), "--FeatureMatching.use_gpu", gpu, *_sift_matching_args(guided=False)],
                          env, out / "logs", "02_match"))

    attempts = []
    for k, extra in enumerate(MAPPER_ATTEMPTS):
        sp = out / f"sparse_attempt{k}"
        sp.mkdir()
        st = run_step(colmap, "mapper", ["--database_path", str(db), "--image_path", str(out / "images"), "--output_path", str(sp), *extra],
                      env, out / "logs", f"03_mapper_attempt{k}")
        models = []
        for d in sorted(p for p in sp.iterdir() if p.is_dir() and p.name.isdigit()):
            steps.append(run_step(colmap, "model_converter", ["--input_path", str(d), "--output_path", str(d), "--output_type", "TXT"], env,
                                  out / "logs", f"04_convert_attempt{k}_model{d.name}"))
            models.append(parse_model(d))
        stderr = (out / "logs" / f"03_mapper_attempt{k}.stderr.log").read_text(encoding="utf-8")
        key = [ln for ln in stderr.splitlines() if any(s in ln for s in ("initial image pair", "Discarding", "No good", "Failed", "Registering", "Registered", "bad"))]
        attempts.append({**st, "extra_args": list(extra), "models": models, "key_log_lines": key[-40:]})
        if models and max(len(m["registered_images"]) for m in models) >= 0.8 * len(names):
            break  # the backend's own stop rule (WEAK_REGISTRATION_SHARE)

    d = read_database(db)
    reg = {n for at in attempts for m in at["models"] for n in m["registered_images"]}
    id_by_name = {v["name"]: k for k, v in d["images"].items()}
    per_image = []
    for n in names:
        iid = id_by_name.get(n)
        cam = d["cameras"].get(d["images"][iid]["camera_id"]) if iid else None
        kp = d["keypoints"].get(iid, np.zeros((0, 2), np.float32))
        w, h = (cam["width"], cam["height"]) if cam else (0, 0)
        per_image.append({"image": n, "keypoints": int(len(kp)), "spatial_coverage_8x8": coverage(kp[:, :2], w, h),
                          "camera": {"model": cam["model"], "params": cam["params"], "size": [w, h]} if cam else None,
                          "registered_in_any_attempt": n in reg, **image_facts(Path(a.images) / n)})
    per_pair = []
    for (i1, n1), (i2, n2) in itertools.combinations(sorted((k, v["name"]) for k, v in d["images"].items()), 2):
        pairs, cfg = d["two_view"].get(i1 * PAIR_BASE + i2, (np.zeros((0, 2), np.uint32), 0))
        raw = d["raw_matches"].get(i1 * PAIR_BASE + i2, 0)
        disp = None
        if len(pairs):
            k1, k2 = d["keypoints"][i1][:, :2], d["keypoints"][i2][:, :2]
            disp = float(np.median(np.linalg.norm(k1[pairs[:, 0]] - k2[pairs[:, 1]], axis=1)))
        w = d["cameras"][d["images"][i1]["camera_id"]]["width"]
        per_pair.append({"a": n1, "b": n2, "raw_matches": raw, "verified_inliers": int(len(pairs)), "geometry": GEOMETRY.get(cfg, str(cfg)),
                         "inlier_ratio": round(len(pairs) / raw, 3) if raw else None,
                         "median_inlier_displacement_px": round(disp, 1) if disp is not None else None,
                         "median_displacement_share_of_width": round(disp / w, 4) if disp is not None and w else None})
    report = {"images": names,
              "configuration": "plain SIFT, GPU (NOT production)" if a.plain else "production: robust SIFT (affine shape + DSP), CPU, exhaustive, guided off",
              "steps": steps, "mapper_attempts": attempts, "per_image": per_image, "per_pair": per_pair,
              "summary": {"n_images": len(names), "registered_any_attempt": len(reg), "pairs_total": len(per_pair),
                          "pairs_with_verified_inliers": sum(1 for p in per_pair if p["verified_inliers"] > 0),
                          "pairs_ge_15_inliers": sum(1 for p in per_pair if p["verified_inliers"] >= 15)}}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"]), "->", out / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
