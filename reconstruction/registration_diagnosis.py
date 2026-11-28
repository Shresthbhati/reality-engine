"""Why did cameras not register? Reproducible diagnosis of one image set (Sprint 9 P1).

Builds on ``reconstruction.diagnostics`` (the per-model findings, thresholds and ``Diagnostic`` record) and on
``reconstruction.capture_quality`` (photo metrics). Adds what those cannot see: the COLMAP *database* (keypoints, raw
matches, geometrically verified inliers per pair) and the mapper *log* (initial pair, registration order).

Three layers, each usable without COLMAP:

  * ``read_database`` / ``read_model`` / ``parse_mapper_log``   parse COLMAP's outputs (sqlite, text model, log)
  * ``analyse``                                               pure function of those -> a JSON-serialisable report
  * ``capture_guidance``                                      report -> user-facing strings ({kind, message, basis})

``run_colmap`` + ``diagnose_image_set`` drive the real binary (CPU only unless told otherwise) with the production
flags (affine+DSP SIFT, exhaustive matching, the production mapper attempts) and write ``diagnosis.json`` +
``summary.txt``. Nothing here changes any production default.

Honesty rules
  * An unregistered image gets a *derived* reason. COLMAP never logs why an image was skipped (verified on 4.2.0: no
    "could not register" line exists), so the reason is computed from the database and the final model, and the
    numbers behind it are stored next to it.
  * Root-cause classes are EXPERIMENTAL rules over named thresholds below; each cause carries its evidence. ``wrong
    intrinsics`` is never inferred from one run: it needs the A/B in ``intrinsics_variant``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.diagnostics import Diagnostic

# ------------------------------------------------------------------ thresholds (EXPERIMENTAL, named) ------------
#: COLMAP's own two-view minimum: a pair with fewer verified inliers is not an edge (TwoViewGeometry min_num_inliers).
MIN_EDGE_INLIERS = 15
#: an edge strong enough to carry registration (COLMAP abs_pose_min_num_inliers default).
STRONG_EDGE_INLIERS = 30
ABS_POSE_MIN_INLIERS = 30
#: COLMAP's initial-pair requirement (Mapper.init_min_num_inliers); the mapper only halves it when it relaxes.
INIT_MIN_INLIERS = 100
#: a pair with a baseline: triangulation angle below this (degrees) is "no usable baseline" (COLMAP min_tri_angle 1.5).
MIN_TRI_ANGLE_DEG = 1.5
#: median keypoints per image below this = too little texture to carry matching (calibrated in the report from the
#: south_building controls vs the synthetic sets; see docs/audits/CAMERA_REGISTRATION_DIAGNOSIS.md).
LOW_KEYPOINTS = 500
LOW_SHARPNESS = 100.0            # capture_quality.BLUR_WARN_BELOW (Laplacian variance at <=640 px)
MOTION_ANISOTROPY = 2.5          # gradient energy ratio between the two image axes that suggests directional blur
PANORAMIC_PAIR_SHARE = 0.5       # share of verified pairs COLMAP labelled PANORAMIC / PLANAR_OR_PANORAMIC
LOW_INLIER_RATIO = 0.25          # median verified/raw ratio below this = matches mostly wrong (repetition/low texture)
FOCAL_DISAGREE = 0.25            # |solved - initial| / initial focal above this = initial intrinsics doubtful
REGISTERED_OK_SHARE = 0.8        # production WEAK_REGISTRATION_SHARE

#: COLMAP two_view_geometries.config values
CONFIG_NAMES = {0: "undefined", 1: "degenerate", 2: "calibrated", 3: "uncalibrated", 4: "planar", 5: "panoramic",
                6: "planar_or_panoramic", 7: "watermark", 8: "multiple"}
_ROTATION_ONLY_CONFIGS = {5, 6}
MAX_PAIR_ID = 2147483647


def _pair_ids(pair_id: int) -> Tuple[int, int]:
    return pair_id // MAX_PAIR_ID, pair_id % MAX_PAIR_ID


# ------------------------------------------------------------------ COLMAP outputs ----------------------------
def read_database(db_path: Path) -> dict:
    """Cameras, images, keypoint counts/positions, raw match counts and verified inlier matches per pair."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        cams = {}
        for cid, model, w, h, params, prior in con.execute(
                "select camera_id, model, width, height, params, prior_focal_length from cameras"):
            cams[cid] = {"model_id": model, "width": w, "height": h, "prior_focal_length": bool(prior),
                         "params": [float(v) for v in np.frombuffer(params, dtype=np.float64)]}
        images = {iid: {"name": name, "camera_id": cid} for iid, name, cid in con.execute(
            "select image_id, name, camera_id from images")}
        kps: Dict[int, np.ndarray] = {}
        for iid, rows, cols, blob in con.execute("select image_id, rows, cols, data from keypoints"):
            kps[iid] = (np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)[:, :2].copy()
                        if rows and blob else np.zeros((0, 2), np.float32))
        raw = {}
        for pid, rows in con.execute("select pair_id, rows from matches"):
            raw[_pair_ids(pid)] = rows
        pairs = {}
        for pid, rows, cols, blob, config, qvec, tvec in con.execute(
                "select pair_id, rows, cols, data, config, qvec, tvec from two_view_geometries"):
            key = _pair_ids(pid)
            idx = (np.frombuffer(blob, dtype=np.uint32).reshape(rows, cols).copy() if rows and blob
                   else np.zeros((0, 2), np.uint32))
            rel = None
            if qvec is not None and len(qvec) == 32:
                q = np.frombuffer(qvec, dtype=np.float64)
                rel = float(np.degrees(2 * math.acos(min(1.0, abs(q[0])))))
            pairs[key] = {"inliers": int(rows), "config": int(config), "matches": idx, "rel_rotation_deg": rel}
        return {"cameras": cams, "images": images, "keypoints": kps, "raw_matches": raw, "pairs": pairs}
    finally:
        con.close()


_PROJ = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5, "OPENCV": 8}


def read_model(model_dir: Path) -> dict:
    """A COLMAP TEXT model (cameras.txt / images.txt / points3D.txt) -> cameras, images (with 2D->3D links), points."""
    model_dir = Path(model_dir)
    cams = {}
    for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if p and not p[0].startswith("#"):
            cams[int(p[0])] = {"model": p[1], "width": int(p[2]), "height": int(p[3]), "params": [float(v) for v in p[4:]]}
    images = {}
    lines = [l for l in (model_dir / "images.txt").read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    for head, pts in zip(lines[0::2], lines[1::2]):
        p = head.split()
        if len(p) < 10:
            continue
        flat = pts.split()
        xy = np.array([[float(flat[i]), float(flat[i + 1])] for i in range(0, len(flat), 3)]) if flat else np.zeros((0, 2))
        p3 = np.array([int(flat[i + 2]) for i in range(0, len(flat), 3)], dtype=np.int64) if flat else np.zeros(0, np.int64)
        images[int(p[0])] = {"q": np.array([float(v) for v in p[1:5]]), "t": np.array([float(v) for v in p[5:8]]),
                             "camera_id": int(p[8]), "name": p[9], "xy": xy, "point3d": p3}
    points = {}
    for line in (model_dir / "points3D.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if p and not p[0].startswith("#"):
            tr = p[8:]
            points[int(p[0])] = {"xyz": np.array([float(v) for v in p[1:4]]), "error": float(p[7]),
                                 "track": [(int(tr[i]), int(tr[i + 1])) for i in range(0, len(tr), 2)]}
    return {"cameras": cams, "images": images, "points": points}


def _rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _project(cam: dict, pc: np.ndarray) -> Optional[np.ndarray]:
    """Camera-frame points (n,3) -> pixels (n,2) for the models COLMAP picks by default; None for any other model."""
    m, p = cam["model"], cam["params"]
    if m not in _PROJ or len(p) < _PROJ[m]:
        return None
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    r2 = x * x + y * y
    if m == "SIMPLE_PINHOLE":
        return np.stack([p[0] * x + p[1], p[0] * y + p[2]], 1)
    if m == "PINHOLE":
        return np.stack([p[0] * x + p[2], p[1] * y + p[3]], 1)
    if m == "SIMPLE_RADIAL":
        d = 1 + p[3] * r2
        return np.stack([p[0] * x * d + p[1], p[0] * y * d + p[2]], 1)
    if m == "RADIAL":
        d = 1 + p[3] * r2 + p[4] * r2 * r2
        return np.stack([p[0] * x * d + p[1], p[0] * y * d + p[2]], 1)
    k1, k2, p1, p2 = p[4:8]
    d = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * d + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * d + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([p[0] * xd + p[2], p[1] * yd + p[3]], 1)


def model_metrics(model: dict) -> dict:
    """Per registered image: observations, RMS reprojection error (own projection, all model-linked keypoints), camera
    centre and viewing direction; plus the median triangulation angle of the model's points (degrees)."""
    per, centres, dirs = {}, {}, {}
    pts = model["points"]
    for iid, im in model["images"].items():
        R = _rot(im["q"])
        centres[iid] = -R.T @ im["t"]
        dirs[iid] = R.T @ np.array([0.0, 0.0, 1.0])
        mask = im["point3d"] >= 0
        ids = [int(i) for i in im["point3d"][mask] if int(i) in pts]
        rms = None
        if ids:
            X = np.array([pts[i]["xyz"] for i in ids])
            pc = (R @ X.T).T + im["t"]
            front = pc[:, 2] > 1e-9
            proj = _project(model["cameras"][im["camera_id"]], pc[front]) if front.any() else None
            if proj is not None:
                obs = im["xy"][mask][[k for k, i in enumerate(im["point3d"][mask]) if int(i) in pts]][front]
                rms = float(np.sqrt(np.mean(np.sum((proj - obs) ** 2, axis=1))))
        per[iid] = {"name": im["name"], "observations": len(ids), "reproj_rms_px": None if rms is None else round(rms, 3)}
    angles = []
    for pt in list(pts.values())[:5000]:
        cs = [centres[i] for i, _ in pt["track"] if i in centres]
        if len(cs) < 2:
            continue
        rays = np.array([c - pt["xyz"] for c in cs])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
        angles.append(float(np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1, 1)))))
    return {"per_image": per, "centres": centres, "dirs": dirs,
            "median_tri_angle_deg": round(median(angles), 2) if angles else None,
            "share_tri_angle_below_min": round(float(np.mean([a < MIN_TRI_ANGLE_DEG for a in angles])), 3) if angles else None}


_RE_INIT = re.compile(r"Registering initial image pair #(\d+) and #(\d+)")
_RE_REG = re.compile(r"Registering image #(\d+) \(num_reg_frames=(\d+)\)")
_RE_SEES = re.compile(r"Image sees (\d+) / (\d+) points")


def parse_mapper_log(text: str) -> dict:
    """What the mapper itself says. COLMAP 4.2 has no per-image failure line, so this holds only initialisations and
    the order/strength of successful registrations."""
    out = {"initial_pairs": [], "registered_order": [], "no_initial_pair": text.count("No good initial image pair found"),
           "relaxed_initialisation": text.count("Relaxing the initialization constraints"),
           "models_kept": text.count("Keeping successful reconstruction"), "discarded_models": text.count("Discarding reconstruction")}
    cur = None
    for line in text.splitlines():
        m = _RE_INIT.search(line)
        if m:
            out["initial_pairs"].append([int(m.group(1)), int(m.group(2))])
        m = _RE_REG.search(line)
        if m:
            cur = {"image_id": int(m.group(1)), "num_registered": int(m.group(2))}
            out["registered_order"].append(cur)
            continue
        m = _RE_SEES.search(line)
        if m and cur is not None and "sees" not in cur:
            cur["sees"], cur["of_points"] = int(m.group(1)), int(m.group(2))
    return out


# ------------------------------------------------------------------ per-image evidence ------------------------
def image_quality(path: Path) -> dict:
    """Resolution, sharpness (Laplacian variance), exposure, edge density and directional-blur anisotropy."""
    import cv2

    from reconstruction.capture_quality import ANALYSIS_MAX_SIDE_PX, photo_metrics

    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return {"readable": False}
    h, w = g.shape
    s = ANALYSIS_MAX_SIDE_PX / max(h, w)
    small = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else g
    m = photo_metrics(small)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1)
    ex, ey = float((gx ** 2).mean()) + 1e-9, float((gy ** 2).mean()) + 1e-9
    edges = cv2.Canny(small, 50, 150)
    m.update(readable=True, width=w, height=h, megapixels=round(w * h / 1e6, 2),
             edge_density=round(float((edges > 0).mean()), 4),
             gradient_anisotropy=round(max(ex / ey, ey / ex), 2),
             intensity_std=round(float(small.std()), 1))
    return m


def image_exif(path: Path) -> dict:
    from PIL import Image

    try:
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)
            fl, f35 = sub.get(37386), sub.get(41989)
            return {"make": ex.get(271), "model": ex.get(272),
                    "focal_mm": float(fl) if fl else None, "focal_35mm": float(f35) if f35 else None}
    except Exception:  # noqa: BLE001 -- unreadable/non-EXIF image: report no EXIF, never guess
        return {"make": None, "model": None, "focal_mm": None, "focal_35mm": None}


def intrinsics_report(db: dict, model: Optional[dict], exif: Dict[str, dict], sizes: Dict[str, Tuple[int, int]]) -> dict:
    """Per image: where COLMAP's *initial* focal came from (EXIF prior vs 1.2*max-side default), the distortion model,
    the focal it solved, and the EXIF-implied focal for comparison (35mm-equivalent -> pixels on the long side)."""
    out = {}
    names_by_cam: Dict[int, List[str]] = {}
    for iid, im in db["images"].items():
        names_by_cam.setdefault(im["camera_id"], []).append(im["name"])
    solved_by_name = {}
    if model:
        for iid, im in model["images"].items():
            solved_by_name[im["name"]] = model["cameras"][im["camera_id"]]
    for iid, im in db["images"].items():
        cam = db["cameras"][im["camera_id"]]
        ex = exif.get(im["name"], {})
        w, h = sizes.get(im["name"], (cam["width"], cam["height"]))
        init_f = cam["params"][0]
        exif_f = ex.get("focal_35mm") / 36.0 * max(w, h) if ex.get("focal_35mm") else None
        s = solved_by_name.get(im["name"])
        solved_f = s["params"][0] if s else None
        out[im["name"]] = {
            "source": "exif_prior" if cam["prior_focal_length"] else "default_1.2x_max_side",
            "camera_id": im["camera_id"], "shared_with": len(names_by_cam[im["camera_id"]]),
            "initial_focal_px": round(init_f, 1), "exif_implied_focal_px": None if exif_f is None else round(exif_f, 1),
            "distortion_model": s["model"] if s else None,
            "distortion_k1": round(s["params"][3], 4) if s and s["model"] in ("SIMPLE_RADIAL", "RADIAL") else None,
            "solved_focal_px": None if solved_f is None else round(solved_f, 1),
            "focal_change_ratio": None if solved_f is None else round(abs(solved_f - init_f) / init_f, 3),
            "exif": ex}
    return out


# ------------------------------------------------------------------ analysis ----------------------------------
def _components(nodes: Iterable[int], edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: Dict[int, List[int]] = {}
    for n in parent:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g))


def pair_table(db: dict) -> List[dict]:
    rows = []
    for (a, b), p in sorted(db["pairs"].items()):
        raw = db["raw_matches"].get((a, b), 0)
        rows.append({"a": db["images"][a]["name"], "b": db["images"][b]["name"], "raw_matches": raw,
                     "verified_inliers": p["inliers"], "inlier_ratio": round(p["inliers"] / raw, 3) if raw else None,
                     "config": CONFIG_NAMES.get(p["config"], str(p["config"])), "rel_rotation_deg": p["rel_rotation_deg"]})
    return rows


def graph_summary(db: dict) -> dict:
    ids = sorted(db["images"])
    edges = [(a, b) for (a, b), p in db["pairs"].items() if p["inliers"] >= MIN_EDGE_INLIERS]
    strong = [(a, b) for (a, b), p in db["pairs"].items() if p["inliers"] >= STRONG_EDGE_INLIERS]
    comps = _components(ids, edges)
    deg = {i: 0 for i in ids}
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    name = lambda i: db["images"][i]["name"]  # noqa: E731
    live = [p for p in db["pairs"].values() if p["inliers"] >= MIN_EDGE_INLIERS]
    ratios = [p["inliers"] / db["raw_matches"][k] for k, p in db["pairs"].items()
              if p["inliers"] >= MIN_EDGE_INLIERS and db["raw_matches"].get(k)]
    cfg = {}
    for p in live:
        cfg[CONFIG_NAMES.get(p["config"], str(p["config"]))] = cfg.get(CONFIG_NAMES.get(p["config"], str(p["config"])), 0) + 1
    return {
        "images": len(ids), "pairs_total": len(db["pairs"]), "edges_ge_15": len(edges), "edges_ge_30": len(strong),
        "edge_density": round(len(edges) / max(1, len(ids) * (len(ids) - 1) / 2), 3),
        "components": [[name(i) for i in c] for c in comps], "n_components": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "isolated_images": [name(i) for i in ids if deg[i] == 0],
        "degree": {name(i): deg[i] for i in ids},
        "median_inliers_of_edges": median([p["inliers"] for p in live]) if live else 0,
        "best_pair_inliers": max((p["inliers"] for p in db["pairs"].values()), default=0),
        "pairs_ge_init_min": sum(1 for p in db["pairs"].values() if p["inliers"] >= INIT_MIN_INLIERS),
        "median_best_edge_per_image": median([max([p["inliers"] for k, p in db["pairs"].items() if i in k] or [0]) for i in ids]) if ids else 0,
        "median_inlier_ratio_of_edges": round(median(ratios), 3) if ratios else None,
        "config_counts_of_edges": cfg,
        "rotation_only_share": round(sum(v for k, v in cfg.items() if k in ("panoramic", "planar_or_panoramic")) / max(1, len(live)), 3)}


def unregistered_reasons(db: dict, models: List[dict], registered_names: set) -> Dict[str, dict]:
    """Derived reason for every image no sub-model contains, with the numbers behind it (see module docstring)."""
    by_name = {im["name"]: iid for iid, im in db["images"].items()}
    graph = graph_summary(db)
    comp_of = {n: ci for ci, c in enumerate(graph["components"]) for n in c}
    reg_comp = {comp_of[n] for n in registered_names if n in comp_of}
    out = {}
    for name, iid in sorted(by_name.items()):
        if name in registered_names:
            continue
        best_edge = max((p["inliers"] for (a, b), p in db["pairs"].items() if iid in (a, b)), default=0)
        reg_ids = {by_name[n] for n in registered_names if n in by_name}
        best_to_reg = max((p["inliers"] for (a, b), p in db["pairs"].items()
                           if iid in (a, b) and (b if a == iid else a) in reg_ids), default=0)
        best_cand, cand_models = 0, []
        for mi, md in enumerate(models):
            kp_to_pts: Dict[int, set] = {}
            neigh = 0
            for rid, rim in md["images"].items():
                rname = rim["name"]
                rdb = by_name.get(rname)
                key = (min(iid, rdb), max(iid, rdb)) if rdb else None
                p = db["pairs"].get(key) if key else None
                if not p or p["inliers"] < MIN_EDGE_INLIERS:
                    continue
                neigh += 1
                u_col, r_col = (0, 1) if iid < rdb else (1, 0)
                for row in p["matches"]:
                    j = int(row[r_col])
                    if j < len(rim["point3d"]) and rim["point3d"][j] >= 0:
                        kp_to_pts.setdefault(int(row[u_col]), set()).add(int(rim["point3d"][j]))
            cand_pts = {pid for s in kp_to_pts.values() for pid in s}
            cand_models.append({"model": mi, "registered_neighbours": neigh, "candidate_2d3d": len(kp_to_pts),
                                "distinct_3d_points": len(cand_pts)})
            best_cand = max(best_cand, len(kp_to_pts))
        if not models and best_edge >= MIN_EDGE_INLIERS:
            kind = "no_initial_pair"
            why = (f"no model was built at all: the best pair in the set has {graph['best_pair_inliers']} verified inliers "
                   f"(initialisation wants >= {INIT_MIN_INLIERS}, halved when relaxed); this photo's best is {best_edge}")
        elif best_edge < MIN_EDGE_INLIERS:
            kind = "no_verified_overlap"
            why = (f"no pair with >= {MIN_EDGE_INLIERS} geometrically verified inliers (best {best_edge}): "
                   "it overlaps no other photo enough to be matched")
        elif comp_of.get(name) not in reg_comp:
            kind = "disconnected_component"
            why = (f"verified edges exist (best {best_edge}) but only to photos that are not in any registered model: "
                   "its overlap group never connected to the model")
        elif best_to_reg < STRONG_EDGE_INLIERS and best_edge >= STRONG_EDGE_INLIERS:
            kind = "weakly_bridged_cluster"
            why = (f"it matches other UNPLACED photos strongly ({best_edge} inliers) but a placed photo only weakly "
                   f"({best_to_reg} < {STRONG_EDGE_INLIERS}): a cluster that overlaps itself, thinly tied to the model")
        elif best_cand < ABS_POSE_MIN_INLIERS:
            kind = "insufficient_2d3d_inliers"
            why = (f"best {best_to_reg} verified inliers to a placed photo ({best_edge} to any), but only {best_cand} of its "
                   f"keypoints reach a triangulated 3D point (< {ABS_POSE_MIN_INLIERS} needed for absolute pose)")
        else:
            kind = "pose_estimation_rejected"
            why = (f"{best_cand} 2D-3D candidates (>= {ABS_POSE_MIN_INLIERS}) exist yet the mapper did not place it: "
                   "RANSAC inlier ratio, bundle-adjustment filtering or seed order rejected it")
        out[name] = {"reason": kind, "explanation": why, "best_pair_inliers": best_edge,
                     "best_pair_inliers_to_registered": best_to_reg,
                     "candidate_2d3d_best_model": best_cand, "per_model": cand_models}
    return out


def classify_causes(report: dict) -> List[dict]:
    """Root-cause candidates from the numbers in ``report``, strongest first. Each: {cause, confidence, evidence}."""
    g, q, k = report["graph"], report["image_summary"], report["mapper"]
    n = g["images"]
    reg = report["registered"]["count"]
    if n == 0 or reg >= REGISTERED_OK_SHARE * n:
        return []
    causes = []

    def add(cause, conf, **ev):
        causes.append({"cause": cause, "confidence": conf, "evidence": ev})

    low_tex = q["median_keypoints"] < LOW_KEYPOINTS
    if low_tex:
        add("low_texture", "high" if q["median_keypoints"] < LOW_KEYPOINTS / 2 else "medium",
            median_keypoints=q["median_keypoints"], threshold=LOW_KEYPOINTS, median_edge_density=q["median_edge_density"],
            median_sharpness=q["median_sharpness"])
    if q["median_sharpness"] < LOW_SHARPNESS and q["median_anisotropy"] >= MOTION_ANISOTROPY:
        add("motion_blur", "medium", median_sharpness=q["median_sharpness"], median_anisotropy=q["median_anisotropy"])
    elif q["median_sharpness"] < LOW_SHARPNESS and not low_tex:
        add("soft_focus_or_blur", "low", median_sharpness=q["median_sharpness"])
    tri = report["coverage"].get("median_tri_angle_deg")
    # COLMAP labels a pair PLANAR_OR_PANORAMIC for a flat scene as well as for a pure rotation, so the share alone proves
    # nothing: it only counts when the model (if any) also shows a tiny triangulation angle, or there is no model.
    if g["rotation_only_share"] >= PANORAMIC_PAIR_SHARE and g["edges_ge_15"] and (tri is None or tri < MIN_TRI_ANGLE_DEG * 2):
        add("rotation_only_or_planar", "high" if g["rotation_only_share"] >= 0.8 else "medium",
            panoramic_pair_share=g["rotation_only_share"], config_counts=g["config_counts_of_edges"],
            median_tri_angle_deg=report["coverage"].get("median_tri_angle_deg"))
    elif tri is not None and tri < MIN_TRI_ANGLE_DEG * 2:
        add("rotation_only_or_planar", "medium", median_tri_angle_deg=tri, minimum=MIN_TRI_ANGLE_DEG)
    ratio = g["median_inlier_ratio_of_edges"]
    if ratio is not None and ratio < LOW_INLIER_RATIO:
        add("ambiguous_matches_repetition_or_low_texture", "medium", median_inlier_ratio=ratio, threshold=LOW_INLIER_RATIO)
    thin = g["best_pair_inliers"] < INIT_MIN_INLIERS or g["median_best_edge_per_image"] < INIT_MIN_INLIERS / 2
    if g["isolated_images"] or g["n_components"] > 1 or thin:
        add("insufficient_overlap", "high" if len(g["isolated_images"]) >= max(1, n // 3) or g["n_components"] > 1 or reg == 0 else "medium",
            isolated=len(g["isolated_images"]), components=g["n_components"], largest_component=g["largest_component"],
            edge_density=g["edge_density"], median_inliers_of_edges=g["median_inliers_of_edges"],
            best_pair_inliers=g["best_pair_inliers"], pairs_ge_init_min=g["pairs_ge_init_min"],
            median_best_edge_per_image=g["median_best_edge_per_image"], median_keypoints=q["median_keypoints"])
    focal = [i["focal_change_ratio"] for i in report["intrinsics"].values() if i["focal_change_ratio"] is not None]
    defaults = sum(1 for i in report["intrinsics"].values() if i["source"] != "exif_prior")
    if focal and median(focal) > FOCAL_DISAGREE:
        add("suspect_intrinsics", "low", median_focal_change=round(median(focal), 3), images_on_default_focal=defaults,
            note="needs the intrinsics A/B to be attributed; one run cannot separate this from poor geometry")
    reasons = {v["reason"] for v in report["unregistered"].values()}
    if "pose_estimation_rejected" in reasons:
        add("mapper_config", "medium", images=[n_ for n_, v in report["unregistered"].items() if v["reason"] == "pose_estimation_rejected"],
            attempts=[(a["attempt"], a["registered"]) for a in k["attempts"]])
    if len({a["registered"] for a in k["attempts"]}) > 1:
        add("mapper_seed_sensitivity", "medium", attempts=[(a["attempt"], a["registered"]) for a in k["attempts"]])
    if k.get("no_initial_pair_all_attempts"):
        add("no_initial_pair", "high", best_pair_inliers=g["best_pair_inliers"], pairs_ge_init_min=g["pairs_ge_init_min"],
            relaxations=sum(l.get("relaxed_initialisation", 0) for l in k["logs"].values()),
            note="the mapper found no pair with enough verified inliers and triangulation angle to start")
    return causes


def coverage_report(model_metrics_list: List[dict], n_input: int, registered: int) -> dict:
    """Registered-camera coverage in the largest model: spread of centres vs scene depth, widest viewing separation."""
    if not model_metrics_list:
        return {"registered_fraction": 0.0}
    mm = max(model_metrics_list, key=lambda m: len(m["centres"]))
    cs = np.array(list(mm["centres"].values()))
    ds = np.array(list(mm["dirs"].values()))
    spread = float(np.linalg.norm(cs.max(0) - cs.min(0))) if len(cs) > 1 else 0.0
    sep = float(np.degrees(np.arccos(np.clip((ds @ ds.T).min(), -1, 1)))) if len(ds) > 1 else 0.0
    return {"registered_fraction": round(registered / max(1, n_input), 3), "cameras_in_largest_model": len(cs),
            "camera_centre_spread_model_units": round(spread, 3), "max_viewing_separation_deg": round(sep, 1),
            "median_tri_angle_deg": mm["median_tri_angle_deg"], "share_points_tri_angle_below_min": mm["share_tri_angle_below_min"]}


def analyse(db: dict, models: List[dict], mapper_attempts: List[dict], logs: Dict[str, dict],
            qualities: Dict[str, dict], exif: Dict[str, dict], sizes: Dict[str, Tuple[int, int]],
            chosen: Optional[int] = None) -> dict:
    """Pure analysis. ``models`` = text-model dicts of the CHOSEN attempt (all its sub-models); ``mapper_attempts`` =
    [{attempt, args, registered, points, models}]; ``logs`` = parse_mapper_log output per attempt label."""
    names = {im["name"] for im in db["images"].values()}
    registered_names = {im["name"] for md in models for im in md["images"].values()}
    mms = [model_metrics(md) for md in models]
    per_image = {}
    kp = {db["images"][i]["name"]: len(v) for i, v in db["keypoints"].items()}
    for iid, im in db["images"].items():
        nm = im["name"]
        v = db["keypoints"].get(iid, np.zeros((0, 2)))
        cam = db["cameras"][im["camera_id"]]
        cells = 0
        if len(v):
            gx = np.minimum((v[:, 0] / cam["width"] * 4).astype(int), 3)
            gy = np.minimum((v[:, 1] / cam["height"] * 4).astype(int), 3)
            cells = len(set(zip(gx.tolist(), gy.tolist())))
        per_image[nm] = {"keypoints": kp[nm], "keypoint_grid_coverage_16": round(cells / 16, 2), "quality": qualities.get(nm, {}),
                         "registered": nm in registered_names, "reproj_rms_px": None, "observations": None}
    for mm in mms:
        for r in mm["per_image"].values():
            per_image[r["name"]].update(reproj_rms_px=r["reproj_rms_px"], observations=r["observations"])
    qv = lambda key: [v["quality"].get(key) for v in per_image.values() if v["quality"].get(key) is not None]  # noqa: E731
    summary = {"median_keypoints": median(kp.values()) if kp else 0, "min_keypoints": min(kp.values()) if kp else 0,
               "median_sharpness": median(qv("blur_laplacian_var")) if qv("blur_laplacian_var") else None,
               "median_edge_density": median(qv("edge_density")) if qv("edge_density") else None,
               "median_anisotropy": median(qv("gradient_anisotropy")) if qv("gradient_anisotropy") else None,
               "median_mean_luma": median(qv("mean_luma")) if qv("mean_luma") else None,
               "resolutions": sorted({f"{v['quality'].get('width')}x{v['quality'].get('height')}" for v in per_image.values() if v["quality"]})}
    if summary["median_sharpness"] is None:
        summary["median_sharpness"] = float("inf")
        summary["median_anisotropy"] = 1.0
        summary["median_edge_density"] = None
    report = {
        "images": per_image, "image_summary": summary, "graph": graph_summary(db), "pairs": pair_table(db),
        "intrinsics": intrinsics_report(db, models[0] if models else None, exif, sizes),
        "registered": {"count": len(registered_names), "of": len(names), "names": sorted(registered_names),
                       "chosen_attempt": chosen, "sub_models": [len(m["images"]) for m in models]},
        "unregistered": unregistered_reasons(db, models, registered_names),
        "coverage": coverage_report(mms, len(names), len(registered_names)),
        "mapper": {"attempts": mapper_attempts, "logs": logs,
                   "no_initial_pair_all_attempts": bool(logs) and all(not l["initial_pairs"] for l in logs.values())},
        "thresholds": {"MIN_EDGE_INLIERS": MIN_EDGE_INLIERS, "ABS_POSE_MIN_INLIERS": ABS_POSE_MIN_INLIERS,
                       "LOW_KEYPOINTS": LOW_KEYPOINTS, "LOW_SHARPNESS": LOW_SHARPNESS, "LOW_INLIER_RATIO": LOW_INLIER_RATIO,
                       "PANORAMIC_PAIR_SHARE": PANORAMIC_PAIR_SHARE, "MIN_TRI_ANGLE_DEG": MIN_TRI_ANGLE_DEG},
    }
    report["root_causes"] = classify_causes(report)
    report["diagnostics"] = [d.to_dict() for d in as_diagnostics(report)]
    return report


def as_diagnostics(report: dict) -> List[Diagnostic]:
    """The report's findings as ``diagnostics.Diagnostic`` records (same shape the correction engine already reads)."""
    out = []
    for name, u in report["unregistered"].items():
        out.append(Diagnostic("UNREGISTERED_IMAGE", "warning", name, u["explanation"],
                              measured={"reason": u["reason"], "best_pair_inliers": u["best_pair_inliers"],
                                        "candidate_2d3d": u["candidate_2d3d_best_model"]},
                              recommended=capture_advice(u["reason"])))
    for c in report["root_causes"]:
        out.append(Diagnostic("REGISTRATION_ROOT_CAUSE:" + c["cause"], "warning", "reconstruction",
                              f"Likely cause of low registration: {c['cause']} ({c['confidence']} confidence).",
                              measured=c["evidence"], recommended=capture_advice(c["cause"])))
    return out


# ------------------------------------------------------------------ user-facing guidance ----------------------
_ADVICE = {
    "no_verified_overlap": "Retake this photo from a spot 0.3-0.5 m away from a photo that already shows the same wall or furniture, so about 60-80% of it appears in both.",
    "disconnected_component": "Add 2-3 photos that show the join between this group of photos and the rest of the room, each overlapping both groups by 60-80%.",
    "insufficient_2d3d_inliers": "This photo overlaps its neighbour but shares too few matchable details with the placed photos; retake it a little closer to the neighbouring view, with textured surfaces (books, art, patterned objects) in frame.",
    "weakly_bridged_cluster": "These photos overlap each other but barely the rest of the room: add 2-3 photos taken in between, each sharing 60-80% of its view with both the group and the main set.",
    "pose_estimation_rejected": "This photo was close to being placed; adding one more overlapping photo next to it will usually settle it.",
    "low_texture": "Include textured surfaces (shelves, art, rugs, plants, patterned objects) in every photo; plain painted walls, glossy floors and flat colour give the matcher nothing to lock onto.",
    "motion_blur": "Hold the phone still for each photo (brace elbows, tap the shutter gently) and use bright light; blurred photos lose the sharp corners matching needs.",
    "soft_focus_or_blur": "Tap to focus, wait for the focus to settle, and keep the lens clean; the photos are soft.",
    "rotation_only_or_planar": "Move your feet between photos: take each photo from a spot at least 0.3-0.5 m from the previous one. Turning on the spot, or photographing one flat wall, gives no depth.",
    "ambiguous_matches_repetition_or_low_texture": "Repeating patterns (tiles, blinds, identical windows) cause false matches; include distinctive items in each view, and keep neighbouring photos closer together.",
    "insufficient_overlap": "Walk around the room in a loop and take a photo about every 0.5 m (or every 15-20 degrees of turn) so each photo shares 60-80% of its view with the next.",
    "suspect_intrinsics": "Take photos with the same lens and zoom setting throughout (no digital zoom, no switching between wide and ultra-wide).",
    "mapper_seed_sensitivity": "The result depends on chance; adding 3-4 more overlapping photos makes the layout stable.",
    "mapper_config": "Add a few more overlapping photos; the engine was on the edge of placing some.",
    "no_initial_pair": "No two photos had enough shared detail AND enough sideways movement to start from. Take two photos of the same textured area about 0.3-0.5 m apart, then continue around the room.",
}


def capture_advice(key: str) -> str:
    return _ADVICE.get(key, "Add photographs that overlap each other from different positions.")


def capture_guidance(report: dict, max_items: int = 5) -> List[dict]:
    """Diagnosis -> user-facing ``{kind, message, basis}`` items, the shape ``engine/pipeline/guidance.build_guidance``
    already emits (so the product can append them to ``report['guidance']``). Empty when registration is fine. Every
    ``basis`` quotes the measurement, so the advice can be checked."""
    items: List[dict] = []
    reg, n = report["registered"]["count"], report["registered"]["of"]
    if n and reg >= REGISTERED_OK_SHARE * n:
        return items
    g = report["graph"]
    for c in report["root_causes"]:
        ev = c["evidence"]
        if c["cause"] == "low_texture":
            basis = f"median {ev['median_keypoints']:.0f} keypoints per photo (need >= {LOW_KEYPOINTS})"
        elif c["cause"] == "motion_blur":
            basis = f"median sharpness {ev['median_sharpness']:.0f}, directional blur ratio {ev['median_anisotropy']:.1f}"
        elif c["cause"] == "rotation_only_or_planar":
            basis = ("; ".join(f"{k} {v}" for k, v in ev.items() if k in ("panoramic_pair_share", "median_tri_angle_deg") and v is not None))
        elif c["cause"] == "insufficient_overlap":
            basis = (f"best pair shares {ev['best_pair_inliers']} verified matches (a start needs {INIT_MIN_INLIERS}); "
                     f"{ev['isolated']} photo(s) with none, {ev['components']} unconnected group(s)")
        elif c["cause"] == "ambiguous_matches_repetition_or_low_texture":
            basis = f"only {ev['median_inlier_ratio']:.0%} of raw matches survive geometric verification"
        else:
            basis = "; ".join(f"{k}={v}" for k, v in list(ev.items())[:3])
        items.append({"kind": c["cause"], "message": capture_advice(c["cause"]), "basis": basis})
    shown = {u["reason"] for u in report["unregistered"].values()}
    for r in ("no_initial_pair", "no_verified_overlap", "disconnected_component", "weakly_bridged_cluster", "insufficient_2d3d_inliers",
              "pose_estimation_rejected"):
        if r in shown:
            ims = sorted(nm for nm, u in report["unregistered"].items() if u["reason"] == r)
            items.append({"kind": r, "message": capture_advice(r),
                          "basis": f"{len(ims)} photo(s) not placed: {', '.join(ims[:4])}{'...' if len(ims) > 4 else ''}"})
    seen, out = set(), []
    for it in items:
        if it["kind"] not in seen:
            seen.add(it["kind"])
            out.append(it)
    return out[:max_items]


# ------------------------------------------------------------------ driving COLMAP ----------------------------
@dataclass
class RunConfig:
    """What the run does. Defaults mirror production (``ColmapReconstructionBackend``): affine+DSP SIFT on CPU,
    exhaustive matching, the production mapper attempts. ``use_gpu`` defaults False on purpose (shared GPU)."""
    robust_sift: bool = True
    use_gpu: bool = False
    single_camera: Optional[bool] = None       # None = production rule: one shared camera when every image has one size
    camera_model: Optional[str] = None         # None = COLMAP default (SIMPLE_RADIAL)
    camera_params: Optional[str] = None        # "f,cx,cy[,k]" pins initial intrinsics (diagnostic A/B only)
    pin_intrinsics: bool = False               # also stop bundle adjustment refining focal/distortion (A/B only)
    max_image_size: Optional[int] = None       # None = COLMAP default (3200)
    extra_extractor: List[str] = field(default_factory=list)
    extra_matcher: List[str] = field(default_factory=list)
    extra_mapper: List[str] = field(default_factory=list)
    attempts: Optional[Sequence[Sequence[str]]] = None   # None = production MAPPER_ATTEMPTS


def _env(colmap: str) -> dict:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    plugins = Path(colmap).resolve().parent.parent / "plugins"
    if plugins.is_dir():
        env.setdefault("QT_PLUGIN_PATH", str(plugins))
    return env


def _step(colmap: str, step: str, args: Sequence[str], log: Path, env: dict, timeout: int = 3600, check: bool = True) -> int:
    p = subprocess.run([colmap, step, *args], capture_output=True, text=True, env=env, timeout=timeout)
    log.write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
    if check and p.returncode != 0:
        raise RuntimeError(f"colmap {step} failed ({p.returncode}); see {log}")
    return p.returncode


def colmap_available(colmap: str) -> bool:
    try:
        return subprocess.run([colmap, "-h"], capture_output=True, text=True, timeout=30, env=_env(colmap)).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _uniform_size(paths: Sequence[Path]) -> bool:
    from PIL import Image

    sizes = set()
    for p in paths:
        with Image.open(p) as im:
            sizes.add(im.size)
    return len(paths) >= 2 and len(sizes) == 1


def run_colmap(images: Sequence[Path], work: Path, colmap: str, cfg: Optional[RunConfig] = None) -> dict:
    """Copy ``images`` into ``work/images`` and run extract -> exhaustive match -> mapper attempts. Returns paths."""
    cfg = cfg or RunConfig()
    from reconstruction.backend.colmap_backend import MAPPER_ATTEMPTS, WEAK_REGISTRATION_SHARE  # production values

    work = Path(work)
    shutil.rmtree(work, ignore_errors=True)
    (work / "images").mkdir(parents=True)
    for p in images:
        shutil.copy(p, work / "images" / Path(p).name)
    env, gpu = _env(colmap), "1" if cfg.use_gpu else "0"
    single = cfg.single_camera if cfg.single_camera is not None else _uniform_size(list(images))
    db = work / "database.db"
    ext = ["--database_path", str(db), "--image_path", str(work / "images"), "--FeatureExtraction.use_gpu", "0" if cfg.robust_sift else gpu]
    if single:
        ext += ["--ImageReader.single_camera", "1"]
    if cfg.camera_model:
        ext += ["--ImageReader.camera_model", cfg.camera_model]
    if cfg.camera_params:
        ext += ["--ImageReader.camera_params", cfg.camera_params]
    if cfg.max_image_size:
        ext += ["--FeatureExtraction.max_image_size", str(cfg.max_image_size)]
    if cfg.robust_sift:
        ext += ["--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"]
    _step(colmap, "feature_extractor", ext + cfg.extra_extractor, work / "feature_extractor.log", env)
    # CPU matching can die with an allocator error when the machine is shared and memory is tight; retry with fewer
    # threads (same results, slower) rather than report a capture failure that is really a host failure.
    for threads in (None, "4", "1"):
        rc = _step(colmap, "exhaustive_matcher", ["--database_path", str(db), "--FeatureMatching.use_gpu", gpu,
                                                  *(["--FeatureMatching.num_threads", threads] if threads else []), *cfg.extra_matcher],
                   work / "exhaustive_matcher.log", env, check=False)
        if rc == 0:
            break
    else:
        raise RuntimeError(f"colmap exhaustive_matcher failed (exit {rc}) even single-threaded; see {work / 'exhaustive_matcher.log'}")
    pinned = ["--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0"] if cfg.pin_intrinsics else []
    attempts = cfg.attempts if cfg.attempts is not None else MAPPER_ATTEMPTS
    out = {"work": work, "db": db, "single_camera": single, "robust_sift": cfg.robust_sift}
    out["attempts"] = _run_mappers(colmap, work, len(images), attempts, [*pinned, *cfg.extra_mapper], env, WEAK_REGISTRATION_SHARE)
    return out


def _run_mappers(colmap: str, work: Path, n_images: int, attempts: Sequence[Sequence[str]], base_args: Sequence[str],
                 env: dict, stop_share: float) -> List[dict]:
    """The mapper attempts on ``work/database.db`` (production order; stops once ``stop_share`` of the photos are placed)."""
    out = []
    for k, extra in enumerate(attempts):
        sp = work / f"sparse_try{k}"
        sp.mkdir()
        log = work / f"mapper_try{k}.log"
        # COLMAP exits 1 with "Failed to create any sparse model" when no initial pair exists: that is the registers-0
        # outcome we must report, not an error to raise.
        rc = _step(colmap, "mapper", ["--database_path", str(work / "database.db"), "--image_path", str(work / "images"),
                                      "--output_path", str(sp), *base_args, *extra], log, env, check=False)
        models = []
        for d in sorted((d for d in sp.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name)):
            _step(colmap, "model_converter", ["--input_path", str(d), "--output_path", str(d), "--output_type", "TXT"],
                  d / "convert.log", env)
            models.append(d)
        if rc != 0 and not models and "Failed to create any sparse model" not in log.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"colmap mapper failed ({rc}) without the no-model message; see {log}")
        parsed = [read_model(d) for d in models]
        n_reg = len({i["name"] for m in parsed for i in m["images"].values()})
        out.append({"attempt": k, "args": list(extra), "dir": sp, "log": log, "models": models, "exit_code": rc,
                    "registered": n_reg, "points": sum(len(m["points"]) for m in parsed)})
        if n_reg >= max(3, math.ceil(stop_share * n_images)):
            break      # production stops retrying here
    return out


def remap(src_work: Path, dst_work: Path, colmap: str, mapper_args: Sequence[str], label: str = "") -> dict:
    """Diagnostic A/B on the MAPPER only: copy a finished run's database + images and re-run a single mapper attempt with
    ``mapper_args`` (e.g. relaxed initial-pair thresholds). Shows whether a failure is the data or the mapper settings.
    Never a default; the result is reported next to the production run."""
    src, dst = Path(src_work), Path(dst_work)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    shutil.copytree(src / "images", dst / "images")
    shutil.copy(src / "database.db", dst / "database.db")
    n = len(list((dst / "images").iterdir()))
    _run_mappers(colmap, dst, n, [()], list(mapper_args), _env(colmap), 1.0)
    return diagnose_workdir(dst, colmap, {"label": label, "attempt_args": {0: list(mapper_args)}, "mapper_ab": list(mapper_args)})


def diagnose_image_set(images: Sequence[Path], work: Path, colmap: str, cfg: Optional[RunConfig] = None, label: str = "") -> dict:
    """Run COLMAP on ``images`` and return the full diagnosis (also written to ``work/diagnosis.json`` + ``summary.txt``)."""
    cfg = cfg or RunConfig()
    run = run_colmap(images, work, colmap, cfg)
    meta = {"attempt_args": {a["attempt"]: a["args"] for a in run["attempts"]}, "label": label, "gpu": cfg.use_gpu, "robust_sift": cfg.robust_sift, "single_camera": run["single_camera"],
            "camera_model": cfg.camera_model, "camera_params": cfg.camera_params, "pin_intrinsics": cfg.pin_intrinsics}
    return diagnose_workdir(Path(work), colmap, meta)


def diagnose_workdir(work: Path, colmap: str = "colmap", meta: Optional[dict] = None) -> dict:
    """Re-derive the whole diagnosis from an existing run directory (database.db, images/, sparse_try*/, mapper_try*.log)
    without running COLMAP again: how the analysis was re-run after a rule change, and how a production workspace can be
    diagnosed after the fact."""
    work = Path(work)
    db = read_database(work / "database.db")
    attempts, per_attempt_models = [], {}
    for sp in sorted(work.glob("sparse_try*"), key=lambda d: int(d.name.replace("sparse_try", ""))):
        k = int(sp.name.replace("sparse_try", ""))
        dirs = sorted((d for d in sp.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
        mods = [read_model(d) for d in dirs]
        per_attempt_models[k] = mods
        extra = (meta or {}).get("attempt_args", {}).get(k, [])
        attempts.append({"attempt": k, "args": extra, "registered": len({i["name"] for m in mods for i in m["images"].values()}),
                         "points": sum(len(m["points"]) for m in mods), "sub_models": len(mods)})
    best = max(attempts, key=lambda a: (a["registered"], a["points"]))      # production picks by (cameras, points)
    logs = {}
    for a in attempts:
        lp = work / f"mapper_try{a['attempt']}.log"
        logs[f"attempt{a['attempt']}"] = parse_mapper_log(lp.read_text(encoding="utf-8", errors="replace")) if lp.exists() else parse_mapper_log("")
    for l in logs.values():     # the log speaks in image ids; the report speaks in file names
        l["initial_pairs_named"] = [[db["images"][a]["name"], db["images"][b]["name"]] for a, b in l["initial_pairs"]
                                    if a in db["images"] and b in db["images"]]
    paths = {p.name: p for p in sorted((work / "images").iterdir())}
    quals = {n: image_quality(p) for n, p in paths.items()}
    exif = {n: image_exif(p) for n, p in paths.items()}
    sizes = {n: (q.get("width", 0), q.get("height", 0)) for n, q in quals.items()}
    rep = analyse(db, per_attempt_models[best["attempt"]], attempts, logs, quals, exif, sizes, best["attempt"])
    banner = ""
    try:
        banner = subprocess.run([colmap, "-h"], capture_output=True, text=True, env=_env(colmap), timeout=30).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    rep["run"] = {"colmap": banner, "images": sorted(paths), **(meta or {})}
    rep["guidance"] = capture_guidance(rep)
    Path(work, "diagnosis.json").write_text(json.dumps(rep, indent=1, default=_jsonable), encoding="utf-8")
    Path(work, "summary.txt").write_text(summarize(rep), encoding="utf-8")
    return rep


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def summarize(rep: dict) -> str:
    r, g, q = rep["registered"], rep["graph"], rep["image_summary"]
    lines = [f"[{rep.get('run', {}).get('label', '')}] registered {r['count']}/{r['of']} (sub-models {r['sub_models']}), "
             f"attempts {[(a['attempt'], a['registered']) for a in rep['mapper']['attempts']]}",
             f"  keypoints median {q['median_keypoints']:.0f} (min {q['min_keypoints']}); sharpness median {q['median_sharpness']:.0f}; "
             f"edge density {q['median_edge_density']}; resolutions {q['resolutions']}",
             f"  graph: {g['edges_ge_15']} edges>=15 ({g['edges_ge_30']}>=30), density {g['edge_density']}, components "
             f"{[len(c) for c in g['components']]}, isolated {len(g['isolated_images'])}, median inliers/edge "
             f"{g['median_inliers_of_edges']}, inlier ratio {g['median_inlier_ratio_of_edges']}, configs {g['config_counts_of_edges']}",
             f"  coverage: {rep['coverage']}"]
    for nm, u in rep["unregistered"].items():
        lines.append(f"  NOT PLACED {nm}: {u['reason']} - {u['explanation']}")
    for c in rep["root_causes"]:
        lines.append(f"  CAUSE {c['cause']} ({c['confidence']}): {c['evidence']}")
    for gd in rep.get("guidance", []):
        lines.append(f"  ADVICE [{gd['kind']}] {gd['message']}  <- {gd['basis']}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ subset selection (from an overlap graph) --
def select_subset(db: dict, k: int, strategy: str) -> List[str]:
    """Pick ``k`` image names from an overlap graph. ``dense_overlap`` greedily adds the image sharing the most verified
    inliers with the chosen set; ``sparse_chain`` adds the image whose best link to the chosen set is the weakest that
    is still an edge (a thin chain); ``stride`` takes every n-th name (what dropping photos evenly looks like)."""
    names = {i: im["name"] for i, im in db["images"].items()}
    ids = sorted(names, key=lambda i: names[i])
    if strategy == "stride":
        step = len(ids) / k
        return [names[ids[int(i * step)]] for i in range(k)]

    def inl(a, b):
        return db["pairs"].get((min(a, b), max(a, b)), {"inliers": 0})["inliers"]

    start = max(ids, key=lambda i: sum(inl(i, j) for j in ids))
    chosen = [start]
    while len(chosen) < k:
        cand = [(max(inl(c, i) for c in chosen), i) for i in ids if i not in chosen]
        cand = [c for c in cand if c[0] >= MIN_EDGE_INLIERS] or cand
        _, pick = (max if strategy == "dense_overlap" else min)(cand, key=lambda t: (t[0], -t[1]) if strategy == "dense_overlap" else (t[0], t[1]))
        chosen.append(pick)
    return sorted(names[i] for i in chosen)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Diagnose camera registration for one image set (CPU COLMAP).")
    ap.add_argument("images", help="directory of photos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--colmap", default=os.environ.get("COLMAP_BINARY", "colmap"))
    ap.add_argument("--names", help="comma-separated file names to use (default: all)")
    ap.add_argument("--label", default="")
    ap.add_argument("--camera-params", help="pin initial intrinsics 'f,cx,cy,k' (A/B diagnostic only)")
    ap.add_argument("--camera-model")
    ap.add_argument("--pin-intrinsics", action="store_true")
    ap.add_argument("--per-image-cameras", action="store_true")
    ns = ap.parse_args(argv)
    d = Path(ns.images)
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if ns.names:
        want = set(ns.names.split(","))
        files = [p for p in files if p.name in want]
    cfg = RunConfig(camera_params=ns.camera_params, camera_model=ns.camera_model, pin_intrinsics=ns.pin_intrinsics,
                    single_camera=False if ns.per_image_cameras else None)
    rep = diagnose_image_set(files, Path(ns.out), ns.colmap, cfg, ns.label or d.name)
    print(summarize(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
