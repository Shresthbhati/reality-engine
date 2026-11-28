"""Capture readiness: can THESE photographs be registered into one shared 3D frame, and if not, which photographs are needed.

Computed BEFORE reconstruction (``assess_readiness``) from independent CPU feature matching (OpenCV SIFT + ratio test +
RANSAC), and again AFTER a partial registration (``after_registration``) from the stored pair facts plus, when a sparse model
exists, its reprojection / triangulation statistics. It reuses ``capture_quality.photo_metrics / judge_photo`` for blur and
exposure; it does not replace the scene gate (``scene_identity``, LightGlue) or COLMAP.

Honesty rules
  * Every threshold is PROVISIONAL (see ``THRESHOLDS``): set from synthetic stress images and the recorded runs in
    docs/reconstruction/REAL_RUN_STAGE_REPORT.md; none is a calibrated operating point.
  * Whatever the checks cannot decide is reported as ``unknown`` and listed in ``cannot_decide``; it never passes. ``ready`` needs
    every pair of every usable photograph evaluated and a positive finding on each check.
  * Pose/parallax numbers from a pair use an ASSUMED focal (EXIF 35 mm equivalent when present, else 1.2 * max side, the COLMAP
    default) and say so. They diagnose the capture; they do not measure the room.
  * This module never produces poses. A photograph not placed by the multi-view solver stays an independent hypothesis, and
    ``frame`` says ``shared_frame: false`` for it.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:  # optional runtime dependency of the repo (same convention as capture_quality)
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from reconstruction.capture_quality import (
    BLUR_DROP_BELOW,
    BLUR_WARN_BELOW,
    judge_photo,
    photo_metrics,
)

SCHEMA = "capture_readiness/1"
READY, NEEDS_MORE, BLOCKED, UNKNOWN = "ready", "needs_more_photos", "blocked", "unknown"

THRESHOLDS: dict[str, float] = {
    "analysis_max_side_px": 800,       # images are shrunk to this before SIFT (runtime cap, not a quality claim)
    "sift_features": 1500,
    "ratio_test": 0.8,                 # Lowe ratio
    "ransac_px": 2.0,                  # fundamental-matrix RANSAC threshold at analysis resolution
    "verified_min_inliers": 30,        # the figure the recorded runs used (REAL_RUN_STAGE_REPORT)
    "verified_min_inlier_ratio": 0.25,  # inliers / ratio-test matches; rejects pairs that pass only by volume
    "weak_min_inliers": 18,            # unrelated textured images gave <=12 RANSAC inliers in the stress tests; below = "no relation"
    "weak_min_inlier_ratio": 0.2,
    "weak_bridge_inliers": 60,         # a graph bridge with fewer inliers than this is a weak bridge
    "duplicate_shift_fraction": 0.01,  # median shift (fraction of image diagonal) at or below this = same viewpoint
    "duplicate_hash_bits": 6,          # pHash Hamming distance that makes a duplicate CANDIDATE (needs feature confirmation)
    "low_parallax_homography": 0.90,   # one homography explains this fraction of inliers = rotation-only / planar
    "low_parallax_tri_deg": 2.0,       # median triangulation angle below this = narrow baseline
    "window": 8,                       # +-window in capture order when all pairs do not fit in max_pairs
    "max_pairs": 800,
    "budget_s": 90.0,
    "blur_relative": 0.35,             # Laplacian variance below this fraction of the set median = relatively blurred
    "crop_scale_ratio": 1.25,          # cross-size verified pair whose feature scale ratio is outside [1/x, x]
}
PROVISIONAL_NOTE = ("Thresholds are provisional (synthetic stress images + recorded runs; not calibrated on a real room capture). "
                    "Pose and parallax numbers use an assumed focal length. 'unknown' means the check could not decide.")
_PROBLEM_CAUSES = {"isolated", "no_verified_overlap", "blurred", "unreadable", "too_low_parallax", "intrinsics_mismatch",
                   "disconnected_group", "other"}


def _f(x: Any, nd: int = 3) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _read_gray_and_meta(path: Path) -> dict | None:
    """Decode once: grey analysis image, true size, EXIF facts. None when the file is not a decodable image."""
    try:
        full = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)   # fromfile: unicode paths on Windows
    except Exception:  # noqa: BLE001 -- a bad file is a result (readable=False), not an exception
        return None
    if full is None or min(full.shape) < 32:
        return None
    h, w = full.shape
    exif: dict[str, Any] = {"present": False}
    try:
        from PIL import Image

        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769) if ex else {}
            exif = {"make": ex.get(0x010F), "model": ex.get(0x0110), "focal_mm": _f(sub.get(0x920A)),
                    "focal_35mm": _f(sub.get(0xA405)), "lens": sub.get(0xA434)}
            exif["present"] = any(exif.values())
    except Exception:  # noqa: BLE001 -- EXIF is optional evidence
        pass
    s = THRESHOLDS["analysis_max_side_px"] / max(h, w)
    small = cv2.resize(full, (max(32, round(w * s)), max(32, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else full
    return {"gray": small, "width": w, "height": h, "scale": small.shape[1] / w, "exif": exif}


def _phash(gray: np.ndarray) -> int:
    d = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()
    return int("".join("1" if b else "0" for b in d > np.median(d[1:])), 2)


def _focal_prior_px(width: int, height: int, exif: Mapping[str, Any]) -> tuple[float, str]:
    if exif.get("focal_35mm"):
        return float(exif["focal_35mm"]) * math.hypot(width, height) / 43.267, "exif_35mm_equivalent"
    return 1.2 * max(width, height), "assumed_1.2_x_max_side"


def _components(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[list[str]]:
    nodes = list(nodes)
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g, key=nodes.index) for g in groups.values()), key=lambda g: (-len(g), nodes.index(g[0])))


# ---------------------------------------------------------------------------------------------- pairwise matching
def _match_pair(fa: dict, fb: dict, ra: dict, rb: dict) -> dict:
    """SIFT ratio-test matches -> F-RANSAC inliers; homography share, shift, scale ratio; rotation/triangulation angle (assumed
    focal) when the pair is verified."""
    out: dict[str, Any] = {"putative": 0, "inliers": 0, "inlier_ratio": None, "homography_fraction": None,
                           "median_shift_fraction": None, "rotation_deg": None, "triangulation_deg": None, "scale_ratio": None}
    da, db = fa["des"], fb["des"]
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return out
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < THRESHOLDS["ratio_test"] * m[1].distance]
    out["putative"] = len(good)
    if len(good) < THRESHOLDS["weak_min_inliers"]:   # cannot reach even a weak pair; skips the RANSAC (the dominant cost)
        return out
    p1 = np.array([fa["pts"][m.queryIdx] for m in good], np.float32)
    p2 = np.array([fb["pts"][m.trainIdx] for m in good], np.float32)
    cv2.setRNGSeed(0)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, THRESHOLDS["ransac_px"], 0.999, 1500)
    if F is None or mask is None:
        return out
    inl = mask.ravel().astype(bool)
    n_in = int(inl.sum())
    out["inliers"], out["inlier_ratio"] = n_in, _f(n_in / len(good))
    if n_in < 8:
        return out
    q1, q2 = p1[inl], p2[inl]
    ca, cb = np.float32(fa["wh"]) / 2, np.float32(fb["wh"]) / 2
    n1, n2 = (q1 - ca) / fa["wh"], (q2 - cb) / fb["wh"]                       # normalised to [-0.5, 0.5] per image
    out["median_shift_fraction"] = _f(float(np.median(np.hypot(*(n1 - n2).T))) / math.sqrt(2.0), 4)
    H, hm = cv2.findHomography(q1, q2, cv2.RANSAC, 1.5 * THRESHOLDS["ransac_px"])
    out["homography_fraction"] = _f(float(hm.sum()) / n_in) if H is not None and hm is not None else None
    sa = np.array([fa["kp_size"][m.queryIdx] for m in np.array(good)[inl]])
    sb = np.array([fb["kp_size"][m.trainIdx] for m in np.array(good)[inl]])
    out["scale_ratio"] = _f(float(np.median(sb / np.maximum(sa, 1e-6))))
    if n_in >= THRESHOLDS["verified_min_inliers"]:
        try:
            ka, kb = ra["f_analysis"], rb["f_analysis"]
            e1, e2 = (q1 - ca) / ka, (q2 - cb) / kb
            E, em = cv2.findEssentialMat(e1, e2, np.eye(3), cv2.RANSAC, 0.999, THRESHOLDS["ransac_px"] / ((ka + kb) / 2))
            if E is not None and E.shape == (3, 3):
                _, R, t, pm = cv2.recoverPose(E, e1, e2, np.eye(3), mask=em)
                keep = pm.ravel() > 0
                if keep.sum() >= 8:
                    out["rotation_deg"] = _f(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))), 2)
                    X = cv2.triangulatePoints(np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, t]), e1[keep].T, e2[keep].T)
                    X = (X[:3] / X[3]).T
                    v1, v2 = X, X - (-R.T @ t).ravel()
                    cosang = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
                    out["triangulation_deg"] = _f(float(np.median(np.degrees(np.arccos(np.clip(cosang, -1, 1))))), 2)
        except cv2.error:
            pass
    return out


def _pair_status(p: Mapping[str, Any]) -> str:
    if p.get("not_evaluated"):
        return "not_evaluated"
    n, ratio = p["inliers"], p["inlier_ratio"] or 0.0
    if n >= THRESHOLDS["verified_min_inliers"] and ratio >= THRESHOLDS["verified_min_inlier_ratio"]:
        return "verified"
    return "weak" if n >= THRESHOLDS["weak_min_inliers"] and ratio >= THRESHOLDS["weak_min_inlier_ratio"] else "none"


def _pair_low_parallax(p: Mapping[str, Any]) -> bool | None:
    """True / False when decidable from the pair, None when neither number exists."""
    h, t = p.get("homography_fraction"), p.get("triangulation_deg")
    if (t is not None and t < THRESHOLDS["low_parallax_tri_deg"]) or (h is not None and h >= THRESHOLDS["low_parallax_homography"]):
        return True
    return False if (t is not None or h is not None) else None


# ---------------------------------------------------------------------------------------------- the report
def assess_readiness(images: Sequence[tuple[str, Any]], *, labels: Mapping[str, str] | None = None,
                     registered_ids: Sequence[str] | None = None, budget_s: float | None = None,
                     max_pairs: int | None = None) -> dict:
    """``images`` = [(id, path)] in capture order. Returns the JSON-ready report (pre-registration unless ``registered_ids`` is
    given). Deterministic for fixed pixels; runtime bounded by ``budget_s``: pairs left unevaluated are reported as such and make
    the verdict ``unknown`` rather than ``ready``."""
    t0 = time.monotonic()
    budget = THRESHOLDS["budget_s"] if budget_s is None else budget_s
    max_pairs = int(THRESHOLDS["max_pairs"] if max_pairs is None else max_pairs)
    labels = dict(labels or {})
    ids = [i for i, _ in images]
    base: dict[str, Any] = {"schema": SCHEMA, "thresholds": dict(THRESHOLDS), "provisional_note": PROVISIONAL_NOTE,
                            "images": [], "pairs": [], "pairing": {}, "duplicates": [], "intrinsics": {}, "graph": {},
                            "diversity": {}, "cannot_decide": [],
                            "pose_consistency": {"state": "not_available", "reason": "no sparse model supplied"}}
    if cv2 is None:
        return _finish_blocked(base, ids, "OpenCV is not installed, so no photograph could be analysed.", t0)

    recs: dict[str, dict] = {}
    feats: dict[str, dict] = {}
    sift = cv2.SIFT_create(  # type: ignore[attr-defined]
        nfeatures=int(THRESHOLDS["sift_features"]))
    for idx, (iid, path) in enumerate(images):
        rec: dict[str, Any] = {"id": iid, "label": labels.get(iid, iid), "index": idx + 1, "readable": False}
        base["images"].append(rec)
        got = _read_gray_and_meta(Path(path))
        if got is None:
            rec["reason"] = "could not be decoded as an image"
            continue
        g = got["gray"]
        m = photo_metrics(g)
        verdict, reasons = judge_photo(m)
        f_px, f_src = _focal_prior_px(got["width"], got["height"], got["exif"])
        rec.update(readable=True, width=got["width"], height=got["height"], size=f"{got['width']}x{got['height']}",
                   aspect=_f(got["width"] / got["height"]), exif=got["exif"], metrics=m, quality=verdict, quality_reasons=reasons,
                   phash=f"{_phash(g):016x}", focal_prior_px=_f(f_px, 1), focal_prior_source=f_src)
        rec["f_analysis"] = f_px * got["scale"]
        recs[iid] = rec
        if time.monotonic() - t0 > budget:
            rec["features"] = "not_computed_budget"
            continue
        kps, des = sift.detectAndCompute(g, None)
        feats[iid] = {"des": des, "pts": np.array([k.pt for k in kps], np.float32).reshape(-1, 2), "kp_size": [k.size for k in kps],
                      "wh": np.array([g.shape[1], g.shape[0]], np.float32)}
        rec["n_keypoints"] = len(kps)

    readable = [i for i in ids if i in recs]
    if not readable:
        return _finish_blocked(base, ids, "None of the photographs could be decoded as an image.", t0)

    lap = [recs[i]["metrics"]["blur_laplacian_var"] for i in readable]
    med = float(np.median(lap))   # a uniformly blurry set has no outlier, so the absolute verdict still applies
    for i in readable:
        v = recs[i]["metrics"]["blur_laplacian_var"]
        recs[i]["blur_relative_to_median"] = _f(v / med if med > 0 else None)
        recs[i]["blurred"] = bool(v < BLUR_DROP_BELOW or (v < BLUR_WARN_BELOW and v < THRESHOLDS["blur_relative"] * med))

    # all pairs when they fit, else a +-window in capture order; nearest in capture order first so a budget cut keeps the most
    # informative pairs
    order = {i: k for k, i in enumerate(readable)}
    allp = sorted(itertools.combinations(readable, 2), key=lambda ab: (order[ab[1]] - order[ab[0]], order[ab[0]]))
    mode = "all"
    if len(allp) > max_pairs:
        mode = "window"
        allp = [p for p in allp if order[p[1]] - order[p[0]] <= int(THRESHOLDS["window"])][:max_pairs]
    rows: dict[tuple[str, str], dict] = {}
    for a, b in allp:
        row: dict[str, Any] = {"a": a, "b": b}
        if a not in feats or b not in feats or time.monotonic() - t0 > budget:
            row["not_evaluated"] = True
        else:
            row.update(_match_pair(feats[a], feats[b], recs[a], recs[b]))
        row["status"] = _pair_status(row)
        rows[(a, b)] = row
    base["pairs"] = [rows[k] for k in allp]
    n_eval = sum(1 for r in rows.values() if r["status"] != "not_evaluated")
    base["pairing"] = {"mode": mode, "possible": len(readable) * (len(readable) - 1) // 2, "listed": len(allp), "evaluated": n_eval,
                       "budget_s": budget, "budget_exhausted": n_eval < len(allp)}
    _intrinsics_and_duplicates(base, recs, readable, rows)
    status, reason = _judge(base, ids, registered_ids)
    return _finish(base, ids, status, reason, registered_ids, t0)


def _intrinsics_and_duplicates(base: dict, recs: dict, readable: Sequence[str], rows: dict) -> None:
    sizes: dict[str, list[str]] = {}
    aspects: dict[str, list[str]] = {}
    cams: dict[str, list[str]] = {}
    for i in readable:
        r = recs[i]
        sizes.setdefault(r["size"], []).append(i)
        aspects.setdefault(f"{r['aspect']:.2f}", []).append(i)
        ex = r["exif"]
        if ex.get("make") or ex.get("model"):
            cams.setdefault(f"{ex.get('make') or ''} {ex.get('model') or ''}".strip(), []).append(i)
    n_exif = sum(1 for i in readable if recs[i]["exif"]["present"])
    n_focal = sum(1 for i in readable if recs[i]["exif"].get("focal_35mm") or recs[i]["exif"].get("focal_mm"))
    hints: list[str] = []
    if len(sizes) > 1:
        hints.append("photographs have different pixel sizes (different cameras, crops or generators); each size is solved as its own camera")
    if len(aspects) > 1:
        hints.append("photographs have different aspect ratios: some were cropped or come from a different source")
    if len(cams) > 1:
        hints.append("EXIF says more than one camera model was used")
    if 0 < n_exif < len(readable):
        hints.append(f"only {n_exif} of {len(readable)} photographs carry EXIF")
    if n_focal == 0:
        hints.append("no photograph states a focal length, so the lens is unknown (COLMAP will assume 1.2 x the longer side)")
    lo, hi = 1 / THRESHOLDS["crop_scale_ratio"], THRESHOLDS["crop_scale_ratio"]
    for (a, b), p in rows.items():
        sr = p.get("scale_ratio")
        if p["status"] == "verified" and recs[a]["size"] != recs[b]["size"] and sr and not lo <= sr <= hi:
            hints.append(f"{recs[a]['label']} / {recs[b]['label']}: matching features differ in scale by x{sr:.2f} across different "
                         "sizes (crop or zoom, or a different distance; not distinguishable here)")
    base["intrinsics"] = {
        "size_groups": sizes, "aspect_groups": aspects, "camera_groups_exif": cams, "n_with_exif": n_exif, "n_with_focal": n_focal,
        "mixed_size": len(sizes) > 1,
        "distortion": {"state": "unknown", "note": "lens distortion cannot be measured from unlabelled photographs by this check"},
        "hints": hints}
    dup_edges, dup_rows = [], []
    for a, b in itertools.combinations(readable, 2):
        p = rows.get((a, b))
        hd = bin(int(recs[a]["phash"], 16) ^ int(recs[b]["phash"], 16)).count("1")
        confirmed = bool(p and p["status"] == "verified" and p["median_shift_fraction"] is not None
                         and p["median_shift_fraction"] <= THRESHOLDS["duplicate_shift_fraction"])
        if confirmed or hd <= THRESHOLDS["duplicate_hash_bits"]:
            state = "confirmed" if confirmed else ("not_a_duplicate" if p and p["status"] != "not_evaluated" else "unconfirmed")
            dup_rows.append({"a": a, "b": b, "hash_distance": hd, "state": state,
                             "median_shift_fraction": p.get("median_shift_fraction") if p else None})
            if confirmed:
                dup_edges.append((a, b))
    keep_of: dict[str, str] = {}
    for grp in _components(list(readable), dup_edges):
        keep = max(grp, key=lambda i: recs[i]["metrics"]["blur_laplacian_var"])
        keep_of.update({i: keep for i in grp if i != keep})
    for i in readable:
        recs[i]["duplicate_of"] = keep_of.get(i)
    base["duplicates"] = dup_rows


def _spread(sel: Sequence[Mapping[str, Any]]) -> dict:
    """Viewpoint diversity of a set of VERIFIED pairs."""
    tri = [p["triangulation_deg"] for p in sel if p["triangulation_deg"] is not None]
    rot = [p["rotation_deg"] for p in sel if p["rotation_deg"] is not None]
    sh = [p["median_shift_fraction"] for p in sel if p["median_shift_fraction"] is not None]
    low = [_pair_low_parallax(p) for p in sel]
    out: dict[str, Any] = {
        "verified_pairs": len(sel), "pairs_with_pose": len(tri), "intrinsics_assumed": True,
        "median_shift_fraction": _f(np.median(sh), 4) if sh else None,
        "triangulation_deg_median": _f(np.median(tri), 2) if tri else None, "triangulation_deg_p90": _f(np.percentile(tri, 90), 2) if tri else None,
        "rotation_deg_median": _f(np.median(rot), 2) if rot else None, "rotation_deg_max": _f(max(rot), 2) if rot else None,
        "low_parallax_fraction": _f(sum(1 for x in low if x) / len(low)) if sel else None}
    if len(sel) < 2 or not tri:
        out["verdict"], out["note"] = UNKNOWN, "fewer than 2 verified pairs with a pose estimate; baseline and rotation spread cannot be judged"
    elif out["triangulation_deg_p90"] < THRESHOLDS["low_parallax_tri_deg"]:
        out["verdict"], out["note"] = "narrow_baseline", "almost every verified pair has a very small triangulation angle: move sideways between shots, not just turn"
    elif out["low_parallax_fraction"] >= 0.75:
        out["verdict"], out["note"] = "mostly_low_parallax", "most verified pairs are explained by one homography (rotation-only or a single plane)"
    else:
        out["verdict"], out["note"] = "adequate", "verified pairs show usable parallax"
    return out


def _judge(base: dict, ids: Sequence[str], registered_ids: Sequence[str] | None) -> tuple[str, str | None]:
    """Fills graph, diversity, per-image diagnosis and cannot_decide from the stored image/pair facts; returns (status, reason).
    Pure over ``base`` so ``after_registration`` can re-run it without pixels."""
    recs = {r["id"]: r for r in base["images"] if r.get("readable")}
    readable = [i for i in ids if i in recs]
    rows = {(p["a"], p["b"]): p for p in base["pairs"]}

    def pair(a: str, b: str) -> dict | None:
        return rows.get((a, b)) or rows.get((b, a))

    verified = [k for k, p in rows.items() if p["status"] == "verified"]
    usable = [i for i in readable if not recs[i].get("duplicate_of")]
    vv = [(a, b) for a, b in verified if a in usable and b in usable]
    comps = _components(usable, vv)
    bridges, weak_bridges = [], []
    if len(vv) <= 3000:   # ponytail: O(E*(V+E)) bridge test; switch to Tarjan low-link beyond a few thousand edges
        for e in vv:
            comp = next(c for c in comps if e[0] in c)
            if len(_components(comp, [x for x in vv if x != e and x[0] in comp])) > 1:
                row = {"a": e[0], "b": e[1], "inliers": rows[e]["inliers"]}
                bridges.append(row)
                if row["inliers"] < THRESHOLDS["weak_bridge_inliers"]:
                    weak_bridges.append(row)
    base["graph"] = {"components": comps, "n_components": len(comps), "largest": len(comps[0]) if comps else 0,
                     "verified_pairs": len(verified), "bridges": bridges, "weak_bridges": weak_bridges,
                     "weak_pairs": [{"a": a, "b": b, "inliers": p["inliers"]} for (a, b), p in rows.items() if p["status"] == "weak"],
                     "bridge_check": "done" if len(vv) <= 3000 else "skipped_too_many_edges"}
    base["diversity"] = {"overall": _spread([rows[k] for k in verified]),
                         "per_component": [_spread([rows[k] for k in verified if k[0] in c]) for c in comps]}
    registered = set(registered_ids) if registered_ids is not None else None
    reg_comp = {i for c in comps if registered and any(x in registered for x in c) for i in c}
    cannot: list[str] = []
    for r in base["images"]:
        if not r.get("readable"):
            r["diagnosis"] = {"cause": "unreadable", "certainty": "measured", "evidence": {"reason": r.get("reason")},
                              "predicted_registration": "unlikely"}
            continue
        i = r["id"]
        nb = [(j, p) for j in readable if j != i for p in [pair(i, j)] if p is not None]
        evald = [p for _, p in nb if p["status"] != "not_evaluated"]
        n_poss = len(readable) - 1
        links = sorted(((p["inliers"], j) for j, p in nb if p["status"] == "verified"), reverse=True)
        best = max((p["inliers"] for p in evald), default=0)
        ev: dict[str, Any] = {"verified_links": [{"image": recs[j]["label"], "inliers": n} for n, j in links[:5]],
                              "n_verified_links": len(links), "best_inliers": best, "pairs_evaluated": f"{len(evald)}/{n_poss}",
                              "size": r["size"], "sharpness": r["metrics"]["blur_laplacian_var"]}
        complete = len(evald) == n_poss
        cause, cert = None, "measured"
        if r.get("duplicate_of"):
            cause, ev["duplicate_of"] = "duplicate", recs[r["duplicate_of"]]["label"]
        elif r.get("blurred"):
            cause = "blurred"
        elif n_poss == 0:
            pass        # a lone photograph has nothing to overlap with; handled at the status level
        elif not links:
            if any(p["status"] == "weak" for p in evald):
                cause = "no_verified_overlap"
            elif complete:
                cause = "isolated"
            else:
                cause, cert = "unknown", "unknown"
                cannot.append(f"{r['label']}: only {len(evald)}/{n_poss} pairs were evaluated, so isolation cannot be confirmed")
        else:
            nb_sizes = {recs[j]["size"] for _, j in links}
            lows = [_pair_low_parallax(rows.get((i, j)) or rows[(j, i)]) for _, j in links]
            if len(base["intrinsics"]["size_groups"]) > 1 and r["size"] not in nb_sizes:
                cause, cert, ev["neighbour_sizes"] = "intrinsics_mismatch", "candidate", sorted(nb_sizes)
            elif all(x is True for x in lows):
                cause = "too_low_parallax"
            elif registered is not None and i not in registered:
                cause, cert = ("disconnected_group", "measured") if reg_comp and i not in reg_comp else ("other", "unexplained")
        if cause in ("isolated", "no_verified_overlap", "blurred"):
            pred = "unlikely"
        elif cause == "unknown" or not complete:
            pred = "unknown"
        elif cause == "duplicate":
            pred = "redundant"
        else:
            pred = "likely" if links else "unknown"
        if registered is not None:
            r["registered"] = i in registered
            if i in registered:
                cause, cert = None, "measured"
        r["diagnosis"] = {"cause": cause, "certainty": cert, "evidence": ev, "predicted_registration": pred}

    div = base["diversity"]["overall"]["verdict"]
    pairing = base["pairing"]
    incomplete = pairing["mode"] == "window" or pairing["budget_exhausted"]
    problems = [r for r in base["images"] if (r.get("diagnosis") or {}).get("cause") in _PROBLEM_CAUSES]
    unknowns = [r for r in base["images"] if (r.get("diagnosis") or {}).get("cause") == "unknown"]
    base["cannot_decide"] = cannot + ["lens distortion (not measurable here)"] + \
        (["viewpoint spread (too few verified pairs with a pose estimate)"] if div == UNKNOWN else []) + \
        (["pairs outside the +-window were not evaluated"] if pairing["mode"] == "window" else []) + \
        (["pairs not evaluated before the time budget ran out"] if pairing["budget_exhausted"] else [])
    if registered is not None:
        missed = [i for i in usable if i not in registered]
        if len(registered) >= 2 and not missed:
            return READY, "every usable photograph was registered into one shared frame"
        return NEEDS_MORE, f"{len(registered)} of {len(usable)} usable photographs were registered; {len(missed)} are not placed in the shared frame"
    if len(usable) < 2 and not problems:
        return NEEDS_MORE, "only one distinct usable photograph: a shared 3D frame needs at least two overlapping photographs"
    if problems:
        return NEEDS_MORE, "some photographs have no verified overlap or cannot be used"
    if unknowns or incomplete:   # component / bridge claims are only definite when every pair was evaluated
        return UNKNOWN, "the checks could not decide: not every pair was evaluated"
    if len(comps) > 1 or weak_bridges:
        return NEEDS_MORE, (f"the photographs form {len(comps)} separate groups" if len(comps) > 1 else "the photographs are held together by weak links")
    if div == UNKNOWN:
        return UNKNOWN, "the checks could not decide: viewpoint spread is undecidable"
    if div in ("narrow_baseline", "mostly_low_parallax"):
        return NEEDS_MORE, base["diversity"]["overall"]["note"]
    return READY, "all usable photographs are linked by verified overlap with adequate parallax (a prediction, not a registration)"


def _frame(ids: Sequence[str], registered_ids: Sequence[str] | None) -> dict:
    if registered_ids is None:
        return {"shared_frame": None, "registered_ids": None, "unregistered_ids": None,
                "note": "registration has not been attempted yet; no shared frame exists"}
    reg = [i for i in ids if i in set(registered_ids)]
    un = [i for i in ids if i not in set(registered_ids)]
    return {"shared_frame": len(reg) >= 2, "registered_ids": reg, "unregistered_ids": un,
            "unregistered_label": "independent single-image hypotheses: relative placement unknown, not part of the shared frame",
            "note": ("one photograph carries its own assumed pose only; there is no shared frame" if len(reg) == 1 else
                     "no photograph was placed by the multi-view solver" if not reg else
                     "registered photographs share one frame; the unregistered ones are not in it" if un else "all photographs share one frame")}


def _actions(base: dict, ids: Sequence[str]) -> list[dict]:
    recs = {r["id"]: r for r in base["images"]}
    nm = lambda xs: ", ".join(recs[x]["label"] for x in xs)  # noqa: E731
    by: dict[str, list[str]] = {}
    for i in ids:
        c = (recs[i].get("diagnosis") or {}).get("cause")
        if c and c != "duplicate":
            by.setdefault(c, []).append(i)
    acts: list[dict] = []
    if "isolated" in by:
        acts.append({"kind": "isolated", "images": by["isolated"],
                     "text": f"{nm(by['isolated'])} do not overlap any other photograph. Take photographs that overlap by about 60% with the "
                             "existing set and with these, standing between them and the rest."})
    if "no_verified_overlap" in by:
        acts.append({"kind": "no_verified_overlap", "images": by["no_verified_overlap"],
                     "text": f"{nm(by['no_verified_overlap'])} match other photographs only weakly (no verified overlap). Add one or two "
                             "photographs from positions between them and the rest, about 60% overlap each."})
    multi = [c for c in base["graph"]["components"] if len(c) > 1]
    if len(multi) > 1:
        acts.append({"kind": "disconnected_groups", "images": [x for c in multi for x in c],
                     "text": f"The photographs fall into {len(multi)} separate groups ({'; '.join(nm(c) for c in multi)}) with no overlap between "
                             "groups. Take 2-3 photographs that show part of each group (about 60% overlap with both)."})
    for wb in base["graph"]["weak_bridges"]:
        acts.append({"kind": "weak_bridge", "images": [wb["a"], wb["b"]],
                     "text": f"{recs[wb['a']]['label']} and {recs[wb['b']]['label']} are the only link between two parts of the capture and it is "
                             f"weak ({wb['inliers']} matched points). Take extra photographs between them."})
    if "disconnected_group" in by:
        acts.append({"kind": "disconnected_group", "images": by["disconnected_group"],
                     "text": f"{nm(by['disconnected_group'])} overlap each other but not the registered photographs. Take photographs that show part of both."})
    low = by.get("too_low_parallax", [])
    if low or base["diversity"]["overall"].get("verdict") in ("narrow_baseline", "mostly_low_parallax"):
        acts.append({"kind": "low_parallax", "images": low,
                     "text": (f"{nm(low)}: " if low else "") + "the viewpoints barely change. Move 0.5-1 m sideways between shots instead of only turning."})
    for kind, text in (("blurred", "Retake {}: too blurry or without texture (hold still, include edges)."),
                       ("intrinsics_mismatch", "{} differ in size from the photographs they overlap with (possible different camera or crop). Use one camera and no cropping."),
                       ("other", "{} overlap other photographs but were not registered; this check cannot say why. Retake from new positions."),
                       ("unreadable", "{} could not be opened as images.")):
        if kind in by:
            acts.append({"kind": kind if kind != "other" else "unexplained", "images": by[kind], "text": text.format(nm(by[kind]))})
    dups = [i for i in ids if recs[i].get("duplicate_of")]
    if dups:
        acts.append({"kind": "duplicate", "images": dups, "text": f"{nm(dups)} repeat another photograph's viewpoint and add nothing; they are not counted."})
    intr = base["intrinsics"]
    if intr.get("mixed_size"):
        acts.append({"kind": "mixed_size", "images": [], "text": "Photographs come in different sizes (" +
                     ", ".join(f"{k}: {len(v)}" for k, v in intr["size_groups"].items()) + "). One camera with one setting registers best."})
    if intr.get("n_with_focal") == 0:
        acts.append({"kind": "no_exif", "images": [], "text": "No photograph carries lens (EXIF) information, so the focal length is unknown. "
                     "Original camera files, not screenshots or generated images, give the best result."})
    return acts


def _message(status: str, reason: str | None, acts: list[dict], frame: Mapping[str, Any]) -> str:
    head = {READY: "Capture is ready: ", NEEDS_MORE: "More photographs are needed: ", BLOCKED: "Cannot assess this capture: ",
            UNKNOWN: "Capture readiness is undecided: "}[status] + (reason or "") + "."
    if frame.get("unregistered_ids"):
        head += (f" The {len(frame['unregistered_ids'])} unregistered photograph(s) stay independent single-photo hypotheses "
                 "and are NOT placed in a shared 3D frame.")
    return "\n".join([head, *(f"- {a['text']}" for a in acts)])


def _finish(base: dict, ids: Sequence[str], status: str, reason: str | None, registered_ids: Sequence[str] | None, t0: float) -> dict:
    base["stage"] = "pre_registration" if registered_ids is None else "post_registration"
    base["status"], base["reason"] = status, reason
    base["frame"] = _frame(ids, registered_ids)
    base["actions"] = _actions(base, ids)
    base["user_message"] = _message(status, reason, base["actions"], base["frame"])
    base["summary"] = {"n_images": len(ids), "n_readable": sum(1 for r in base["images"] if r.get("readable")),
                       "n_components": base["graph"]["n_components"], "verified_pairs": base["graph"]["verified_pairs"],
                       "pairs_evaluated": base["pairing"]["evaluated"], "pairs_possible": base["pairing"]["possible"]}
    base["timing"] = {"runtime_s": round(time.monotonic() - t0, 2)}
    for r in base["images"]:
        r.pop("f_analysis", None)
    return base


def _finish_blocked(base: dict, ids: Sequence[str], reason: str, t0: float) -> dict:
    base["graph"] = {"components": [], "n_components": 0, "largest": 0, "verified_pairs": 0, "bridges": [], "weak_bridges": [], "weak_pairs": []}
    base["diversity"] = {"overall": {"verdict": UNKNOWN}, "per_component": []}
    base["intrinsics"] = {"mixed_size": False, "n_with_focal": 0, "hints": []}
    base["pairing"] = {"mode": "none", "possible": 0, "listed": 0, "evaluated": 0, "budget_exhausted": False}
    for r in base["images"]:
        r["diagnosis"] = {"cause": "unreadable", "certainty": "measured", "evidence": {"reason": r.get("reason")}}
    return _finish(base, ids, BLOCKED, reason, None, t0)


def after_registration(report: Mapping[str, Any], registered_ids: Sequence[str], sparse: Mapping[str, Any] | None = None) -> dict:
    """The same report re-judged with the multi-view outcome: a cause for every photograph that did not register, the explicit
    ``frame`` (shared or not) and, when a sparse model is supplied, its pose-consistency statistics. No pixels are read."""
    t0 = time.monotonic()
    base = copy.deepcopy(dict(report))
    for k in ("artifact", "timing", "status", "reason", "user_message", "actions", "frame", "summary", "stage"):
        base.pop(k, None)
    ids = [r["id"] for r in base["images"]]
    status, reason = _judge(base, ids, registered_ids)
    base["pose_consistency"] = (pose_consistency(sparse, base) if sparse else
                                {"state": "not_available", "reason": "no sparse model was supplied (nothing registered or model unreadable)"})
    pc = base["pose_consistency"]
    if status == READY and pc.get("verdict") == "inconsistent":
        status, reason = UNKNOWN, "registration succeeded but its own geometry looks inconsistent: " + pc["note"]
    return _finish(base, ids, status, reason, registered_ids, t0)


def pose_consistency(sparse: Mapping[str, Any], base: Mapping[str, Any] | None = None) -> dict:
    """Reprojection and triangulation statistics of a sparse model, plus agreement of its relative rotations with this module's own
    independent pair estimates. ``sparse`` = {"cameras": {id: {"K": 3x3, "R": 3x3 world->cam, "t": 3}},
    "points": [{"xyz": [x,y,z], "track": [(id, x, y), ...]}]} (see ``load_colmap_text_model``)."""
    cams, pts = sparse.get("cameras") or {}, sparse.get("points") or []
    if len(cams) < 2 or not pts:
        return {"state": "not_available", "reason": f"model has {len(cams)} camera(s) and {len(pts)} point(s); statistics need >=2 cameras and points"}
    reproj: dict[str, list[float]] = {c: [] for c in cams}
    tri_deg, track_len = [], []
    for p in pts:
        X = np.asarray(p["xyz"], float)
        centres = []
        for cid, x, y in p["track"]:
            c = cams.get(cid)
            if c is None:
                continue
            R, t = np.asarray(c["R"], float), np.asarray(c["t"], float)
            Xc = R @ X + t
            if Xc[2] <= 0:
                reproj[cid].append(float("inf"))
                continue
            uv = np.asarray(c["K"], float) @ (Xc / Xc[2])
            reproj[cid].append(float(math.hypot(uv[0] - x, uv[1] - y)))
            centres.append(-R.T @ t)
        track_len.append(len(p["track"]))
        if len(centres) >= 2:
            rays = [(X - c) / (np.linalg.norm(X - c) + 1e-12) for c in centres]
            tri_deg.append(max(math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b))))) for a, b in itertools.combinations(rays, 2)))
    fin = [x for v in reproj.values() for x in v if np.isfinite(x)]
    out: dict[str, Any] = {
        "state": "ok", "n_cameras": len(cams), "n_points": len(pts),
        "per_camera": {c: {"observations": len(v), "median_reprojection_px": _f(np.median([x for x in v if np.isfinite(x)]), 2) if any(np.isfinite(v)) else None}
                       for c, v in reproj.items()},
        "reprojection_px_median": _f(np.median(fin), 2) if fin else None, "reprojection_px_p95": _f(np.percentile(fin, 95), 2) if fin else None,
        "triangulation_deg_median": _f(np.median(tri_deg), 2) if tri_deg else None,
        "fraction_points_under_2deg": _f(sum(1 for a in tri_deg if a < 2.0) / len(tri_deg)) if tri_deg else None,
        "track_length_median": _f(np.median(track_len), 1), "fraction_tracks_len_ge3": _f(sum(1 for n in track_len if n >= 3) / len(track_len))}
    notes = []
    if out["reprojection_px_p95"] is not None and out["reprojection_px_p95"] > 4.0:
        notes.append(f"95th-percentile reprojection error {out['reprojection_px_p95']} px")
    if out["fraction_points_under_2deg"] is not None and out["fraction_points_under_2deg"] > 0.5:
        notes.append(f"{out['fraction_points_under_2deg']:.0%} of points are triangulated at <2 degrees (weakly constrained depth)")
    if out["fraction_tracks_len_ge3"] < 0.3:
        notes.append(f"only {out['fraction_tracks_len_ge3']:.0%} of points are seen by 3 or more cameras")
    if base is not None:   # relative rotations of the sparse model vs this module's independent (assumed-focal, so loose) estimates
        diffs = []
        for p in base.get("pairs", []):
            a, b, rd = cams.get(p["a"]), cams.get(p["b"]), p.get("rotation_deg")
            if a and b and rd is not None and p["status"] == "verified":
                Rrel = np.asarray(b["R"], float) @ np.asarray(a["R"], float).T
                diffs.append(abs(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rrel) - 1) / 2)))) - rd))
        out["rotation_agreement_deg_median"], out["rotation_agreement_pairs"] = (_f(np.median(diffs), 2) if diffs else None), len(diffs)
    out["verdict"] = "inconsistent" if len(notes) >= 2 else "weak" if notes else "consistent"
    out["note"] = "; ".join(notes) or "no reprojection/triangulation red flags"
    return out


def load_colmap_text_model(model_dir: str | Path, name_to_id: Mapping[str, str] | None = None) -> dict | None:
    """Parse a COLMAP TXT model (cameras/images/points3D.txt) into ``pose_consistency`` input. ``name_to_id`` maps an image file
    name or stem to an evidence id (default: the stem). None when the files are missing or unparsable (never invented)."""
    d = Path(model_dir)
    try:
        cam_k: dict[int, np.ndarray] = {}
        for ln in (d / "cameras.txt").read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            par = [float(x) for x in t[4:]]
            if t[1] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE"):
                fx = fy = par[0]
                cx, cy = par[1], par[2]
            elif t[1] in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
                fx, fy, cx, cy = par[:4]
            else:
                continue
            cam_k[int(t[0])] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        lines = [ln for ln in (d / "images.txt").read_text().splitlines() if not ln.startswith("#")]
        cams: dict[str, dict] = {}
        image_id: dict[int, str] = {}
        for hdr in lines[0::2]:
            t = hdr.split()
            if len(t) < 10 or int(t[8]) not in cam_k:
                continue
            w, x, y, z = np.array([float(v) for v in t[1:5]]) / np.linalg.norm([float(v) for v in t[1:5]])
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            iid = (name_to_id or {}).get(t[9]) or (name_to_id or {}).get(Path(t[9]).stem) or Path(t[9]).stem
            cams[iid] = {"K": cam_k[int(t[8])], "R": R, "t": np.array([float(v) for v in t[5:8]])}
            image_id[int(t[0])] = iid
        obs: dict[tuple[str, int], tuple[float, float]] = {}
        for hdr, pl in zip(lines[0::2], lines[1::2]):
            oid = image_id.get(int(hdr.split()[0]))
            if oid is None:
                continue
            v = pl.split()
            for k in range(0, len(v) - 2, 3):
                obs[(oid, k // 3)] = (float(v[k]), float(v[k + 1]))
        pts = []
        for ln in (d / "points3D.txt").read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            track = [(image_id.get(int(t[k])), int(t[k + 1])) for k in range(8, len(t) - 1, 2)]
            pts.append({"xyz": [float(v) for v in t[1:4]],
                        "track": [(i, *obs[(i, j)]) for i, j in track if i is not None and (i, j) in obs]})
        return {"cameras": cams, "points": pts}
    except Exception:  # noqa: BLE001 -- an unreadable model means "not available", never invented statistics
        return None


# ---------------------------------------------------------------------------------------------- artifact + job glue
def canonical_bytes(report: Mapping[str, Any]) -> bytes:
    """Stable JSON of the report without run-dependent fields (timing, artifact pointer)."""
    body = {k: v for k, v in report.items() if k not in ("timing", "artifact")}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def store_report(report: dict, store: Any | None) -> dict:
    """Attach ``artifact`` = {sha256, uri, bytes} (hash of the canonical JSON); also stores the bytes when a store is given."""
    data = canonical_bytes(report)
    digest, uri = hashlib.sha256(data).hexdigest(), None
    if store is not None:
        try:
            uri, digest = store.put(data)
        except Exception:  # noqa: BLE001 -- losing the artifact copy must not lose the report
            uri = None
    report["artifact"] = {"sha256": digest, "uri": uri, "bytes": len(data)}
    return report


def stage_entry(report: Mapping[str, Any] | None) -> dict | None:
    """The small ``stages.capture_readiness`` entry (status, reason, counts, artifact pointer) for the job report."""
    if not report:
        return None
    return {"status": report["status"], "reason": report.get("reason"), "stage": report.get("stage"), "summary": report.get("summary"),
            "shared_frame": (report.get("frame") or {}).get("shared_frame"), "artifact": report.get("artifact")}


def for_job(entries: Sequence[tuple[str, Any, str]], store: Any | None = None) -> dict:
    """Pre-registration report for the production job path. ``entries`` = [(evidence_id, path, display name)]. Never raises: a
    failure becomes a ``blocked`` report carrying the error, and the job continues (the check is advisory)."""
    try:
        rep = assess_readiness([(i, p) for i, p, _ in entries], labels={i: n for i, _, n in entries})
    except Exception as exc:  # noqa: BLE001 -- advisory stage: record, never fail the job
        rep = {"schema": SCHEMA, "stage": "pre_registration", "status": BLOCKED,
               "reason": f"readiness check failed: {type(exc).__name__}: {exc}", "user_message": "Capture readiness could not be assessed.",
               "actions": [], "images": [], "pairs": [], "summary": {}, "frame": {"shared_frame": None}}
    return store_report(rep, store)


def for_job_after(pre: dict | None, registered_ids: Sequence[str], model_dir: str | Path | None = None,
                  name_to_id: Mapping[str, str] | None = None, store: Any | None = None) -> dict | None:
    """Post-registration report; ``pre`` None (check disabled/failed) gives None. Never raises."""
    if not pre or not pre.get("images"):
        return pre
    try:
        sparse = load_colmap_text_model(model_dir, name_to_id) if model_dir else None
        return store_report(after_registration(pre, registered_ids, sparse), store)
    except Exception as exc:  # noqa: BLE001
        return {**pre, "post_registration_error": f"{type(exc).__name__}: {exc}"}
