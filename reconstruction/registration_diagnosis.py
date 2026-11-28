"""Explain a COLMAP run from its own database + sparse model: which images registered, why the others did not, and how trustworthy the geometry is.

Everything here is MEASURED from `database.db` (keypoints, raw matches, two-view geometries) and `sparse/<n>/` (poses, intrinsics, 3-D tracks); nothing is
inferred from a log that was not kept. The output is a plain dict (JSON-able) so it can be persisted beside the run, attached to a failure
explanation, and diffed between runs.

A pair counts as *verified* when its two-view geometry kept at least `MIN_VERIFIED_INLIERS` inliers (COLMAP's own default for a usable pair is 15).
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_VERIFIED_INLIERS = 15
# COLMAP's TwoViewGeometry::ConfigurationType: 0 UNDEFINED, 1 DEGENERATE, 2 CALIBRATED, 3 UNCALIBRATED, 4 PLANAR, 5 PANORAMIC, 6 PLANAR_OR_PANORAMIC, 7 WATERMARK, 8 MULTIPLE
_CONFIG_NAMES = {0: "undefined", 1: "degenerate", 2: "calibrated", 3: "uncalibrated", 4: "planar", 5: "panoramic", 6: "planar_or_panoramic", 7: "watermark", 8: "multiple"}
_PAIR_MOD = 2147483647
_GRID = 4  # spatial-coverage grid per image side


def _pair_ids(pair_id: int) -> tuple[int, int]:
    return pair_id // _PAIR_MOD, pair_id % _PAIR_MOD


def _blob(data: bytes | None, rows: int, cols: int, dtype) -> np.ndarray:
    if not data or rows == 0:
        return np.zeros((0, cols), dtype=dtype)
    return np.frombuffer(data, dtype=dtype).reshape(rows, cols)


@dataclass
class _Db:
    names: dict[int, str]
    keypoints: dict[int, np.ndarray]  # image_id -> (n,>=2) float32
    camera_of: dict[int, int]
    raw_matches: dict[tuple[int, int], int]
    verified: dict[tuple[int, int], dict]
    cameras: dict[int, dict]


def _read_database(db_path: Path) -> _Db:
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        names, camera_of = {}, {}
        for iid, name, cid in con.execute("select image_id, name, camera_id from images"):
            names[iid], camera_of[iid] = name, cid
        keypoints = {iid: _blob(data, rows, cols, np.float32) for iid, rows, cols, data in con.execute("select image_id, rows, cols, data from keypoints")}
        raw = {_pair_ids(pid): rows for pid, rows in con.execute("select pair_id, rows from matches")}
        verified: dict[tuple[int, int], dict] = {}
        for pid, rows, cols, data, config in con.execute("select pair_id, rows, cols, data, config from two_view_geometries"):
            a, b = _pair_ids(pid)
            idx = _blob(data, rows, cols, np.uint32) if rows else np.zeros((0, 2), dtype=np.uint32)
            verified[(a, b)] = {"inliers": int(rows), "config": _CONFIG_NAMES.get(int(config), str(config)), "idx": idx}
        cameras = {}
        for cid, model, w, h, params in con.execute("select camera_id, model, width, height, params from cameras"):
            cameras[cid] = {"model_id": int(model), "width": int(w), "height": int(h),
                            "params": [round(float(x), 6) for x in np.frombuffer(params, dtype=np.float64)] if params else []}
        return _Db(names, keypoints, camera_of, raw, verified, cameras)
    finally:
        con.close()


def _coverage(kps: np.ndarray, idx_col: np.ndarray, w: float, h: float) -> float:
    """Fraction of a GRID x GRID image grid that holds at least one inlier keypoint (1.0 = matches spread over the whole frame)."""
    if len(idx_col) == 0 or w <= 0 or h <= 0:
        return 0.0
    xy = kps[idx_col, :2]
    gx = np.clip((xy[:, 0] / w * _GRID).astype(int), 0, _GRID - 1)
    gy = np.clip((xy[:, 1] / h * _GRID).astype(int), 0, _GRID - 1)
    return len(set(zip(gx.tolist(), gy.tolist(), strict=True))) / float(_GRID * _GRID)


def _components(nodes: list[int], edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = {n: n for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted(groups.values(), key=len, reverse=True)


def _triangulation_angles(rec) -> np.ndarray:
    """Largest viewing-ray angle (degrees) of every 3-D point; small angles mean a poorly conditioned depth."""
    centres = {iid: np.asarray(im.projection_center()) for iid, im in rec.images.items() if im.has_pose}
    out = []
    for p in rec.points3D.values():
        xyz = np.asarray(p.xyz)
        rays = []
        for el in p.track.elements:
            c = centres.get(el.image_id)
            if c is None:
                continue
            v = xyz - c
            n = np.linalg.norm(v)
            if n > 0:
                rays.append(v / n)
        if len(rays) < 2:
            out.append(0.0)
            continue
        r = np.asarray(rays)
        cos = np.clip(r @ r.T, -1.0, 1.0)
        out.append(float(np.degrees(np.arccos(cos.min()))))
    return np.asarray(out)


def _pct(a: np.ndarray, q: float) -> float | None:
    return round(float(np.percentile(a, q)), 3) if len(a) else None


def diagnose(session_dir: str | Path, *, expected_images: int | None = None) -> dict:
    """Diagnose the COLMAP session at `session_dir` (holds `database.db`, `images/`, `sparse/<model>/`). Returns a JSON-able report."""
    import pycolmap

    sd = Path(session_dir)
    db = _read_database(sd / "database.db")
    models = sorted(p for p in (sd / "sparse").glob("*") if p.is_dir()) if (sd / "sparse").exists() else []
    rec_list = []
    for m in models:
        try:
            rec_list.append((m.name, pycolmap.Reconstruction(str(m))))
        except Exception:  # noqa: BLE001 - an unreadable model is reported, not hidden
            rec_list.append((m.name, None))
    best_name, best = max(((n, r) for n, r in rec_list if r is not None), key=lambda t: t[1].num_reg_images(), default=(None, None))

    ids = sorted(db.names)
    reg_ids = [iid for iid in (best.reg_image_ids() if best else []) if iid in db.names]
    registered = set(reg_ids)
    # the registration order of a COLMAP model is its incremental order, so the first two images are the initial pair it seeded from
    init_pair = [db.names[i] for i in reg_ids[:2]] if len(reg_ids) >= 2 else None

    cam = None
    if best is not None and best.cameras:
        c = next(iter(best.cameras.values()))
        cam = {"model": str(c.model).split(".")[-1], "width": int(c.width), "height": int(c.height), "params": [round(float(x), 6) for x in c.params],
               "n_cameras": len(best.cameras), "prior_focal_length": bool(getattr(c, "has_prior_focal_length", False))}
    # without a model (a failed run) the database camera still gives the sensor size and the focal COLMAP started from
    db_cam = next(iter(db.cameras.values()), None)
    if cam is None and db_cam is not None:
        cam = {"model": f"colmap_model_id_{db_cam['model_id']}", "width": db_cam["width"], "height": db_cam["height"], "params": db_cam["params"],
               "n_cameras": len(db.cameras), "prior_focal_length": None, "source": "database (no sparse model was produced)"}
    wh = {iid: (cam["width"], cam["height"]) if cam else (0, 0) for iid in ids}

    verified = {k: v for k, v in db.verified.items() if v["inliers"] >= MIN_VERIFIED_INLIERS}
    pair_rows = []
    for (a, b), v in db.verified.items():
        raw = db.raw_matches.get((a, b), 0)
        covs = []
        if v["inliers"] and len(v["idx"]):
            covs = [_coverage(db.keypoints[a], v["idx"][:, 0], *wh[a]), _coverage(db.keypoints[b], v["idx"][:, 1], *wh[b])]
        pair_rows.append({"a": db.names[a], "b": db.names[b], "raw_matches": raw, "inliers": v["inliers"], "config": v["config"],
                          "inlier_ratio": round(v["inliers"] / raw, 4) if raw else None, "coverage": round(min(covs), 3) if covs else 0.0})
    ratios = np.asarray([p["inlier_ratio"] for p in pair_rows if p["inlier_ratio"] is not None and p["raw_matches"] >= MIN_VERIFIED_INLIERS])
    covs_all = np.asarray([p["coverage"] for p in pair_rows if p["inliers"] >= MIN_VERIFIED_INLIERS])

    degree = {i: 0 for i in ids}
    best_link: dict[int, tuple[int, int]] = {}  # image -> (inliers, neighbour) strongest verified link to a REGISTERED image
    for (a, b), v in verified.items():
        degree[a] += 1
        degree[b] += 1
    for (a, b), v in db.verified.items():
        for x, y in ((a, b), (b, a)):
            if y in registered and v["inliers"] > best_link.get(x, (0, 0))[0]:
                best_link[x] = (v["inliers"], y)
    comps = _components(ids, list(verified))
    comp_of = {i: n for n, comp in enumerate(comps) for i in comp}

    n_obs = {i: 0 for i in ids}
    err_sum = {i: 0.0 for i in ids}
    if best is not None:
        for p in best.points3D.values():
            for el in p.track.elements:
                if el.image_id in n_obs:
                    n_obs[el.image_id] += 1
                    err_sum[el.image_id] += float(p.error)

    images = []
    for i in ids:
        row = {"name": db.names[i], "keypoints": int(len(db.keypoints.get(i, []))), "registered": i in registered, "verified_neighbours": degree[i],
               "component": comp_of[i], "points3d_observed": n_obs[i] if i in registered else 0}
        if i in registered and n_obs[i]:
            row["mean_track_error_px"] = round(err_sum[i] / n_obs[i], 3)
        if i not in registered:
            bl = best_link.get(i)
            row["strongest_link_to_registered"] = {"image": db.names[bl[1]], "inliers": bl[0]} if bl else None
            row["why_unregistered"] = _why_unregistered(degree[i], bl, comp_of[i], comps, registered, db.keypoints.get(i))
        images.append(row)

    angles = _triangulation_angles(best) if best is not None else np.zeros(0)
    errs = np.asarray([float(p.error) for p in best.points3D.values()]) if best is not None else np.zeros(0)
    tracks = np.asarray([p.track.length() for p in best.points3D.values()]) if best is not None else np.zeros(0)
    n_total = len(ids)
    if expected_images is not None and expected_images != n_total:
        n_total_note = f"database holds {n_total} images but {expected_images} were expected"
    else:
        n_total_note = None
    return {
        "images_in_database": n_total, "images_registered": len(registered), "images_unregistered": n_total - len(registered), "note": n_total_note,
        "models_found": [{"name": n, "registered_images": r.num_reg_images() if r else None, "points3d": r.num_points3D() if r else None} for n, r in rec_list],
        "model_used": best_name, "initial_pair": init_pair, "camera": cam,
        "sparse": {"points3d": int(len(errs)), "mean_reproj_error_px": round(float(errs.mean()), 4) if len(errs) else None,
                   "median_track_length": float(np.median(tracks)) if len(tracks) else None,
                   "triangulation_angle_deg": {"p5": _pct(angles, 5), "median": _pct(angles, 50), "p95": _pct(angles, 95),
                                               "fraction_below_2deg": round(float((angles < 2.0).mean()), 3) if len(angles) else None}},
        "pairs": {"total_with_geometry": len(db.verified), "verified": len(verified), "min_verified_inliers": MIN_VERIFIED_INLIERS,
                  "inlier_ratio_median": _pct(ratios, 50), "inlier_ratio_p10": _pct(ratios, 10), "match_coverage_median": _pct(covs_all, 50),
                  "geometry_configs": _count([p["config"] for p in pair_rows])},
        "components": [{"size": len(c), "registered": sum(1 for i in c if i in registered), "images": [db.names[i] for i in c]} for c in comps],
        "keypoints_per_image": {"min": min((len(v) for v in db.keypoints.values()), default=0),
                                "median": int(statistics.median([len(v) for v in db.keypoints.values()])) if db.keypoints else 0},
        "images": images,
    }


def _count(xs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _why_unregistered(degree: int, best_link, comp: int, comps: list[list[int]], registered: set[int], kps) -> str:
    n_kp = 0 if kps is None else len(kps)
    if n_kp < 100:
        return f"too few features ({n_kp} keypoints): the image is blurred, textureless or tiny"
    if degree == 0 and best_link is None:
        return "no pair of this image verified geometrically with any other image: no overlap with the rest of the capture, or a pure rotation"
    if not any(i in registered for i in comps[comp]):
        return "its connected group of overlapping photos never registered (the group itself failed to initialise or lacked a wide baseline)"
    if best_link is None or best_link[0] < MIN_VERIFIED_INLIERS:
        return "overlaps only unregistered photos, or its link to the registered set is weaker than a usable pair"
    return (f"verified overlap exists ({best_link[0]} inliers with a registered photo) but too few of its matches hit already-triangulated 3-D points "
            "for a stable pose: the overlap is on structure that was never triangulated (narrow baseline or sparse track)")


def capture_guidance(report: dict) -> list[str]:
    """Plain-language advice derived from the measured report. Empty when the capture is sound."""
    out: list[str] = []
    n, reg = report["images_in_database"], report["images_registered"]
    if reg < 3:
        out.append("Fewer than three photos could be placed. Take photos with 60-80% overlap between neighbours, moving the camera sideways between shots (not just rotating on the spot).")
    unreg = [i for i in report["images"] if not i["registered"]]
    isolated = [i["name"] for i in unreg if i["verified_neighbours"] == 0]
    if isolated:
        out.append(f"{len(isolated)} photo(s) share no verified overlap with any other photo ({', '.join(isolated[:4])}): re-shoot that area from positions that also see already-covered surfaces.")
    comps = [c for c in report["components"] if c["size"] > 1]
    if len(comps) > 1:
        out.append(f"The photos form {len(comps)} separate overlap groups; no photo bridges them. Add 2-3 photos that each see both groups, to join them.")
    sparse = report["sparse"]
    frac = (sparse.get("triangulation_angle_deg") or {}).get("fraction_below_2deg")
    if frac is not None and frac > 0.5:
        out.append(f"{round(frac * 100)}% of the reconstructed points were seen from nearly the same direction (under 2 degrees): depth is poorly constrained. Move the camera further between shots.")
    cov = report["pairs"].get("match_coverage_median")
    if cov is not None and cov < 0.25:
        out.append("Feature matches cover only a small part of each frame: the shared view is dominated by one patch. Include more textured surfaces in the overlap.")
    if reg < n and not out:
        out.append(f"{n - reg} of {n} photos were not placed; see each photo's 'why_unregistered'.")
    return out
