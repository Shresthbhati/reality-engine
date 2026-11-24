"""Reproducible camera-registration diagnosis (Sprint 1A, P2).

Runs the INSTALLED COLMAP binary over a named photograph subset and records, per stage: exit status,
wall time, stderr excerpt, and the counts COLMAP itself wrote into its database (keypoints per image,
raw + verified matches, two-view inlier counts) plus the registered cameras and track statistics of
every model the mapper produced. Nothing is inferred: every number is read back from COLMAP's own
database or model files.

    python scripts/sprint1/registration_diagnosis.py --images datasets/south_building/images \
        --out artifacts/sprint1/reg10 --select stride:3 --limit 10

``--select`` is ``stride:N`` (every Nth file in sorted order), ``first:N`` or an explicit
comma-separated list of file names. The report is JSON; it never claims a camera was registered
when the mapper did not register it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _exif(path: Path) -> dict:
    """Camera metadata the file itself reports (absent fields stay null, never defaulted)."""
    out: dict = {"file": path.name, "bytes": path.stat().st_size}
    try:
        from PIL import Image, ExifTags

        with Image.open(path) as im:
            out["width"], out["height"] = im.size
            ex = {ExifTags.TAGS.get(k, str(k)): v for k, v in (im.getexif() or {}).items()}
            for key in ("Make", "Model", "FocalLength", "FocalLengthIn35mmFilm",
                        "PixelXDimension", "PixelYDimension", "Orientation", "DateTimeOriginal"):
                val = ex.get(key)
                out[key] = str(val) if val is not None else None
    except Exception as exc:  # noqa: BLE001 - unreadable EXIF is a finding, not a crash
        out["exif_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _run(step: str, args: list[str], log_dir: Path, timeout: float = 3600.0) -> dict:
    env = dict(**__import__("os").environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(["colmap", step, *args], capture_output=True, text=True,
                              timeout=timeout, shell=False, env=env)
        rec = {"step": step, "argv": ["colmap", step, *args], "exit": proc.returncode,
               "seconds": round(time.perf_counter() - t0, 2),
               "stderr_tail": (proc.stderr or "")[-1200:], "stdout_tail": (proc.stdout or "")[-1200:]}
    except subprocess.TimeoutExpired:
        rec = {"step": step, "argv": ["colmap", step, *args], "exit": "timeout",
               "seconds": round(time.perf_counter() - t0, 2), "stderr_tail": ""}
    (log_dir / f"{step}.log").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def _connected_components(pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Vertex sets of the verified-pair graph. A capture that registers nothing usually has a graph that is one or two
    small components, not a connected set of overlaps -- that is what the mapper's initialiser needs."""
    adj: dict[str, set[str]] = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen: set[str] = set()
    out: list[list[str]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for m in sorted(adj[n]):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        out.append(sorted(comp))
    return sorted(out, key=len, reverse=True)


def _db_facts(db: Path) -> dict:
    """COLMAP's own database counts: per-image keypoints, match totals, verified two-view inliers, per-pair detail and
    the connected components of the verified-pair graph."""
    if not db.is_file():
        return {"status": "database missing"}
    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()
        # COLMAP >= 3.8 stores one blob row per image/pair; the count is the `rows` column, not COUNT(*).
        kp = dict(cur.execute("SELECT image_id, rows FROM keypoints").fetchall())
        names = dict(cur.execute("SELECT image_id, name FROM images").fetchall())
        raw_rows = [r[0] for r in cur.execute("SELECT rows FROM matches").fetchall()]
        n_raw = sum(int(v) for v in raw_rows)
        try:
            n_ver = cur.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0").fetchone()[0]
            inl = [r[0] for r in cur.execute(
                "SELECT rows FROM two_view_geometries WHERE rows > 0").fetchall()]
        except sqlite3.Error as exc:
            n_ver, inl = None, []
            kp["two_view_error"] = str(exc)
        # per-pair detail: which two photographs, how many raw matches, how many survived COLMAP's own two-view verification
        detail: list[dict] = []
        try:
            # COLMAP stores one `pair_id` per pair, encoded as image_id_1 * 2^31-1 + image_id_2 (the smaller id first).
            for pair_id, inl_rows in cur.execute(
                "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"
            ).fetchall():
                a, b = int(pair_id) // 2147483647, int(pair_id) % 2147483647
                detail.append({"a": names.get(a, str(a)), "b": names.get(b, str(b)), "inliers": int(inl_rows)})
        except sqlite3.Error as exc:
            detail = [{"error": str(exc)}]
        detail.sort(key=lambda d: -int(d.get("inliers", 0)))
        return {
            "images": len(names),
            "keypoints_per_image": {names.get(k, str(k)): v for k, v in sorted(kp.items())},
            "raw_match_pairs_with_matches": int(n_ver or 0),
            "raw_matches_total": int(n_raw),
            "verified_pairs": None if n_ver is None else int(n_ver),
            "verified_inliers": sorted(int(v) for v in inl),
            "verified_inliers_median": float(statistics.median(inl)) if inl else None,
            "verified_inliers_max": int(max(inl)) if inl else None,
            "verified_pairs_detail": detail,
            "verified_graph_components": _connected_components([(d["a"], d["b"]) for d in detail if "a" in d]),
            "possible_pairs": len(names) * (len(names) - 1) // 2,
        }
    finally:
        con.close()


def _model_facts(model_dir: Path, image_names: list[str]) -> dict:
    """Registered cameras and track statistics of one COLMAP model, read with pycolmap (text or binary)."""
    try:
        import pycolmap

        rec = pycolmap.Reconstruction(str(model_dir))
    except Exception as exc:  # noqa: BLE001 - an unreadable model is a finding
        return {"status": f"model unreadable: {type(exc).__name__}: {exc}"}
    registered = sorted(Path(im.name).name for im in rec.images.values())
    tracks = [int(p.track.length()) for p in rec.points3D.values()]
    return {
        "registered": len(registered),
        "registered_names": registered,
        "not_registered": sorted(set(image_names) - set(registered)),
        "points": len(tracks),
        "track_median": float(statistics.median(tracks)) if tracks else None,
        "track_mean": round(sum(tracks) / len(tracks), 2) if tracks else None,
        "track_max": max(tracks) if tracks else None,
        "mean_reprojection_error": round(float(rec.compute_mean_reprojection_error()), 3) if tracks else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", default="stride:3")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-image-size", type=int, default=1024)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--robust-sift", action="store_true",
                    help="the production backend's default SIFT config (affine-shape estimation + domain-size pooling, CPU only)")
    args = ap.parse_args(argv)

    src = Path(args.images)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    work = out / "images"
    work.mkdir(parents=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file())
    sel = args.select.strip()
    if sel.startswith("stride:"):
        chosen = files[:: int(sel.split(":", 1)[1])][: args.limit]
    elif sel.startswith("first:"):
        chosen = files[: int(sel.split(":", 1)[1])]
    else:
        wanted = {s.strip() for s in sel.split(",") if s.strip()}
        chosen = [p for p in files if p.name in wanted]
    if len(chosen) < 2:
        print(f"need >= 2 photographs, selected {len(chosen)}")
        return 2
    for p in chosen:
        shutil.copy2(p, work / p.name)

    sift_extra = (["--SiftExtraction.estimate_affine_shape", "1",
                   "--SiftExtraction.domain_size_pooling", "1"] if args.robust_sift else [])
    use_gpu = "0" if args.robust_sift else str(args.gpu)   # affine-shape SIFT is CPU-only in COLMAP
    report: dict = {
        "images_source": str(src), "selected_count": len(chosen), "selection": sel,
        "colmap_version": subprocess.run(["colmap", "--version"], capture_output=True, text=True).stdout.strip(),
        "parameters": {"max_image_size": args.max_image_size, "gpu": bool(args.gpu), "robust_sift": bool(args.robust_sift),
                       "camera_model": "SIMPLE_RADIAL", "single_camera": "0 (one camera per image)"},
        "exif": [_exif(p) for p in chosen],
        "stages": [],
    }

    db = out / "database.db"
    sparse = out / "sparse"
    sparse.mkdir()
    report["stages"].append(_run("feature_extractor", [
        "--database_path", str(db), "--image_path", str(work),
        "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "0",
        "--FeatureExtraction.use_gpu", use_gpu, *sift_extra,
        "--FeatureExtraction.max_image_size", str(args.max_image_size)], out))
    report["stages"].append(_run("exhaustive_matcher", [
        "--database_path", str(db), "--FeatureMatching.use_gpu", use_gpu], out))
    report["stages"].append(_run("mapper", [
        "--database_path", str(db), "--image_path", str(work), "--output_path", str(sparse),
        "--Mapper.multiple_models", "1", "--Mapper.min_model_size", "2",
        "--Mapper.structure_less_registration_fallback", "1",
        "--Mapper.random_seed", "0", "--default_random_seed", "0"], out))
    report["database"] = _db_facts(db)
    models = []
    for d in sorted(p for p in sparse.iterdir() if p.is_dir()):
        models.append({"model": d.name, **_model_facts(d, [p.name for p in chosen])})
    report["models"] = models
    report["any_registered"] = any(m.get("registered", 0) > 0 for m in models)
    # COLMAP's own words for why a model is missing, so a failure reason is quoted, not guessed
    mlog = (report["stages"][-1].get("stderr_tail") or "") + (report["stages"][-1].get("stdout_tail") or "")
    report["mapper_failure_lines"] = [ln.strip() for ln in mlog.splitlines()
                                      if "No good initial image pair" in ln or "bad initial pair" in ln
                                      or "Failed to create any sparse model" in ln]
    (out / "diagnosis.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({
        "selected": len(chosen),
        "models": [(m["model"], m.get("registered"), m.get("points"), m.get("track_median")) for m in models],
        "keypoints_total": sum((report["database"].get("keypoints_per_image") or {}).values()),
        "verified_pairs": report["database"].get("verified_pairs"),
        "verified_inliers_max": report["database"].get("verified_inliers_max"),
        "report": str(out / "diagnosis.json"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
