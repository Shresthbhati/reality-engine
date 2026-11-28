"""Controlled registration experiments (Sprint 9 P1). CPU-only COLMAP, production flags.

    python scripts/registration_experiments.py --out <dir> --colmap <bin> \
        --set room_capture=<dir> --set south_building=<dir> ...

For each ``--set`` it runs: the full set; subsets of 11 and 6 photos chosen from that set's own overlap graph
(``dense_overlap``, ``sparse_chain``, ``stride``); and, for a set called ``south_building``, consecutive and
wide-baseline windows. ``--intrinsics-ab name=f,cx,cy,k`` adds the pinned-intrinsics diagnostic A/B (reported
separately, never used as a default). Results land in ``<out>/results.json`` plus one folder (diagnosis.json,
summary.txt, COLMAP logs) per run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.registration_diagnosis import (  # noqa: E402
    RunConfig, diagnose_image_set, diagnose_workdir, read_database, select_subset, summarize)


def _compact(rep: dict) -> dict:
    g, q = rep["graph"], rep["image_summary"]
    return {"registered": rep["registered"]["count"], "of": rep["registered"]["of"], "sub_models": rep["registered"]["sub_models"],
            "attempts": [(a["attempt"], a["registered"]) for a in rep["mapper"]["attempts"]],
            "median_keypoints": q["median_keypoints"], "median_sharpness": q["median_sharpness"],
            "edges_ge_15": g["edges_ge_15"], "edge_density": g["edge_density"], "components": [len(c) for c in g["components"]],
            "isolated": len(g["isolated_images"]), "median_inliers_edge": g["median_inliers_of_edges"],
            "median_inlier_ratio": g["median_inlier_ratio_of_edges"], "rotation_only_share": g["rotation_only_share"],
            "tri_angle_deg": rep["coverage"].get("median_tri_angle_deg"),
            "max_view_sep_deg": rep["coverage"].get("max_viewing_separation_deg"),
            "unregistered": {k: v["reason"] for k, v in rep["unregistered"].items()},
            "causes": [(c["cause"], c["confidence"]) for c in rep["root_causes"]],
            "reproj_rms_median": sorted(r["reproj_rms_px"] for r in rep["images"].values() if r["reproj_rms_px"] is not None)[
                len([1 for r in rep["images"].values() if r["reproj_rms_px"] is not None]) // 2] if any(
                r["reproj_rms_px"] is not None for r in rep["images"].values()) else None,
            "run": rep["run"]["label"], "images": rep["run"]["images"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--set", action="append", default=[], help="name=dir")
    ap.add_argument("--sizes", default="11,6")
    ap.add_argument("--intrinsics-ab", action="append", default=[], help="name=f,cx,cy,k (pinned, diagnostic only)")
    ap.add_argument("--only", help="comma-separated substrings; run only matching labels")
    ns = ap.parse_args()
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    only = ns.only.split(",") if ns.only else None

    def go(label, files, cfg=None):
        if only and not any(o in label for o in only):
            return None
        print(f"== {label} ({len(files)} photos)", flush=True)
        if (out / label / "diagnosis.json").exists():     # already run: re-derive the report, do not repeat COLMAP
            rep = diagnose_workdir(out / label, ns.colmap, {"label": label})
        else:
            try:
                rep = diagnose_image_set(files, out / label, ns.colmap, cfg, label)
            except Exception as exc:  # noqa: BLE001 -- one failed run must not stop the matrix; record it
                print(f"!! {label} failed: {exc}", flush=True)
                results[label] = {"error": str(exc)[:300]}
                res_path.write_text(json.dumps(results, indent=1, default=str))
                return None
        print(summarize(rep), flush=True)
        results[label] = _compact(rep)
        res_path.write_text(json.dumps(results, indent=1, default=str))
        return rep

    ab = dict(a.split("=", 1) for a in ns.intrinsics_ab)
    for spec in ns.set:
        name, d = spec.split("=", 1)
        files = {p.name: p for p in sorted(Path(d).iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        go(f"{name}__full", list(files.values()))
        full_db = out / f"{name}__full" / "database.db"
        if not full_db.exists():
            continue
        db = read_database(full_db)
        for k in (int(s) for s in ns.sizes.split(",")):
            if k >= len(files):
                continue
            for strat in ("dense_overlap", "sparse_chain", "stride"):
                pick = select_subset(db, k, strat)
                go(f"{name}__{k}_{strat}", [files[n] for n in pick])
        if name.startswith("south_building"):
            names = sorted(files)
            go(f"{name}__window_consecutive12", [files[n] for n in names[:12]])
            go(f"{name}__window_stride3_11", [files[n] for n in names[::3]])
            go(f"{name}__window_stride5_7", [files[n] for n in names[::5]])
        if name in ab:
            f, cx, cy, k = ab[name].split(",")
            go(f"{name}__AB_pinned_pinhole_truth", list(files.values()),
               RunConfig(camera_model="SIMPLE_RADIAL", camera_params=f"{f},{cx},{cy},{k}", pin_intrinsics=True))
            go(f"{name}__AB_per_image_cameras", list(files.values()), RunConfig(single_camera=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
