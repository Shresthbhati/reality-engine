"""Measure how the detail chain's thresholds behave on REAL data (committed South Building sparse model).

Usage:  python scripts/measure_detail_thresholds.py [--out docs/engineering/detail_threshold_measurements.json]

What this can and cannot establish (stated so the numbers are not over-read):
  * The data is a real, committed COLMAP sparse model (32 photographs of a building, ~50k points) in COLMAP's own
    arbitrary scale, so every spatial threshold is expressed as a FRACTION OF THE SCENE EXTENT and converted to model
    units per run. Nothing here is a metric claim.
  * ``--extent robust`` (default) uses perception.detail.voxel.robust_scene_extent -- the diagonal of the 1st-99th
    percentile box, the same definition the product's voxel rule uses. ``--extent bbox`` reproduces the 2026-09-30
    run, whose bounding-box diagonal (67.9) turned out to be ~10x the real footprint (6.3): see
    docs/engineering/DETAIL_CALIBRATION.md.
  * There is no ground truth for "which details exist", so "false positive" and "missed detail" cannot be scored.
    What IS measured: how many cells/ROIs each setting produces, how many refine vs refuse, the measured residuals
    of what refines, runtime, and STABILITY -- the overlap of the ROI cell sets between neighbouring settings. A
    threshold on a plateau (neighbouring values give the same ROIs) is defensible; one on a cliff is not.
Every number printed comes from the run; nothing is tuned to look good.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPARSE = REPO / "datasets" / "south_building" / "sparse"


def load_real_scene():
    from reconstruction.backend.colmap_backend import _parse_images_txt, _parse_points3d_txt
    from reconstruction.backend.interface import ReconstructionResult

    images_txt = (SPARSE / "images.txt").read_text()
    names = {}
    for line in images_txt.splitlines():
        parts = line.split()
        if line.strip() and not line.startswith("#") and len(parts) >= 10:
            names[parts[0]] = parts[9]
    poses = _parse_images_txt(images_txt, {n: n for n in names.values()})
    points = _parse_points3d_txt((SPARSE / "points3D.txt").read_text(), names)
    return ReconstructionResult(points=points, camera_poses=poses, registration_status="success")


def cameras_for(result):
    from reconstruction.calibration.camera import CameraIntrinsics, camera_from_pose

    cam_line = [l for l in (SPARSE / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")][0].split()
    w, h, f, cx, cy = int(cam_line[2]), int(cam_line[3]), float(cam_line[4]), float(cam_line[5]), float(cam_line[6])
    intr = CameraIntrinsics(fx=f, fy=f, cx=cx, cy=cy, width=w, height=h)
    return [camera_from_pose(intr, p) for p in result.camera_poses]


def extent_of(points) -> float:
    lo = [min(p.position[i] for p in points) for i in range(3)]
    hi = [max(p.position[i] for p in points) for i in range(3)]
    return math.dist(lo, hi)


def run_config(result, quality, voxel_frac, curvature, planarity, extent):
    from perception.detail.discovery import discover_detail
    from perception.detail.refinement import refine_rois
    from perception.detail.roi import generate_rois

    voxel = voxel_frac * extent
    t0 = time.perf_counter()
    cands = discover_detail(result, quality, voxel_size=voxel, curvature_threshold=curvature,
                            planarity_threshold=planarity)
    t1 = time.perf_counter()
    rois = generate_rois(cands, voxel_size=voxel, include_structure=True)
    t2 = time.perf_counter()
    registry = {p.track_id: p.position for p in result.points}
    outcomes = refine_rois(rois, point_lookup=lambda pid: registry.get(pid))
    t3 = time.perf_counter()
    refined = [o for o in outcomes if o.status == "refined"]
    rms = [o.refinement.rms_residual_m for o in refined if o.refinement is not None]
    return {
        "voxel_frac_of_extent": voxel_frac, "voxel_model_units": round(voxel, 4), "curvature": curvature,
        "planarity": planarity, "candidates": len(cands), "rois": len(rois), "refined": len(refined),
        "refused": len(outcomes) - len(refined),
        "rms_median_frac_of_extent": round(statistics.median(rms) / extent, 6) if rms else None,
        "roi_size_points_median": int(statistics.median([r.n_points for r in rois])) if rois else 0,
        "seconds": {"discovery": round(t1 - t0, 2), "roi": round(t2 - t1, 2), "refine": round(t3 - t2, 2)},
        "_cells": sorted({c for r in rois for c in r.detail_cells}),
    }


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return round(len(a & b) / len(a | b), 3) if (a | b) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extent", choices=("robust", "bbox"), default="robust")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        name = "detail_threshold_measurements_robust_extent.json" if args.extent == "robust"             else "detail_threshold_measurements.json"
        args.out = str(REPO / "docs" / "engineering" / name)

    from perception.quality.assessment import assess_evidence_quality

    result = load_real_scene()
    cams = cameras_for(result)
    from perception.detail.voxel import robust_scene_extent

    bbox = extent_of(result.points)
    robust = robust_scene_extent(result.points)
    extent = robust if args.extent == "robust" else bbox
    quality = assess_evidence_quality(result, cams)
    print(f"real scene: {len(result.points)} points, {len(result.camera_poses)} cameras, extent[{args.extent}] "
          f"{extent:.3f} model units (bbox diagonal {bbox:.3f}, robust {robust:.3f})")
    print(f"quality: tier={quality.detail_tier} gsd={quality.gsd_mm_per_px} (model-unit scale: not metric)")

    if args.extent == "robust":
        voxel_fracs = [0.05, 0.10, 0.16, 0.25, 0.40]
        base = (0.16, 0.1, 0.95)
    else:
        voxel_fracs = [0.01, 0.02, 0.04, 0.08]
        base = (0.04, 0.1, 0.95)
    curvatures = [0.05, 0.1, 0.2]
    planarities = [0.9, 0.95, 0.98]
    # one-factor-at-a-time around the shipped defaults (a full grid would hide which factor moves the result)
    plan = [base] + [(v, base[1], base[2]) for v in voxel_fracs if v != base[0]] \
        + [(base[0], c, base[2]) for c in curvatures if c != base[1]] \
        + [(base[0], base[1], p) for p in planarities if p != base[2]]
    rows = []
    for v, c, p in plan:
        row = run_config(result, quality, v, c, p, extent)
        rows.append(row)
        print(f"voxel {v:>5} curv {c:>4} plan {p:>4}: cand {row['candidates']:>4} roi {row['rois']:>4} "
              f"refined {row['refined']:>4} refused {row['refused']:>3} rms/ext {row['rms_median_frac_of_extent']} "
              f"{row['seconds']}")
    base_cells = rows[0]["_cells"]
    for row in rows:
        same_grid = row["voxel_frac_of_extent"] == base[0]      # cell ids are only comparable on the same voxel grid
        row["roi_cell_overlap_with_base"] = jaccard(base_cells, row["_cells"]) if same_grid else None
        row.pop("_cells")
    out = {"dataset": "south_building sparse (committed, real)", "points": len(result.points),
           "cameras": len(result.camera_poses), "extent_definition": args.extent,
           "extent_model_units": round(extent, 4), "bbox_diagonal_model_units": round(bbox, 4),
           "robust_extent_model_units": round(robust, 4),
           "scale": "COLMAP arbitrary (not metric)",
           "base": {"voxel_frac_of_extent": base[0], "curvature": base[1], "planarity": base[2]}, "runs": rows,
           "note": "no ground truth: false positives / missed details are not scored; see module docstring"}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
