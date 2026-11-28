#!/usr/bin/env python3
"""Real-photograph accuracy benchmark through the PRODUCTION reconstruction path, scored with reconstruction.quality_metrics.

Unlike earlier benchmark scripts this one does NOT disable depth / perception / mesh / dense: it calls
``scripts/stage_report.run_pipeline`` (the same functions ``apps/api/jobs.py`` runs: scene gate -> intrinsics ->
progressive ladder -> pose, scale, depth, layout, perception, mesh, dense, detail, finalize) and then scores the
returned ProgressiveResult against the committed COLMAP reference model of the same photographs.

HONEST LIMITS (recorded in the output, not hidden):
  * the reference is a COLMAP SfM model, itself reconstructed from the same photos. It is NOT independently measured
    ground truth (no tape / laser measurements for these photos), so it can validate registration consistency and
    shape, never metric scale. ``independent_ground_truth`` is therefore always False here.
  * pose/point errors are measured AFTER a Sim(3) gauge fit to the reference (``residual_is_independent_metric_accuracy``
    is False); a metric figure is only reported if the pipeline itself declared a metric scale.

Usage (one dataset per process; a 6 GB GPU cannot take two heavy jobs):
    python scripts/real_photo_accuracy_benchmark.py --name south_building_12 --subset 12
Output: docs/benchmarks/real_photo_accuracy_<name>.json
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATASET = ROOT / "datasets" / "south_building"
OUT_DIR = ROOT / "docs" / "benchmarks"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_status(rep: dict, name: str) -> dict:
    for s in rep.get("stages", []):
        if s.get("stage") == name:
            return {"status": s.get("status"), "wall_s": s.get("wall_s"), "metrics": s.get("metrics"),
                    "warnings": s.get("warnings"), "outputs": s.get("outputs")}
    return {"status": "not_recorded"}


def score(prog, rep: dict, gt_images: dict, gt_points: dict) -> dict:
    import numpy as np

    from reconstruction import quality_metrics as qm
    from reconstruction.evaluation import evaluate_absolute_poses, evaluate_point_cloud

    out: dict = {}
    out["registration_rate"] = qm.registration_rate(len(prog.registered_ids), len(prog.input_ids))
    out["scale_validity"] = qm.scale_validity(prog.scale_state, None)

    id_to_name = rep.get("id_to_name", {})
    from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint

    # ProgressiveResult carries poses as (evidence_id, position, quaternion) tuples and points as positions.
    renamed = [ReconstructedCameraPose(evidence_id=id_to_name[eid], position=tuple(pos), rotation=tuple(rot))
               for eid, pos, rot in prog.camera_poses if eid in id_to_name]
    points = [ReconstructedPoint(position=tuple(q[:3]) if not hasattr(q, "position") else tuple(q.position),
                                 track_id=str(i), source_evidence_ids=[]) for i, q in enumerate(prog.points)]
    pose = evaluate_absolute_poses(renamed, gt_images, estimated_scale_state=prog.scale_state)
    transform = pose.pop("gauge_transform", None)
    out["pose_vs_reference"] = pose

    if transform is not None and points:
        out["point_cloud_vs_reference"] = evaluate_point_cloud(points, gt_points, transform)
        R, t, s = transform
        est = np.asarray([p.position for p in points], dtype=np.float64)
        aligned = s * (est @ np.asarray(R, dtype=np.float64).T) + np.asarray(t, dtype=np.float64)
        ref = np.asarray([gt_points[k] for k in sorted(gt_points)], dtype=np.float64)
        spread = float(np.linalg.norm(ref.max(axis=0) - ref.min(axis=0)))
        tau = 0.02 * spread       # 2% of the reference extent: reference units (COLMAP gauge), not metres
        comp = qm.point_cloud_completeness(aligned, ref, tau, same_frame=True)
        comp["tau_definition"] = "2% of reference bounding-box diagonal, in reference (gauge) units"
        comp["frame_note"] = "estimate aligned to the reference by a Sim(3) fit (scale fitted): shape agreement only"
        out["point_cloud_completeness"] = comp
    else:
        out["point_cloud_vs_reference"] = {"refused": True, "reason": "no gauge transform or no points"}

    dense = _stage_status(rep, "dense")
    out["dense"] = dense
    out["dense_invoked"] = dense.get("status") not in (None, "not_recorded", "skipped")
    out["dense_points_added"] = (dense.get("metrics") or {}).get("dense_points_added")
    out["pose_graph_cycle_error"] = {
        "value": None,
        "refused": "absolute poses are cycle-consistent by construction; pairwise-edge cycle residuals are produced by "
                   "scripts/run_multi_engine_real_capture.py (perception.camera_pose_graph)"}
    out["stage_status"] = {n: _stage_status(rep, n).get("status")
                           for n in ("pose", "scale", "depth", "layout", "perception", "mesh", "dense", "detail", "finalize")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", default=str(DATASET / "images"))
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    import stage_report                                  # noqa: E402  (scripts/ on sys.path)
    from run_south_building_e2e import _load_gt_images, _load_gt_points  # noqa: E402

    imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if a.subset:
        imgs = imgs[:a.subset]
    work = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix=f"real_photo_bench_{a.name}_"))
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("COLMAP", os.environ.get("COLMAP_BINARY", "colmap"))

    t0 = time.time()
    rep = stage_report.run_pipeline(imgs, work)
    prog = stage_report.LAST_PROG
    result: dict = {
        "name": a.name, "git_sha": _git_sha(), "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_images": len(imgs), "image_sha256": {p.name: _sha(p) for p in imgs},
        "production_path": "scripts/stage_report.run_pipeline -> progressive.run_progressive -> vertical_slice (all stages enabled)",
        "independent_ground_truth": False,
        "reference": "datasets/south_building/sparse (COLMAP SfM model of the same photos; NOT independently measured)",
        "wall_s": round(time.time() - t0, 1), "ladder": rep.get("ladder"),
    }
    if prog is None:
        result["error"] = "ladder raised: no ProgressiveResult to score"
    else:
        gt_images = _load_gt_images(DATASET / "sparse" / "images.txt")
        gt_points = _load_gt_points(DATASET / "sparse" / "points3D.txt")
        result["scores"] = score(prog, rep, gt_images, gt_points)
        result["level_name"], result["scale_state"] = prog.level_name, prog.scale_state
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"real_photo_accuracy_{a.name}.json"
    out.write_text(json.dumps(stage_report._clean(result), indent=1, default=str), encoding="utf-8")
    print(f"wrote {out} wall={result['wall_s']}s")
    return 0 if prog is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
