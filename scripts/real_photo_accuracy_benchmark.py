"""Real-photo reconstruction accuracy benchmark (KiloCode RCPAS Workstream 2).

Runs the production vertical_slice pipeline (real COLMAP backend) on the
committed room_capture dataset and measures reconstruction accuracy against
the manifest's ground-truth camera positions and measured baselines.

All metrics are MEASURED (not manufactured):
- Pose error: similarity-aligned (Kabsch + scale) reconstructed camera
  positions vs manifest ground-truth positions, in metres.
- Scale error: metres-per-unit recovered from measured baselines vs the
  manifest baselines (only when scale_state == "metric"; otherwise reported
  as "relative" and NOT reported as a metric accuracy).
- Reconstruction completeness: registered cameras / total images,
  reconstructed 3D points.
- Runtime: wall-clock seconds for the full vertical_slice pipeline.
- Peak memory (VRAM proxy): peak RSS of the process.

Results are written to a machine-readable JSON report and a human-readable
markdown report. Synthetic, real-photo and physically measured results are
kept separate: everything here is REAL-PHOTO (room_capture).

Run: python scripts/real_photo_accuracy_benchmark.py [--images N] [--out report.json]
"""
from __future__ import annotations

import argparse, json, math, sys, tempfile, time, tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice
from evidence.session import EvidenceItem, EvidenceKind
from sdk.reality import validate
from reconstruction.scale import ScaleReference
from worldstore.store import WorldStore


DATASET = Path("datasets/room_capture")
IMAGES = DATASET / "images"
MANIFEST = DATASET / "manifest.json"


def load_manifest():
    return json.loads(MANIFEST.read_text())


def build_evidence(limit=None):
    imgs = sorted(IMAGES.glob("*.jpg"))
    if limit:
        imgs = imgs[:limit]
    return [EvidenceItem(
        id=p.name, kind=EvidenceKind.PHOTO,
        source_uri=p.resolve().as_posix(),
        metadata={"size": [1280, 960]},
    ) for p in imgs]


def similarity_align(est_points, gt_points):
    """Similarity transform (scale s, rotation R, translation t) mapping est -> gt."""
    est = np.array(est_points, dtype=float)
    gt = np.array(gt_points, dtype=float)
    est_c = est.mean(axis=0)
    gt_c = gt.mean(axis=0)
    est_cen = est - est_c
    gt_cen = gt - gt_c
    est_scale = np.linalg.norm(est_cen)
    gt_scale = np.linalg.norm(gt_cen)
    s = gt_scale / est_scale if est_scale > 0 else 1.0
    H = est_cen.T @ gt_cen
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = gt_c - s * (R @ est_c)
    aligned = (s * (R @ est_cen.T)).T + gt_c
    return float(s), R, t, aligned


def run_benchmark(limit=None):
    manifest = load_manifest()
    imgs = sorted(IMAGES.glob("*.jpg"))
    if limit:
        imgs = imgs[:limit]
    items = [EvidenceItem(
        id=p.name, kind=EvidenceKind.PHOTO,
        source_uri=p.resolve().as_posix(),
        metadata={"size": [1280, 960]},
    ) for p in imgs]

    opts = VerticalSliceOptions(
        colmap_binary="colmap",
        reconstruction_backend=None,
        depth_model=None,
        perception_model=None,
        mesh_enabled=False,
        detail_enabled=False,
        dense_mvs_enabled=False,
        image_size=[1280, 960],
        scene_gate=False,
        measured_baselines=[
            ScaleReference(evidence_id_a=b["evidence_id_a"],
                           evidence_id_b=b["evidence_id_b"],
                           distance_m=b["distance_m"],
                           method="manual_measurement")
            for b in manifest["measured_baselines"]
        ],
    )

    tracemalloc.start()
    t0 = time.time()
    try:
        res = vertical_slice(items, opts)
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "images": len(items)}
    elapsed = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Pose error vs ground truth (similarity-aligned)
    gt = {im["file"]: im for im in manifest["images"]}
    cam_poses = {cp[0]: cp for cp in res.camera_poses}
    est_pts, gt_poses, matched = [], [], []
    for ev_id, cp in cam_poses.items():
        if ev_id in gt:
            est_pts.append(list(cp[1]))
            gt_poses.append(list(g["position_m"] for g in [gt[ev_id]])[0])
            matched.append(ev_id)

    pose_result = {"matched_cameras": len(matched), "pose_error_m": None}
    if len(est_pts) >= 3:
        s, R, t, aligned = similarity_align(np.array(est_pts), np.array(gt_poses))
        errs = [float(np.linalg.norm(a - b)) for a, b in zip(aligned, np.array(gt_poses))]
        pose_result = {
            "matched_cameras": len(matched),
            "similarity_scale_s": s,
            "pose_error_mean_m": float(sum(errs) / len(errs)),
            "pose_error_max_m": float(max(errs)),
            "pose_error_min_m": float(min(errs)),
            "pose_errors_m": {eid: float(e) for eid, e in zip(matched, errs)},
        }

    # Scale error: metres-per-unit from measured baselines vs manifest
    scale_result = {
        "scale_state": res.scale_state,
        "meters_per_unit": res.meters_per_unit,
        "metric_scale_available": res.scale_state == "metric" and res.meters_per_unit is not None,
    }
    if res.scale_state == "metric" and res.meters_per_unit:
        # compare recovered scale against manifest baselines
        baseline_errors = []
        for b in manifest["measured_baselines"]:
            a_id, b_id, dist_m = b["evidence_id_a"], b["evidence_id_b"], b["distance_m"]
            if a_id in cam_poses and b_id in cam_poses:
                pa = np.array(cam_poses[a_id][1])
                pb = np.array(cam_poses[b_id][1])
                model_sep = float(np.linalg.norm(pa - pb))
                recovered_m = model_sep * res.meters_per_unit
                baseline_errors.append({
                    "pair": f"{a_id}<->{b_id}",
                    "measured_m": dist_m,
                    "recovered_m": recovered_m,
                    "error_m": abs(recovered_m - dist_m),
                })
        scale_result["baseline_errors_m"] = baseline_errors
        if baseline_errors:
            scale_result["scale_error_mean_m"] = float(sum(b["error_m"] for b in baseline_errors) / len(baseline_errors))
            scale_result["scale_error_max_m"] = float(max(b["error_m"] for b in baseline_errors))

    completeness = {
        "images_total": len(items),
        "cameras_registered": res.cameras_registered,
        "camera_registration_ratio": res.cameras_registered / len(items) if items else 0.0,
        "points_total": res.points_total,
        "registration_status": res.registration_status,
    }

    report = {
        "dataset": manifest["dataset"],
        "dataset_path": str(DATASET),
        "image_size": [1280, 960],
        "intrinsics": manifest["intrinsics_px"],
        "evidence_kind": "real_photograph",
        "benchmark_type": "real_photo_reconstruction",
        "elapsed_seconds": round(elapsed, 3),
        "peak_memory_bytes": peak,
        "pose": pose_result,
        "scale": scale_result,
        "completeness": completeness,
        "world_entities": len(res.world.entities),
        "world_valid": validate(res.world).is_valid(),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description="Real-photo reconstruction accuracy benchmark")
    ap.add_argument("--images", type=int, default=None, help="Number of images to use (default: all)")
    ap.add_argument("--out", type=str, default=None, help="Write JSON report to this file")
    ap.add_argument("--report", type=str, default=None, help="Write human-readable markdown report to this file")
    args = ap.parse_args()

    report = run_benchmark(limit=args.images)

    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON report to {args.out}")
    if args.report:
        lines = [
            "# Real-Photo Reconstruction Accuracy Benchmark (KiloCode RCPAS WS2)",
            "",
            f"- Dataset: {report['dataset']} ({report['evidence_kind']})",
            f"- Images: {report['completeness']['images_total']}",
            f"- Elapsed: {report['elapsed_seconds']} s",
            f"- Peak memory: {report['peak_memory_bytes'] / (1024*1024):.2f} MB",
            "",
            "## Pose error (similarity-aligned vs ground truth)",
            f"- Matched cameras: {report['pose']['matched_cameras']}",
            f"- Mean error: {report['pose']['pose_error_mean_m']:.4f} m" if report['pose']['pose_error_mean_m'] is not None else "- Pose error: n/a",
            f"- Max error: {report['pose']['pose_error_max_m']:.4f} m" if report['pose']['pose_error_max_m'] is not None else "",
            f"- Min error: {report['pose']['pose_error_min_m']:.4f} m" if report['pose']['pose_error_min_m'] is not None else "",
            "",
            "## Scale",
            f"- Scale state: {report['scale']['scale_state']}",
            f"- Meters per unit: {report['scale']['meters_per_unit']}",
            f"- Metric scale available: {report['scale']['metric_scale_available']}",
            "",
            "## Completeness",
            f"- Cameras registered: {report['completeness']['cameras_registered']} / {report['completeness']['images_total']}",
            f"- Registration ratio: {report['completeness']['camera_registration_ratio']:.3f}",
            f"- Points total: {report['completeness']['points_total']}",
            f"- Registration status: {report['completeness']['registration_status']}",
            "",
            "## World validation",
            f"- World entities: {report['world_entities']}",
            f"- World valid: {report['world_valid']}",
        ]
        Path(args.report).write_text("\n".join(lines))
        print(f"Wrote markdown report to {args.report}")
    return 0 if report["pose"]["pose_error_mean_m"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
