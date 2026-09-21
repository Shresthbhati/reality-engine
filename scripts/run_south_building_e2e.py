#!/usr/bin/env python3
"""Real end-to-end run over the committed south-building dataset
(RECONSTRUCTION_REAL_WORLD_ROBUSTNESS P0: a reproducible REAL execution
path another developer can run from repository data alone).

Chain exercised (real, no fake backend anywhere):

  datasets/south_building/images (32 real photographs)
    -> evidence.importers.import_folder    (real decode, EXIF, quality)
    -> EvidencePackage.to_evidence_items   (quality in metadata["quality"])
    -> reconstruction.robustness_admission (quality-aware admission gate)
    -> ColmapReconstructionBackend         (REAL COLMAP feature_extractor
                                            -> exhaustive_matcher -> mapper
                                            -> model_converter)
    -> robustness.run-level classification  (ACCEPTED/DEGRADED/...)
    -> GT comparison against datasets/south_building/sparse
       (cameras.txt/images.txt/points3D.txt, provenance ids preserved)
    -> fusion into canonical FusableObservations + fusion diagnostics
    -> recorded run record JSON (dataset identity, versions, runtime,
       output statistics, failures) under datasets/south_building/runs/

Usage:
  python scripts/run_south_building_e2e.py [--gpu]

`--gpu` passes COLMAP's use_gpu=1 (this machine's COLMAP 4.2.0 is a
CUDA build). Default is CPU for cross-machine reproducibility.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATASET = REPO / "datasets" / "south_building"


def _colmap_version(binary: str) -> str:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
        return out.stdout.splitlines()[0].strip() if out.stdout else "unknown"
    except Exception as exc:  # probe only; the backend reports the real failure
        return f"unavailable ({type(exc).__name__})"


def _load_gt_images(path: Path):
    """Parse images.txt -> {name: (qw,qx,qy,qz,tx,ty,tz)}."""
    gt = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for i in range(0, len(lines) - 1, 2):
        parts = lines[i].split()
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        qw, qx, qy, qz, tx, ty, tz = (float(x) for x in parts[1:8])
        camera_idx = int(parts[8])
        name = parts[9]
        gt[name] = (qw, qx, qy, qz, tx, ty, tz)
    return gt


def _load_gt_points(path: Path):
    pts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 8 or not p[0].isdigit():
            continue
        pts[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    return pts


def _quat_angle_deg(a, b) -> float:
    """Geodesic angle between two (w,x,y,z) quaternions, degrees."""
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = min(1.0, dot)
    import math
    return math.degrees(2.0 * math.acos(dot))


def _evidence_id_to_name(items) -> dict:
    """Map evidence_id -> original image filename.

    ColmapReconstructionBackend copies each image into its workspace as
    ``{item.id}{suffix}``, so reconstructed poses are keyed by evidence
    id -- NOT by the original filename the GT model uses. The mapping
    goes through source_uri (the only place the original name survives
    ingestion). Reuses the backend's own _uri_to_path so the two sides
    can never disagree about URI parsing."""
    from reconstruction.backend.colmap_backend import _uri_to_path

    mapping = {}
    for item in items:
        src = _uri_to_path(item.source_uri)
        mapping[item.id] = src.name
    return mapping


def _compare_to_gt(poses, gt_images, id_to_name=None):
    """Absolute-pose fidelity vs the reference model, measured through
    reconstruction.evaluation (the canonical evaluator). Returns
    (report, gauge_transform) -- the transform is reused for the
    point-cloud fidelity measurement (ONE gauge explains orientations,
    centers and points).

    Honesty correction (P1 fidelity mission): an incremental SfM run is
    recovered in an ARBITRARY gauge -- comparing its quaternions
    directly against the reference model (the previous implementation)
    measured mostly the gauge difference, not fidelity (the recorded
    8.94 deg "median disagreement" was dominated by gauge). The
    canonical evaluator least-squares-aligns the estimate to the
    reference (Horn rotation on orientation frames + scale/translation
    from camera-center geometry) and only then measures per-camera
    disagreement. Refuses rather than guesses when too few common
    cameras or degenerate geometry cannot define the alignment."""
    from reconstruction.evaluation import evaluate_absolute_poses

    by_name = {p.evidence_id: p for p in poses}
    if id_to_name:
        # The evaluator matches poses to the reference by the pose's
        # evidence_id; rebuild each frozen pose under its ORIGINAL
        # filename (the only name the reference model knows).
        import dataclasses

        renamed = []
        for eid, pose in by_name.items():
            name = id_to_name.get(eid)
            if name is not None:
                renamed.append(dataclasses.replace(pose, evidence_id=name))
        by_name = {p.evidence_id: p for p in renamed}
    report = evaluate_absolute_poses(list(by_name.values()), gt_images)
    if report.get("refused"):
        return {"refused": True, "reason": report.get("reason", "unspecified")}, None
    out = dict(report)
    transform = out.pop("gauge_transform")
    common = len(set(by_name) & set(gt_images))
    out["registered_of_gt"] = f"{common}/{len(gt_images)}"
    out["note"] = (
        "measured after least-squares gauge alignment (Horn rotation on "
        "orientation frames; scale/translation from camera-center "
        "geometry) -- raw pre-alignment disagreement is gauge, not error"
    )
    return out, transform


def _load_run_output_model(workdir: Path):
    """Parse the backend's converted text model output (0/ subdirectory
    written by ColmapReconstructionBackend)."""
    model_dirs = sorted(d for d in workdir.rglob("0") if d.is_dir())
    if not model_dirs:
        return None, None
    model = model_dirs[0]
    images_txt = model / "images.txt"
    points_txt = model / "points3D.txt"
    poses = None
    points = None
    if images_txt.is_file():
        from reconstruction.backend.colmap_backend import _parse_images_txt
        poses = _parse_images_txt(images_txt.read_text(encoding="utf-8"), {})
    if points_txt.is_file():
        n = sum(1 for l in points_txt.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#"))
        points = n
    return poses, points


def _verify_artifact(store, result) -> dict:
    """Round-trip proof that the dense cloud is a persisted, traceable
    canonical artifact -- not an orphaned file:

    1. store.get(result.dense_report['artifact_uri']) RESOLVES (the
       artifact outlived the deleted workspace);
    2. its bytes decode to the same vertex count the report recorded;
    3. the canonical float64 encoding round-trips byte-identically.
    """
    from world_ir.geometry_data import PointCloudData

    report = result.dense_report
    uri = report.get("artifact_uri")
    if not uri:
        return {"resolved": False, "reason": report.get("artifact", "")}
    blob = store.get(uri)
    cloud = PointCloudData.from_bytes(blob)
    return {
        "resolved": True,
        "decodes": len(cloud.points) == report["artifact_vertex_count"],
        "n_points_decoded": len(cloud.points),
        "round_trips": blob == PointCloudData(points=cloud.points).to_bytes(),
        "sha256_recorded": report["artifact_sha256"],
    }


def _dense_vs_sparse_fidelity(sparse_points, dense_points):
    """Measured agreement between the two independent point clouds the
    SAME run produced (sparse SfM tracks vs dense MVS fusion):
    for each sparse point, the distance to its nearest dense neighbor
    (KD-tree, no downsampling). This is a consistency check between two
    real outputs, not a fabricated score."""
    if not sparse_points or not dense_points:
        return None
    from scipy.spatial import cKDTree

    import numpy as np

    d = np.asarray([p.position for p in dense_points], dtype=np.float64)
    s = np.asarray([p.position for p in sparse_points], dtype=np.float64)
    tree = cKDTree(d)
    dist, _ = tree.query(s, k=1)
    return {
        "n_sparse": int(len(s)),
        "n_dense": int(len(d)),
        "sparse_to_dense_median_m": round(float(np.median(dist)), 5),
        "sparse_to_dense_p95_m": round(float(np.percentile(dist, 95)), 5),
        "note": "distances in COLMAP's own SfM scale (no metric anchor; "
                "recorded honestly as model units)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true",
                        help="enable COLMAP GPU (CUDA build required)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="limit images (subset runs); default all 32")
    parser.add_argument("--dense", action="store_true",
                        help="continue the sparse model through COLMAP's "
                             "dense MVS chain (image_undistorter -> "
                             "patch_match_stereo -> stereo_fusion); the "
                             "fused cloud is ingested into the canonical "
                             "artifact store and carried through fidelity "
                             "evaluation")
    args = parser.parse_args()

    if args.dense:
        from reconstruction.dense_pipeline import dense_mvs_available
        if not dense_mvs_available("colmap"):
            print("DENSE UNAVAILABLE: the 'colmap' binary on PATH does not "
                  "list patch_match_stereo -- this build cannot run dense "
                  "MVS. Nothing was faked; re-run without --dense.",
                  file=sys.stderr)
            return 2

    from evidence.importers import import_folder
    from evidence.packages import DeterministicPackageBuilder
    from provenance import Provenance
    from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
    from reconstruction.fusion.fusion import FusableObservation, fuse_quantity
    from reconstruction.orchestrator import ReconstructionOrchestrator
    from reconstruction.robustness import classify_reconstruction_run
    from reconstruction.robustness_admission import admit_for_reconstruction
    from world_ir.artifact_store import FileArtifactStore

    started = time.perf_counter()
    record = {
        "dataset": "south-building (committed 32-image subset)",
        "dataset_identity": json.loads((DATASET / "MANIFEST.json").read_text(encoding="utf-8"))["identity"],
        "python": platform.python_version(),
        "colmap_version": _colmap_version("colmap"),
        "gpu": bool(args.gpu),
    }

    # 1. Real ingestion (decode + EXIF + measured quality metrics).
    builder = DeterministicPackageBuilder(seed="south-building-e2e")
    report = import_folder(builder, str(DATASET / "images"))
    items = builder.build().to_evidence_items()
    if args.max_images:
        items = items[: args.max_images]
    record["ingested"] = {
        "imported": len(report.imported),
        "duplicates_skipped": len(report.duplicates_skipped),
        "unhandled": len(report.unhandled_paths),
        "items": len(items),
    }

    # 2. Admission gate (quality-aware compute).
    admitted, decision = admit_for_reconstruction(items)
    record["admission"] = decision.to_dict()["classification"]["counts"]

    # 3. REAL COLMAP reconstruction through the orchestrator.
    # Dense runs get a PERSISTENT artifact store under the dataset's
    # runs dir: the backend must ingest the fused cloud inside its
    # workspace lifetime or the artifact dies with the temp dir.
    store = FileArtifactStore(root=DATASET / "runs" / "artifacts") if args.dense else None
    backend = ColmapReconstructionBackend(colmap_binary="colmap",
                                          use_gpu=args.gpu, dense_mvs=args.dense,
                                          artifact_store=store)
    orchestrator = ReconstructionOrchestrator([backend])
    t0 = time.perf_counter()
    run = orchestrator.run(admitted)
    record["reconstruction"] = {
        # Diagnostics MUST survive: a false sub-model merge with a
        # discarded report left no trace in earlier runs (measured
        # 2026-09-20 -- run_20260920T200832 was poisoned by exactly
        # this). When present, the serialized CrossSessionReport lists
        # every sub-model's status, transform, and refusal reason.
        "merge_report": getattr(run.result, "merge_report", None),
        "backend": run.diagnostics.backend_name,
        "status": run.diagnostics.final_status,
        "duration_s": round(time.perf_counter() - t0, 1),
        "poses": len(run.result.camera_poses),
        "points": len(run.result.points),
    }

    # 3b. Dense continuation: the backend ingested the fused cloud
    # through the canonical chain INSIDE its workspace lifetime (a
    # fused.ply left in the temp workspace would be an orphaned
    # artifact by construction -- the workspace is deleted on return;
    # measured 2026-09-21). Here the record only VERIFIES the
    # persisted artifact: it resolves, decodes, and matches the cloud
    # that traveled on the result.
    if args.dense:
        dense_report = run.result.dense_report
        assert dense_report is not None  # dense_mvs=True guarantees it
        record["dense"] = {
            **dense_report,
            "artifact_verification": _verify_artifact(
                backend.artifact_store, run.result),
            "fidelity_vs_sparse": _dense_vs_sparse_fidelity(
                run.result.points, run.result.dense_points),
        }

    outcome = classify_reconstruction_run(run.result)
    record["outcome"] = outcome.to_dict()

    # 4. GT comparison. The backend stores camera-to-world quaternions
    # (COLMAP's q is world->camera; the backend conjugates) and camera
    # centers. GT is world->camera, so conjugate before comparing.
    # Pose keys are evidence ids (the backend renames images into its
    # workspace), so translate to the GT's original filenames first.
    gt_images = _load_gt_images(DATASET / "sparse" / "images.txt")
    record["gt_comparison"], gauge_transform = _compare_to_gt(
        run.result.camera_poses, gt_images,
        _evidence_id_to_name(admitted),
    )
    # Geometric fidelity: how far are reconstructed points from the
    # reference surface, after the SAME gauge transform? This is the
    # mission's "are the surfaces actually correct" metric.
    if gauge_transform is not None:
        from reconstruction.evaluation import evaluate_point_cloud

        gt_points = _load_gt_points(DATASET / "sparse" / "points3D.txt")
        record["point_fidelity"] = evaluate_point_cloud(
            run.result.points, gt_points, gauge_transform
        )
    # Keep the COLMAP workspace artifacts on disk for inspection; also
    # record where they are so the run is reproducible/auditable.
    record["colmap_note"] = (
        "backend runs COLMAP in a temporary workspace (database.db, "
        "feature/match/mapper outputs); it is removed after the run -- "
        "only the parsed result and this record persist"
    )

    # 5. Fusion sanity: fuse camera-height observations (a quantity with
    # real units) through the canonical fusion engine -- it must run on
    # the real SfM output and preserve provenance.
    from engine.core.units import Unit
    obs = [
        FusableObservation(
            value=float(pose.position[2]),
            unit=Unit.METER,
            source=f"colmap-pose:{pose.evidence_id}",
            provenance=Provenance.ESTIMATED,
            confidence=float(pose.uncertainty.confidence) or 0.5,
            precision=0.05,
            evidence_ids=(pose.evidence_id,),
        )
        for pose in run.result.camera_poses
    ]
    fused = fuse_quantity(obs, "camera_height_model_units")
    record["fusion"] = {
        "observations": len(obs),
        "fused_value": round(fused.value, 4),
        "fused_precision": round(fused.precision, 5),
        "fused_provenance": fused.provenance.value,
        "conflicts_preserved": len(getattr(fused, "conflicts", ()) or ()),
    }

    # 6. Persist the run record.
    runs_dir = DATASET / "runs"
    runs_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out = runs_dir / f"run_{stamp}{'_gpu' if args.gpu else ''}.json"
    record["total_runtime_s"] = round(time.perf_counter() - started, 1)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print("run record:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
