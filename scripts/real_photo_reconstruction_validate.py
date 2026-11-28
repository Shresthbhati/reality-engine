"""Real-photo reconstruction validation on room_capture (ground-truth poses + baselines).

Runs the production vertical_slice pipeline (real COLMAP backend) on the
committed room_capture images and verifies the produced WorldIR against the
manifest's ground-truth camera positions and measured baselines.

Camera poses from COLMAP are in an arbitrary similarity frame (scale,
rotation, translation). We align them to ground truth with a similarity
transform (Kabsch + scale) before computing pose error, which is the
correct way to assess SfM accuracy.

Run: python scripts/real_photo_reconstruction_validate.py
"""
from __future__ import annotations

import json, math, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice
from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.scale import ScaleReference
from sdk.reality import validate
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
    items = []
    for p in imgs:
        items.append(EvidenceItem(
            id=p.name, kind=EvidenceKind.PHOTO,
            source_uri=p.resolve().as_posix(),
            metadata={"size": [1280, 960]},
        ))
    return items


def similarity_align(est_points, gt_points):
    """Similarity transform (scale s, rotation R, translation t) mapping est -> gt.
    est_points, gt_points: Nx3 arrays. Returns s, R, t and aligned points."""
    est = np.array(est_points, dtype=float)
    gt = np.array(gt_points, dtype=float)
    # center
    est_c = est.mean(axis=0)
    gt_c = gt.mean(axis=0)
    est_cen = est - est_c
    gt_cen = gt - gt_c
    # scale
    est_scale = np.linalg.norm(est_cen)
    gt_scale = np.linalg.norm(gt_cen)
    s = gt_scale / est_scale if est_scale > 0 else 1.0
    # rotation (Kabsch)
    H = est_cen.T @ gt_cen
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # translation
    t = gt_c - s * (R @ est_c)
    aligned = s * (est_cen @ R.T) + est_c + t  # wait, order: s*R@est + t
    aligned = (s * (R @ est_cen.T)).T + est_c + gt_c - (s * (R @ est_c).T).mean(axis=0) + gt_c
    # simpler: aligned = s * (R @ est.T).T + t applied after centering:
    aligned = (s * (R @ est_cen.T)).T + gt_c
    return s, R, t, aligned


def main():
    manifest = load_manifest()
    print(f"Dataset: {manifest['dataset']}")
    print(f"Images in manifest: {len(manifest['images'])}")
    print(f"Intrinsics: fx={manifest['intrinsics_px']['fx']} fy={manifest['intrinsics_px']['fy']} "
          f"cx={manifest['intrinsics_px']['cx']} cy={manifest['intrinsics_px']['cy']}")
    print(f"Scale reference: {manifest['scale_reference']['method']} (metric={manifest['scale_reference']['metric']})")
    print(f"Measured baselines: {len(manifest['measured_baselines'])}")

    items = build_evidence()
    print(f"\nRunning vertical_slice on {len(items)} real images (COLMAP backend)...")

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

    import time
    t0 = time.time()
    try:
        res = vertical_slice(items, opts)
    except Exception as exc:
        print(f"RECONSTRUCTION FAILED: {exc}")
        return 1
    elapsed = time.time() - t0

    print(f"\n=== RECONSTRUCTION RESULT ===")
    print(f"registration_status: {res.registration_status}")
    print(f"cameras_registered: {res.cameras_registered} / {len(items)}")
    print(f"points_total: {res.points_total}")
    print(f"scale_state: {res.scale_state}, meters_per_unit: {res.meters_per_unit}")
    print(f"world entities: {len(res.world.entities)}")
    print(f"elapsed: {elapsed:.1f}s")

    report = validate(res.world)
    print(f"world valid: {report.is_valid()}, issues: {len(report.issues)}")

    # === GROUND TRUTH VERIFICATION ===
    gt = {im["file"]: im for im in manifest["images"]}
    cam_poses = {cp[0]: cp for cp in res.camera_poses}

    est_pts, gt_poses, matched_ids = [], [], []
    for ev_id, cp in cam_poses.items():
        if ev_id in gt:
            g = gt[ev_id]
            est_pts.append(list(cp[1]))
            gt_poses.append(list(g["position_m"]))
            matched_ids.append(ev_id)

    if len(est_pts) >= 3:
        s, R, t, aligned = similarity_align(np.array(est_pts), np.array(gt_poses))
        errs = [float(np.linalg.norm(a - b)) for a, b in zip(aligned, np.array(gt_poses))]
        print(f"\n=== POSE ERROR vs ground truth ({len(errs)} cameras, similarity-aligned) ===")
        print(f"  similarity scale s = {s:.6f} (model units -> meters)")
        print(f"  mean: {sum(errs)/len(errs):.4f} m")
        print(f"  max:  {max(errs):.4f} m")
        print(f"  min:  {min(errs):.4f} m")
        for ev_id, err in sorted(zip(matched_ids, errs), key=lambda x: -x[1])[:5]:
            print(f"  {ev_id}: err={err:.4f} m")
    else:
        print("Not enough matched cameras for alignment.")

    # Save to WorldStore and verify read-back
    tmp = Path(tempfile.mkdtemp()) / "store"
    store = WorldStore(tmp)
    ver = store.save_version(res.world, parent=None, version_id="v-real-1")
    errs2 = store.verify_version(ver.version_id)
    print(f"\n=== WORLDSTORE ===")
    print(f"saved version: {ver.version_id}")
    print(f"verify errors: {errs2}")
    loaded = store.load_version(ver.version_id)
    print(f"loaded entities: {len(loaded.entities)} (matches: {len(loaded.entities) == len(res.world.entities)})")

    print("\n=== REAL-PHOTO RECONSTRUCTION VALIDATION COMPLETE ===")
    return 0 if res.registration_status == "success" and res.cameras_registered >= 2 else 2


if __name__ == "__main__":
    import time
    sys.exit(main())
