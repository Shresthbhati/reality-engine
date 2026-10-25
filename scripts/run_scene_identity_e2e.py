#!/usr/bin/env python3
"""End-to-end scene-identity acceptance run with REAL images and REAL COLMAP.

Dataset (synthetic, deterministic, genuinely shared 3D scene with KNOWN camera poses -- see
scripts/render_room_dataset.py): scene A = the textured 2.5x2x2.25 m room rendered from its capture trajectory;
scene B = the SAME room layout and trajectory with different surface texture (a "similar but different place").

    python scripts/run_scene_identity_e2e.py --work <dir> --report <report.json> [--n 24] [--b-count 8]

Steps (each recorded in the report; nothing is asserted that was not measured):
  1. pair statistics: verified inliers for A-A, B-B and A-B pairs (threshold calibration data)
  2. grouping of the mixed A+B upload -> expect two groups, never one
  3. COLMAP reconstruction of the primary group through the bounded correction loop -> diagnostics + outcome
  4. pose accuracy vs the known ground-truth poses (Sim3-aligned)
  5. registry commit + reload (identity / persistence)
  6. insufficient-overlap subset -> honest outcome
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import render_room_dataset as R  # noqa: E402

from evidence.session import EvidenceItem, EvidenceKind  # noqa: E402
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend  # noqa: E402
from reconstruction.evaluation import _matrix_to_quat, evaluate_absolute_poses  # noqa: E402
from reconstruction.scene_identity import (  # noqa: E402
    SceneRegistry, SiftPairVerifier, group_images, reconstruct_group,
)


def render_scene(out: Path, prefix: str, n: int, seed_shift: int, count=None):
    out.mkdir(parents=True, exist_ok=True)
    R.NOISE_OCTAVES = tuple((f, s + seed_shift) for f, s in ((2.0, 11), (8.0, 23), (32.0, 47)))
    cams = R.camera_trajectory(n)
    gt, paths = {}, {}
    for i, cam in enumerate(cams[: count or len(cams)]):
        Rm = R.look_at_matrix(cam["position"], cam["look_at"])
        t = -Rm @ cam["position"]
        eid = f"{prefix}_{i:04d}"
        p = out / f"{eid}.jpg"
        if not p.exists():
            Image.fromarray(R.render_image(Rm, t)).save(p, quality=95)
        w, x, y, z = _matrix_to_quat(Rm.tolist())
        gt[eid] = (w, x, y, z, *[float(v) for v in t])
        paths[eid] = p
    return paths, gt


def evidence(paths):
    return [EvidenceItem(id=k, kind=EvidenceKind.PHOTO, source_uri=p.resolve().as_uri()) for k, p in paths.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--b-count", type=int, default=8)
    a = ap.parse_args()
    work = Path(a.work)
    report: dict = {"commands": " ".join(sys.argv)}
    t0 = time.time()
    pa, gt_a = render_scene(work / "A", "A", a.n, 0)
    pb, _ = render_scene(work / "B", "B", a.n, 1000, count=a.b_count)
    report["dataset"] = {"A_images": len(pa), "B_images": len(pb), "render_s": round(time.time() - t0, 1),
                         "kind": "synthetic shared 3D room, known poses; B = same layout, different texture"}
    paths = {**pa, **pb}
    verifier = SiftPairVerifier(paths)

    # 1. pair statistics
    ids = sorted(paths)
    stats = {"AA": [], "BB": [], "AB": []}
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            ev = verifier(x, y)
            stats[x[0] + y[0] if x[0] == y[0] else "AB"].append((ev.inliers, round(ev.inlier_ratio, 3)))
    report["pair_stats"] = {k: {"n": len(v), "inliers_min": min(i for i, _ in v), "inliers_max": max(i for i, _ in v),
                                "inliers_p10": int(np.percentile([i for i, _ in v], 10)),
                                "inliers_p90": int(np.percentile([i for i, _ in v], 90)),
                                "ratio_max": max(r for _, r in v)} for k, v in stats.items() if v}
    print(json.dumps(report["pair_stats"], indent=1))

    # 2. grouping of the mixed upload
    grouping = group_images(ids, verifier)
    report["grouping_mixed"] = grouping.to_dict() | {"group_sizes": [len(g.members) for g in grouping.groups]}
    print("groups:", [(g.group_id, len(g.members)) for g in grouping.groups], "unlinked:", grouping.unlinked)
    primary = grouping.primary
    report["group_purity"] = {g.group_id: sorted({m[0] for m in g.members}) for g in grouping.groups}

    # 3. COLMAP on the primary group, through the correction loop
    reg = SceneRegistry(work / "registry")
    reg.record_grouping("world-e2e", grouping)
    members = {m: paths[m] for m in primary.members}
    trace = reconstruct_group(
        evidence(members),
        lambda cfg: ColmapReconstructionBackend(robust_sift=cfg["robust_sift"], guided_matching=cfg["guided_matching"]),
        grouping=grouping)
    report["reconstruction"] = trace.to_dict()
    print("stop:", trace.stop_reason, "outcome:", trace.best.outcome.value if trace.best else None)

    # 4. accuracy vs ground truth
    if trace.best_result is not None:
        report["pose_accuracy_vs_gt"] = evaluate_absolute_poses(
            list(trace.best_result.camera_poses), {k: v for k, v in gt_a.items() if k in members})

    # 5. persistence
    if trace.best is not None:
        ok, why = reg.commit_reconstruction("world-e2e", primary.group_id, trace.best, "v1",
                                            [p.evidence_id for p in trace.best_result.camera_poses])
        reloaded = SceneRegistry(work / "registry").load("world-e2e")
        report["persistence"] = {
            "adopted": ok, "decision": why,
            "reloaded_group_ids": [g["group_id"] for g in reloaded["groups"]],
            "reloaded_last_valid_registered": (reloaded["last_valid"]["diagnostics"]["registered_images"]
                                               if reloaded["last_valid"] else None)}

    # 6. insufficient overlap: three views from different sides of the room
    sparse_ids = [i for i in ("A_0000", "A_0006", "A_0012") if i in pa]
    g2 = group_images(sparse_ids, verifier)
    report["insufficient_overlap"] = {"ids": sparse_ids, "grouping": g2.to_dict()}
    if g2.primary is not None and len(g2.primary.members) >= 2:
        t2 = reconstruct_group(evidence({m: pa[m] for m in g2.primary.members}),
                               lambda cfg: ColmapReconstructionBackend(**cfg), grouping=g2)
        report["insufficient_overlap"]["trace"] = t2.to_dict()
    report["total_s"] = round(time.time() - t0, 1)
    Path(a.report).write_text(json.dumps(report, indent=1, default=str))
    print("report ->", a.report)


if __name__ == "__main__":
    main()
