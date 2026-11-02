#!/usr/bin/env python3
"""Evaluate a FINISHED product-path reconstruction against the committed South Building reference.

The product path never sees the reference: this tool reads a workspace after the fact (the app's SQLite database and the
persistent COLMAP session under the WorldStore root) and only then compares with datasets/south_building/sparse.

Reports, as one JSON document:
  - which photographs registered / did not (original file names), sparse point count
  - pose error against the reference after Sim(3) alignment (reconstruction.evaluation), with outlier cameras named
  - reference-free structure: per-camera support (sparse points only), tie groups (reconstruction.diagnostics)
  - for every unregistered photograph: its verified-match partners and inlier counts from COLMAP's own database
  - optionally the fused dense cloud (--dense-ply) measured against the reference points

Usage:
  python scripts/evaluate_session_model.py --db data/expdense3.db --workspace data/expdense3_ws --world wld_xxx \
         [--dense-ply <artifact file> --report <GET /api/worlds/{id}/report saved as json>] [--out result.json]

The dense cloud is stored in the re-solved world frame; it is mapped back through the registered camera centres
(report["cameras"]), and the fit residual is reported so a bad mapping cannot pass silently.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REFERENCE = REPO / "datasets" / "south_building" / "sparse"


def _read_images(path: Path):
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return {lines[i].split()[0]: lines[i].split()[9] for i in range(0, len(lines), 2)}, \
        {lines[i].split()[9]: tuple(float(x) for x in lines[i].split()[1:8]) for i in range(0, len(lines), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--dense-ply")
    ap.add_argument("--report")
    ap.add_argument("--out")
    a = ap.parse_args()

    from reconstruction.backend.colmap_backend import _parse_images_txt, _parse_points3d_txt
    from reconstruction.diagnostics import analyze
    from reconstruction.evaluation import evaluate_absolute_poses, evaluate_point_cloud

    name_of = {r[0]: r[1] for r in sqlite3.connect(a.db).execute("select id,name from evidence")}
    session = Path(a.workspace) / "colmap-sessions" / a.world / "current"
    model = session / "sparse" / "0"
    id_by_image, _ = _read_images(model / "images.txt")
    ev_by_image = {k: v[:-4] if v.endswith(".jpg") else v for k, v in id_by_image.items()}
    poses = [replace(p, evidence_id=name_of[p.evidence_id[:-4]])
             for p in _parse_images_txt((model / "images.txt").read_text(), {})]
    points = _parse_points3d_txt((model / "points3D.txt").read_text(), {k: name_of[v] for k, v in ev_by_image.items()})
    input_names = sorted(name_of.values())

    out: dict = {"world": a.world, "input": len(input_names), "registered": sorted(p.evidence_id for p in poses),
                 "unregistered": [n for n in input_names if n not in {p.evidence_id for p in poses}],
                 "sparse_points": len(points)}
    out["diagnostics"] = analyze(points, poses, input_names)

    _, ref_poses = _read_images(REFERENCE / "images.txt")
    ev = evaluate_absolute_poses(poses, ref_poses)
    out["pose_evaluation"] = ev
    gt_points = {}
    for ln in (REFERENCE / "points3D.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            f = ln.split()
            gt_points[int(f[0])] = tuple(float(x) for x in f[1:4])
    if not ev.get("refused"):
        out["sparse_vs_reference"] = evaluate_point_cloud(points, gt_points, tuple(ev["gauge_transform"]))

    # why each unregistered photograph failed: the strongest verified partners COLMAP itself found
    con = sqlite3.connect(session / "database.db")
    imgs = {i: n for i, n in con.execute("select image_id,name from images")}
    keypoints = {i: r for i, r in con.execute("select image_id,rows from keypoints")}
    nm = lambda i: name_of[imgs[i][:-4]]  # noqa: E731
    inliers = {}
    for pid, rows, _cfg in con.execute("select pair_id,rows,config from two_view_geometries"):
        lo, hi = divmod(pid, 2147483647)
        if rows:
            inliers[(lo, hi)] = rows
    registered = set(out["registered"])
    out["unregistered_detail"] = []
    for name in out["unregistered"]:
        iid = next(i for i in imgs if nm(i) == name)
        partners = sorted(((v, nm(x if y == iid else y)) for (x, y), v in inliers.items() if iid in (x, y)), reverse=True)
        out["unregistered_detail"].append({
            "image": name, "keypoints": keypoints.get(iid), "verified_pairs": len(partners),
            "registered_partners_ge_15_inliers": sum(1 for v, p in partners if p in registered and v >= 15),
            "top_partners": [{"image": p, "inliers": v, "registered": p in registered} for v, p in partners[:5]]})

    if a.dense_ply and a.report:
        import numpy as np
        from scipy.spatial import cKDTree
        from reconstruction.evaluation import umeyama_sim3

        cams = json.load(open(a.report))["cameras"]
        raw = _parse_images_txt((model / "images.txt").read_text(), {})
        centres = {p.evidence_id[:-4]: np.array(p.position) for p in raw}
        ids = sorted(centres)
        R, t, s = umeyama_sim3([tuple(centres[i]) for i in ids], [tuple(cams[i]) for i in ids])
        R, t = np.array(R), np.array(t)
        resid = float(np.linalg.norm(np.array([s * (R @ centres[i]) + t for i in ids]) - np.array([cams[i] for i in ids]), axis=1).max())
        raw_bytes = Path(a.dense_ply).read_bytes()
        n = int(raw_bytes.split(b"element vertex ")[1].split(b"\n")[0])
        xyz = np.frombuffer(raw_bytes[raw_bytes.index(b"end_header\n") + 11:], dtype="<f4").reshape(-1, 3).astype(float)
        dense_n = n - len(points)
        dense = xyz[-dense_n:]
        Rg, tg, sg = ev["gauge_transform"]
        mapped = sg * (((dense - t) / s) @ R @ np.array(Rg).T) + np.array(tg)
        ref = np.array(list(gt_points.values()))
        lo, hi = np.percentile(ref, 2, axis=0), np.percentile(ref, 98, axis=0)
        diag = float(np.linalg.norm(hi - lo))
        d, _ = cKDTree(ref).query(mapped)
        cover, _ = cKDTree(mapped).query(ref)
        out["dense_vs_reference"] = {
            "frame_fit_residual_max": resid, "dense_points": int(dense_n), "scene_diagonal": round(diag, 3),
            "dist_median": float(np.median(d)), "dist_p90": float(np.percentile(d, 90)), "dist_p99": float(np.percentile(d, 99)),
            "share_within_1pct_diag": float((d < 0.01 * diag).mean()), "share_within_5pct_diag": float((d < 0.05 * diag).mean()),
            "reference_points_covered_within_2pct_diag": float((cover < 0.02 * diag).mean()),
            "note": "distance to the reference's SPARSE points: a proxy, not a surface-to-surface error"}

    text = json.dumps(out, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
