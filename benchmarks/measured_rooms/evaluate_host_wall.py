"""Host-wall attachment precision / recall of a production run against the dataset's OWN Kinect depth (estimated reference, not a survey).

    PYTHONPATH=. python benchmarks/measured_rooms/evaluate_host_wall.py <run_dir> <frames_dir>

Reference (never given to the pipeline): every frame's 16-bit depth + KinectFusion pose are fused into one reference cloud and large planes are
fitted with Open3D RANSAC. A wall-class object (socket, switch plate, vent, mirror, TV, ...) is REFERENCE-ATTACHED when >= 60 % of the Kinect
points inside its detection boxes lie within 4 cm of a large VERTICAL reference plane. "Vertical" uses the up axis of the pipeline's canonical
frame mapped into the reference frame (the one pipeline-derived input; disclosed). The prediction is the world's mounted_on / embedded_in
relationship. Reported: TP / FP / FN / TN, precision, recall, false-attachment rate (FP / predicted) and, for true positives, whether the
predicted host wall is the reference plane. With a handful of candidates the numbers have wide uncertainty: counts are printed, not just rates.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from reconstruction.similarity import umeyama

K = (585.0, 585.0, 320.0, 240.0)
WALL_CLASS = ("socket", "switch", "vent", "mirror", "outlet", "tv", "television", "display", "picture", "door", "window", "thermostat", "light")
ON_PLANE_TOL_M, ON_PLANE_FRACTION, MIN_POINTS = 0.04, 0.60, 50


def _small(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L").resize((32, 24)), float)


def _reference_planes(frames: Path, names: list[str], up_gt: np.ndarray) -> list[dict]:
    import open3d as o3d

    fx, fy, cx, cy = K
    pts = []
    for n in names:
        d = np.array(Image.open(frames / f"{n}.depth.png")).astype(float) / 1000
        P = np.loadtxt(frames / f"{n}.pose.txt").reshape(4, 4)
        v, u = np.mgrid[0:480:6, 0:640:6]
        z = d[v, u]
        ok = (z > 0.3) & (z < 4.5)
        pc = np.stack([(u[ok] - cx) * z[ok] / fx, (v[ok] - cy) * z[ok] / fy, z[ok]], 1)
        pts.append(pc @ P[:3, :3].T + P[:3, 3])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(pts))).voxel_down_sample(0.02)
    planes, rest = [], cloud
    for _ in range(8):
        if len(rest.points) < 3000:
            break
        model, idx = rest.segment_plane(0.02, 3, 2000)
        if len(idx) < 3000:
            break
        n = np.asarray(model[:3], float)
        nrm = float(np.linalg.norm(n))
        n = n / nrm
        planes.append({"normal": n, "offset": float(-model[3] / nrm), "inliers": len(idx), "vertical": abs(float(n @ up_gt)) < 0.25})
        rest = rest.select_by_index(idx, invert=True)
    return planes


def main(run_dir: Path, frames: Path) -> int:
    from apps.api.evidence_world_stage import head_version
    from engine.multiengine.evidence_matrix import EvidenceMatrix
    from worldstore.store import WorldStore

    run = json.loads((run_dir / "run.json").read_text())
    ev_root = Path([p for p in glob.glob(str(run_dir / "ws/evidence-world/*/*/job_*")) if os.path.isdir(p)][0])
    world = WorldStore(ev_root / "worldstore").load_version(head_version(ev_root))
    names = sorted(p.name.replace(".color.png", "") for p in frames.glob("frame-*.color.png"))
    gt_by_hash = {hashlib.sha256(f.read_bytes()).hexdigest(): f.name.replace(".color.png", "") for f in frames.glob("frame-*.color.png")}
    h2n = {v: gt_by_hash.get(v, k.replace(".color.png", "")) for k, v in run["hashes"].items()}
    cams = None
    for f in glob.glob(str(run_dir / "art/**/*"), recursive=True):
        if os.path.isfile(f) and os.path.getsize(f) < 20000:
            b = open(f, "rb").read()
            if b[:1] == b"{" and b"rotation_convention" in b[:600]:
                cams = json.loads(b)
    cam_pos = {c["evidence_id"]: np.array(c["position_m"]) for c in cams["cameras"]}
    orig = {n: _small(frames / f"{n}.color.png") for n in names}
    sess = glob.glob(str(run_dir / "ws/colmap-sessions/*/current/images"))[0]
    ev2n = {os.path.splitext(f)[0]: min(orig, key=lambda n: ((orig[n] - _small(Path(sess) / f)) ** 2).mean()) for f in os.listdir(sess)}
    ks = sorted(set(cam_pos) & set(ev2n))
    A = np.array([cam_pos[k] for k in ks])
    B = np.array([np.loadtxt(frames / f"{ev2n[k]}.pose.txt").reshape(4, 4)[:3, 3] for k in ks])
    s, R, t = umeyama(A, B)
    up_gt = R @ np.array([0.0, 1.0, 0.0])  # the canonical frame's +Y (pipeline-derived), expressed in the reference frame
    planes = _reference_planes(frames, names, up_gt)
    verts = [p for p in planes if p["vertical"]]
    print(f"reference planes: {len(planes)} large, {len(verts)} vertical; inlier counts {[p['inliers'] for p in planes]}")

    m = EvidenceMatrix(ev_root)
    fx, fy, cx, cy = K
    depth_cache: dict[str, tuple] = {}
    rows = []
    for e in world.entities.values():
        path = (((e.custom_properties or {}).get("evidence") or {}).get("ontology") or {}).get("path", "")
        if not e.id.startswith("ent-") or not any(k in path.lower() for k in WALL_CLASS):
            continue
        pts = []
        for mid in ((e.custom_properties or {}).get("evidence") or {}).get("members", []):
            o = m.store.get(mid)
            if o.quantity != "bbox" or o.image_region is None:
                continue
            n = ev2n.get(os.path.splitext(os.path.basename(o.source_id))[0]) or h2n.get(os.path.basename(o.source_id))
            if n is None:
                continue
            if n not in depth_cache:
                depth_cache[n] = (np.array(Image.open(frames / f"{n}.depth.png")).astype(float) / 1000, np.loadtxt(frames / f"{n}.pose.txt").reshape(4, 4))
            d, P = depth_cache[n]
            x0, y0, x1, y1 = (max(0, int(round(v))) for v in o.image_region)
            x1, y1 = min(640, x1), min(480, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            v, u = np.mgrid[y0:y1:2, x0:x1:2]
            z = d[v, u]
            ok = (z > 0.3) & (z < 4.5)
            if ok.sum() < 10:
                continue
            pc = np.stack([(u[ok] - cx) * z[ok] / fx, (v[ok] - cy) * z[ok] / fy, z[ok]], 1)
            pts.append(pc @ P[:3, :3].T + P[:3, 3])
        pred = [r for r in e.relationships if str(getattr(r.kind, "value", r.kind)) in ("mounted_on", "embedded_in")]
        npts = int(sum(len(p) for p in pts))
        row = {"entity": e.id[-10:], "class": path, "ref_points": npts, "predicted_attached": bool(pred), "predicted_host": pred[0].target_id if pred else None}
        if npts >= MIN_POINTS and verts:
            X = np.vstack(pts)
            frac, plane = max(((float(np.mean(np.abs(X @ p["normal"] - p["offset"]) <= ON_PLANE_TOL_M)), p) for p in verts), key=lambda x: x[0])
            row.update(ref_on_plane_fraction=round(frac, 2), ref_attached=frac >= ON_PLANE_FRACTION, ref_plane_inliers=plane["inliers"])
            if pred and row["ref_attached"]:  # does the predicted host wall coincide with the reference plane?
                g = world.geometries.get(world.entities[pred[0].target_id].geometry_ids[0])
                pl = (g.quality_metrics or {}).get("plane") if g else None
                if pl:
                    ng = R @ np.asarray(pl["normal"], float)
                    cg = (s * np.asarray(g.quality_metrics["observed_hull"], float) @ R.T + t).mean(0)
                    row["host_matches_reference"] = bool(abs(float(ng @ plane["normal"])) > 0.94 and abs(float(cg @ plane["normal"] - plane["offset"])) < 0.10)
        else:
            row["ref_attached"] = None  # too few Kinect points / no vertical reference plane: NOT measured
        rows.append(row)
    ev = [r for r in rows if r["ref_attached"] is not None]
    tp = sum(1 for r in ev if r["predicted_attached"] and r["ref_attached"])
    fp = sum(1 for r in ev if r["predicted_attached"] and not r["ref_attached"])
    fn = sum(1 for r in ev if not r["predicted_attached"] and r["ref_attached"])
    tn = sum(1 for r in ev if not r["predicted_attached"] and not r["ref_attached"])
    res = {"candidates_wall_class": len(rows), "evaluable": len(ev), "not_measured": len(rows) - len(ev), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
           "precision": (tp / (tp + fp)) if tp + fp else None, "recall": (tp / (tp + fn)) if tp + fn else None,
           "false_attachment_rate": (fp / (tp + fp)) if tp + fp else None,
           "host_matches_reference_on_TP": [r.get("host_matches_reference") for r in ev if r["predicted_attached"] and r["ref_attached"]],
           "reference": "Kinect depth + KinectFusion poses (estimates); up axis from the pipeline's canonical frame (disclosed)",
           "tolerances": {"on_plane_m": ON_PLANE_TOL_M, "on_plane_fraction": ON_PLANE_FRACTION, "min_points": MIN_POINTS}, "rows": rows}
    (run_dir / "host_wall_evaluation.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, default=str))
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
