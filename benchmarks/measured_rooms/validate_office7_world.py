"""Validate the WORLD a production run persisted from 7-Scenes "office" photographs against the dataset's own ground truth.

    PYTHONPATH=. python benchmarks/measured_rooms/validate_office7_world.py <run_dir> <frames_dir>

``run_dir`` is the output of ``run_office7_production.py``; ``frames_dir`` holds the ``frame-*.color.png / .depth.png / .pose.txt`` files.
Reports: a FRESH-PROCESS reload of the persisted world; how many entities are positioned / multi-view; and how far each positioned
entity's centre lies from the nearest Kinect surface point after a similarity (fitted on camera centres) maps the job frame to the
ground-truth frame. Honest limits: Kinect depth / KinectFusion poses are estimates with centimetre error; "near a surface" shows the
position lies on observed geometry, NOT that the object's class or full extent is right; and the metres come from the ground-truth
similarity -- the pipeline itself stays unanchored (``NOT_MEASURED``)."""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from reconstruction.similarity import umeyama

K = (585.0, 585.0, 320.0, 240.0)  # published 7-Scenes Kinect RGB intrinsics (fx, fy, cx, cy)


def _small(p):
    return np.asarray(Image.open(p).convert("L").resize((32, 24)), float)


def _extents(P: np.ndarray) -> np.ndarray:
    """5th-95th percentile extent along each axis (the lifter's own robust extent), after a depth-band inlier filter."""
    return np.percentile(P, 95, axis=0) - np.percentile(P, 5, axis=0)


def _dimension_check(run, run_dir: Path, ev_root: Path, frames: Path, names, s: float, R: np.ndarray, t: np.ndarray, ev2n: dict | None = None) -> None:
    """Lifting-step accuracy: re-measure each lifted box's pixels from the KINECT depth (same box, same robust extent, expressed in the
    job's axes and converted to metres by the ground-truth similarity) and compare with the pipeline's lifted dimensions. This tests the
    lifting (depth alignment + unprojection), NOT the detector: the box itself is taken as given."""
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    import hashlib

    # content hash -> the ground-truth frame whose bytes the run was given (exact, unlike the run's possibly renamed upload names)
    gt_by_hash = {hashlib.sha256(f.read_bytes()).hexdigest(): f.name.replace(".color.png", "") for f in frames.glob("frame-*.color.png")}
    h2n = {v: gt_by_hash.get(v, k.replace(".color.png", "")) for k, v in run["hashes"].items()}
    m = EvidenceMatrix(ev_root)
    fx, fy, cx, cy = K
    ratios, abs_err, rows = [], [], 0
    zs: list[float] = []  # |error| / predicted sigma, for the axes that carry a stated covariance (uncertainty calibration)
    cache: dict[str, tuple] = {}
    for o in m.store.query(quantity="dimensions"):
        if not o.engine_id.startswith("lift:") or o.image_region is None or m.status(o.observation_id) == "rejected":
            continue
        n = (ev2n or {}).get(os.path.splitext(os.path.basename(o.source_id))[0]) or h2n.get(os.path.basename(o.source_id))
        if n is None:
            continue
        if n not in cache:
            cache[n] = (np.array(Image.open(frames / f"{n}.depth.png")).astype(float) / 1000, np.loadtxt(frames / f"{n}.pose.txt").reshape(4, 4))
        d, P = cache[n]
        x0, y0, x1, y1 = (int(round(v)) for v in o.image_region)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(640, x1), min(480, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        v, u = np.mgrid[y0:y1, x0:x1]
        z = d[v, u]
        ok = (z > 0.2) & (z < 5)
        if ok.sum() < 20:
            continue
        zz, med = z[ok], np.median(z[ok])
        mad = np.median(np.abs(zz - med))
        keep = np.abs(zz - med) <= (max(0.05, 3 * mad) if mad > 0.001 else 0.2 * med)
        pc = np.stack([(u[ok][keep] - cx) * zz[keep] / fx, (v[ok][keep] - cy) * zz[keep] / fy, zz[keep]], 1)
        world_gt = pc @ P[:3, :3].T + P[:3, 3]
        in_job = (world_gt - t) @ R / s  # inverse of dst = s * R @ src + t, in the job's axes
        gt_ext = _extents(in_job) * s  # metres
        got = np.asarray(o.predicted["value"], float) * s
        big = gt_ext > 0.05  # an axis with no real extent has no meaningful ratio
        if big.any():
            ratios.extend((got[big] / gt_ext[big]).tolist())
            abs_err.extend(np.abs(got - gt_ext)[big].tolist())
            cov = (o.uncertainty or {}).get("covariance")
            if cov is not None:
                sig = np.sqrt(np.clip(np.diag(np.asarray(cov, float)), 1e-18, None)) * s
                zs.extend((np.abs(got - gt_ext)[big] / sig[big]).tolist())
            rows += 1
    if not ratios:
        print("dimension check: no comparable boxes")
        return
    r, a = np.array(ratios), np.array(abs_err)
    print(f"LIFTING dimension check vs Kinect depth over {rows} boxes ({len(r)} axes with extent > 5 cm): "
          f"lifted/Kinect extent ratio median {np.median(r):.2f} (p25 {np.percentile(r, 25):.2f}, p75 {np.percentile(r, 75):.2f}); "
          f"absolute error median {np.median(a) * 100:.1f} cm, p90 {np.percentile(a, 90) * 100:.1f} cm")
    if zs:
        z = np.array(zs)
        print(f"UNCERTAINTY CALIBRATION (dimensions, {len(z)} axes with a stated sigma): |error|/sigma median {np.median(z):.1f}, "
              f"within 1 sigma {np.mean(z <= 1):.0%}, within 2 sigma {np.mean(z <= 2):.0%}, within 3 sigma {np.mean(z <= 3):.0%} "
              "(a calibrated 1-sigma would give ~68% / 95% / 99.7%); the stated sigma is a sampling-spread LOWER BOUND, so under-coverage is expected")


def _surface_check(world, gt_points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> None:
    """Does the Kinect see a real surface where each persisted plane says there is one? For every plane geometry: the Kinect points inside
    the plane's observed footprint and within 5 cm of it (support count) and their median distance to it. A real plane has thousands of
    supporting points at ~1-2 cm; a spurious one has few. Says nothing about the plane's CLASS (floor vs table) -- the Kinect frame has
    no known up axis."""
    from scipy.spatial import Delaunay

    in_job = (gt_points - t) @ R / s  # Kinect points expressed in the job frame
    rows = []
    for e in world.entities.values():
        if not e.id.startswith("surf-") or not e.geometry_ids:
            continue
        q = world.geometries[e.geometry_ids[0]].quality_metrics or {}
        if "plane" not in q:
            continue
        n, off = np.asarray(q["plane"]["normal"], float), float(q["plane"]["offset"])
        hull = np.asarray(q["observed_hull"], float)
        a = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        c = hull.mean(0)
        to2 = lambda P: np.stack([(P - c) @ a, (P - c) @ b], 1)  # noqa: E731
        try:
            inside = Delaunay(to2(hull)).find_simplex(to2(in_job)) >= 0
        except Exception:  # noqa: BLE001
            continue
        dist = np.abs(in_job @ n - off) * s * 100  # cm
        near = inside & (dist < 5.0)
        rows.append((e.id, int(near.sum()), float(np.median(dist[near])) if near.any() else None, int(inside.sum())))
    print("SURFACES vs Kinect (plane id: Kinect points within 5 cm of the plane inside its observed footprint, median residual cm, footprint points):")
    for r in rows:
        print(f"  {r[0]}: {r[1]} pts, median {r[2]:.1f} cm, of {r[3]} in footprint" if r[2] is not None else f"  {r[0]}: 0 pts (no Kinect surface on this plane)")


def main(run_dir: Path, frames: Path) -> int:
    from apps.api.evidence_world_stage import head_version
    from worldstore.store import WorldStore

    run = json.loads((run_dir / "run.json").read_text())
    names = [k.replace(".color.png", "") for k in run["hashes"]]
    ev_root = Path([p for p in glob.glob(str(run_dir / "ws/evidence-world/*/*/job_*")) if os.path.isdir(p)][0])
    code = ("import json,sys;from worldstore.store import WorldStore;from apps.api.evidence_world_stage import head_version;from pathlib import Path;"
            "r=Path(sys.argv[1]);v=head_version(r);s=WorldStore(r/'worldstore');w=s.load_version(v);"
            "print(json.dumps({'version':v,'integrity_problems':s.verify_version(v),'entities':len(w.entities),"
            "'positioned':sum(1 for e in w.entities.values() if (e.transform or {}).get('position')),'geometries':len(w.geometries)}))")
    fresh = subprocess.run([sys.executable, "-c", code, str(ev_root)], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    print("FRESH-PROCESS RELOAD:", fresh.stdout.strip() or fresh.stderr[-300:])
    w = WorldStore(ev_root / "worldstore").load_version(head_version(ev_root))

    cams_payload = None  # the job's served cameras.json: {"cameras": [{evidence_id, position_m, rotation_wxyz}]}
    for f in glob.glob(str(run_dir / "art/**/*"), recursive=True):
        if os.path.isfile(f) and os.path.getsize(f) < 20000:
            b = open(f, "rb").read()
            if b[:1] == b"{" and b"rotation_convention" in b[:600]:
                cams_payload = json.loads(b)
    cams = {c["evidence_id"]: np.array(c["position_m"]) for c in cams_payload["cameras"]}
    # candidates come from the ground-truth directory itself: the uploaded files may have been renamed (a subset copied as frame-000000..N),
    # so the run's own file names are not a key into the pose / depth files
    names = sorted(p.name.replace(".color.png", "") for p in frames.glob("frame-*.color.png"))
    orig = {n: _small(frames / f"{n}.color.png") for n in names}
    sess = glob.glob(str(run_dir / "ws/colmap-sessions/*/current/images"))[0]
    ev2n = {os.path.splitext(f)[0]: min(orig, key=lambda n: ((orig[n] - _small(os.path.join(sess, f))) ** 2).mean()) for f in os.listdir(sess)}
    ks = sorted(set(cams) & set(ev2n))
    A = np.array([cams[k] for k in ks])
    B = np.array([np.loadtxt(frames / f"{ev2n[k]}.pose.txt").reshape(4, 4)[:3, 3] for k in ks])
    s, R, t = umeyama(A, B)
    print(f"job->GT similarity: scale {s:.4f}, camera-centre RMSE {np.sqrt((np.linalg.norm(s * A @ R.T + t - B, axis=1) ** 2).mean()):.4f} m")

    pts = []
    fx, fy, cx, cy = K
    for n in orig:
        d = np.array(Image.open(frames / f"{n}.depth.png")).astype(float) / 1000
        P = np.loadtxt(frames / f"{n}.pose.txt").reshape(4, 4)
        v, u = np.mgrid[0:480:4, 0:640:4]
        z = d[v, u]
        ok = (z > 0.2) & (z < 5)
        pc = np.stack([(u[ok] - cx) * z[ok] / fx, (v[ok] - cy) * z[ok] / fy, z[ok]], 1)
        pts.append(pc @ P[:3, :3].T + P[:3, 3])
    G = np.vstack(pts)
    tree = cKDTree(G)
    ents = [e for e in w.entities.values() if (e.transform or {}).get("position")]
    pos = np.array([[e.transform["position"][k] for k in "xyz"] for e in ents])
    d, _ = tree.query(s * pos @ R.T + t)
    print(f"positioned entities {len(ents)}; centre -> nearest Kinect surface point: median {np.median(d) * 100:.1f} cm, "
          f"p75 {np.percentile(d, 75) * 100:.1f} cm, p90 {np.percentile(d, 90) * 100:.1f} cm; within 10 cm {(d < 0.10).mean() * 100:.0f}%, "
          f"within 25 cm {(d < 0.25).mean() * 100:.0f}%")
    _dimension_check(run, run_dir, ev_root, frames, orig, s, R, t, ev2n)
    _surface_check(w, G, s, R, t)
    multi = [e for e in w.entities.values() if len({o.data_uri for o in e.observations if o.metadata.get("quantity") == "bbox"}) > 1]
    print(f"entities with bbox evidence from >1 photograph: {len(multi)} of {len(w.entities)} (single-view {len(w.entities) - len(multi)}); "
          f"positioned and multi-view: {sum(1 for e in multi if (e.transform or {}).get('position'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
