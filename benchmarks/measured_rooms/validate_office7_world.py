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
    tree = cKDTree(np.vstack(pts))
    ents = [e for e in w.entities.values() if (e.transform or {}).get("position")]
    pos = np.array([[e.transform["position"][k] for k in "xyz"] for e in ents])
    d, _ = tree.query(s * pos @ R.T + t)
    print(f"positioned entities {len(ents)}; centre -> nearest Kinect surface point: median {np.median(d) * 100:.1f} cm, "
          f"p75 {np.percentile(d, 75) * 100:.1f} cm, p90 {np.percentile(d, 90) * 100:.1f} cm; within 10 cm {(d < 0.10).mean() * 100:.0f}%, "
          f"within 25 cm {(d < 0.25).mean() * 100:.0f}%")
    multi = [e for e in w.entities.values() if len({o.data_uri for o in e.observations if o.metadata.get("quantity") == "bbox"}) > 1]
    print(f"entities with bbox evidence from >1 photograph: {len(multi)} of {len(w.entities)} (single-view {len(w.entities) - len(multi)}); "
          f"positioned and multi-view: {sum(1 for e in multi if (e.transform or {}).get('position'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
