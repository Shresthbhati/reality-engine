"""Ground-truth benchmark of the real photos -> world chain on a KNOWN synthetic box room.

Drives the product API exactly as the UI does (POST /api/reconstructions, poll /api/worlds/{id}/status, read worldir /
points / cameras) over a temp database, then scores the result against the ground truth ``synthetic.room_photos``
rendered. Scale-free metrics only; photos carry no metric scale, so lengths are "model units".

    python -m benchmarks.room_e2e --capture <dir> --mode images|video [--out results.json]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def app_client(root: Path):
    """A fresh app's worth of state (database, artifacts, WorldStore) under ``root``."""
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(root / 'app.db').as_posix()}"
    os.environ["STORAGE_ROOT"] = str(root / "artifacts")
    os.environ["WORLDSTORE_ROOT"] = str(root / "ws")
    from fastapi.testclient import TestClient
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws
    ws._store = None
    with TestClient(main_mod.app) as c:
        yield c


def run_chain(c, paths, timeout: float = 1800.0) -> dict:
    """Upload ``paths`` through the real route and wait for the job. Returns raw API answers + wall time."""
    mime = lambda p: "video/mp4" if p.suffix.lower() == ".mp4" else "image/jpeg"
    files = [("files", (p.name, p.read_bytes(), mime(p))) for p in paths]
    t0 = time.time()
    r = c.post("/api/reconstructions", files=files)
    ctype = r.headers.get("content-type", "")
    out = {"post_status": r.status_code, "post": r.json() if ctype.startswith("application/json") else r.text}
    if r.status_code != 202:
        return out
    wid = out["post"]["world_id"]
    stages, st = [], None
    while time.time() - t0 < timeout:
        st = c.get(f"/api/worlds/{wid}/status").json()
        stg = (st.get("job") or {}).get("stage")
        if stg and (not stages or stages[-1][0] != stg):
            stages.append((stg, round(time.time() - t0, 1)))
        if not st["in_progress"]:
            break
        time.sleep(0.5)
    out.update(wall_s=round(time.time() - t0, 1), stages=stages, status=st, world_id=wid)
    if st and st.get("has_model"):
        out["worldir"] = c.get(f"/api/worlds/{wid}/worldir").json()
        out["cameras"] = c.get(f"/api/worlds/{wid}/cameras").json()
        pr = c.get(f"/api/worlds/{wid}/points")
        out["points_ply_bytes"] = len(pr.content) if pr.status_code == 200 else pr.status_code
        out["_points_raw"] = pr.content if pr.status_code == 200 else b""
    return out


def _load_points(root: Path, data_uri: str):
    import numpy as np
    from world_ir.artifact_store import FileArtifactStore
    from world_ir.geometry_data import PointCloudData
    for sub in ("pipeline-artifacts", "artifacts"):
        try:
            raw = FileArtifactStore(root / "ws" / sub).get(data_uri)
        except Exception:
            continue
        if raw[:8] == b"RESPC001":
            return np.asarray([tuple(p[:3]) for p in PointCloudData.from_bytes(raw).points], float)
    return np.zeros((0, 3))


def evaluate(res: dict, gt: dict, root: Path) -> dict:
    """Score one run (``run_chain`` output) against ``ground_truth.json``. Scale-free; see benchmarks.room_eval."""
    import re
    import numpy as np
    from benchmarks import room_eval as ev
    out = {"registered_images": None, "state": (res.get("status") or {}).get("state")}
    st, wir, cams = res.get("status") or {}, res.get("worldir"), res.get("cameras")
    if not (wir and cams):
        return out
    room = gt["room"]
    diag = float(np.sqrt(room["width"] ** 2 + room["depth"] ** 2 + room["height"] ** 2))
    names = {e["id"]: e["name"] for e in st.get("evidence", [])}
    est_c, est_R, gt_c, gt_R = [], [], [], []
    by_img = {c["image"]: c for c in gt["images"]}
    for cam in cams["cameras"]:
        nm = names.get(cam["evidence_id"], "")
        m = re.search(r"#frame-(\d+)", nm)
        if m:
            fi = int(m.group(1))
            g = gt["video_frames"][fi] if fi < len(gt["video_frames"]) else None
        else:
            g = by_img.get(nm)
        if g is None:
            continue
        est_c.append(cam["position_m"])
        est_R.append(ev.quat_wxyz_to_R(cam["rotation_wxyz"]))
        gt_c.append(g["C"])
        gt_R.append(np.asarray(g["R_wc"]))
    out["registered_images"] = len(est_c)
    out["total_cameras_reported"] = len(cams["cameras"])
    if len(est_c) < 4:
        out["note"] = "fewer than 4 cameras matched to ground truth; no alignment possible"
        return out
    cam_score, sim = ev.score_cameras(np.asarray(est_c), np.asarray(est_R), np.asarray(gt_c), np.asarray(gt_R), diag)
    out["cameras"] = cam_score
    ents, geoms = wir["entities"], wir["geometries"]
    dense, planes = None, []
    for gid, g in geoms.items():
        if g["type"] == "pointcloud" and (dense is None or g["vertex_count"] > dense[0]):
            dense = (g["vertex_count"], g["data_uri"])
    for e in ents.values():
        if e["type"] in ("wall", "floor", "ceiling"):
            for gid in e["geometry_ids"]:
                g = geoms.get(gid)
                if g and g.get("data_uri"):
                    p = _load_points(root, g["data_uri"])
                    if len(p):
                        planes.append(p)
    out["entities"] = {t: sum(1 for e in ents.values() if e["type"] == t) for t in sorted({e["type"] for e in ents.values()})}
    if dense:
        out["dense_points"] = ev.score_points(_load_points(root, dense[1]), sim, room)
    out["planes"] = ev.score_planes(planes, sim, room)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--mode", choices=["images", "video"], default="images")
    ap.add_argument("--reeval", default="", help="re-score a saved results.json against --capture/--workdir without re-running")
    ap.add_argument("--n", type=int, default=0, help="use only the first N images")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cap = Path(a.capture)
    if a.reeval:
        res = json.loads(Path(a.reeval).read_text())
        print(json.dumps(evaluate(res, json.loads((cap / "ground_truth.json").read_text()), Path(a.workdir)), indent=1, default=str))
        return
    paths = sorted((cap / "images").glob("*.jpg")) if a.mode == "images" else [cap / "room.mp4"]
    if a.n:
        paths = paths[: a.n]
    import tempfile
    root = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="room_e2e_"))
    with app_client(root) as c:
        res = run_chain(c, paths)
    raw = res.pop("_points_raw", b"")
    gtp = cap / "ground_truth.json"
    if gtp.exists() and res.get("worldir"):
        res["evaluation"] = evaluate(res, json.loads(gtp.read_text()), root)
        print(json.dumps(res["evaluation"], indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
        if raw:
            Path(a.out).with_suffix(".ply").write_bytes(raw)
    st = res.get("status") or {}
    print(json.dumps({k: res.get(k) for k in ("post_status", "wall_s", "stages")}, default=str))
    keep = ("level", "level_name", "model_state", "outcome", "images_used", "images_registered", "scale")
    print(json.dumps({"state": st.get("state"), "model": {k: v for k, v in (st.get("model") or {}).items() if k in keep},
                      "failure": st.get("failure")}, default=str)[:2000])


if __name__ == "__main__":
    main()
