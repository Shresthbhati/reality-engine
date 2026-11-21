"""Run the REAL production job (real COLMAP, real models, evidence-world stage ON) on photographs of a furnished office from the
Microsoft 7-Scenes "office" sequences, and keep every artefact so the result can be validated independently.

7-Scenes (Shotton et al. 2013): Kinect RGB-D frames with per-frame camera poses from KinectFusion and 16-bit depth in millimetres.
LICENCE: research / non-commercial (Microsoft Research). The photographs are NOT committed; this script only reads a local folder.

    python benchmarks/measured_rooms/run_office7_production.py <frames_dir> <out_dir>

``frames_dir`` holds ``frame-XXXXXX.color.png`` files (the depth / pose files beside them are NOT given to the pipeline; they are the
independent ground truth used by ``validate_office7.py``). Only the colour photographs are uploaded."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(frames: Path, out: Path) -> int:
    photos = sorted(frames.glob("frame-*.color.png"))
    if len(photos) < 6:
        print(f"need >= 6 photographs, found {len(photos)}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    env = {
        "DATABASE_URL": f"sqlite+aiosqlite:///{(out / 'app.db').as_posix()}", "STORAGE_ROOT": str(out / "art"),
        "WORLDSTORE_ROOT": str(out / "ws"), "RE_EVIDENCE_WORLD": "1",
        "RE_EVIDENCE_WORLD_DEPTH_ENGINES": os.environ.get("RE_EVIDENCE_WORLD_DEPTH_ENGINES", "midas,depth_anything"),
    }
    os.environ.update(env)
    os.environ.pop("REALITY_TEST_BACKEND", None)  # the REAL backend
    from fastapi.testclient import TestClient

    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws

    ws._store = None
    H = {"x-tenant-id": "tenant-office7"}
    t0 = time.time()
    with TestClient(main_mod.app) as c:
        sid = c.post("/api/sessions", json={"name": "office7"}, headers=H).json()["id"]
        wid = c.post("/api/worlds", json={"name": "office7"}, headers=H).json()["id"]
        assert c.post(f"/api/worlds/{wid}/attach/{sid}", headers=H).status_code == 200
        hashes = {}
        for p in photos:
            data = p.read_bytes()
            r = c.post(f"/api/uploads?session_id={sid}", headers=H, files={"file": (p.name, data, "image/png")})
            assert r.status_code == 201, r.text
            hashes[p.name] = hashlib.sha256(data).hexdigest()
        jid = c.post(f"/api/sessions/{sid}/reconstruct", headers=H).json()["job_id"]
        stages = []
        job = {}
        while time.time() - t0 < 3600:
            job = c.get(f"/api/jobs/{jid}", headers=H).json()
            if not stages or stages[-1][1] != job.get("stage"):
                stages.append((round(time.time() - t0, 1), job.get("stage"), job.get("status")))
            if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
                break
            time.sleep(5)
        rd = c.get(f"/api/worlds/{wid}/evidence-world/{jid}/readiness", headers=H).json()
        run = {"job_id": jid, "world_id": wid, "session_id": sid, "tenant": "tenant-office7", "hashes": hashes,
               "seconds": round(time.time() - t0, 1), "stage_timeline": stages, "job": job, "readiness": rd}
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"status": job.get("status"), "seconds": run["seconds"], "readiness_world": rd.get("world")}, default=str))
    return 0 if job.get("status") in ("succeeded", "partial") else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
