"""Reproduce / guard: repeated and large exports must never terminate the API process.

Starts the real API (uvicorn, own process) on a persisted data directory, then for N rounds exports the world in several formats, downloads each
artifact, and polls /healthz, while sampling the server's RSS. Reports the exit status the moment the process dies (with the last log lines), so the
cause is observed, not guessed.

    PYTHONPATH=. python scripts/repro_export_stability.py --world wld_xxx --rounds 6 --db data/verify_app.db --storage data/verify_artifacts \
        --worldstore data/verify_worldstore [--formats gltf,glb,obj,ply,stl,usda] [--port 8301]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil


def _req(method: str, url: str, body: dict | None = None, timeout: float = 120.0) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default=None, help="an existing world id; omit when --photos is given")
    ap.add_argument("--photos", default=None, help="directory of photographs: create a world, upload, RECONSTRUCT in the same server, then export")
    ap.add_argument("--n-photos", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--formats", default="gltf,glb,obj,ply,stl,usda")
    ap.add_argument("--port", type=int, default=8301)
    ap.add_argument("--db", required=True)
    ap.add_argument("--storage", required=True)
    ap.add_argument("--worldstore", required=True)
    a = ap.parse_args()
    env = {**os.environ, "DATABASE_URL": f"sqlite+aiosqlite:///{Path(a.db).resolve().as_posix()}", "STORAGE_ROOT": str(Path(a.storage).resolve()),
           "WORLDSTORE_ROOT": str(Path(a.worldstore).resolve()), "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.getcwd()}
    log = Path(tempfile.gettempdir()) / f"export_stability_{a.port}.log"
    with open(log, "wb") as lf:
        proc = subprocess.Popen([sys.executable, "-c", f"import uvicorn; uvicorn.run('apps.api.main:app', host='127.0.0.1', port={a.port})"],
                                env=env, stdout=lf, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{a.port}"
    ps = psutil.Process(proc.pid)
    rss = [ps.memory_info().rss]
    t0 = time.time()
    while time.time() - t0 < 90:
        try:
            if _req("GET", f"{base}/healthz", timeout=2)[0] == 200:
                break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    results, dead = [], None
    formats = a.formats.split(",")
    timeline = []  # (seconds, rss_mb, system_available_mb, phase)

    def sample(phase: str) -> None:
        if proc.poll() is None:
            timeline.append((round(time.time() - t0, 1), round(ps.memory_info().rss / 2**20, 1), round(psutil.virtual_memory().available / 2**20), phase))

    if a.photos:
        import httpx

        h = httpx.Client(base_url=base, timeout=300)
        sid = h.post("/api/sessions", json={"name": "repro"}).json()["id"]
        a.world = h.post("/api/worlds", json={"name": "repro world"}).json()["id"]
        h.post(f"/api/worlds/{a.world}/attach/{sid}").raise_for_status()
        for f in sorted(Path(a.photos).glob("*.[jJ][pP][gG]"))[: a.n_photos]:
            h.post(f"/api/uploads?session_id={sid}", files={"file": (f.name, f.read_bytes(), "image/jpeg")}).raise_for_status()
        job = h.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
        t_job = time.time()
        while proc.poll() is None:
            sample("reconstruct")
            j = h.get(f"/api/jobs/{job}").json()
            if j.get("terminal"):
                results.append({"phase": "reconstruct", "job": job, "status": j["status"], "seconds": round(time.time() - t_job, 1),
                                "stages": [{"name": i["name"], "status": i["status"], "duration_s": i.get("duration_s"),
                                            **{k: v for k, v in (i.get("metrics") or {}).items() if k.startswith("mem_")}}
                                           for i in h.get(f"/api/jobs/{job}/stages").json()["items"]]})
                break
            time.sleep(5)
        if proc.poll() is not None:
            dead = {"exit_code": proc.returncode, "during": "reconstruction"}
    for rnd in range(a.rounds):
        for fmt in formats:
            sample(f"export {fmt}")
            if proc.poll() is not None:
                dead = {"exit_code": proc.returncode, "during": f"round {rnd} format {fmt}"}
                break
            t = time.time()
            code, raw = _req("POST", f"{base}/api/worlds/{a.world}/export", {"format": fmt})
            row = {"round": rnd, "format": fmt, "export_status": code, "export_s": round(time.time() - t, 2)}
            if code == 200:
                url = json.loads(raw)["download_url"]
                c2, body = _req("GET", base + url)
                row.update(download_status=c2, bytes=len(body))
            if proc.poll() is None:
                row["healthz"] = _req("GET", f"{base}/healthz", timeout=5)[0]
                rss.append(ps.memory_info().rss)
                row["rss_mb"] = round(rss[-1] / 2**20, 1)
            results.append(row)
        if dead:
            break
    if proc.poll() is not None and not dead:
        dead = {"exit_code": proc.returncode, "during": "after the last export"}
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=20)
    tail = log.read_text(errors="replace").splitlines()[-12:]
    out = {"rounds": a.rounds, "formats": formats, "api_died": dead, "requests": len(results), "peak_rss_mb": round(max(rss) / 2**20, 1),
           "first_rss_mb": round(rss[0] / 2**20, 1), "last_rss_mb": round(rss[-1] / 2**20, 1), "min_system_available_mb": min((t[2] for t in timeline), default=None),
           "world": a.world, "log": str(log), "log_tail": tail, "memory_timeline": timeline[:: max(1, len(timeline) // 40)], "results": results}
    print(json.dumps(out, indent=1))
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
