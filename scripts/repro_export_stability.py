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
    ap.add_argument("--world", required=True)
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
    for rnd in range(a.rounds):
        for fmt in formats:
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
           "first_rss_mb": round(rss[0] / 2**20, 1), "last_rss_mb": round(rss[-1] / 2**20, 1), "log": str(log), "log_tail": tail, "results": results}
    print(json.dumps(out, indent=1))
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
