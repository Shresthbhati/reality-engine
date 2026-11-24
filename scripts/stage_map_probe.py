"""Re-runnable production-path probe: runs ONE real API reconstruction job (real COLMAP + real models, NOT
REALITY_TEST_BACKEND) on a folder of photographs and emits ``docs/audits/stage_map_run.json`` with measured data.

    python scripts/stage_map_probe.py <photos_dir> [--pattern "IMG_*.jpg"] [--scratch <dir>] [--no-evidence-world]

What it measures (all observed, none estimated):
  * per pipeline stage (the 15 ``job_stage_runs`` rows): wall time, started/ended, recorded peak RSS, warnings, output hashes
  * a 0.5 s sampler: RSS of this process, RSS of its child processes (COLMAP), and GPU memory via nvidia-smi (total used and
    per-process), each sample labelled with the job's current user-facing stage; peaks are reported per label
  * the job payload, the committed version's report (scale/depth/timings), artifact content hashes, and one call of
    every available export format with the content hash it returned
Photographs are copied to the scratch directory (never committed). The scratch DB / stores are throw-away."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_JSON = ROOT / "docs" / "audits" / "stage_map_run.json"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Sampler(threading.Thread):
    """RSS (self + children) and VRAM, every ``period`` s, tagged with ``label`` (set by the poller)."""

    def __init__(self, period: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.period, self.label, self.samples, self._halt = period, "init", [], threading.Event()
        self.t0 = time.time()

    @staticmethod
    def _vram() -> tuple[float | None, dict]:
        try:
            tot = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            per = {}
            for ln in apps:
                if "," in ln:
                    pid, mb = [x.strip() for x in ln.split(",", 1)]
                    per[pid] = float(mb) if mb.replace(".", "").isdigit() else None
            return float(tot[0]), per
        except Exception:  # noqa: BLE001
            return None, {}

    def run(self) -> None:
        import psutil

        me = psutil.Process()
        while not self._halt.is_set():
            try:
                rs = me.memory_info().rss / 2**20
                kids = {}
                for c in me.children(recursive=True):
                    try:
                        kids[c.name()] = kids.get(c.name(), 0.0) + c.memory_info().rss / 2**20
                    except psutil.Error:
                        pass
                used, per = self._vram()
                self.samples.append({"t": round(time.time() - self.t0, 2), "label": self.label, "rss_self_mb": round(rs, 1),
                                     "rss_children_mb": {k: round(v, 1) for k, v in kids.items()},
                                     "vram_used_mb": used, "vram_per_pid": per})
            except Exception:  # noqa: BLE001 -- sampling must never kill the run
                pass
            self._halt.wait(self.period)

    def stop(self) -> None:
        self._halt.set()

    def peaks_by_label(self) -> dict:
        out: dict = {}
        for s in self.samples:
            d = out.setdefault(s["label"], {"samples": 0, "peak_rss_self_mb": 0.0, "peak_rss_children_mb": 0.0,
                                            "peak_rss_total_mb": 0.0, "peak_vram_used_mb": 0.0, "children_seen": set()})
            kids = sum(s["rss_children_mb"].values())
            d["samples"] += 1
            d["peak_rss_self_mb"] = max(d["peak_rss_self_mb"], s["rss_self_mb"])
            d["peak_rss_children_mb"] = max(d["peak_rss_children_mb"], kids)
            d["peak_rss_total_mb"] = max(d["peak_rss_total_mb"], s["rss_self_mb"] + kids)
            if s["vram_used_mb"] is not None:
                d["peak_vram_used_mb"] = max(d["peak_vram_used_mb"], s["vram_used_mb"])
            d["children_seen"].update(s["rss_children_mb"].keys())
        for d in out.values():
            d["children_seen"] = sorted(d["children_seen"])
        return out


def _tree_hashes(root: Path, limit: int = 4000) -> dict:
    out = {}
    if not root.exists():
        return out
    for i, p in enumerate(sorted(x for x in root.rglob("*") if x.is_file())):
        if i >= limit:
            out["__truncated__"] = True
            break
        out[p.relative_to(root).as_posix()] = {"bytes": p.stat().st_size, "sha256": _file_sha(p)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", type=Path)
    ap.add_argument("--pattern", default="IMG_*.jpg")
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--no-evidence-world", action="store_true", help="run the DEFAULT production config (RE_EVIDENCE_WORLD unset)")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    a = ap.parse_args()

    src = sorted(a.photos.glob(a.pattern))
    if len(src) < 6:
        print(f"need >= 6 photographs, found {len(src)}")
        return 2
    scratch = a.scratch or Path(tempfile.mkdtemp(prefix="stage_map_"))
    photos_dir = scratch / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    for p in src:  # photographs live only in the scratch dir
        shutil.copy2(p, photos_dir / p.name)
    photos = sorted(photos_dir.glob(a.pattern))

    env = {"DATABASE_URL": f"sqlite+aiosqlite:///{(scratch / 'app.db').as_posix()}", "STORAGE_ROOT": str(scratch / "art"),
           "WORLDSTORE_ROOT": str(scratch / "ws"), "RE_MEMORY_ADMISSION": os.environ.get("RE_MEMORY_ADMISSION", "record")}
    if not a.no_evidence_world:
        env["RE_EVIDENCE_WORLD"] = "1"
        env["RE_EVIDENCE_WORLD_DEPTH_ENGINES"] = os.environ.get("RE_EVIDENCE_WORLD_DEPTH_ENGINES", "midas,depth_anything")
    os.environ.update(env)
    os.environ.pop("REALITY_TEST_BACKEND", None)  # the REAL backend
    from fastapi.testclient import TestClient

    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws

    ws._store = None
    H = {"x-tenant-id": "tenant-stagemap"}
    sampler = Sampler()
    run: dict = {"probe": "scripts/stage_map_probe.py", "photos": len(photos), "photo_sha256": {}, "env": {k: v for k, v in env.items() if "DATABASE" not in k},
                 "evidence_world_enabled": not a.no_evidence_world,
                 "contention_note": "COLMAP/GPU may have been shared with another process during this run; see host_processes_at_start",
                 "python": sys.version.split()[0]}
    try:
        import psutil

        run["host_processes_at_start"] = sorted({p.name() for p in psutil.process_iter() if "colmap" in p.name().lower()})
        run["host_cpu_count"] = psutil.cpu_count()
        run["host_cpu_percent_at_start"] = psutil.cpu_percent(interval=0.5)
    except Exception:  # noqa: BLE001
        pass
    import faulthandler  # where is every thread every 3 min: a hang's location is then on disk

    _fh = open(scratch / "stacks.log", "w")
    faulthandler.dump_traceback_later(180, repeat=True, file=_fh)
    t0 = time.time()
    sampler.t0 = t0
    sampler.start()
    stages_timeline: list = []
    with TestClient(main_mod.app, raise_server_exceptions=False) as c:
        sid = c.post("/api/sessions", json={"name": "stage-map"}, headers=H).json()["id"]
        wid = c.post("/api/worlds", json={"name": "stage-map"}, headers=H).json()["id"]
        assert c.post(f"/api/worlds/{wid}/attach/{sid}", headers=H).status_code == 200
        t_up = time.time()
        for p in photos:
            data = p.read_bytes()
            r = c.post(f"/api/uploads?session_id={sid}", headers=H, files={"file": (p.name, data, "image/jpeg")})
            assert r.status_code == 201, r.text
            run["photo_sha256"][p.name] = _sha(data)
        run["upload_seconds"] = round(time.time() - t_up, 2)
        jid = c.post(f"/api/sessions/{sid}/reconstruct", headers=H).json()["job_id"]
        t_job = time.time()
        job: dict = {}
        poll_errors: list = []
        while time.time() - t0 < a.timeout:
            try:
                r = c.get(f"/api/jobs/{jid}", headers=H)
                if r.status_code != 200:  # observed: SQLite 'database is locked' -> HTTP 500 while the stage thread writes
                    poll_errors.append({"t": round(time.time() - t0, 1), "http": r.status_code, "body": r.text[:120]})
                    time.sleep(2.0)
                    continue
                job = r.json()
            except Exception as exc:  # noqa: BLE001
                poll_errors.append({"t": round(time.time() - t0, 1), "exc": f"{type(exc).__name__}: {str(exc)[:120]}"})
                time.sleep(2.0)
                continue
            st = job.get("stage")
            sampler.label = str(st)
            if int(time.time() - t0) % 60 == 0:  # interim record so an interruption loses nothing
                try:
                    a.out.with_suffix(".partial.json").write_text(json.dumps({"timeline": stages_timeline, "job": job, "poll_errors": poll_errors,
                                                                              "peaks": sampler.peaks_by_label(), "elapsed_s": round(time.time() - t0)}, default=str), encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
            if not stages_timeline or stages_timeline[-1]["stage"] != st:
                stages_timeline.append({"t_since_start_s": round(time.time() - t0, 1), "t_since_job_s": round(time.time() - t_job, 1),
                                        "stage": st, "status": job.get("status")})
            if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
                break
            time.sleep(1.0)
        run["poll_errors"] = poll_errors
        run["job_wall_seconds"] = round(time.time() - t_job, 1)
        run["job"] = job
        run["stage_timeline"] = stages_timeline
        sampler.label = "post_job_api_reads"
        calls: dict = {}

        def get(name: str, url: str) -> dict | None:
            try:
                r = c.get(url, headers=H)
                body = r.json() if "json" in r.headers.get("content-type", "") else None
                calls[name] = {"status": r.status_code, "bytes": len(r.content), "sha256": _sha(r.content)}
                return body
            except Exception as exc:  # noqa: BLE001
                calls[name] = {"error": f"{type(exc).__name__}: {exc}"}
                return None

        run["world_report"] = get("report", f"/api/worlds/{wid}/report")
        get("worldir", f"/api/worlds/{wid}/worldir")
        get("points", f"/api/worlds/{wid}/points")
        get("cameras", f"/api/worlds/{wid}/cameras")
        run["quality"] = get("quality", f"/api/worlds/{wid}/quality")
        run["versions"] = get("versions", f"/api/worlds/{wid}/versions")
        run["spatial_intelligence"] = get("spatial_intelligence", f"/api/worlds/{wid}/spatial-intelligence")
        run["evidence_world_runs"] = get("evidence_world_list", f"/api/worlds/{wid}/evidence-world")
        if run["evidence_world_runs"]:
            run["evidence_world_readiness"] = get("evidence_world_readiness", f"/api/worlds/{wid}/evidence-world/{jid}/readiness")
            run["evidence_world_manifest"] = get("evidence_world_manifest", f"/api/worlds/{wid}/evidence-world/{jid}/manifest")
        fm = get("export_formats", f"/api/worlds/{wid}/export/formats")
        exports = {}
        for f in (fm or {}).get("formats", []):
            if not f.get("available"):
                exports[f["format"]] = {"available": False, "reason": f.get("reason")}
                continue
            try:
                r = c.post(f"/api/worlds/{wid}/export", json={"format": f["format"]}, headers=H)
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                exports[f["format"]] = {"http": r.status_code, "content_hash": body.get("content_hash"), "artifact_uri": body.get("artifact_uri"),
                                        "entities_exported": len(body.get("entities_exported") or []), "skip_reasons": (body.get("skip_reasons") or [])[:5],
                                        "scale": body.get("scale"), "evidence_summary": body.get("evidence_summary"),
                                        "detail": (body.get("detail") if r.status_code >= 400 else None)}
            except Exception as exc:  # noqa: BLE001
                exports[f["format"]] = {"error": f"{type(exc).__name__}: {exc}"}
        run["exports"] = exports
        run["api_reads"] = calls
        run["viewer"] = {"note": "no browser driven by this probe: only that the viewer's data endpoints (/points, /cameras, /worldir) answered", "endpoints": {k: calls.get(k) for k in ("points", "cameras", "worldir")}}
        run["world_id"], run["session_id"], run["job_id"] = wid, sid, jid
    sampler.stop()
    sampler.join(timeout=5)
    run["total_seconds"] = round(time.time() - t0, 1)
    run["peaks_by_job_stage"] = sampler.peaks_by_label()
    run["peak_overall"] = {
        "rss_self_mb": max((s["rss_self_mb"] for s in sampler.samples), default=None),
        "rss_children_mb": max((sum(s["rss_children_mb"].values()) for s in sampler.samples), default=None),
        "rss_total_mb": max((s["rss_self_mb"] + sum(s["rss_children_mb"].values()) for s in sampler.samples), default=None),
        "vram_used_mb": max((s["vram_used_mb"] for s in sampler.samples if s["vram_used_mb"] is not None), default=None),
        "samples": len(sampler.samples), "period_s": sampler.period,
    }
    run["sampler_trace_every_10th"] = sampler.samples[::10]
    # per pipeline stage rows, straight from the DB the job wrote
    try:
        con = sqlite3.connect(str(scratch / "app.db"))
        con.row_factory = sqlite3.Row
        rows = con.execute("select name, version, status, attempt, inputs_hash, outputs, metrics, warnings, error, peak_rss_mb, started_at, ended_at "
                           "from job_stage_runs where job_id=? order by started_at", (run["job_id"],)).fetchall()
        stage_rows = []
        for r in rows:
            d = dict(r)
            for k in ("outputs", "metrics", "warnings"):
                try:
                    d[k] = json.loads(d[k]) if isinstance(d[k], str) else d[k]
                except Exception:  # noqa: BLE001
                    pass
            try:
                from datetime import datetime

                s, e = datetime.fromisoformat(str(d["started_at"])), datetime.fromisoformat(str(d["ended_at"]))
                d["wall_s"] = round((e - s).total_seconds(), 3)
            except Exception:  # noqa: BLE001
                d["wall_s"] = None
            stage_rows.append(d)
        run["job_stage_runs"] = stage_rows
        con.close()
    except Exception as exc:  # noqa: BLE001
        run["job_stage_runs_error"] = f"{type(exc).__name__}: {exc}"
    run["artifact_hashes"] = {"worldstore": _tree_hashes(scratch / "ws"), "storage_root_files": len(list((scratch / "art").rglob("*"))) if (scratch / "art").exists() else 0}
    run["scratch_dir"] = str(scratch)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(run, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"status": job.get("status"), "job_wall_s": run["job_wall_seconds"], "peak": run["peak_overall"], "out": str(a.out)}, default=str))
    return 0 if job.get("status") in ("succeeded", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
