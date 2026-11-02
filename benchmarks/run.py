#!/usr/bin/env python3
"""Benchmark runner (blueprint task 2.1, metrics from docs/blueprint/18).

Drives the REAL product pipeline -- upload -> API -> job worker -> real COLMAP /
MiDaS -> WorldIR -> WorldStore -- through the same FastAPI app the Studio uses
(the same route the golden journey tests use), then reads the measured result
back through the product's own read routes.

    python benchmarks/run.py --dataset south_building [--max-images N]
    python benchmarks/run.py --dataset living_room        # reports "dataset not found" if absent

Output: benchmarks/results/<YYYY-MM-DD>-<short-git-sha>.json

Honesty rule: a metric that cannot be measured is written as
``{"value": null, "reason": "..."}``. Nothing is defaulted or invented. Only
metrics that exist for this run are emitted with a number.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = REPO / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _m(value, reason: str | None = None) -> dict:
    """A metric: a measured value, or null with the reason it is not available."""
    if value is None:
        return {"value": None, "reason": reason or "not measured"}
    return {"value": value}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "nogit"


# --------------------------------------------------------------- resource sampler


class ResourceSampler(threading.Thread):
    """Samples RSS (this process + all children, e.g. the COLMAP subprocess) and GPU memory
    (nvidia-smi, whole device minus the baseline taken before the run)."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.peak_rss = 0
        self.peak_vram_mib: float | None = None
        self.vram_baseline_mib: float | None = None
        self.vram_error: str | None = None
        try:
            import psutil

            self._proc = psutil.Process()
        except Exception as exc:  # pragma: no cover
            self._proc = None
            self.rss_error = f"psutil unavailable: {exc}"
        else:
            self.rss_error = None
        self.vram_baseline_mib = self._vram()

    def _vram(self) -> float | None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=5, stderr=subprocess.DEVNULL,
            )
            return float(out.strip().splitlines()[0])
        except Exception as exc:
            self.vram_error = f"nvidia-smi unavailable: {exc.__class__.__name__}"
            return None

    def _sample(self) -> None:
        if self._proc is not None:
            try:
                rss = self._proc.memory_info().rss
                for ch in self._proc.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except Exception:
                        pass
                self.peak_rss = max(self.peak_rss, rss)
            except Exception:
                pass
        v = self._vram()
        if v is not None and (self.peak_vram_mib is None or v > self.peak_vram_mib):
            self.peak_vram_mib = v

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._sample()
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=5)
        self._sample()


# --------------------------------------------------------------- the product pipeline


@contextmanager
def app_client(root: Path):
    """The product API over a scratch directory (own DB + WorldStore)."""
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


def _find_key(obj, key: str):
    """First value for ``key`` anywhere in a nested JSON structure (None if absent)."""
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _stage_timings(report: dict) -> dict:
    """Per-stage wall seconds as recorded by the pipeline itself (report['stages']['timings_s']), plus the
    COLMAP sub-stage split (extract_match / mapper) when the report carries it."""
    stages = (report or {}).get("stages") or {}
    out: dict = {}
    t = stages.get("timings_s")
    if isinstance(t, dict):
        out.update({k: float(v) for k, v in t.items() if isinstance(v, (int, float))})
    cs = ((stages.get("reconstruction") or {}).get("colmap_session") or {}).get("seconds")
    if isinstance(cs, dict):
        out["colmap_substages"] = {k: float(v) for k, v in cs.items() if isinstance(v, (int, float))}
    return out


def run_pipeline(images: list[Path], timeout: float) -> dict:
    root = Path(tempfile.mkdtemp(prefix="bench_run_"))
    try:
        with app_client(root) as c:
            files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in images]
            t0 = time.perf_counter()
            r = c.post("/api/reconstructions", files=files)
            upload_s = time.perf_counter() - t0
            if r.status_code != 202:
                raise RuntimeError(f"upload rejected: {r.status_code} {r.text[:300]}")
            wid = r.json()["world_id"]
            t1 = time.perf_counter()
            end = time.time() + timeout
            st = None
            while time.time() < end:
                st = c.get(f"/api/worlds/{wid}/status").json()
                if not st.get("in_progress"):
                    break
                time.sleep(1.0)
            else:
                raise TimeoutError(f"world {wid} did not settle in {timeout}s; last status: {st}")
            pipeline_s = time.perf_counter() - t1
            wir = c.get(f"/api/worlds/{wid}/worldir")
            worldir = wir.json() if wir.status_code == 200 else {}
            rep = c.get(f"/api/worlds/{wid}/report")
            report = rep.json() if rep.status_code == 200 else {}
            return {"status": st, "worldir": worldir, "report": report,
                    "upload_s": upload_s, "pipeline_s": pipeline_s}
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------- metrics


def derive_metrics(n_images: int, run: dict, sampler: ResourceSampler, total_s: float) -> dict:
    st, wir, report = run["status"], run["worldir"], run["report"]
    model = (st or {}).get("model") or {}
    ents = (wir or {}).get("entities") or {}
    metrics: dict = {}

    used, reg = model.get("images_used"), model.get("images_registered")
    if used:
        metrics["registration_ratio"] = {"value": round(reg / used, 4), "registered": reg, "total": used}
    elif model:
        metrics["registration_ratio"] = _m(None, "no multi-view reconstruction attempt recorded")
    else:
        metrics["registration_ratio"] = _m(None, f"no model produced (state={st.get('state')})")

    err = _find_key(report, "mean_reprojection_error_px")
    if err is None:
        err = _find_key(st, "mean_reprojection_error_px")
    metrics["mean_reprojection_error_px"] = (
        _m(round(float(err), 4)) if err is not None
        else _m(None, "the COLMAP backend computes it (reconstruction/backend/colmap_backend.py) but the "
                      "API status/report routes do not surface it, so it is not observable through the product path")
    )

    by_type: dict[str, int] = {}
    for e in ents.values():
        by_type[e.get("type", "unknown")] = by_type.get(e.get("type", "unknown"), 0) + 1
    metrics["entity_counts_by_type"] = {"value": by_type, "total": len(ents)} if ents else _m(None, "WorldIR has no entities")

    ident = model.get("identity")
    dup = _find_key(ident, "duplicates") if ident else None
    metrics["duplicate_entity_count"] = _m(
        len(dup) if isinstance(dup, (list, dict)) else (dup if isinstance(dup, int) else None),
        "no ground-truth entity identities for this dataset and the pipeline reports no "
        "duplicate-candidate list for this run; duplicate rate is not measurable",
    )

    metrics["runtime_s"] = {
        "value": {"total_wall": round(total_s, 3), "upload": round(run["upload_s"], 3),
                  "pipeline_job": round(run["pipeline_s"], 3)},
        "per_stage_s": _stage_timings(report) or None,
        "per_stage_note": None if _stage_timings(report) else
        "pipeline report exposes no per-stage wall times; only upload and whole-job wall time are measured",
    }

    metrics["peak_ram_mib"] = (
        _m(round(sampler.peak_rss / 2**20, 1)) if sampler.peak_rss
        else _m(None, sampler.rss_error or "no RSS sample taken")
    )
    if sampler.peak_vram_mib is None:
        metrics["peak_vram_mib"] = _m(None, sampler.vram_error or "no GPU memory sample (CPU-only)")
    else:
        metrics["peak_vram_mib"] = {
            "value": round(sampler.peak_vram_mib, 1),
            "baseline_before_run_mib": sampler.vram_baseline_mib,
            "note": "whole-device nvidia-smi memory.used peak (includes other GPU processes)",
        }
    return metrics


def collect_images(name: str, max_images: int | None) -> tuple[list[Path] | None, str | None]:
    ds = DATASETS / name
    img_dir = ds / "images"
    if not ds.is_dir():
        return None, f"dataset not found: {ds}"
    src = img_dir if img_dir.is_dir() else ds
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not imgs:
        return None, f"dataset not found: no images under {src}"
    return (imgs[:max_images] if max_images else imgs), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="south_building", help="folder under datasets/ (south_building | living_room)")
    ap.add_argument("--max-images", type=int, default=None, help="use only the first N name-sorted images")
    ap.add_argument("--dump-raw", type=Path, default=None, help="also write the raw status/report JSON here (debug)")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds to wait for the job to settle")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{dt.date.today().isoformat()}-{_git_sha()}.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    result: dict = {
        "schema": "reality-engine.benchmark/1",
        "git_sha": _git_sha(),
        "date": dt.date.today().isoformat(),
        "platform": {"python": platform.python_version(), "os": platform.platform()},
    }
    result["runs"] = existing.get("runs", {}) if existing.get("git_sha") == result["git_sha"] else {}

    images, problem = collect_images(args.dataset, args.max_images)
    if images is None:
        result["runs"][args.dataset] = {"status": "dataset not found", "reason": problem}
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{args.dataset}: dataset not found ({problem}); recorded in {out}")
        return 0

    run_rec: dict = {
        "dataset": args.dataset, "images_available": len(list(
            p for p in (DATASETS / args.dataset / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)),
        "images_used": len(images), "max_images_flag": args.max_images,
        "pipeline": "apps.api POST /api/reconstructions -> job worker (apps/api/jobs.py) -> real COLMAP/MiDaS",
    }
    sampler = ResourceSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        run = run_pipeline(images, args.timeout)
    except Exception as exc:
        sampler.stop()
        run_rec["status"] = "failed"
        run_rec["error"] = f"{exc.__class__.__name__}: {exc}"
        result["runs"][args.dataset] = run_rec
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(run_rec, indent=2))
        return 1
    total = time.perf_counter() - t0
    sampler.stop()
    if args.dump_raw:
        args.dump_raw.write_text(json.dumps({"status": run["status"], "report": run["report"]}, indent=1, default=str),
                                 encoding="utf-8")

    st = run["status"] or {}
    run_rec["status"] = "ok"
    run_rec["world_state"] = st.get("state")
    run_rec["model_level"] = (st.get("model") or {}).get("level_name")
    # the pipeline's own account of what it tried (e.g. why COLMAP found no initial pair)
    run_rec["pipeline_attempts"] = (st.get("model") or {}).get("attempts")
    run_rec["pipeline_report_stage_keys"] = sorted(((run["report"] or {}).get("stages") or {}).keys()) or None
    run_rec["metrics"] = derive_metrics(len(images), run, sampler, total)
    result["runs"][args.dataset] = run_rec
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
