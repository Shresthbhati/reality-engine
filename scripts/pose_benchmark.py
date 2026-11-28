"""Benchmark camera-pose candidates on one dataset against its reference model.

    python scripts/pose_benchmark.py --dataset datasets/south_building \
        --colmap C:/Users/shres/tools/colmap-extracted/bin/colmap --out benchmarks/results/pose_benchmark_south_building.json

Candidates (same images, same SIFT features + exhaustive matches, only the mapper differs):
  incremental   `colmap mapper`
  global        `colmap global_mapper` (GLOMAP-style rotation + position averaging)

The reference is the COLMAP model shipped with the dataset. It is NOT independent ground truth (it is
itself a COLMAP result) and its scale is arbitrary, so scores measure agreement with it, in its units.
Every run records commit, image ids, config, hardware, versions, runtime, peak VRAM and failure logs.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.camera_pose_graph import read_colmap_text_model  # noqa: E402
from perception.pose_validation import evaluate_trajectory, validate_poses  # noqa: E402


def _run(cmd, log):
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    log.append({"cmd": " ".join(str(c) for c in cmd[1:3]), "rc": p.returncode, "s": round(time.time() - t0, 2),
                "stderr_tail": p.stderr[-400:] if p.returncode else ""})
    return p.returncode == 0


class _Vram(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.halt = 0, False

    def run(self):
        while not self.halt:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout.split()[0]
                self.peak = max(self.peak, int(out))
            except Exception:
                pass
            time.sleep(0.25)


def _model_poses(txt_dir):
    return {n: (p.R, p.t) for n, p in read_colmap_text_model(txt_dir).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/south_building")
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = Path(a.dataset)
    ref = _model_poses(ds / "sparse")
    work = Path(tempfile.mkdtemp(prefix="posebench_"))
    db = work / "db.db"
    log: list = []
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    ver = subprocess.run([a.colmap, "help"], capture_output=True, text=True).stdout.splitlines()[0]
    import numpy
    import scipy
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    result = {"commit": sha, "dataset": str(ds), "split": "all images (no split)", "colmap": ver,
              "hardware": {"gpu": gpu, "platform": platform.platform()},
              "deps": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__},
              "image_ids": sorted(p.name for p in (ds / "images").iterdir()),
              "reference": {"kind": "COLMAP model shipped with dataset (not independent)", "scale": "relative",
                            "cameras": len(ref)},
              "config": {"features": "SIFT default", "matching": "exhaustive", "checkpoints": "none (classical)"},
              "candidates": {}}
    vram = _Vram()
    vram.start()
    time.sleep(0.6)
    base = vram.peak
    ok = _run([a.colmap, "feature_extractor", "--database_path", db, "--image_path", ds / "images"], log) and \
        _run([a.colmap, "exhaustive_matcher", "--database_path", db], log)
    result["shared_stage_log"] = log
    for name, sub in (("incremental", "mapper"), ("global", "global_mapper")):
        if not ok:
            result["candidates"][name] = {"unavailable": "feature extraction/matching failed"}
            continue
        l2: list = []
        out = work / name
        out.mkdir()
        t0 = time.time()
        good = _run([a.colmap, sub, "--database_path", db, "--image_path", ds / "images", "--output_path", out], l2)
        runtime = round(time.time() - t0, 2)
        models = sorted(p for p in out.iterdir() if p.is_dir()) if good else []
        if not models:
            result["candidates"][name] = {"unavailable": "no model produced", "log": l2}
            continue
        # largest model = most registered images; smaller ones are reported as disconnected components
        sizes = {}
        for m in models:
            txt = work / f"{name}_{m.name}_txt"
            txt.mkdir()
            if _run([a.colmap, "model_converter", "--input_path", m, "--output_path", txt, "--output_type", "TXT"], l2):
                sizes[m.name] = (len(_model_poses(txt)), txt)
        k = max(sizes, key=lambda x: sizes[x][0])
        est = _model_poses(sizes[k][1])
        order = sorted(est)
        result["candidates"][name] = {
            "mapper_runtime_s": runtime, "models": {m: s[0] for m, s in sizes.items()},
            "registered": len(est), "of": len(result["image_ids"]),
            "unregistered": sorted(set(result["image_ids"]) - set(est)),
            "eval": evaluate_trajectory(est, ref, est_scale_state="relative", ref_scale_state="relative",
                                        order=order),
            "validation": validate_poses(est, order=order), "log": l2}
    vram.halt = True
    result["peak_vram_mib_incl_baseline"] = vram.peak
    result["baseline_vram_mib"] = base
    result["note"] = "image order for RPE is filename order (a photo set, not a video): RPE is indicative only"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, default=float))
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({n: {k: v for k, v in c.items() if k in ("registered", "mapper_runtime_s", "unavailable")}
                      for n, c in result["candidates"].items()}, indent=1))


if __name__ == "__main__":
    main()
