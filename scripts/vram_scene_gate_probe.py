#!/usr/bin/env python3
"""Measure GPU memory before/after ``apps.api.jobs._scene_gate`` (D3). One variant per process:

    python scripts/vram_scene_gate_probe.py datasets/user_room_nov14/images            # with the fix (release_vram in finally)
    python scripts/vram_scene_gate_probe.py datasets/user_room_nov14/images --no-release   # control: release_vram patched to a no-op

Prints torch allocated/reserved and whole-device free VRAM (nvidia-smi) before the gate, right after it, and after the
gate's locals are gone. Whole-device numbers include other processes; run on an otherwise idle GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def smi() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return dict(zip(("used_mb", "free_mb", "total_mb"), (int(x) for x in out)))


def snap(torch) -> dict:
    return {"torch_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
            "torch_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1), **{f"smi_{k}": v for k, v in smi().items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("--no-release", action="store_true")
    a = ap.parse_args()
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA not available; nothing to measure"}))
        return 1
    import engine.core.device as dev
    from apps.api import jobs, worldstore_service
    import tempfile

    if a.no_release:
        dev.release_vram = lambda: None
    tmp = Path(tempfile.mkdtemp())
    worldstore_service.worldstore_root = lambda: tmp
    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    evals = [SimpleNamespace(item=SimpleNamespace(id=f"ev_{i:03d}"), facts=SimpleNamespace(geometry_eligible=True), path=p)
             for i, p in enumerate(paths)]
    torch.zeros(1, device="cuda")                       # create the CUDA context first so "before" is a fair baseline
    before = snap(torch)
    out = jobs._scene_gate("probe", evals, {})
    after = snap(torch)
    gc.collect()
    settled = snap(torch)
    print(json.dumps({"variant": "no-release (control)" if a.no_release else "release_vram in finally", "images": len(paths),
                      "gate_state": out["state"], "before": before, "after_gate": after, "after_gc": settled}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
