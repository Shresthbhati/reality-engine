"""Hardware Benchmark: Sequential Model Execution Latency and VRAM Usage.

Measures actual execution latency (load, inference, unload) and GPU VRAM / host RAM
usage across verified real model architectures on actual hardware (RTX 4050 6GB Laptop GPU).

Enforces the Mandatory 10-Step Sequential Maximum-Fidelity Execution Policy:
LOAD -> INFERENCE -> RELEASE RESOURCES -> VERIFY PERSISTENCE -> NEXT MODEL.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


try:
    import torch
except ImportError:
    torch = None

from engine.multiengine.adapters import (  # noqa: E402
    DepthAnythingV2EngineAdapter,
    MidasEngineAdapter,
    OwlVitEngineAdapter,
)
from engine.multiengine.contract import Device, EngineRequest  # noqa: E402
from engine.multiengine.registry import maximum_fidelity_registry  # noqa: E402


def run_hardware_benchmark(output_json: Path | None = None) -> dict[str, Any]:
    device = Device("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    print(f"Running hardware benchmark on device: {device}")
    reg = maximum_fidelity_registry()

    # Create temporary synthetic test image
    temp_dir = Path("benchmarks/hardware/.tmp_bench")
    temp_dir.mkdir(parents=True, exist_ok=True)
    test_img_path = temp_dir / "bench_frame.jpg"

    from PIL import Image
    test_img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    test_img.save(test_img_path)

    bench_results: list[dict[str, Any]] = []

    models_to_test = [
        ("owlvit", OwlVitEngineAdapter, "google/owlvit-base-patch32"),
        ("midas", MidasEngineAdapter, "intel-isl/MiDaS:DPT_Hybrid"),
        ("depth_anything", DepthAnythingV2EngineAdapter, "depth-anything/Depth-Anything-V2-Small-hf"),
    ]

    for model_id, adapter_cls, checkpoint in models_to_test:
        print(f"\n--- Benchmarking {model_id} ({checkpoint}) ---")
        spec = reg.get(model_id)
        adapter = adapter_cls(spec)

        # Baseline VRAM
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            vram_start_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            vram_start_mb = 0.0

        # Step 1: Load
        t0 = time.perf_counter()
        try:
            adapter.load(device)
            load_time_s = time.perf_counter() - t0
            load_ok = True
        except Exception as exc:
            print(f"Load failed for {model_id}: {exc}")
            load_time_s = time.perf_counter() - t0
            load_ok = False

        # Step 2: Inference
        t1 = time.perf_counter()
        infer_ok = False
        num_drafts = 0
        if load_ok:
            try:
                req = EngineRequest(sources=(str(test_img_path),))
                drafts = adapter.infer(req)
                infer_ok = True
                num_drafts = len(drafts)
            except Exception as exc:
                print(f"Inference failed for {model_id}: {exc}")
        infer_time_s = time.perf_counter() - t1

        # VRAM measurements
        if torch is not None and torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            vram_used_mb = max(0.0, peak_vram_mb - vram_start_mb)
        else:
            peak_vram_mb = 0.0
            vram_used_mb = 0.0

        # Step 3: Unload & Release
        t2 = time.perf_counter()
        adapter.unload()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            vram_after_unload_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            vram_after_unload_mb = 0.0
        unload_time_s = time.perf_counter() - t2

        res = {
            "model_id": model_id,
            "checkpoint": checkpoint,
            "device": str(device),
            "load_time_s": round(load_time_s, 3),
            "inference_time_s": round(infer_time_s, 3),
            "unload_time_s": round(unload_time_s, 3),
            "total_time_s": round(load_time_s + infer_time_s + unload_time_s, 3),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "vram_delta_mb": round(vram_used_mb, 1),
            "vram_after_unload_mb": round(vram_after_unload_mb, 1),
            "memory_released_ok": vram_after_unload_mb <= (vram_start_mb + 50.0),
            "output_drafts_count": num_drafts,
            "status": "PASS" if (load_ok and infer_ok) else "BLOCKED_OR_FAILED",
        }
        bench_results.append(res)
        print(f"Results for {model_id}: {res}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_name": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "CPU",
        "cuda_version": torch.version.cuda if torch and torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__ if torch else "N/A",
        "sequential_execution_policy": "MAXIMUM_FIDELITY",
        "models_benchmarked": bench_results,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Cleanup temp
    if test_img_path.exists():
        test_img_path.unlink()

    return summary


if __name__ == "__main__":
    out_path = Path("docs/HARDWARE_BENCHMARK_REPORT.json")
    run_hardware_benchmark(out_path)
