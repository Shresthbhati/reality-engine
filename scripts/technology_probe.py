"""Measure the technologies that are actually runnable here (Master Directive Rule 2): load time, warm latency, peak VRAM.

  python scripts/technology_probe.py IMAGE [--out docs/engineering/technology_probe.json] [--repeats 3]

Only locally cached weights are used (no downloads). A model that cannot load is recorded as such with the error, never
skipped silently. ``accuracy`` is always null here: accuracy needs ground truth (see benchmarks/reality_benchmark). The
numbers describe THIS machine, image size and library versions, and are recorded with them."""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import sys
import time
from typing import Callable, Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(name: str, build: Callable[[], Tuple[Callable[[], object], Callable[[], None]]], repeats: int) -> dict:
    """``build`` returns (run, unload). Latency is the median of ``repeats`` warm runs after one warm-up."""
    import torch

    cuda = torch.cuda.is_available()
    rec: Dict[str, object] = {"name": name, "accuracy": None, "accuracy_note": "no ground truth: not measured"}
    try:
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        run, unload = build()
        run()                                                   # warm-up (includes lazy init)
        _sync()
        rec["load_and_first_run_s"] = round(time.time() - t0, 2)
        times: List[float] = []
        for _ in range(repeats):
            _sync()
            t = time.time()
            run()
            _sync()
            times.append(time.time() - t)
        rec["latency_s_median"] = round(statistics.median(times), 3)
        rec["latency_s_all"] = [round(x, 3) for x in times]
        rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None
        unload()
        rec["status"] = "measured"
    except Exception as exc:  # noqa: BLE001 -- recorded, never hidden
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        try:
            if cuda:
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return rec


def _hf_depth(model_id: str, half: bool, image):
    def build():
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        proc = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True).to("cuda").eval()
        if half:
            model = model.half()

        def run():
            inp = proc(images=image, return_tensors="pt").to("cuda")
            if half:
                inp["pixel_values"] = inp["pixel_values"].half()
            with torch.no_grad():
                return model(**inp).predicted_depth

        return run, lambda: None
    return build


def _midas(model_type: str, image_path: str):
    def build():
        from types import SimpleNamespace

        from evidence.session import EvidenceKind
        from perception.depth.midAS_backend import MiDaSDepthBackend

        be = MiDaSDepthBackend(model_type=model_type)
        item = SimpleNamespace(id="probe", source_uri=pathlib.Path(image_path).as_uri(), metadata={}, kind=EvidenceKind.PHOTO)
        return (lambda: be.estimate_depth([item])), (lambda: None)
    return build


def _maskrcnn(image):
    def build():
        import numpy as np
        import torch
        import torchvision

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").to("cuda").eval()
        x = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float().div(255).to("cuda")

        def run():
            with torch.no_grad():
                return model([x])

        return run, lambda: None
    return build


def _gdino(image_path: str, half: bool, batch: int):
    def build():
        import torch

        import engine.pipeline.open_vocab_furniture as o

        proc, model, dev = o._load()
        model = model.half() if half else model.float()
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        prompts = ["sofa", "chair", "wall socket", "light switch", "ceiling fan", "door handle", "window", "plant"][:batch]

        def run():
            inp = proc(images=[img] * len(prompts), text=[f"{p}." for p in prompts], return_tensors="pt", padding=True).to(dev)
            if half:
                inp["pixel_values"] = inp["pixel_values"].half()
            with torch.no_grad():
                return model(**inp)

        return run, o.unload_open_vocab_furniture
    return build


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default="docs/engineering/technology_probe.json")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)

    import torch
    from PIL import Image

    image = Image.open(args.image).convert("RGB")
    probes = [
        ("Grounding DINO tiny, fp32, 2 prompts/pass", _gdino(args.image, False, 2)),
        ("Grounding DINO tiny, fp16, 2 prompts/pass", _gdino(args.image, True, 2)),
        ("DPT_Hybrid (MiDaS, current default depth)", _midas("DPT_Hybrid", args.image)),
        ("Depth Anything V2 Small, fp16", _hf_depth("depth-anything/Depth-Anything-V2-Small-hf", True, image)),
        ("Depth Anything V2 Metric-Indoor Small, fp16", _hf_depth("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", True, image)),
        ("Mask R-CNN R50-FPN (COCO)", _maskrcnn(image)),
    ]
    results = [measure(n, b, args.repeats) for n, b in probes]
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "image": pathlib.Path(args.image).name, "image_size": list(image.size),
              "machine": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
                          "torch": torch.__version__, "python": platform.python_version(), "os": platform.platform()},
              "results": results}
    out = pathlib.Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for r in results:
        print(f"{r['name']:48s} {r['status']:9s} latency {r.get('latency_s_median')} s  peak VRAM {r.get('peak_vram_gb')} GB  {r.get('error', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
