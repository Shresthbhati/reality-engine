"""Reproducible A3 Perception Benchmark Runner (A3.11).

Runs the production ``PipelineDetectionEngineAdapter`` + ``build_intelligence_bundle``
on real photographs or a labelled COCO indoor dataset, emitting a structured
``BenchmarkReport`` without conflating unlabelled counts with physical accuracy.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.perception.evaluator import BenchmarkReport, EvaluationMode, run_perception_benchmark
from benchmarks.perception.indoor_dataset_adapter import load_coco_indoor_annotations
from engine.multiengine.adapters import create_adapters_for_registry, pipeline_detection
from engine.multiengine.contract import Device, EngineRequest
from engine.multiengine.registry import maximum_fidelity_registry
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox


def run_benchmark_on_images(
    image_paths: list[str],
    *,
    dataset_name: str = "unlabelled_real_capture",
    mode: EvaluationMode = EvaluationMode.UNLABELLED_REAL,
    coco_json: str | None = None,
    detector_id: str = "grounding_dino",
) -> dict[str, Any]:
    """Execute the production detection pipeline and evaluate via ``run_perception_benchmark``."""
    t0 = time.perf_counter()
    reg = maximum_fidelity_registry()
    adapters = create_adapters_for_registry(reg)
    eid = pipeline_detection.install(reg, adapters, detector_id=detector_id)
    eng = adapters[eid]
    dev = Device("cuda", 0) if torch.cuda.is_available() else Device("cpu")
    eng.load(dev)
    try:
        drafts = eng.infer(EngineRequest(sources=tuple(image_paths), payload={"run_id": f"bench_{dataset_name}"}))
    finally:
        eng.unload()
    wall_s = time.perf_counter() - t0
    peak_vram = round(torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0), 1) if torch.cuda.is_available() else None

    preds_by_img: dict[str, list[DetectionBox]] = {}
    for d in drafts:
        if d.quantity != "bbox" or d.image_region is None:
            continue
        key = Path(d.source_id).name
        preds_by_img.setdefault(key, []).append(DetectionBox(
            box_id=str(d.predicted.get("pipeline_box_id") or f"b_{len(preds_by_img.get(key, []))}"),
            class_name=str(d.predicted.get("label") or "object"),
            box_2d=tuple(float(v) for v in d.image_region),
            confidence=float(d.confidence) if d.confidence is not None else 0.5,
            source_engine=str(d.predicted.get("inner_engine") or detector_id),
            source_image_id=key,
        ))

    gt_by_img = None
    if coco_json:
        _, gt_by_img = load_coco_indoor_annotations(coco_json, dataset_name=dataset_name)

    report: BenchmarkReport = run_perception_benchmark(
        mode=mode,
        dataset_name=dataset_name,
        predictions_by_image=preds_by_img,
        gt_by_image=gt_by_img,
        wall_time_s=wall_s,
        peak_vram_mb=peak_vram,
    )
    return {
        "benchmark_report": report.to_dict(),
        "pipeline_facts": eng.last_facts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Reality Engine A3 Perception Benchmark")
    parser.add_argument("--images-glob", default="datasets/south_building/images/*.JPG")
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--dataset-name", default="south_building")
    parser.add_argument("--coco-json", default=None)
    parser.add_argument("--mode", choices=[m.value for m in EvaluationMode], default=EvaluationMode.UNLABELLED_REAL.value)
    args = parser.parse_args(argv)

    imgs = sorted(glob.glob(args.images_glob))[: args.max_images]
    if not imgs:
        print(json.dumps({"error": f"No images matched {args.images_glob!r}"}))
        return 1
    out = run_benchmark_on_images(
        imgs,
        dataset_name=args.dataset_name,
        mode=EvaluationMode(args.mode),
        coco_json=args.coco_json,
    )
    print(json.dumps(out["benchmark_report"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
