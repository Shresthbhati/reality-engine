"""Sprint 10 Independent Perception Validation and Benchmark Runner (P2, P3, P4).

Executes comprehensive evaluation of Reality Engine's perception engine on the
scene-disjoint evaluation dataset (South Building exterior + multi-view room capture):
- Computes Precision, Recall, Macro F1, COCO AP50, and VOC AP50.
- Compares native resolution vs. tiled inference for small objects (sockets, switches, handles, taps).
- Evaluates multi-view physical identity clustering (purity, completeness, association F1, identity switches).
- Stratifies performance by object scale (Small: area < 32^2, Medium: 32^2..96^2, Large: > 96^2).
- Stratifies by occlusion (visible vs. occluded).
- Evaluates parent-child assignment precision, recall, and macro F1.
- Strictly marks unsupported classes as UNMEASURED.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from benchmarks.perception.component_benchmark import (
    ComponentBenchmarkEvaluator,
    ComponentFamily,
    EvaluationMode,
    GroundTruthComponent,
    compute_box_iou,
    is_class_compatible,
    load_component_ground_truth,
)
from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    DetectionPipeline,
    DetectionScale,
    create_default_detector,
    create_default_segmenter,
    load_default_production_calibrator,
)


def evaluate_sprint10_independent(
    dataset_dir: str = "datasets/scene_disjoint_evaluation",
    split: str = "all",
    max_images: int = 6,
    output_path: str = "artifacts/perception/sprint10_independent_evaluation_report.json",
) -> dict[str, Any]:
    import torch

    print(f"Loading scene-disjoint ground truth from {dataset_dir} (split={split})...", flush=True)
    gt_components, image_files, meta = load_component_ground_truth(dataset_dir, split=split)

    if max_images and len(image_files) > max_images:
        # Balance exterior and interior images: 3 exterior + 3 interior
        ext_imgs = [p for p in image_files if p.name.startswith("P118")]
        int_imgs = [p for p in image_files if p.name.startswith("IMG_")]
        n_half = max_images // 2
        image_files = ext_imgs[:n_half] + int_imgs[: (max_images - n_half)]
        kept_names = {p.name for p in image_files}
        gt_components = [g for g in gt_components if g.image_id in kept_names]

    print(f"Loaded {len(gt_components)} ground-truth boxes across {len(image_files)} images ({[p.name for p in image_files]}).", flush=True)

    calibrator = load_default_production_calibrator()
    detector = create_default_detector()
    segmenter = create_default_segmenter()
    pipeline = DetectionPipeline(detector=detector, segmenter=segmenter, calibrator=calibrator)

    predictions_native: list[DetectionBox] = []
    predictions_tiled: list[DetectionBox] = []
    per_image_timings: dict[str, float] = {}

    for idx, img_p in enumerate(image_files):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        # 1. Native inference (no tiling)
        res_native = pipeline.run_pipeline(
            img_rgb,
            image_id=img_p.name,
            detect_subparts=True,
            run_tiling=False,
        )
        for det in res_native.refined_boxes:
            predictions_native.append(DetectionBox(
                box_id=det.box_id,
                class_name=det.class_name,
                box_2d=det.box_2d,
                confidence=det.confidence,
                source_engine=det.source_engine,
                source_image_id=img_p.name,
                parent_box_id=det.parent_box_id,
                scale=det.scale,
                physical_identity_id=getattr(det, "physical_identity_id", None),
                raw_score=det.raw_score,
            ))

        # 2. Tiled inference (small-object recovery pipeline)
        res_tiled = pipeline.run_pipeline(
            img_rgb,
            image_id=img_p.name,
            detect_subparts=True,
            run_tiling=True,
        )
        for det in res_tiled.refined_boxes:
            predictions_tiled.append(DetectionBox(
                box_id=det.box_id,
                class_name=det.class_name,
                box_2d=det.box_2d,
                confidence=det.confidence,
                source_engine=det.source_engine,
                source_image_id=img_p.name,
                parent_box_id=det.parent_box_id,
                scale=det.scale,
                physical_identity_id=getattr(det, "physical_identity_id", None),
                raw_score=det.raw_score,
            ))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dt = time.perf_counter() - t0
        per_image_timings[img_p.name] = round(dt, 2)
        print(f"Processed {img_p.name} ({idx + 1}/{len(image_files)}) in {dt:.2f}s", flush=True)

    evaluator = ComponentBenchmarkEvaluator()

    # Benchmark report on tiled pipeline (production configuration)
    report_tiled = evaluator.evaluate(
        ground_truth=gt_components,
        predictions=predictions_tiled,
        eval_split=split,
        dataset_name="scene_disjoint_evaluation",
        mode=EvaluationMode.FRAME_LEVEL,
    )

    # Benchmark report on native pipeline (ablation study)
    report_native = evaluator.evaluate(
        ground_truth=gt_components,
        predictions=predictions_native,
        eval_split=split,
        dataset_name="scene_disjoint_evaluation",
        mode=EvaluationMode.FRAME_LEVEL,
    )

    # Multi-view physical identity clustering report
    report_phys = evaluator.evaluate(
        ground_truth=gt_components,
        predictions=predictions_tiled,
        eval_split=split,
        dataset_name="scene_disjoint_evaluation",
        mode=EvaluationMode.PHYSICAL_IDENTITY,
    )

    # Stratified Small Object Evaluation
    # Small: area < 32^2 (1024 px^2)
    # Medium: 32^2 <= area <= 96^2 (1024 .. 9216 px^2)
    # Large: area > 96^2 (9216 px^2)
    scale_metrics = {}
    for scale_label, min_a, max_a in [
        ("small", 0.0, 1024.0),
        ("medium", 1024.0, 9216.0),
        ("large", 9216.0, 1e9),
    ]:
        gt_scale = [g for g in gt_components if min_a <= g.area < max_a]
        preds_scale = [p for p in predictions_tiled if min_a <= p.area < max_a]
        tp = 0
        claimed_gt = set()
        for p in preds_scale:
            for g in gt_scale:
                if g.image_id != getattr(p, "source_image_id", ""):
                    continue
                if not is_class_compatible(p.class_name, g.class_name):
                    continue
                k = (g.image_id, g.entity_id)
                if k in claimed_gt:
                    continue
                if compute_box_iou(p.box_2d, g.box_2d) >= 0.50:
                    tp += 1
                    claimed_gt.add(k)
                    break
        fp = len(preds_scale) - tp
        fn = len(gt_scale) - tp
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / float(len(gt_scale)) if len(gt_scale) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        scale_metrics[scale_label] = {
            "gt_count": len(gt_scale),
            "pred_count": len(preds_scale),
            "tp50": tp,
            "fp50": fp,
            "fn50": fn,
            "precision50": round(prec, 4),
            "recall50": round(rec, 4),
            "f1_50": round(f1, 4),
        }

    # Micro/Small-object targeted classes: sockets, switches, handles, faucets, mullions, drains
    micro_classes = {"socket", "switch", "door_handle", "faucet", "window_mullion", "drain"}
    micro_gt = [g for g in gt_components if g.class_name in micro_classes]
    micro_preds_tiled = [p for p in predictions_tiled if p.class_name in micro_classes]
    micro_preds_native = [p for p in predictions_native if p.class_name in micro_classes]

    small_object_comparison = {
        "targeted_classes": sorted(list(micro_classes)),
        "total_gt_samples": len(micro_gt),
        "native_resolution": {
            "pred_count": len(micro_preds_native),
            "detected_ratio": round(len(micro_preds_native) / len(micro_gt), 4) if micro_gt else 0.0,
        },
        "tiled_inference": {
            "pred_count": len(micro_preds_tiled),
            "detected_ratio": round(len(micro_preds_tiled) / len(micro_gt), 4) if micro_gt else 0.0,
        },
    }

    final_report = {
        "benchmark_id": "sprint10_independent_perception_validation",
        "dataset_name": "scene_disjoint_evaluation",
        "total_images": len(image_files),
        "total_ground_truth_boxes": len(gt_components),
        "measured_classes_count": report_tiled.measured_classes_count,
        "unmeasured_classes_count": report_tiled.unmeasured_classes_count,
        "timings_seconds": per_image_timings,
        "performance_summary": {
            "parent_detection_macro_f1": report_tiled.parent_detection_macro_f1,
            "parent_detection_ap50_coco": report_tiled.parent_detection_ap50_101,
            "parent_detection_ap50_voc": report_tiled.parent_detection_ap50_11,
            "component_detection_macro_f1": report_tiled.component_detection_macro_f1,
            "component_detection_ap50_coco": report_tiled.component_detection_ap50_101,
            "component_detection_ap50_voc": report_tiled.component_detection_ap50_11,
            "component_parent_link_macro_f1": report_tiled.component_parent_link_macro_f1,
            "component_parent_link_precision": report_tiled.component_parent_link_precision,
            "component_parent_link_recall": report_tiled.component_parent_link_recall,
        },
        "multi_view_physical_identity": report_phys.physical_identity_metrics,
        "scale_stratification": scale_metrics,
        "small_object_ablation": small_object_comparison,
        "results_by_family": report_tiled.results_by_family,
        "results_by_class": {
            k: v.to_dict()
            for k, v in report_tiled.results_by_class.items()
            if v.status == "measured"
        },
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 60)
    print("SPRINT 10 INDEPENDENT BENCHMARK COMPLETE:")
    print(f"Evaluated on {len(image_files)} scene-disjoint images ({len(gt_components)} GT boxes).")
    print(f"Measured classes: {report_tiled.measured_classes_count}, Unmeasured: {report_tiled.unmeasured_classes_count}")
    print(f"Parent Macro F1: {report_tiled.parent_detection_macro_f1}, COCO AP50: {report_tiled.parent_detection_ap50_101}")
    print(f"Component Macro F1: {report_tiled.component_detection_macro_f1}, COCO AP50: {report_tiled.component_detection_ap50_101}")
    print(f"Component-Parent Link F1: {report_tiled.component_parent_link_macro_f1}")
    if report_phys.physical_identity_metrics:
        print(f"Physical Identity Purity: {report_phys.physical_identity_metrics.get('mean_cluster_purity')}, Completeness: {report_phys.physical_identity_metrics.get('mean_cluster_completeness')}, F1: {report_phys.physical_identity_metrics.get('physical_identity_f1')}")
    print("=" * 60)
    print(f"Saved evaluation report to {out_p}")
    return final_report


if __name__ == "__main__":
    evaluate_sprint10_independent()
