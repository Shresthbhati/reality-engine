"""Production Perception Evaluation: Per-Class P/R/F1/AP, Localization, OCR & Production Connection."""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from perception.attributes.recognizer import EntityAttributeBundle, extract_observed_attributes
from perception.engine_evidence import ObservationRecord
from perception.evaluation.gt_benchmark import (
    GTBoxAnnotation,
    PredBoxAnnotation,
    _box_iou,
    evaluate_detection_coco,
    evaluate_multiview_identity,
    evaluate_scene_graph,
    load_coco_gt_bundle,
)
from perception.reasoning.scene_graph_builder import (
    SceneGraphReport,
    build_scene_graph_from_observations,
)


@dataclass(frozen=True)
class PerClassMetrics:
    category: str
    precision: float
    recall: float
    f1: float
    ap50: float
    ap50_95: float
    mean_center_err_px: float | None
    n_gt: int
    n_pred: int


@dataclass(frozen=True)
class OCRCandidateResult:
    candidate_id: str
    status: str
    exact_accuracy: float | None
    normalized_accuracy: float | None
    mean_cer: float | None
    n_regions: int
    error_message: str = ""


def _center_error(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = 0.5 * (a[0] + a[2]), 0.5 * (a[1] + a[3])
    bx, by = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
    return float(np.hypot(ax - bx, ay - by))


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def evaluate_per_class_metrics(
    gt: Sequence[GTBoxAnnotation],
    preds: Sequence[PredBoxAnnotation],
    iou_thresh: float = 0.50,
) -> list[PerClassMetrics]:
    """Compute per-class Precision, Recall, F1, AP50, AP50:95, and localization error."""
    coco = evaluate_detection_coco(gt, preds)
    classes = sorted({g.category for g in gt})
    results: list[PerClassMetrics] = []

    for cls in classes:
        c_gt = [g for g in gt if g.category == cls]
        c_pred = [p for p in preds if p.category == cls]
        matched: set[int] = set()
        errs: list[float] = []
        tp = 0
        for p in sorted(c_pred, key=lambda x: x.score, reverse=True):
            best_i = -1
            best_iou = 0.0
            for idx, g in enumerate(c_gt):
                if idx in matched or g.image_id != p.image_id:
                    continue
                iou = _box_iou(p.bbox_xyxy, g.bbox_xyxy)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_i = idx
            if best_i >= 0:
                matched.add(best_i)
                tp += 1
                errs.append(_center_error(p.bbox_xyxy, c_gt[best_i].bbox_xyxy))

        prec = tp / len(c_pred) if c_pred else (1.0 if not c_gt else 0.0)
        rec = tp / len(c_gt) if c_gt else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        results.append(PerClassMetrics(
            category=cls,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            ap50=coco.per_class_ap50.get(cls, 0.0),
            ap50_95=coco.per_class_ap50_95.get(cls, 0.0),
            mean_center_err_px=round(float(np.mean(errs)), 3) if errs else None,
            n_gt=len(c_gt),
            n_pred=len(c_pred),
        ))
    return results


def evaluate_ocr_candidates(
    ocr_regions: Sequence[dict[str, Any]],
    image_lookup: dict[str, np.ndarray],
    reader_fn: Callable[[np.ndarray, tuple[float, float, float, float]], str],
    candidate_id: str = "florence2_ocr",
) -> OCRCandidateResult:
    """Evaluate an OCR candidate on annotated text regions; reports unavailable honestly on failure."""
    if not ocr_regions:
        return OCRCandidateResult(candidate_id, "succeeded", 1.0, 1.0, 0.0, 0)

    exact_hits = 0
    norm_hits = 0
    cers: list[float] = []
    try:
        for reg in ocr_regions:
            img_name = str(reg["image_file"])
            img = image_lookup[img_name]
            box = tuple(float(v) for v in reg["bbox_xyxy"])
            pred_txt = reader_fn(img, box).strip()  # type: ignore[arg-type]
            gt_txt = str(reg["text"]).strip()
            if pred_txt == gt_txt:
                exact_hits += 1
            if " ".join(pred_txt.lower().split()) == " ".join(gt_txt.lower().split()):
                norm_hits += 1
            cers.append(_levenshtein(pred_txt.lower(), gt_txt.lower()) / max(1, len(gt_txt)))
    except Exception as exc:
        return OCRCandidateResult(
            candidate_id=candidate_id,
            status="unavailable",
            exact_accuracy=None,
            normalized_accuracy=None,
            mean_cer=None,
            n_regions=len(ocr_regions),
            error_message=f"{type(exc).__name__}: {exc}",
        )

    n = len(ocr_regions)
    return OCRCandidateResult(
        candidate_id=candidate_id,
        status="succeeded",
        exact_accuracy=round(exact_hits / n, 4),
        normalized_accuracy=round(norm_hits / n, 4),
        mean_cer=round(float(np.mean(cers)), 4),
        n_regions=n,
    )


def connect_attributes_and_scene_graph(
    observations: Sequence[ObservationRecord],
    image_arrays: dict[str, np.ndarray] | None = None,
    depth_maps: dict[str, np.ndarray] | None = None,
) -> tuple[list[EntityAttributeBundle], SceneGraphReport]:
    """Wire production ObservationRecords directly into observed attribute & scene-graph extractors."""
    bundles = [
        extract_observed_attributes(
            obs,
            image_rgb=image_arrays.get(obs.source_image_id) if image_arrays else None,
            depth_map=depth_maps.get(obs.source_image_id) if depth_maps else None,
        )
        for obs in observations
    ]
    sg = build_scene_graph_from_observations(observations, depth_maps=depth_maps)
    return bundles, sg


def load_expanded_gt_with_ocr(
    json_path: str | Path,
) -> tuple[list[GTBoxAnnotation], Any, list[dict[str, Any]]]:
    gt_boxes, gt_rels = load_coco_gt_bundle(json_path)
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    imgs = {im["id"]: im["file_name"] for im in data.get("images", [])}
    ocr_regions = [
        {**r, "image_file": imgs.get(r["image_id"], str(r["image_id"]))}
        for r in data.get("ocr_regions", [])
    ]
    return gt_boxes, gt_rels, ocr_regions


__all__ = [
    "OCRCandidateResult",
    "PerClassMetrics",
    "connect_attributes_and_scene_graph",
    "evaluate_multiview_identity",
    "evaluate_ocr_candidates",
    "evaluate_per_class_metrics",
    "evaluate_scene_graph",
    "load_expanded_gt_with_ocr",
]
