"""Indoor perception dataset adapter & benchmark evaluator (A2.15).

Normalizes COCO-style / SUN RGB-D / ScanNet-style 2D bounding-box annotations into
canonical Reality Engine taxonomy labels, while enforcing dataset provenance honesty
(synthetic fixtures are explicitly flagged ``is_physically_surveyed=False``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    compute_box_iou,
)
from engine.multiengine.spatial_intelligence.taxonomy import resolve_taxonomy_class


@dataclass(frozen=True)
class GroundTruthBox:
    image_id: str
    raw_label: str
    canonical_label: str
    box_2d: tuple[float, float, float, float]
    is_occluded: bool = False
    is_truncated: bool = False


@dataclass(frozen=True)
class IndoorDatasetManifest:
    dataset_name: str
    license: str
    is_physically_surveyed: bool
    samples: tuple[dict[str, Any], ...]


def load_coco_indoor_annotations(
    coco_json_path: Path | str,
    *,
    dataset_name: str = "coco_indoor",
    license_name: str = "unknown",
    is_physically_surveyed: bool = False,
) -> tuple[IndoorDatasetManifest, dict[str, list[GroundTruthBox]]]:
    """Load COCO-format 2D annotations and map labels to canonical taxonomy classes."""
    p = Path(coco_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    cats = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    imgs = {int(im["id"]): im for im in data.get("images", [])}
    gt_by_image: dict[str, list[GroundTruthBox]] = defaultdict(list)

    for ann in data.get("annotations", []):
        img_info = imgs.get(int(ann["image_id"]))
        if not img_info:
            continue
        file_name = str(img_info.get("file_name", ann["image_id"]))
        raw_lbl = cats.get(int(ann["category_id"]), "unknown")
        canon = resolve_taxonomy_class(raw_lbl).name
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0 or h <= 0:
            continue
        gt_by_image[file_name].append(GroundTruthBox(
            image_id=file_name,
            raw_label=raw_lbl,
            canonical_label=canon,
            box_2d=(x, y, x + w, y + h),
        ))

    manifest = IndoorDatasetManifest(
        dataset_name=dataset_name,
        license=license_name,
        is_physically_surveyed=bool(is_physically_surveyed),
        samples=tuple({"image_id": k, "gt_count": len(v)} for k, v in sorted(gt_by_image.items())),
    )
    return manifest, dict(gt_by_image)


def evaluate_detections_against_ground_truth(
    predictions_by_image: dict[str, Sequence[DetectionBox]],
    gt_by_image: dict[str, Sequence[GroundTruthBox]],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute per-class precision, recall, and F1 against labelled ground truth."""
    tp_cls: dict[str, int] = defaultdict(int)
    fp_cls: dict[str, int] = defaultdict(int)
    fn_cls: dict[str, int] = defaultdict(int)

    all_images = sorted(set(predictions_by_image.keys()) | set(gt_by_image.keys()))
    for img_id in all_images:
        preds = sorted(predictions_by_image.get(img_id, ()), key=lambda d: d.confidence, reverse=True)
        gts = list(gt_by_image.get(img_id, ()))
        matched_gt: set[int] = set()

        for pred in preds:
            p_canon = resolve_taxonomy_class(pred.class_name).name
            best_iou = 0.0
            best_idx: int | None = None
            for gi, gt in enumerate(gts):
                if gi in matched_gt:
                    continue
                if gt.canonical_label != p_canon:
                    continue
                iou = compute_box_iou(pred.box_2d, gt.box_2d)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_idx is not None:
                matched_gt.add(best_idx)
                tp_cls[p_canon] += 1
            else:
                fp_cls[p_canon] += 1

        for gi, gt in enumerate(gts):
            if gi not in matched_gt:
                fn_cls[gt.canonical_label] += 1

    per_class: dict[str, dict[str, Any]] = {}
    all_classes = sorted(set(tp_cls.keys()) | set(fp_cls.keys()) | set(fn_cls.keys()))
    for cls in all_classes:
        tp, fp, fn = tp_cls[cls], fp_cls[cls], fn_cls[cls]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    total_tp = sum(tp_cls.values())
    total_fp = sum(fp_cls.values())
    total_fn = sum(fn_cls.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    return {
        "iou_threshold": iou_threshold,
        "images_evaluated": len(all_images),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "per_class": per_class,
    }


__all__ = [
    "GroundTruthBox",
    "IndoorDatasetManifest",
    "evaluate_detections_against_ground_truth",
    "load_coco_indoor_annotations",
]
