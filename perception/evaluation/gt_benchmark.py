"""Ground-Truth COCO + Multi-View Identity + Scene-Graph Benchmark Evaluator.

Computes honest, uninflated quantitative metrics against annotated ground truth:
- Detection: AP@0.50, AP@0.50:0.95 (10 IoU thresholds), per-class AP, small-object recall@0.50.
- Segmentation: Mean matched Mask IoU.
- Multi-View Identity: Pairwise same-instance precision, recall, F1, and ID switches.
- Scene Graph: Directed relationship precision, recall, F1.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GTBoxAnnotation:
    image_id: str
    category: str
    bbox_xyxy: tuple[float, float, float, float]
    track_id: str | None = None
    mask: np.ndarray | None = None
    is_small: bool = False


@dataclass(frozen=True)
class PredBoxAnnotation:
    image_id: str
    category: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    track_id: str | None = None
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class GTRelationship:
    source_track_id: str
    target_track_id: str
    kind: str


@dataclass(frozen=True)
class PredRelationship:
    source_track_id: str
    target_track_id: str
    kind: str
    provenance: str = "observed"


@dataclass(frozen=True)
class DetectionEvalReport:
    ap50: float
    ap50_95: float
    per_class_ap50: dict[str, float]
    per_class_ap50_95: dict[str, float]
    small_object_recall50: float
    mean_mask_iou: float | None
    n_gt: int
    n_pred: int
    n_small_gt: int


@dataclass(frozen=True)
class IdentityEvalReport:
    precision: float
    recall: float
    f1: float
    id_switches: int
    n_gt_pairs: int
    n_pred_pairs: int


@dataclass(frozen=True)
class SceneGraphEvalReport:
    precision: float
    recall: float
    f1: float
    n_gt: int
    n_pred: int


@dataclass(frozen=True)
class FullBenchmarkReport:
    detection: DetectionEvalReport
    identity: IdentityEvalReport
    scene_graph: SceneGraphEvalReport
    metadata: dict[str, Any] = field(default_factory=dict)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    ba = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if ba.shape != bb.shape:
        return 0.0
    inter = int(np.logical_and(ba, bb).sum())
    union = int(np.logical_or(ba, bb).sum())
    return float(inter / union) if union > 0 else 0.0


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point COCO interpolated Average Precision."""
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    recall_levels = np.linspace(0.0, 1.0, 101)
    ap = float(np.mean([np.max(mpre[mrec >= r]) if np.any(mrec >= r) else 0.0 for r in recall_levels]))
    return ap


def _eval_class_at_iou(
    gt_boxes: Sequence[GTBoxAnnotation],
    pred_boxes: Sequence[PredBoxAnnotation],
    iou_thresh: float,
) -> tuple[float, int, list[float]]:
    if not gt_boxes:
        return (0.0 if pred_boxes else 1.0), 0, []
    if not pred_boxes:
        return 0.0, 0, []

    gt_by_img: dict[str, list[tuple[int, GTBoxAnnotation]]] = defaultdict(list)
    for idx, g in enumerate(gt_boxes):
        gt_by_img[g.image_id].append((idx, g))

    matched_gt: set[int] = set()
    sorted_preds = sorted(pred_boxes, key=lambda p: p.score, reverse=True)
    tp = np.zeros(len(sorted_preds), dtype=np.float64)
    fp = np.zeros(len(sorted_preds), dtype=np.float64)
    mask_ious: list[float] = []

    for i, p in enumerate(sorted_preds):
        best_iou = 0.0
        best_gt_idx = -1
        best_gt_obj: GTBoxAnnotation | None = None
        for g_idx, g in gt_by_img.get(p.image_id, []):
            if g_idx in matched_gt:
                continue
            iou = _box_iou(p.bbox_xyxy, g.bbox_xyxy)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
                best_gt_obj = g
        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            tp[i] = 1.0
            if p.mask is not None and best_gt_obj is not None and best_gt_obj.mask is not None:
                mask_ious.append(_mask_iou(p.mask, best_gt_obj.mask))
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / len(gt_boxes)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return _compute_ap(recalls, precisions), len(matched_gt), mask_ious


def evaluate_detection_coco(
    gt: Sequence[GTBoxAnnotation],
    preds: Sequence[PredBoxAnnotation],
) -> DetectionEvalReport:
    """Compute COCO AP50, AP50:95, per-class AP, small-object recall, and mask IoU."""
    classes = sorted({g.category for g in gt})
    thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    per_class_ap50: dict[str, float] = {}
    per_class_ap50_95: dict[str, float] = {}
    all_mask_ious: list[float] = []

    for cls in classes:
        c_gt = [g for g in gt if g.category == cls]
        c_pred = [p for p in preds if p.category == cls]
        aps: list[float] = []
        for t in thresholds:
            ap, _n_matched, m_ious = _eval_class_at_iou(c_gt, c_pred, t)
            aps.append(ap)
            if t == 0.50:
                per_class_ap50[cls] = round(ap, 4)
                all_mask_ious.extend(m_ious)
        per_class_ap50_95[cls] = round(float(np.mean(aps)), 4)

    ap50 = round(float(np.mean(list(per_class_ap50.values()))) if per_class_ap50 else 0.0, 4)
    ap50_95 = round(float(np.mean(list(per_class_ap50_95.values()))) if per_class_ap50_95 else 0.0, 4)

    # Small-object recall @ IoU=0.50
    small_gts = [g for g in gt if g.is_small or ((g.bbox_xyxy[2] - g.bbox_xyxy[0]) * (g.bbox_xyxy[3] - g.bbox_xyxy[1]) < 32 * 32)]
    small_matched = 0
    for g in small_gts:
        if any(
            p.image_id == g.image_id and p.category == g.category and _box_iou(p.bbox_xyxy, g.bbox_xyxy) >= 0.50
            for p in preds
        ):
            small_matched += 1
    small_recall = round(small_matched / len(small_gts), 4) if small_gts else 1.0

    mean_mask_iou = round(float(np.mean(all_mask_ious)), 4) if all_mask_ious else None

    return DetectionEvalReport(
        ap50=ap50,
        ap50_95=ap50_95,
        per_class_ap50=per_class_ap50,
        per_class_ap50_95=per_class_ap50_95,
        small_object_recall50=small_recall,
        mean_mask_iou=mean_mask_iou,
        n_gt=len(gt),
        n_pred=len(preds),
        n_small_gt=len(small_gts),
    )


def evaluate_multiview_identity(
    gt: Sequence[GTBoxAnnotation],
    preds: Sequence[PredBoxAnnotation],
    iou_thresh: float = 0.50,
) -> IdentityEvalReport:
    """Evaluate cross-view instance identity linking against GT track_ids."""
    # Match each prediction to its best GT box in the same image/category
    gt_to_pred_tracks: dict[str, set[str]] = defaultdict(set)
    matched_preds: list[tuple[str, str]] = []  # (gt_track_id, pred_track_id)

    for g in gt:
        if g.track_id is None:
            continue
        best_p: PredBoxAnnotation | None = None
        best_iou = 0.0
        for p in preds:
            if p.image_id != g.image_id or p.category != g.category or p.track_id is None:
                continue
            iou = _box_iou(p.bbox_xyxy, g.bbox_xyxy)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_p = p
        if best_p is not None and best_p.track_id is not None:
            gt_to_pred_tracks[g.track_id].add(best_p.track_id)
            matched_preds.append((g.track_id, best_p.track_id))

    id_switches = sum(max(0, len(ptids) - 1) for ptids in gt_to_pred_tracks.values())

    # Pairwise same-track evaluation over matched detections
    gt_pairs: set[tuple[int, int]] = set()
    pred_pairs: set[tuple[int, int]] = set()
    n = len(matched_preds)
    for i in range(n):
        for j in range(i + 1, n):
            if matched_preds[i][0] == matched_preds[j][0]:
                gt_pairs.add((i, j))
            if matched_preds[i][1] == matched_preds[j][1]:
                pred_pairs.add((i, j))

    tp = len(gt_pairs & pred_pairs)
    prec = tp / len(pred_pairs) if pred_pairs else (1.0 if not gt_pairs else 0.0)
    rec = tp / len(gt_pairs) if gt_pairs else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return IdentityEvalReport(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        id_switches=id_switches,
        n_gt_pairs=len(gt_pairs),
        n_pred_pairs=len(pred_pairs),
    )


def evaluate_scene_graph(
    gt_rels: Sequence[GTRelationship],
    pred_rels: Sequence[PredRelationship],
) -> SceneGraphEvalReport:
    """Evaluate directed scene-graph relationships against GT."""
    gt_set = {(r.source_track_id, r.target_track_id, r.kind) for r in gt_rels}
    pred_set = {(r.source_track_id, r.target_track_id, r.kind) for r in pred_rels}
    tp = len(gt_set & pred_set)
    prec = tp / len(pred_set) if pred_set else (1.0 if not gt_set else 0.0)
    rec = tp / len(gt_set) if gt_set else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return SceneGraphEvalReport(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        n_gt=len(gt_set),
        n_pred=len(pred_set),
    )


def load_coco_gt_bundle(
    json_path: str | Path,
) -> tuple[list[GTBoxAnnotation], list[GTRelationship]]:
    """Load COCO-format GT extended with track_id, is_small, and relationships."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    imgs = {im["id"]: im["file_name"] for im in data.get("images", [])}

    gt_boxes: list[GTBoxAnnotation] = []
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        gt_boxes.append(GTBoxAnnotation(
            image_id=str(imgs.get(ann["image_id"], ann["image_id"])),
            category=str(cats.get(ann["category_id"], ann["category_id"])),
            bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
            track_id=ann.get("track_id"),
            is_small=bool(ann.get("is_small", False)),
        ))

    gt_rels = [
        GTRelationship(
            source_track_id=str(r["source_track_id"]),
            target_track_id=str(r["target_track_id"]),
            kind=str(r["kind"]),
        )
        for r in data.get("relationships", [])
    ]
    return gt_boxes, gt_rels
