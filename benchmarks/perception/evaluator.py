"""Reproducible Perception Benchmark & Evaluation Framework (A3.11).

Measures:
  - Precision, Recall, F1, Average Precision (AP), IoU
  - Mask quality & duplicate rate
  - Cross-view identity accuracy
  - Classification accuracy & Calibration error (ECE, Brier) when labels permit
  - Runtime, peak VRAM, and failure rate

Strictly separates:
  - ``synthetic``
  - ``unlabelled_real``
  - ``labelled_real``
  - ``physically_measured``
Never reports synthetic or unlabelled detection counts as physical accuracy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from benchmarks.perception.indoor_dataset_adapter import GroundTruthBox
from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    LearnedConfidenceCalibrator,
    compute_box_iou,
)
from engine.multiengine.spatial_intelligence.taxonomy import resolve_taxonomy_class
from perception.intelligence.data_model import CrossViewIdentityHypothesis


class EvaluationMode(str, Enum):
    SYNTHETIC = "synthetic"
    UNLABELLED_REAL = "unlabelled_real"
    LABELLED_REAL = "labelled_real"
    PHYSICALLY_MEASURED = "physically_measured"


@dataclass(frozen=True)
class BenchmarkReport:
    mode: EvaluationMode
    dataset_name: str
    images_evaluated: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_ap: float | None
    mean_iou: float | None
    duplicate_rate: float
    mask_quality_mean: float | None
    identity_accuracy: float | None
    classification_accuracy: float | None
    ece: float | None
    brier_score: float | None
    wall_time_s: float
    peak_vram_mb: float | None
    failure_rate: float
    per_class: dict[str, dict[str, Any]]
    accuracy_claim_allowed: bool
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "dataset_name": self.dataset_name,
            "images_evaluated": self.images_evaluated,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_ap": self.mean_ap,
            "mean_iou": self.mean_iou,
            "duplicate_rate": self.duplicate_rate,
            "mask_quality_mean": self.mask_quality_mean,
            "identity_accuracy": self.identity_accuracy,
            "classification_accuracy": self.classification_accuracy,
            "ece": self.ece,
            "brier_score": self.brier_score,
            "wall_time_s": self.wall_time_s,
            "peak_vram_mb": self.peak_vram_mb,
            "failure_rate": self.failure_rate,
            "per_class": self.per_class,
            "accuracy_claim_allowed": self.accuracy_claim_allowed,
            "caveat": self.caveat,
        }


def _compute_ap(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    if not recalls or not precisions:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return float(ap)


def evaluate_cross_view_identity_accuracy(
    predicted_identities: Sequence[CrossViewIdentityHypothesis],
    gt_groups_by_obs_id: Mapping[str, str],
) -> float | None:
    """Pairwise precision/accuracy of predicted cross-view clusters vs ground-truth physical IDs."""
    if not gt_groups_by_obs_id or not predicted_identities:
        return None
    correct = 0
    total = 0
    for ident in predicted_identities:
        members = [m for m in ident.member_observation_ids if m in gt_groups_by_obs_id]
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                total += 1
                if gt_groups_by_obs_id[members[i]] == gt_groups_by_obs_id[members[j]]:
                    correct += 1
    if total == 0:
        return 1.0 if predicted_identities else None
    return round(correct / float(total), 4)


def run_perception_benchmark(
    *,
    mode: EvaluationMode,
    dataset_name: str,
    predictions_by_image: Mapping[str, Sequence[DetectionBox]],
    gt_by_image: Mapping[str, Sequence[GroundTruthBox]] | None = None,
    mask_quality_scores: Sequence[float] = (),
    predicted_identities: Sequence[CrossViewIdentityHypothesis] = (),
    gt_identity_map: Mapping[str, str] | None = None,
    wall_time_s: float = 0.0,
    peak_vram_mb: float | None = None,
    failed_images: int = 0,
    iou_threshold: float = 0.50,
) -> BenchmarkReport:
    """Evaluate predictions cleanly without conflating unlabelled counts with accuracy."""
    all_images = sorted(set(predictions_by_image.keys()) | set((gt_by_image or {}).keys()))
    n_imgs = len(all_images)
    failure_rate = round(failed_images / max(1, n_imgs), 4)
    mask_mean = round(float(np.mean(mask_quality_scores)), 4) if mask_quality_scores else None

    # Duplicate rate: fraction of per-image same-class predictions overlapping >= 0.65
    total_preds = 0
    dup_preds = 0
    for img_id in all_images:
        plist = list(predictions_by_image.get(img_id, ()))
        total_preds += len(plist)
        for i, a in enumerate(plist):
            for b in plist[i + 1:]:
                if a.class_name == b.class_name and compute_box_iou(a.box_2d, b.box_2d) >= 0.65:
                    dup_preds += 1
                    break
    dup_rate = round(dup_preds / max(1, total_preds), 4)

    if mode is EvaluationMode.UNLABELLED_REAL or not gt_by_image:
        return BenchmarkReport(
            mode=mode,
            dataset_name=dataset_name,
            images_evaluated=n_imgs,
            precision=None,
            recall=None,
            f1=None,
            mean_ap=None,
            mean_iou=None,
            duplicate_rate=dup_rate,
            mask_quality_mean=mask_mean,
            identity_accuracy=None,
            classification_accuracy=None,
            ece=None,
            brier_score=None,
            wall_time_s=round(wall_time_s, 3),
            peak_vram_mb=peak_vram_mb,
            failure_rate=failure_rate,
            per_class={},
            accuracy_claim_allowed=False,
            caveat="Unlabelled dataset: precision/recall/AP/ECE are None; counts are not accuracy.",
        )

    tp_scores: list[tuple[str, float, int]] = []  # (class, score, is_tp)
    gt_counts: dict[str, int] = defaultdict(int)
    matched_ious: list[float] = []
    tp_cls: dict[str, int] = defaultdict(int)
    fp_cls: dict[str, int] = defaultdict(int)
    fn_cls: dict[str, int] = defaultdict(int)
    cal_scores: list[float] = []
    cal_labels: list[int] = []

    for img_id in all_images:
        preds = sorted(predictions_by_image.get(img_id, ()), key=lambda d: d.confidence, reverse=True)
        gts = list(gt_by_image.get(img_id, ()))
        for g in gts:
            gt_counts[g.canonical_label] += 1
        used_gt: set[int] = set()
        for pred in preds:
            p_cls = resolve_taxonomy_class(pred.class_name).name
            best_iou = 0.0
            best_gi: int | None = None
            for gi, gt in enumerate(gts):
                if gi in used_gt or gt.canonical_label != p_cls:
                    continue
                iou = compute_box_iou(pred.box_2d, gt.box_2d)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            is_tp = 1 if best_gi is not None else 0
            if best_gi is not None:
                used_gt.add(best_gi)
                matched_ious.append(best_iou)
                tp_cls[p_cls] += 1
            else:
                fp_cls[p_cls] += 1
            tp_scores.append((p_cls, float(pred.confidence), is_tp))
            cal_scores.append(float(pred.confidence))
            cal_labels.append(is_tp)
        for gi, gt in enumerate(gts):
            if gi not in used_gt:
                fn_cls[gt.canonical_label] += 1

    per_class: dict[str, dict[str, Any]] = {}
    aps: list[float] = []
    for cls in sorted(set(gt_counts.keys()) | set(tp_cls.keys()) | set(fp_cls.keys())):
        tp, fp, fn = tp_cls[cls], fp_cls[cls], fn_cls[cls]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        cls_items = sorted([x for x in tp_scores if x[0] == cls], key=lambda x: x[1], reverse=True)
        n_gt = gt_counts.get(cls, 0)
        ap = 0.0
        if n_gt > 0 and cls_items:
            tp_acc = 0
            fp_acc = 0
            precs, recs = [], []
            for _, _, flag in cls_items:
                if flag:
                    tp_acc += 1
                else:
                    fp_acc += 1
                precs.append(tp_acc / (tp_acc + fp_acc))
                recs.append(tp_acc / n_gt)
            ap = _compute_ap(recs, precs)
            aps.append(ap)
        per_class[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "ap": round(ap, 4),
        }

    tot_tp = sum(tp_cls.values())
    tot_fp = sum(fp_cls.values())
    tot_fn = sum(fn_cls.values())
    p_tot = round(tot_tp / (tot_tp + tot_fp), 4) if (tot_tp + tot_fp) > 0 else 0.0
    r_tot = round(tot_tp / (tot_tp + tot_fn), 4) if (tot_tp + tot_fn) > 0 else 0.0
    f1_tot = round(2 * p_tot * r_tot / (p_tot + r_tot), 4) if (p_tot + r_tot) > 0 else 0.0
    mean_ap = round(float(np.mean(aps)), 4) if aps else 0.0
    mean_iou = round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0
    cal_m = LearnedConfidenceCalibrator.compute_calibration_metrics(cal_scores, cal_labels)
    id_acc = evaluate_cross_view_identity_accuracy(predicted_identities, gt_identity_map or {})
    cls_acc = round(tot_tp / max(1, tot_tp + tot_fp), 4)

    allow_claim = mode in (EvaluationMode.LABELLED_REAL, EvaluationMode.PHYSICALLY_MEASURED)
    caveat = (
        "Synthetic benchmark: do not report as physical-world accuracy."
        if mode is EvaluationMode.SYNTHETIC
        else "Measured against labelled ground truth."
    )
    return BenchmarkReport(
        mode=mode,
        dataset_name=dataset_name,
        images_evaluated=n_imgs,
        precision=p_tot,
        recall=r_tot,
        f1=f1_tot,
        mean_ap=mean_ap,
        mean_iou=mean_iou,
        duplicate_rate=dup_rate,
        mask_quality_mean=mask_mean,
        identity_accuracy=id_acc,
        classification_accuracy=cls_acc,
        ece=round(cal_m.ece, 4),
        brier_score=round(cal_m.brier_score, 4),
        wall_time_s=round(wall_time_s, 3),
        peak_vram_mb=peak_vram_mb,
        failure_rate=failure_rate,
        per_class=per_class,
        accuracy_claim_allowed=allow_claim,
        caveat=caveat,
    )


__all__ = [
    "BenchmarkReport",
    "EvaluationMode",
    "evaluate_cross_view_identity_accuracy",
    "run_perception_benchmark",
]
