"""Real-Data Perception, Part Hierarchy, Temporal Identity, and Reconstruction Benchmark.

Evaluates:
1. Object Detection and Fine-Grained Part Classification (Precision, Recall, F1 per class and macro/micro).
2. Part Hierarchy Linking Accuracy (parent-child part relationship correctness).
3. Instance Mask Quality (Mask IoU and Boundary IoU).
4. Temporal Tracking Performance (IDF1, MOTA, ID Switches, Track Fragmentation).
5. Multi-View Association Integrity (Association Precision, Recall, and 2D/3D Reprojection Residuals).
6. Hardware and Execution Telemetry (Runtime latency, peak RSS memory, and peak CUDA VRAM).

Enforces Zero Fabrication:
- Ground truth is strictly independent of detector predictions.
- No synthetic scale is reported as physical metres.
- Ambiguous associations and rejections are preserved and reported.
"""

from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



@dataclass(frozen=True)
class GroundTruthBox:
    """Labelled 2D bounding box with class, part parentage, and physical entity ground truth."""

    entity_id: str
    class_name: str
    box_2d: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    image_id: str
    parent_entity_id: str | None = None
    frame_index: int | None = None
    mask_2d: np.ndarray | None = None
    visibility: str = "visible"
    occlusion_level: str = "none"


@dataclass(frozen=True)
class DetectionMetrics:
    """Quantitative detection performance metrics at specified IoU threshold."""

    iou_threshold: float
    per_class_precision: dict[str, float | None]
    per_class_recall: dict[str, float | None]
    per_class_f1: dict[str, float | None]
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    micro_precision: float | None
    micro_recall: float | None
    micro_f1: float | None
    total_gt: int
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    average_precision: float | None = None  # Ranking-based 101-point interpolated PR AP
    per_class_ap: dict[str, float | None] = field(default_factory=dict)
    scale_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_calibration_error: float | None = None
    brier_score: float | None = None
    status: str = "measured"  # "measured" or "unmeasured_no_gt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "status": self.status,
            "macro_precision": round(self.macro_precision, 4) if self.macro_precision is not None else None,
            "macro_recall": round(self.macro_recall, 4) if self.macro_recall is not None else None,
            "macro_f1": round(self.macro_f1, 4) if self.macro_f1 is not None else None,
            "micro_precision": round(self.micro_precision, 4) if self.micro_precision is not None else None,
            "micro_recall": round(self.micro_recall, 4) if self.micro_recall is not None else None,
            "micro_f1": round(self.micro_f1, 4) if self.micro_f1 is not None else None,
            "average_precision": round(self.average_precision, 4) if self.average_precision is not None else None,
            "expected_calibration_error": round(self.expected_calibration_error, 4) if self.expected_calibration_error is not None else None,
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "per_class_f1": {k: round(v, 4) if v is not None else None for k, v in self.per_class_f1.items()},
            "per_class_ap": {k: round(v, 4) if v is not None else None for k, v in self.per_class_ap.items()},
            "scale_breakdown": self.scale_breakdown,
        }


@dataclass(frozen=True)
class PartHierarchyMetrics:
    """Evaluation of hierarchical part-parent link precision and recall."""

    total_gt_part_links: int
    predicted_part_links: int
    correct_part_links: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_parent_assignments: int = 0
    missed_components: int = 0
    part_detection_precision: float | None = None
    part_detection_recall: float | None = None
    parent_assignment_precision: float | None = None
    parent_assignment_recall: float | None = None
    incorrect_parent_assignments: int = 0
    duplicate_component_detections: int = 0
    status: str = "measured"  # "measured" or "unmeasured_no_gt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_gt_part_links": self.total_gt_part_links,
            "predicted_part_links": self.predicted_part_links,
            "correct_part_links": self.correct_part_links,
            "false_parent_assignments": self.false_parent_assignments,
            "missed_components": self.missed_components,
            "incorrect_parent_assignments": self.incorrect_parent_assignments,
            "duplicate_component_detections": self.duplicate_component_detections,
            "part_detection_precision": round(self.part_detection_precision, 4) if self.part_detection_precision is not None else None,
            "part_detection_recall": round(self.part_detection_recall, 4) if self.part_detection_recall is not None else None,
            "parent_assignment_precision": round(self.parent_assignment_precision, 4) if self.parent_assignment_precision is not None else None,
            "parent_assignment_recall": round(self.parent_assignment_recall, 4) if self.parent_assignment_recall is not None else None,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1": round(self.f1, 4) if self.f1 is not None else None,
            "status": self.status,
        }


@dataclass(frozen=True)
class MaskQualityMetrics:
    """Instance mask overlap and boundary quality metrics."""

    evaluated_masks: int
    mean_mask_iou: float | None
    median_mask_iou: float | None
    mean_boundary_iou: float | None
    mean_boundary_f_score: float | None = None
    masks_above_50_iou: float | None = None
    masks_above_75_iou: float | None = None
    total_gt_masks: int = 0
    status: str = "measured"  # "measured", "unmeasured_no_gt", or "no_matches"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_gt_masks": self.total_gt_masks,
            "evaluated_masks": self.evaluated_masks,
            "mean_mask_iou": round(self.mean_mask_iou, 4) if self.mean_mask_iou is not None else None,
            "median_mask_iou": round(self.median_mask_iou, 4) if self.median_mask_iou is not None else None,
            "mean_boundary_iou": round(self.mean_boundary_iou, 4) if self.mean_boundary_iou is not None else None,
            "mean_boundary_f_score": round(self.mean_boundary_f_score, 4) if self.mean_boundary_f_score is not None else None,
            "masks_above_50_iou_fraction": round(self.masks_above_50_iou, 4) if self.masks_above_50_iou is not None else None,
            "masks_above_75_iou_fraction": round(self.masks_above_75_iou, 4) if self.masks_above_75_iou is not None else None,
            "status": self.status,
        }


@dataclass(frozen=True)
class TrackingMetrics:
    """Multi-object temporal tracking metrics."""

    total_gt_trajectories: int
    total_predicted_tracks: int
    id_switches: int
    track_fragmentations: int
    idf1: float | None
    id_precision: float | None
    id_recall: float | None
    mota: float | None
    iou_threshold: float = 0.50
    total_gt_instances: int = 0
    total_predicted_instances: int = 0
    frame_tp: int = 0
    frame_fn: int = 0
    frame_fp: int = 0
    id_tp: int = 0
    hota: float | None = None
    deta: float | None = None
    assa: float | None = None
    track_coverage: float | None = None
    status: str = "measured"  # "measured" or "unmeasured_no_gt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "total_gt_trajectories": self.total_gt_trajectories,
            "total_gt_instances": self.total_gt_instances,
            "total_predicted_tracks": self.total_predicted_tracks,
            "total_predicted_instances": self.total_predicted_instances,
            "frame_tp": self.frame_tp,
            "frame_fn": self.frame_fn,
            "frame_fp": self.frame_fp,
            "id_tp": self.id_tp,
            "id_switches": self.id_switches,
            "track_fragmentations": self.track_fragmentations,
            "idf1": round(self.idf1, 4) if self.idf1 is not None else None,
            "id_precision": round(self.id_precision, 4) if self.id_precision is not None else None,
            "id_recall": round(self.id_recall, 4) if self.id_recall is not None else None,
            "mota": round(self.mota, 4) if self.mota is not None else None,
            "hota": round(self.hota, 4) if self.hota is not None else None,
            "deta": round(self.deta, 4) if self.deta is not None else None,
            "assa": round(self.assa, 4) if self.assa is not None else None,
            "track_coverage": round(self.track_coverage, 4) if self.track_coverage is not None else None,
            "status": self.status,
        }


@dataclass(frozen=True)
class MultiViewAssociationMetrics:
    """Multi-view observation association and reprojection residual metrics."""

    candidate_pairs_evaluated: int
    merged_pairs: int
    refused_pairs: int
    association_precision: float | None
    association_recall: float | None
    inlier_mean_reprojection_residual_px: float | None = None
    inlier_median_reprojection_residual_px: float | None = None
    inlier_p90_reprojection_residual_px: float | None = None
    inlier_p25_reprojection_residual_px: float | None = None
    inlier_p75_reprojection_residual_px: float | None = None
    inlier_p95_reprojection_residual_px: float | None = None
    all_pairs_mean_residual_px: float | None = None
    total_gt_pairs: int = 0
    cluster_purity: float | None = None
    identity_completeness: float | None = None
    false_merge_rate: float | None = None
    false_split_rate: float | None = None
    single_view_entity_rate: float | None = None
    multi_view_entity_rate: float | None = None
    results_by_class: dict[str, dict[str, Any]] | None = None
    results_by_view_count: dict[str, dict[str, Any]] | None = None
    status: str = "measured"  # "measured" or "unmeasured_no_gt"

    @property
    def mean_reprojection_residual_px(self) -> float:
        return self.inlier_mean_reprojection_residual_px if self.inlier_mean_reprojection_residual_px is not None else 0.0

    @property
    def median_reprojection_residual_px(self) -> float:
        return self.inlier_median_reprojection_residual_px if self.inlier_median_reprojection_residual_px is not None else 0.0

    @property
    def p90_reprojection_residual_px(self) -> float:
        return self.inlier_p90_reprojection_residual_px if self.inlier_p90_reprojection_residual_px is not None else 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "candidate_pairs_evaluated": self.candidate_pairs_evaluated,
            "merged_pairs": self.merged_pairs,
            "refused_pairs": self.refused_pairs,
            "total_gt_pairs": self.total_gt_pairs,
            "association_precision": round(self.association_precision, 4) if self.association_precision is not None else None,
            "association_recall": round(self.association_recall, 4) if self.association_recall is not None else None,
            "false_merge_rate": round(self.false_merge_rate, 4) if self.false_merge_rate is not None else None,
            "false_split_rate": round(self.false_split_rate, 4) if self.false_split_rate is not None else None,
            "cluster_purity": round(self.cluster_purity, 4) if self.cluster_purity is not None else None,
            "identity_completeness": round(self.identity_completeness, 4) if self.identity_completeness is not None else None,
            "single_view_entity_rate": round(self.single_view_entity_rate, 4) if self.single_view_entity_rate is not None else None,
            "multi_view_entity_rate": round(self.multi_view_entity_rate, 4) if self.multi_view_entity_rate is not None else None,
            "inlier_mean_reprojection_residual_px": round(self.inlier_mean_reprojection_residual_px, 2) if self.inlier_mean_reprojection_residual_px is not None else None,
            "inlier_median_reprojection_residual_px": round(self.inlier_median_reprojection_residual_px, 2) if self.inlier_median_reprojection_residual_px is not None else None,
            "inlier_p25_reprojection_residual_px": round(self.inlier_p25_reprojection_residual_px, 2) if self.inlier_p25_reprojection_residual_px is not None else None,
            "inlier_p75_reprojection_residual_px": round(self.inlier_p75_reprojection_residual_px, 2) if self.inlier_p75_reprojection_residual_px is not None else None,
            "inlier_p90_reprojection_residual_px": round(self.inlier_p90_reprojection_residual_px, 2) if self.inlier_p90_reprojection_residual_px is not None else None,
            "inlier_p95_reprojection_residual_px": round(self.inlier_p95_reprojection_residual_px, 2) if self.inlier_p95_reprojection_residual_px is not None else None,
            "all_pairs_mean_residual_px": round(self.all_pairs_mean_residual_px, 2) if self.all_pairs_mean_residual_px is not None else None,
            "mean_reprojection_residual_px": round(self.mean_reprojection_residual_px, 2),
            "median_reprojection_residual_px": round(self.median_reprojection_residual_px, 2),
            "p90_reprojection_residual_px": round(self.p90_reprojection_residual_px, 2),
            "status": self.status,
        }
        if self.results_by_class is not None:
            d["results_by_class"] = self.results_by_class
        if self.results_by_view_count is not None:
            d["results_by_view_count"] = self.results_by_view_count
        return d


@dataclass(frozen=True)
class BenchmarkExecutionSummary:
    """Complete perception benchmark evaluation summary."""

    benchmark_id: str
    detection_iou50: DetectionMetrics
    detection_iou75: DetectionMetrics
    part_hierarchy: PartHierarchyMetrics
    mask_quality: MaskQualityMetrics
    tracking: TrackingMetrics
    multi_view: MultiViewAssociationMetrics
    telemetry: dict[str, Any]
    split: str = "all"

    @property
    def detection_mAP50(self) -> DetectionMetrics:
        """Compatibility property mapping to detection metrics at IoU 0.50."""
        return self.detection_iou50

    @property
    def detection_mAP75(self) -> DetectionMetrics:
        """Compatibility property mapping to detection metrics at IoU 0.75."""
        return self.detection_iou75

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "split": self.split,
            "detection_iou50": self.detection_iou50.to_dict(),
            "detection_iou75": self.detection_iou75.to_dict(),
            "detection_mAP50": self.detection_iou50.to_dict(),
            "detection_mAP75": self.detection_iou75.to_dict(),
            "part_hierarchy": self.part_hierarchy.to_dict(),
            "mask_quality": self.mask_quality.to_dict(),
            "tracking": self.tracking.to_dict(),
            "multi_view": self.multi_view.to_dict(),
            "telemetry": self.telemetry,
        }


# ============================================================================
# Evaluator Implementations
# ============================================================================

def compute_box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-Union between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def compute_boundary_iou(mask_a: np.ndarray, mask_b: np.ndarray, dilation_radius: int | None = None) -> float:
    """Evaluates Boundary IoU between two boolean segmentation masks using Euclidean Distance Transform."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    from scipy.ndimage import distance_transform_edt

    if dilation_radius is None:
        # Standard boundary thickness is ~1.5% of min dimension (Cheng et al., CVPR 2021)
        dilation_radius = max(2, int(round(0.015 * min(mask_a.shape[:2]))))

    rows_a, cols_a = np.where(mask_a)
    rows_b, cols_b = np.where(mask_b)
    if len(rows_a) == 0 and len(rows_b) == 0:
        return 1.0
    if len(rows_a) == 0 or len(rows_b) == 0:
        return 0.0

    r1 = max(0, min(rows_a.min(), rows_b.min()) - dilation_radius - 2)
    r2 = min(mask_a.shape[0], max(rows_a.max(), rows_b.max()) + dilation_radius + 3)
    c1 = max(0, min(cols_a.min(), cols_b.min()) - dilation_radius - 2)
    c2 = min(mask_a.shape[1], max(cols_a.max(), cols_b.max()) + dilation_radius + 3)

    crop_a = mask_a[r1:r2, c1:c2].astype(bool)
    crop_b = mask_b[r1:r2, c1:c2].astype(bool)

    d_bg_a = distance_transform_edt(crop_a)
    d_fg_a = distance_transform_edt(~crop_a)
    bound_a = ((d_bg_a <= dilation_radius) & (d_bg_a > 0)) | ((d_fg_a <= dilation_radius) & (d_fg_a > 0))

    d_bg_b = distance_transform_edt(crop_b)
    d_fg_b = distance_transform_edt(~crop_b)
    bound_b = ((d_bg_b <= dilation_radius) & (d_bg_b > 0)) | ((d_fg_b <= dilation_radius) & (d_fg_b > 0))

    inter = np.logical_and(bound_a, bound_b).sum()
    union = np.logical_or(bound_a, bound_b).sum()
    return float(inter / union) if union > 0 else 1.0 if not bound_a.any() and not bound_b.any() else 0.0


def compute_boundary_f_score(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    distance_threshold: float | None = None,
) -> float:
    """Computes standard Boundary F-score (BF-score) between two binary instance masks."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    if not np.any(mask_a) and not np.any(mask_b):
        return 1.0
    if not np.any(mask_a) or not np.any(mask_b):
        return 0.0
    from scipy.ndimage import distance_transform_edt, binary_erosion

    if distance_threshold is None:
        distance_threshold = max(2.0, 0.015 * min(mask_a.shape[:2]))

    # Exact 1-pixel boundary extraction via 4-connected morphological erosion
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    m_a = mask_a.astype(bool)
    m_b = mask_b.astype(bool)
    bound_a = m_a & (~binary_erosion(m_a, structure=struct))
    bound_b = m_b & (~binary_erosion(m_b, structure=struct))

    n_a = int(bound_a.sum())
    n_b = int(bound_b.sum())
    if n_a == 0 and n_b == 0:
        return 1.0
    if n_a == 0 or n_b == 0:
        return 0.0

    rows_a, cols_a = np.where(m_a)
    rows_b, cols_b = np.where(m_b)
    pad = int(math.ceil(distance_threshold)) + 4
    r1 = max(0, min(rows_a.min(), rows_b.min()) - pad)
    r2 = min(mask_a.shape[0], max(rows_a.max(), rows_b.max()) + pad)
    c1 = max(0, min(cols_a.min(), cols_b.min()) - pad)
    c2 = min(mask_a.shape[1], max(cols_a.max(), cols_b.max()) + pad)

    bound_a_crop = bound_a[r1:r2, c1:c2]
    bound_b_crop = bound_b[r1:r2, c1:c2]

    dist_to_b = distance_transform_edt(~bound_b_crop)
    dist_to_a = distance_transform_edt(~bound_a_crop)

    prec = float(np.sum(dist_to_b[bound_a_crop] <= distance_threshold) / n_a)
    rec = float(np.sum(dist_to_a[bound_b_crop] <= distance_threshold) / n_b)
    return float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0


def compute_class_average_precision(
    gt_boxes_cls: Sequence[GroundTruthBox],
    pred_boxes_cls: Sequence[Any],
    iou_threshold: float = 0.50,
) -> float | None:
    """Computes standard 101-point interpolated Average Precision for a single class."""
    total_gt = len(gt_boxes_cls)
    if total_gt == 0:
        return None
    if not pred_boxes_cls:
        return 0.0

    # Sort predictions by confidence descending
    sorted_preds = sorted(pred_boxes_cls, key=lambda p: float(getattr(p, "confidence", 0.0)), reverse=True)

    gt_by_img: dict[str, list[GroundTruthBox]] = defaultdict(list)
    for g in gt_boxes_cls:
        gt_by_img[g.image_id].append(g)

    matched_gt: dict[str, set[int]] = defaultdict(set)
    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for p in sorted_preds:
        img_id = getattr(p, "source_image_id", getattr(p, "image_id", ""))
        p_box = tuple(getattr(p, "box_2d", (0, 0, 0, 0)))
        img_gts = gt_by_img.get(img_id, [])

        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g in enumerate(img_gts):
            if g_idx in matched_gt[img_id]:
                continue
            iou = compute_box_iou(p_box, g.box_2d)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_gt_idx >= 0:
            matched_gt[img_id].add(best_gt_idx)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum(fp_flags)

    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / (cum_tp + cum_fp)

    # Standard 101-point COCO-style interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated_precisions: list[float] = []
    for r_thresh in recall_thresholds:
        prec_at_r = precisions[recalls >= r_thresh]
        if len(prec_at_r) > 0:
            interpolated_precisions.append(float(np.max(prec_at_r)))
        else:
            interpolated_precisions.append(0.0)

    return float(np.mean(interpolated_precisions))


def evaluate_detections(
    gt_boxes: Sequence[GroundTruthBox],
    pred_boxes: Sequence[Any],
    iou_threshold: float = 0.50,
) -> DetectionMetrics:
    """Evaluates detection Precision, Recall, and F1 at specified IoU threshold with PR-ranking AP."""
    from engine.multiengine.spatial_intelligence.taxonomy import are_semantically_compatible

    total_gt = len(gt_boxes)
    total_preds = len(pred_boxes)

    gt_by_img: dict[str, list[GroundTruthBox]] = defaultdict(list)
    for g in gt_boxes:
        gt_by_img[g.image_id].append(g)

    pred_by_img: dict[str, list[Any]] = defaultdict(list)
    for p in pred_boxes:
        img_id = getattr(p, "source_image_id", getattr(p, "image_id", "img_001"))
        pred_by_img[img_id].append(p)

    all_images = set(gt_by_img.keys()) | set(pred_by_img.keys())

    tp_per_class: dict[str, int] = defaultdict(int)
    fp_per_class: dict[str, int] = defaultdict(int)
    fn_per_class: dict[str, int] = defaultdict(int)
    all_classes: set[str] = {g.class_name for g in gt_boxes} | {
        getattr(p, "class_name", "") for p in pred_boxes if getattr(p, "class_name", "")
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0

    pred_confidences: list[float] = []
    pred_matches: list[int] = []
    scale_tp: dict[str, int] = defaultdict(int)
    scale_fp: dict[str, int] = defaultdict(int)

    for img_id in all_images:
        gts = list(gt_by_img.get(img_id, []))
        preds = list(pred_by_img.get(img_id, []))
        preds.sort(key=lambda x: getattr(x, "confidence", 0.5), reverse=True)

        matched_gt: set[int] = set()

        for p in preds:
            p_box = tuple(getattr(p, "box_2d", (0, 0, 0, 0)))
            p_cls = getattr(p, "class_name", "")
            p_conf = float(getattr(p, "calibrated_confidence", None) if getattr(p, "calibrated_confidence", None) is not None else getattr(p, "confidence", 0.5))
            p_area = max(0.0, p_box[2] - p_box[0]) * max(0.0, p_box[3] - p_box[1])
            s_bin = "small" if p_area < 1024.0 else ("medium" if p_area < 9216.0 else "large")

            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g in enumerate(gts):
                if g_idx in matched_gt:
                    continue
                if not are_semantically_compatible(p_cls, g.class_name):
                    continue
                iou = compute_box_iou(p_box, g.box_2d)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                tp_per_class[p_cls] += 1
                total_tp += 1
                scale_tp[s_bin] += 1
                pred_confidences.append(p_conf)
                pred_matches.append(1)
            else:
                fp_per_class[p_cls] += 1
                total_fp += 1
                scale_fp[s_bin] += 1
                pred_confidences.append(p_conf)
                pred_matches.append(0)

        for g_idx, g in enumerate(gts):
            if g_idx not in matched_gt:
                fn_per_class[g.class_name] += 1
                total_fn += 1

    per_class_p: dict[str, float | None] = {}
    per_class_r: dict[str, float | None] = {}
    per_class_f1: dict[str, float | None] = {}
    per_class_ap: dict[str, float | None] = {}

    for cls in all_classes:
        tp = tp_per_class[cls]
        fp = fp_per_class[cls]
        fn = fn_per_class[cls]
        cls_gt_count = sum(1 for g in gt_boxes if g.class_name == cls)
        cls_preds = [p for p in pred_boxes if getattr(p, "class_name", "") == cls]

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else (None if cls_gt_count == 0 else 0.0)
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else (None if cls_gt_count == 0 else 0.0)
        f1 = (
            float(2 * p * r / (p + r))
            if (p is not None and r is not None and (p + r) > 0)
            else (None if cls_gt_count == 0 else 0.0)
        )
        per_class_p[cls] = p
        per_class_r[cls] = r
        per_class_f1[cls] = f1
        per_class_ap[cls] = compute_class_average_precision(
            [g for g in gt_boxes if g.class_name == cls],
            cls_preds,
            iou_threshold=iou_threshold,
        )

    if total_gt == 0:
        macro_p = 0.0 if total_preds > 0 else None
        macro_r = None
        macro_f1 = None
        micro_p = 0.0 if total_preds > 0 else None
        micro_r = None
        micro_f1 = None
        overall_ap = None
        status = "unmeasured_no_gt"
    else:
        valid_ps = [v for v in per_class_p.values() if v is not None]
        valid_rs = [v for v in per_class_r.values() if v is not None]
        valid_f1s = [v for v in per_class_f1.values() if v is not None]
        macro_p = float(np.mean(valid_ps)) if valid_ps else 0.0
        macro_r = float(np.mean(valid_rs)) if valid_rs else 0.0
        macro_f1 = float(np.mean(valid_f1s)) if valid_f1s else 0.0

        micro_p = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        micro_r = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

        valid_aps = [v for v in per_class_ap.values() if v is not None]
        overall_ap = float(np.mean(valid_aps)) if valid_aps else 0.0
        status = "measured"

    # Scale breakdown
    scale_breakdown: dict[str, dict[str, Any]] = {}
    gt_scale_counts: dict[str, int] = defaultdict(int)
    for g in gt_boxes:
        g_area = max(0.0, g.box_2d[2] - g.box_2d[0]) * max(0.0, g.box_2d[3] - g.box_2d[1])
        bin_name = "small" if g_area < 1024.0 else ("medium" if g_area < 9216.0 else "large")
        gt_scale_counts[bin_name] += 1

    for bin_name in ["small", "medium", "large"]:
        s_tp = scale_tp[bin_name]
        s_fp = scale_fp[bin_name]
        s_gt = gt_scale_counts[bin_name]
        s_fn = max(0, s_gt - s_tp)
        s_p = float(s_tp / (s_tp + s_fp)) if (s_tp + s_fp) > 0 else None
        s_r = float(s_tp / s_gt) if s_gt > 0 else None
        s_f1 = float(2 * s_p * s_r / (s_p + s_r)) if (s_p is not None and s_r is not None and (s_p + s_r) > 0) else None
        scale_breakdown[bin_name] = {
            "gt_count": s_gt,
            "true_positives": s_tp,
            "false_positives": s_fp,
            "false_negatives": s_fn,
            "precision": round(s_p, 4) if s_p is not None else None,
            "recall": round(s_r, 4) if s_r is not None else None,
            "f1": round(s_f1, 4) if s_f1 is not None else None,
        }

    # Expected Calibration Error (ECE) and Brier Score
    brier_val = None
    ece_val = None
    if pred_confidences:
        confs_arr = np.array(pred_confidences, dtype=float)
        matches_arr = np.array(pred_matches, dtype=float)
        brier_val = float(np.mean((confs_arr - matches_arr) ** 2))

        # 10 equal bins from 0.0 to 1.0
        bin_edges = np.linspace(0.0, 1.0, 11)
        total_n = len(pred_confidences)
        ece_sum = 0.0
        for b_idx in range(10):
            b_low, b_high = bin_edges[b_idx], bin_edges[b_idx + 1]
            in_bin = (confs_arr >= b_low) & (confs_arr < b_high) if b_idx < 9 else (confs_arr >= b_low) & (confs_arr <= b_high)
            n_b = int(np.sum(in_bin))
            if n_b > 0:
                avg_conf = float(np.mean(confs_arr[in_bin]))
                avg_acc = float(np.mean(matches_arr[in_bin]))
                ece_sum += (n_b / total_n) * abs(avg_acc - avg_conf)
        ece_val = float(ece_sum)

    return DetectionMetrics(
        iou_threshold=iou_threshold,
        per_class_precision=per_class_p,
        per_class_recall=per_class_r,
        per_class_f1=per_class_f1,
        per_class_ap=per_class_ap,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        average_precision=overall_ap,
        total_gt=total_gt,
        total_predictions=total_preds,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        scale_breakdown=scale_breakdown,
        expected_calibration_error=ece_val,
        brier_score=brier_val,
        status=status,
    )


def evaluate_part_hierarchy(
    gt_boxes: Sequence[GroundTruthBox],
    pred_boxes: Sequence[Any],
) -> PartHierarchyMetrics:
    """Evaluates parent-child link accuracy between detected subparts and host objects."""
    gt_links: dict[tuple[str, str], str] = {}
    for g in gt_boxes:
        if g.parent_entity_id:
            gt_links[(g.image_id, g.entity_id)] = g.parent_entity_id

    pred_by_id = {getattr(p, "box_id", getattr(p, "observation_id", "")): p for p in pred_boxes}
    pred_links: dict[str, str] = {}
    for p in pred_boxes:
        p_id = getattr(p, "box_id", getattr(p, "observation_id", ""))
        parent_id = getattr(p, "parent_box_id", None)
        if parent_id and parent_id in pred_by_id:
            pred_links[p_id] = parent_id

    from engine.multiengine.spatial_intelligence.taxonomy import are_semantically_compatible

    pred_to_gt_entity: dict[str, str] = {}
    for p_id, p in pred_by_id.items():
        p_box = getattr(p, "box_2d", (0, 0, 0, 0))
        img_id = getattr(p, "source_image_id", getattr(p, "image_id", "img_001"))
        p_cls = getattr(p, "class_name", "")
        p_is_subpart = getattr(p, "parent_box_id", None) is not None
        best_iou = 0.15 if p_is_subpart else 0.20
        best_gt = None
        for g in gt_boxes:
            if g.image_id == img_id:
                g_is_subpart = g.parent_entity_id is not None
                if p_is_subpart != g_is_subpart:
                    continue
                if g.class_name != p_cls and not are_semantically_compatible(g.class_name, p_cls):
                    continue
                iou = compute_box_iou(p_box, g.box_2d)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g.entity_id
        if best_gt:
            pred_to_gt_entity[p_id] = best_gt

    correct_links = 0
    false_parent_assignments = 0
    claimed_gt_parts: set[tuple[str, str]] = set()
    duplicate_component_detections = 0
    detected_part_candidates = 0

    for child_pid, parent_pid in pred_links.items():
        child_p = pred_by_id[child_pid]
        img_id = getattr(child_p, "source_image_id", getattr(child_p, "image_id", "img_001"))
        child_gt = pred_to_gt_entity.get(child_pid)
        parent_gt = pred_to_gt_entity.get(parent_pid)
        expected_parent = gt_links.get((img_id, child_gt)) if child_gt else None

        if child_gt and (img_id, child_gt) in gt_links:
            detected_part_candidates += 1
            if (img_id, child_gt) in claimed_gt_parts:
                duplicate_component_detections += 1
            else:
                claimed_gt_parts.add((img_id, child_gt))

        if child_gt and expected_parent is not None:
            if parent_gt == expected_parent:
                correct_links += 1
            else:
                false_parent_assignments += 1
        else:
            false_parent_assignments += 1

    total_gt = len(gt_links)
    total_pred = len(pred_links)
    missed_components = max(0, total_gt - correct_links)
    incorrect_parent_assignments = false_parent_assignments

    if total_gt == 0:
        prec = 0.0 if total_pred > 0 else None
        rec = None
        f1_val = None
        part_det_prec = 0.0 if total_pred > 0 else None
        part_det_rec = None
        parent_assign_prec = 0.0 if total_pred > 0 else None
        parent_assign_rec = None
        status = "unmeasured_no_gt"
    else:
        prec = float(correct_links / total_pred) if total_pred > 0 else 0.0
        rec = float(correct_links / total_gt)
        f1_val = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        part_det_prec = float(detected_part_candidates / total_pred) if total_pred > 0 else 0.0
        part_det_rec = float(len(claimed_gt_parts) / total_gt)
        parent_assign_prec = float(correct_links / detected_part_candidates) if detected_part_candidates > 0 else 0.0
        parent_assign_rec = float(correct_links / total_gt)
        status = "measured"

    return PartHierarchyMetrics(
        total_gt_part_links=total_gt,
        predicted_part_links=total_pred,
        correct_part_links=correct_links,
        false_parent_assignments=false_parent_assignments,
        missed_components=missed_components,
        part_detection_precision=part_det_prec,
        part_detection_recall=part_det_rec,
        parent_assignment_precision=parent_assign_prec,
        parent_assignment_recall=parent_assign_rec,
        incorrect_parent_assignments=incorrect_parent_assignments,
        duplicate_component_detections=duplicate_component_detections,
        precision=prec,
        recall=rec,
        f1=f1_val,
        status=status,
    )


def evaluate_mask_quality(
    gt_boxes: Sequence[GroundTruthBox],
    pred_boxes: Sequence[Any],
) -> MaskQualityMetrics:
    """Evaluates Mask IoU and Boundary IoU for instances with segmentation masks."""
    total_gt_masks = sum(1 for g in gt_boxes if g.mask_2d is not None)
    if total_gt_masks == 0:
        return MaskQualityMetrics(
            total_gt_masks=0,
            evaluated_masks=0,
            mean_mask_iou=None,
            median_mask_iou=None,
            mean_boundary_iou=None,
            mean_boundary_f_score=None,
            masks_above_50_iou=None,
            masks_above_75_iou=None,
            status="unmeasured_no_gt",
        )

    ious: list[float] = []
    bound_ious: list[float] = []
    bf_scores: list[float] = []

    for g in gt_boxes:
        if g.mask_2d is None:
            continue
        for p in pred_boxes:
            p_mask = getattr(p, "mask_2d", None)
            if p_mask is None or p_mask.shape != g.mask_2d.shape:
                continue
            b_iou = compute_box_iou(getattr(p, "box_2d", (0, 0, 0, 0)), g.box_2d)
            if b_iou >= 0.50:
                inter = np.logical_and(p_mask.astype(bool), g.mask_2d.astype(bool)).sum()
                union = np.logical_or(p_mask.astype(bool), g.mask_2d.astype(bool)).sum()
                m_iou = float(inter / union) if union > 0 else 0.0
                ious.append(m_iou)
                bnd_iou = compute_boundary_iou(p_mask.astype(bool), g.mask_2d.astype(bool))
                bound_ious.append(bnd_iou)
                bf = compute_boundary_f_score(p_mask.astype(bool), g.mask_2d.astype(bool))
                bf_scores.append(bf)
                break

    if not ious:
        return MaskQualityMetrics(
            total_gt_masks=total_gt_masks,
            evaluated_masks=0,
            mean_mask_iou=0.0,
            median_mask_iou=0.0,
            mean_boundary_iou=0.0,
            mean_boundary_f_score=0.0,
            masks_above_50_iou=0.0,
            masks_above_75_iou=0.0,
            status="no_matches",
        )

    return MaskQualityMetrics(
        total_gt_masks=total_gt_masks,
        evaluated_masks=len(ious),
        mean_mask_iou=float(np.mean(ious)),
        median_mask_iou=float(np.median(ious)),
        mean_boundary_iou=float(np.mean(bound_ious)),
        mean_boundary_f_score=float(np.mean(bf_scores)) if bf_scores else 0.0,
        masks_above_50_iou=float(np.mean([1.0 if v >= 0.50 else 0.0 for v in ious])),
        masks_above_75_iou=float(np.mean([1.0 if v >= 0.75 else 0.0 for v in ious])),
        status="measured",
    )


def evaluate_temporal_tracking(
    gt_trajectories: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    pred_tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    iou_threshold: float = 0.50,
) -> TrackingMetrics:
    """Computes global bipartite IDF1 (Ristani et al., 2016), standard CLEAR MOTA (Bernardin & Stiefelhagen, 2008),
    and standard HOTA, DetA, AssA (Luiten et al., 2021)."""
    from scipy.optimize import linear_sum_assignment

    total_gt_trajectories = len(gt_trajectories)
    total_gt_instances = sum(len(frames) for frames in gt_trajectories.values())
    total_pred_tracks = len(pred_tracks)
    total_pred_instances = sum(len(frames) for frames in pred_tracks.values())

    if total_gt_instances == 0:
        return TrackingMetrics(
            total_gt_trajectories=0,
            total_gt_instances=0,
            total_predicted_tracks=total_pred_tracks,
            total_predicted_instances=total_pred_instances,
            id_switches=0,
            track_fragmentations=0,
            idf1=None,
            id_precision=0.0 if total_pred_instances > 0 else None,
            id_recall=None,
            mota=None,
            hota=None,
            deta=None,
            assa=None,
            track_coverage=None,
            iou_threshold=iou_threshold,
            status="unmeasured_no_gt",
        )

    # 1. Global bipartite matching for IDF1 (Ristani et al., ECCV 2016)
    gt_id_list = list(gt_trajectories.keys())
    pred_id_list = list(pred_tracks.keys())
    cost_matrix = np.zeros((len(gt_id_list), len(pred_id_list)), dtype=float)

    for i, g_id in enumerate(gt_id_list):
        g_map = {f[0]: f[1] for f in gt_trajectories[g_id]}
        for j, p_id in enumerate(pred_id_list):
            overlap = 0
            for f_idx, p_box in pred_tracks[p_id]:
                if f_idx in g_map and compute_box_iou(p_box, g_map[f_idx]) >= iou_threshold:
                    overlap += 1
            cost_matrix[i, j] = -overlap

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    id_tp = int(-cost_matrix[row_ind, col_ind].sum()) if len(row_ind) > 0 else 0

    id_prec = float(id_tp / total_pred_instances) if total_pred_instances > 0 else 0.0
    id_rec = float(id_tp / total_gt_instances) if total_gt_instances > 0 else 0.0
    idf1 = float(2 * id_tp / (total_gt_instances + total_pred_instances)) if (total_gt_instances + total_pred_instances) > 0 else 0.0

    # 2. CLEAR MOT metrics (Bernardin & Stiefelhagen, EURASIP 2008):
    # In each frame:
    # 1. Matches from the previous frame are preserved if both entities exist and IoU >= threshold.
    # 2. Unmatched GT and hypotheses are paired via bipartite Hungarian matching on IoU.
    # 3. An ID switch occurs when a GT was mapped to track P_prev and is now mapped to P_curr != P_prev.
    all_frames = sorted(set(
        [f[0] for frames in gt_trajectories.values() for f in frames] +
        [f[0] for frames in pred_tracks.values() for f in frames]
    ))
    id_switches = 0
    fragmentations = 0
    prev_frame_matches: dict[str, str] = {}  # g_id -> p_id in immediate previous frame
    last_matched_track: dict[str, str] = {}  # g_id -> last observed matched track ID
    frame_tp_total = 0
    frame_fn_total = 0
    frame_fp_total = 0

    for f_idx in all_frames:
        f_gts = {g_id: box for g_id, frames in gt_trajectories.items() for fi, box in frames if fi == f_idx}
        f_preds = {p_id: box for p_id, frames in pred_tracks.items() for fi, box in frames if fi == f_idx}

        curr_frame_matches: dict[str, str] = {}
        unmatched_gts = set(f_gts.keys())
        unmatched_preds = set(f_preds.keys())

        # Step 2a: Preserve correspondence from previous frame if IoU >= threshold
        for g_id, prev_pid in list(prev_frame_matches.items()):
            if g_id in unmatched_gts and prev_pid in unmatched_preds:
                iou = compute_box_iou(f_gts[g_id], f_preds[prev_pid])
                if iou >= iou_threshold:
                    curr_frame_matches[g_id] = prev_pid
                    unmatched_gts.remove(g_id)
                    unmatched_preds.remove(prev_pid)

        # Step 2b: Bipartite matching for remaining unmatched GTs and predictions
        if unmatched_gts and unmatched_preds:
            rem_gts = list(unmatched_gts)
            rem_preds = list(unmatched_preds)
            iou_mat = np.zeros((len(rem_gts), len(rem_preds)), dtype=float)
            for gi, g_id in enumerate(rem_gts):
                for pi, p_id in enumerate(rem_preds):
                    iou_mat[gi, pi] = compute_box_iou(f_gts[g_id], f_preds[p_id])

            r_idx, c_idx = linear_sum_assignment(-iou_mat)
            for r, c in zip(r_idx, c_idx):
                if iou_mat[r, c] >= iou_threshold:
                    g_id = rem_gts[r]
                    p_id = rem_preds[c]
                    curr_frame_matches[g_id] = p_id
                    if g_id in last_matched_track and last_matched_track[g_id] != p_id:
                        id_switches += 1
                    last_matched_track[g_id] = p_id
                    unmatched_gts.remove(g_id)
                    unmatched_preds.remove(p_id)

        # Update fragmentations: if a GT was tracked in previous frame but untracked now
        for g_id in f_gts:
            if g_id not in curr_frame_matches and g_id in prev_frame_matches:
                fragmentations += 1

        prev_frame_matches = curr_frame_matches

        frame_tp = len(curr_frame_matches)
        frame_fn = len(f_gts) - frame_tp
        frame_fp = len(f_preds) - frame_tp
        frame_tp_total += frame_tp
        frame_fn_total += frame_fn
        frame_fp_total += frame_fp

    # Standard CLEAR MOT metrics (Bernardin & Stiefelhagen, EURASIP 2008):
    # MOTA = 1 - (sum_t FN_t + sum_t FP_t + sum_t IDSW_t) / sum_t GT_t
    mota = 1.0 - float(frame_fn_total + frame_fp_total + id_switches) / float(total_gt_instances) if total_gt_instances > 0 else 0.0
    track_cov = float(frame_tp_total / total_gt_instances) if total_gt_instances > 0 else 0.0

    # 3. Standard HOTA (Luiten et al., IJCV 2021) across 19 localization thresholds alpha in [0.05, 0.95]
    alpha_thresholds = np.linspace(0.05, 0.95, 19)
    hota_alphas: list[float] = []
    deta_alphas: list[float] = []
    assa_alphas: list[float] = []

    for alpha in alpha_thresholds:
        frame_matches_alpha: dict[int, list[tuple[str, str]]] = defaultdict(list)
        unmatched_gt_alpha = 0
        unmatched_pred_alpha = 0

        for f_idx in all_frames:
            f_gts = [(g_id, box) for g_id, frames in gt_trajectories.items() for fi, box in frames if fi == f_idx]
            f_preds = [(p_id, box) for p_id, frames in pred_tracks.items() for fi, box in frames if fi == f_idx]
            if not f_gts and not f_preds:
                continue
            if not f_gts:
                unmatched_pred_alpha += len(f_preds)
                continue
            if not f_preds:
                unmatched_gt_alpha += len(f_gts)
                continue

            iou_mat = np.zeros((len(f_gts), len(f_preds)), dtype=float)
            for gi, (_, g_box) in enumerate(f_gts):
                for pi, (_, p_box) in enumerate(f_preds):
                    iou_mat[gi, pi] = compute_box_iou(g_box, p_box)

            r_idx, c_idx = linear_sum_assignment(-iou_mat)
            matched_r = set()
            matched_c = set()
            for r, c in zip(r_idx, c_idx):
                if iou_mat[r, c] >= alpha:
                    frame_matches_alpha[f_idx].append((f_gts[r][0], f_preds[c][0]))
                    matched_r.add(r)
                    matched_c.add(c)
            unmatched_gt_alpha += (len(f_gts) - len(matched_r))
            unmatched_pred_alpha += (len(f_preds) - len(matched_c))

        total_tp_alpha = sum(len(m) for m in frame_matches_alpha.values())
        if total_tp_alpha == 0:
            deta_alphas.append(0.0)
            assa_alphas.append(0.0)
            hota_alphas.append(0.0)
            continue

        deta_a = float(total_tp_alpha / (total_tp_alpha + unmatched_gt_alpha + unmatched_pred_alpha))
        deta_alphas.append(deta_a)

        # Compute association score A(c) for each matched detection pair c = (g_id, p_id)
        pair_match_counts: Counter[tuple[str, str]] = Counter()
        for f_idx, m_list in frame_matches_alpha.items():
            for g_id, p_id in m_list:
                pair_match_counts[(g_id, p_id)] += 1

        ass_scores: list[float] = []
        for f_idx, m_list in frame_matches_alpha.items():
            for g_id, p_id in m_list:
                tpa = pair_match_counts[(g_id, p_id)]
                p_len = len(pred_tracks[p_id])
                fpa = p_len - tpa
                g_len = len(gt_trajectories[g_id])
                fna = g_len - tpa
                denom = tpa + fpa + fna
                a_score = float(tpa / denom) if denom > 0 else 0.0
                ass_scores.append(a_score)

        assa_a = float(np.mean(ass_scores)) if ass_scores else 0.0
        assa_alphas.append(assa_a)
        hota_alphas.append(math.sqrt(deta_a * assa_a))

    hota = float(np.mean(hota_alphas)) if hota_alphas else 0.0
    deta = float(np.mean(deta_alphas)) if deta_alphas else 0.0
    assa = float(np.mean(assa_alphas)) if assa_alphas else 0.0

    return TrackingMetrics(
        total_gt_trajectories=total_gt_trajectories,
        total_gt_instances=total_gt_instances,
        total_predicted_tracks=total_pred_tracks,
        total_predicted_instances=total_pred_instances,
        frame_tp=frame_tp_total,
        frame_fn=frame_fn_total,
        frame_fp=frame_fp_total,
        id_tp=id_tp,
        id_switches=id_switches,
        track_fragmentations=fragmentations,
        idf1=idf1,
        id_precision=id_prec,
        id_recall=id_rec,
        mota=mota,
        hota=hota,
        deta=deta,
        assa=assa,
        track_coverage=track_cov,
        status="measured",
    )


def evaluate_multiview_association(
    gt_entity_observations: dict[str, set[str]],  # gt_entity_id -> set of observation_ids
    merged_entities: Sequence[Any],
    association_decisions: Sequence[Any],
    gt_entity_classes: dict[str, str] | None = None,
) -> MultiViewAssociationMetrics:
    """Evaluates cross-view association precision, recall, and reprojection residuals."""
    obs_to_gt: dict[str, str] = {}
    for gt_id, obs_set in gt_entity_observations.items():
        for o in obs_set:
            obs_to_gt[o] = gt_id

    merged_count = 0
    refused_count = 0
    correct_merges = 0
    incorrect_merges = 0

    inlier_residuals: list[float] = []
    all_residuals: list[float] = []

    valid_decisions = [
        d for d in association_decisions
        if getattr(d, "obs_a_id", "") and getattr(d, "obs_b_id", "") and getattr(d, "obs_a_id", "") != getattr(d, "obs_b_id", "")
    ]

    for d in valid_decisions:
        outcome = getattr(d, "outcome", "")
        is_merged = (outcome == "merged" or getattr(outcome, "value", "") == "merged")
        obs_a = getattr(d, "obs_a_id", "")
        obs_b = getattr(d, "obs_b_id", "")

        err = getattr(d, "reprojection_err_px", None)
        if err is not None and math.isfinite(err):
            all_residuals.append(float(err))
            if is_merged:
                inlier_residuals.append(float(err))

        if is_merged:
            merged_count += 1
            gt_a = obs_to_gt.get(obs_a)
            gt_b = obs_to_gt.get(obs_b)
            if gt_a and gt_b and gt_a == gt_b:
                correct_merges += 1
            else:
                incorrect_merges += 1
        else:
            refused_count += 1

    total_gt_pairs = 0
    for obs_set in gt_entity_observations.values():
        n = len(obs_set)
        total_gt_pairs += (n * (n - 1)) // 2

    if total_gt_pairs == 0:
        return MultiViewAssociationMetrics(
            candidate_pairs_evaluated=len(valid_decisions),
            merged_pairs=merged_count,
            refused_pairs=refused_count,
            total_gt_pairs=0,
            association_precision=0.0 if merged_count > 0 else None,
            association_recall=None,
            inlier_mean_reprojection_residual_px=float(np.mean(inlier_residuals)) if inlier_residuals else None,
            inlier_median_reprojection_residual_px=float(np.median(inlier_residuals)) if inlier_residuals else None,
            inlier_p90_reprojection_residual_px=float(np.percentile(inlier_residuals, 90)) if inlier_residuals else None,
            all_pairs_mean_residual_px=float(np.mean(all_residuals)) if all_residuals else None,
            status="unmeasured_no_gt",
        )

    assoc_prec = float(correct_merges / merged_count) if merged_count > 0 else 0.0
    assoc_rec = float(correct_merges / total_gt_pairs)
    false_merge_rate = float(incorrect_merges / merged_count) if merged_count > 0 else 0.0

    # 1. Cluster Purity: fraction of cluster members belonging to the dominant true entity
    purity_numer = 0
    purity_denom = 0
    for entity in merged_entities:
        obs_ids = sorted(set(getattr(entity, "member_observation_ids", getattr(entity, "constituent_observation_ids", ()))))
        if not obs_ids:
            continue
        gt_counts: Counter[str] = Counter()
        for oid in obs_ids:
            gt_id = obs_to_gt.get(oid)
            if gt_id:
                gt_counts[gt_id] += 1
        if gt_counts:
            purity_numer += gt_counts.most_common(1)[0][1]
        purity_denom += len(obs_ids)

    cluster_purity = float(purity_numer / purity_denom) if purity_denom > 0 else 1.0

    # 2. Independent Identity Completeness (fraction of true entity observations in its dominant cluster)
    total_mapped_gt_obs = sum(len(s) for s in gt_entity_observations.values())
    completeness_sum = 0
    for gt_id, obs_set in gt_entity_observations.items():
        if not obs_set:
            continue
        c_counts: Counter[int] = Counter()
        for c_idx, entity in enumerate(merged_entities):
            c_obs = set(getattr(entity, "member_observation_ids", getattr(entity, "constituent_observation_ids", ())))
            intersect = len(obs_set & c_obs)
            if intersect > 0:
                c_counts[c_idx] = intersect
        if c_counts:
            completeness_sum += c_counts.most_common(1)[0][1]

    identity_completeness = float(completeness_sum / total_mapped_gt_obs) if total_mapped_gt_obs > 0 else 0.0

    # 2b. False-Split Rate: fraction of multi-observation GT entities fragmented across multiple clusters
    split_gt_entities = 0
    multiview_gt_count = sum(1 for s in gt_entity_observations.values() if len(s) > 1)
    for gt_id, obs_set in gt_entity_observations.items():
        if len(obs_set) <= 1:
            continue
        matching_clusters = sum(
            1 for entity in merged_entities
            if len(obs_set & set(getattr(entity, "member_observation_ids", getattr(entity, "constituent_observation_ids", ())))) > 0
        )
        if matching_clusters > 1:
            split_gt_entities += 1
    false_split_rate = float(split_gt_entities / multiview_gt_count) if multiview_gt_count > 0 else 0.0

    # 3. Single-view and Multi-view entity rates
    tot_entities = len(merged_entities)
    single_view_cnt = sum(1 for e in merged_entities if len(getattr(e, "member_observation_ids", getattr(e, "constituent_observation_ids", ()))) <= 1)
    multi_view_cnt = sum(1 for e in merged_entities if len(getattr(e, "member_observation_ids", getattr(e, "constituent_observation_ids", ()))) >= 2)
    single_view_rate = float(single_view_cnt / tot_entities) if tot_entities > 0 else 0.0
    multi_view_rate = float(multi_view_cnt / tot_entities) if tot_entities > 0 else 0.0

    mean_inlier = float(np.mean(inlier_residuals)) if inlier_residuals else None
    med_inlier = float(np.median(inlier_residuals)) if inlier_residuals else None
    p25_inlier = float(np.percentile(inlier_residuals, 25)) if inlier_residuals else None
    p75_inlier = float(np.percentile(inlier_residuals, 75)) if inlier_residuals else None
    p90_inlier = float(np.percentile(inlier_residuals, 90)) if inlier_residuals else None
    p95_inlier = float(np.percentile(inlier_residuals, 95)) if inlier_residuals else None
    all_mean = float(np.mean(all_residuals)) if all_residuals else None

    # Breakdown by class
    results_by_class: dict[str, dict[str, Any]] = {}
    if gt_entity_classes:
        for cname in sorted(set(gt_entity_classes.values())):
            c_gt_ids = {gid for gid, c in gt_entity_classes.items() if c == cname}
            c_gt_pairs = sum((len(gt_entity_observations[gid]) * (len(gt_entity_observations[gid]) - 1)) // 2 for gid in c_gt_ids if gid in gt_entity_observations)
            c_merged = 0
            c_correct = 0
            c_residuals: list[float] = []
            for d in valid_decisions:
                outcome = getattr(d, "outcome", "")
                is_merged = (outcome == "merged" or getattr(outcome, "value", "") == "merged")
                obs_a = getattr(d, "obs_a_id", "")
                obs_b = getattr(d, "obs_b_id", "")
                gt_a = obs_to_gt.get(obs_a)
                gt_b = obs_to_gt.get(obs_b)
                if (gt_a in c_gt_ids) or (gt_b in c_gt_ids):
                    if is_merged:
                        c_merged += 1
                        if gt_a and gt_b and gt_a == gt_b:
                            c_correct += 1
                            err = getattr(d, "reprojection_err_px", None)
                            if err is not None and math.isfinite(err):
                                c_residuals.append(float(err))
            results_by_class[cname] = {
                "total_gt_pairs": c_gt_pairs,
                "merged_pairs": c_merged,
                "correct_merges": c_correct,
                "precision": round(c_correct / c_merged, 4) if c_merged > 0 else 0.0,
                "recall": round(c_correct / c_gt_pairs, 4) if c_gt_pairs > 0 else 0.0,
                "inlier_mean_residual_px": round(float(np.mean(c_residuals)), 2) if c_residuals else None,
            }

    # Breakdown by view count (1, 2, 3+ views)
    results_by_view_count: dict[str, dict[str, Any]] = {}
    for vc_bucket, vc_filter in [
        ("1_view", lambda n: n == 1),
        ("2_views", lambda n: n == 2),
        ("3+_views", lambda n: n >= 3),
    ]:
        b_gt_ids = {gid for gid, s in gt_entity_observations.items() if vc_filter(len(s))}
        b_gt_pairs = sum((len(gt_entity_observations[gid]) * (len(gt_entity_observations[gid]) - 1)) // 2 for gid in b_gt_ids)
        b_merged = 0
        b_correct = 0
        b_residuals: list[float] = []
        for d in valid_decisions:
            outcome = getattr(d, "outcome", "")
            is_merged = (outcome == "merged" or getattr(outcome, "value", "") == "merged")
            obs_a = getattr(d, "obs_a_id", "")
            obs_b = getattr(d, "obs_b_id", "")
            gt_a = obs_to_gt.get(obs_a)
            gt_b = obs_to_gt.get(obs_b)
            if (gt_a in b_gt_ids) or (gt_b in b_gt_ids):
                if is_merged:
                    b_merged += 1
                    if gt_a and gt_b and gt_a == gt_b:
                        b_correct += 1
                        err = getattr(d, "reprojection_err_px", None)
                        if err is not None and math.isfinite(err):
                            b_residuals.append(float(err))
        results_by_view_count[vc_bucket] = {
            "entity_count": len(b_gt_ids),
            "total_gt_pairs": b_gt_pairs,
            "merged_pairs": b_merged,
            "correct_merges": b_correct,
            "precision": round(b_correct / b_merged, 4) if b_merged > 0 else 0.0,
            "recall": round(b_correct / b_gt_pairs, 4) if b_gt_pairs > 0 else 0.0,
            "inlier_mean_residual_px": round(float(np.mean(b_residuals)), 2) if b_residuals else None,
        }

    return MultiViewAssociationMetrics(
        candidate_pairs_evaluated=len(valid_decisions),
        merged_pairs=merged_count,
        refused_pairs=refused_count,
        total_gt_pairs=total_gt_pairs,
        association_precision=assoc_prec,
        association_recall=assoc_rec,
        cluster_purity=cluster_purity,
        identity_completeness=identity_completeness,
        false_merge_rate=false_merge_rate,
        false_split_rate=false_split_rate,
        single_view_entity_rate=single_view_rate,
        multi_view_entity_rate=multi_view_rate,
        inlier_mean_reprojection_residual_px=mean_inlier,
        inlier_median_reprojection_residual_px=med_inlier,
        inlier_p25_reprojection_residual_px=p25_inlier,
        inlier_p75_reprojection_residual_px=p75_inlier,
        inlier_p90_reprojection_residual_px=p90_inlier,
        inlier_p95_reprojection_residual_px=p95_inlier,
        all_pairs_mean_residual_px=all_mean,
        results_by_class=results_by_class if results_by_class else None,
        results_by_view_count=results_by_view_count if results_by_view_count else None,
        status="measured",
    )


def measure_hardware_telemetry() -> dict[str, Any]:
    """Measures peak process RSS and peak CUDA VRAM memory usage."""
    telemetry: dict[str, Any] = {
        "timestamp": time.time(),
        "peak_rss_mb": None,
        "peak_cuda_vram_mb": None,
    }
    try:
        import psutil

        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None) or getattr(mi, "rss", None)
        if peak:
            telemetry["peak_rss_mb"] = round(peak / (1024 * 1024), 2)
    except Exception:
        pass
    if telemetry["peak_rss_mb"] is None:  # no psutil: POSIX getrusage (KiB on Linux, bytes on macOS)
        try:
            import resource
            import sys as _sys

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            telemetry["peak_rss_mb"] = round(rss / (1024 * 1024) if _sys.platform == "darwin" else rss / 1024, 2)
        except Exception:
            pass

    try:
        import torch

        if torch.cuda.is_available():
            telemetry["peak_cuda_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    except Exception:
        pass

    return telemetry


# ============================================================================
# Dataset Loading & End-to-End Benchmark Execution
# ============================================================================

def load_ground_truth_from_dataset(
    dataset_path: Path | str,
    split: str = "all",
    include_subparts: bool = False,
) -> tuple[list[GroundTruthBox], dict[str, Any]]:
    """Loads ground-truth 2D annotations and physical entity metadata from a dataset directory or JSON file.

    Supports split partitions:
    - 'dev': IMG_0000.jpg to IMG_0010.jpg (development split for tuning)
    - 'test': IMG_0011.jpg to IMG_0020.jpg (strictly held-out test split)
    - 'all': complete evaluation dataset

    Enforces zero fabrication:
    - Never fabricates ground truth where absent.
    - Preserves unmeasured / blocked dataset status flags.
    """
    path = Path(dataset_path)
    if path.is_dir():
        gt_file = path / "ground_truth_identities.json"
        if not gt_file.is_file():
            gt_file = path / "ground_truth.json"
    else:
        gt_file = path

    if not gt_file.is_file():
        raise FileNotFoundError(f"Ground truth file not found at {gt_file}")

    import json

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_boxes: list[GroundTruthBox] = []
    image_annotations = data.get("image_annotations", {})
    sorted_image_names = sorted(image_annotations.keys())
    img_to_frame_idx = {name: idx for idx, name in enumerate(sorted_image_names)}

    import re

    for img_name, annotations in image_annotations.items():
        # Try to parse frame index from name (e.g. IMG_0004.jpg -> 4) or fallback to sorted index
        m = re.search(r"(\d+)", img_name)
        frame_idx = int(m.group(1)) if m else img_to_frame_idx.get(img_name, 0)

        # Apply train/test split filtering
        norm_split = split.lower().strip()
        if norm_split == "dev" and frame_idx > 10:
            continue
        elif norm_split == "test" and frame_idx <= 10:
            continue

        # Determine image dimensions dynamically
        img_w, img_h = 4032, 3024
        img_candidates = [
            path / img_name,
            path / "images" / img_name,
        ]
        if path.is_file():
            img_candidates.extend([
                path.parent / img_name,
                path.parent / "images" / img_name,
            ])
        for cand in img_candidates:
            if cand.is_file():
                try:
                    from PIL import Image
                    with Image.open(cand) as im:
                        img_w, img_h = im.size
                    break
                except Exception:
                    pass
        else:
            if "image_size" in data:
                img_w, img_h = int(data["image_size"][0]), int(data["image_size"][1])

        for ann in annotations:
            parent_id = ann.get("parent_entity_id")
            if not include_subparts and parent_id:
                continue
            gt_id = ann.get("ground_truth_id") or ann.get("entity_id", "")
            cls_name = ann.get("class_name") or ann.get("label", "")
            b = ann.get("bbox_2d") or ann.get("box_2d", [0, 0, 0, 0])
            vis_state = ann.get("visibility", "visible")
            occ_lvl = ann.get("occlusion_level", "none")

            mask_2d = None
            poly = ann.get("polygon") or ann.get("mask_polygon")
            if poly and len(poly) >= 3:
                try:
                    from PIL import Image, ImageDraw
                    mask_img = Image.new("1", (img_w, img_h), 0)
                    ImageDraw.Draw(mask_img).polygon([tuple(pt) for pt in poly], fill=1)
                    mask_2d = np.asarray(mask_img, dtype=bool)
                except Exception:
                    mask_2d = None
            elif ann.get("has_mask"):
                try:
                    from PIL import Image, ImageDraw
                    mask_img = Image.new("1", (img_w, img_h), 0)
                    x1 = max(0.0, float(b[0]))
                    y1 = max(0.0, float(b[1]))
                    x2 = min(float(img_w), float(b[2]))
                    y2 = min(float(img_h), float(b[3]))
                    ImageDraw.Draw(mask_img).rectangle([x1, y1, x2, y2], fill=1)
                    mask_2d = np.asarray(mask_img, dtype=bool)
                except Exception:
                    mask_2d = None

            gt_boxes.append(GroundTruthBox(
                entity_id=gt_id,
                class_name=cls_name,
                box_2d=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                image_id=img_name,
                parent_entity_id=parent_id,
                frame_index=frame_idx,
                mask_2d=mask_2d,
                visibility=vis_state,
                occlusion_level=occ_lvl,
            ))

    metadata = {
        "dataset": data.get("dataset", path.name),
        "description": data.get("description", ""),
        "is_physically_surveyed": data.get("is_physically_surveyed", False),
        "milestone_status": data.get("milestone_status", "PROVISIONAL"),
        "total_images_annotated": len(sorted_image_names),
        "split": split,
        "images_in_split": len(set(g.image_id for g in gt_boxes)),
        "total_boxes": len(gt_boxes),
    }
    return gt_boxes, metadata


def load_colmap_calibration(dataset_dir: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Loads camera intrinsics and extrinsics from COLMAP cameras.txt and images.txt in a dataset directory."""
    dpath = Path(dataset_dir)
    intrinsics: dict[str, Any] = {}
    extrinsics: dict[str, Any] = {}

    cam_file = dpath / "cameras.txt"
    if cam_file.is_file():
        from reconstruction.colmap_intrinsics import read_colmap_cameras
        cams = read_colmap_cameras(dpath)
        for idx, c in enumerate(cams):
            intrinsics[str(idx + 1)] = c

    img_file = dpath / "images.txt"
    if img_file.is_file():
        from reconstruction.backend.colmap_backend import _parse_images_txt
        text = img_file.read_text(encoding="utf-8")
        parsed_poses = _parse_images_txt(text, {})
        for pose in parsed_poses:
            extrinsics[pose.evidence_id] = {
                "position": pose.position,
                "rotation": pose.rotation,
            }

    return intrinsics, extrinsics


def run_real_perception_benchmark(
    dataset_dir: Path | str,
    output_dir: Path | str | None = None,
    benchmark_id: str = "real_perception_benchmark_001",
    detector_engine: Any = None,
    segmenter_engine: Any = None,
    iou_thresholds: tuple[float, float] = (0.50, 0.75),
    split: str = "all",
    text_prompts: Sequence[str] | None = None,
    target_resolution: tuple[int, int] | None = None,
) -> BenchmarkExecutionSummary:
    """Executes the full perception, tracking, multi-view association, and telemetry benchmark on a real dataset.

    Returns a complete, verified BenchmarkExecutionSummary with confidence-ranked PR AP,
    unambiguous fixed-IoU F1, and clean inlier reprojection metrics.
    """
    dpath = Path(dataset_dir)
    t_start = time.monotonic()

    # 1. Load ground truth for designated split
    gt_boxes, gt_meta = load_ground_truth_from_dataset(dpath, split=split, include_subparts=False)
    gt_all_boxes, _ = load_ground_truth_from_dataset(dpath, split=split, include_subparts=True)

    # Scale ground truth boxes and masks if target_resolution is specified
    if target_resolution is not None:
        from PIL import Image as PILImage
        import cv2
        scaled_gt_boxes: list[GroundTruthBox] = []
        scaled_gt_all: list[GroundTruthBox] = []

        def _scale_gt(g_in: GroundTruthBox) -> GroundTruthBox:
            img_p = dpath / g_in.image_id
            if img_p.is_file():
                with PILImage.open(img_p) as im_ref:
                    orig_w, orig_h = im_ref.size
            else:
                orig_w, orig_h = (4032, 3024)
            w_scale = target_resolution[0] / orig_w
            h_scale = target_resolution[1] / orig_h
            new_box = (
                g_in.box_2d[0] * w_scale,
                g_in.box_2d[1] * h_scale,
                g_in.box_2d[2] * w_scale,
                g_in.box_2d[3] * h_scale,
            )
            new_mask = None
            if g_in.mask_2d is not None:
                new_mask = cv2.resize(
                    g_in.mask_2d.astype(np.uint8),
                    target_resolution,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            return GroundTruthBox(
                entity_id=g_in.entity_id,
                class_name=g_in.class_name,
                box_2d=new_box,
                image_id=g_in.image_id,
                mask_2d=new_mask,
                parent_entity_id=g_in.parent_entity_id,
                frame_index=g_in.frame_index,
                visibility=g_in.visibility,
            )

        for g in gt_boxes:
            scaled_gt_boxes.append(_scale_gt(g))
        for g in gt_all_boxes:
            scaled_gt_all.append(_scale_gt(g))
        gt_boxes = scaled_gt_boxes
        gt_all_boxes = scaled_gt_all

    gt_by_image: dict[str, list[GroundTruthBox]] = defaultdict(list)
    for g in gt_boxes:
        gt_by_image[g.image_id].append(g)

    # 2. Setup perception pipeline with real detector adapter if available
    from engine.multiengine.spatial_intelligence.detection_pipeline import (
        DetectionBox,
        DetectionPipeline,
        DetectionScale,
        LearnedConfidenceCalibrator,
        create_default_detector,
        create_default_segmenter,
    )
    from engine.multiengine.spatial_intelligence.association import (
        CrossViewRayCandidate,
        MultiViewIntelligenceEngine,
        TemporalIdentityTracker,
        build_camera_ray,
    )
    from engine.multiengine.spatial_intelligence.taxonomy import Domain

    if detector_engine is None:
        detector_engine = create_default_detector()

    if segmenter_engine is None:
        segmenter_engine = create_default_segmenter()

    calibrator = None
    cal_file = Path(output_dir if output_dir else "artifacts/perception") / "calibrator_dev.json"
    if not cal_file.is_file():
        cal_file = Path("artifacts/perception/calibrator_dev.json")
    if cal_file.is_file():
        try:
            calibrator = LearnedConfidenceCalibrator.load(cal_file)
            print(f"Loaded learned confidence calibrator from {cal_file} (T={calibrator.temperature:.4f}, bias={calibrator.bias:.4f}, classes={len(calibrator.class_params)})")
        except Exception as e:
            print(f"Warning: could not load calibrator from {cal_file}: {e}")

    pipe = DetectionPipeline(detector=detector_engine, segmenter=segmenter_engine, calibrator=calibrator)

    if text_prompts:
        prompts_to_use = list(text_prompts)
    else:
        annotated_classes = sorted(set(g.class_name for g in gt_all_boxes))
        prompts_to_use = annotated_classes if annotated_classes else [
            "door", "window", "table", "chair", "lamp",
            "door_handle", "window_frame", "table_top", "chair_seat",
        ]

    # 3. Process each annotated image in the split
    all_predicted_boxes: list[DetectionBox] = []
    frame_observations: dict[int, list[DetectionBox]] = defaultdict(list)
    obs_by_image: dict[str, list[DetectionBox]] = defaultdict(list)

    sorted_image_names = sorted(gt_by_image.keys())

    for loop_idx, img_name in enumerate(sorted_image_names):
        img_path = dpath / img_name
        if not img_path.is_file():
            continue

        m = re.search(r"(\d+)", img_name)
        frame_idx = int(m.group(1)) if m else loop_idx

        print(f"  [{loop_idx + 1}/{len(sorted_image_names)}] Inferring perception on {img_name} ({split} split)...", flush=True)
        from PIL import Image
        with Image.open(img_path) as im:
            if target_resolution is not None and im.size != target_resolution:
                im = im.resize(target_resolution, Image.Resampling.BILINEAR)
            rgb = np.asarray(im.convert("RGB"))

        detect_parts = any(g.parent_entity_id is not None for g in gt_all_boxes)
        res = pipe.run_pipeline(
            rgb,
            domain=Domain.INTERIOR,
            image_id=img_name,
            frame_index=frame_idx,
            timestamp_sec=frame_idx * 0.1,  # 10 fps nominal
            run_tiling=False,  # default standard scale for evaluation
            detect_subparts=detect_parts,
            text_prompts=prompts_to_use,
        )

        for b in res.refined_boxes:
            all_predicted_boxes.append(b)
            if not b.parent_box_id:
                frame_observations[frame_idx].append(b)
                obs_by_image[img_name].append(b)

        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 4. Temporal Tracking Evaluation
    tracker = TemporalIdentityTracker(max_missed_frames=5, min_iou_match=0.15, max_centroid_shift_px=450.0)
    predicted_tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    assignments: dict[str, str] = {}

    for frame_idx in sorted(frame_observations.keys()):
        obs_list = frame_observations[frame_idx]
        ts = frame_idx * 0.1
        frame_assignments = tracker.process_frame_observations(frame_idx, ts, obs_list)
        assignments.update(frame_assignments)
        for obs in obs_list:
            trk_id = frame_assignments.get(obs.box_id)
            if trk_id:
                predicted_tracks[trk_id].append((frame_idx, tuple(obs.box_2d)))

    # Form ground truth trajectories
    gt_trajectories: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for g in gt_boxes:
        if g.frame_index is not None:
            gt_trajectories[g.entity_id].append((g.frame_index, tuple(g.box_2d)))

    tracking_metrics = evaluate_temporal_tracking(gt_trajectories, predicted_tracks)

    # 5. Multi-View Association Evaluation
    intrinsics, extrinsics = load_colmap_calibration(dpath)
    ray_candidates: list[CrossViewRayCandidate] = []

    from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
    from engine.math import Quat, Vec3

    for img_name, boxes in obs_by_image.items():
        pose = extrinsics.get(img_name)
        if pose is None:
            # Camera is uncalibrated / not registered in SfM; cannot construct 3D ray without fabricating camera pose
            continue
        pos = pose["position"]
        rot = pose["rotation"]

        fx, fy, px, py = 1000.0, 1000.0, 1000.0, 750.0
        w_cam, h_cam = 4032, 3024
        if intrinsics and "1" in intrinsics:
            c_info = intrinsics["1"]
            w_cam = c_info.get("width", 4032)
            h_cam = c_info.get("height", 3024)
            params = c_info.get("params", [])
            if len(params) >= 4:
                fx, fy, px, py = float(params[0]), float(params[1]), float(params[2]), float(params[3])

        if target_resolution is not None:
            s_x = target_resolution[0] / w_cam
            s_y = target_resolution[1] / h_cam
            w_cam, h_cam = target_resolution
            fx *= s_x
            fy *= s_y
            px *= s_x
            py *= s_y

        cam_intr = CameraIntrinsics(fx=fx, fy=fy, cx=px, cy=py, width=w_cam, height=h_cam)
        cam_extr = CameraExtrinsics(
            position=Vec3(float(pos[0]), float(pos[1]), float(pos[2])),
            rotation=Quat(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
        )
        cam = PinholeCamera(intrinsics=cam_intr, extrinsics=cam_extr)

        for b in boxes:
            cand = build_camera_ray(
                observation_id=b.box_id,
                source_image_id=img_name,
                class_name=b.class_name,
                box_2d=b.box_2d,
                camera=cam,
                scale_state="sfm_gauge",
                scale_units="sfm_units",
                is_physically_metric=False,
                coordinate_frame="sfm_world",
                track_id=assignments.get(b.box_id),
            )
            ray_candidates.append(cand)

    # Calibrated geometry association with tight reprojection gating (30px)
    mv_engine = MultiViewIntelligenceEngine(
        max_ray_distance_m=0.35,
        max_reproj_err_px=30.0,
        min_match_score=0.25,
    )
    merged_entities, assoc_decisions, _ = mv_engine.associate_candidates(ray_candidates)

    # Map GT entities to predicted observation IDs via best box IoU
    # Only evaluate on images with valid calibration extrinsics to avoid inflating denominator with uncalibrated observations
    calibrated_images = set(extrinsics.keys()) if extrinsics else set()
    gt_entity_obs: dict[str, set[str]] = defaultdict(set)
    for g in gt_boxes:
        if calibrated_images and g.image_id not in calibrated_images:
            continue
        img_preds = obs_by_image.get(g.image_id, [])
        best_iou = 0.0
        best_box_id = None
        for p in img_preds:
            iou = compute_box_iou(g.box_2d, p.box_2d)
            if iou > best_iou:
                best_iou = iou
                best_box_id = p.box_id
        if best_box_id and best_iou >= 0.30:
            gt_entity_obs[g.entity_id].add(best_box_id)

    gt_entity_classes = {g.entity_id: g.class_name for g in gt_boxes}
    mv_metrics = evaluate_multiview_association(
        gt_entity_obs,
        merged_entities,
        assoc_decisions,
        gt_entity_classes=gt_entity_classes,
    )

    # 6. Quantitative Detection, Part, and Mask Metrics
    # Primary detections are evaluated on primary GT boxes
    primary_classes = set(g.class_name for g in gt_boxes)
    primary_pred_boxes = [b for b in all_predicted_boxes if b.class_name in primary_classes and not b.parent_box_id]
    if not primary_pred_boxes:
        primary_pred_boxes = [b for b in all_predicted_boxes if b.class_name in primary_classes]
    det_iou50 = evaluate_detections(gt_boxes, primary_pred_boxes, iou_threshold=iou_thresholds[0])
    det_iou75 = evaluate_detections(gt_boxes, primary_pred_boxes, iou_threshold=iou_thresholds[1])
    part_metrics = evaluate_part_hierarchy(gt_all_boxes, all_predicted_boxes)
    mask_metrics = evaluate_mask_quality(gt_boxes, primary_pred_boxes)

    # 7. Hardware & Resource Telemetry
    telemetry = measure_hardware_telemetry()
    telemetry["total_execution_seconds"] = round(time.monotonic() - t_start, 2)
    telemetry["images_evaluated"] = len(sorted_image_names)
    telemetry["total_gt_boxes"] = len(gt_boxes)
    telemetry["total_predicted_boxes"] = len(primary_pred_boxes)
    telemetry["split"] = split
    telemetry["target_resolution"] = f"{target_resolution[0]}x{target_resolution[1]}" if target_resolution else "native"
    telemetry["detector_name"] = getattr(detector_engine, "name", detector_engine.__class__.__name__)

    summary = BenchmarkExecutionSummary(
        benchmark_id=benchmark_id,
        detection_iou50=det_iou50,
        detection_iou75=det_iou75,
        part_hierarchy=part_metrics,
        mask_quality=mask_metrics,
        tracking=tracking_metrics,
        multi_view=mv_metrics,
        telemetry=telemetry,
        split=split,
    )

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_p / f"{benchmark_id}.json", "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

    return summary


def fit_calibrator_on_dev_split(
    dataset_dir: str | Path = "datasets/real_room_capture",
    output_dir: str | Path = "artifacts/perception",
    detector_engine: Any = None,
) -> LearnedConfidenceCalibrator:
    """Fits LearnedConfidenceCalibrator on the development split (IMG_0000..IMG_0010) with full provenance."""
    from engine.multiengine.spatial_intelligence.detection_pipeline import (
        DetectionPipeline,
        LearnedConfidenceCalibrator,
        create_default_detector,
        create_default_segmenter,
    )
    from PIL import Image
    import hashlib
    import json

    ds_p = Path(dataset_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    gt_boxes, metadata = load_ground_truth_from_dataset(ds_p, split="dev")
    images_dir = ds_p / "images"

    dev_image_ids = sorted(set(b.image_id for b in gt_boxes))
    gt_by_img: dict[str, list[GroundTruthBox]] = defaultdict(list)
    for b in gt_boxes:
        gt_by_img[b.image_id].append(b)

    if detector_engine is None:
        detector_engine = create_default_detector()
    segmenter_engine = create_default_segmenter()

    pipe = DetectionPipeline(detector=detector_engine, segmenter=segmenter_engine, calibrator=None)
    annotated_classes = sorted(set(g.class_name for g in gt_boxes))

    raw_scores: list[float] = []
    binary_labels: list[int] = []
    class_names: list[str] = []

    print(f"Fitting calibrator on {len(dev_image_ids)} development images...", flush=True)
    for idx_img, img_id in enumerate(dev_image_ids):
        img_file = ds_p / img_id if (ds_p / img_id).is_file() else (images_dir / img_id)
        if not img_file.is_file():
            continue
        print(f"  [{idx_img + 1}/{len(dev_image_ids)}] Extracting calibration samples from {img_id}...", flush=True)
        with Image.open(img_file) as pil_img:
            img_rgb = np.array(pil_img.convert("RGB"))

        pipeline_res = pipe.run_pipeline(img_rgb, image_id=img_id, run_tiling=False, text_prompts=annotated_classes)
        img_gt = gt_by_img.get(img_id, [])

        matched_gt = set()
        for p in pipeline_res.refined_boxes:
            best_iou = 0.0
            best_idx = None
            for idx, g in enumerate(img_gt):
                if idx in matched_gt or g.class_name != p.class_name:
                    continue
                iou = compute_box_iou(p.box_2d, g.box_2d)
                if iou >= 0.50 and iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            label = 0
            if best_idx is not None:
                label = 1
                matched_gt.add(best_idx)

            raw_scores.append(float(p.confidence))
            binary_labels.append(label)
            class_names.append(p.class_name)

    dev_split_hash = hashlib.sha256(json.dumps(dev_image_ids).encode()).hexdigest()[:16]

    calibrator = LearnedConfidenceCalibrator()
    calibrator.fit(
        raw_scores,
        binary_labels,
        class_names=class_names,
        provenance={
            "model_name": getattr(detector_engine, "name", "grounding_dino_swint_ogc"),
            "checkpoint_hash": getattr(detector_engine, "checkpoint_hash", "default"),
            "dataset_split": "dev",
            "dataset_split_hash": dev_split_hash,
            "sample_count": len(raw_scores),
            "dev_images": dev_image_ids,
        },
    )
    cal_file = out_p / "calibrator_dev.json"
    calibrator.save(cal_file)
    print(f"Fitted and saved LearnedConfidenceCalibrator to {cal_file} (T={calibrator.temperature:.4f}, bias={calibrator.bias:.4f}, {len(calibrator.class_params)} classes)")
    return calibrator


def main() -> int:
    """CLI runner for Real-Data Perception Benchmark."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Real-Data Perception Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="datasets/real_room_capture", help="Path to dataset directory")
    parser.add_argument("--out-dir", type=str, default="artifacts/perception", help="Output directory for reports")
    parser.add_argument("--benchmark-id", type=str, default="perception_real_benchmark", help="Benchmark run identifier")
    parser.add_argument("--split", type=str, default="all", choices=["dev", "test", "all"], help="Dataset split to evaluate")
    parser.add_argument("--target-resolution", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"), help="Target resolution (e.g. 1280 960)")
    parser.add_argument("--fit-calibrator", action="store_true", help="Fit confidence calibrator on dev split before benchmark")
    args = parser.parse_args()

    if args.fit_calibrator:
        print(f"Fitting confidence calibrator on development split of {args.dataset}...")
        fit_calibrator_on_dev_split(dataset_dir=args.dataset, output_dir=args.out_dir)

    target_res = tuple(args.target_resolution) if args.target_resolution else None
    print(f"Running Real-Data Perception Benchmark on {args.dataset} (split: {args.split}, resolution: {target_res or 'native'})...")
    summary = run_real_perception_benchmark(
        dataset_dir=args.dataset,
        output_dir=args.out_dir,
        benchmark_id=args.benchmark_id,
        split=args.split,
        target_resolution=target_res,
    )

    print("\n" + "=" * 60)
    print("REAL-DATA PERCEPTION BENCHMARK EXECUTION REPORT")
    print("=" * 60)
    print(f"Benchmark ID: {summary.benchmark_id}")
    print(f"Split: {summary.split}")
    print(f"Detector: {summary.telemetry.get('detector_name')}")
    print(f"Images Evaluated: {summary.telemetry.get('images_evaluated')}")
    print(f"Execution Latency: {summary.telemetry.get('total_execution_seconds')}s")
    print(f"Peak Process Memory: {summary.telemetry.get('peak_rss_mb')} MB")
    print(f"Peak CUDA VRAM: {summary.telemetry.get('peak_cuda_vram_mb')} MB")
    print("-" * 60)
    print("DETECTION PERFORMANCE:")
    f1_50_str = f"{summary.detection_iou50.macro_f1:.4f}" if summary.detection_iou50.macro_f1 is not None else "N/A"
    prec_50_str = f"{summary.detection_iou50.macro_precision:.4f}" if summary.detection_iou50.macro_precision is not None else "N/A"
    rec_50_str = f"{summary.detection_iou50.macro_recall:.4f}" if summary.detection_iou50.macro_recall is not None else "N/A"
    ap_50_str = f"{summary.detection_iou50.average_precision:.4f}" if summary.detection_iou50.average_precision is not None else "N/A"
    print(f"  IoU@0.50: F1={f1_50_str} (Prec={prec_50_str}, Rec={rec_50_str}) | AP50={ap_50_str}")

    f1_75_str = f"{summary.detection_iou75.macro_f1:.4f}" if summary.detection_iou75.macro_f1 is not None else "N/A"
    prec_75_str = f"{summary.detection_iou75.macro_precision:.4f}" if summary.detection_iou75.macro_precision is not None else "N/A"
    rec_75_str = f"{summary.detection_iou75.macro_recall:.4f}" if summary.detection_iou75.macro_recall is not None else "N/A"
    ap_75_str = f"{summary.detection_iou75.average_precision:.4f}" if summary.detection_iou75.average_precision is not None else "N/A"
    print(f"  IoU@0.75: F1={f1_75_str} (Prec={prec_75_str}, Rec={rec_75_str}) | AP75={ap_75_str}")
    print(f"  True Positives (0.50): {summary.detection_iou50.true_positives} / {summary.detection_iou50.total_gt}")
    print("-" * 60)
    print("INSTANCE MASK & BOUNDARY QUALITY:")
    m_iou_str = f"{summary.mask_quality.mean_mask_iou:.4f}" if summary.mask_quality.mean_mask_iou is not None else "N/A"
    b_iou_str = f"{summary.mask_quality.mean_boundary_iou:.4f}" if summary.mask_quality.mean_boundary_iou is not None else "N/A"
    bf_str = f"{summary.mask_quality.mean_boundary_f_score:.4f}" if summary.mask_quality.mean_boundary_f_score is not None else "N/A"
    m_50_str = f"{summary.mask_quality.masks_above_50_iou:.2%}" if summary.mask_quality.masks_above_50_iou is not None else "N/A"
    print(f"  Status: {summary.mask_quality.status} | Mean Mask IoU: {m_iou_str}, Mean Boundary IoU: {b_iou_str}, Boundary F1: {bf_str}")
    print(f"  Evaluated Masks: {summary.mask_quality.evaluated_masks} / {summary.mask_quality.total_gt_masks} | Masks > 50% IoU: {m_50_str}")
    print("-" * 60)
    print("PART HIERARCHY ACCURACY:")
    part_prec = f"{summary.part_hierarchy.precision:.4f}" if summary.part_hierarchy.precision is not None else "N/A"
    part_rec = f"{summary.part_hierarchy.recall:.4f}" if summary.part_hierarchy.recall is not None else "N/A"
    part_f1 = f"{summary.part_hierarchy.f1:.4f}" if summary.part_hierarchy.f1 is not None else "N/A"
    p_det_prec = f"{summary.part_hierarchy.part_detection_precision:.4f}" if summary.part_hierarchy.part_detection_precision is not None else "N/A"
    p_det_rec = f"{summary.part_hierarchy.part_detection_recall:.4f}" if summary.part_hierarchy.part_detection_recall is not None else "N/A"
    p_ass_prec = f"{summary.part_hierarchy.parent_assignment_precision:.4f}" if summary.part_hierarchy.parent_assignment_precision is not None else "N/A"
    p_ass_rec = f"{summary.part_hierarchy.parent_assignment_recall:.4f}" if summary.part_hierarchy.parent_assignment_recall is not None else "N/A"
    print(f"  Status: {summary.part_hierarchy.status} | Links Precision: {part_prec}, Recall: {part_rec}, F1: {part_f1}")
    print(f"  Part Detection: Prec={p_det_prec}, Rec={p_det_rec} | Parent Assignment: Prec={p_ass_prec}, Rec={p_ass_rec}")
    print(f"  Total Links Evaluated: {summary.part_hierarchy.correct_part_links} / {summary.part_hierarchy.total_gt_part_links} | False Parent: {summary.part_hierarchy.false_parent_assignments}, Duplicates: {summary.part_hierarchy.duplicate_component_detections}")
    print("-" * 60)
    print("TEMPORAL IDENTITY TRACKING:")
    trk_idf1 = f"{summary.tracking.idf1:.4f}" if summary.tracking.idf1 is not None else "N/A"
    trk_hota = f"{summary.tracking.hota:.4f}" if summary.tracking.hota is not None else "N/A"
    trk_mota = f"{summary.tracking.mota:.4f}" if summary.tracking.mota is not None else "N/A"
    trk_cov = f"{summary.tracking.track_coverage:.2%}" if summary.tracking.track_coverage is not None else "N/A"
    print(f"  Status: {summary.tracking.status} | IDF1: {trk_idf1}, HOTA: {trk_hota}, MOTA: {trk_mota}, Track Coverage: {trk_cov}")
    print(f"  Frame TP={summary.tracking.frame_tp}, FN={summary.tracking.frame_fn}, FP={summary.tracking.frame_fp} | ID Switches: {summary.tracking.id_switches}, Fragmentations: {summary.tracking.track_fragmentations}")
    print("-" * 60)
    print("MULTI-VIEW ASSOCIATION:")
    mv_prec = f"{summary.multi_view.association_precision:.4f}" if summary.multi_view.association_precision is not None else "N/A"
    mv_rec = f"{summary.multi_view.association_recall:.4f}" if summary.multi_view.association_recall is not None else "N/A"
    mv_purity = f"{summary.multi_view.cluster_purity:.4f}" if summary.multi_view.cluster_purity is not None else "N/A"
    mv_comp = f"{summary.multi_view.identity_completeness:.4f}" if summary.multi_view.identity_completeness is not None else "N/A"
    mv_fmr = f"{summary.multi_view.false_merge_rate:.4f}" if summary.multi_view.false_merge_rate is not None else "N/A"
    mv_fsr = f"{summary.multi_view.false_split_rate:.4f}" if summary.multi_view.false_split_rate is not None else "N/A"
    inlier_mean = f"{summary.multi_view.inlier_mean_reprojection_residual_px:.2f}px" if summary.multi_view.inlier_mean_reprojection_residual_px is not None else "N/A"
    inlier_med = f"{summary.multi_view.inlier_median_reprojection_residual_px:.2f}px" if summary.multi_view.inlier_median_reprojection_residual_px is not None else "N/A"
    inlier_p25 = f"{summary.multi_view.inlier_p25_reprojection_residual_px:.2f}px" if summary.multi_view.inlier_p25_reprojection_residual_px is not None else "N/A"
    inlier_p75 = f"{summary.multi_view.inlier_p75_reprojection_residual_px:.2f}px" if summary.multi_view.inlier_p75_reprojection_residual_px is not None else "N/A"
    inlier_p90 = f"{summary.multi_view.inlier_p90_reprojection_residual_px:.2f}px" if summary.multi_view.inlier_p90_reprojection_residual_px is not None else "N/A"
    inlier_p95 = f"{summary.multi_view.inlier_p95_reprojection_residual_px:.2f}px" if summary.multi_view.inlier_p95_reprojection_residual_px is not None else "N/A"
    print(f"  Status: {summary.multi_view.status} | Association Precision: {mv_prec}, Recall: {mv_rec}")
    print(f"  Cluster Purity: {mv_purity}, Identity Completeness: {mv_comp} | False Merges: {mv_fmr}, False Splits: {mv_fsr}")
    print(f"  Inlier Reprojection Residuals: Mean={inlier_mean}, Med={inlier_med}, P25={inlier_p25}, P75={inlier_p75}, P90={inlier_p90}, P95={inlier_p95}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

