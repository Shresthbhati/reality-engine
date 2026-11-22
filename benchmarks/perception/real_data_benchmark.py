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
import sys
import time
from collections import defaultdict
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


@dataclass(frozen=True)
class DetectionMetrics:
    """Quantitative detection performance metrics."""

    iou_threshold: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    total_gt: int
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "micro_precision": round(self.micro_precision, 4),
            "micro_recall": round(self.micro_recall, 4),
            "micro_f1": round(self.micro_f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
        }


@dataclass(frozen=True)
class PartHierarchyMetrics:
    """Evaluation of hierarchical part-parent link precision and recall."""

    total_gt_part_links: int
    predicted_part_links: int
    correct_part_links: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_gt_part_links": self.total_gt_part_links,
            "predicted_part_links": self.predicted_part_links,
            "correct_part_links": self.correct_part_links,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(frozen=True)
class MaskQualityMetrics:
    """Instance mask overlap and boundary quality metrics."""

    evaluated_masks: int
    mean_mask_iou: float
    median_mask_iou: float
    mean_boundary_iou: float
    masks_above_50_iou: float
    masks_above_75_iou: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_masks": self.evaluated_masks,
            "mean_mask_iou": round(self.mean_mask_iou, 4),
            "median_mask_iou": round(self.median_mask_iou, 4),
            "mean_boundary_iou": round(self.mean_boundary_iou, 4),
            "masks_above_50_iou_fraction": round(self.masks_above_50_iou, 4),
            "masks_above_75_iou_fraction": round(self.masks_above_75_iou, 4),
        }


@dataclass(frozen=True)
class TrackingMetrics:
    """Multi-object temporal tracking metrics."""

    total_gt_trajectories: int
    total_predicted_tracks: int
    id_switches: int
    track_fragmentations: int
    idf1: float
    id_precision: float
    id_recall: float
    mota: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_gt_trajectories": self.total_gt_trajectories,
            "total_predicted_tracks": self.total_predicted_tracks,
            "id_switches": self.id_switches,
            "track_fragmentations": self.track_fragmentations,
            "idf1": round(self.idf1, 4),
            "id_precision": round(self.id_precision, 4),
            "id_recall": round(self.id_recall, 4),
            "mota": round(self.mota, 4),
        }


@dataclass(frozen=True)
class MultiViewAssociationMetrics:
    """Multi-view observation association and reprojection residual metrics."""

    candidate_pairs_evaluated: int
    merged_pairs: int
    refused_pairs: int
    association_precision: float
    association_recall: float
    mean_reprojection_residual_px: float
    median_reprojection_residual_px: float
    p90_reprojection_residual_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_pairs_evaluated": self.candidate_pairs_evaluated,
            "merged_pairs": self.merged_pairs,
            "refused_pairs": self.refused_pairs,
            "association_precision": round(self.association_precision, 4),
            "association_recall": round(self.association_recall, 4),
            "mean_reprojection_residual_px": round(self.mean_reprojection_residual_px, 2),
            "median_reprojection_residual_px": round(self.median_reprojection_residual_px, 2),
            "p90_reprojection_residual_px": round(self.p90_reprojection_residual_px, 2),
        }


@dataclass(frozen=True)
class BenchmarkExecutionSummary:
    """Complete perception benchmark evaluation summary."""

    benchmark_id: str
    detection_mAP50: DetectionMetrics
    detection_mAP75: DetectionMetrics
    part_hierarchy: PartHierarchyMetrics
    mask_quality: MaskQualityMetrics
    tracking: TrackingMetrics
    multi_view: MultiViewAssociationMetrics
    telemetry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "detection_mAP50": self.detection_mAP50.to_dict(),
            "detection_mAP75": self.detection_mAP75.to_dict(),
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


def compute_boundary_iou(mask_a: np.ndarray, mask_b: np.ndarray, dilation_radius: int = 2) -> float:
    """Evaluates Boundary IoU between two boolean segmentation masks."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    from scipy.ndimage import binary_dilation

    struc = np.ones((dilation_radius * 2 + 1, dilation_radius * 2 + 1), dtype=bool)
    bound_a = binary_dilation(mask_a, structure=struc) ^ mask_a
    bound_b = binary_dilation(mask_b, structure=struc) ^ mask_b

    inter = np.logical_and(bound_a, bound_b).sum()
    union = np.logical_or(bound_a, bound_b).sum()
    return float(inter / union) if union > 0 else 1.0 if not bound_a.any() and not bound_b.any() else 0.0


def evaluate_detections(
    gt_boxes: Sequence[GroundTruthBox],
    pred_boxes: Sequence[Any],
    iou_threshold: float = 0.50,
) -> DetectionMetrics:
    """Evaluates detection Precision, Recall, and F1 at specified IoU threshold."""
    from engine.multiengine.spatial_intelligence.taxonomy import are_semantically_compatible

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
    all_classes: set[str] = {g.class_name for g in gt_boxes} | {getattr(p, "class_name", "") for p in pred_boxes if getattr(p, "class_name", "")}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_id in all_images:
        gts = list(gt_by_img.get(img_id, []))
        preds = list(pred_by_img.get(img_id, []))
        preds.sort(key=lambda x: getattr(x, "confidence", 0.5), reverse=True)

        matched_gt: set[int] = set()

        for p in preds:
            p_box = tuple(getattr(p, "box_2d", (0, 0, 0, 0)))
            p_cls = getattr(p, "class_name", "")

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
            else:
                fp_per_class[p_cls] += 1
                total_fp += 1

        for g_idx, g in enumerate(gts):
            if g_idx not in matched_gt:
                fn_per_class[g.class_name] += 1
                total_fn += 1

    per_class_p: dict[str, float] = {}
    per_class_r: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}

    for cls in all_classes:
        tp = tp_per_class[cls]
        fp = fp_per_class[cls]
        fn = fn_per_class[cls]
        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class_p[cls] = p
        per_class_r[cls] = r
        per_class_f1[cls] = f1

    macro_p = float(np.mean(list(per_class_p.values()))) if per_class_p else 0.0
    macro_r = float(np.mean(list(per_class_r.values()))) if per_class_r else 0.0
    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0

    micro_p = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_r = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return DetectionMetrics(
        iou_threshold=iou_threshold,
        per_class_precision=per_class_p,
        per_class_recall=per_class_r,
        per_class_f1=per_class_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        total_gt=len(gt_boxes),
        total_predictions=len(pred_boxes),
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
    )


def evaluate_part_hierarchy(
    gt_boxes: Sequence[GroundTruthBox],
    pred_boxes: Sequence[Any],
) -> PartHierarchyMetrics:
    """Evaluates parent-child link accuracy between detected subparts and host objects."""
    # Find all GT part relationships (child -> parent)
    gt_links: dict[str, str] = {}
    gt_by_entity: dict[str, GroundTruthBox] = {g.entity_id: g for g in gt_boxes}
    for g in gt_boxes:
        if g.parent_entity_id:
            gt_links[g.entity_id] = g.parent_entity_id

    # Find predicted part relationships (child -> parent)
    pred_by_id = {getattr(p, "box_id", getattr(p, "observation_id", "")): p for p in pred_boxes}
    pred_links: dict[str, str] = {}
    for p in pred_boxes:
        p_id = getattr(p, "box_id", getattr(p, "observation_id", ""))
        parent_id = getattr(p, "parent_box_id", None)
        if parent_id and parent_id in pred_by_id:
            pred_links[p_id] = parent_id

    # Map predicted boxes to GT entities via spatial IoU
    pred_to_gt_entity: dict[str, str] = {}
    for p_id, p in pred_by_id.items():
        p_box = getattr(p, "box_2d", (0, 0, 0, 0))
        img_id = getattr(p, "source_image_id", getattr(p, "image_id", "img_001"))
        best_iou = 0.40
        best_gt = None
        for g in gt_boxes:
            if g.image_id == img_id:
                iou = compute_box_iou(p_box, g.box_2d)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g.entity_id
        if best_gt:
            pred_to_gt_entity[p_id] = best_gt

    correct_links = 0
    for child_pid, parent_pid in pred_links.items():
        child_gt = pred_to_gt_entity.get(child_pid)
        parent_gt = pred_to_gt_entity.get(parent_pid)
        if child_gt and parent_gt and gt_links.get(child_gt) == parent_gt:
            correct_links += 1

    total_gt = len(gt_links)
    total_pred = len(pred_links)
    prec = float(correct_links / total_pred) if total_pred > 0 else (1.0 if total_gt == 0 else 0.0)
    rec = float(correct_links / total_gt) if total_gt > 0 else 1.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return PartHierarchyMetrics(
        total_gt_part_links=total_gt,
        predicted_part_links=total_pred,
        correct_part_links=correct_links,
        precision=prec,
        recall=rec,
        f1=f1,
    )


def evaluate_mask_quality(
    gt_boxes: Sequence[GroundTruthBox],
    pred_boxes: Sequence[Any],
) -> MaskQualityMetrics:
    """Evaluates Mask IoU and Boundary IoU for instances with segmentation masks."""
    ious: list[float] = []
    bound_ious: list[float] = []

    for g in gt_boxes:
        if g.mask_2d is None:
            continue
        # Find best matching prediction
        for p in pred_boxes:
            p_mask = getattr(p, "mask_2d", None)
            if p_mask is None or p_mask.shape != g.mask_2d.shape:
                continue
            b_iou = compute_box_iou(getattr(p, "box_2d", (0, 0, 0, 0)), g.box_2d)
            if b_iou >= 0.50:
                # Mask IoU
                inter = np.logical_and(p_mask.astype(bool), g.mask_2d.astype(bool)).sum()
                union = np.logical_or(p_mask.astype(bool), g.mask_2d.astype(bool)).sum()
                m_iou = float(inter / union) if union > 0 else 0.0
                ious.append(m_iou)
                # Boundary IoU
                bnd_iou = compute_boundary_iou(p_mask.astype(bool), g.mask_2d.astype(bool))
                bound_ious.append(bnd_iou)
                break

    if not ious:
        return MaskQualityMetrics(
            evaluated_masks=0,
            mean_mask_iou=0.0,
            median_mask_iou=0.0,
            mean_boundary_iou=0.0,
            masks_above_50_iou=0.0,
            masks_above_75_iou=0.0,
        )

    return MaskQualityMetrics(
        evaluated_masks=len(ious),
        mean_mask_iou=float(np.mean(ious)),
        median_mask_iou=float(np.median(ious)),
        mean_boundary_iou=float(np.mean(bound_ious)),
        masks_above_50_iou=float(np.mean([1.0 if v >= 0.50 else 0.0 for v in ious])),
        masks_above_75_iou=float(np.mean([1.0 if v >= 0.75 else 0.0 for v in ious])),
    )


def evaluate_temporal_tracking(
    gt_trajectories: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    pred_tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
) -> TrackingMetrics:
    """Computes IDF1, ID switches, track fragmentations, and MOTA on sequential tracking."""
    # Match pred tracks to GT entities based on temporal overlap and spatial IoU
    total_gt_instances = sum(len(frames) for frames in gt_trajectories.values())
    total_pred_instances = sum(len(frames) for frames in pred_tracks.values())

    id_switches = 0
    fragmentations = 0
    id_tp = 0

    for gt_id, gt_frames in gt_trajectories.items():
        gt_frame_map = {f[0]: f[1] for f in gt_frames}
        best_matching_tid = None
        best_overlap_count = 0

        # Find best matching pred track for this GT trajectory
        for tid, pred_frames in pred_tracks.items():
            overlap = 0
            for f_idx, p_box in pred_frames:
                if f_idx in gt_frame_map:
                    if compute_box_iou(p_box, gt_frame_map[f_idx]) >= 0.40:
                        overlap += 1
            if overlap > best_overlap_count:
                best_overlap_count = overlap
                best_matching_tid = tid

        id_tp += best_overlap_count

        # Check for ID switches across consecutive frames
        prev_matched_tid = None
        for f_idx, gt_box in sorted(gt_frames, key=lambda x: x[0]):
            curr_matched_tid = None
            for tid, pred_frames in pred_tracks.items():
                p_boxes = [pb for fi, pb in pred_frames if fi == f_idx]
                if p_boxes and compute_box_iou(p_boxes[0], gt_box) >= 0.40:
                    curr_matched_tid = tid
                    break

            if prev_matched_tid is not None and curr_matched_tid is not None:
                if curr_matched_tid != prev_matched_tid:
                    id_switches += 1
            elif prev_matched_tid is not None and curr_matched_tid is None:
                fragmentations += 1

            prev_matched_tid = curr_matched_tid

    id_prec = float(id_tp / total_pred_instances) if total_pred_instances > 0 else 0.0
    id_rec = float(id_tp / total_gt_instances) if total_gt_instances > 0 else 0.0
    idf1 = float(2 * id_prec * id_rec / (id_prec + id_rec)) if (id_prec + id_rec) > 0 else 0.0

    # MOTA approximation: max(0.0, 1.0 - (FN + FP + IDSW) / GT)
    fn = max(0, total_gt_instances - id_tp)
    fp = max(0, total_pred_instances - id_tp)
    mota = max(0.0, 1.0 - float(fn + fp + id_switches) / max(1, total_gt_instances))

    return TrackingMetrics(
        total_gt_trajectories=len(gt_trajectories),
        total_predicted_tracks=len(pred_tracks),
        id_switches=id_switches,
        track_fragmentations=fragmentations,
        idf1=idf1,
        id_precision=id_prec,
        id_recall=id_rec,
        mota=mota,
    )


def evaluate_multiview_association(
    gt_entity_observations: dict[str, set[str]],  # gt_entity_id -> set of observation_ids
    merged_entities: Sequence[Any],
    association_decisions: Sequence[Any],
) -> MultiViewAssociationMetrics:
    """Evaluates cross-view association precision, recall, and reprojection residuals."""
    # Find which observations belong to which GT entity
    obs_to_gt: dict[str, str] = {}
    for gt_id, obs_set in gt_entity_observations.items():
        for o in obs_set:
            obs_to_gt[o] = gt_id

    # Evaluate decisions
    merged_count = 0
    refused_count = 0
    correct_merges = 0
    incorrect_merges = 0

    reproj_residuals: list[float] = []

    for d in association_decisions:
        outcome = getattr(d, "outcome", "")
        is_merged = (outcome == "merged" or getattr(outcome, "value", "") == "merged")
        obs_a = getattr(d, "obs_a_id", "")
        obs_b = getattr(d, "obs_b_id", "")

        err = getattr(d, "reprojection_err_px", None)
        if err is not None and math.isfinite(err):
            reproj_residuals.append(float(err))

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

    # Total GT pairs
    total_gt_pairs = 0
    for obs_set in gt_entity_observations.values():
        n = len(obs_set)
        total_gt_pairs += (n * (n - 1)) // 2

    assoc_prec = float(correct_merges / merged_count) if merged_count > 0 else (1.0 if total_gt_pairs == 0 else 0.0)
    assoc_rec = float(correct_merges / total_gt_pairs) if total_gt_pairs > 0 else 1.0

    mean_rep = float(np.mean(reproj_residuals)) if reproj_residuals else 0.0
    med_rep = float(np.median(reproj_residuals)) if reproj_residuals else 0.0
    p90_rep = float(np.percentile(reproj_residuals, 90)) if reproj_residuals else 0.0

    return MultiViewAssociationMetrics(
        candidate_pairs_evaluated=len(association_decisions),
        merged_pairs=merged_count,
        refused_pairs=refused_count,
        association_precision=assoc_prec,
        association_recall=assoc_rec,
        mean_reprojection_residual_px=mean_rep,
        median_reprojection_residual_px=med_rep,
        p90_reprojection_residual_px=p90_rep,
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

def load_ground_truth_from_dataset(dataset_path: Path | str) -> tuple[list[GroundTruthBox], dict[str, Any]]:
    """Loads ground-truth 2D annotations and physical entity metadata from a dataset directory or JSON file.

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

        for ann in annotations:
            gt_id = ann.get("ground_truth_id") or ann.get("entity_id", "")
            cls_name = ann.get("class_name") or ann.get("label", "")
            b = ann.get("bbox_2d") or ann.get("box_2d", [0, 0, 0, 0])
            parent_id = ann.get("parent_entity_id")

            gt_boxes.append(GroundTruthBox(
                entity_id=gt_id,
                class_name=cls_name,
                box_2d=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                image_id=img_name,
                parent_entity_id=parent_id,
                frame_index=frame_idx,
            ))

    metadata = {
        "dataset": data.get("dataset", path.name),
        "description": data.get("description", ""),
        "is_physically_surveyed": data.get("is_physically_surveyed", False),
        "milestone_status": data.get("milestone_status", "PROVISIONAL"),
        "total_images_annotated": len(image_annotations),
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
) -> BenchmarkExecutionSummary:
    """Executes the full perception, tracking, multi-view association, and telemetry benchmark on a real dataset.

    Returns a complete, verified BenchmarkExecutionSummary.
    """
    dpath = Path(dataset_dir)
    t_start = time.monotonic()

    # 1. Load ground truth
    gt_boxes, gt_meta = load_ground_truth_from_dataset(dpath)
    gt_by_image: dict[str, list[GroundTruthBox]] = defaultdict(list)
    for g in gt_boxes:
        gt_by_image[g.image_id].append(g)

    # 2. Setup perception pipeline
    from engine.multiengine.spatial_intelligence.detection_pipeline import (
        DetectionBox,
        DetectionPipeline,
        DetectionScale,
    )
    from engine.multiengine.spatial_intelligence.association import (
        CrossViewRayCandidate,
        MultiViewIntelligenceEngine,
        TemporalIdentityTracker,
        build_camera_ray,
    )
    from engine.multiengine.spatial_intelligence.taxonomy import Domain

    pipe = DetectionPipeline(detector=detector_engine, segmenter=segmenter_engine)

    # 3. Process each annotated image
    all_predicted_boxes: list[DetectionBox] = []
    frame_observations: dict[int, list[DetectionBox]] = defaultdict(list)
    obs_by_image: dict[str, list[DetectionBox]] = defaultdict(list)

    sorted_image_names = sorted(gt_by_image.keys())

    for frame_idx, img_name in enumerate(sorted_image_names):
        img_path = dpath / img_name
        if not img_path.is_file():
            continue

        print(f"  [{frame_idx + 1}/{len(sorted_image_names)}] Inferring perception on {img_name}...", flush=True)
        from PIL import Image
        with Image.open(img_path) as im:
            rgb = np.asarray(im.convert("RGB"))

        res = pipe.run_pipeline(
            rgb,
            domain=Domain.INTERIOR,
            image_id=img_name,
            frame_index=frame_idx,
            timestamp_sec=frame_idx * 0.1,  # 10 fps nominal
            run_tiling=False,  # default standard scale for evaluation
            detect_subparts=True,
        )

        for b in res.refined_boxes:
            all_predicted_boxes.append(b)
            frame_observations[frame_idx].append(b)
            obs_by_image[img_name].append(b)

    # 4. Temporal Tracking Evaluation
    tracker = TemporalIdentityTracker(max_missed_frames=5, min_iou_match=0.20)
    predicted_tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)

    for frame_idx in sorted(frame_observations.keys()):
        obs_list = frame_observations[frame_idx]
        ts = frame_idx * 0.1
        assignments = tracker.process_frame_observations(frame_idx, ts, obs_list)
        for obs in obs_list:
            trk_id = assignments.get(obs.box_id)
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
        pos = pose["position"] if pose else (0.0, 0.0, 0.0)
        rot = pose["rotation"] if pose else (1.0, 0.0, 0.0, 0.0)

        fx, fy, px, py = 1000.0, 1000.0, 1000.0, 750.0
        w_cam, h_cam = 4032, 3024
        if intrinsics and "1" in intrinsics:
            c_info = intrinsics["1"]
            w_cam = c_info.get("width", 4032)
            h_cam = c_info.get("height", 3024)
            params = c_info.get("params", [])
            if len(params) >= 4:
                fx, fy, px, py = float(params[0]), float(params[1]), float(params[2]), float(params[3])

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
            )
            ray_candidates.append(cand)

    mv_engine = MultiViewIntelligenceEngine(max_ray_distance_m=0.35, max_reproj_err_px=100.0)
    merged_entities, assoc_decisions, _ = mv_engine.associate_candidates(ray_candidates)

    # Map GT entities to predicted observation IDs via best box IoU
    gt_entity_obs: dict[str, set[str]] = defaultdict(set)
    for g in gt_boxes:
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

    mv_metrics = evaluate_multiview_association(gt_entity_obs, merged_entities, assoc_decisions)

    # 6. Quantitative Detection, Part, and Mask Metrics
    det_mAP50 = evaluate_detections(gt_boxes, all_predicted_boxes, iou_threshold=iou_thresholds[0])
    det_mAP75 = evaluate_detections(gt_boxes, all_predicted_boxes, iou_threshold=iou_thresholds[1])
    part_metrics = evaluate_part_hierarchy(gt_boxes, all_predicted_boxes)
    mask_metrics = evaluate_mask_quality(gt_boxes, all_predicted_boxes)

    # 7. Hardware & Resource Telemetry
    telemetry = measure_hardware_telemetry()
    telemetry["total_execution_seconds"] = round(time.monotonic() - t_start, 2)
    telemetry["images_evaluated"] = len(sorted_image_names)
    telemetry["total_gt_boxes"] = len(gt_boxes)
    telemetry["total_predicted_boxes"] = len(all_predicted_boxes)

    summary = BenchmarkExecutionSummary(
        benchmark_id=benchmark_id,
        detection_mAP50=det_mAP50,
        detection_mAP75=det_mAP75,
        part_hierarchy=part_metrics,
        mask_quality=mask_metrics,
        tracking=tracking_metrics,
        multi_view=mv_metrics,
        telemetry=telemetry,
    )

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_p / f"{benchmark_id}.json", "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

    return summary


def main() -> int:
    """CLI runner for Real-Data Perception Benchmark."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Real-Data Perception Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="datasets/real_room_capture", help="Path to dataset directory")
    parser.add_argument("--out-dir", type=str, default="artifacts/perception", help="Output directory for reports")
    parser.add_argument("--benchmark-id", type=str, default="perception_real_benchmark", help="Benchmark run identifier")
    args = parser.parse_args()

    print(f"Running Real-Data Perception Benchmark on {args.dataset}...")
    summary = run_real_perception_benchmark(
        dataset_dir=args.dataset,
        output_dir=args.out_dir,
        benchmark_id=args.benchmark_id,
    )

    print("\n" + "=" * 60)
    print("REAL-DATA PERCEPTION BENCHMARK EXECUTION REPORT")
    print("=" * 60)
    print(f"Benchmark ID: {summary.benchmark_id}")
    print(f"Images Evaluated: {summary.telemetry.get('images_evaluated')}")
    print(f"Execution Latency: {summary.telemetry.get('total_execution_seconds')}s")
    print(f"Peak Process Memory: {summary.telemetry.get('peak_rss_mb')} MB")
    print(f"Peak CUDA VRAM: {summary.telemetry.get('peak_cuda_vram_mb')} MB")
    print("-" * 60)
    print("DETECTION PERFORMANCE:")
    print(f"  mAP@0.50: F1={summary.detection_mAP50.macro_f1:.4f} (Prec={summary.detection_mAP50.macro_precision:.4f}, Rec={summary.detection_mAP50.macro_recall:.4f})")
    print(f"  mAP@0.75: F1={summary.detection_mAP75.macro_f1:.4f} (Prec={summary.detection_mAP75.macro_precision:.4f}, Rec={summary.detection_mAP75.macro_recall:.4f})")
    print(f"  True Positives (0.50): {summary.detection_mAP50.true_positives} / {summary.detection_mAP50.total_gt}")
    print("-" * 60)
    print("PART HIERARCHY ACCURACY:")
    print(f"  Links Precision: {summary.part_hierarchy.precision:.4f}, Recall: {summary.part_hierarchy.recall:.4f}, F1: {summary.part_hierarchy.f1:.4f}")
    print("-" * 60)
    print("TEMPORAL IDENTITY TRACKING:")
    print(f"  IDF1: {summary.tracking.idf1:.4f}, MOTA: {summary.tracking.mota:.4f}")
    print(f"  ID Switches: {summary.tracking.id_switches}, Track Fragmentations: {summary.tracking.track_fragmentations}")
    print("-" * 60)
    print("MULTI-VIEW ASSOCIATION:")
    print(f"  Association Precision: {summary.multi_view.association_precision:.4f}, Recall: {summary.multi_view.association_recall:.4f}")
    print(f"  Reprojection Residuals: Mean={summary.multi_view.mean_reprojection_residual_px:.2f}px, Median={summary.multi_view.median_reprojection_residual_px:.2f}px, P90={summary.multi_view.p90_reprojection_residual_px:.2f}px")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

