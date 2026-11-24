"""Component-Level Perception Benchmark (Sprint 8).

Evaluates fine-grained component recognition across architectural, electrical,
furniture, plumbing/HVAC, and exterior structural elements.
Distinguishes parent-object detection from component detection,
measures parent-child assignment accuracy, and reports sample counts,
support, and explicit UNMEASURED status for classes without ground truth.

Strict rule: Never manufactures ground truth from model predictions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from engine.multiengine.spatial_intelligence.detection_pipeline import (
    GENUINE_STRUCTURAL_PARTS,
    DetectionBox,
    DetectionPipeline,
    DetectionScale,
    create_default_detector,
    create_default_segmenter,
    load_default_production_calibrator,
)
from engine.multiengine.spatial_intelligence.taxonomy import (
    TaxonomyClass,
    are_semantically_compatible,
    resolve_taxonomy_class,
)


class ComponentFamily(str, Enum):
    """Canonical component families required by Sprint 8 & 9."""

    DOORS_AND_WINDOWS = "doors_and_windows"
    ELECTRICAL = "electrical"
    FURNITURE = "furniture"
    PLUMBING_AND_HVAC = "plumbing_and_hvac"
    BUILDING_AND_EXTERIOR = "building_and_exterior"


class EvaluationMode(str, Enum):
    """Evaluation modes for benchmark integrity (Sprint 9 P0)."""

    FRAME_LEVEL = "frame_level"
    PHYSICAL_IDENTITY = "physical_identity"


def is_class_compatible(pred_class: str, gt_class: str) -> bool:
    """Checks whether prediction class is compatible with GT class or known synonym."""
    if pred_class == gt_class:
        return True
    pred_clean = pred_class.lower().strip()
    gt_clean = gt_class.lower().strip()
    if pred_clean == gt_clean:
        return True

    # Synonyms mapping
    synonyms = {
        "door_knob": {"door_handle", "door_knob", "door_lever"},
        "door_handle": {"door_handle", "door_knob", "door_lever"},
        "door_lever": {"door_handle", "door_knob", "door_lever"},
        "window_glass": {"window_glazing", "window_glass"},
        "window_glazing": {"window_glazing", "window_glass"},
        "socket": {"socket", "socket_plate", "outlet_plate", "plug_socket"},
        "socket_plate": {"socket", "socket_plate", "outlet_plate", "plug_socket"},
        "outlet_plate": {"socket", "socket_plate", "outlet_plate", "plug_socket"},
        "plug_socket": {"socket", "socket_plate", "outlet_plate", "plug_socket"},
        "switch": {"switch", "switch_plate", "control_switch"},
        "switch_plate": {"switch", "switch_plate", "control_switch"},
        "control_switch": {"switch", "switch_plate", "control_switch"},
        "tap": {"tap", "faucet"},
        "faucet": {"tap", "faucet"},
        "table_leg": {"table_leg", "furniture_leg", "table_support"},
        "chair_back": {"chair_back", "chair_backrest"},
        "chair_backrest": {"chair_back", "chair_backrest"},
    }
    if gt_clean in synonyms and pred_clean in synonyms[gt_clean]:
        return True

    # Fallback to semantic compatibility
    return are_semantically_compatible(pred_clean, gt_clean)


COMPONENT_FAMILY_MEMBERS: dict[ComponentFamily, tuple[str, ...]] = {
    ComponentFamily.DOORS_AND_WINDOWS: (
        "door", "door_leaf", "door_frame", "door_panel", "door_glazing",
        "door_handle", "door_knob", "door_lever", "door_hinge", "door_lock",
        "door_latch", "door_threshold",
        "window", "window_frame", "window_sash", "window_glazing", "window_glass",
        "window_mullion", "window_sill", "window_handle", "window_hinge", "window_shutter",
    ),
    ComponentFamily.ELECTRICAL: (
        "outlet_plate", "socket_plate", "socket", "plug_socket",
        "switch", "switch_plate", "control_switch", "plug_head",
        "cable", "conduit", "distribution_board",
        "light_fixture", "ceiling_light", "lamp", "diffuser", "light_housing",
    ),
    ComponentFamily.FURNITURE: (
        "table", "table_top", "table_leg", "furniture_leg", "table_support",
        "desk", "drawer", "drawer_handle",
        "cabinet", "cabinet_carcass", "cabinet_door", "cabinet_handle", "shelf",
        "wardrobe",
        "chair", "chair_seat", "chair_back", "chair_backrest", "chair_armrest", "chair_leg",
    ),
    ComponentFamily.PLUMBING_AND_HVAC: (
        "sink", "basin", "tap", "faucet", "faucet_handle", "spout",
        "pipe", "valve", "shower_fixture", "drain", "vent", "radiator",
    ),
    ComponentFamily.BUILDING_AND_EXTERIOR: (
        "wall", "floor", "ceiling", "column", "beam",
        "stairs", "step", "railing", "balustrade",
        "curb", "footpath", "road_edge", "road",
        "lamppost", "sign", "signboard",
        "tree", "trunk", "branch",
    ),
}

# Invert family lookup
CLASS_TO_FAMILY: dict[str, ComponentFamily] = {}
for fam, members in COMPONENT_FAMILY_MEMBERS.items():
    for m in members:
        CLASS_TO_FAMILY[m] = fam


@dataclass
class GroundTruthComponent:
    """Traceable ground truth annotation for a component or parent object."""

    entity_id: str
    class_name: str
    image_id: str
    box_2d: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    parent_entity_id: str | None = None
    physical_identity_id: str | None = None
    visibility: str = "visible"  # "visible", "occluded", "truncated", "absent"
    occlusion_level: str = "none"  # "none", "partial", "heavy"
    mask_polygon: tuple[tuple[float, float], ...] | None = None
    dataset_source: str = "unknown"  # "synthetic", "public_real", "independent_captured"
    is_component: bool = False

    @property
    def frame_instance_key(self) -> tuple[str, str]:
        """Unique key for frame-level evaluation: (image_id, entity_id)."""
        return (self.image_id, self.entity_id)

    @property
    def physical_key(self) -> str:
        """Physical identity across sequential frames/viewpoints."""
        return self.physical_identity_id if self.physical_identity_id else self.entity_id

    @property
    def area(self) -> float:
        return max(0.0, self.box_2d[2] - self.box_2d[0]) * max(0.0, self.box_2d[3] - self.box_2d[1])


@dataclass
class ClassComponentMetrics:
    """Rigorous evaluation metrics for a single class with sample counts."""

    class_name: str
    family: str
    is_component: bool
    status: str  # "measured" or "UNMEASURED"
    gt_count: int = 0
    gt_visible_count: int = 0
    gt_occluded_count: int = 0
    gt_truncated_count: int = 0
    gt_absent_count: int = 0
    pred_count: int = 0
    true_positives_50: int = 0
    false_positives_50: int = 0
    false_negatives_50: int = 0
    true_positives_75: int = 0
    false_positives_75: int = 0
    false_negatives_75: int = 0
    precision_50: float | None = None
    recall_50: float | None = None
    f1_50: float | None = None
    ap_50: float | None = None
    ap50_101: float | None = None
    ap50_11: float | None = None
    precision_75: float | None = None
    recall_75: float | None = None
    f1_75: float | None = None
    ap_75: float | None = None
    ap75_101: float | None = None
    ap75_11: float | None = None
    ap_coco_50_95: float | None = None
    mean_mask_iou: float | None = None
    mean_boundary_iou: float | None = None
    boundary_f1: float | None = None
    # Parent-child metrics if component
    parent_assignment_tp: int = 0
    parent_assignment_fp: int = 0
    parent_assignment_fn: int = 0
    parent_assignment_precision: float | None = None
    parent_assignment_recall: float | None = None
    parent_assignment_f1: float | None = None
    unmeasured_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "family": self.family,
            "is_component": self.is_component,
            "status": self.status,
            "sample_counts": {
                "gt_total": self.gt_count,
                "gt_visible": self.gt_visible_count,
                "gt_occluded": self.gt_occluded_count,
                "gt_truncated": self.gt_truncated_count,
                "gt_absent": self.gt_absent_count,
                "pred_total": self.pred_count,
                "tp50": self.true_positives_50,
                "fp50": self.false_positives_50,
                "fn50": self.false_negatives_50,
                "tp75": self.true_positives_75,
                "fp75": self.false_positives_75,
                "fn75": self.false_negatives_75,
            },
            "metrics_iou50": {
                "precision": round(self.precision_50, 4) if self.precision_50 is not None else None,
                "recall": round(self.recall_50, 4) if self.recall_50 is not None else None,
                "f1": round(self.f1_50, 4) if self.f1_50 is not None else None,
                "ap50_101": round(self.ap50_101, 4) if self.ap50_101 is not None else (round(self.ap_50, 4) if self.ap_50 is not None else None),
                "ap50_11": round(self.ap50_11, 4) if self.ap50_11 is not None else None,
                "ap50": round(self.ap_50, 4) if self.ap_50 is not None else None,
            },
            "metrics_iou75": {
                "precision": round(self.precision_75, 4) if self.precision_75 is not None else None,
                "recall": round(self.recall_75, 4) if self.recall_75 is not None else None,
                "f1": round(self.f1_75, 4) if self.f1_75 is not None else None,
                "ap75_101": round(self.ap75_101, 4) if self.ap75_101 is not None else (round(self.ap_75, 4) if self.ap_75 is not None else None),
                "ap75_11": round(self.ap75_11, 4) if self.ap75_11 is not None else None,
                "ap75": round(self.ap_75, 4) if self.ap_75 is not None else None,
            },
            "metrics_coco": {
                "map_50_95": round(self.ap_coco_50_95, 4) if self.ap_coco_50_95 is not None else None,
            },
            "mask_quality": {
                "mean_mask_iou": round(self.mean_mask_iou, 4) if self.mean_mask_iou is not None else None,
                "mean_boundary_iou": round(self.mean_boundary_iou, 4) if self.mean_boundary_iou is not None else None,
                "boundary_f1": round(self.boundary_f1, 4) if self.boundary_f1 is not None else None,
            },
            "parent_assignment": {
                "tp": self.parent_assignment_tp,
                "fp": self.parent_assignment_fp,
                "fn": self.parent_assignment_fn,
                "precision": round(self.parent_assignment_precision, 4) if self.parent_assignment_precision is not None else None,
                "recall": round(self.parent_assignment_recall, 4) if self.parent_assignment_recall is not None else None,
                "f1": round(self.parent_assignment_f1, 4) if self.parent_assignment_f1 is not None else None,
            } if self.is_component else None,
            "unmeasured_reason": self.unmeasured_reason,
        }


@dataclass
class ComponentBenchmarkReport:
    """Comprehensive component-level benchmark audit report."""

    benchmark_id: str
    dataset_name: str
    dataset_type: str  # "synthetic", "public_real", "independent_captured"
    split: str
    total_images: int
    evaluated_classes_count: int
    measured_classes_count: int
    unmeasured_classes_count: int
    parent_detection_macro_f1: float | None
    parent_detection_ap50: float | None
    component_detection_macro_f1: float | None
    component_detection_ap50: float | None
    component_parent_link_macro_f1: float | None
    results_by_class: dict[str, ClassComponentMetrics]
    results_by_family: dict[str, dict[str, Any]]
    telemetry: dict[str, Any] = field(default_factory=dict)
    mode: str = "frame_level"
    parent_detection_ap50_101: float | None = None
    parent_detection_ap50_11: float | None = None
    component_detection_ap50_101: float | None = None
    component_detection_ap50_11: float | None = None
    component_parent_link_precision: float | None = None
    component_parent_link_recall: float | None = None
    physical_identity_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type,
            "split": self.split,
            "mode": self.mode,
            "total_images": self.total_images,
            "summary": {
                "evaluated_classes": self.evaluated_classes_count,
                "measured_classes": self.measured_classes_count,
                "unmeasured_classes": self.unmeasured_classes_count,
                "parent_macro_f1": round(self.parent_detection_macro_f1, 4) if self.parent_detection_macro_f1 is not None else None,
                "parent_ap50_101": round(self.parent_detection_ap50_101, 4) if self.parent_detection_ap50_101 is not None else (round(self.parent_detection_ap50, 4) if self.parent_detection_ap50 is not None else None),
                "parent_ap50_11": round(self.parent_detection_ap50_11, 4) if self.parent_detection_ap50_11 is not None else None,
                "parent_ap50": round(self.parent_detection_ap50, 4) if self.parent_detection_ap50 is not None else None,
                "component_macro_f1": round(self.component_detection_macro_f1, 4) if self.component_detection_macro_f1 is not None else None,
                "component_ap50_101": round(self.component_detection_ap50_101, 4) if self.component_detection_ap50_101 is not None else (round(self.component_detection_ap50, 4) if self.component_detection_ap50 is not None else None),
                "component_ap50_11": round(self.component_detection_ap50_11, 4) if self.component_detection_ap50_11 is not None else None,
                "component_ap50": round(self.component_detection_ap50, 4) if self.component_detection_ap50 is not None else None,
                "component_parent_link_macro_f1": round(self.component_parent_link_macro_f1, 4) if self.component_parent_link_macro_f1 is not None else None,
                "component_parent_link_precision": round(self.component_parent_link_precision, 4) if self.component_parent_link_precision is not None else None,
                "component_parent_link_recall": round(self.component_parent_link_recall, 4) if self.component_parent_link_recall is not None else None,
            },
            "results_by_family": self.results_by_family,
            "results_by_class": {k: v.to_dict() for k, v in self.results_by_class.items()},
            "physical_identity_metrics": self.physical_identity_metrics,
            "telemetry": self.telemetry,
        }


def compute_box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def compute_voc_ap_11(
    precisions: Sequence[float],
    recalls: Sequence[float],
) -> float:
    """Computes legacy 11-point interpolated Average Precision (PASCAL VOC style)."""
    if not precisions or not recalls:
        return 0.0
    rec_arr = np.array(recalls)
    prec_arr = np.array(precisions)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        p_candidates = prec_arr[rec_arr >= t]
        p_interp = float(np.max(p_candidates)) if len(p_candidates) > 0 else 0.0
        ap += p_interp / 11.0
    return float(ap)


def compute_coco_ap_101(
    precisions: Sequence[float],
    recalls: Sequence[float],
) -> float:
    """Computes standard 101-point interpolated Average Precision (COCO style)."""
    if not precisions or not recalls:
        return 0.0
    rec_arr = np.array(recalls)
    prec_arr = np.array(precisions)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        p_candidates = prec_arr[rec_arr >= t]
        p_interp = float(np.max(p_candidates)) if len(p_candidates) > 0 else 0.0
        ap += p_interp / 101.0
    return float(ap)


def compute_average_precision(
    precisions: Sequence[float],
    recalls: Sequence[float],
) -> float:
    """Default AP computation: uses 101-point COCO interpolation."""
    return compute_coco_ap_101(precisions, recalls)


class ComponentBenchmarkEvaluator:
    """Evaluates component-level recognition performance on ground-truth annotated datasets."""

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        ground_truth: Sequence[GroundTruthComponent],
        predictions: Sequence[DetectionBox],
        dataset_name: str = "dataset",
        dataset_type: str = "synthetic",
        eval_split: str = "test",
        split: str | None = None,
        benchmark_id: str = "component_benchmark",
        mode: EvaluationMode | str = EvaluationMode.FRAME_LEVEL,
        telemetry: dict[str, Any] | None = None,
    ) -> ComponentBenchmarkReport:
        """Alias for evaluate_dataset."""
        actual_split = split if split is not None else eval_split
        return self.evaluate_dataset(
            ground_truth=ground_truth,
            predictions=predictions,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            split=actual_split,
            benchmark_id=benchmark_id,
            mode=mode,
            telemetry=telemetry,
        )

    def evaluate_dataset(
        self,
        ground_truth: Sequence[GroundTruthComponent],
        predictions: Sequence[DetectionBox],
        dataset_name: str = "dataset",
        dataset_type: str = "synthetic",
        split: str = "test",
        benchmark_id: str = "component_benchmark",
        mode: EvaluationMode | str = EvaluationMode.FRAME_LEVEL,
        telemetry: dict[str, Any] | None = None,
    ) -> ComponentBenchmarkReport:
        """Evaluates ground truth vs predictions across all canonical classes and component families."""
        eval_mode = EvaluationMode(mode) if isinstance(mode, str) else mode

        # 1. Index GT and Predictions
        gt_by_class: dict[str, list[GroundTruthComponent]] = defaultdict(list)
        gt_by_img: dict[str, list[GroundTruthComponent]] = defaultdict(list)
        all_gt_images: set[str] = set()

        for g in ground_truth:
            gt_by_class[g.class_name].append(g)
            gt_by_img[g.image_id].append(g)
            all_gt_images.add(g.image_id)

        pred_by_class: dict[str, list[DetectionBox]] = defaultdict(list)
        pred_by_img: dict[str, list[DetectionBox]] = defaultdict(list)
        for p in predictions:
            pred_by_class[p.class_name].append(p)
            img_id = getattr(p, "source_image_id", "img")
            pred_by_img[img_id].append(p)
            all_gt_images.add(img_id)

        # Collect all classes across taxonomy families
        all_eval_classes: list[str] = []
        for fam_classes in COMPONENT_FAMILY_MEMBERS.values():
            all_eval_classes.extend(fam_classes)
        all_eval_classes = sorted(set(all_eval_classes))

        results_by_class: dict[str, ClassComponentMetrics] = {}

        parent_f1s: list[float] = []
        parent_aps_101: list[float] = []
        parent_aps_11: list[float] = []
        comp_f1s: list[float] = []
        comp_aps_101: list[float] = []
        comp_aps_11: list[float] = []
        comp_link_f1s: list[float] = []
        comp_link_precs: list[float] = []
        comp_link_recs: list[float] = []

        measured_count = 0
        unmeasured_count = 0

        # Evaluate each class
        for cname in all_eval_classes:
            fam = CLASS_TO_FAMILY.get(cname, ComponentFamily.BUILDING_AND_EXTERIOR).value
            tax_cls = resolve_taxonomy_class(cname)
            is_comp = (tax_cls.parent_class is not None) or (cname not in ("door", "window", "table", "desk", "chair", "cabinet", "wall", "floor", "ceiling"))

            gt_items = gt_by_class.get(cname, [])
            pred_items = pred_by_class.get(cname, [])

            if not gt_items:
                # UNMEASURED: no ground truth in this dataset
                unmeasured_count += 1
                n_preds = len(pred_items)
                results_by_class[cname] = ClassComponentMetrics(
                    class_name=cname,
                    family=fam,
                    is_component=is_comp,
                    status="UNMEASURED",
                    gt_count=0,
                    pred_count=n_preds,
                    false_positives_50=n_preds,
                    false_positives_75=n_preds,
                    unmeasured_reason="NO_GROUND_TRUTH_IN_DATASET",
                )
                continue

            # Measured class
            measured_count += 1
            n_gt = len(gt_items)
            n_vis = sum(1 for g in gt_items if g.visibility == "visible")
            n_occ = sum(1 for g in gt_items if g.occlusion_level in ("partial", "heavy"))
            n_trunc = sum(1 for g in gt_items if g.visibility == "truncated")
            n_absent = sum(1 for g in gt_items if g.visibility == "absent")
            n_pred = len(pred_items)

            # Match predictions to GT at IoU 0.50 and 0.75
            tp50, fp50, fn50, prec50, rec50, f1_50, ap50_101, ap50_11 = self._evaluate_box_matches(
                gt_items, pred_items, iou_thresh=0.50, mode=eval_mode
            )
            tp75, fp75, fn75, prec75, rec75, f1_75, ap75_101, ap75_11 = self._evaluate_box_matches(
                gt_items, pred_items, iou_thresh=0.75, mode=eval_mode
            )

            # COCO 10-threshold mAP [0.50:0.05:0.95]
            coco_aps = []
            for iou_t in np.linspace(0.50, 0.95, 10):
                _, _, _, _, _, _, ap_t, _ = self._evaluate_box_matches(
                    gt_items, pred_items, iou_thresh=float(iou_t), mode=eval_mode
                )
                coco_aps.append(ap_t)
            ap_coco_50_95 = float(np.mean(coco_aps)) if coco_aps else 0.0

            # Evaluate parent-child link accuracy if component
            link_tp, link_fp, link_fn = 0, 0, 0
            link_prec, link_rec, link_f1 = None, None, None

            if is_comp:
                link_tp, link_fp, link_fn = self._evaluate_parent_child_links(
                    gt_items, pred_items, ground_truth, predictions, mode=eval_mode
                )
                if (link_tp + link_fp) > 0:
                    link_prec = float(link_tp) / float(link_tp + link_fp)
                    comp_link_precs.append(link_prec)
                if (link_tp + link_fn) > 0:
                    link_rec = float(link_tp) / float(link_tp + link_fn)
                    comp_link_recs.append(link_rec)
                if link_prec is not None and link_rec is not None and (link_prec + link_rec) > 0:
                    link_f1 = 2.0 * link_prec * link_rec / (link_prec + link_rec)
                if link_f1 is not None:
                    comp_link_f1s.append(link_f1)

            results_by_class[cname] = ClassComponentMetrics(
                class_name=cname,
                family=fam,
                is_component=is_comp,
                status="measured",
                gt_count=n_gt,
                gt_visible_count=n_vis,
                gt_occluded_count=n_occ,
                gt_truncated_count=n_trunc,
                gt_absent_count=n_absent,
                pred_count=n_pred,
                true_positives_50=tp50,
                false_positives_50=fp50,
                false_negatives_50=fn50,
                true_positives_75=tp75,
                false_positives_75=fp75,
                false_negatives_75=fn75,
                precision_50=prec50,
                recall_50=rec50,
                f1_50=f1_50,
                ap_50=ap50_101,
                ap50_101=ap50_101,
                ap50_11=ap50_11,
                precision_75=prec75,
                recall_75=rec75,
                f1_75=f1_75,
                ap_75=ap75_101,
                ap75_101=ap75_101,
                ap75_11=ap75_11,
                ap_coco_50_95=ap_coco_50_95,
                parent_assignment_tp=link_tp,
                parent_assignment_fp=link_fp,
                parent_assignment_fn=link_fn,
                parent_assignment_precision=link_prec,
                parent_assignment_recall=link_rec,
                parent_assignment_f1=link_f1,
            )

            if is_comp:
                if f1_50 is not None:
                    comp_f1s.append(f1_50)
                if ap50_101 is not None:
                    comp_aps_101.append(ap50_101)
                if ap50_11 is not None:
                    comp_aps_11.append(ap50_11)
            else:
                if f1_50 is not None:
                    parent_f1s.append(f1_50)
                if ap50_101 is not None:
                    parent_aps_101.append(ap50_101)
                if ap50_11 is not None:
                    parent_aps_11.append(ap50_11)

        # Aggregate by family
        results_by_family: dict[str, dict[str, Any]] = {}
        for fam in ComponentFamily:
            fam_members = COMPONENT_FAMILY_MEMBERS[fam]
            fam_res = [results_by_class[c] for c in fam_members if c in results_by_class]
            measured_fam = [r for r in fam_res if r.status == "measured"]
            f1s = [r.f1_50 for r in measured_fam if r.f1_50 is not None]
            results_by_family[fam.value] = {
                "total_classes": len(fam_members),
                "measured_classes": len(measured_fam),
                "unmeasured_classes": len(fam_members) - len(measured_fam),
                "mean_f1_50": round(float(np.mean(f1s)), 4) if f1s else None,
                "total_gt_samples": sum(r.gt_count for r in measured_fam),
                "total_pred_samples": sum(r.pred_count for r in fam_res),
            }

        # Multi-view physical identity metrics if requested
        phys_metrics = None
        if eval_mode == EvaluationMode.PHYSICAL_IDENTITY:
            phys_metrics = self._evaluate_physical_identity_clustering(ground_truth, predictions)

        return ComponentBenchmarkReport(
            benchmark_id=benchmark_id,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            split=split,
            mode=eval_mode.value,
            total_images=len(all_gt_images),
            evaluated_classes_count=len(all_eval_classes),
            measured_classes_count=measured_count,
            unmeasured_classes_count=unmeasured_count,
            parent_detection_macro_f1=float(np.mean(parent_f1s)) if parent_f1s else None,
            parent_detection_ap50=float(np.mean(parent_aps_101)) if parent_aps_101 else None,
            parent_detection_ap50_101=float(np.mean(parent_aps_101)) if parent_aps_101 else None,
            parent_detection_ap50_11=float(np.mean(parent_aps_11)) if parent_aps_11 else None,
            component_detection_macro_f1=float(np.mean(comp_f1s)) if comp_f1s else None,
            component_detection_ap50=float(np.mean(comp_aps_101)) if comp_aps_101 else None,
            component_detection_ap50_101=float(np.mean(comp_aps_101)) if comp_aps_101 else None,
            component_detection_ap50_11=float(np.mean(comp_aps_11)) if comp_aps_11 else None,
            component_parent_link_macro_f1=float(np.mean(comp_link_f1s)) if comp_link_f1s else None,
            component_parent_link_precision=float(np.mean(comp_link_precs)) if comp_link_precs else None,
            component_parent_link_recall=float(np.mean(comp_link_recs)) if comp_link_recs else None,
            results_by_class=results_by_class,
            results_by_family=results_by_family,
            physical_identity_metrics=phys_metrics,
            telemetry=telemetry or {},
        )

    def _evaluate_box_matches(
        self,
        gt_boxes: Sequence[GroundTruthComponent],
        pred_boxes: Sequence[DetectionBox],
        iou_thresh: float = 0.50,
        mode: EvaluationMode | str = EvaluationMode.FRAME_LEVEL,
    ) -> tuple[int, int, int, float, float, float, float, float]:
        """Greedy matching of predicted boxes to GT boxes ordered by confidence.

        Returns:
            (tp, fp, fn, prec, rec, f1, ap50_101, ap50_11)
        """
        if not gt_boxes:
            return 0, len(pred_boxes), 0, 0.0, 0.0, 0.0, 0.0, 0.0

        eval_mode = EvaluationMode(mode) if isinstance(mode, str) else mode
        sorted_preds = sorted(pred_boxes, key=lambda p: p.confidence, reverse=True)

        # In FRAME_LEVEL mode, identity key is (image_id, entity_id)
        # In PHYSICAL_IDENTITY mode, identity key is physical_key
        matched_gt_keys: set[Any] = set()

        tp = 0
        fp = 0
        precisions: list[float] = []
        recalls: list[float] = []

        for p in sorted_preds:
            p_img = getattr(p, "source_image_id", "img")
            best_iou = 0.0
            best_gt_key = None

            for g in gt_boxes:
                if g.image_id != p_img:
                    continue
                if not is_class_compatible(p.class_name, g.class_name):
                    continue

                gt_key = g.frame_instance_key if eval_mode == EvaluationMode.FRAME_LEVEL else g.physical_key
                if gt_key in matched_gt_keys:
                    continue

                iou = compute_box_iou(p.box_2d, g.box_2d)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_key = gt_key

            if best_iou >= iou_thresh and best_gt_key is not None:
                tp += 1
                matched_gt_keys.add(best_gt_key)
            else:
                fp += 1

            precisions.append(float(tp) / float(tp + fp))
            recalls.append(float(tp) / float(len(gt_boxes)))

        fn = len(gt_boxes) - tp
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / float(len(gt_boxes)) if len(gt_boxes) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        ap_101 = compute_coco_ap_101(precisions, recalls)
        ap_11 = compute_voc_ap_11(precisions, recalls)

        return tp, fp, fn, prec, rec, f1, ap_101, ap_11

    def _evaluate_parent_child_links(
        self,
        gt_components: Sequence[GroundTruthComponent],
        pred_components: Sequence[DetectionBox],
        all_gt: Sequence[GroundTruthComponent],
        all_preds: Sequence[DetectionBox],
        mode: EvaluationMode | str = EvaluationMode.FRAME_LEVEL,
    ) -> tuple[int, int, int]:
        """Evaluates whether components were correctly linked to their designated host parents.

        Strict 1-to-1 matching:
        - A GT component can be claimed at most once.
        - Redundant component predictions claiming the same GT are penalized as false positives.
        - Predictions with missing or incorrect parent boxes are penalized as false positives.
        - Unclaimed GT components are counted as false negatives.
        """
        # Map GT entity IDs / frame keys to parent entity IDs
        gt_parent_map: dict[tuple[str, str], str] = {
            (g.image_id, g.entity_id): g.parent_entity_id
            for g in gt_components if g.parent_entity_id
        }

        # Build prediction box ID -> DetectionBox lookup
        pred_lookup = {p.box_id: p for p in all_preds}

        # Match predicted components to GT components strictly 1-to-1
        link_tp = 0
        link_fp = 0
        claimed_gt_keys: set[tuple[str, str]] = set()

        sorted_pred_comps = sorted(pred_components, key=lambda p: p.confidence, reverse=True)

        for p in sorted_pred_comps:
            p_img = getattr(p, "source_image_id", "img")
            best_iou = 0.0
            matched_g: GroundTruthComponent | None = None

            for g in gt_components:
                if g.image_id != p_img:
                    continue
                if not is_class_compatible(p.class_name, g.class_name):
                    continue
                iou = compute_box_iou(p.box_2d, g.box_2d)
                if iou > best_iou:
                    best_iou = iou
                    matched_g = g

            if best_iou >= 0.35 and matched_g is not None:
                gt_key = (matched_g.image_id, matched_g.entity_id)
                # 1-to-1 enforcement: if this GT component was already claimed, penalize as FP
                if gt_key in claimed_gt_keys:
                    link_fp += 1
                    continue

                expected_parent_gt_id = gt_parent_map.get(gt_key)

                # Check if predicted component has a valid parent box
                if p.parent_box_id and p.parent_box_id in pred_lookup:
                    parent_pred = pred_lookup[p.parent_box_id]
                    # Find which GT parent object parent_pred matches
                    best_parent_iou = 0.0
                    matched_parent_gt_id = None
                    for g_parent in all_gt:
                        if g_parent.image_id != p_img:
                            continue
                        p_iou = compute_box_iou(parent_pred.box_2d, g_parent.box_2d)
                        if p_iou > best_parent_iou:
                            best_parent_iou = p_iou
                            matched_parent_gt_id = g_parent.entity_id

                    if best_parent_iou >= 0.35 and matched_parent_gt_id == expected_parent_gt_id:
                        link_tp += 1
                        claimed_gt_keys.add(gt_key)
                    else:
                        link_fp += 1
                else:
                    link_fp += 1
            else:
                if p.parent_box_id:
                    link_fp += 1

        link_fn = len(gt_parent_map) - len(claimed_gt_keys)
        return link_tp, link_fp, max(0, link_fn)

    def _evaluate_physical_identity_clustering(
        self,
        ground_truth: Sequence[GroundTruthComponent],
        predictions: Sequence[DetectionBox],
    ) -> dict[str, Any]:
        """Evaluates physical identity clustering across multiple viewpoints."""
        gt_phys_map: dict[tuple[str, str], str] = {
            (g.image_id, g.entity_id): g.physical_key for g in ground_truth
        }
        pred_phys_clusters: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for p in predictions:
            p_img = getattr(p, "source_image_id", "img")
            p_id = getattr(p, "physical_identity_id", None) or getattr(p, "track_id", None) or p.box_id
            best_iou = 0.0
            matched_gt_key = None
            for g in ground_truth:
                if g.image_id != p_img:
                    continue
                if not is_class_compatible(p.class_name, g.class_name):
                    continue
                iou = compute_box_iou(p.box_2d, g.box_2d)
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_key = (g.image_id, g.entity_id)
            if best_iou >= 0.50 and matched_gt_key is not None:
                pred_phys_clusters[str(p_id)].append(matched_gt_key)

        purities = []
        for cluster_id, items in pred_phys_clusters.items():
            if not items:
                continue
            gt_phys_counts = defaultdict(int)
            for it in items:
                gt_phys_counts[gt_phys_map.get(it, "unknown")] += 1
            dominant_count = max(gt_phys_counts.values())
            purities.append(float(dominant_count) / float(len(items)))

        mean_cluster_purity = float(np.mean(purities)) if purities else 1.0
        return {
            "total_predicted_clusters": len(pred_phys_clusters),
            "mean_cluster_purity": round(mean_cluster_purity, 4),
            "evaluated_cluster_instances": sum(len(c) for c in pred_phys_clusters.values()),
        }


def load_component_ground_truth(
    dataset_path: Path | str,
    split: str = "all",
) -> tuple[list[GroundTruthComponent], list[Path], dict[str, Any]]:
    """Loads traceable component ground truth from a dataset directory.

    Strict rules:
    - Never fabricates ground truth.
    - If no annotations file is present, returns empty GT with explicit unmeasured status.
    - Parses parent_entity_id for component-to-parent validation.
    """
    path = Path(dataset_path)
    gt_file = path / "ground_truth_identities.json"
    if not gt_file.is_file():
        gt_file = path / "ground_truth.json"

    # Find image files
    img_dir = path / "images" if (path / "images").is_dir() else path
    all_images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))

    if not gt_file.is_file():
        return [], all_images, {
            "status": "UNMEASURED",
            "reason": f"No ground truth annotation file found in {path}",
            "image_count": len(all_images),
        }

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_annotations = data.get("image_annotations", {})
    sorted_image_names = sorted(image_annotations.keys())
    img_to_frame_idx = {name: idx for idx, name in enumerate(sorted_image_names)}

    import re

    gt_components: list[GroundTruthComponent] = []
    filtered_images: list[Path] = []

    for img_name, annotations in image_annotations.items():
        m = re.search(r"(\d+)", img_name)
        frame_idx = int(m.group(1)) if m else img_to_frame_idx.get(img_name, 0)

        norm_split = split.lower().strip()
        if norm_split == "dev" and frame_idx > 10:
            continue
        elif norm_split == "test" and frame_idx <= 10:
            continue

        # Find actual file path
        found_path = None
        for candidate in [path / img_name, img_dir / img_name]:
            if candidate.is_file():
                found_path = candidate
                break
        if found_path:
            filtered_images.append(found_path)

        for ann in annotations:
            bbox = ann.get("bbox_2d") or ann.get("box_2d")
            if not bbox or len(bbox) != 4:
                continue
            phys_id = ann.get("ground_truth_id") or ann.get("physical_identity_id") or ann.get("entity_id")
            entity_id = ann.get("instance_id") or ann.get("entity_id") or phys_id or f"gt_{len(gt_components)}"
            cls_name = ann.get("class_name", "unknown")
            parent_id = ann.get("parent_entity_id") or ann.get("parent_id")
            is_occl = bool(ann.get("is_occluded", False))
            is_trunc = bool(ann.get("is_truncated", False))

            vis = "occluded" if is_occl else ("truncated" if is_trunc else "visible")
            occ_lvl = "partial" if is_occl else "none"
            tax = resolve_taxonomy_class(cls_name)
            is_comp = bool(parent_id is not None or tax.parent_class is not None)

            gt_components.append(GroundTruthComponent(
                entity_id=entity_id,
                physical_identity_id=phys_id,
                class_name=cls_name,
                image_id=img_name,
                box_2d=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                parent_entity_id=parent_id,
                visibility=vis,
                occlusion_level=occ_lvl,
                is_component=is_comp,
            ))

    return gt_components, sorted(filtered_images), {
        "status": "MEASURED" if gt_components else "EMPTY_SPLIT",
        "total_gt_count": len(gt_components),
        "split": split,
        "image_count": len(filtered_images),
    }


def run_component_benchmark(
    dataset_path: Path | str,
    split: str = "test",
    output_path: Path | str | None = None,
    target_width: int = 1280,
    target_height: int = 960,
    max_images: int | None = None,
    mode: str = "frame_level",
) -> ComponentBenchmarkReport:
    """Executes the component-level perception benchmark on a dataset split."""
    import cv2
    import torch

    dataset_path = Path(dataset_path)
    gt_components, image_files, meta = load_component_ground_truth(dataset_path, split=split)

    if max_images and len(image_files) > max_images:
        image_files = image_files[:max_images]
        kept_img_names = {p.name for p in image_files}
        gt_components = [g for g in gt_components if g.image_id in kept_img_names]

    # Initialize production pipeline with calibrator
    calibrator = load_default_production_calibrator()
    pipeline = DetectionPipeline(
        detector=create_default_detector(),
        segmenter=create_default_segmenter(),
        calibrator=calibrator,
    )

    all_predictions: list[DetectionBox] = []

    print(f"Starting component benchmark on {dataset_path.name} (split={split}, mode={mode}). Images: {len(image_files)}, GT boxes: {len(gt_components)}")

    for idx, img_path in enumerate(image_files):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Scale to evaluation resolution if requested
        if (orig_w, orig_h) != (target_width, target_height):
            img_resized = cv2.resize(img_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            scale_x = float(orig_w) / float(target_width)
            scale_y = float(orig_h) / float(target_height)
        else:
            img_resized = img_rgb
            scale_x, scale_y = 1.0, 1.0

        res = pipeline.run_pipeline(
            img_resized,
            image_id=img_path.name,
            detect_subparts=True,
            run_tiling=True,
        )

        # Map detections back to original ground-truth coordinate system if needed
        for det in res.refined_boxes:
            if scale_x != 1.0 or scale_y != 1.0:
                mapped_box = (
                    det.x1 * scale_x,
                    det.y1 * scale_y,
                    det.x2 * scale_x,
                    det.y2 * scale_y,
                )
            else:
                mapped_box = det.box_2d

            all_predictions.append(DetectionBox(
                box_id=det.box_id,
                class_name=det.class_name,
                box_2d=mapped_box,
                confidence=det.confidence,
                source_engine=det.source_engine,
                source_image_id=img_path.name,
                parent_box_id=det.parent_box_id,
                scale=det.scale,
                physical_identity_id=getattr(det, "physical_identity_id", None),
                attributes=det.attributes,
                raw_score=det.raw_score,
            ))

    evaluator = ComponentBenchmarkEvaluator()
    report = evaluator.evaluate(
        ground_truth=gt_components,
        predictions=all_predictions,
        eval_split=split,
        dataset_name=dataset_path.name,
        mode=mode,
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Component benchmark report written to {out_p}")

    return report


def main() -> int:
    """CLI entrypoint for running the component-level perception benchmark."""
    parser = argparse.ArgumentParser(description="Reality Engine Component-Level Perception Benchmark (Sprint 9)")
    parser.add_argument("--dataset", type=str, default="datasets/real_room_capture", help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev", "all"], help="Evaluation split")
    parser.add_argument("--mode", type=str, default="frame_level", choices=["frame_level", "physical_identity"], help="Evaluation mode")
    parser.add_argument("--output", type=str, default="artifacts/perception/component_benchmark_eval_sprint9.json", help="Output JSON path")
    parser.add_argument("--width", type=int, default=1280, help="Target evaluation width")
    parser.add_argument("--height", type=int, default=960, help="Target evaluation height")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate")

    args = parser.parse_args()
    report = run_component_benchmark(
        dataset_path=args.dataset,
        split=args.split,
        mode=args.mode,
        output_path=args.output,
        target_width=args.width,
        target_height=args.height,
        max_images=args.max_images,
    )
    print("\n" + "=" * 60)
    print(f"COMPONENT BENCHMARK SUMMARY ({report.dataset_name} - {report.split} - mode={report.mode}):")
    print(f"Total taxonomy classes evaluated: {report.evaluated_classes_count}")
    print(f"Measured classes: {report.measured_classes_count}, Unmeasured classes: {report.unmeasured_classes_count}")
    p_f1 = report.parent_detection_macro_f1 if report.parent_detection_macro_f1 is not None else 0.0
    p_ap_coco = report.parent_detection_ap50_101 if report.parent_detection_ap50_101 is not None else 0.0
    p_ap_voc = report.parent_detection_ap50_11 if report.parent_detection_ap50_11 is not None else 0.0
    c_f1 = report.component_detection_macro_f1 if report.component_detection_macro_f1 is not None else 0.0
    c_ap_coco = report.component_detection_ap50_101 if report.component_detection_ap50_101 is not None else 0.0
    link_f1 = report.component_parent_link_macro_f1 if report.component_parent_link_macro_f1 is not None else 0.0
    print(f"Parent Detection Macro F1: {p_f1:.4f}, COCO AP50: {p_ap_coco:.4f}, VOC AP50: {p_ap_voc:.4f}")
    print(f"Component Detection Macro F1: {c_f1:.4f}, COCO AP50: {c_ap_coco:.4f}")
    print(f"Component-Parent Link Macro F1: {link_f1:.4f}")
    if report.physical_identity_metrics:
        print(f"Physical Identity Purity: {report.physical_identity_metrics.get('mean_cluster_purity', 0.0):.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
