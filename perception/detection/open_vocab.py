"""Reality Engine — Open-Vocabulary Detection via Model Registry (Task 8.2).

Rule Zero: ML proposes, geometry verifies (P3).
A detector returns Observations only. It never writes entities, surfaces, or measurements.
Prompts originate strictly from the versioned ontology (`config/ontology.yaml`).
Enforces resolution-aware `min_pixels` gating (P10/P11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

import numpy as np

from perception import model_registry
from perception.detection.interface import Detector, DetectorUnavailable, ImageDetection, get_detector
from perception.ontology import Ontology, OntologyNode, get_default_ontology
from world_ir.observations import DepthSource, Observation


@dataclass
class DetectionEvaluation:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def box_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x_min, y_min, x_max, y_max) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


class OpenVocabDetector:
    """Open-vocabulary object detector driven by the versioned ontology."""

    def __init__(
        self,
        model_name: str = "owlvit_base_patch32",
        backend: Optional[Detector] = None,
        ontology: Optional[Ontology] = None,
        confidence_threshold: float = 0.25,
        allowed_groups: Optional[Set[str]] = None,
    ):
        self.model_name = model_name
        self.ontology = ontology or get_default_ontology()
        self.confidence_threshold = confidence_threshold
        self.allowed_groups = allowed_groups

        # Lookup model metadata in the registry
        self.spec = model_registry.get(model_name)
        self.model_label = f"{self.spec.name}@{self.spec.status}" if self.spec else model_name
        self.license = self.spec.license if self.spec else "Apache-2.0"

        # Obtain backend: injected, or resolved from detector registry
        if backend is not None:
            self._backend = backend
        else:
            self._backend = get_detector(model_name)

    def is_available(self) -> Tuple[bool, str]:
        """Check if backend is available locally without downloading."""
        return self._backend.available()

    def get_ontology_prompts(self) -> List[str]:
        """Collect detector text queries from active ontology classes."""
        prompts = []
        for node in self.ontology.nodes.values():
            if self.allowed_groups and node.group not in self.allowed_groups:
                continue
            if node.path.startswith("fallback/"):
                continue
            prompts.extend(node.prompts)
        return sorted(list(set(prompts)))

    def detect_observations(
        self,
        image_rgb: np.ndarray,
        image_id: str,
        queries: Optional[Sequence[str]] = None,
        camera_matrix: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        scale_state: str = "relative",
    ) -> List[Observation]:
        """Run open-vocabulary detection on an image and return Observations only.
        
        Rule Zero: Returns observations only. Never writes entities or surfaces.
        Enforces resolution-aware min_pixels gating.
        """
        active_queries = list(queries) if queries else self.get_ontology_prompts()
        if not active_queries:
            return []

        # Run backend detection
        raw_detections: List[ImageDetection] = self._backend.detect(image_rgb, active_queries)

        h, w = image_rgb.shape[:2]
        observations: List[Observation] = []

        for det in raw_detections:
            if det.score < self.confidence_threshold:
                continue

            x0, y0, x1, y1 = det.box_xyxy
            box_w = max(0.0, x1 - x0)
            box_h = max(0.0, y1 - y0)
            pixel_area = box_w * box_h

            # Resolve canonical ontology node
            onto_node: OntologyNode = self.ontology.resolve_or_unknown(det.label)

            # Resolution-aware gating: reject detections below the class's min_pixels
            if pixel_area < onto_node.min_pixels:
                continue

            # Determine 3D approximate position from depth map if available
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            world_pos = (0.0, 0.0, 0.0)
            depth_src = DepthSource.RELATIVE
            sigma = 0.5 if scale_state == "metric" else 1.0

            if depth_map is not None and 0 <= int(cy) < depth_map.shape[0] and 0 <= int(cx) < depth_map.shape[1]:
                d = float(depth_map[int(cy), int(cx)])
                if np.isfinite(d) and d > 0:
                    depth_src = DepthSource.METRIC if scale_state == "metric" else DepthSource.RELATIVE
                    if camera_matrix is not None:
                        fx = float(camera_matrix[0, 0])
                        fy = float(camera_matrix[1, 1])
                        px = float(camera_matrix[0, 2])
                        py = float(camera_matrix[1, 2])
                        X = (cx - px) * d / fx
                        Y = (cy - py) * d / fy
                        Z = d
                        world_pos = (X, Y, Z)
                    else:
                        world_pos = (0.0, 0.0, d)

            obs = Observation(
                observation_id=f"obs-det-{uuid4().hex[:10]}",
                image_id=image_id,
                label=onto_node.path,
                world_pos=world_pos,
                sigma_m=sigma,
                depth_source=depth_src,
                model=self.model_label,
                license=self.license,
                confidence=float(det.score),
                bbox_px=(float(x0), float(y0), float(x1), float(y1)),
            )
            observations.append(obs)

        return observations


def evaluate_detection_metrics(
    observations: Sequence[Observation],
    ground_truth: Sequence[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, DetectionEvaluation]:
    """Calculate per-class Precision, Recall, and F1 on a benchmark dataset.
    
    Each ground_truth item is { 'label': str, 'box_xyxy': tuple, 'image_id': str }.
    """
    classes = set()
    for o in observations:
        classes.add(o.label)
    for gt in ground_truth:
        classes.add(gt["label"])

    evals: Dict[str, DetectionEvaluation] = {cls: DetectionEvaluation() for cls in classes}

    # Group by image
    images = set([o.image_id for o in observations] + [gt.get("image_id", "default") for gt in ground_truth])

    for img_id in images:
        img_obs = [o for o in observations if o.image_id == img_id]
        img_gt = [gt for gt in ground_truth if gt.get("image_id", "default") == img_id]

        matched_gt_indices = set()

        for obs in sorted(img_obs, key=lambda x: x.confidence, reverse=True):
            if obs.bbox_px is None:
                continue

            best_iou = 0.0
            best_gt_idx = -1

            for idx, gt in enumerate(img_gt):
                if idx in matched_gt_indices:
                    continue
                if gt["label"] != obs.label:
                    continue

                iou = box_iou(obs.bbox_px, tuple(gt["box_xyxy"]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            cls_stat = evals[obs.label]
            if best_iou >= iou_threshold and best_gt_idx != -1:
                cls_stat.true_positives += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                cls_stat.false_positives += 1

        # Count unmatched ground truths as false negatives
        for idx, gt in enumerate(img_gt):
            if idx not in matched_gt_indices:
                evals[gt["label"]].false_negatives += 1

    return evals
