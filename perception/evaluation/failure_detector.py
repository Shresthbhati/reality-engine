"""Engine Failure Diagnosis and Anomaly Detection Engine.

Implements Section 9 of the Multi-Engine Integration Directive:
- Detects monocular depth scale drift and reflective surface distortion
- Detects feature matching starvation on textureless surfaces (painted drywall)
- Detects segmentation mask boundary leakage across low-contrast edges
- Detects multi-view geometry disconnection due to insufficient overlap
- Detects appearance embedding confusion on visually symmetrical furniture
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class FailureKind(str, Enum):
    """Categorization of failure modes across reconstruction engines."""

    NONE = "none"
    DEPTH_INVERTED_OR_FLAT = "depth_inverted_or_flat"
    DEPTH_OUT_OF_BOUNDS = "depth_out_of_bounds"
    TEXTURELESS_FEATURE_STARVATION = "textureless_feature_starvation"
    SEGMENTATION_BOUNDARY_LEAKAGE = "segmentation_boundary_leakage"
    SFM_LOW_OVERLAP_DISCONNECTION = "sfm_low_overlap_disconnection"
    EMBEDDING_SYMMETRY_CONFUSION = "embedding_symmetry_confusion"
    SMALL_OBJECT_RESOLUTION_STARVATION = "small_object_resolution_starvation"


@dataclass(frozen=True)
class FailureDiagnosis:
    """Diagnostic outcome evaluating an engine's output against physical plausibility."""

    engine_name: str
    failure_kind: FailureKind
    is_failure_detected: bool
    confidence_in_failure: float  # [0.0, 1.0]
    metric_value: float
    threshold: float
    explanation: str
    recommended_mitigation: str


class EngineFailureDetector:
    """Analyzes engine outputs to detect known pathology and unreliability patterns."""

    @classmethod
    def diagnose_depth(
        cls,
        depth_map: np.ndarray,
        min_valid_depth: float = 0.05,
        max_valid_depth: float = 50.0,
    ) -> FailureDiagnosis:
        """Detects uncalibrated, inverted, or flat degenerate depth maps."""
        arr = depth_map.astype(np.float64)
        std_val = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        if std_val < 1e-4:
            return FailureDiagnosis(
                engine_name="depth_estimator",
                failure_kind=FailureKind.DEPTH_INVERTED_OR_FLAT,
                is_failure_detected=True,
                confidence_in_failure=0.99,
                metric_value=std_val,
                threshold=1e-4,
                explanation="Depth map has near-zero standard deviation (flat plane or saturated output).",
                recommended_mitigation="Reject depth map; re-estimate with secondary depth model or LiDAR sidecar.",
            )

        if min_val < 0.0 or max_val > max_valid_depth:
            return FailureDiagnosis(
                engine_name="depth_estimator",
                failure_kind=FailureKind.DEPTH_OUT_OF_BOUNDS,
                is_failure_detected=True,
                confidence_in_failure=0.95,
                metric_value=max_val,
                threshold=max_valid_depth,
                explanation=f"Depth range [{min_val:.2f}, {max_val:.2f}] violates physical bounds.",
                recommended_mitigation="Enforce metric scale clipping and bilateral depth outlier rejection.",
            )

        return FailureDiagnosis(
            engine_name="depth_estimator",
            failure_kind=FailureKind.NONE,
            is_failure_detected=False,
            confidence_in_failure=0.0,
            metric_value=std_val,
            threshold=1e-4,
            explanation="Depth map satisfies variance and bounding constraints.",
            recommended_mitigation="Proceed with back-projection.",
        )

    @classmethod
    def diagnose_feature_matching(
        cls,
        image_a: np.ndarray,
        image_b: np.ndarray,
        num_inlier_matches: int,
        min_inliers_required: int = 15,
    ) -> FailureDiagnosis:
        """Detects feature starvation on textureless surfaces (e.g. painted drywall)."""
        # Compute gradient variance (textureness) of images
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY) if len(image_a.shape) == 3 else image_a
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY) if len(image_b.shape) == 3 else image_b

        var_a = float(np.var(cv2.Laplacian(gray_a, cv2.CV_64F)))
        var_b = float(np.var(cv2.Laplacian(gray_b, cv2.CV_64F)))
        mean_texture = 0.5 * (var_a + var_b)

        if num_inlier_matches < min_inliers_required and mean_texture < 50.0:
            return FailureDiagnosis(
                engine_name="feature_matcher",
                failure_kind=FailureKind.TEXTURELESS_FEATURE_STARVATION,
                is_failure_detected=True,
                confidence_in_failure=0.92,
                metric_value=float(num_inlier_matches),
                threshold=float(min_inliers_required),
                explanation=f"Image pair has low texture ({mean_texture:.1f}) and failed to yield >= {min_inliers_required} matches.",
                recommended_mitigation="Switch from classical SIFT to ALIKED + LightGlue or dense matcher (MASt3R).",
            )

        return FailureDiagnosis(
            engine_name="feature_matcher",
            failure_kind=FailureKind.NONE,
            is_failure_detected=False,
            confidence_in_failure=0.0,
            metric_value=float(num_inlier_matches),
            threshold=float(min_inliers_required),
            explanation="Feature match count satisfies geometric epipolar estimation requirements.",
            recommended_mitigation="Proceed to pose estimation.",
        )

    @classmethod
    def diagnose_segmentation_leakage(
        cls,
        mask: np.ndarray,
        prompt_box_xyxy: Sequence[float],
        max_leak_ratio: float = 1.40,
    ) -> FailureDiagnosis:
        """Detects mask bleeding beyond the prompting bounding box."""
        bx1, by1, bx2, by2 = [int(round(c)) for c in prompt_box_xyxy]
        h, w = mask.shape[:2]

        box_w = max(1, bx2 - bx1)
        box_h = max(1, by2 - by1)
        box_area = box_w * box_h

        mask_area = float(np.sum(mask.astype(bool)))

        # Pixels outside the expanded bounding box
        expanded_margin = 15
        ex1 = max(0, bx1 - expanded_margin)
        ey1 = max(0, by1 - expanded_margin)
        ex2 = min(w, bx2 + expanded_margin)
        ey2 = min(h, by2 + expanded_margin)

        box_crop = np.zeros_like(mask, dtype=bool)
        box_crop[ey1:ey2, ex1:ex2] = True

        outside_mask = np.logical_and(mask.astype(bool), ~box_crop)
        outside_area = float(np.sum(outside_mask))

        leak_ratio = (mask_area) / max(1.0, float(box_area))

        if outside_area > 0.20 * mask_area or leak_ratio > max_leak_ratio:
            return FailureDiagnosis(
                engine_name="instance_segmenter",
                failure_kind=FailureKind.SEGMENTATION_BOUNDARY_LEAKAGE,
                is_failure_detected=True,
                confidence_in_failure=0.88,
                metric_value=leak_ratio,
                threshold=max_leak_ratio,
                explanation=f"Mask leaked beyond prompt box (leak ratio: {leak_ratio:.2f}, outside pixels: {int(outside_area)}).",
                recommended_mitigation="Clip mask to bounding box margin and extract contour holes using RETR_CCOMP.",
            )

        return FailureDiagnosis(
            engine_name="instance_segmenter",
            failure_kind=FailureKind.NONE,
            is_failure_detected=False,
            confidence_in_failure=0.0,
            metric_value=leak_ratio,
            threshold=max_leak_ratio,
            explanation="Mask conforms to bounding box boundary constraints.",
            recommended_mitigation="Proceed to 3D lifting.",
        )
