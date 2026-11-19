"""Tests for Engine Failure Diagnosis and Anomaly Detection.

Validates:
- Detection of near-zero variance flat depth maps and negative/out-of-bounds values
- Detection of feature matching starvation on low-texture drywall surfaces
- Detection of segmentation mask boundary bleeding/leakage beyond prompt boxes
"""

from __future__ import annotations

import numpy as np

from perception.evaluation import (
    EngineFailureDetector,
    FailureDiagnosis,
    FailureKind,
)


def test_diagnose_flat_depth_map():
    # Saturated flat depth map
    flat_depth = np.ones((50, 50), dtype=np.float32) * 3.0
    diag = EngineFailureDetector.diagnose_depth(flat_depth)
    assert isinstance(diag, FailureDiagnosis)
    assert diag.is_failure_detected is True
    assert diag.failure_kind == FailureKind.DEPTH_INVERTED_OR_FLAT
    assert diag.confidence_in_failure > 0.95


def test_diagnose_out_of_bounds_depth():
    # Negative or extreme depth values
    invalid_depth = np.ones((50, 50), dtype=np.float32) * 2.0
    invalid_depth[0, 0] = -1.5
    diag = EngineFailureDetector.diagnose_depth(invalid_depth, min_valid_depth=0.05, max_valid_depth=50.0)
    assert diag.is_failure_detected is True
    assert diag.failure_kind == FailureKind.DEPTH_OUT_OF_BOUNDS


def test_diagnose_valid_depth_passes():
    h, w = 50, 50
    y, x = np.mgrid[:h, :w]
    valid_depth = (x / float(w) * 3.0 + 1.0).astype(np.float32)
    diag = EngineFailureDetector.diagnose_depth(valid_depth)
    assert diag.is_failure_detected is False
    assert diag.failure_kind == FailureKind.NONE


def test_diagnose_textureless_feature_starvation():
    # Featureless flat white drywall
    blank_img_a = np.ones((100, 100, 3), dtype=np.uint8) * 240
    blank_img_b = np.ones((100, 100, 3), dtype=np.uint8) * 240

    # Only 4 inliers recovered
    diag = EngineFailureDetector.diagnose_feature_matching(
        blank_img_a, blank_img_b, num_inlier_matches=4, min_inliers_required=15
    )
    assert diag.is_failure_detected is True
    assert diag.failure_kind == FailureKind.TEXTURELESS_FEATURE_STARVATION
    assert "low texture" in diag.explanation


def test_diagnose_textured_surface_passes():
    # High-contrast checkerboard image
    textured_a = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(np.uint8)
    textured_b = textured_a.copy()

    diag = EngineFailureDetector.diagnose_feature_matching(
        textured_a, textured_b, num_inlier_matches=45, min_inliers_required=15
    )
    assert diag.is_failure_detected is False
    assert diag.failure_kind == FailureKind.NONE


def test_diagnose_segmentation_mask_leakage():
    # Target bounding box is [40, 40, 60, 60] (area = 400)
    # Mask leaks across half the image [10, 10, 90, 90] (area = 6400)
    leaked_mask = np.zeros((100, 100), dtype=bool)
    leaked_mask[10:90, 10:90] = True

    box = [40.0, 40.0, 60.0, 60.0]
    diag = EngineFailureDetector.diagnose_segmentation_leakage(leaked_mask, box)
    assert diag.is_failure_detected is True
    assert diag.failure_kind == FailureKind.SEGMENTATION_BOUNDARY_LEAKAGE


def test_diagnose_well_bounded_mask_passes():
    # Target bounding box is [30, 30, 70, 70]
    # Mask fits neatly inside [32, 32, 68, 68]
    valid_mask = np.zeros((100, 100), dtype=bool)
    valid_mask[32:68, 32:68] = True

    box = [30.0, 30.0, 70.0, 70.0]
    diag = EngineFailureDetector.diagnose_segmentation_leakage(valid_mask, box)
    assert diag.is_failure_detected is False
    assert diag.failure_kind == FailureKind.NONE
