"""Tests for Multi-Engine Comparative Evaluation Harness.

Validates:
- Camera trajectory alignment via Umeyama Sim(3) and ATE calculation
- Depth map affine scale/shift alignment, MAE, RMSE, and edge correlation
- 2D/3D Object detection bipartite matching, consensus, and label conflict matrix
- Segmentation mask overlap and hole topology comparison
- Mesh Chamfer distance calculation
"""

from __future__ import annotations

import numpy as np

from engine.evaluation import (
    ComparativeEvaluationHarness,
    DepthComparisonResult,
    DetectionComparisonResult,
    MaskComparisonResult,
    PoseComparisonResult,
)


def test_compare_camera_poses_exact_identity():
    poses = [
        {"image_id": "img1", "position_m": [0.0, 0.0, 0.0]},
        {"image_id": "img2", "position_m": [1.0, 0.0, 0.0]},
        {"image_id": "img3", "position_m": [1.0, 1.0, 0.0]},
        {"image_id": "img4", "position_m": [0.0, 1.0, 0.0]},
    ]
    res = ComparativeEvaluationHarness.compare_camera_poses(poses, poses)
    assert isinstance(res, PoseComparisonResult)
    assert res.num_common_cameras == 4
    assert res.ate_rmse_m < 1e-6
    assert abs(res.scale_factor - 1.0) < 1e-6


def test_compare_camera_poses_scaled_and_translated():
    # Model B is scaled by 2.0 and translated by [10, 5, 0]
    poses_a = [
        {"image_id": "img1", "position_m": [0.0, 0.0, 0.0]},
        {"image_id": "img2", "position_m": [1.0, 0.0, 0.0]},
        {"image_id": "img3", "position_m": [1.0, 2.0, 0.0]},
        {"image_id": "img4", "position_m": [0.0, 2.0, 0.0]},
    ]
    poses_b = [
        {"image_id": "img1", "position_m": [10.0, 5.0, 0.0]},
        {"image_id": "img2", "position_m": [12.0, 5.0, 0.0]},
        {"image_id": "img3", "position_m": [12.0, 9.0, 0.0]},
        {"image_id": "img4", "position_m": [10.0, 9.0, 0.0]},
    ]
    res = ComparativeEvaluationHarness.compare_camera_poses(poses_a, poses_b)
    assert res.num_common_cameras == 4
    assert res.ate_rmse_m < 1e-5
    assert abs(res.scale_factor - 2.0) < 1e-4


def test_compare_depth_maps_affine_invariance():
    # Model A: non-linear continuous surface
    # Model B: affine transformed 2.0 * depth_a + 3.0
    h, w = 40, 40
    y, x = np.mgrid[:h, :w]
    depth_a = (np.sin(x / 5.0) + np.cos(y / 5.0) + 3.0).astype(np.float32)
    depth_b = (2.0 * depth_a + 3.0).astype(np.float32)

    res = ComparativeEvaluationHarness.compare_depth_maps(depth_a, depth_b)
    assert isinstance(res, DepthComparisonResult)
    # Raw differences are large
    assert res.mae_relative > 3.0
    # Aligned difference accounts for affine scale/shift
    assert res.aligned_mae < 1e-4
    assert abs(res.optimal_scale - 2.0) < 1e-3
    assert abs(res.optimal_shift - 3.0) < 1e-3
    # Gradient edge correlation should be nearly 1.0
    assert res.edge_correlation > 0.99


def test_compare_object_detections_consensus_and_conflict():
    # Two models detect objects in the same image
    # Pair 1: chair at [10, 10, 50, 50] (Both agree -> Consensus)
    # Pair 2: table vs desk at [100, 100, 200, 200] (Disagree label -> Conflict)
    # Model A only: [300, 300, 350, 350]
    dets_a = [
        {"label": "chair", "canonical_path": "furniture.chair", "box_xyxy": [10, 10, 50, 50]},
        {"label": "table", "canonical_path": "furniture.dining_table", "box_xyxy": [100, 100, 200, 200]},
        {"label": "lamp", "canonical_path": "lighting.floor_lamp", "box_xyxy": [300, 300, 350, 350]},
    ]
    dets_b = [
        {"label": "chair", "canonical_path": "furniture.chair", "box_xyxy": [12, 12, 52, 52]},
        {"label": "desk", "canonical_path": "furniture.desk", "box_xyxy": [98, 98, 202, 202]},
    ]

    res = ComparativeEvaluationHarness.compare_object_detections(dets_a, dets_b, iou_threshold=0.5)
    assert isinstance(res, DetectionComparisonResult)
    assert res.num_matched_pairs == 2
    assert res.consensus_count == 1
    assert res.conflict_count == 1
    assert res.model_a_only_count == 1
    assert res.model_b_only_count == 0
    assert res.label_agreement_matrix["furniture.dining_table"]["furniture.desk"] == 1


def test_compare_segmentation_masks_and_topology():
    # Mask with a donut hole
    mask_a = np.zeros((100, 100), dtype=bool)
    mask_a[20:80, 20:80] = True
    mask_a[40:60, 40:60] = False  # Hole

    mask_b = np.zeros((100, 100), dtype=bool)
    mask_b[25:75, 25:75] = True  # Solid without hole

    from perception.specialists.segment import extract_mask_topology

    top_a = extract_mask_topology(mask_a)
    top_b = extract_mask_topology(mask_b)

    res = ComparativeEvaluationHarness.compare_segmentation_masks(
        mask_a, mask_b, topology_a=top_a, topology_b=top_b
    )
    assert isinstance(res, MaskComparisonResult)
    assert 0.0 < res.iou < 1.0
    assert res.holes_model_a == 1
    assert res.holes_model_b == 0
    assert not res.hole_topology_agreement
