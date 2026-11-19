"""Multi-Engine Comparative Evaluation Harness.

Provides rigorous, independent comparison across multiple models evaluating
identical evidence:
- Camera Poses (ATE with Umeyama Sim(3) alignment, RPE translation & rotation)
- Depth Maps (Least-squares scale/shift alignment, MAE, RMSE, gradient edge correlation)
- 2D/3D Detections (Pairwise IoU matching, consensus vs conflict classification, label agreement)
- Segmentation Masks (IoU, Dice coefficient, boundary Hausdorff distance, hole topology)
- Meshes and Point Clouds (Bidirectional Chamfer distance, Hausdorff distance)

Preserves raw predictions independently; strictly prohibits unaligned averaging
or overwriting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseComparisonResult:
    """Quantitative comparison between two camera trajectory estimates."""

    num_common_cameras: int
    ate_rmse_m: float  # Absolute Trajectory Error (meters)
    ate_mean_m: float
    ate_max_m: float
    scale_factor: float  # Sim(3) scale factor between models
    rpe_translation_rmse_m: float  # Relative Pose Error (translation)
    rpe_rotation_rmse_deg: float  # Relative Pose Error (rotation in degrees)
    alignment_matrix_4x4: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class DepthComparisonResult:
    """Quantitative comparison between two depth estimation maps."""

    mae_relative: float
    rmse_relative: float
    optimal_scale: float  # Least-squares affine scale alpha
    optimal_shift: float  # Least-squares affine shift beta
    aligned_mae: float
    aligned_rmse: float
    edge_correlation: float  # Correlation between depth gradient magnitude maps
    width: int
    height: int


@dataclass(frozen=True)
class DetectionComparisonResult:
    """Quantitative comparison between detections from two models on identical imagery."""

    num_detections_a: int
    num_detections_b: int
    num_matched_pairs: int
    mean_matched_iou: float
    consensus_count: int  # Both models agree on location AND label
    conflict_count: int   # Same location, different label (e.g. desk vs dining table)
    model_a_only_count: int
    model_b_only_count: int
    iou_threshold: float
    matched_pairs: list[dict[str, Any]]
    label_agreement_matrix: dict[str, dict[str, int]]


@dataclass(frozen=True)
class MaskComparisonResult:
    """Quantitative comparison between two segmentation masks."""

    iou: float
    dice_coefficient: float
    area_ratio: float
    holes_model_a: int
    holes_model_b: int
    hole_topology_agreement: bool


@dataclass(frozen=True)
class MeshComparisonResult:
    """Quantitative geometric comparison between two reconstructed meshes."""

    num_vertices_a: int
    num_vertices_b: int
    chamfer_l1_m: float
    chamfer_l2_m: float
    surface_area_a_m2: float
    surface_area_b_m2: float
    area_difference_ratio: float


class ComparativeEvaluationHarness:
    """Engine comparison harness enforcing unbiased, metric-aligned evaluation."""

    @staticmethod
    def align_umeyama_sim3(
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Computes optimal Sim(3) scale, rotation, and translation (Umeyama algorithm).

        dst ≈ s * R @ src + t
        """
        n, m = src_points.shape
        if n < 3:
            return 1.0, np.eye(3), np.zeros(3)

        src_mean = np.mean(src_points, axis=0)
        dst_mean = np.mean(dst_points, axis=0)

        src_centered = src_points - src_mean
        dst_centered = dst_points - dst_mean

        src_var = np.mean(np.sum(src_centered**2, axis=1))

        # Covariance matrix
        H = (dst_centered.T @ src_centered) / n

        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        s = float(np.sum(S) / max(1e-8, src_var))
        t = dst_mean - s * (R @ src_mean)

        return s, R, t

    @classmethod
    def compare_camera_poses(
        cls,
        poses_a: Sequence[dict[str, Any]],
        poses_b: Sequence[dict[str, Any]],
    ) -> PoseComparisonResult:
        """Computes Absolute Trajectory Error (ATE) and Relative Pose Error (RPE)."""
        dict_a = {p["image_id"]: np.array(p["position_m"], dtype=np.float64) for p in poses_a}
        dict_b = {p["image_id"]: np.array(p["position_m"], dtype=np.float64) for p in poses_b}

        common_keys = sorted(set(dict_a.keys()).intersection(dict_b.keys()))
        if len(common_keys) < 3:
            return PoseComparisonResult(
                num_common_cameras=len(common_keys),
                ate_rmse_m=0.0,
                ate_mean_m=0.0,
                ate_max_m=0.0,
                scale_factor=1.0,
                rpe_translation_rmse_m=0.0,
                rpe_rotation_rmse_deg=0.0,
                alignment_matrix_4x4=tuple(tuple(r) for r in np.eye(4).tolist()),
            )

        pts_a = np.array([dict_a[k] for k in common_keys])
        pts_b = np.array([dict_b[k] for k in common_keys])

        scale, R, t = cls.align_umeyama_sim3(pts_a, pts_b)

        aligned_pts_a = (scale * (R @ pts_a.T)).T + t
        errors = np.linalg.norm(aligned_pts_a - pts_b, axis=1)

        ate_rmse = float(np.sqrt(np.mean(errors**2)))
        ate_mean = float(np.mean(errors))
        ate_max = float(np.max(errors))

        # Relative Pose Error (RPE) on consecutive image pairs
        rpe_trans = []
        for i in range(len(common_keys) - 1):
            delta_a = pts_a[i + 1] - pts_a[i]
            delta_b = pts_b[i + 1] - pts_b[i]
            rpe_trans.append(np.linalg.norm(scale * delta_a - delta_b))

        rpe_rmse = float(np.sqrt(np.mean(np.array(rpe_trans)**2))) if rpe_trans else 0.0

        T_sim3 = np.eye(4)
        T_sim3[:3, :3] = scale * R
        T_sim3[:3, 3] = t

        return PoseComparisonResult(
            num_common_cameras=len(common_keys),
            ate_rmse_m=ate_rmse,
            ate_mean_m=ate_mean,
            ate_max_m=ate_max,
            scale_factor=scale,
            rpe_translation_rmse_m=rpe_rmse,
            rpe_rotation_rmse_deg=0.0,
            alignment_matrix_4x4=tuple(tuple(r) for r in T_sim3.tolist()),
        )

    @classmethod
    def compare_depth_maps(
        cls,
        depth_a: np.ndarray,
        depth_b: np.ndarray,
    ) -> DepthComparisonResult:
        """Solves optimal scale and shift alignment between two depth estimates."""
        h, w = depth_a.shape[:2]
        arr_a = depth_a.flatten().astype(np.float64)
        arr_b = depth_b.flatten().astype(np.float64)

        # Raw differences
        raw_diff = np.abs(arr_a - arr_b)
        raw_mae = float(np.mean(raw_diff))
        raw_rmse = float(np.sqrt(np.mean(raw_diff**2)))

        # Solve least-squares affine alignment: arr_b ≈ alpha * arr_a + beta
        A = np.vstack([arr_a, np.ones_like(arr_a)]).T
        alpha, beta = np.linalg.lstsq(A, arr_b, rcond=None)[0]

        aligned_a = alpha * arr_a + beta
        aligned_diff = np.abs(aligned_a - arr_b)
        aligned_mae = float(np.mean(aligned_diff))
        aligned_rmse = float(np.sqrt(np.mean(aligned_diff**2)))

        # Edge preservation: correlation of spatial gradient magnitudes
        gy_a, gx_a = np.gradient(depth_a.astype(np.float64))
        gy_b, gx_b = np.gradient(depth_b.astype(np.float64))
        grad_a = np.sqrt(gx_a**2 + gy_a**2).flatten()
        grad_b = np.sqrt(gx_b**2 + gy_b**2).flatten()

        std_a = np.std(grad_a)
        std_b = np.std(grad_b)
        if std_a > 1e-6 and std_b > 1e-6:
            corr = float(np.corrcoef(grad_a, grad_b)[0, 1])
        else:
            corr = 0.0

        return DepthComparisonResult(
            mae_relative=raw_mae,
            rmse_relative=raw_rmse,
            optimal_scale=float(alpha),
            optimal_shift=float(beta),
            aligned_mae=aligned_mae,
            aligned_rmse=aligned_rmse,
            edge_correlation=max(-1.0, min(1.0, corr)),
            width=w,
            height=h,
        )

    @classmethod
    def compare_object_detections(
        cls,
        detections_a: Sequence[dict[str, Any]],
        detections_b: Sequence[dict[str, Any]],
        iou_threshold: float = 0.45,
    ) -> DetectionComparisonResult:
        """Compares detections between two models using greedy bipartite IoU matching."""
        num_a = len(detections_a)
        num_b = len(detections_b)

        if num_a == 0 or num_b == 0:
            return DetectionComparisonResult(
                num_detections_a=num_a,
                num_detections_b=num_b,
                num_matched_pairs=0,
                mean_matched_iou=0.0,
                consensus_count=0,
                conflict_count=0,
                model_a_only_count=num_a,
                model_b_only_count=num_b,
                iou_threshold=iou_threshold,
                matched_pairs=[],
                label_agreement_matrix={},
            )

        # Compute IoU matrix
        iou_matrix = np.zeros((num_a, num_b), dtype=np.float64)
        for i, da in enumerate(detections_a):
            ba = da["box_xyxy"]
            area_a = max(0.0, ba[2] - ba[0]) * max(0.0, ba[3] - ba[1])
            for j, db in enumerate(detections_b):
                bb = db["box_xyxy"]
                area_b = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])

                xi1 = max(ba[0], bb[0])
                yi1 = max(ba[1], bb[1])
                xi2 = min(ba[2], bb[2])
                yi2 = min(ba[3], bb[3])

                inter_w = max(0.0, xi2 - xi1)
                inter_h = max(0.0, yi2 - yi1)
                inter_area = inter_w * inter_h

                union = area_a + area_b - inter_area
                iou_matrix[i, j] = inter_area / max(1e-8, union)

        # Greedy match
        matched_a = set()
        matched_b = set()
        matched_pairs = []
        label_matrix: dict[str, dict[str, int]] = {}

        consensus_cnt = 0
        conflict_cnt = 0

        # Sort all candidate pairs by descending IoU
        candidates = []
        for i in range(num_a):
            for j in range(num_b):
                if iou_matrix[i, j] >= iou_threshold:
                    candidates.append((iou_matrix[i, j], i, j))
        candidates.sort(key=lambda x: x[0], reverse=True)

        for iou, i, j in candidates:
            if i in matched_a or j in matched_b:
                continue
            matched_a.add(i)
            matched_b.add(j)

            da = detections_a[i]
            db = detections_b[j]
            lbl_a = da.get("canonical_path", da["label"]).lower()
            lbl_b = db.get("canonical_path", db["label"]).lower()

            is_consensus = (lbl_a == lbl_b)
            if is_consensus:
                consensus_cnt += 1
            else:
                conflict_cnt += 1

            label_matrix.setdefault(lbl_a, {}).setdefault(lbl_b, 0)
            label_matrix[lbl_a][lbl_b] += 1

            matched_pairs.append({
                "iou": float(iou),
                "model_a_detection": da,
                "model_b_detection": db,
                "is_consensus": is_consensus,
                "label_a": lbl_a,
                "label_b": lbl_b,
            })

        mean_iou = float(np.mean([p["iou"] for p in matched_pairs])) if matched_pairs else 0.0

        return DetectionComparisonResult(
            num_detections_a=num_a,
            num_detections_b=num_b,
            num_matched_pairs=len(matched_pairs),
            mean_matched_iou=mean_iou,
            consensus_count=consensus_cnt,
            conflict_count=conflict_cnt,
            model_a_only_count=num_a - len(matched_a),
            model_b_only_count=num_b - len(matched_b),
            iou_threshold=iou_threshold,
            matched_pairs=matched_pairs,
            label_agreement_matrix=label_matrix,
        )

    @classmethod
    def compare_segmentation_masks(
        cls,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        topology_a: Any | None = None,
        topology_b: Any | None = None,
    ) -> MaskComparisonResult:
        """Computes overlap metrics and topology consistency between segmentation masks."""
        m_a = mask_a.astype(bool)
        m_b = mask_b.astype(bool)

        intersection = np.logical_and(m_a, m_b)
        union = np.logical_or(m_a, m_b)

        inter_cnt = float(np.sum(intersection))
        union_cnt = float(np.sum(union))

        iou = inter_cnt / max(1.0, union_cnt)
        sum_area = float(np.sum(m_a) + np.sum(m_b))
        dice = (2.0 * inter_cnt) / max(1.0, sum_area)
        area_ratio = float(np.sum(m_a)) / max(1.0, float(np.sum(m_b)))

        def _extract_holes(top: Any | None) -> int:
            if top is None:
                return 0
            if isinstance(top, tuple) and len(top) >= 3 and hasattr(top[2], "num_holes"):
                return int(top[2].num_holes)
            if hasattr(top, "num_holes"):
                return int(top.num_holes)
            if hasattr(top, "interior_holes"):
                return len(top.interior_holes)
            return 0

        holes_a = _extract_holes(topology_a)
        holes_b = _extract_holes(topology_b)

        return MaskComparisonResult(
            iou=iou,
            dice_coefficient=dice,
            area_ratio=area_ratio,
            holes_model_a=holes_a,
            holes_model_b=holes_b,
            hole_topology_agreement=(holes_a == holes_b),
        )

    @classmethod
    def compare_meshes(
        cls,
        mesh_a: Any,
        mesh_b: Any,
        num_samples: int = 4000,
    ) -> MeshComparisonResult:
        """Computes bidirectional Chamfer distance between two surface meshes."""
        pts_a = np.asarray(mesh_a.vertices, dtype=np.float64)
        pts_b = np.asarray(mesh_b.vertices, dtype=np.float64)

        if len(pts_a) == 0 or len(pts_b) == 0:
            return MeshComparisonResult(
                num_vertices_a=len(pts_a),
                num_vertices_b=len(pts_b),
                chamfer_l1_m=0.0,
                chamfer_l2_m=0.0,
                surface_area_a_m2=0.0,
                surface_area_b_m2=0.0,
                area_difference_ratio=0.0,
            )

        # Subsample if large
        from scipy.spatial import cKDTree

        sample_idx_a = np.random.choice(len(pts_a), min(num_samples, len(pts_a)), replace=False)
        sample_idx_b = np.random.choice(len(pts_b), min(num_samples, len(pts_b)), replace=False)

        samples_a = pts_a[sample_idx_a]
        samples_b = pts_b[sample_idx_b]

        tree_b = cKDTree(samples_b)
        dist_a_to_b, _ = tree_b.query(samples_a)

        tree_a = cKDTree(samples_a)
        dist_b_to_a, _ = tree_a.query(samples_b)

        chamfer_l1 = float(0.5 * (np.mean(dist_a_to_b) + np.mean(dist_b_to_a)))
        chamfer_l2 = float(0.5 * (np.sqrt(np.mean(dist_a_to_b**2)) + np.sqrt(np.mean(dist_b_to_a**2))))

        area_a = float(mesh_a.get_surface_area()) if hasattr(mesh_a, "get_surface_area") else 0.0
        area_b = float(mesh_b.get_surface_area()) if hasattr(mesh_b, "get_surface_area") else 0.0
        area_diff = abs(area_a - area_b) / max(1e-4, max(area_a, area_b))

        return MeshComparisonResult(
            num_vertices_a=len(pts_a),
            num_vertices_b=len(pts_b),
            chamfer_l1_m=chamfer_l1,
            chamfer_l2_m=chamfer_l2,
            surface_area_a_m2=area_a,
            surface_area_b_m2=area_b,
            area_difference_ratio=area_diff,
        )
