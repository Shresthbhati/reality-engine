"""Reconstruction completeness and coverage metrics (Master Development Plan Phase 4, Tasks 4.9 & 4.10).

Quantifies how thoroughly an environment was reconstructed:
1. Registration completeness: registered cameras / total input cameras.
2. Geometric redundancy: track length distribution (share of 3D points supported by >= 3 cameras).
3. Spatial coverage: bounding volume extent and voxel occupancy density across the observed volume.
4. Scale reliability distinction: separates geometric consistency from metric scale anchoring
   (verified physical metric vs device LiDAR vs uncalibrated relative).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.backend.interface import ReconstructionResult


@dataclass(frozen=True)
class CompletenessMetrics:
    """Quantitative completeness and scale reliability of a 3D reconstruction."""

    cameras_input: int
    cameras_registered: int
    registration_ratio: float
    points_total: int
    scene_bounds_min: Tuple[float, float, float]
    scene_bounds_max: Tuple[float, float, float]
    scene_volume_m3: float
    voxel_occupancy_count: int
    voxel_size_m: float
    mean_track_length: float
    redundancy_ratio_3plus: float
    scale_state: str
    scale_reliability: str
    completeness_score: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["registration_ratio"] = round(self.registration_ratio, 4)
        d["scene_volume_m3"] = round(self.scene_volume_m3, 4)
        d["mean_track_length"] = round(self.mean_track_length, 2)
        d["redundancy_ratio_3plus"] = round(self.redundancy_ratio_3plus, 4)
        d["completeness_score"] = round(self.completeness_score, 4)
        return d


def measure_reconstruction_completeness(
    result: ReconstructionResult,
    total_input_cameras: int,
    scale_metadata: Optional[Dict[str, Any]] = None,
    voxel_size_m: float = 0.25,
) -> CompletenessMetrics:
    """Measure reconstruction completeness, spatial voxel coverage, and scale reliability."""
    c_reg = len(result.camera_poses)
    c_in = max(c_reg, total_input_cameras)
    reg_ratio = c_reg / c_in if c_in > 0 else 0.0

    points = result.points
    num_points = len(points)

    if not points:
        return CompletenessMetrics(
            cameras_input=c_in,
            cameras_registered=c_reg,
            registration_ratio=reg_ratio,
            points_total=0,
            scene_bounds_min=(0.0, 0.0, 0.0),
            scene_bounds_max=(0.0, 0.0, 0.0),
            scene_volume_m3=0.0,
            voxel_occupancy_count=0,
            voxel_size_m=voxel_size_m,
            mean_track_length=0.0,
            redundancy_ratio_3plus=0.0,
            scale_state="none",
            scale_reliability="none",
            completeness_score=0.0,
        )

    coords = np.array([p.position for p in points], dtype=np.float64)
    b_min = tuple(float(x) for x in coords.min(axis=0))
    b_max = tuple(float(x) for x in coords.max(axis=0))

    dx = max(0.01, b_max[0] - b_min[0])
    dy = max(0.01, b_max[1] - b_min[1])
    dz = max(0.01, b_max[2] - b_min[2])
    volume = dx * dy * dz

    # Voxel occupancy: quantized 3D grid
    voxel_indices = np.floor(coords / voxel_size_m).astype(np.int64)
    unique_voxels = len(np.unique(voxel_indices, axis=0))

    # Track length statistics
    track_lengths = [len(getattr(p, "source_evidence_ids", []) or []) for p in points]
    if any(track_lengths):
        mean_track = float(np.mean(track_lengths))
        redundant_count = sum(1 for tl in track_lengths if tl >= 3)
        redundancy_ratio = redundant_count / num_points
    else:
        mean_track = 2.0  # default minimum two views
        redundancy_ratio = 0.5

    # Scale reliability classification
    scale_meta = scale_metadata or {}
    scale_state = str(scale_meta.get("state") or "relative").lower()
    if scale_state == "metric":
        basis = str(scale_meta.get("note") or scale_meta.get("basis") or "").lower()
        if "tape" in basis or "ground_truth" in basis or "measured" in basis:
            scale_reliability = "tape_verified"
        elif "device_depth" in basis or "lidar" in basis:
            scale_reliability = "device_lidar"
        else:
            scale_reliability = "estimated_prior"
    else:
        scale_reliability = "relative_arbitrary"

    # Composite completeness score (harmonic balance of registration ratio, track redundancy, and voxel support)
    # Scaled from 0.0 to 1.0
    voxel_score = min(1.0, unique_voxels / 200.0)
    composite = (reg_ratio * 0.4) + (redundancy_ratio * 0.3) + (voxel_score * 0.3)

    return CompletenessMetrics(
        cameras_input=c_in,
        cameras_registered=c_reg,
        registration_ratio=reg_ratio,
        points_total=num_points,
        scene_bounds_min=b_min,
        scene_bounds_max=b_max,
        scene_volume_m3=volume,
        voxel_occupancy_count=unique_voxels,
        voxel_size_m=voxel_size_m,
        mean_track_length=mean_track,
        redundancy_ratio_3plus=redundancy_ratio,
        scale_state=scale_state,
        scale_reliability=scale_reliability,
        completeness_score=min(1.0, max(0.0, composite)),
    )
