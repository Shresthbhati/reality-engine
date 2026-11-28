#!/usr/bin/env python3
"""Phase 2: Baselines and test harness for ML Reconstruction.

Creates repeatable measurements for camera registration, reprojection error,
reconstruction geometry, wall/opening dimensions, entity identity, occlusion,
and runtime/memory metrics.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CameraRegistrationMetrics:
    """Metrics for camera registration quality."""
    num_cameras: int
    num_registered: int
    registration_ratio: float
    median_rotation_error_deg: float
    median_translation_error: float
    inlier_count: int
    inlier_ratio: float


@dataclass
class ReprojectionErrorMetrics:
    """Metrics for reprojection error after alignment."""
    mean_error_pixels: float
    median_error_pixels: float
    rmse_pixels: float
    percent_within_1px: float
    percent_within_5px: float
    percent_within_10px: float
    mean_error_meters: float
    median_error_meters: float


@dataclass
class ReconstructionGeometryMetrics:
    """Metrics for reconstruction geometry quality."""
    num_points: int
    num_cameras: int
    point_density_per_cam: float
    bounding_box_meters: Tuple[float, float, float]
    median_point_distance_m: float
    min_point_distance_m: float
    max_point_distance_m: float
    surface_coverage_percent: float


@dataclass
class WallOpeningMetrics:
    """Metrics for wall and opening dimensions."""
    wall_count: int
    opening_count: int
    wall_heights_m: List[float]
    wall_widths_m: List[float]
    opening_widths_m: List[float]
    avg_wall_height_m: float
    avg_opening_width_m: float
    dimension_errors_m: List[float]


@dataclass
class EntityIdentityMetrics:
    """Metrics for cross-view entity identity."""
    total_entities: int
    duplicate_entities: int
    incorrectly_merged: int
    correctly_separated: int
    identity_switches: int
    entity_types: Dict[str, int]


@dataclass
class OcclusionMetrics:
    """Metrics for occlusion analysis."""
    observed_fraction: float
    occluded_fraction: float
    unobserved_fraction: float
    shadowed_fraction: float
    visibility_scores: Dict[str, float]


@dataclass
class RuntimeMemoryMetrics:
    """Metrics for runtime and memory."""
    total_runtime_seconds: float
    per_stage_runtime: Dict[str, float]
    peak_memory_mb: float
    mean_memory_mb: float
    memory_samples: List[float]