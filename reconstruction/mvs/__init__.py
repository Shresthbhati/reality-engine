"""Multi-View Stereo and dense reconstruction subsystem (Master Development Plan Tasks 4.4, 4.5, 4.8, Epic E03-1)."""

from .dense_cloud import DenseCloudResult, MultiViewDenseCloudBuilder
from .point_cloud_ops import (
    chamfer_distance,
    compute_point_cloud_density,
    estimate_point_normals,
    export_point_cloud_ply,
    f_score_tau,
    radius_outlier_removal,
    statistical_outlier_removal,
    voxel_downsample,
)

__all__ = [
    "DenseCloudResult",
    "MultiViewDenseCloudBuilder",
    "chamfer_distance",
    "compute_point_cloud_density",
    "estimate_point_normals",
    "export_point_cloud_ply",
    "f_score_tau",
    "radius_outlier_removal",
    "statistical_outlier_removal",
    "voxel_downsample",
]
