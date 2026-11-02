"""Multi-View Stereo and dense reconstruction subsystem (Master Development Plan Tasks 4.4, 4.5, 4.8)."""

from .dense_cloud import DenseCloudResult, MultiViewDenseCloudBuilder
from .point_cloud_ops import (
    estimate_point_normals,
    radius_outlier_removal,
    statistical_outlier_removal,
    voxel_downsample,
)

__all__ = [
    "DenseCloudResult",
    "MultiViewDenseCloudBuilder",
    "estimate_point_normals",
    "radius_outlier_removal",
    "statistical_outlier_removal",
    "voxel_downsample",
]
