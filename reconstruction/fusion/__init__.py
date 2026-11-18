"""Evidence fusion public API (spec sec 14)."""

from reconstruction.fusion.fusion import (
    CONFLICT_SIGMA,
    FusableObservation,
    FusedEstimate,
    FusionError,
    MeasurementConflict,
    fused_to_measurement,
    fuse_quantity,
)
from reconstruction.fusion.robust_fusion import (
    RobustFusionResult,
    fuse_multisensor_points_3d,
    robust_fuse_depth_observations,
    robust_fuse_quantity,
)

__all__ = [
    "CONFLICT_SIGMA",
    "FusableObservation",
    "FusedEstimate",
    "FusionError",
    "MeasurementConflict",
    "RobustFusionResult",
    "fuse_multisensor_points_3d",
    "fused_to_measurement",
    "fuse_quantity",
    "robust_fuse_depth_observations",
    "robust_fuse_quantity",
]
