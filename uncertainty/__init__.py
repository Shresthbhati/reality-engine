"""Uncertainty propagation (P10-02):
docs/future/uncertainty/UNCERTAINTY_PROPAGATION.md.

`Uncertain` is the canonical uncertain scalar (value + optional
standard deviation + mandatory derivation basis). The operators in
`propagation` compose uncertain quantities with standard first-order
error propagation over the repo's geometry types, and UNKNOWN
propagates as UNKNOWN through every operator."""

from uncertainty.propagation import (
    Covariance3,
    PropagationError,
    compose_pose_covariances,
    depth_to_world_covariance,
    difference,
    linear_propagate,
    propagate_chain,
    propagate_point_through_pose,
    rotate_covariance,
    scale,
    sum,
    transform_point_covariance,
)
from uncertainty.quantity import UNKNOWN, BASES, Uncertain

__all__ = [
    "UNKNOWN",
    "BASES",
    "Uncertain",
    "Covariance3",
    "PropagationError",
    "depth_to_world_covariance",
    "propagate_chain",
    "propagate_point_through_pose",
    "sum",
    "difference",
    "scale",
    "linear_propagate",
    "rotate_covariance",
    "transform_point_covariance",
    "compose_pose_covariances",
]
