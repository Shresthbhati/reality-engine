"""Compatibility exports for the pipeline's spatial-global job helpers."""

from engine.pipeline.spatial_global_stage import (
    ENV_FLAG,
    ENV_SOLVER,
    SpatialGlobalError,
    enabled,
    facts_for_report,
    inputs_for_job,
    open_store,
    register_inputs_provider,
    solver,
    store_dir,
)

__all__ = [
    "ENV_FLAG",
    "ENV_SOLVER",
    "SpatialGlobalError",
    "enabled",
    "facts_for_report",
    "inputs_for_job",
    "open_store",
    "register_inputs_provider",
    "solver",
    "store_dir",
]
