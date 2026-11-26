"""Pipeline-facing adapters for perception services used by the API worker."""

from __future__ import annotations

from typing import Any

from perception.spatial_adapter import adapt_spatial_inputs_v1
from perception.spatial_inputs import (
    SPATIAL_INPUTS_FILE,
    CameraInput,
    ScaleStateInput,
    SpatialInputsV1,
    StructuralInput,
)

__all__ = [
    "SPATIAL_INPUTS_FILE",
    "CameraInput",
    "ScaleStateInput",
    "SpatialInputsV1",
    "StructuralInput",
    "adapt_spatial_inputs_v1",
    "extract_structural_regions",
    "get_inputs_provider",
    "process_video_file",
    "run_inference_plan",
    "run_multi_engine_for_job",
    "select_inference_plan_for_inputs",
    "world_lineage_block",
]


def process_video_file(*args: Any, **kwargs: Any) -> Any:
    from perception.evidence_enrichment import process_video_file as process

    return process(*args, **kwargs)


def run_multi_engine_for_job(*args: Any, **kwargs: Any) -> Any:
    from perception.multi_engine_stage import run_multi_engine_for_job as run

    return run(*args, **kwargs)


def world_lineage_block(*args: Any, **kwargs: Any) -> Any:
    from perception.engine_matrix_bridge import world_lineage_block as build

    return build(*args, **kwargs)


def get_inputs_provider(*args: Any, **kwargs: Any) -> Any:
    from perception.spatial_inputs import get_inputs_provider as get

    return get(*args, **kwargs)


def run_inference_plan(*args: Any, **kwargs: Any) -> Any:
    from perception.orchestrator import run_inference_plan as run

    return run(*args, **kwargs)


def select_inference_plan_for_inputs(*args: Any, **kwargs: Any) -> Any:
    from perception.orchestrator import select_inference_plan_for_inputs as select

    return select(*args, **kwargs)


def extract_structural_regions(*args: Any, **kwargs: Any) -> Any:
    from perception.scene_intelligence import extract_structural_regions as extract

    return extract(*args, **kwargs)
