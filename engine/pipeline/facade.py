"""API-facing facade over engine/libraries.

The API layer (apps/api) must not import perception/reconstruction/model
backends directly (architecture layer rule). This module exposes thin,
pure-Python wrappers over the small set of library functions the API needs,
so apps/api depends on engine/pipeline instead of reaching into sibling
packages.

All functions here are synchronous and side-effect free: they take a
WorldIR (or data already in hand) and return a value. Heavy compute
(COLMAP, ML models) stays behind the pipeline / job runner.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from perception.architecture.space_graph import InteriorSpaceGraph
from perception.reasoning.physical_plausibility import validate_physical_plausibility
from perception.style.analyzer import analyze_world_design_style
from reconstruction.capture_quality import (
    assess_capture,
    pick_keyframes,
    score_video_frames,
    summarize_stored,
    wall_coverage,
)
from reconstruction.scale import (
    KNOWN_LENGTH,
    MODEL_SPACE_METHODS,
    PRIOR_ACCEPTED,
    ScaleAnchoringError,
    ScaleReference,
    gate_ratios,
    suggest_scale_from_world,
)
from reconstruction.scale.base import ScaleSuggestion
from world_ir.world_v1 import WorldIR
from reconstruction.calibration.camera import camera_from_pose


def create_interior_space_graph(graph_dict: dict) -> InteriorSpaceGraph:
    """Build an InteriorSpaceGraph from a serialized WorldIR space-graph dict."""
    return InteriorSpaceGraph.from_dict(graph_dict)


def suggest_scale_for_world(world: WorldIR) -> Optional[ScaleSuggestion]:
    """Suggest a metres-per-unit scale from a RELATIVE compiled WorldIR.

    Returns a :class:`ScaleSuggestion` or None when the world is already
    metric or no door with a measured height exists. Never applies
    anything -- the caller decides whether to accept.
    """
    return suggest_scale_from_world(world)


def scale_symbols() -> dict:
    """Return the scale-related symbols the API needs for route logic.

    Keys: KNOWN_LENGTH, MODEL_SPACE_METHODS, PRIOR_ACCEPTED,
    ScaleAnchoringError, ScaleReference, gate_ratios.
    """
    return {
        "KNOWN_LENGTH": KNOWN_LENGTH,
        "MODEL_SPACE_METHODS": MODEL_SPACE_METHODS,
        "PRIOR_ACCEPTED": PRIOR_ACCEPTED,
        "ScaleAnchoringError": ScaleAnchoringError,
        "ScaleReference": ScaleReference,
        "gate_ratios": gate_ratios,
    }


def validate_world_plausibility(world: WorldIR) -> Any:
    """Run physical-plausibility validation on a compiled world."""
    return validate_physical_plausibility(world)


def analyze_world_style(world: WorldIR) -> Any:
    """Analyze the architectural design style of a compiled world."""
    return analyze_world_design_style(world)


# --- capture quality --------------------------------------------------

def assess_capture_quality(photos: Sequence[tuple[str, bytes]]) -> dict:
    """Assess capture quality for a list of (path, bytes) photo tuples."""
    return assess_capture(photos)


def summarize_capture_quality(per_photo: Any) -> dict:
    """Summarize per-photo capture-quality results."""
    return summarize_stored(per_photo)


def score_and_pick_video_frames(
    path: str, frame_count: int, budget: int
) -> tuple[list, list]:
    """Score video frames and pick keyframes; returns (scores, chosen_indices)."""
    scores = score_video_frames(path, frame_count, budget)
    idx = pick_keyframes(scores, budget)
    return scores, idx


def wall_coverage_for_world(polygon_xyz: Any, cameras: Any, basis: Any = None) -> Any:
    """Compute wall coverage from a room footprint polygon and camera poses."""
    return wall_coverage(polygon_xyz, cameras, basis)


def camera_from_pose_for_world(intrinsics: Any, pose: Any) -> PinholeCamera:
    """Build a PinholeCamera from a backend camera pose."""
    return camera_from_pose(intrinsics, pose)
