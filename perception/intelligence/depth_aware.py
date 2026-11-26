"""Depth-Aware Perception Subsystem (A4.7).

Integrates depth outputs from any depth provider through a common contract
without claiming ownership of depth reconstruction, without treating relative
depth as metric depth, and without mixing incompatible depth conventions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.intelligence.data_model import NormalizedPerceptionObservation


@dataclass(frozen=True)
class DepthMapContract:
    image_id: str
    depth_array: np.ndarray
    scale_state: str  # "relative" | "sfm_gauge" | "metric"
    units: str  # "relative" | "sfm_units" | "meters"
    model_id: str
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.scale_state == "metric" and self.units != "meters":
            raise ValueError("metric scale_state requires units='meters'")
        if self.scale_state == "relative" and self.units == "meters":
            raise ValueError("relative depth must never claim units='meters'")


@dataclass(frozen=True)
class ObjectDepthSummary:
    observation_id: str
    scale_state: str
    units: str
    valid_fraction: float
    median_depth: float | None
    iqr_depth: float | None
    boundary_discontinuity_score: float
    relative_rank: int | None = None
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class DepthAwarePerceptionResult:
    image_id: str
    scale_state: str
    units: str
    object_depths: tuple[ObjectDepthSummary, ...]
    front_to_back_order: tuple[str, ...]
    inconsistencies: tuple[dict[str, Any], ...]


def associate_observations_with_depth(
    observations: Sequence[NormalizedPerceptionObservation],
    depth_contract: DepthMapContract,
) -> DepthAwarePerceptionResult:
    """Associate detections/masks with depth while enforcing scale-state honesty."""
    dmap = np.asarray(depth_contract.depth_array, dtype=np.float32)
    dh, dw = dmap.shape[:2]
    base_valid = np.isfinite(dmap) & (dmap > 0.0)
    if depth_contract.valid_mask is not None and depth_contract.valid_mask.shape[:2] == (dh, dw):
        base_valid &= np.asarray(depth_contract.valid_mask, dtype=bool)

    summaries: list[ObjectDepthSummary] = []
    inconsistencies: list[dict[str, Any]] = []

    for o in observations:
        if o.provenance.source_image_id != depth_contract.image_id:
            continue
        sx = dw / float(o.coords.image_width)
        sy = dh / float(o.coords.image_height)
        x1 = max(0, min(dw - 1, int(round(o.bbox_px[0] * sx))))
        y1 = max(0, min(dh - 1, int(round(o.bbox_px[1] * sy))))
        x2 = max(0, min(dw, int(round(o.bbox_px[2] * sx))))
        y2 = max(0, min(dh, int(round(o.bbox_px[3] * sy))))

        diags: list[str] = []
        if x2 <= x1 or y2 <= y1:
            summaries.append(ObjectDepthSummary(
                o.observation_id, depth_contract.scale_state, depth_contract.units,
                0.0, None, None, 0.0, diagnostics=("degenerate_projected_box",),
            ))
            continue

        patch = dmap[y1:y2, x1:x2]
        vpatch = base_valid[y1:y2, x1:x2]
        valid_frac = float(np.mean(vpatch)) if vpatch.size > 0 else 0.0
        vals = patch[vpatch]

        if vals.size < 4 or valid_frac < 0.15:
            diags.append("insufficient_valid_depth_pixels")
            summaries.append(ObjectDepthSummary(
                o.observation_id, depth_contract.scale_state, depth_contract.units,
                round(valid_frac, 4), None, None, 0.0, diagnostics=tuple(diags),
            ))
            continue

        med = float(np.median(vals))
        q75, q25 = float(np.percentile(vals, 75)), float(np.percentile(vals, 25))
        iqr = max(0.0, q75 - q25)

        # Boundary discontinuity: compare interior depth vs outer ring if box allows
        disc = 0.0
        if (x2 - x1) >= 8 and (y2 - y1) >= 8:
            inner = patch[2:-2, 2:-2][vpatch[2:-2, 2:-2]]
            if inner.size > 0:
                disc = round(abs(float(np.median(inner)) - med) / max(1e-4, med), 4)

        summaries.append(ObjectDepthSummary(
            observation_id=o.observation_id,
            scale_state=depth_contract.scale_state,
            units=depth_contract.units,
            valid_fraction=round(valid_frac, 4),
            median_depth=round(med, 4),
            iqr_depth=round(iqr, 4),
            boundary_discontinuity_score=disc,
            diagnostics=tuple(diags),
        ))

    # Relative front-to-back ordering (only valid when all items share the same depth map)
    with_med = [s for s in summaries if s.median_depth is not None]
    with_med.sort(key=lambda s: float(s.median_depth))
    rank_map = {s.observation_id: idx + 1 for idx, s in enumerate(with_med)}
    ranked = tuple(
        ObjectDepthSummary(
            s.observation_id, s.scale_state, s.units, s.valid_fraction,
            s.median_depth, s.iqr_depth, s.boundary_discontinuity_score,
            relative_rank=rank_map.get(s.observation_id), diagnostics=s.diagnostics,
        )
        for s in summaries
    )
    return DepthAwarePerceptionResult(
        image_id=depth_contract.image_id,
        scale_state=depth_contract.scale_state,
        units=depth_contract.units,
        object_depths=ranked,
        front_to_back_order=tuple(s.observation_id for s in with_med),
        inconsistencies=tuple(inconsistencies),
    )


__all__ = [
    "DepthAwarePerceptionResult",
    "DepthMapContract",
    "ObjectDepthSummary",
    "associate_observations_with_depth",
]
