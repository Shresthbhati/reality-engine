"""Multi-View Perception Engine (A4.6).

Uses camera geometry supplied by Claude's reconstruction workstream (``CameraPoseConstraint``)
together with appearance embeddings, mask correspondence, visibility/occlusion tracking,
and architectural continuity to associate observations across views without creating a
competing reconstruction system.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from perception.intelligence.data_model import NormalizedPerceptionObservation
from perception.intelligence.identity import (
    CameraPoseConstraint,
    IdentityResolutionReport,
    resolve_cross_view_identities,
)


@dataclass(frozen=True)
class MultiViewPerceptionReport:
    identity_report: IdentityResolutionReport
    visibility_tracks: dict[str, tuple[str, ...]]  # identity_id -> ordered image_ids
    architectural_continuity: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def _camera_baseline_compat(
    cam_a: CameraPoseConstraint | None,
    cam_b: CameraPoseConstraint | None,
    obs_a: NormalizedPerceptionObservation,
    obs_b: NormalizedPerceptionObservation,
) -> float | None:
    """Geometric compatibility heuristic from camera baseline and normalized box centroids."""
    if cam_a is None or cam_b is None:
        return None
    dx = cam_a.position_m[0] - cam_b.position_m[0]
    dy = cam_a.position_m[1] - cam_b.position_m[1]
    dz = cam_a.position_m[2] - cam_b.position_m[2]
    baseline = math.sqrt(dx * dx + dy * dy + dz * dz)

    ca_x = 0.5 * (obs_a.bbox_norm[0] + obs_a.bbox_norm[2])
    ca_y = 0.5 * (obs_a.bbox_norm[1] + obs_a.bbox_norm[3])
    cb_x = 0.5 * (obs_b.bbox_norm[0] + obs_b.bbox_norm[2])
    cb_y = 0.5 * (obs_b.bbox_norm[1] + obs_b.bbox_norm[3])
    shift = math.hypot(ca_x - cb_x, ca_y - cb_y)

    # Small baseline (<0.5m) expects small image-space shift; larger baselines allow wider shifts
    tol = 0.25 + 0.15 * min(3.0, baseline)
    return round(max(0.0, min(1.0, 1.0 - shift / max(0.1, tol))), 4)


def run_multiview_perception(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    cameras: Mapping[str, CameraPoseConstraint] | None = None,
    match_threshold: float = 0.60,
) -> MultiViewPerceptionReport:
    """Run multi-view identity resolution, visibility tracking, and architectural continuity."""
    cam_map = cameras or {}

    def _geom_fn(a: NormalizedPerceptionObservation, b: NormalizedPerceptionObservation) -> float | None:
        return _camera_baseline_compat(
            cam_map.get(a.provenance.source_image_id),
            cam_map.get(b.provenance.source_image_id),
            a,
            b,
        )

    id_rep = resolve_cross_view_identities(
        observations,
        epipolar_fn=_geom_fn if cam_map else None,
        match_threshold=match_threshold,
    )
    obs_by_id = {o.observation_id: o for o in observations}

    vis_tracks: dict[str, tuple[str, ...]] = {}
    arch_cont: list[dict[str, Any]] = []
    for ident in id_rep.identities:
        imgs = tuple(sorted({obs_by_id[m].provenance.source_image_id for m in ident.member_observation_ids}))
        vis_tracks[ident.identity_id] = imgs
        if len(imgs) >= 2 and ident.canonical_label in ("wall", "floor", "ceiling", "door", "window", "column"):
            arch_cont.append({
                "identity_id": ident.identity_id,
                "canonical_label": ident.canonical_label,
                "views": list(imgs),
                "identity_score": ident.identity_score,
            })

    return MultiViewPerceptionReport(
        identity_report=id_rep,
        visibility_tracks=vis_tracks,
        architectural_continuity=tuple(arch_cont),
        summary={
            **id_rep.summary,
            "multi_view_tracked": sum(1 for v in vis_tracks.values() if len(v) >= 2),
            "continuous_architectural_elements": len(arch_cont),
        },
    )


__all__ = [
    "MultiViewPerceptionReport",
    "run_multiview_perception",
]
