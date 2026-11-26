"""Indoor Architectural Intelligence Engine (A3.2).

Recognizes and distinguishes:
  - Primary surfaces: walls, wall segments, corners, floor planes, ceiling planes
  - Structural supports: columns, beams
  - Openings & framing: door openings, windows, door frames, window frames, window sills, structural openings
  - Circulation & boundaries: stairs, railings, structural boundaries, room boundaries

Enforces honesty invariants:
  - A 2D rectangular detection is an ``opening_candidate`` until corroborated by
    frame parts, multi-view evidence, or depth discontinuity; it is never prematurely
    promoted to a ``verified_opening``.
  - Host-wall association is only asserted when an explicit wall observation
    contains/hosts the opening (IoA >= 0.35); otherwise ``host_wall_hypothesis_state="unknown"``.
  - Partial/truncated observations remain flagged ``is_partial=True``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import compute_box_iou
from perception.intelligence.data_model import (
    ArchitecturalObservation,
    NormalizedPerceptionObservation,
)

PRIMARY_SURFACES = frozenset({"wall", "wall_segment", "floor", "ceiling", "corner"})
STRUCTURAL_SUPPORTS = frozenset({"column", "beam"})
OPENING_CLASSES = frozenset({"door", "window", "opening", "structural_opening", "doorway", "archway"})
OPENING_FRAME_CLASSES = frozenset({"door_frame", "window_frame", "window_sill", "window_mullion", "window_sash"})
CIRCULATION_CLASSES = frozenset({"staircase", "stairs", "railing"})
BOUNDARY_CLASSES = frozenset({"structural_boundary", "room_boundary", "baseboard"})

ALL_ARCHITECTURAL_CLASSES = (
    PRIMARY_SURFACES
    | STRUCTURAL_SUPPORTS
    | OPENING_CLASSES
    | OPENING_FRAME_CLASSES
    | CIRCULATION_CLASSES
    | BOUNDARY_CLASSES
)


@dataclass(frozen=True)
class ArchitecturalAnalysisResult:
    observations: tuple[ArchitecturalObservation, ...]
    wall_host_links: dict[str, str]
    opening_to_frame_links: dict[str, tuple[str, ...]]
    competing_structural_pairs: tuple[tuple[str, str, float], ...]
    summary: dict[str, Any]


def _box_ioa(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    inter = ix * iy
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def _boundary_continuity(obs: NormalizedPerceptionObservation, peers: Sequence[NormalizedPerceptionObservation]) -> float:
    """Estimate boundary continuity [0, 1] from alignment with adjacent same-class wall/floor/ceiling segments."""
    if obs.canonical_label not in PRIMARY_SURFACES | BOUNDARY_CLASSES:
        return 0.0
    best = 0.25 if not obs.is_truncated else 0.15
    for peer in peers:
        if peer.observation_id == obs.observation_id:
            continue
        if peer.coords.frame_id != obs.coords.frame_id:
            continue
        if peer.canonical_label != obs.canonical_label:
            continue
        # Check horizontal or vertical edge proximity
        dx = min(abs(obs.bbox_px[2] - peer.bbox_px[0]), abs(peer.bbox_px[0] - peer.bbox_px[2]))
        dy = min(abs(obs.bbox_px[3] - peer.bbox_px[1]), abs(peer.bbox_px[1] - peer.bbox_px[3]))
        if dx < 30.0 or dy < 30.0:
            best = max(best, 0.75)
    return round(best, 4)


def analyze_architecture(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    min_wall_containment_ioa: float = 0.35,
) -> ArchitecturalAnalysisResult:
    """Analyze architectural observations, host-wall hypotheses, frames, and structural conflicts."""
    arch_obs = [o for o in observations if o.canonical_label in ALL_ARCHITECTURAL_CLASSES or o.category == "architecture"]
    by_image: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for o in arch_obs:
        by_image[o.provenance.source_image_id].append(o)

    wall_host_links: dict[str, str] = {}
    opening_to_frame: dict[str, list[str]] = defaultdict(list)
    competing_pairs: list[tuple[str, str, float]] = []
    results: list[ArchitecturalObservation] = []

    for img_id, img_obs in by_image.items():
        walls = [o for o in img_obs if o.canonical_label in ("wall", "wall_segment")]
        openings = [o for o in img_obs if o.canonical_label in OPENING_CLASSES]
        frames = [o for o in img_obs if o.canonical_label in OPENING_FRAME_CLASSES]

        # Link frames to openings in the same image
        frame_parent: dict[str, str] = {}
        for fr in frames:
            best_op: str | None = None
            best_iou = 0.0
            for op in openings:
                iou = compute_box_iou(fr.bbox_px, op.bbox_px)
                ioa = _box_ioa(fr.bbox_px, op.bbox_px)
                if (iou >= 0.30 or ioa >= 0.40) and max(iou, ioa) > best_iou:
                    best_iou = max(iou, ioa)
                    best_op = op.observation_id
            if best_op is not None:
                frame_parent[fr.observation_id] = best_op
                opening_to_frame[best_op].append(fr.observation_id)

        # Detect competing structural hypotheses on overlapping regions (e.g. door vs window, wall vs door)
        competing_map: dict[str, list[str]] = defaultdict(list)
        for i, a in enumerate(img_obs):
            for b in img_obs[i + 1:]:
                if a.canonical_label == b.canonical_label:
                    continue
                if a.observation_id in frame_parent and frame_parent[a.observation_id] == b.observation_id:
                    continue
                if b.observation_id in frame_parent and frame_parent[b.observation_id] == a.observation_id:
                    continue
                # A wall hosting an opening is expected, not a contradiction
                if {a.canonical_label, b.canonical_label} & {"wall", "wall_segment"} and {a.canonical_label, b.canonical_label} & OPENING_CLASSES:
                    continue
                iou = compute_box_iou(a.bbox_px, b.bbox_px)
                if iou >= 0.55:
                    competing_pairs.append((a.observation_id, b.observation_id, round(iou, 4)))
                    competing_map[a.observation_id].append(b.canonical_label)
                    competing_map[b.observation_id].append(a.canonical_label)

        for o in img_obs:
            lbl = o.canonical_label
            host_wall_id: str | None = None
            host_state = "unknown"
            is_verified_opening = False

            if lbl in PRIMARY_SURFACES:
                role = "primary_surface"
            elif lbl in STRUCTURAL_SUPPORTS:
                role = "structural_support"
            elif lbl in OPENING_CLASSES:
                # Check if a wall in this image hosts the opening
                best_w_id: str | None = None
                best_ioa = 0.0
                for w in walls:
                    ioa = _box_ioa(o.bbox_px, w.bbox_px)
                    if ioa >= min_wall_containment_ioa and ioa > best_ioa:
                        best_ioa = ioa
                        best_w_id = w.observation_id
                if best_w_id is not None:
                    host_wall_id = best_w_id
                    host_state = "observed"
                    wall_host_links[o.observation_id] = best_w_id

                # An opening is only "verified_opening" if corroborated by a frame part,
                # segmentation mask, or explicit 3D/depth support (never from a bare 2D box alone)
                has_frame = bool(opening_to_frame.get(o.observation_id))
                has_mask = o.segmentation is not None and o.segmentation.quality_score >= 0.45
                is_verified_opening = has_frame or has_mask
                role = "verified_opening" if is_verified_opening else "opening_candidate"
            elif lbl in OPENING_FRAME_CLASSES:
                role = "frame"
                if o.observation_id in frame_parent:
                    parent_op = frame_parent[o.observation_id]
                    if parent_op in wall_host_links:
                        host_wall_id = wall_host_links[parent_op]
                        host_state = "hypothesized"
            elif lbl in CIRCULATION_CLASSES:
                role = "circulation"
            else:
                role = "boundary"

            cont = _boundary_continuity(o, img_obs)
            results.append(ArchitecturalObservation(
                observation=o,
                structural_role=role,
                host_wall_id=host_wall_id,
                host_wall_hypothesis_state=host_state,
                boundary_continuity_score=cont,
                is_verified_opening=is_verified_opening,
                competing_structural_labels=tuple(sorted(set(competing_map.get(o.observation_id, ())))),
            ))

    summary = {
        "total_architectural": len(results),
        "primary_surfaces": sum(1 for r in results if r.structural_role == "primary_surface"),
        "opening_candidates": sum(1 for r in results if r.structural_role == "opening_candidate"),
        "verified_openings": sum(1 for r in results if r.structural_role == "verified_opening"),
        "host_wall_linked": len(wall_host_links),
        "competing_structural_pairs": len(competing_pairs),
    }
    return ArchitecturalAnalysisResult(
        observations=tuple(results),
        wall_host_links=wall_host_links,
        opening_to_frame_links={k: tuple(v) for k, v in opening_to_frame.items()},
        competing_structural_pairs=tuple(competing_pairs),
        summary=summary,
    )


__all__ = [
    "ALL_ARCHITECTURAL_CLASSES",
    "ArchitecturalAnalysisResult",
    "OPENING_CLASSES",
    "OPENING_FRAME_CLASSES",
    "PRIMARY_SURFACES",
    "analyze_architecture",
]
