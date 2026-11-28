"""Architectural Elevation & Room-Envelope Analyzer (A4.3 / A4.4).

Computes image-space elevation profiles for walls and openings:
  - Left-to-right wall ordering across the image horizon
  - Opening-to-wall area ratios & sill/lintel normalized height bands
  - Room-envelope completeness scoring (floor, ceiling, walls, openings)
  - Structural bay / span segmentation for columns & beams

All quantities remain explicitly in ``image_space`` unless metric poses and
scale are supplied externally.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from perception.intelligence.data_model import NormalizedPerceptionObservation


@dataclass(frozen=True)
class WallElevationProfile:
    wall_id: str
    image_id: str
    x_span_norm: tuple[float, float]
    y_span_norm: tuple[float, float]
    contained_opening_ids: tuple[str, ...]
    opening_to_wall_area_ratio: float
    sill_height_norm: float | None
    lintel_height_norm: float | None
    is_metric: bool = False


@dataclass(frozen=True)
class RoomEnvelopeCompleteness:
    image_id: str
    has_floor: bool
    has_ceiling: bool
    wall_count: int
    opening_count: int
    column_or_beam_count: int
    completeness_score: float
    missing_elements: tuple[str, ...]


@dataclass(frozen=True)
class ArchitecturalElevationReport:
    profiles: tuple[WallElevationProfile, ...]
    envelopes: tuple[RoomEnvelopeCompleteness, ...]
    ordered_wall_ids_by_image: dict[str, tuple[str, ...]]
    summary: dict[str, Any]


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _ioa(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(1e-6, _box_area(inner))
    return (ix * iy) / area


def analyze_architectural_elevations(
    observations: Sequence[NormalizedPerceptionObservation],
) -> ArchitecturalElevationReport:
    """Build image-space wall elevation profiles and room-envelope completeness scores."""
    by_img: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for o in observations:
        by_img[o.provenance.source_image_id].append(o)

    profiles: list[WallElevationProfile] = []
    envelopes: list[RoomEnvelopeCompleteness] = []
    ordered_walls: dict[str, tuple[str, ...]] = {}

    for img_id, obs_list in sorted(by_img.items()):
        walls = sorted(
            [o for o in obs_list if o.canonical_label in ("wall", "wall_segment")],
            key=lambda w: 0.5 * (w.bbox_norm[0] + w.bbox_norm[2]),
        )
        ordered_walls[img_id] = tuple(w.observation_id for w in walls)

        openings = [o for o in obs_list if o.canonical_label in ("door", "window", "opening")]
        floors = [o for o in obs_list if o.canonical_label == "floor"]
        ceilings = [o for o in obs_list if o.canonical_label == "ceiling"]
        struct_elems = [o for o in obs_list if o.canonical_label in ("column", "beam", "stairs")]

        for w in walls:
            w_area = max(1e-6, _box_area(w.bbox_px))
            contained: list[NormalizedPerceptionObservation] = []
            for op in openings:
                if _ioa(op.bbox_px, w.bbox_px) >= 0.35:
                    contained.append(op)
            op_area = sum(_box_area(op.bbox_px) for op in contained)
            ratio = round(min(1.0, op_area / w_area), 4)

            sill_vals = [1.0 - op.bbox_norm[3] for op in contained if op.canonical_label == "window"]
            lintel_vals = [1.0 - op.bbox_norm[1] for op in contained]
            profiles.append(WallElevationProfile(
                wall_id=w.observation_id,
                image_id=img_id,
                x_span_norm=(round(w.bbox_norm[0], 4), round(w.bbox_norm[2], 4)),
                y_span_norm=(round(w.bbox_norm[1], 4), round(w.bbox_norm[3], 4)),
                contained_opening_ids=tuple(op.observation_id for op in contained),
                opening_to_wall_area_ratio=ratio,
                sill_height_norm=round(min(sill_vals), 4) if sill_vals else None,
                lintel_height_norm=round(max(lintel_vals), 4) if lintel_vals else None,
                is_metric=False,
            ))

        has_floor = len(floors) > 0
        has_ceiling = len(ceilings) > 0
        missing: list[str] = []
        if not has_floor:
            missing.append("floor")
        if not has_ceiling:
            missing.append("ceiling")
        if len(walls) < 2:
            missing.append("multi_wall_boundary")
        if not openings:
            missing.append("openings")

        score = (
            (0.25 if has_floor else 0.0)
            + (0.20 if has_ceiling else 0.0)
            + min(0.35, 0.175 * len(walls))
            + min(0.20, 0.10 * len(openings))
        )
        envelopes.append(RoomEnvelopeCompleteness(
            image_id=img_id,
            has_floor=has_floor,
            has_ceiling=has_ceiling,
            wall_count=len(walls),
            opening_count=len(openings),
            column_or_beam_count=len(struct_elems),
            completeness_score=round(min(1.0, score), 4),
            missing_elements=tuple(missing),
        ))

    mean_comp = (
        round(sum(e.completeness_score for e in envelopes) / len(envelopes), 4)
        if envelopes
        else 0.0
    )
    return ArchitecturalElevationReport(
        profiles=tuple(profiles),
        envelopes=tuple(envelopes),
        ordered_wall_ids_by_image=ordered_walls,
        summary={
            "images_analyzed": len(envelopes),
            "wall_elevation_profiles": len(profiles),
            "mean_envelope_completeness": mean_comp,
            "is_metric": False,
        },
    )


__all__ = [
    "ArchitecturalElevationReport",
    "RoomEnvelopeCompleteness",
    "WallElevationProfile",
    "analyze_architectural_elevations",
]
