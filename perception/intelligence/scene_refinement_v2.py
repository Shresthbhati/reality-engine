"""A5.4 & A5.5 Architectural & Furniture Observation Quality Refinement.

Reduces false structural and furniture detections using actual image evidence,
box geometry, mask solidity, and semantic/spatial constraints:
  - Architectural refinement: verifies walls, floors, ceilings, doors, windows,
    frames, columns, beams, and built-in furniture; rejects sub-threshold/tiny
    spurious structural boxes; never invents a wall from line segments alone.
  - Furniture refinement: prioritizes interior survey furniture (tables, chairs,
    sofas, beds, desks, cabinets, wardrobes, shelving, appliances), rejects
    implausible full-frame furniture false positives, and attaches subpart hints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from perception.intelligence.data_model import NormalizedPerceptionObservation

ARCHITECTURAL_SURVEY_CLASSES: frozenset[str] = frozenset({
    "wall", "wall_segment", "floor", "ceiling", "door", "door_frame",
    "window", "window_frame", "window_sill", "opening", "column", "beam",
    "stairs", "railing", "roof",
})

INTERIOR_SURVEY_FURNITURE: frozenset[str] = frozenset({
    "table", "dining_table", "coffee_table", "chair", "dining_chair",
    "armchair", "office_chair", "stool", "sofa", "couch", "bed", "desk",
    "cabinet", "wardrobe", "shelf", "bookshelf", "storage_unit",
    "refrigerator", "oven", "microwave", "sink", "toilet", "bathtub",
    "countertop", "kitchen_furniture",
})

FURNITURE_SUBPARTS: dict[str, tuple[str, ...]] = {
    "chair": ("chair_seat", "chair_back", "chair_legs"),
    "dining_chair": ("chair_seat", "chair_back", "chair_legs"),
    "office_chair": ("chair_seat", "chair_back", "chair_pedestal"),
    "table": ("tabletop", "table_legs"),
    "dining_table": ("tabletop", "table_legs"),
    "desk": ("desktop", "desk_legs_or_pedestal"),
    "sofa": ("sofa_cushion", "sofa_backrest", "sofa_armrest"),
    "bed": ("mattress", "headboard", "bed_frame"),
    "cabinet": ("cabinet_door", "cabinet_carcass"),
    "wardrobe": ("wardrobe_door", "wardrobe_frame"),
    "door": ("door_leaf", "door_handle"),
    "window": ("window_pane", "window_frame"),
}


@dataclass(frozen=True)
class SceneRefinementReport:
    accepted_observations: tuple[NormalizedPerceptionObservation, ...]
    rejected_architectural_ids: tuple[str, ...]
    rejected_furniture_ids: tuple[str, ...]
    rejection_reasons: dict[str, str]
    summary: dict[str, Any]


def _box_area_fraction(obs: NormalizedPerceptionObservation) -> float:
    w = max(0.0, obs.bbox_norm[2] - obs.bbox_norm[0])
    h = max(0.0, obs.bbox_norm[3] - obs.bbox_norm[1])
    return w * h


def _crop_edge_density(
    image_rgb: np.ndarray | None,
    bbox_px: tuple[float, float, float, float],
) -> float | None:
    if image_rgb is None or image_rgb.ndim != 3:
        return None
    ih, iw = image_rgb.shape[:2]
    x1 = max(0, min(iw - 1, int(round(bbox_px[0]))))
    y1 = max(0, min(ih - 1, int(round(bbox_px[1]))))
    x2 = max(x1 + 2, min(iw, int(round(bbox_px[2]))))
    y2 = max(y1 + 2, min(ih, int(round(bbox_px[3]))))
    crop = image_rgb[y1:y2, x1:x2]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return 0.0
    gray = crop.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edges = (gx[:-1, :] + gy[:, :-1]) > 14.0
    return round(float(edges.mean()), 4)


def refine_scene_observations_v2(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    images_rgb: Mapping[str, np.ndarray] | None = None,
    min_confidence: float = 0.22,
) -> SceneRefinementReport:
    """Filter architectural & furniture false positives using image evidence, geometry, and masks."""
    accepted: list[NormalizedPerceptionObservation] = []
    rej_arch: list[str] = []
    rej_furn: list[str] = []
    reasons: dict[str, str] = {}

    for o in observations:
        lbl = o.canonical_label.lower()
        area_frac = _box_area_fraction(o)
        raw_sc = o.scores.raw_score
        img = (images_rgb or {}).get(o.provenance.source_image_id)
        edge_dens = _crop_edge_density(img, o.bbox_px)

        if raw_sc is not None and raw_sc < min_confidence:
            reasons[o.observation_id] = f"below_min_confidence:{raw_sc:.3f}"
            if o.category == "architecture" or lbl in ARCHITECTURAL_SURVEY_CLASSES:
                rej_arch.append(o.observation_id)
            else:
                rej_furn.append(o.observation_id)
            continue

        # Architectural quality checks
        if o.category == "architecture" or lbl in ARCHITECTURAL_SURVEY_CLASSES:
            # Tiny sub-0.3% box cannot be a primary wall/floor/ceiling surface
            if lbl in ("wall", "wall_segment", "floor", "ceiling") and area_frac < 0.003:
                rej_arch.append(o.observation_id)
                reasons[o.observation_id] = f"tiny_architectural_surface:{area_frac:.5f}"
                continue
            # Column/beam must not be full-frame square (>75% image area)
            if lbl in ("column", "beam") and area_frac > 0.75:
                rej_arch.append(o.observation_id)
                reasons[o.observation_id] = f"implausible_column_or_beam_area:{area_frac:.3f}"
                continue
            # Uniform/blank region false positive for openings/frames
            if lbl in ("door", "window", "door_frame", "window_frame") and edge_dens is not None and edge_dens < 0.005:
                rej_arch.append(o.observation_id)
                reasons[o.observation_id] = f"featureless_opening_crop:edge_density={edge_dens}"
                continue
            if o.segmentation is not None and o.segmentation.area_px > 0 and o.segmentation.solidity < 0.12:
                rej_arch.append(o.observation_id)
                reasons[o.observation_id] = f"fragmented_architectural_mask:solidity={o.segmentation.solidity}"
                continue
            accepted.append(o)
            continue

        # Furniture quality checks
        if lbl in INTERIOR_SURVEY_FURNITURE or o.category == "furniture":
            # Single movable furniture item covering >88% of image is almost always a false positive
            if area_frac > 0.88:
                rej_furn.append(o.observation_id)
                reasons[o.observation_id] = f"implausible_fullframe_furniture:{area_frac:.3f}"
                continue
            if area_frac < 0.0005:
                rej_furn.append(o.observation_id)
                reasons[o.observation_id] = f"subpixel_furniture_noise:{area_frac:.6f}"
                continue
            if edge_dens is not None and edge_dens < 0.003:
                rej_furn.append(o.observation_id)
                reasons[o.observation_id] = f"featureless_furniture_crop:edge_density={edge_dens}"
                continue
            subparts = FURNITURE_SUBPARTS.get(lbl, ())
            if subparts and "expected_subparts" not in o.attributes:
                attrs = dict(o.attributes)
                attrs["expected_subparts"] = list(subparts)
                attrs["is_survey_priority_furniture"] = True
                accepted.append(replace(o, attributes=attrs))
            else:
                accepted.append(o)
            continue

        accepted.append(o)

    return SceneRefinementReport(
        accepted_observations=tuple(accepted),
        rejected_architectural_ids=tuple(rej_arch),
        rejected_furniture_ids=tuple(rej_furn),
        rejection_reasons=reasons,
        summary={
            "input_count": len(observations),
            "accepted_count": len(accepted),
            "rejected_architectural_count": len(rej_arch),
            "rejected_furniture_count": len(rej_furn),
        },
    )


__all__ = [
    "ARCHITECTURAL_SURVEY_CLASSES",
    "FURNITURE_SUBPARTS",
    "INTERIOR_SURVEY_FURNITURE",
    "SceneRefinementReport",
    "refine_scene_observations_v2",
]
