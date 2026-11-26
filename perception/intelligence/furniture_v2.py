"""Furniture Perception 2.0 (A4.5).

Expands furniture intelligence beyond classification:
  - Subtype & part recognition
  - Instance separation & occlusion reasoning
  - Support, floor, and wall proximity relationships
  - Furniture arrangement grouping (e.g., dining_set, workstation, seating_group)
  - Explicit distinction between movable furniture, built-in furniture,
    appliances, fixtures, and architectural structures.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from perception.intelligence.data_model import NormalizedPerceptionObservation

MOBILITY_CLASSES: dict[str, str] = {
    "chair": "movable_furniture",
    "dining_chair": "movable_furniture",
    "armchair": "movable_furniture",
    "office_chair": "movable_furniture",
    "stool": "movable_furniture",
    "sofa": "movable_furniture",
    "couch": "movable_furniture",
    "bed": "movable_furniture",
    "table": "movable_furniture",
    "dining_table": "movable_furniture",
    "coffee_table": "movable_furniture",
    "desk": "movable_furniture",
    "cabinet": "movable_furniture",
    "wardrobe": "movable_furniture",
    "shelf": "movable_furniture",
    "storage_unit": "movable_furniture",
    "kitchen_furniture": "built_in_furniture",
    "countertop": "built_in_furniture",
    "refrigerator": "appliance",
    "oven": "appliance",
    "microwave": "appliance",
    "sink": "fixture",
    "toilet": "fixture",
    "bathtub": "fixture",
    "bathroom_fixture": "fixture",
    "lamp": "fixture",
    "ceiling_light": "fixture",
}


@dataclass(frozen=True)
class FurnitureArrangementGroup:
    group_id: str
    arrangement_kind: str  # "dining_or_meeting_set" | "workstation" | "seating_lounge"
    member_observation_ids: tuple[str, ...]
    image_id: str


@dataclass(frozen=True)
class Furniture2AnalysisResult:
    mobility_assignments: dict[str, str]
    arrangements: tuple[FurnitureArrangementGroup, ...]
    occlusion_pairs: tuple[tuple[str, str], ...]
    support_links: dict[str, str]  # object_id -> supporting_surface_id
    summary: dict[str, Any]


def classify_object_mobility(canonical_label: str, *, is_wall_contained: bool = False) -> str:
    lbl = canonical_label.lower().replace(" ", "_")
    if is_wall_contained and lbl in ("cabinet", "wardrobe", "shelf", "storage_unit"):
        return "built_in_furniture"
    return MOBILITY_CLASSES.get(lbl, "movable_furniture")


def analyze_furniture_v2(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    builtin_ids: Sequence[str] = (),
) -> Furniture2AnalysisResult:
    """Analyze furniture mobility, arrangements, occlusion pairs, and support surfaces."""
    builtin_set = set(builtin_ids)
    by_img: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for o in observations:
        by_img[o.provenance.source_image_id].append(o)

    mobility: dict[str, str] = {}
    arrangements: list[FurnitureArrangementGroup] = []
    occlusion_pairs: list[tuple[str, str]] = []
    support_links: dict[str, str] = {}

    for img_id, obs_list in sorted(by_img.items()):
        floors = [o for o in obs_list if o.canonical_label == "floor"]
        tables = [o for o in obs_list if o.canonical_label in ("table", "dining_table", "coffee_table")]
        desks = [o for o in obs_list if o.canonical_label == "desk"]
        chairs = [o for o in obs_list if o.canonical_label in ("chair", "dining_chair", "armchair", "office_chair", "stool")]
        sofas = [o for o in obs_list if o.canonical_label in ("sofa", "couch")]

        for o in obs_list:
            if o.category in ("furniture", "fixture", "appliance") or o.canonical_label in MOBILITY_CLASSES:
                mobility[o.observation_id] = classify_object_mobility(
                    o.canonical_label, is_wall_contained=(o.observation_id in builtin_set)
                )
                # Floor support link
                if floors and mobility[o.observation_id] in ("movable_furniture", "appliance"):
                    if o.bbox_norm[3] >= floors[0].bbox_norm[1] - 0.05:
                        support_links[o.observation_id] = floors[0].observation_id

        # Table-top / desk-top object support (e.g. lamp on table/desk)
        for o in obs_list:
            if o.canonical_label in ("lamp", "microwave"):
                for sup in [*tables, *desks]:
                    cx = 0.5 * (o.bbox_px[0] + o.bbox_px[2])
                    if sup.bbox_px[0] <= cx <= sup.bbox_px[2] and o.bbox_px[3] <= sup.bbox_px[3]:
                        support_links[o.observation_id] = sup.observation_id
                        break

        # Arrangement grouping: table + nearby chairs
        for tbl in tables:
            near_chairs = [
                c.observation_id for c in chairs
                if abs(0.5 * (c.bbox_norm[0] + c.bbox_norm[2]) - 0.5 * (tbl.bbox_norm[0] + tbl.bbox_norm[2])) <= 0.35
            ]
            if near_chairs:
                arrangements.append(FurnitureArrangementGroup(
                    group_id=f"arr_{img_id}_{tbl.observation_id}",
                    arrangement_kind="dining_or_meeting_set",
                    member_observation_ids=(tbl.observation_id, *near_chairs),
                    image_id=img_id,
                ))

        for dsk in desks:
            near_chairs = [
                c.observation_id for c in chairs
                if abs(0.5 * (c.bbox_norm[0] + c.bbox_norm[2]) - 0.5 * (dsk.bbox_norm[0] + dsk.bbox_norm[2])) <= 0.30
            ]
            if near_chairs:
                arrangements.append(FurnitureArrangementGroup(
                    group_id=f"arr_{img_id}_{dsk.observation_id}",
                    arrangement_kind="workstation",
                    member_observation_ids=(dsk.observation_id, *near_chairs),
                    image_id=img_id,
                ))

        if sofas and any(t.canonical_label == "coffee_table" for t in tables):
            ct = next(t for t in tables if t.canonical_label == "coffee_table")
            arrangements.append(FurnitureArrangementGroup(
                group_id=f"arr_{img_id}_lounge",
                arrangement_kind="seating_lounge",
                member_observation_ids=(ct.observation_id, *(s.observation_id for s in sofas)),
                image_id=img_id,
            ))

        # Occlusion pairs: overlapping foreground objects in the same image
        f_objs = [o for o in obs_list if o.observation_id in mobility]
        for i, a in enumerate(f_objs):
            for b in f_objs[i + 1:]:
                ix = max(0.0, min(a.bbox_px[2], b.bbox_px[2]) - max(a.bbox_px[0], b.bbox_px[0]))
                iy = max(0.0, min(a.bbox_px[3], b.bbox_px[3]) - max(a.bbox_px[1], b.bbox_px[1]))
                inter = ix * iy
                min_area = max(1e-6, min(
                    (a.bbox_px[2] - a.bbox_px[0]) * (a.bbox_px[3] - a.bbox_px[1]),
                    (b.bbox_px[2] - b.bbox_px[0]) * (b.bbox_px[3] - b.bbox_px[1]),
                ))
                if inter / min_area >= 0.25:
                    # Lower bottom-edge in image space is typically closer (occluder)
                    occluder, occluded = (a, b) if a.bbox_px[3] >= b.bbox_px[3] else (b, a)
                    occlusion_pairs.append((occluder.observation_id, occluded.observation_id))

    return Furniture2AnalysisResult(
        mobility_assignments=mobility,
        arrangements=tuple(arrangements),
        occlusion_pairs=tuple(occlusion_pairs),
        support_links=support_links,
        summary={
            "classified_objects": len(mobility),
            "arrangements": len(arrangements),
            "occlusion_pairs": len(occlusion_pairs),
            "support_links": len(support_links),
        },
    )


__all__ = [
    "Furniture2AnalysisResult",
    "FurnitureArrangementGroup",
    "MOBILITY_CLASSES",
    "analyze_furniture_v2",
    "classify_object_mobility",
]
