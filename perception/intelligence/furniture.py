"""Indoor Furniture & Fixture Intelligence Engine (A3.3).

Covers tables, chairs, sofas, beds, desks, cabinets, wardrobes, shelves,
storage units, kitchen furniture, bathroom fixtures, lighting fixtures,
and appliances.

Provides:
  - Hierarchical classification paths
  - Component / subpart ownership binding (legs, tops, seats, handles, drawers)
  - Occlusion-aware duplicate suppression
  - Separation of movable furniture from architectural structures
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import compute_box_iou
from perception.intelligence.data_model import (
    FurnitureObservation,
    NormalizedPerceptionObservation,
)

FURNITURE_HIERARCHY: dict[str, tuple[str, ...]] = {
    "chair": ("Furniture", "Seating", "Chair"),
    "dining_chair": ("Furniture", "Seating", "Chair", "DiningChair"),
    "armchair": ("Furniture", "Seating", "Chair", "Armchair"),
    "office_chair": ("Furniture", "Seating", "Chair", "OfficeChair"),
    "stool": ("Furniture", "Seating", "Stool"),
    "sofa": ("Furniture", "Seating", "Sofa"),
    "couch": ("Furniture", "Seating", "Sofa"),
    "bench": ("Furniture", "Seating", "Bench"),
    "bed": ("Furniture", "Sleeping", "Bed"),
    "table": ("Furniture", "Surface", "Table"),
    "dining_table": ("Furniture", "Surface", "Table", "DiningTable"),
    "coffee_table": ("Furniture", "Surface", "Table", "CoffeeTable"),
    "desk": ("Furniture", "Surface", "Desk"),
    "cabinet": ("Furniture", "Storage", "Cabinet"),
    "wardrobe": ("Furniture", "Storage", "Wardrobe"),
    "shelf": ("Furniture", "Storage", "Shelf"),
    "bookshelf": ("Furniture", "Storage", "Bookshelf"),
    "storage_unit": ("Furniture", "Storage", "StorageUnit"),
    "kitchen_furniture": ("Furniture", "Kitchen", "KitchenFurniture"),
    "countertop": ("Furniture", "Kitchen", "Countertop"),
    "bathroom_fixture": ("Fixtures", "Bathroom", "BathroomFixture"),
    "sink": ("Fixtures", "Plumbing", "Sink"),
    "toilet": ("Fixtures", "Plumbing", "Toilet"),
    "bathtub": ("Fixtures", "Plumbing", "Bathtub"),
    "lamp": ("Fixtures", "Lighting", "Lamp"),
    "ceiling_light": ("Fixtures", "Lighting", "CeilingLight"),
    "refrigerator": ("Appliances", "Kitchen", "Refrigerator"),
    "oven": ("Appliances", "Kitchen", "Oven"),
    "microwave": ("Appliances", "Kitchen", "Microwave"),
}

FURNITURE_COMPONENTS = frozenset({
    "table_top", "table_leg", "furniture_leg", "chair_seat", "chair_backrest",
    "chair_armrest", "drawer", "drawer_handle", "furniture_handle",
})

COMPONENT_HOST_RULES: dict[str, frozenset[str]] = {
    "table_top": frozenset({"table", "desk", "dining_table", "coffee_table"}),
    "table_leg": frozenset({"table", "desk", "dining_table", "coffee_table"}),
    "furniture_leg": frozenset({"table", "chair", "sofa", "desk", "bed", "cabinet"}),
    "chair_seat": frozenset({"chair", "dining_chair", "armchair", "office_chair", "sofa"}),
    "chair_backrest": frozenset({"chair", "dining_chair", "armchair", "office_chair", "sofa"}),
    "chair_armrest": frozenset({"chair", "armchair", "office_chair", "sofa"}),
    "drawer": frozenset({"desk", "cabinet", "storage_unit", "wardrobe", "kitchen_furniture"}),
    "drawer_handle": frozenset({"drawer", "cabinet", "wardrobe", "desk", "storage_unit"}),
    "furniture_handle": frozenset({"cabinet", "wardrobe", "drawer", "storage_unit", "kitchen_furniture"}),
}


@dataclass(frozen=True)
class FurnitureAnalysisResult:
    items: tuple[FurnitureObservation, ...]
    component_links: dict[str, str]
    suppressed_duplicates: tuple[str, ...]
    summary: dict[str, Any]


def classify_furniture_hierarchy(canonical_label: str) -> tuple[str, ...]:
    key = canonical_label.strip().lower().replace(" ", "_")
    return FURNITURE_HIERARCHY.get(key, ("Furniture", "General", key.title()))


def _ioa(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (ix * iy) / area


def analyze_furniture(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    duplicate_iou: float = 0.65,
) -> FurnitureAnalysisResult:
    """Analyze furniture & fixture observations, deduplicate per-image, and bind subparts."""
    furn_obs = [
        o for o in observations
        if o.canonical_label.lower().replace(" ", "_") in FURNITURE_HIERARCHY
        or o.canonical_label.lower().replace(" ", "_") in FURNITURE_COMPONENTS
        or o.category in ("furniture", "fixture", "appliance")
    ]
    by_img: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for o in furn_obs:
        by_img[o.provenance.source_image_id].append(o)

    kept_all: list[NormalizedPerceptionObservation] = []
    suppressed: list[str] = []
    component_links: dict[str, str] = {}
    host_children: dict[str, list[str]] = defaultdict(list)

    for _img_id, obs_list in by_img.items():
        hosts = [o for o in obs_list if o.canonical_label not in FURNITURE_COMPONENTS]
        parts = [o for o in obs_list if o.canonical_label in FURNITURE_COMPONENTS]

        # Same-class duplicate suppression (never suppress across distinct classes)
        hosts_sorted = sorted(
            hosts,
            key=lambda o: float(o.scores.raw_score if o.scores.raw_score is not None else 0.5),
            reverse=True,
        )
        kept_hosts: list[NormalizedPerceptionObservation] = []
        for h in hosts_sorted:
            is_dup = False
            for existing in kept_hosts:
                if existing.canonical_label == h.canonical_label and compute_box_iou(h.bbox_px, existing.bbox_px) >= duplicate_iou:
                    is_dup = True
                    break
            if is_dup:
                suppressed.append(h.observation_id)
            else:
                kept_hosts.append(h)

        # Bind furniture subparts to their best compatible host
        for p in parts:
            allowed_hosts = COMPONENT_HOST_RULES.get(p.canonical_label, frozenset())
            best_host: str | None = None
            best_ioa = 0.0
            for h in kept_hosts:
                if allowed_hosts and h.canonical_label not in allowed_hosts:
                    continue
                val = _ioa(p.bbox_px, h.bbox_px)
                if val >= 0.35 and val > best_ioa:
                    best_ioa = val
                    best_host = h.observation_id
            if best_host is not None:
                component_links[p.observation_id] = best_host
                host_children[best_host].append(p.observation_id)

        kept_all.extend(kept_hosts)

    items = tuple(
        FurnitureObservation(
            observation=o,
            hierarchy_path=classify_furniture_hierarchy(o.canonical_label),
            component_ids=tuple(host_children.get(o.observation_id, ())),
            mobility="fixed_fixture" if o.canonical_label in ("sink", "toilet", "bathtub", "countertop", "ceiling_light") else "movable",
        )
        for o in kept_all
    )
    summary = {
        "total_furniture_items": len(items),
        "components_linked": len(component_links),
        "suppressed_duplicates": len(suppressed),
        "partial_or_occluded": sum(1 for i in items if i.observation.is_partial or i.observation.occlusion_level != "none"),
    }
    return FurnitureAnalysisResult(
        items=items,
        component_links=component_links,
        suppressed_duplicates=tuple(suppressed),
        summary=summary,
    )


__all__ = [
    "COMPONENT_HOST_RULES",
    "FURNITURE_COMPONENTS",
    "FURNITURE_HIERARCHY",
    "FurnitureAnalysisResult",
    "analyze_furniture",
    "classify_furniture_hierarchy",
]
