"""A small built-in furniture catalog (plan T-4b.1): nominal sizes for the classes the viewer already draws procedurally.

These are NOT measurements of anything in the room and not product data: they are typical overall sizes (metres) used as GENERATED
defaults when a designer swaps one piece for another. There are no glTF models yet (no licensed CC0 asset set has been
vetted); the viewer draws each class from boxes, sized from these dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CatalogItem:
    id: str
    name: str
    category: str       # the viewer's furniture class (room-shell-model.ts)
    length_m: float
    width_m: float
    height_m: float

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "category": self.category, "length_m": self.length_m,
                "width_m": self.width_m, "height_m": self.height_m, "basis": "nominal catalog size, not a measurement"}


_ITEMS = (
    CatalogItem("sofa-3-seat", "Sofa (3 seat)", "sofa", 2.1, 0.9, 0.85),
    CatalogItem("sofa-2-seat", "Sofa (2 seat)", "sofa", 1.6, 0.9, 0.85),
    CatalogItem("armchair", "Armchair", "armchair", 0.85, 0.85, 0.85),
    CatalogItem("chair", "Chair", "chair", 0.45, 0.45, 0.9),
    CatalogItem("stool", "Stool", "stool", 0.35, 0.35, 0.65),
    CatalogItem("dining-table-6", "Dining table (6 seat)", "dining_table", 1.8, 0.9, 0.75),
    CatalogItem("coffee-table", "Coffee table", "coffee_table", 1.0, 0.55, 0.42),
    CatalogItem("desk", "Desk", "desk", 1.4, 0.7, 0.75),
    CatalogItem("bed-double", "Bed (double)", "bed", 2.0, 1.6, 0.5),
    CatalogItem("bed-single", "Bed (single)", "bed", 2.0, 0.9, 0.5),
    CatalogItem("wardrobe", "Wardrobe", "wardrobe", 1.2, 0.6, 2.0),
    CatalogItem("bookshelf", "Bookshelf", "bookshelf", 0.8, 0.3, 1.8),
    CatalogItem("dresser", "Dresser", "dresser", 1.2, 0.5, 0.85),
    CatalogItem("nightstand", "Nightstand", "nightstand", 0.45, 0.4, 0.5),
    CatalogItem("tv-stand", "TV stand", "tv_stand", 1.5, 0.4, 0.5),
    CatalogItem("tv", "Television", "tv", 1.2, 0.08, 0.7),
    CatalogItem("floor-lamp", "Floor lamp", "lamp", 0.35, 0.35, 1.5),
    CatalogItem("cabinet", "Cabinet", "cabinet", 0.9, 0.45, 0.9),
)

CATALOG: Dict[str, CatalogItem] = {i.id: i for i in _ITEMS}


def catalog_items() -> List[dict]:
    return [i.to_dict() for i in _ITEMS]
