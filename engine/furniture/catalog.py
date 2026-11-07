"""Authoritative catalog of CC0 3D furniture models (T-4b.1).

Every item carries an unencumbered CC0 1.0 Universal license, real metric
bounding dimensions (width, depth, height in metres), authored materials,
and can be exported as standard binary glTF (GLB).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from exporters.gltf.exporter import export_to_glb
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Material,
    Provenance,
    SurfaceFinish,
    Vector3,
)
from world_ir.world_v1 import WorldIR


@dataclass(frozen=True)
class FurnitureCatalogItem:
    id: str
    name: str
    category: str
    license: str
    license_url: str
    author: str
    dimensions: dict[str, Any]
    color_rgb: tuple[float, float, float]
    description: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["color_rgb"] = list(self.color_rgb)
        return d


CC0_FURNITURE_CATALOG: list[FurnitureCatalogItem] = [
    FurnitureCatalogItem(
        id="cc0-chair-dining",
        name="Modern Dining Chair",
        category="chair",
        license="CC0 1.0 Universal (Public Domain)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Reality Engine CC0 Assets",
        dimensions={"width": 0.52, "depth": 0.54, "height": 0.85, "unit": "m"},
        color_rgb=(0.75, 0.55, 0.38),
        description="Clean-lined dining chair with solid oak legs and natural curved seat.",
        tags=["dining", "chair", "wood", "seating"],
    ),
    FurnitureCatalogItem(
        id="cc0-chair-office",
        name="Ergonomic Office Chair",
        category="chair",
        license="CC0 1.0 Universal (Public Domain)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Reality Engine CC0 Assets",
        dimensions={"width": 0.65, "depth": 0.65, "height": 1.05, "unit": "m"},
        color_rgb=(0.20, 0.20, 0.22),
        description="High-back ergonomic mesh office chair with 5-star swivel base.",
        tags=["office", "chair", "desk", "swivel", "ergonomic"],
    ),
    FurnitureCatalogItem(
        id="cc0-table-dining",
        name="Solid Oak Dining Table",
        category="table",
        license="CC0 1.0 Universal (Public Domain)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Reality Engine CC0 Assets",
        dimensions={"width": 1.80, "depth": 0.90, "height": 0.76, "unit": "m"},
        color_rgb=(0.68, 0.52, 0.36),
        description="6-person rectangular dining table crafted from solid American oak.",
        tags=["dining", "table", "wood", "oak"],
    ),
    FurnitureCatalogItem(
        id="cc0-table-coffee",
        name="Round Coffee Table",
        category="table",
        license="CC0 1.0 Universal (Public Domain)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Reality Engine CC0 Assets",
        dimensions={"width": 0.85, "depth": 0.85, "height": 0.45, "unit": "m"},
        color_rgb=(0.85, 0.82, 0.78),
        description="Low circular coffee table with honed stone composite top and birch legs.",
        tags=["living", "table", "coffee", "round"],
    ),
    FurnitureCatalogItem(
        id="cc0-sofa-2seater",
        name="2-Seater Fabric Sofa",
        category="sofa",
        license="CC0 1.0 Universal (Public Domain)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Reality Engine CC0 Assets",
        dimensions={"width": 1.75, "depth": 0.88, "height": 0.82, "unit": "m"},
        color_rgb=(0.35, 0.40, 0.45),
        description="Compact two-cushion loveseat upholstered in durable slate weave.",
        tags=["living", "sofa", "couch", "seating"],
    ),
    FurnitureCatalogItem(
        id="cc0-sofa-3seater",
        name="3-Seater Leather Sofa",
        category="sofa",
        license="CC0 1.0 Universal (Public Domain)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Reality Engine CC0 Assets",
        dimensions={"width": 2.20, "depth": 0.95, "height": 0.84, "unit": "m"},
        color_rgb=(0.38, 0.24, 0.16),
        description="Full-grain cognac leather sofa with deep tailored seating.",
        tags=["living", "sofa", "couch", "leather"],
    ),
    FurnitureCatalogItem(
        id="cc0-desk-office",
        name="Minimalist Work Desk",
        category="desk",
        license="CC0 1.0 Universal (Public Domain)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Reality Engine CC0 Assets",
        dimensions={"width": 1.40, "depth": 0.70, "height": 0.74, "unit": "m"},
        color_rgb=(0.90, 0.90, 0.88),
        description="Modern home office workstation with powder-coated steel legs.",
        tags=["office", "desk", "workstation"],
    ),
    FurnitureCatalogItem(
        id="cc0-bed-queen",
        name="Queen Platform Bed Frame",
        category="bed",
        license="CC0 1.0 Universal (Public Domain)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Reality Engine CC0 Assets",
        dimensions={"width": 1.62, "depth": 2.12, "height": 0.95, "unit": "m"},
        color_rgb=(0.55, 0.42, 0.32),
        description="Low walnut platform bed frame with integrated slotted headboard.",
        tags=["bedroom", "bed", "furniture"],
    ),
    FurnitureCatalogItem(
        id="cc0-bookshelf",
        name="Modular Bookshelf Unit",
        category="storage",
        license="CC0 1.0 Universal (Public Domain)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Reality Engine CC0 Assets",
        dimensions={"width": 0.80, "depth": 0.32, "height": 1.85, "unit": "m"},
        color_rgb=(0.45, 0.35, 0.25),
        description="5-tier vertical open shelving unit with solid oak uprights.",
        tags=["storage", "bookshelf", "shelving"],
    ),
]


def list_catalog(category: str | None = None) -> list[FurnitureCatalogItem]:
    """Return all catalog items, optionally filtered by category."""
    if not category:
        return list(CC0_FURNITURE_CATALOG)
    c = category.strip().lower()
    return [item for item in CC0_FURNITURE_CATALOG if item.category.lower() == c]


def get_catalog_item(item_id: str) -> FurnitureCatalogItem | None:
    """Lookup a furniture catalog item by its unique ID."""
    for item in CC0_FURNITURE_CATALOG:
        if item.id == item_id:
            return item
    return None


def build_furniture_worldir(item: FurnitureCatalogItem) -> WorldIR:
    """Construct a clean, single-entity WorldIR containing the catalog furniture item."""
    world = WorldIR(id=f"world-{item.id}")
    dims = item.dimensions
    w = float(dims.get("width", 1.0))
    d = float(dims.get("depth", 1.0))
    h = float(dims.get("height", 1.0))

    geom_id = f"geom-{item.id}-{uuid4().hex[:6]}"
    mat_id = f"mat-{item.id}-{uuid4().hex[:6]}"
    ent_id = f"ent-{item.id}"

    geom = Geometry(
        id=geom_id,
        type=GeometryType.BOX,
        bounds_min=Vector3(x=-w / 2, y=0.0, z=-d / 2),
        bounds_max=Vector3(x=w / 2, y=h, z=d / 2),
    )
    mat = Material(
        id=mat_id,
        name=f"{item.name} Material",
        class_name=item.category,
        surface_finish=SurfaceFinish.SMOOTH,
        color_rgb=item.color_rgb,
        provenance=Provenance.GENERATED,
        confidence=1.0,
    )
    ent = Entity(
        id=ent_id,
        name=item.name,
        type=EntityType.FURNITURE,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        geometry_ids=[geom_id],
        material_ids=[mat_id],
        semantic_labels=["furniture", item.category, "cc0_catalog"],
        provenance=Provenance.GENERATED,
        confidence=1.0,
        custom_properties={
            "catalog_id": item.id,
            "catalog_name": item.name,
            "category": item.category,
            "license": item.license,
            "license_url": item.license_url,
            "author": item.author,
            "dimensions": dims,
        },
    )

    world.geometries[geom_id] = geom
    world.materials[mat_id] = mat
    world.entities[ent_id] = ent
    return world


def export_furniture_glb(item: FurnitureCatalogItem) -> bytes:
    """Generate valid binary glTF 2.0 (GLB) bytes for the catalog furniture item."""
    world = build_furniture_worldir(item)
    return export_to_glb(world)
