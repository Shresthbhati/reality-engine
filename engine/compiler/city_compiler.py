"""City world compiler core (P14-01: "Terrain/roads/buildings/
vegetation/infrastructure/water ... outputs WorldIR").

Scope decision (honest, recorded): the full city compiler -- GIS/OSM/
satellite/LiDAR ingestion, terrain rasters, 3DCityDB/CityGML outputs --
is a program of work, not one module. This pass delivers the core seam
everything else plugs into:

    CityFeature (data: kind + footprint + height + evidence note)
        -> compile_city_world()
    WorldIR entities with real AABB geometry, deterministic ids/order,
    correct entity types, provenance recorded.

The seam is real, not a stub: features compiled here flow through the
existing exporters (CityJSON verified in tests) and the P13 spatial
tiles; OSM ingestion becomes "produce CityFeatures" later, without
touching this compiler.

Honesty rules:

  - Bounds derive from the feature's REAL footprint and height; a flat
    feature (no height) gets the 1 cm floored slab every other
    exporter uses for degenerate axes.
  - Unknown kinds are REFUSED (ValueError), never silently re-typed.
  - Evidence notes ride on the entity's semantic_labels -- traceable,
    not buried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from provenance import Provenance
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Vector3,
)
from world_ir.world_v1 import WorldIR

#: CityFeature kind -> WorldIR EntityType. Unknown kinds refuse.
_KIND_MAP = {
    "building": EntityType.BUILDING,
    "road": EntityType.ROAD,
    "water": EntityType.WATER,
    "vegetation": EntityType.VEGETATION,
    "infrastructure": EntityType.INFRASTRUCTURE,
    "terrain": EntityType.TERRAIN,
}

#: Flat features (roads, water) get this z extent instead of an
#: invisible zero-height AABB (same convention as the exporters).
_MIN_HEIGHT = 0.01


@dataclass
class CityFeature:
    """One city-scale feature as data.

    `footprint` is a convex-decomposable polygon in the XY plane (the
    compiler uses its axis-aligned bounding extent -- the honest
    simplification this pass makes; concave footprints are recorded as
    an open item, not approximated silently).
    `height` is the feature's measured/evidence height above ground;
    None for flat features (roads, water bodies).
    """

    kind: str
    name: str
    footprint: Sequence[Tuple[float, float]]
    height: Optional[float] = None
    evidence_note: str = ""

    def __post_init__(self):
        if self.kind not in _KIND_MAP:
            raise ValueError(
                f"unknown city feature kind {self.kind!r} -- known kinds: "
                f"{sorted(_KIND_MAP)}")
        if len(self.footprint) < 3:
            raise ValueError("footprint needs >= 3 vertices")


def compile_city_world(features: Sequence[CityFeature],
                       name: str = "city") -> WorldIR:
    """Compile city features into a deterministic WorldIR world.

    Entity ids are `<world name>-<kind>-<index>` in the features'
    given order (stable, traceable back to the input list).
    """
    world = WorldIR(id=f"world-city-{name}",
                    main_branch_id=f"branch-main-{name}")
    for index, feature in enumerate(features):
        etype = _KIND_MAP[feature.kind]
        xs = [float(p[0]) for p in feature.footprint]
        ys = [float(p[1]) for p in feature.footprint]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        z1 = float(feature.height) if feature.height is not None else _MIN_HEIGHT
        z0 = 0.0

        entity_id = f"{name}-{feature.kind}-{index}"
        geom_id = f"geom-{entity_id}"
        world.geometries[geom_id] = Geometry(
            id=geom_id,
            type=GeometryType.BOX,
            bounds_min=Vector3(x0, y0, z0),
            bounds_max=Vector3(x1, y1, z1),
            provenance=Provenance.GENERATED,
        )
        # Footprint center as the entity transform: the exporters
        # (CityJSON/glTF/Blender) place objects by transform position,
        # so city features carry theirs explicitly.
        entity = Entity(
            id=entity_id,
            name=feature.name,
            type=etype,
            transform={"position": {"x": (x0 + x1) / 2.0,
                                    "y": (y0 + y1) / 2.0,
                                    "z": (z0 + z1) / 2.0}},
            geometry_ids=[geom_id],
            statement_state=_feature_statement_state(),
            provenance=Provenance.GENERATED,
        )
        if feature.evidence_note:
            entity.semantic_labels.append(feature.evidence_note)
        world.entities[entity_id] = entity
    return world


def _feature_statement_state():
    from world_ir.statement_state import StatementState
    # Synthesized city mass from feature data is PROCEDURAL-adjacent;
    # the compiler's outputs are GENERATED data whose semantics are
    # "constructed from declared feature data", so PROCEDURAL is the
    # honest statement class.
    return StatementState.PROCEDURAL
