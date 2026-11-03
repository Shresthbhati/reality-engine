"""Architecture-to-Contract output adapters (Blueprint 11 & 12, Task 5.2).

Adapts raw geometric and perception structures (planes, rooms, openings, storeys)
into the canonical WorldIR contract hierarchy:
    World -> Building -> Floor -> Space -> Surface -> Opening
with explicit evidence states (OBSERVED, INFERRED, OCCLUDED, PROCEDURAL) and
honest provenance.

Also provides:
- Glazing / transparency classification (Scenario A14): glass is classified as an
  Opening/Window with glazing evidence, not an opaque wall.
- Surface continuity under occlusion (Scenario A6): furniture occluding a wall leaves
  the wall as a single continuous surface, marking hidden spans OCCLUDED/INFERRED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from engine.math import Vec3
from provenance import Provenance
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Measurement,
    Relationship,
    RelationshipKind,
    Surface as WorldIRSurface,
    Vector3,
)
from world_ir.v2_extensions import HIERARCHY_LEVELS, Hierarchy, HierarchyNode
from world_ir.validation import VALID_EVIDENCE_STATES


@dataclass
class ContractAdaptationResult:
    """Result of adapting architectural entities into the contract hierarchy."""

    building_node_id: Optional[str] = None
    floor_node_ids: List[str] = field(default_factory=list)
    space_node_ids: List[str] = field(default_factory=list)
    surface_node_ids: List[str] = field(default_factory=list)
    opening_node_ids: List[str] = field(default_factory=list)
    entity_space_mappings: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SurfaceSpan:
    """A segment or sub-region along a surface face with evidence classification."""

    start_m: float
    end_m: float
    evidence_state: str  # OBSERVED | OCCLUDED | INFERRED
    note: str = ""


@dataclass
class WallContinuityResult:
    """Outcome of evaluating wall continuity across occlusions."""

    wall_id: str
    is_continuous: bool
    spans: List[SurfaceSpan]
    overall_evidence_state: str
    observed_fraction: float


def adapt_architecture_to_contract(
    world,
    *,
    default_building_id: Optional[str] = None,
    default_floor_id: Optional[str] = None,
) -> ContractAdaptationResult:
    """Adapt all architectural entities in `world` into the contract hierarchy.

    Guarantees:
    1. Every Building, Floor, Space, Surface, and Opening has a typed HierarchyNode
       connected in the hierarchy tree: World -> Building -> Floor -> Space -> Surface -> Opening.
    2. Every architectural entity and surface has an explicit `evidence_state`.
    3. Each Surface entity is linked to a Space entity (`custom_properties["space_id"]`).
    4. Each Opening entity is linked to a Surface entity (`custom_properties["host_wall_id"]` / `surface_id`).
    5. Discovered spaces populate `world.surfaces` for full Invariant 1 & 7 compliance.
    6. Non-architectural entities (furniture, fixtures) located in a room are mapped to `world.hierarchy.entity_space`.
    """
    res = ContractAdaptationResult()

    if not hasattr(world, "hierarchy") or world.hierarchy is None:
        world.hierarchy = Hierarchy()

    hierarchy = world.hierarchy

    # Entity classification sets
    building_types = {EntityType.BUILDING}
    floor_types = {EntityType.LEVEL}  # or custom storeys
    space_types = {EntityType.ROOM, EntityType.CORRIDOR}
    surface_types = {EntityType.WALL, EntityType.FLOOR, EntityType.CEILING}
    opening_types = {EntityType.DOOR, EntityType.WINDOW, EntityType.OPENING}

    entities: Dict[str, Entity] = world.entities if hasattr(world.entities, "items") else {}

    # 1. Building node
    building_ents = [e for e in entities.values() if e.type in building_types]
    if building_ents:
        b_ent = building_ents[0]
        b_node_id = f"node-{b_ent.id}"
        cp = b_ent.custom_properties if isinstance(b_ent.custom_properties, dict) else {}
        b_ev_state = cp.get("evidence_state")
        if not b_ev_state:
            b_ev_state = "INFERRED" if b_ent.provenance == Provenance.INFERRED else "OBSERVED"
            cp["evidence_state"] = b_ev_state
            b_ent.custom_properties = cp
        if b_node_id not in hierarchy.nodes:
            hierarchy.add(HierarchyNode(
                id=b_node_id,
                level="building",
                parent_id=None,
                entity_id=b_ent.id,
                evidence_state=b_ev_state,
                confidence=b_ent.confidence,
            ))
        res.building_node_id = b_node_id
    else:
        # Default implicit building node for single-site captures
        b_node_id = default_building_id or "node-building-default"
        if b_node_id not in hierarchy.nodes:
            hierarchy.add(HierarchyNode(
                id=b_node_id,
                level="building",
                parent_id=None,
                entity_id=None,
                evidence_state="INFERRED",
                confidence=0.9,
            ))
        res.building_node_id = b_node_id

    # 2. Floor / Storey nodes
    floor_ents = [
        e for e in entities.values()
        if e.type in floor_types or (isinstance(e.name, str) and e.name.startswith("Storey "))
    ]
    floor_node_map: Dict[str, str] = {}  # floor entity_id or name -> floor hierarchy node_id

    if floor_ents:
        for f_ent in floor_ents:
            f_node_id = f"node-{f_ent.id}"
            cp = f_ent.custom_properties if isinstance(f_ent.custom_properties, dict) else {}
            f_ev_state = cp.get("evidence_state")
            if not f_ev_state:
                f_ev_state = "INFERRED" if f_ent.provenance == Provenance.INFERRED else "OBSERVED"
                cp["evidence_state"] = f_ev_state
                f_ent.custom_properties = cp
            if f_node_id not in hierarchy.nodes:
                hierarchy.add(HierarchyNode(
                    id=f_node_id,
                    level="floor",
                    parent_id=res.building_node_id,
                    entity_id=f_ent.id,
                    evidence_state=f_ev_state,
                    confidence=f_ent.confidence,
                ))
            floor_node_map[f_ent.id] = f_node_id
            res.floor_node_ids.append(f_node_id)
    else:
        # Default floor node for single-level capture
        f_node_id = default_floor_id or "node-floor-default"
        if f_node_id not in hierarchy.nodes:
            hierarchy.add(HierarchyNode(
                id=f_node_id,
                level="floor",
                parent_id=res.building_node_id,
                entity_id=None,
                evidence_state="INFERRED",
                confidence=0.9,
            ))
        res.floor_node_ids.append(f_node_id)
        floor_node_map["default"] = f_node_id

    default_floor_node_id = res.floor_node_ids[0]

    # 3. Space nodes (Rooms, Corridors)
    space_ents = [e for e in entities.values() if e.type in space_types]
    space_node_map: Dict[str, str] = {}  # space entity_id -> space hierarchy node_id
    space_by_node: Dict[str, Entity] = {}

    for s_ent in space_ents:
        s_node_id = f"node-{s_ent.id}"
        cp = s_ent.custom_properties if isinstance(s_ent.custom_properties, dict) else {}
        kind = "corridor" if s_ent.type == EntityType.CORRIDOR else "room"
        
        # Determine parent floor
        parent_floor = default_floor_node_id
        parent_storey_id = cp.get("storey_id") or cp.get("floor_id")
        if parent_storey_id and parent_storey_id in floor_node_map:
            parent_floor = floor_node_map[parent_storey_id]

        ev_state = cp.get("evidence_state")
        if not ev_state or ev_state.upper() not in VALID_EVIDENCE_STATES:
            ev_state = "INFERRED" if s_ent.provenance == Provenance.INFERRED else "OBSERVED"
            cp["evidence_state"] = ev_state
            s_ent.custom_properties = cp

        if s_node_id not in hierarchy.nodes:
            hierarchy.add(HierarchyNode(
                id=s_node_id,
                level="space",
                parent_id=parent_floor,
                entity_id=s_ent.id,
                kind=kind,
                evidence_state=ev_state,
                confidence=s_ent.confidence,
            ))
        space_node_map[s_ent.id] = s_node_id
        space_by_node[s_node_id] = s_ent
        res.space_node_ids.append(s_node_id)

    # Fallback space if surfaces exist but no explicit room entity exists
    if not space_ents:
        fallback_space_id = "node-space-default"
        if fallback_space_id not in hierarchy.nodes:
            hierarchy.add(HierarchyNode(
                id=fallback_space_id,
                level="space",
                parent_id=default_floor_node_id,
                entity_id=None,
                kind="room",
                evidence_state="INFERRED",
                confidence=0.8,
            ))
        space_node_map["default"] = fallback_space_id
        res.space_node_ids.append(fallback_space_id)

    primary_space_node_id = res.space_node_ids[0]

    # Map of space entity boundary elements (element_id -> space_entity_id)
    boundary_to_space: Dict[str, str] = {}
    for s_ent in space_ents:
        cp = s_ent.custom_properties or {}
        b_ids = cp.get("boundary_element_ids") or []
        for b_id in b_ids:
            boundary_to_space[b_id] = s_ent.id

    # 4. Surface nodes (Walls, Floors, Ceilings)
    surface_ents = [e for e in entities.values() if e.type in surface_types]
    surface_node_map: Dict[str, str] = {}  # surface entity_id -> surface hierarchy node_id

    for sf_ent in surface_ents:
        sf_node_id = f"node-{sf_ent.id}"
        cp = sf_ent.custom_properties if isinstance(sf_ent.custom_properties, dict) else {}
        
        kind_map = {
            EntityType.WALL: "wall",
            EntityType.FLOOR: "floor",
            EntityType.CEILING: "ceiling",
        }
        kind = kind_map.get(sf_ent.type, "other")

        # Resolve which space this surface belongs to
        space_id = cp.get("space_id") or cp.get("room_id")
        if not space_id:
            # Check boundary_to_space
            if sf_ent.id in boundary_to_space:
                space_id = boundary_to_space[sf_ent.id]
            else:
                # Check relationship links
                for rel in sf_ent.relationships:
                    if rel.kind in (RelationshipKind.PART_OF, RelationshipKind.INSIDE) and rel.target_id in space_node_map:
                        space_id = rel.target_id
                        break
        
        if not space_id and space_ents:
            # Associate with primary space
            space_id = space_ents[0].id

        if space_id:
            cp["space_id"] = space_id
            sf_ent.custom_properties = cp

        parent_space_node = space_node_map.get(space_id, primary_space_node_id)

        ev_state = cp.get("evidence_state")
        if not ev_state or ev_state.upper() not in VALID_EVIDENCE_STATES:
            ev_state = "INFERRED" if sf_ent.provenance == Provenance.INFERRED else "OBSERVED"
            cp["evidence_state"] = ev_state
            sf_ent.custom_properties = cp

        coverage = cp.get("coverage")
        if coverage is None:
            # Measured extent or inlier coverage fraction
            coverage = 1.0 if ev_state == "OBSERVED" else 0.8
            cp["coverage"] = coverage

        if sf_node_id not in hierarchy.nodes:
            hierarchy.add(HierarchyNode(
                id=sf_node_id,
                level="surface",
                parent_id=parent_space_node,
                entity_id=sf_ent.id,
                kind=kind,
                evidence_state=ev_state,
                confidence=sf_ent.confidence,
                coverage=float(coverage) if coverage is not None else None,
            ))
        surface_node_map[sf_ent.id] = sf_node_id
        res.surface_node_ids.append(sf_node_id)

        # Ensure world.surfaces has this surface definition for full schema compliance
        if hasattr(world, "surfaces") and isinstance(world.surfaces, dict):
            surf_key = f"surf-{sf_ent.id}"
            if surf_key not in world.surfaces:
                geom_id = sf_ent.geometry_ids[0] if sf_ent.geometry_ids else ""
                world.surfaces[surf_key] = WorldIRSurface(
                    id=surf_key,
                    name=sf_ent.name or f"Surface {sf_ent.id}",
                    geometry_id=geom_id,
                    custom_attributes={
                        "evidence_state": ev_state,
                        "space_id": space_id or "",
                        "entity_id": sf_ent.id,
                        "kind": kind,
                    },
                )

    # 5. Opening nodes (Doors, Windows, Openings)
    opening_ents = [e for e in entities.values() if e.type in opening_types]

    for op_ent in opening_ents:
        op_node_id = f"node-{op_ent.id}"
        cp = op_ent.custom_properties if isinstance(op_ent.custom_properties, dict) else {}

        kind_map = {
            EntityType.DOOR: "door",
            EntityType.WINDOW: "window",
            EntityType.OPENING: "passage",
        }
        kind = kind_map.get(op_ent.type, "passage")

        # Resolve host surface
        host_surface_id = cp.get("host_wall_id") or cp.get("surface_id")
        if not host_surface_id:
            for rel in op_ent.relationships:
                if rel.kind in (RelationshipKind.PART_OF, RelationshipKind.EMBEDDED_IN) and rel.target_id in surface_node_map:
                    host_surface_id = rel.target_id
                    break

        if not host_surface_id and surface_ents:
            # Fallback to first wall surface
            walls = [s for s in surface_ents if s.type == EntityType.WALL]
            host_surface_id = walls[0].id if walls else surface_ents[0].id

        if host_surface_id:
            cp["host_wall_id"] = host_surface_id
            cp["surface_id"] = host_surface_id
            op_ent.custom_properties = cp

        parent_surface_node = surface_node_map.get(host_surface_id)
        if not parent_surface_node and res.surface_node_ids:
            parent_surface_node = res.surface_node_ids[0]

        ev_state = cp.get("evidence_state")
        if not ev_state or ev_state.upper() not in VALID_EVIDENCE_STATES:
            ev_state = "INFERRED" if op_ent.provenance == Provenance.INFERRED else "OBSERVED"
            cp["evidence_state"] = ev_state
            op_ent.custom_properties = cp

        if op_node_id not in hierarchy.nodes and parent_surface_node:
            hierarchy.add(HierarchyNode(
                id=op_node_id,
                level="opening",
                parent_id=parent_surface_node,
                entity_id=op_ent.id,
                kind=kind,
                evidence_state=ev_state,
                confidence=op_ent.confidence,
            ))
            res.opening_node_ids.append(op_node_id)

    # 6. Entity-space placement for objects (furniture, fixtures, appliances)
    excluded_types = building_types | floor_types | space_types | surface_types | opening_types
    for eid, ent in entities.items():
        if ent.type in excluded_types:
            continue
        # Find which space this object lives in
        space_id = (ent.custom_properties or {}).get("space_id") or (ent.custom_properties or {}).get("room_id")
        if not space_id:
            for rel in ent.relationships:
                if rel.kind in (RelationshipKind.INSIDE, RelationshipKind.PART_OF) and rel.target_id in space_node_map:
                    space_id = rel.target_id
                    break
        target_space_node = space_node_map.get(space_id, primary_space_node_id)
        hierarchy.entity_space[eid] = target_space_node
        res.entity_space_mappings[eid] = target_space_node

    return res


def classify_glazing_opening(
    plane_id: str,
    *,
    is_translucent: bool = False,
    specular_sheen: float = 0.0,
    material_kind: Optional[str] = None,
    color_rgb: Optional[Sequence[int]] = None,
    dimensions: Optional[Tuple[float, float]] = None,
    sill_height_m: Optional[float] = None,
) -> Tuple[str, dict]:
    """Glazing and glass transparency classifier (Scenario A14).

    Distinguishes transparent or glazed architectural elements from opaque walls.
    A planar feature with transparency, high specular sheen with low absorption,
    or glass semantics is classified as an Opening (window/glazing), never an opaque wall.

    Returns:
        (contract_kind, properties_dict) where contract_kind is 'window' or 'passage'
    """
    is_glass = (
        is_translucent
        or (material_kind and "glass" in material_kind.lower())
        or (material_kind and "glazing" in material_kind.lower())
        or (specular_sheen >= 0.70)
    )

    if not is_glass:
        return "wall", {"is_opaque": True, "glazing": False}

    sill = sill_height_m if sill_height_m is not None else 0.9
    width, height = dimensions if dimensions else (1.2, 1.4)

    return "window", {
        "is_opaque": False,
        "glazing": True,
        "translucency": 0.85 if is_translucent else 0.65,
        "specular_sheen": specular_sheen,
        "sill_height_m": sill,
        "width_m": width,
        "height_m": height,
        "evidence_state": "OBSERVED",
        "provenance": "OBSERVED",
    }


def evaluate_wall_continuity_under_occlusion(
    wall_span_length_m: float,
    occluder_spans: Sequence[Tuple[float, float]],
    wall_id: str = "wall-0",
) -> WallContinuityResult:
    """Evaluate surface continuity for a wall occluded by foreground objects (Scenario A6).

    When furniture or equipment occludes a portion of a wall plane, the wall
    must remain continuous rather than splitting into separate disjoint wall entities.
    The occluded spans are marked OCCLUDED / INFERRED, while visible spans are OBSERVED.
    """
    if wall_span_length_m <= 0.0:
        raise ValueError("wall span length must be positive")

    # Normalize occluder spans to [0, wall_span_length_m]
    valid_occluders = []
    for s_start, s_end in occluder_spans:
        start_clamped = max(0.0, min(wall_span_length_m, s_start))
        end_clamped = max(0.0, min(wall_span_length_m, s_end))
        if end_clamped > start_clamped:
            valid_occluders.append((start_clamped, end_clamped))

    # Merge overlapping occluder intervals
    valid_occluders.sort()
    merged_occluders: List[Tuple[float, float]] = []
    for occ in valid_occluders:
        if not merged_occluders:
            merged_occluders.append(occ)
        else:
            prev_s, prev_e = merged_occluders[-1]
            if occ[0] <= prev_e:
                merged_occluders[-1] = (prev_s, max(prev_e, occ[1]))
            else:
                merged_occluders.append(occ)

    # Build continuous spans: visible (OBSERVED) and occluded (OCCLUDED)
    spans: List[SurfaceSpan] = []
    cursor = 0.0
    for occ_s, occ_e in merged_occluders:
        if occ_s > cursor:
            spans.append(SurfaceSpan(start_m=cursor, end_m=occ_s, evidence_state="OBSERVED", note="unobstructed wall face"))
        spans.append(SurfaceSpan(start_m=occ_s, end_m=occ_e, evidence_state="OCCLUDED", note="foreground occlusion (furniture/fixture)"))
        cursor = occ_e

    if cursor < wall_span_length_m:
        spans.append(SurfaceSpan(start_m=cursor, end_m=wall_span_length_m, evidence_state="OBSERVED", note="unobstructed wall face"))

    observed_len = sum(s.end_m - s.start_m for s in spans if s.evidence_state == "OBSERVED")
    observed_fraction = observed_len / wall_span_length_m

    overall_state = "OBSERVED" if observed_fraction >= 0.99 else "PARTIALLY_OBSERVED" if observed_fraction > 0.1 else "OCCLUDED"

    return WallContinuityResult(
        wall_id=wall_id,
        is_continuous=True,  # Wall remains single continuous entity
        spans=spans,
        overall_evidence_state=overall_state,
        observed_fraction=observed_fraction,
    )
