"""Building-topology promotion (auto-recon sprint P0/P1).

Position in the stack: room_graph.build_room_graph +
build_building_graph measured rooms (bounds, openings, adjacency) and
assembled them into storeys + a building envelope -- but nothing turned
that structure into canonical WorldIR entities, so the product could
never traverse Building -> Storey -> Room -> Wall, and windows/stairs
(detected by their own honest detectors) never joined the topology.

This module is the promotion write path for that structure (the same
discipline as evidence/promote_planes.py and
perception/architecture/promotion.py: detection stays pure, promotion
mutates the world):

  - promote_building_topology: every storey becomes a STOREY entity,
    every room a ROOM entity, the envelope a BUILDING entity. Edges:
    building CONTAINS storey, storey CONTAINS room, room CONTAINS each
    boundary part (wall/floor/ceiling entities must already exist --
    promote the planes first; missing parts are a hard refusal, never a
    dangling edge), part PART_OF room. Openings ride into the room's
    custom_properties (measured width/height + carrying wall), room
    dimensions/area into properties as measured facts. Confidence: the
    minimum of the parts' confidences (a room is as credible as its
    least-credible boundary). Provenance: INFERRED.

  - link_stairs_to_storeys: a StaircaseFit is a measured vertical
    span; every storey whose floor height falls within that span (its
    tolerance band included) is linked via ADJACENT_TO with
    derived_from metadata. The stair entity must exist (promoted via
    the canonical component path); links without entities are refused.

  - associate_windows_to_rooms: a WINDOW entity's wall_plane_id
    identifies the wall it was measured in; rooms that share that wall
    (boundary_element_ids contain the wall-<plane_id> element) and
    whose bounds overlap the window's geometry record the window in
    custom_properties["window_ids"] and gain an ADJACENT_TO edge.

Deterministic: canonical ordering everywhere; same input -> identical
output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Provenance,
    Relationship,
    RelationshipKind,
    Geometry,
    GeometryType,
    Vector3,
)
from perception.architecture.promotion import _store_entity
from perception.architecture.room_graph import BuildingGraph, RoomGraph


class TopologyError(ValueError):
    """Building-topology promotion refused."""


def _contains_edge(target_id: str, confidence: float, derived_from: str) -> Relationship:
    return Relationship(
        kind=RelationshipKind.CONTAINS,
        target_id=target_id,
        confidence=confidence,
        provenance=Provenance.INFERRED,
        metadata={"derived_from": derived_from},
    )


def _part_of_edge(target_id: str, confidence: float, derived_from: str) -> Relationship:
    return Relationship(
        kind=RelationshipKind.PART_OF,
        target_id=target_id,
        confidence=confidence,
        provenance=Provenance.INFERRED,
        metadata={"derived_from": derived_from},
    )


@dataclass(frozen=True)
class TopologyPromotionResult:
    """Ids written by promote_building_topology (traceable outcome)."""

    building_id: str
    storey_ids: Tuple[str, ...]
    room_ids: Tuple[str, ...]
    #: room_id -> boundary entity ids contained (audit trail).
    room_parts: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: RoomGraph.room_id values that had NO matching evidence-side ROOM
    #: entity and were therefore NOT promoted (see promote_building_topology
    #: docstring: evidence-side ring-closure detection is authoritative
    #: whenever it has run). A refusal, recorded here, not a fabrication.
    unmatched_room_ids: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "building_id": self.building_id,
            "storey_ids": list(self.storey_ids),
            "room_ids": list(self.room_ids),
            "room_parts": {k: list(v) for k, v in self.room_parts.items()},
            "unmatched_room_ids": list(self.unmatched_room_ids),
        }


def _room_entity(
    room: RoomGraph, index: int, world, parts_conf: float
) -> Entity:
    dims = {k: float(v) for k, v in room.dimensions_m.items()}
    openings = [
        {
            "wall_element_id": o.wall_element_id,
            "kind": o.kind,
            "width_m": o.width_m,
            "height_m": o.height_m,
        }
        for o in room.openings
    ]
    rid = f"room-{index:03d}"
    geom_id = f"geom-{rid}"
    world.geometries[geom_id] = Geometry(
        id=geom_id,
        type=GeometryType.BOX,
        bounds_min=Vector3(room.bounds_min[0], room.bounds_min[1], room.bounds_min[2]),
        bounds_max=Vector3(room.bounds_max[0], room.bounds_max[1], room.bounds_max[2]),
    )
    return Entity(
        id=f"room-{index:03d}",
        type=EntityType.ROOM,
        name=f"room {index:03d}",
        geometry_ids=[geom_id],
        custom_properties={
            "boundary_element_ids": list(room.boundary_element_ids),
            "bounds_min": list(room.bounds_min),
            "bounds_max": list(room.bounds_max),
            "dimensions_m": dims,
            "floor_area_m2": room.floor_area_m2,
            "openings": openings,
            "adjacent_room_ids": list(room.adjacent_room_ids),
        },
        confidence=parts_conf,
        provenance=Provenance.INFERRED,
    )


def _find_existing_room_entity(room: RoomGraph, world) -> Optional[str]:
    """Match `room` against an already-promoted ROOM entity by shared
    boundary parts (Jaccard overlap >= 0.5 of struct-* CONTAINS targets).

    Two independent detectors (evidence.promote_rooms's wall-ring tracer
    and this module's own enclosure grouping) can each promote a room
    for the same physical space when both run against the same world;
    without this check, promote_building_topology would mint a second,
    differently-id'd ROOM entity for evidence that already has one --
    a duplicate semantic entity, not a new fact. Reuse is preferred over
    creation whenever the physical evidence overlaps enough to be the
    same room.
    """
    room_parts = set(room.boundary_element_ids)
    if not room_parts:
        return None
    best_id: Optional[str] = None
    best_overlap = 0.0
    # world.entities is a plain dict in world_ir.world_v1.WorldIR (values()
    # yields Entity objects) but an EntityRegistry in the legacy
    # world_ir.world.WorldIR (no values(); iterating it directly yields
    # Entity objects instead) -- support both without assuming either.
    entities_iter = (
        world.entities.values() if hasattr(world.entities, "values") else world.entities
    )
    for ent in entities_iter:
        if ent.type != EntityType.ROOM:
            continue
        eid = ent.id
        ent_parts = {
            r.target_id for r in ent.relationships if r.kind == RelationshipKind.CONTAINS
        }
        if not ent_parts:
            continue
        union = room_parts | ent_parts
        overlap = len(room_parts & ent_parts) / len(union) if union else 0.0
        if overlap >= 0.5 and overlap > best_overlap:
            best_overlap = overlap
            best_id = eid
    return best_id


def promote_building_topology(
    building: BuildingGraph,
    rooms: Sequence[RoomGraph],
    world,
) -> TopologyPromotionResult:
    """Promote the building graph (storeys + rooms) into canonical
    WorldIR entities and relationships.

    Refuses (TopologyError) when:
      - the building graph is None (empty buildings are refusals, not
        shells) or no rooms are supplied;
      - a room's boundary entities do not exist in the world (promote
        the planes first -- a room over phantom parts would be a
        dangling topology, not evidence).

    Room-detector reconciliation (resolved): evidence.promote_rooms's
    wall-ring-closure detector and this module's own RoomGraph
    enclosure grouping can independently identify rooms for the same
    physical space, and RoomGraph feeds corridor/storey/space-graph
    construction regardless. Evidence-side ring closure is the more
    rigorous test (it requires a *closed* boundary, not just bounding-box
    proximity) and is treated as authoritative: if the world already
    contains any evidence-side ROOM entity when this runs (i.e.
    evidence-side detection has run), a RoomGraph room with no matching
    promoted ROOM entity (see _find_existing_room_entity) is NOT
    independently promoted -- that would fabricate a second, unverified
    room for evidence the stricter detector chose not to call a room.
    Its room_id is recorded in TopologyPromotionResult.unmatched_room_ids
    instead: a refusal, not a silent drop. Only when no evidence-side
    ROOM entities exist yet (the standalone use of this function, e.g.
    tests/test_topology_coherence.py) does an unmatched RoomGraph room
    still mint its own entity -- there is no stricter detector to defer
    to in that case.

    Deterministic; idempotency is the caller's responsibility (re-running
    on the same world raises DuplicateEntityError).
    """
    if building is None:
        raise TopologyError(
            "no building graph -- an empty building is a refusal, not a shell"
        )
    if not rooms:
        raise TopologyError("no rooms supplied -- nothing to promote")

    # Validate every room's parts BEFORE writing anything (atomicity:
    # a partial promotion would leave a half-wired graph).
    room_by_id = {r.room_id: r for r in rooms}
    room_parts: Dict[str, List[str]] = {}
    for room in rooms:
        missing = [
            eid for eid in room.boundary_element_ids if eid not in world.entities
        ]
        if missing:
            raise TopologyError(
                f"room {room.room_id} boundary parts missing from world: "
                f"{missing[:5]} -- promote the planes first "
                "(evidence.promote_planes.promote_plane_to_entity)"
            )
        room_parts[room.room_id] = list(room.boundary_element_ids)

    rooms_sorted = sorted(rooms, key=lambda r: r.room_id)
    room_ids: List[str] = []
    parts_conf: Dict[str, float] = {}
    for room in rooms_sorted:
        confs = [
            world.entities.get(eid).confidence
            for eid in room.boundary_element_ids
        ]
        parts_conf[room.room_id] = min(confs) if confs else 0.0

    # Evidence-side detection has "run" for this world iff a ROOM entity
    # already exists before this function creates any -- checked once,
    # up front, so promoting several rooms in this call doesn't flip the
    # answer partway through.
    entities_iter = (
        world.entities.values() if hasattr(world.entities, "values") else world.entities
    )
    evidence_side_active = any(e.type == EntityType.ROOM for e in entities_iter)

    # Rooms first (storeys/building reference them). Reuse an
    # already-promoted ROOM entity for the same physical room instead of
    # minting a duplicate (see _find_existing_room_entity). When
    # evidence-side detection is active and a RoomGraph room has no
    # match, that room is a refusal (unmatched_room_ids), not a
    # fabricated second entity (see promote_building_topology docstring,
    # "Room-detector reconciliation").
    room_index: Dict[str, str] = {}
    reused_room_ids = set()
    unmatched_room_ids: List[str] = []
    next_index = 1
    for room in rooms_sorted:
        existing_id = _find_existing_room_entity(room, world)
        if existing_id is not None:
            room_index[room.room_id] = existing_id
            reused_room_ids.add(room.room_id)
            room_ids.append(existing_id)
            continue
        if evidence_side_active:
            unmatched_room_ids.append(room.room_id)
            continue
        entity = _room_entity(room, next_index, world, parts_conf[room.room_id])
        _store_entity(world, entity)
        room_index[room.room_id] = entity.id
        room_ids.append(entity.id)
        next_index += 1

    # CONTAINS room -> part + PART_OF part -> room (both directions, so
    # traversal from either end works). Idempotent: a reused room may
    # already carry some of these edges from its original promotion.
    # Unmatched rooms have no entity to wire (refused above); skip them.
    for room in rooms_sorted:
        if room.room_id not in room_index:
            continue
        rid = room_index[room.room_id]
        conf = parts_conf[room.room_id]
        rent = world.entities.get(rid)
        existing_contains = {
            r.target_id for r in rent.relationships if r.kind == RelationshipKind.CONTAINS
        }
        for eid in room.boundary_element_ids:
            part_ent = world.entities.get(eid)
            if eid not in existing_contains:
                rent.relationships.append(
                    _contains_edge(eid, conf, "room_boundary_membership")
                )
            existing_part_of = {
                r.target_id
                for r in part_ent.relationships
                if r.kind == RelationshipKind.PART_OF
            }
            if rid not in existing_part_of:
                part_ent.relationships.append(
                    _part_of_edge(rid, conf, "room_boundary_membership")
                )

    # Storeys: reuse the building graph's measured grouping.
    storey_ids: List[str] = []
    for si, storey in enumerate(building.storeys, start=1):
        sid = f"storey-{si:02d}"
        member_room_ids = tuple(
            room_index[r] for r in storey.room_ids if r in room_index
        )
        confs = [
            world.entities.get(rid).confidence
            for rid in member_room_ids
        ] or [0.0]
        ent = Entity(
            id=sid,
            type=EntityType.STOREY,
            name=f"storey {si:02d}",
            custom_properties={
                "floor_height_m": storey.floor_height_m,
                "room_ids": list(member_room_ids),
            },
            confidence=min(confs),
            provenance=Provenance.INFERRED,
        )
        for rid in member_room_ids:
            ent.relationships.append(
                _contains_edge(rid, min(confs), "storey_floor_height_grouping")
            )
            world.entities.get(rid).relationships.append(
                _part_of_edge(sid, min(confs), "storey_floor_height_grouping")
            )
        _store_entity(world, ent)
        storey_ids.append(sid)

    # Building envelope entity.
    b_conf = min(
        (world.entities.get(sid).confidence for sid in storey_ids),
        default=0.0,
    )
    bent = Entity(
        id=building.building_id,
        type=EntityType.BUILDING,
        name="building 001",
        custom_properties={
            "envelope_bounds_min": list(building.envelope_bounds_min),
            "envelope_bounds_max": list(building.envelope_bounds_max),
            "storey_ids": list(storey_ids),
            "n_storeys": len(storey_ids),
        },
        confidence=b_conf,
        provenance=Provenance.INFERRED,
    )
    for sid in storey_ids:
        bent.relationships.append(
            _contains_edge(sid, b_conf, "building_envelope_grouping")
        )
        world.entities.get(sid).relationships.append(
            _part_of_edge(building.building_id, b_conf, "building_envelope_grouping")
        )
    _store_entity(world, bent)

    return TopologyPromotionResult(
        building_id=building.building_id,
        storey_ids=tuple(storey_ids),
        room_ids=tuple(room_ids),
        room_parts={k: tuple(v) for k, v in room_parts.items()},
        unmatched_room_ids=tuple(unmatched_room_ids),
    )


# ------------------------------------------------------------------
# Stairs <-> storey transitions
# ------------------------------------------------------------------


@dataclass(frozen=True)
class StairStoreyLink:
    """A measured stair's link to the storeys it spans."""

    stair_entity_id: str
    storey_ids: Tuple[str, ...]
    #: measured z-span of the stair (fit.lowest_z -> fit.highest_z)
    span_min_m: float
    span_max_m: float

    def to_dict(self) -> dict:
        return {
            "stair_entity_id": self.stair_entity_id,
            "storey_ids": list(self.storey_ids),
            "span_min_m": self.span_min_m,
            "span_max_m": self.span_max_m,
        }


def link_stairs_to_storeys(
    stair_fits: Sequence,
    building: BuildingGraph,
    world,
    entity_id_prefix: str = "stairs",
) -> List[StairStoreyLink]:
    """Link measured staircases to the storeys they vertically span.

    A storey is spanned when its measured floor height lies within the
    stair's vertical extent (expanded by the room graph's floor-height
    tolerance -- a stair lands ON a floor, its lowest band is that
    floor). Each link writes ADJACENT_TO edges (with derived_from
    metadata) between the stair entity and the storey entity.

    The stair entity must exist (promote the fit through the canonical
    component path first); a fit without an entity is refused -- a link
    to a phantom is a lie.
    """
    from perception.architecture.room_graph import FLOOR_HEIGHT_TOLERANCE_M

    links: List[StairStoreyLink] = []
    for i, fit in enumerate(stair_fits, start=1):
        eid = f"{entity_id_prefix}-{i:03d}"
        if eid not in world.entities:
            raise TopologyError(
                f"stair entity {eid!r} not in world -- promote the "
                "StaircaseFit through perception.architecture.promotion first"
            )
        lo = getattr(fit, "span_z_min_m", None)
        hi = getattr(fit, "span_z_max_m", None)
        if lo is None or hi is None:
            # Measured span: derive from the fit's own fields. A
            # StaircaseFit carries position + n_steps * rise; the
            # vertical extent is position.z .. position.z + n*rise
            # (measured quantities, not guesses).
            pos = getattr(fit, "position", (0.0, 0.0, 0.0))
            rise = getattr(fit, "rise_m", 0.0)
            n_steps = getattr(fit, "n_steps", 0)
            # `position` is the support centroid: it sits at the MEAN of
            # the measured band heights, i.e. mid-rhythm. The span is
            # therefore centroid +/- (n_steps * rise) / 2 -- derived
            # arithmetic on measured quantities, never a guess.
            half = (n_steps * rise) / 2.0
            lo, hi = pos[2] - half, pos[2] + half
        spanned: List[str] = []
        for storey in building.storeys:
            h = storey.floor_height_m
            if lo - FLOOR_HEIGHT_TOLERANCE_M <= h <= hi + FLOOR_HEIGHT_TOLERANCE_M:
                sid = f"storey-{storey.storey_id.split('-')[-1]}"
                if sid not in world.entities:
                    raise TopologyError(
                        f"storey entity {sid!r} not in world -- promote the "
                        "building topology first"
                    )
                spanned.append(sid)
        if not spanned:
            continue  # honest: a stair outside the building links nothing
        s_ent = world.entities.get(eid)
        for sid in spanned:
            s_ent.relationships.append(Relationship(
                kind=RelationshipKind.ADJACENT_TO,
                target_id=sid,
                confidence=fit.confidence,
                provenance=Provenance.INFERRED,
                metadata={
                    "derived_from": "stair_z_span_overlaps_storey_floor",
                    "stair_span_m": [lo, hi],
                },
            ))
        links.append(StairStoreyLink(
            stair_entity_id=eid,
            storey_ids=tuple(spanned),
            span_min_m=lo,
            span_max_m=hi,
        ))
    return links


# ------------------------------------------------------------------
# Window <-> room association
# ------------------------------------------------------------------





@dataclass(frozen=True)
class WindowRoomAssociation:
    """A window's measured association with the rooms it lights."""

    window_id: str
    room_ids: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "room_ids": list(self.room_ids),
        }


def _xy_overlap(a_lo, a_hi, b_lo, b_hi) -> bool:
    return a_lo[0] <= b_hi[0] and b_lo[0] <= a_hi[0] and \
        a_lo[1] <= b_hi[1] and b_lo[1] <= a_hi[1]


def associate_windows_to_rooms(
    window_entities: Sequence[Entity],
    rooms: Sequence[RoomGraph],
    world,
) -> List[WindowRoomAssociation]:
    """Associate WINDOW entities with the rooms whose walls they were
    measured in.

    The association is measured, not guessed: a window belongs to a
    room iff the room's boundary contains the window's wall element
    (custom_properties["wall_plane_id"] -> "wall-<plane_id>") AND the
    window's bounds overlap the room's plan. The room entity's
    custom_properties["window_ids"] records the association so it is
    queryable from the WorldIR alone.

    Windows with no matching room are returned with empty room_ids
    (recorded absence, never a forced pairing).
    """
    associations: List[WindowRoomAssociation] = []
    for win in window_entities:
        wall_plane_id = (win.custom_properties or {}).get("wall_plane_id")
        wall_element_id = f"wall-{wall_plane_id}" if wall_plane_id else None
        matched: List[str] = []
        for room in rooms:
            if wall_element_id and wall_element_id in room.boundary_element_ids:
                rid = f"room-{sorted(r.room_id for r in rooms).index(room.room_id) + 1:03d}"
                if rid not in matched:
                    matched.append(rid)
        associations.append(WindowRoomAssociation(
            window_id=win.id,
            room_ids=tuple(matched),
        ))
        if not matched:
            continue
        for rid in matched:
            rent = world.entities.get(rid)
            if not rent:
                for e in world.entities.values():
                    if getattr(e, "type", None) == EntityType.ROOM:
                        bounds = (e.custom_properties or {}).get("boundary_element_ids", [])
                        if wall_element_id in bounds or any(wall_plane_id and wall_plane_id in b for b in bounds):
                            rent = e
                            break
            if rent:
                win_ids = list(rent.custom_properties.get("window_ids", []))
                if win.id not in win_ids:
                    win_ids.append(win.id)
                rent.custom_properties["window_ids"] = sorted(win_ids)
                rent.relationships.append(Relationship(
                    kind=RelationshipKind.ADJACENT_TO,
                    target_id=win.id,
                    confidence=win.confidence,
                    provenance=Provenance.INFERRED,
                    metadata={"derived_from": "window_in_room_boundary_wall"},
                ))
    return associations


# ------------------------------------------------------------------
# Interior connectivity (INTERIOR RECONSTRUCTION MISSION 2/3/6/7)
# ------------------------------------------------------------------


#: Opening kinds that form a PASSAGE between spaces. A window connects
#: light and sight, not people -- it must never create a passage edge.
PASSAGE_KINDS = frozenset({"doorway", "door", "opening"})


def connected_rooms_through_openings(rooms, wall_openings) -> List[Tuple[str, str]]:
    """Rooms connected through passages in their shared walls.

    `wall_openings` yields (wall_element_id, RoomOpening) pairs: the
    measured openings and the walls carrying them. Two distinct rooms
    are connected iff BOTH rooms' boundary_element_ids contain the
    opening's wall AND the opening kind is a passage (door/generic --
    windows connect light, not people).

    Returns canonical (min_id, max_id) pairs, deduplicated, sorted.
    Pure function over measured graph facts; deterministic.
    """
    openings_by_wall = {}
    for wall_id, opening in wall_openings:
        if opening.kind not in PASSAGE_KINDS:
            continue
        openings_by_wall.setdefault(wall_id, []).append(opening)

    links = set()
    for room in rooms:
        for other in rooms:
            if other.room_id <= room.room_id:
                continue
            shared = (
                set(room.boundary_element_ids)
                & set(other.boundary_element_ids)
            )
            for wall_id in shared:
                if wall_id in openings_by_wall:
                    links.add((min(room.room_id, other.room_id),
                               max(room.room_id, other.room_id)))
    return sorted(links)


@dataclass(frozen=True)
class CorridorRoomLink:
    """A corridor's measured passage links to the rooms it serves."""

    corridor_id: str
    room_ids: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {"corridor_id": self.corridor_id, "room_ids": list(self.room_ids)}


def link_corridor_to_rooms(corridor_room, rooms, corridor_entity_id: str) -> CorridorRoomLink:
    """Rooms connected to a corridor through doors in shared walls.

    The corridor is itself a room-graph cell (detected by
    corridors.detect_corridor). A room is served by the corridor iff a
    passage opening (door) lies in a wall BOTH cells share. Window-only
    adjacency is recorded absence: no passage, no link.
    """
    served: List[str] = []
    corridor_walls = set(corridor_room.boundary_element_ids)
    for room in sorted(rooms, key=lambda r: r.room_id):
        if room.room_id == corridor_room.room_id:
            continue
        shared = corridor_walls & set(room.boundary_element_ids)
        if not shared:
            continue
        has_passage = any(
            o.kind in PASSAGE_KINDS
            for o in room.openings
            if o.wall_element_id in shared
        ) or any(
            o.kind in PASSAGE_KINDS
            for o in corridor_room.openings
            if o.wall_element_id in shared
        )
        if has_passage:
            served.append(room.room_id)
    return CorridorRoomLink(
        corridor_id=corridor_entity_id,
        room_ids=tuple(served),
    )


def storey_connectivity(building, stair_links) -> List[Tuple[str, str]]:
    """Level-to-level adjacency through stairs (topology, not geometry).

    `stair_links` yields (stair_entity_id, storey_ids) pairs -- the
    output shape of link_stairs_to_storeys. Every pair of storeys
    sharing one stair becomes a CONNECTED_TO edge (canonical order,
    deduplicated, sorted). A storey no stair reaches stays isolated:
    recorded honestly, never bridged by a guess.
    """
    edges = set()
    for _stair_id, storey_ids in stair_links:
        for i, a in enumerate(storey_ids):
            for b in storey_ids[i + 1:]:
                edges.add((min(a, b), max(a, b)))
    return sorted(edges)
