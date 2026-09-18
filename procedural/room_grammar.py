"""Procedural room grammar (P15-01: "WorldIR -> urban/room grammar ->
constraints -> generated worlds").

The grammar turns a RoomGrammarSpec (DATA: dimensions, openings, seed)
into a real WorldIR world -- deterministic, derivable, and honestly
labeled: every generated entity carries StatementState.PROCEDURAL
(Provenance.GENERATED). Generated worlds are for simulation scenarios,
synthetic benchmark scenes, and testing downstream compilers; they are
never a substitute for capture.

Constraints hold BY CONSTRUCTION (not validated after the fact):

  - Walls bound the floor extent: wall AABBs span the room's width or
    depth at its edges and meet at corners.
  - The ceiling spans the floor at the room height.
  - Openings (doors/windows) are entities with MEASURED bounds carved
    from the wall they belong to -- real geometry in the wall plane,
    not decorative labels.
  - A floor layout places rooms on a deterministic grid and SHARES the
    wall between adjacent rooms (one wall entity per boundary line
    segment), which is what keeps the room graph's adjacency honest.

Determinism: no randomness anywhere. The `seed` field exists so specs
stay forward-compatible with stochastic grammars; the current grammar
is fully derived from the spec's geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from provenance import Provenance
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Vector3,
)
from world_ir.statement_state import StatementState
from world_ir.world_v1 import WorldIR

#: Provenance/state applied to every generated statement.
_PROCEDURAL_STATE = StatementState.PROCEDURAL
_PROCEDURAL_PROVENANCE = Provenance.GENERATED

_WALL_THICKNESS = 0.2


@dataclass
class WallOpening:
    """An opening in a named wall, positioned along it.

    `wall` is one of "north"/"south"/"east"/"west" (grammar-local
    axes: north = -y, south = +y, west = -x, east = +x in the room's
    own frame). lateral_offset is the distance along the wall from its
    minimum axis corner to the opening's minimum corner.
    """

    wall: str
    kind: str            # "door" | "window"
    lateral_offset: float
    width: float
    bottom: float        # opening's bottom above the floor
    top: float           # opening's top above the floor

    def __post_init__(self):
        if self.wall not in {"north", "south", "east", "west"}:
            raise ValueError(f"unknown wall {self.wall!r}")
        if self.kind not in {"door", "window"}:
            raise ValueError(f"unknown opening kind {self.kind!r}")
        if self.width <= 0 or self.top <= self.bottom or self.bottom < 0:
            raise ValueError("opening needs width > 0 and top > bottom >= 0")


@dataclass
class RoomGrammarSpec:
    """One room as data: extents, openings, identity."""

    name: str
    width: float         # along x
    depth: float         # along y
    height: float        # along z
    openings: List[WallOpening] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        if self.width <= 0 or self.depth <= 0 or self.height <= 0:
            raise ValueError("room extents must be positive")


def _entity(entity_id: str, etype: EntityType, name: str,
            bmin: Tuple[float, float, float],
            bmax: Tuple[float, float, float],
            geometry_owner: Dict[str, Geometry]) -> Entity:
    geom_id = f"geom-{entity_id}"
    geometry_owner[geom_id] = Geometry(
        id=geom_id,
        type=GeometryType.BOX,
        bounds_min=Vector3(*bmin),
        bounds_max=Vector3(*bmax),
        provenance=_PROCEDURAL_PROVENANCE,
    )
    return Entity(
        id=entity_id,
        name=name,
        type=etype,
        geometry_ids=[geom_id],
        statement_state=_PROCEDURAL_STATE,
        provenance=_PROCEDURAL_PROVENANCE,
    )


def generate_room(spec: RoomGrammarSpec, world: Optional[WorldIR] = None,
                  origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                  exterior_walls: Optional[set] = None) -> WorldIR:
    """Generate one room's floor/walls/ceiling (+ openings) into
    `world` (a fresh WorldIR when omitted), with the room's minimum
    corner at `origin`.

    `exterior_walls` names which walls to build (default: all four).
    generate_floor_layout uses it to skip shared-wall duplicates.
    Entity ids are namespaced by the room name so layouts never
    collide.
    """
    if world is None:
        # Deterministic world identity: two runs of the same spec
        # produce byte-identical worlds (the WorldIR container's
        # uuid4 id/branch defaults would otherwise break
        # reproducibility).
        world = WorldIR(
            id=f"world-{spec.name}",
            main_branch_id=f"branch-main-{spec.name}",
        )
    ox, oy, oz = origin
    w, d, h = spec.width, spec.depth, spec.height
    t = _WALL_THICKNESS

    if exterior_walls is None:
        exterior_walls = {"north", "south", "east", "west"}

    def add(entity: Entity) -> None:
        world.entities[entity.id] = entity
        world.geometries.update({
            gid: _entity_geometries[entity.id][gid]
            for gid in entity.geometry_ids
        })

    _entity_geometries: Dict[str, Dict[str, Geometry]] = {}

    def build(entity_id: str, etype: EntityType, name: str,
              bmin: Tuple[float, float, float],
              bmax: Tuple[float, float, float]) -> None:
        geoms: Dict[str, Geometry] = {}
        entity = _entity(entity_id, etype, name, bmin, bmax, geoms)
        _entity_geometries[entity_id] = geoms
        world.entities[entity_id] = entity
        world.geometries.update(geoms)

    # Floor slab (thin, under the walking surface).
    build(f"{spec.name}-floor", EntityType.FLOOR, f"{spec.name} floor",
          (ox, oy, oz - _WALL_THICKNESS), (ox + w, oy + d, oz))
    # Ceiling slab at the room height.
    build(f"{spec.name}-ceiling", EntityType.CEILING, f"{spec.name} ceiling",
          (ox, oy, oz + h), (ox + w, oy + d, oz + h + _WALL_THICKNESS))

    # Walls: north = -y edge, south = +y edge, west = -x edge,
    # east = +x edge. Each wall spans its edge and meets the
    # perpendicular walls at the corners.
    wall_defs = {
        "north": (f"{spec.name}-wall-north", EntityType.WALL,
                  f"{spec.name} north wall",
                  (ox, oy - t, oz), (ox + w, oy, oz + h)),
        "south": (f"{spec.name}-wall-south", EntityType.WALL,
                  f"{spec.name} south wall",
                  (ox, oy + d, oz), (ox + w, oy + d + t, oz + h)),
        "west": (f"{spec.name}-wall-west", EntityType.WALL,
                 f"{spec.name} west wall",
                 (ox - t, oy, oz), (ox, oy + d, oz + h)),
        "east": (f"{spec.name}-wall-east", EntityType.WALL,
                 f"{spec.name} east wall",
                 (ox + w, oy, oz), (ox + w + t, oy + d, oz + h)),
    }
    built_walls: Dict[str, Entity] = {}
    for side, (eid, etype, name, bmin, bmax) in wall_defs.items():
        if side not in exterior_walls:
            continue
        build(eid, etype, name, bmin, bmax)
        built_walls[side] = world.entities[eid]

    # The room itself: a semantic entity relating the boundary pieces.
    room = Entity(
        id=f"{spec.name}-room",
        name=f"{spec.name}",
        type=EntityType.ROOM,
        statement_state=_PROCEDURAL_STATE,
        provenance=_PROCEDURAL_PROVENANCE,
    )
    world.entities[room.id] = room

    # Openings: real geometry in the owning wall's plane, measured
    # from the spec. The opening sits at the wall's inner face.
    for i, opening in enumerate(spec.openings):
        if opening.wall not in built_walls:
            raise ValueError(
                f"opening {i} targets wall {opening.wall!r} which is "
                "not built for this room")
        offset = opening.lateral_offset
        if opening.wall in {"north", "south"}:
            x0 = ox + offset
            y0 = (oy - t) if opening.wall == "north" else (oy + d)
            bmin = (x0, y0, oz + opening.bottom)
            bmax = (x0 + opening.width, y0 + t, oz + opening.top)
        else:
            y0 = oy + offset
            x0 = (ox - t) if opening.wall == "west" else (ox + w)
            bmin = (x0, y0, oz + opening.bottom)
            bmax = (x0 + t, y0 + opening.width, oz + opening.top)
        etype = EntityType.DOOR if opening.kind == "door" else EntityType.WINDOW
        build(f"{spec.name}-opening-{i}", etype,
              f"{spec.name} {opening.kind} {i}", bmin, bmax)

    return world


def generate_floor_layout(specs: List[RoomGrammarSpec],
                          columns: int = 2,
                          gap: float = 0.0) -> WorldIR:
    """Generate a deterministic grid layout of rooms sharing walls.

    Rooms are placed in `columns`-wide rows in the given order. Adjacent
    rooms SHARE the wall between them: the earlier room builds it, the
    later room skips its duplicate side (the room graph then sees one
    wall adjoining both rooms, not two coincident ones).
    """
    if columns < 1:
        raise ValueError("columns must be >= 1")
    world = WorldIR(
        id="world-layout-" + "-".join(s.name for s in specs),
        main_branch_id="branch-main-layout",
    )
    # Deterministic placement: row-major grid from the origin, gap
    # between rooms (a shared wall's thickness already separates the
    # interior volumes; gap widens it further).
    y_cursor = 0.0
    max_row_height = 0.0
    for index, spec in enumerate(specs):
        col = index % columns
        if col == 0 and index > 0:
            y_cursor += max_row_height + gap
            max_row_height = 0.0
        x_cursor = col * (max(s.width for s in specs[:index + 1]
                              and [specs[(index // columns) * columns + c]
                                   for c in range(columns)
                                   if (index // columns) * columns + c < len(specs)])
                          or 0.0)
        max_row_height = max(max_row_height, spec.depth)

        # Which sides border a previously placed neighbor -> shared.
        exterior = {"north", "south", "east", "west"}
        if col > 0:
            exterior.discard("west")   # neighbor to the -x side built it
        if index >= columns:
            exterior.discard("north")  # neighbor above (-y) built it

        generate_room(spec, world=world, origin=(x_cursor, y_cursor, 0.0),
                      exterior_walls=exterior)
    return world
