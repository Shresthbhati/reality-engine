"""Procedural corridor and city grammars (extends room_grammar.py).

Same contract as the room grammar: DATA in, a real WorldIR out,
deterministic (same spec + seed -> identical world), every generated
statement StatementState.PROCEDURAL / Provenance.GENERATED. Both reuse
generate_room so walls, floors and ceilings are built one way only.

  - generate_corridor: a straight circulation space. Two long walls, open
    ends, a CORRIDOR entity ADJACENT_TO each room it `connects`.
  - generate_city: a grid of building lots separated by streets. A CITY
    entity CONTAINS BUILDING and STREET entities; each BUILDING CONTAINS
    its room. Building heights come from the seeded RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from provenance import Provenance
from procedural.room_grammar import RoomGrammarSpec, _entity, generate_room
from world_ir.schema_v1 import Entity, EntityType, Geometry, Relationship, RelationshipKind
from world_ir.statement_state import StatementState
from world_ir.world_v1 import WorldIR

MAX_CITY_LOTS = 100
_SLAB = 0.2


def _link(kind: RelationshipKind, target_id: str) -> Relationship:
    return Relationship(kind=kind, target_id=target_id, confidence=1.0,
                        provenance=Provenance.GENERATED)


def _semantic(entity_id: str, etype: EntityType, name: str) -> Entity:
    return Entity(id=entity_id, name=name, type=etype,
                  statement_state=StatementState.PROCEDURAL,
                  provenance=Provenance.GENERATED)


def _fresh_world(name: str) -> WorldIR:
    return WorldIR(id=f"world-{name}", main_branch_id=f"branch-main-{name}")


@dataclass
class CorridorGrammarSpec:
    name: str
    length: float
    width: float = 2.0
    height: float = 2.6
    axis: str = "x"                       # corridor runs along "x" or "y"
    connects: List[str] = field(default_factory=list)  # existing room entity ids
    seed: int = 0

    def __post_init__(self):
        if self.length <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("corridor extents must be positive")
        if self.axis not in {"x", "y"}:
            raise ValueError(f"unknown corridor axis {self.axis!r}")


def generate_corridor(spec: CorridorGrammarSpec, world: Optional[WorldIR] = None,
                      origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> WorldIR:
    if world is None:
        world = _fresh_world(spec.name)
    for rid in spec.connects:
        if rid not in world.entities:
            raise ValueError(f"cannot connect to unknown room {rid!r}")
    if spec.axis == "x":
        room = RoomGrammarSpec(spec.name, spec.length, spec.width, spec.height)
        walls = {"north", "south"}
    else:
        room = RoomGrammarSpec(spec.name, spec.width, spec.length, spec.height)
        walls = {"west", "east"}
    generate_room(room, world=world, origin=origin, exterior_walls=walls)
    # The generated "room" semantic entity is this corridor.
    world.entities.pop(f"{spec.name}-room")
    corridor = _semantic(f"{spec.name}-corridor", EntityType.CORRIDOR, spec.name)
    corridor.relationships = [_link(RelationshipKind.ADJACENT_TO, rid) for rid in spec.connects]
    world.entities[corridor.id] = corridor
    return world


@dataclass
class CityGrammarSpec:
    name: str
    blocks_x: int = 3
    blocks_y: int = 3
    lot_size: float = 20.0
    street_width: float = 8.0
    min_height: float = 6.0
    max_height: float = 30.0
    seed: int = 0

    def __post_init__(self):
        if self.blocks_x < 1 or self.blocks_y < 1:
            raise ValueError("a city needs at least one block in each direction")
        if self.blocks_x * self.blocks_y > MAX_CITY_LOTS:
            raise ValueError(f"a city is limited to {MAX_CITY_LOTS} lots")
        if self.lot_size <= 0 or self.street_width <= 0:
            raise ValueError("lot size and street width must be positive")
        if self.min_height <= 0 or self.max_height < self.min_height:
            raise ValueError("building heights need 0 < min_height <= max_height")


def generate_city(spec: CityGrammarSpec, world: Optional[WorldIR] = None,
                  origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> WorldIR:
    if world is None:
        world = _fresh_world(spec.name)
    ox, oy, oz = origin
    pitch = spec.lot_size + spec.street_width
    total_w = spec.blocks_x * pitch + spec.street_width
    total_d = spec.blocks_y * pitch + spec.street_width
    rng = random.Random(spec.seed)
    city = _semantic(f"{spec.name}-city", EntityType.CITY, spec.name)

    def slab(entity_id: str, etype: EntityType, label: str, x0, y0, x1, y1) -> None:
        geoms: Dict[str, Geometry] = {}
        entity = _entity(entity_id, etype, label, (x0, y0, oz - _SLAB), (x1, y1, oz), geoms)
        world.entities[entity_id] = entity
        world.geometries.update(geoms)
        city.relationships.append(_link(RelationshipKind.CONTAINS, entity_id))

    for k in range(spec.blocks_y + 1):          # streets running along x
        y0 = oy + k * pitch
        slab(f"{spec.name}-street-x{k}", EntityType.STREET, f"{spec.name} street x{k}",
             ox, y0, ox + total_w, y0 + spec.street_width)
    for k in range(spec.blocks_x + 1):          # streets running along y
        x0 = ox + k * pitch
        slab(f"{spec.name}-street-y{k}", EntityType.STREET, f"{spec.name} street y{k}",
             x0, oy, x0 + spec.street_width, oy + total_d)

    for j in range(spec.blocks_y):
        for i in range(spec.blocks_x):
            bname = f"{spec.name}-b{i}-{j}"
            height = round(rng.uniform(spec.min_height, spec.max_height), 1)
            generate_room(
                RoomGrammarSpec(bname, spec.lot_size, spec.lot_size, height),
                world=world,
                origin=(ox + spec.street_width + i * pitch,
                        oy + spec.street_width + j * pitch, oz))
            building = _semantic(f"{bname}-building", EntityType.BUILDING, bname)
            building.relationships.append(_link(RelationshipKind.CONTAINS, f"{bname}-room"))
            world.entities[building.id] = building
            city.relationships.append(_link(RelationshipKind.CONTAINS, building.id))

    world.entities[city.id] = city
    return world
