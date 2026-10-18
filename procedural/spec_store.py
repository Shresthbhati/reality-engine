"""Stored generator specs + scoped regeneration (World Factory, task C1).

Every entity a procedural generator creates is stamped
`custom_properties["generated_by"] = <semantic entity id>`; the semantic
entity (`<name>-room` / `-corridor` / `-city`) also carries the full
`GeneratedSpec` in `custom_properties["spec"]`. That makes a generated
object re-derivable: `regenerate` removes exactly the entities stamped
for one spec, re-runs its generator with edited params, then re-runs the
specs ATTACHED to it (a corridor flush to a room's east edge follows the
room's new size) -- and nothing else. Everything is plain WorldIR data;
no side tables.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Optional

from procedural.apartment_grammar import DEFAULT_HEIGHT, generate_apartment, plan_from_params
from procedural.building_grammar import BuildingSpec, generate_building
from procedural.district_grammar import DistrictSpec, generate_district
from procedural.layout_solver import ApartmentSpec
from procedural.room_grammar import RoomGrammarSpec, WallOpening, generate_room
from procedural.urban_grammar import (
    CityGrammarSpec,
    CorridorGrammarSpec,
    generate_city,
    generate_corridor,
)
from world_ir.world_v1 import WorldIR

GENERATORS = {
    "room": "room_grammar@1",
    "corridor": "corridor_grammar@1",
    "city": "city_grammar@1",
    "apartment": "apartment_grammar@1",
    "building": "building_grammar@1",
    "district": "district_grammar@1",
}
SIDES = ("east", "west", "north", "south")


class UnknownSpec(LookupError):
    """No stored spec for that semantic entity id."""


@dataclass
class GeneratedSpec:
    kind: str                      # room | corridor | city
    name: str
    params: dict                   # exact validated inputs, JSON-safe
    seed: int = 0
    origin: list = None            # [x, y, z], grammar-local axes
    attach: Optional[dict] = None  # {"to": semantic id, "side": east|west|north|south}
    generator: str = ""

    def __post_init__(self):
        if self.kind not in GENERATORS:
            raise ValueError(f"unknown generated kind {self.kind!r}")
        self.origin = list(self.origin or (0.0, 0.0, 0.0))
        self.generator = self.generator or GENERATORS[self.kind]
        if self.attach is not None:
            if self.kind != "corridor":
                raise ValueError("only corridors can be attached")
            if self.attach.get("side") not in SIDES:
                raise ValueError(f"attach side must be one of {SIDES}")
            if not self.attach.get("to"):
                raise ValueError("attach needs a target entity id ('to')")

    @property
    def entity_id(self) -> str:
        return f"{self.name}-{self.kind}"

    @property
    def depends_on(self) -> list:
        deps = [self.attach["to"]] if self.attach else []
        deps += [c for c in self.params.get("connects", []) if c not in deps]
        return deps


# ---------------------------------------------------------------- stamping

def stamp(world: WorldIR, semantic_entity_id: str, spec: GeneratedSpec,
          created_ids) -> None:
    """Mark every entity the generator created as owned by this spec and
    store the spec on the semantic entity."""
    for eid in created_ids:
        world.entities[eid].custom_properties["generated_by"] = semantic_entity_id
    world.entities[semantic_entity_id].custom_properties["spec"] = asdict(spec)


def load_specs(world: WorldIR) -> dict:
    """{semantic entity id: GeneratedSpec} for every stamped spec."""
    out = {}
    for eid, ent in world.entities.items():
        raw = ent.custom_properties.get("spec")
        if isinstance(raw, dict) and ent.custom_properties.get("generated_by") == eid:
            out[eid] = GeneratedSpec(**raw)
    return out


def owned_ids(world: WorldIR, semantic_entity_id: str) -> list:
    return [eid for eid, e in world.entities.items()
            if e.custom_properties.get("generated_by") == semantic_entity_id]


def _remove_owned(world: WorldIR, semantic_entity_id: str) -> list:
    removed = owned_ids(world, semantic_entity_id)
    for eid in removed:
        for gid in world.entities[eid].geometry_ids:
            world.geometries.pop(gid, None)
        del world.entities[eid]
    return removed


def dependents(specs: dict, sid: str, include_connects: bool = False) -> list:
    """Transitive dependents of `sid` in BFS order. Attached specs are
    regenerated with their target; `include_connects` also counts plain
    corridor `connects` (used to protect deletes from dangling links)."""
    def direct(x):
        return [s.entity_id for s in specs.values()
                if (s.attach and s.attach["to"] == x)
                or (include_connects and x in s.params.get("connects", []))]
    out, queue = [], [sid]
    while queue:
        for d in direct(queue.pop(0)):
            if d not in out and d != sid:
                out.append(d)
                queue.append(d)
    return out


# --------------------------------------------------------------- generation

def _corridor_size(spec: GeneratedSpec) -> tuple:
    p = spec.params
    return ((p["length"], p["width"]) if p.get("axis", "x") == "x"
            else (p["width"], p["length"]))


def attached_origin(world: Optional[WorldIR], spec: GeneratedSpec) -> list:
    """Origin placing `spec` flush against `attach.side` of its target,
    from the target's CURRENT floor footprint (world Y-up -> local Z-up)."""
    target = (load_specs(world) if world is not None else {}).get(spec.attach["to"])
    floor = world.entities.get(f"{target.name}-floor") if target else None
    if floor is None or not floor.geometry_ids:
        raise ValueError(
            f"cannot attach to {spec.attach['to']!r}: no stored room/corridor "
            "spec with a floor footprint")
    g = world.geometries[floor.geometry_ids[0]]
    xmin, xmax = g.bounds_min.x, g.bounds_max.x
    ymin, ymax = -g.bounds_max.z, -g.bounds_min.z     # local y = -world z
    oz = g.bounds_max.y
    cw, cd = _corridor_size(spec)
    side = spec.attach["side"]
    ox, oy = {
        "east": (xmax, ymin), "west": (xmin - cw, ymin),
        "north": (xmin, ymin - cd), "south": (xmin, ymax),
    }[side]
    return [ox, oy, oz]


def generate(world: Optional[WorldIR], spec: GeneratedSpec) -> WorldIR:
    """Run the spec's generator into `world` and stamp what it created."""
    if spec.attach:
        spec = replace(spec, origin=attached_origin(world, spec))
    p = spec.params
    origin = tuple(spec.origin)
    before = set(world.entities) if world is not None else set()
    if spec.kind == "room":
        out = generate_room(
            RoomGrammarSpec(spec.name, p["width"], p["depth"], p["height"],
                            [WallOpening(**o) for o in p.get("openings", [])],
                            p.get("seed", 0)),
            world=world, origin=origin)
    elif spec.kind == "corridor":
        out = generate_corridor(
            CorridorGrammarSpec(spec.name, p["length"], p.get("width", 2.0),
                                p.get("height", 2.6), p.get("axis", "x"),
                                list(p.get("connects", []))),
            world=world, origin=origin)
    elif spec.kind == "apartment":
        # params = ApartmentSpec fields (+ optional "rects" / "plan_index" / "height").
        # Stale rects (footprint or room list edited) are re-solved deterministically.
        a_spec = ApartmentSpec.from_dict({"name": spec.name, **p})
        out = generate_apartment(a_spec, plan_from_params(a_spec, p), world=world,
                                 origin=origin, height=p.get("height", DEFAULT_HEIGHT))
    elif spec.kind == "building":
        out = generate_building(BuildingSpec.from_params(spec.name, p), world=world,
                                origin=origin)
    elif spec.kind == "district":
        out = generate_district(DistrictSpec.from_params(spec.name, p), world=world,
                                origin=origin)
    else:
        out = generate_city(
            CityGrammarSpec(spec.name, p["blocks_x"], p["blocks_y"], p["lot_size"],
                            p["street_width"], p["min_height"], p["max_height"],
                            p.get("seed", 0)),
            world=world, origin=origin)
    stamp(out, spec.entity_id, spec, [e for e in out.entities if e not in before])
    return out


# ------------------------------------------------------------ regeneration

def _fingerprints(world: WorldIR) -> dict:
    return {
        eid: json.dumps(
            {"e": e.to_dict(),
             "g": [world.geometries[g].to_dict() for g in e.geometry_ids
                   if g in world.geometries]},
            sort_keys=True).encode()
        for eid, e in world.entities.items()
    }


def regenerate(world: WorldIR, semantic_entity_id: str, new_params: Optional[dict] = None):
    """Edit one generated object. `new_params` is merged over the stored
    params; the reserved keys "origin" and "attach" update those spec
    fields. Returns (new WorldIR, changed_ids, removed_ids, unchanged_ids);
    the input world is never mutated. Only the spec and the specs attached
    to it (transitively) are re-run."""
    world = WorldIR.from_dict(world.to_dict())
    specs = load_specs(world)
    if semantic_entity_id not in specs:
        raise UnknownSpec(semantic_entity_id)
    spec = specs[semantic_entity_id]
    patch = dict(new_params or {})
    origin = patch.pop("origin", spec.origin)
    attach = patch.pop("attach", spec.attach)
    merged = {**spec.params, **patch}
    new_spec = replace(spec, params=merged, seed=merged.get("seed", spec.seed),
                       origin=list(origin), attach=attach)
    new_spec.__post_init__()

    chain = dependents(specs, semantic_entity_id)
    if attach and (attach["to"] == semantic_entity_id or attach["to"] in chain):
        raise ValueError("attach would create a dependency cycle")

    before = _fingerprints(world)
    for sid in [semantic_entity_id] + chain:
        _remove_owned(world, sid)
        generate(world, new_spec if sid == semantic_entity_id else specs[sid])
    after = _fingerprints(world)

    changed = [e for e, b in after.items() if before.get(e) != b]
    removed = [e for e in before if e not in after]
    unchanged = [e for e in after if e not in changed]
    return world, changed, removed, unchanged


def delete_specs(world: WorldIR, sids) -> tuple:
    """Remove the given specs' entities (and geometries). Returns
    (new WorldIR, removed entity ids); the input is never mutated."""
    world = WorldIR.from_dict(world.to_dict())
    removed = []
    for sid in sids:
        removed += _remove_owned(world, sid)
    return world, removed
