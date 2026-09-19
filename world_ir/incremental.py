"""Incremental world updates (P4.21): given a set of directly-changed
entity ids (from `world_ir.diff.diff_worlds` or new evidence), compute the
full "affected region" -- every entity that is structurally connected to a
change and therefore needs re-checking, without touching the rest of a
potentially city-scale world.

This is the missing link between `world_ir/diff.py` (says WHAT changed) and
a real incremental pipeline (needs to know WHAT ELSE might be stale because
of it): if a wall's geometry changes, the room it's `part_of` and anything
it `supports`/`rests_on` are affected even though their own fields are
byte-identical. Walking only the changed entity's own diff would miss that
-- the point of "regional recompilation" per the P4.21 spec is exactly this
propagation, not a whole-world rebuild and not a single-entity patch.

No LLM, no heuristic guessing: this is a deterministic BFS over the
existing `Relationship` edges already on every `Entity` (the same edges
`engine/scene_graph/graph.py`'s `SceneGraph` walks), bounded by which
relationship kinds are considered "propagating" for a given call.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Collection, FrozenSet, Iterable, Optional, Set, Tuple

from world_ir.schema_v1 import Entity, Geometry, RelationshipKind
from world_ir.spatial_tiles import SpatialTiles
from world_ir.world_v1 import WorldIR

#: Relationship kinds that propagate "this might be stale" to a neighbor.
#: ADJACENT_TO/COLLIDES_WITH/OVERLAPS/CROSSES are deliberately excluded by
#: default -- mere proximity/contact doesn't imply the neighbor's own
#: derived state depends on this entity's fields the way containment and
#: load-bearing do. Callers with a real need (e.g. physics re-simulation
#: caring about contact graphs) can pass their own `relationship_kinds`.
DEFAULT_PROPAGATING_KINDS: FrozenSet[RelationshipKind] = frozenset({
    RelationshipKind.PART_OF,
    RelationshipKind.CONTAINS,
    RelationshipKind.ATTACHED_TO,
    RelationshipKind.SUPPORTS,
    RelationshipKind.RESTS_ON,
})


def affected_closure(
    world: WorldIR,
    changed_entity_ids: Iterable[str],
    *,
    relationship_kinds: Optional[Collection[RelationshipKind]] = None,
    max_hops: Optional[int] = None,
) -> Set[str]:
    """Every entity id reachable from `changed_entity_ids` by following
    edges of `relationship_kinds` (both directions -- an edge is a
    dependency regardless of which end declares it), up to `max_hops` hops.
    The changed ids themselves are always included. `max_hops=None` means
    unbounded (walk the whole connected component); `max_hops=0` returns
    exactly `changed_entity_ids` (clamped to ids that actually exist).

    Entity ids not present in `world.entities` are silently dropped -- an
    id from a stale diff naming an entity that no longer exists can't be
    "affected" in a live world.
    """
    kinds = set(relationship_kinds) if relationship_kinds is not None else set(DEFAULT_PROPAGATING_KINDS)
    start = {eid for eid in changed_entity_ids if eid in world.entities}

    if max_hops == 0:
        return start

    # Build an undirected adjacency map lazily, restricted to `kinds`, so a
    # single pass over all entities' relationship lists suffices regardless
    # of how large `start` is.
    adjacency: dict[str, Set[str]] = {}

    def _link(a: str, b: str) -> None:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    for entity_id, entity in world.entities.items():
        for rel in entity.relationships:
            if rel.kind in kinds:
                _link(entity_id, rel.target_id)

    visited: Set[str] = set(start)
    frontier = list(start)
    hops = 0
    while frontier and (max_hops is None or hops < max_hops):
        next_frontier: list[str] = []
        for entity_id in frontier:
            for neighbor in adjacency.get(entity_id, ()):
                if neighbor in world.entities and neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
        hops += 1

    return visited


@dataclass(frozen=True)
class IncrementalUpdateResult:
    """The output of `apply_incremental_update`: the new WorldIR plus a
    complete, deterministic report of exactly what changed, what was
    merely affected (relationship-reachable, not itself modified), and
    what spatial tiles that touches -- so a caller never has to guess
    whether "incremental" actually happened."""

    new_world: WorldIR
    #: Entity/geometry ids directly supplied as new/updated by the caller.
    changed_entity_ids: FrozenSet[str]
    changed_geometry_ids: FrozenSet[str]
    #: affected_closure() over the changed ids -- everything relationship-
    #: reachable that MIGHT need downstream re-derivation. This function
    #: does not itself re-derive them (it has no domain/perception logic);
    #: it reports the set so a real compiler stage can act on it.
    affected_entity_ids: FrozenSet[str]
    #: Tile keys (from world_ir.spatial_tiles.SpatialTiles) touched by any
    #: changed OR affected entity, in EITHER the base world's tile (old
    #: position) or the new world's tile (new position) -- an entity that
    #: moved tiles invalidates both, not just its destination. This is
    #: the "needs re-check" superset: it includes tiles that only hold an
    #: AFFECTED-but-unchanged entity (e.g. a room whose wall changed),
    #: even though nothing in that tile was actually replaced.
    invalidated_tile_ids: FrozenSet[Tuple[int, int, int]]
    #: Tile keys that hold a DIRECTLY changed entity (old or new
    #: position) -- a true content rebuild, not just a relationship
    #: flag. Always a subset of invalidated_tile_ids. In the example
    #: "tile A/B/C, new evidence affects tile B": rebuilt_tile_ids =
    #: {B}; invalidated_tile_ids may additionally contain a tile whose
    #: only connection to the change is an affected (not directly
    #: changed) entity; A and C, having no changed or affected entity
    #: at all, appear in neither set and are fully reused.
    rebuilt_tile_ids: FrozenSet[Tuple[int, int, int]]
    #: Entity/geometry ids present in base_world that were NOT directly
    #: changed -- new_world.entities[id] / new_world.geometries[id] is
    #: the SAME object as base_world's for every id in these sets.
    reused_entity_ids: FrozenSet[str]
    reused_geometry_ids: FrozenSet[str]


def apply_incremental_update(
    base_world: WorldIR,
    updated_entities: Iterable[Entity],
    *,
    updated_geometries: Iterable[Geometry] = (),
    relationship_kinds: Optional[Collection[RelationshipKind]] = None,
    max_hops: Optional[int] = None,
    tile_size: float = 10.0,
) -> IncrementalUpdateResult:
    """The localized WorldIR update primitive: `base_world` (World V1) +
    new/updated entities and geometries (from fresh evidence, already
    promoted by registration/reconstruction/perception) -> World V2,
    without rebuilding anything not touched.

    Contract (what makes this a real incremental update, not a full
    rebuild dressed up as one):
      - `new_world.entities[id] is base_world.entities[id]` for every
        entity id NOT in `changed_entity_ids` -- same object reference,
        not merely equal. Same for geometries. Verified by identity
        assertions in tests/test_world_ir_apply_incremental_update.py,
        not by equality after the fact.
      - `changed_entity_ids`/`changed_geometry_ids` are exactly what the
        caller supplied -- nothing else is mutated or replaced.
      - `affected_entity_ids` is the relationship-closure report
        (world_ir.incremental.affected_closure) over the changed set,
        PLUS any entity whose geometry_ids reference a changed geometry
        (a geometry changing under an entity is itself a reason that
        entity is affected, even if the entity's own fields are
        untouched). This function does NOT re-derive affected entities'
        fields -- it has no perception/compiler logic -- it only reports
        the set so a real compiler stage knows what to re-check.
      - `invalidated_tile_ids` uses the EXISTING world_ir.spatial_tiles.
        SpatialTiles grid (no second tiling system): every tile a
        changed-or-affected entity occupied in base_world OR occupies in
        new_world (an entity that moved tiles invalidates both its old
        and new tile).
      - `new_world.version = base_world.version + 1`; `new_world.id`
        stays the same (same world, next version) -- lineage between the
        two is the caller's job via `WorldStore.save_version(new_world,
        parent=<base_world's stored version id>)`, which independently
        re-derives changed_entity_ids/changed_geometry_ids via
        diff_worlds() -- callers can and should cross-check the two
        agree.

    Known limitation (documented, not hidden): `new_world` is built via
    a shallow copy of `base_world` with only `entities`/`geometries`
    replaced by fresh dict containers -- every OTHER field (materials,
    surfaces, components, branches, temporal_state, ...) is the SAME
    object as base_world's, not just equal. Mutating one of those
    collections on `new_world` after this call would also mutate
    `base_world`'s. This function only guarantees the entity/geometry
    identity contract the incremental-update mission asks for; a
    caller needing full structural independence should not mutate
    those shared collections in place.
    """
    updated_entities = list(updated_entities)
    updated_geometries = list(updated_geometries)
    changed_entity_ids = {e.id for e in updated_entities}
    changed_geometry_ids = {g.id for g in updated_geometries}

    # A changed geometry marks every entity that references it as
    # affected too -- their geometry_ids membership points at evidence
    # that's now stale, even though the entity's OWN fields (name, type,
    # relationships, ...) are untouched.
    geometry_owners = {
        eid for eid, entity in base_world.entities.items()
        if changed_geometry_ids & set(entity.geometry_ids)
    }
    seed_ids = changed_entity_ids | geometry_owners

    affected_entity_ids = affected_closure(
        base_world, seed_ids,
        relationship_kinds=relationship_kinds, max_hops=max_hops,
    )

    # New containers so base_world.entities/geometries are never mutated
    # in place; every value not overwritten below is the SAME object
    # reference copied out of base_world's dict, not a rebuilt copy.
    new_entities = dict(base_world.entities)
    for entity in updated_entities:
        new_entities[entity.id] = entity

    new_geometries = dict(base_world.geometries)
    for geometry in updated_geometries:
        new_geometries[geometry.id] = geometry

    new_world = copy.copy(base_world)
    new_world.entities = new_entities
    new_world.geometries = new_geometries
    new_world.version = base_world.version + 1

    reused_entity_ids = frozenset(base_world.entities) - changed_entity_ids
    reused_geometry_ids = frozenset(base_world.geometries) - changed_geometry_ids

    touch_ids = changed_entity_ids | affected_entity_ids
    entity_tile: dict[str, Set[Tuple[int, int, int]]] = {}
    if touch_ids:
        for tiles in (SpatialTiles(base_world, tile_size=tile_size), SpatialTiles(new_world, tile_size=tile_size)):
            for key in tiles.tile_ids():
                for eid in tiles.entities_in_tile(key):
                    entity_tile.setdefault(eid, set()).add(key)

    rebuilt_tile_ids: Set[Tuple[int, int, int]] = set()
    for eid in changed_entity_ids:
        rebuilt_tile_ids |= entity_tile.get(eid, set())

    invalidated_tile_ids: Set[Tuple[int, int, int]] = set(rebuilt_tile_ids)
    for eid in touch_ids:
        invalidated_tile_ids |= entity_tile.get(eid, set())

    return IncrementalUpdateResult(
        new_world=new_world,
        changed_entity_ids=frozenset(changed_entity_ids),
        changed_geometry_ids=frozenset(changed_geometry_ids),
        affected_entity_ids=frozenset(affected_entity_ids),
        invalidated_tile_ids=frozenset(invalidated_tile_ids),
        rebuilt_tile_ids=frozenset(rebuilt_tile_ids),
        reused_entity_ids=reused_entity_ids,
        reused_geometry_ids=reused_geometry_ids,
    )
