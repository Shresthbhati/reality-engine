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

from typing import Collection, FrozenSet, Iterable, Optional, Set

from world_ir.schema_v1 import RelationshipKind
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
