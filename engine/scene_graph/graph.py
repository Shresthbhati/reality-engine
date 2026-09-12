"""SceneGraph: AI-free structured queries over WorldIR relationships
(spec sec 22 SCENE GRAPH, sec 59 AI-FREE SMARTNESS).

The data already exists: every Entity carries a list of directed
Relationship edges (kind, target_id, provenance, confidence) -- see
world_ir/schema_v1.py. What's missing is a query engine that answers
structured questions ("what's inside Room 4?", "is this window part of
that wall?") by walking those edges deterministically, the way spec sec
59 describes: query -> scene graph -> WorldIR -> deterministic answer,
with no LLM anywhere in the path.

This is read-only, like Inspector and Outliner -- a query engine that
could also write would blur the same "AI/tools query, engine owns truth"
boundary (spec sec 7) those already respect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from world_ir import Entity, RelationshipKind, WorldIR


@dataclass(frozen=True)
class Edge:
    """One directed relationship, with both endpoints resolved (Inspector's
    get_relationships() already returns a target_name for one entity; this
    is the same idea generalized to the whole graph)."""
    source_id: str
    kind: RelationshipKind
    target_id: str
    confidence: float
    provenance: str


class SceneGraph:
    def __init__(self, world: WorldIR):
        self.world = world

    # ---- direct edges ----

    def edges_from(self, entity_id: str, kind: Optional[RelationshipKind] = None) -> List[Edge]:
        """Outgoing edges: relationships this entity declares about itself."""
        entity = self.world.entities.get(entity_id)
        if entity is None:
            return []
        return [
            Edge(entity_id, rel.kind, rel.target_id, rel.confidence, rel.provenance.value)
            for rel in entity.relationships
            if kind is None or rel.kind == kind
        ]

    def edges_to(self, entity_id: str, kind: Optional[RelationshipKind] = None) -> List[Edge]:
        """Incoming edges: every other entity that declares a relationship
        targeting entity_id. O(n) over all entities -- there is no reverse
        index, and building one prematurely for a graph this small would be
        the kind of unrequested optimization the project conventions warn
        against; add one if this becomes a real bottleneck on a large world.
        """
        edges = []
        for source in self.world.entities.values():
            for rel in source.relationships:
                if rel.target_id == entity_id and (kind is None or rel.kind == kind):
                    edges.append(Edge(source.id, rel.kind, entity_id, rel.confidence, rel.provenance.value))
        return edges

    # ---- the item-59 example queries, answered as ordinary graph lookups ----

    def contents_of(self, container_id: str) -> List[Entity]:
        """'Which objects are inside Room 4?' -- entities with a PART_OF or
        CONTAINS edge pointing at container_id, or that container_id
        declares CONTAINS toward."""
        contained_ids: Set[str] = set()
        for edge in self.edges_to(container_id, RelationshipKind.PART_OF):
            contained_ids.add(edge.source_id)
        for edge in self.edges_from(container_id, RelationshipKind.CONTAINS):
            contained_ids.add(edge.target_id)
        return [self.world.entities[eid] for eid in contained_ids if eid in self.world.entities]

    def container_of(self, entity_id: str) -> Optional[Entity]:
        """The inverse of contents_of: what is entity_id part of/contained in."""
        for edge in self.edges_from(entity_id, RelationshipKind.PART_OF):
            container = self.world.entities.get(edge.target_id)
            if container is not None:
                return container
        for edge in self.edges_to(entity_id, RelationshipKind.CONTAINS):
            container = self.world.entities.get(edge.source_id)
            if container is not None:
                return container
        return None

    def supporters_of(self, entity_id: str) -> List[Entity]:
        """'What is holding this up?' -- entities with a SUPPORTS/RESTS_ON
        edge targeting entity_id. Generalizes Inspector.find_supporting()
        (which returns ids) to return resolved Entity objects."""
        ids = {e.source_id for e in self.edges_to(entity_id, RelationshipKind.SUPPORTS)}
        ids |= {e.source_id for e in self.edges_to(entity_id, RelationshipKind.RESTS_ON)}
        return [self.world.entities[eid] for eid in ids if eid in self.world.entities]

    def unknown_relationship_entities(self) -> List[Entity]:
        """'Which surfaces are marked UNKNOWN?' -- generalized: any entity
        with at least one relationship of unresolved kind."""
        return [
            e for e in self.world.entities.values()
            if any(rel.kind == RelationshipKind.UNKNOWN for rel in e.relationships)
        ]

    def query_by_kind(self, kind: RelationshipKind) -> List[Edge]:
        """Every edge of a given kind across the whole world."""
        edges = []
        for source in self.world.entities.values():
            for rel in source.relationships:
                if rel.kind == kind:
                    edges.append(Edge(source.id, rel.kind, rel.target_id, rel.confidence, rel.provenance.value))
        return edges

    # ---- path queries ----

    def path_exists(self, from_id: str, to_id: str, kinds: Optional[List[RelationshipKind]] = None) -> bool:
        """BFS reachability from from_id to to_id, following only outgoing
        edges of the given kinds (any kind if None). Answers questions like
        'is this window structurally connected to that foundation'."""
        if from_id not in self.world.entities or to_id not in self.world.entities:
            return False
        if from_id == to_id:
            return True

        visited = {from_id}
        frontier = [from_id]
        while frontier:
            current = frontier.pop(0)
            for edge in self.edges_from(current):
                if kinds is not None and edge.kind not in kinds:
                    continue
                if edge.target_id == to_id:
                    return True
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    frontier.append(edge.target_id)
        return False

    def all_edges(self) -> List[Edge]:
        """Every relationship in the world, as resolved Edge objects --
        useful for export/inspection/debugging the whole graph at once."""
        edges = []
        for source in self.world.entities.values():
            for rel in source.relationships:
                edges.append(Edge(source.id, rel.kind, rel.target_id, rel.confidence, rel.provenance.value))
        return edges
