"""Outliner: world-hierarchy listing for the Studio (spec §27/§35 - Studio foundation).

Read-only, like Inspector -- an outliner that could mutate the world would
blur the "AI/tools query, engine owns truth" boundary (§7) the same way a
writable Inspector would. Hierarchy comes from Entity.component_ids, which
already exists on the schema (spec: PART_OF containment) -- this module
doesn't invent a new parent/child concept, it just walks the existing one
and packages it as an outliner would need: root nodes, expandable children,
grouped-by-type fallback for entities with no containment relationship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from world_ir import Entity, EntityType, WorldIR


@dataclass(frozen=True)
class OutlineNode:
    entity_id: str
    name: str
    type: EntityType
    provenance: str
    confidence: float
    children: List["OutlineNode"] = field(default_factory=list)


class Outliner:
    """Builds a hierarchical + type-grouped view of a WorldIR's entities."""

    def __init__(self, world: WorldIR):
        self.world = world

    def _node(self, entity: Entity, visited: set) -> OutlineNode:
        children = []
        for child_id in entity.component_ids:
            if child_id in visited:
                continue  # cycle guard -- component_ids is caller-supplied data, not enforced acyclic
            child = self.world.entities.get(child_id)
            if child is None:
                continue
            children.append(self._node(child, visited | {entity.id}))
        return OutlineNode(
            entity_id=entity.id,
            name=entity.name or entity.id,
            type=entity.type,
            provenance=entity.provenance.value,
            confidence=entity.confidence,
            children=children,
        )

    def hierarchy(self) -> List[OutlineNode]:
        """Root nodes: entities not referenced as any other entity's component."""
        child_ids = {cid for e in self.world.entities.values() for cid in e.component_ids}
        roots = [e for e in self.world.entities.values() if e.id not in child_ids]
        return [self._node(root, set()) for root in roots]

    def grouped_by_type(self) -> Dict[str, List[OutlineNode]]:
        """Flat fallback view: every entity, grouped by EntityType, no containment nesting."""
        groups: Dict[str, List[OutlineNode]] = {}
        for entity in self.world.entities.values():
            node = OutlineNode(
                entity_id=entity.id, name=entity.name or entity.id, type=entity.type,
                provenance=entity.provenance.value, confidence=entity.confidence,
            )
            groups.setdefault(entity.type.value, []).append(node)
        return groups

    def find_by_name(self, query: str) -> List[Entity]:
        """Case-insensitive substring search over entity names/ids -- the outliner's search box."""
        q = query.lower()
        return [
            e for e in self.world.entities.values()
            if q in (e.name or "").lower() or q in e.id.lower()
        ]
