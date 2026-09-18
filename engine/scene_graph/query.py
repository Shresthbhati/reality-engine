"""WorldQuery: unified semantic + spatial + filtered query facade (Agent 3
world-construction "Query Engine" deliverable).

`SceneGraph` (graph.py) already answers relationship questions and
`SpatialIndex` (spatial_index.py) already answers geometric questions. Both
take a `predicate: Callable[[Entity], bool]` for extra filtering, which is
how confidence/evidence/temporal/kind filtering can already be composed by a
caller -- but the brief asks for these as named, discoverable query methods
rather than something every caller has to hand-roll. This module is that
thin facade: it owns no new data model, walks no path an existing query
doesn't already walk, and is intentionally free of any LLM/inference step
(spec sec 59 AI-FREE SMARTNESS) -- every answer is a deterministic function
of WorldIR data.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR

from .graph import Edge, SceneGraph
from .spatial_index import Bounds, Point, SpatialIndex, _aabb_overlaps, _entity_bounds


class WorldQuery:
    """Built once from a WorldIR snapshot, like SceneGraph/SpatialIndex
    (which it wraps). Does not observe later mutations -- rebuild after
    edits."""

    def __init__(self, world: WorldIR):
        self.world = world
        self.graph = SceneGraph(world)
        self.spatial = SpatialIndex(world)

    # ---- relationship queries (delegate to SceneGraph) ----

    def contents_of(self, container_id: str) -> List[Entity]:
        return self.graph.contents_of(container_id)

    def container_of(self, entity_id: str) -> Optional[Entity]:
        return self.graph.container_of(entity_id)

    def supporters_of(self, entity_id: str) -> List[Entity]:
        return self.graph.supporters_of(entity_id)

    # ---- spatial queries (delegate to SpatialIndex) ----

    def near(
        self, point: Point, radius: float, predicate: Optional[Callable[[Entity], bool]] = None,
    ) -> List[Tuple[Entity, float]]:
        """Every indexed entity within `radius` of `point`, nearest first."""
        return self.spatial.within_radius(point, radius, predicate=predicate)

    def within(self, bounds_min: Point, bounds_max: Point) -> List[Entity]:
        """Entities positioned inside (or overlapping, for entities with
        real geometry bounds) the given AABB region."""
        return self.spatial.within_region(bounds_min, bounds_max)

    def intersects(self, bounds_min: Point, bounds_max: Point) -> List[Entity]:
        """Entities whose own geometry AABB overlaps the given region.
        Unlike `within`, an entity with no real geometry bounds (a
        transform-only point placement) never matches here -- intersection
        is a statement about an entity's own extent, not its position."""
        region: Bounds = (bounds_min, bounds_max)
        results = []
        for entity in self.world.entities.values():
            bounds = _entity_bounds(self.world, entity)
            if bounds is not None and _aabb_overlaps(bounds, region):
                results.append(entity)
        results.sort(key=lambda e: e.id)
        return results

    # ---- semantic / attribute filters ----

    def query_by_kind(self, kind: EntityType) -> List[Entity]:
        """Every entity of a given EntityType (semantic class), e.g. every
        ROOM, or every CITY-scale DISTRICT."""
        return sorted(
            (e for e in self.world.entities.values() if e.type == kind),
            key=lambda e: e.id,
        )

    def by_confidence(self, min_confidence: float, entities: Optional[List[Entity]] = None) -> List[Entity]:
        """Entities (optionally restricted to an already-filtered list)
        whose confidence is >= min_confidence."""
        pool = entities if entities is not None else list(self.world.entities.values())
        return sorted((e for e in pool if e.confidence >= min_confidence), key=lambda e: e.id)

    def by_evidence(self, evidence_id: str) -> List[Entity]:
        """Entities with at least one Observation whose data_uri or id
        references the given evidence id (session EvidenceItem.id or
        Observation.id/data_uri) -- traces a world entity back to the raw
        capture it was derived from."""
        results = []
        for entity in self.world.entities.values():
            for obs in entity.observations:
                if obs.id == evidence_id or evidence_id in obs.data_uri:
                    results.append(entity)
                    break
        results.sort(key=lambda e: e.id)
        return results

    def as_of(self, timestamp: float) -> List[Entity]:
        """Temporal filter: entities that existed at `timestamp`, i.e. have
        no CREATION event after it and no DESTRUCTION event at or before it.
        An entity with no temporal_events at all is treated as always
        present (nothing observed says otherwise -- never fabricate a
        creation/destruction time)."""
        from world_ir.schema_v1 import TemporalEventType

        results = []
        for entity in self.world.entities.values():
            if not entity.temporal_events:
                results.append(entity)
                continue
            created_after = any(
                ev.type == TemporalEventType.CREATION and ev.timestamp > timestamp
                for ev in entity.temporal_events
            )
            destroyed_by = any(
                ev.type == TemporalEventType.DESTRUCTION and ev.timestamp <= timestamp
                for ev in entity.temporal_events
            )
            if not created_after and not destroyed_by:
                results.append(entity)
        results.sort(key=lambda e: e.id)
        return results
