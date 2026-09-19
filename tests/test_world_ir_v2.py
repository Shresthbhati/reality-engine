"""Tests for world_ir/v2_extensions.py (P9-01 WorldIR 2.0: the open
items recorded in the ledger -- topology, temporal history, external
references -- as ADDITIVE schema extensions).

Spec rules under test (constitution + ledger P9-01 scope):

  - ADDITIVE only: v1 worlds (dicts without the new fields) must load
    unchanged; new fields default to empty and serialize only when
    non-empty (version stays 1-compatible; consumers who never touch
    the new fields see identical dicts).
  - Topology is DERIVED from existing relationship evidence (the
    EntityRegistry's Relationship kinds), never invented: part_of
    builds the containment hierarchy, adjacency/co_located build the
    connectivity graph. A relationship naming a missing target is
    REPORTED (reported_dangling), never silently dropped or
    fabricated into a node.
  - Temporal history is an ORDERED event log: range queries and
    per-entity queries are exact; events arrive already carrying
    their provenance (schema_v1.TemporalEvent) -- the log adds no
    provenance of its own.
  - External references are TYPED links (kind, uri, role, checksum)
    -- the "world points at evidence it does not own" seam (image
    files, COLMAP models, external reports). Round-trip must be
    byte-stable; checksum is recorded, never computed here.
"""

from __future__ import annotations

import pytest

from world_ir.entity import Entity, Relationship
from provenance import Provenance
from world_ir.schema_v1 import TemporalEvent, TemporalEventType
from world_ir.v2_extensions import (
    ExternalReference,
    TemporalHistory,
    TopologyGraph,
)


def _entity(eid: str, etype: str, rels: list[Relationship]) -> Entity:
    return Entity(id=eid, type=etype, relationships=rels)


class TestTopologyGraph:
    def test_part_of_hierarchy_children_and_parent(self):
        entities = [
            _entity("building-1", "building", []),
            _entity("room-1", "room",
                    [Relationship(kind="part_of", target_id="building-1")]),
            _entity("room-2", "room",
                    [Relationship(kind="part_of", target_id="building-1")]),
        ]
        topo = TopologyGraph.from_registry(entities)
        assert topo.parent_of("room-1") == "building-1"
        assert topo.children_of("building-1") == ["room-1", "room-2"]
        assert topo.parent_of("building-1") is None

    def test_ancestors_walks_the_chain(self):
        entities = [
            _entity("city", "city", []),
            _entity("block", "structure",
                    [Relationship(kind="part_of", target_id="city")]),
            _entity("building-1", "building",
                    [Relationship(kind="part_of", target_id="block")]),
        ]
        topo = TopologyGraph.from_registry(entities)
        assert topo.ancestors_of("building-1") == ["block", "city"]

    def test_adjacency_from_adjoining_relationships(self):
        entities = [
            _entity("room-1", "room", []),
            _entity("room-2", "room", []),
            _entity("room-3", "room", []),
            _entity("w1", "wall", [
                Relationship(kind="adjoins", target_id="room-1"),
                Relationship(kind="adjoins", target_id="room-2"),
            ]),
            _entity("w2", "wall", [
                Relationship(kind="adjoins", target_id="room-1"),
                Relationship(kind="adjoins", target_id="room-3"),
            ]),
        ]
        topo = TopologyGraph.from_registry(entities)
        assert topo.neighbors_of("room-1") == ["room-2", "room-3"]
        assert topo.neighbors_of("room-2") == ["room-1"]
        # A wall with a single adjoins target is a leaf, not a seam.
        assert topo.neighbors_of("w1") == []

    def test_dangling_relationship_is_reported_not_dropped(self):
        entities = [
            _entity("room-1", "room",
                    [Relationship(kind="part_of", target_id="ghost")]),
        ]
        topo = TopologyGraph.from_registry(entities)
        assert topo.reported_dangling == (
            ("room-1", "part_of", "ghost"),
        )
        assert topo.parent_of("room-1") is None

    def test_unknown_entity_query_raises(self):
        topo = TopologyGraph.from_registry([_entity("a", "room", [])])
        with pytest.raises(KeyError):
            topo.parent_of("nope")

    def test_deterministic_ordering(self):
        entities = [
            _entity("b", "building", []),
            _entity("a2", "room",
                    [Relationship(kind="part_of", target_id="b")]),
            _entity("a1", "room",
                    [Relationship(kind="part_of", target_id="b")]),
        ]
        topo = TopologyGraph.from_registry(entities)
        assert topo.children_of("b") == ["a1", "a2"]


class TestTemporalHistory:
    def test_events_ordered_by_timestamp(self):
        history = TemporalHistory.from_events([
            TemporalEvent(id="e2", timestamp=5.0, entity_id="a"),
            TemporalEvent(id="e1", timestamp=1.0, entity_id="b"),
        ])
        assert [e.id for e in history.events] == ["e1", "e2"]

    def test_range_query_is_exact(self):
        history = TemporalHistory.from_events([
            TemporalEvent(id="e1", timestamp=1.0, entity_id="a"),
            TemporalEvent(id="e2", timestamp=2.0, entity_id="a"),
            TemporalEvent(id="e3", timestamp=3.0, entity_id="b"),
        ])
        got = history.events_in_range(1.5, 3.0)
        assert [e.id for e in got] == ["e2", "e3"]
        # Boundary inclusive on both ends.
        assert [e.id for e in history.events_in_range(2.0, 2.0)] == ["e2"]

    def test_per_entity_history(self):
        history = TemporalHistory.from_events([
            TemporalEvent(id="e1", timestamp=1.0, entity_id="a"),
            TemporalEvent(id="e2", timestamp=2.0, entity_id="b"),
            TemporalEvent(id="e3", timestamp=3.0, entity_id="a"),
        ])
        assert [e.id for e in history.events_for("a")] == ["e1", "e3"]

    def test_empty_history_is_honest(self):
        history = TemporalHistory.from_events([])
        assert history.events == ()
        assert history.events_in_range(0.0, 1e9) == []


class TestExternalReference:
    def test_roundtrip(self):
        ref = ExternalReference(
            kind="image", uri="file:///captures/img_012.jpg",
            role="supporting_evidence", checksum="sha256:abc123",
            provenance=Provenance.OBSERVED,
        )
        d = ref.to_dict()
        assert ExternalReference.from_dict(d) == ref

    def test_checksum_is_recorded_never_computed(self):
        ref = ExternalReference(kind="image", uri="x", role="r")
        assert ref.checksum is None  # absence is honest
        with pytest.raises(ValueError):
            ExternalReference(kind="image", uri="x", role="r",
                              checksum="not-a-checksum")

    def test_entity_roundtrip_with_references(self):
        e = Entity(id="e1", type="column", external_references=(
            ExternalReference(kind="image", uri="file:///a.jpg",
                              role="supporting_evidence"),
        ))
        from world_ir.entity import Entity as E
        e2 = E.from_dict(e.to_dict())
        assert e2.external_references == e.external_references

    def test_v1_entity_dict_loads_without_references(self):
        e = Entity.from_dict({"id": "e1", "type": "column"})
        assert e.external_references == ()


class TestWorldIntegration:
    def test_worldir_carries_v2_sections_additively(self):
        from world_ir.world import WorldIR
        world = WorldIR(id="w1")
        world.entities.add(Entity(id="a", type="room"))
        world.entities.add(Entity(id="b", type="room"))
        world.entities.add(Entity(
            id="wall", type="wall",
            relationships=[
                Relationship(kind="adjoins", target_id="a"),
                Relationship(kind="adjoins", target_id="b"),
            ]))
        world.temporal_history = TemporalHistory.from_events([
            TemporalEvent(id="ev1", timestamp=1.0, entity_id="a"),
        ])
        d = world.to_dict()
        got = WorldIR.from_dict(d)
        assert got.temporal_history.events[0].id == "ev1"
        assert got.topology().neighbors_of("a") == ["b"]

    def test_v1_world_dict_loads_unchanged(self):
        from world_ir.world import WorldIR
        world = WorldIR(id="w1")
        world.entities.add(Entity(id="a", type="room"))
        d = world.to_dict()
        # Simulate a v1 serialization: no v2 keys at all.
        d.pop("temporal_history", None)
        d.pop("topology", None)
        got = WorldIR.from_dict(d)
        assert got.temporal_history.events == ()
        assert len(got.entities) == 1
