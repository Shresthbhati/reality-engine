"""Tests for the temporal state layer (engine/simulation/temporal.py):
causal event graph over the real EventBus, WorldIR deep-state snapshots,
branch inheritance with structural comparison, and deterministic
non-destructive replay.

Event fixtures use genuine EventBus.emit() streams (including physics
contact/impact events with cause_event_ids chains), not hand-built lists,
so the graph queries are tested against the actual logging path.
"""

from __future__ import annotations

import pytest

from engine.simulation.temporal import (
    BranchManager,
    EventGraph,
    SnapshotStore,
    replay,
)
from events import EventBus
from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Vector3,
)

CONTACT = "ContactEvent"
IMPACT = "ImpactEvent"
BREAK = "BreakEvent"


# ---------------------------------------------------------------------------
# Event fixtures: real bus streams
# ---------------------------------------------------------------------------


def _chain_bus() -> EventBus:
    """wind -> roof damage -> water entry, plus an unrelated contact."""
    bus = EventBus(seed=7)
    e_wind = bus.emit("WindEvent", timestamp=1.0, tick=1,
                      source_refs=("env-wind",), target_refs=("roof-1",),
                      parameters={"speed_mps": 28.0})
    e_roof = bus.emit("DamageEvent", timestamp=1.2, tick=2,
                      source_refs=("env-wind",), target_refs=("roof-1",),
                      cause_event_ids=(e_wind.event_id,),
                      severity="damage")
    bus.emit("FloodEvent", timestamp=1.5, tick=3,
             source_refs=("roof-1",), target_refs=("room-1",),
             cause_event_ids=(e_roof.event_id,),
             severity="water-entry")
    bus.emit(CONTACT, timestamp=1.6, tick=3,
             source_refs=("crate",), target_refs=("floor",),
             parameters={"approach_speed": 0.4})
    return bus


# ---------------------------------------------------------------------------
# Event graph
# ---------------------------------------------------------------------------


class TestEventGraph:
    def test_ancestors_walks_transitive_causes(self):
        graph = EventGraph(_chain_bus())
        flood = [e for e in graph.events if e.type == "FloodEvent"][0]
        ancestors = graph.ancestors_of(flood.event_id)
        types = [a.type for a in ancestors]
        assert types == ["DamageEvent", "WindEvent"]  # nearest first

    def test_descendants_finds_all_consequences(self):
        graph = EventGraph(_chain_bus())
        wind = [e for e in graph.events if e.type == "WindEvent"][0]
        descendants = graph.descendants_of(wind.event_id)
        assert [d.type for d in descendants] == ["DamageEvent", "FloodEvent"]

    def test_root_causes_terminate_the_chain(self):
        graph = EventGraph(_chain_bus())
        flood = [e for e in graph.events if e.type == "FloodEvent"][0]
        roots = graph.root_causes_of(flood.event_id)
        assert [r.type for r in roots] == ["WindEvent"]

    def test_unrelated_events_are_not_ancestors(self):
        graph = EventGraph(_chain_bus())
        flood = [e for e in graph.events if e.type == "FloodEvent"][0]
        assert all(a.type != CONTACT for a in graph.ancestors_of(flood.event_id))

    def test_entity_queries_cover_sources_and_targets(self):
        graph = EventGraph(_chain_bus())
        roof_events = graph.events_affecting_entity("roof-1")
        assert {e.type for e in roof_events} == {"WindEvent", "DamageEvent", "FloodEvent"}

    def test_events_after_is_strict(self):
        graph = EventGraph(_chain_bus())
        wind = [e for e in graph.events if e.type == "WindEvent"][0]
        after = graph.events_after(wind.event_id)
        assert all(e.tick > wind.tick for e in after)
        assert {e.type for e in after} == {"DamageEvent", "FloodEvent", CONTACT}

    def test_unknown_event_raises_not_invented(self):
        graph = EventGraph(_chain_bus())
        with pytest.raises(KeyError):
            graph.ancestors_of("evt-does-not-exist")


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def _small_world() -> WorldIR:
    world = WorldIR(id="w-temporal", name="t", main_branch_id="b-temporal")
    world.entities["e1"] = Entity(
        id="e1", type=EntityType.STRUCTURE, name="box",
        geometry_ids=["g1"], confidence=0.8,
    )
    world.geometries["g1"] = Geometry(
        id="g1", type=GeometryType.BOX,
        bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(1, 1, 1),
    )
    return world


class TestSnapshots:
    def test_capture_and_restore_roundtrip(self):
        store = SnapshotStore()
        world = _small_world()
        snap = store.capture(world, tick=10, timestamp=1.0, label="pre-sim")
        assert snap.entity_count == 1
        assert snap.snapshot_id == "snap-main-00000010-0001"

        world.entities["e2"] = Entity(id="e2", type=EntityType.DEBRIS, name="junk")
        world.entities["e1"].confidence = 0.1

        store.restore(snap.snapshot_id, world)
        assert set(world.entities) == {"e1"}
        assert world.entities["e1"].confidence == 0.8

    def test_snapshot_is_isolated_from_later_mutation(self):
        store = SnapshotStore()
        world = _small_world()
        snap = store.capture(world, tick=0, timestamp=0.0)
        world.entities["e1"].name = "mutated"
        world.entities["e9"] = Entity(id="e9", type=EntityType.DEBRIS, name="x")
        # Restoring into a FRESH world must yield the captured state.
        fresh = WorldIR(id="w2", name="n", main_branch_id="b2")
        store.restore(snap.snapshot_id, fresh)
        assert fresh.entities["e1"].name == "box"
        assert set(fresh.entities) == {"e1"}

    def test_deterministic_snapshot_ids(self):
        store = SnapshotStore()
        world = _small_world()
        s1 = store.capture(world, tick=5, timestamp=0.0)
        s2 = store.capture(world, tick=5, timestamp=0.0)
        assert s2.snapshot_id != s1.snapshot_id  # ordinal disambiguates
        assert s2.snapshot_id.endswith("-0002")

    def test_list_filters_by_branch(self):
        store = SnapshotStore()
        world = _small_world()
        store.capture(world, tick=1, timestamp=0.0, branch_id="main")
        store.capture(world, tick=2, timestamp=0.0, branch_id="scen-a")
        assert [s.branch_id for s in store.list("scen-a")] == ["scen-a"]
        assert len(store.list()) == 2


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------


class TestBranches:
    def test_branch_inherits_parent_state(self):
        mgr = BranchManager()
        parent = _small_world()
        child, record = mgr.create_branch(parent, "scen-a", "main", tick=5,
                                          description="wind scenario")
        assert record.parent_branch_id == "main"
        assert set(child.entities) == set(parent.entities)
        assert child.id == "world-scen-a"

    def test_child_mutation_never_reaches_parent(self):
        mgr = BranchManager()
        parent = _small_world()
        child, _ = mgr.create_branch(parent, "scen-a", "main", tick=5)
        child.entities["e1"].confidence = 0.05
        child.entities["e-added"] = Entity(id="e-added", type=EntityType.DEBRIS, name="x")
        assert parent.entities["e1"].confidence == 0.8
        assert "e-added" not in parent.entities

    def test_branch_registry_semantics(self):
        mgr = BranchManager()
        parent = _small_world()
        mgr.create_branch(parent, "scen-a", "main", tick=5)
        with pytest.raises(ValueError):
            mgr.create_branch(parent, "scen-a", "main", tick=6)  # duplicate
        with pytest.raises(KeyError):
            mgr.create_branch(parent, "scen-b", "no-such-branch", tick=6)
        mgr.delete_branch("scen-a")
        with pytest.raises(KeyError):
            mgr.get("scen-a")
        with pytest.raises(ValueError):
            mgr.delete_branch("main")  # baseline protected

    def test_compare_reports_added_removed_changed(self):
        mgr = BranchManager()
        base = _small_world()
        scenario, _ = mgr.create_branch(base, "scen-a", "main", tick=5)
        scenario.entities["e-debris"] = Entity(
            id="e-debris", type=EntityType.DEBRIS, name="fallen"
        )
        base.entities.pop("e1")  # e1 survives only in the child: "added" there

        diff = mgr.compare(base, scenario, "main", "scen-a")
        assert diff["added"] == ["e-debris", "e1"]  # sorted
        assert diff["removed"] == []

        # Pure-modification comparison on a clean pair:
        base2 = _small_world()
        scen2, _ = mgr.create_branch(base2, "scen-b", "main", tick=5)
        scen2.entities["e1"].confidence = 0.4
        diff2 = mgr.compare(base2, scen2, "main", "scen-b")
        assert diff2["added"] == [] and diff2["removed"] == []
        assert len(diff2["changed"]) == 1
        assert diff2["changed"][0]["fields"]["confidence"] == (0.8, 0.4)

        # And a removal-only comparison:
        base3 = _small_world()
        scen3, _ = mgr.create_branch(base3, "scen-c", "main", tick=5)
        scen3.entities.pop("e1")
        diff3 = mgr.compare(base3, scen3, "main", "scen-c")
        assert diff3["removed"] == ["e1"] and diff3["added"] == []


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class TestReplay:
    def test_replay_is_deterministic(self):
        def add_box(w):
            w.entities["e-added"] = Entity(
                id="e-added", type=EntityType.STRUCTURE, name="added",
                geometry_ids=["g-added"], confidence=0.9,
            )
            w.geometries["g-added"] = Geometry(
                id="g-added", type=GeometryType.BOX,
                bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(1, 1, 1),
            )

        base = _small_world()
        r1, b1 = replay(base, [add_box], seed=42)
        r2, b2 = replay(base, [add_box], seed=42)
        assert set(r1.entities) == set(r2.entities) == {"e1", "e-added"}
        assert r1.to_dict() == r2.to_dict()
        assert b1.to_list() == b2.to_list()

    def test_replay_is_non_destructive(self):
        def add_box(w):
            w.entities["e-added"] = Entity(id="e-added", type=EntityType.DEBRIS, name="x")

        base = _small_world()
        bus = EventBus(seed=1)
        bus.emit(CONTACT, timestamp=0.0, tick=0, source_refs=("a",), target_refs=("b",))
        result, _ = replay(base, [add_box], seed=1)
        # The caller's world and log are untouched.
        assert "e-added" not in base.entities
        assert len(bus.events) == 1

    def test_replay_commands_run_in_order(self):
        calls = []

        def step1(w):
            calls.append(1)
            w.entities["s1"] = Entity(id="s1", type=EntityType.DEBRIS, name="1")

        def step2(w):
            calls.append(2)
            w.entities["s2"] = Entity(id="s2", type=EntityType.DEBRIS, name="2")

        result, _ = replay(_small_world(), [step1, step2], seed=0)
        assert calls == [1, 2]
        assert set(result.entities) == {"e1", "s1", "s2"}
