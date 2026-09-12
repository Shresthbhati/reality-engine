"""Tests for the Studio foundation: Outliner, Selection, StudioSession (Goal G)."""

import pytest

from engine.studio import Outliner, Selection, StudioSession
from provenance import Provenance
from world_ir import Entity, EntityType, WorldIR


def _world_with_entities():
    world = WorldIR(id="w1")
    building = Entity(id="building", name="Building", type=EntityType.BUILDING,
                       component_ids=["wall"], provenance=Provenance.RECONSTRUCTED, confidence=0.9)
    wall = Entity(id="wall", name="Wall", type=EntityType.STRUCTURE,
                  provenance=Provenance.RECONSTRUCTED, confidence=0.8)
    tree = Entity(id="tree", name="Oak Tree", type=EntityType.VEGETATION,
                  provenance=Provenance.INFERRED, confidence=0.4)
    world.entities = {"building": building, "wall": wall, "tree": tree}
    return world


# ---- Outliner ----

def test_outliner_hierarchy_nests_components_under_parent():
    world = _world_with_entities()
    outliner = Outliner(world)
    roots = outliner.hierarchy()

    root_ids = {n.entity_id for n in roots}
    assert root_ids == {"building", "tree"}  # wall is a component of building, not a root

    building_node = next(n for n in roots if n.entity_id == "building")
    assert [c.entity_id for c in building_node.children] == ["wall"]


def test_outliner_hierarchy_guards_against_cycles():
    world = WorldIR(id="w1")
    a = Entity(id="a", name="A", component_ids=["b"])
    b = Entity(id="b", name="B", component_ids=["a"])  # mutual cycle
    world.entities = {"a": a, "b": b}
    outliner = Outliner(world)

    # Each is referenced as the other's component, so neither is a root --
    # the call must simply terminate (no recursion into either), not crash.
    roots = outliner.hierarchy()
    assert roots == []


def test_outliner_grouped_by_type():
    world = _world_with_entities()
    groups = Outliner(world).grouped_by_type()
    assert [n.entity_id for n in groups["building"]] == ["building"]
    assert [n.entity_id for n in groups["structure"]] == ["wall"]
    assert [n.entity_id for n in groups["vegetation"]] == ["tree"]


def test_outliner_find_by_name_case_insensitive():
    world = _world_with_entities()
    results = Outliner(world).find_by_name("oak")
    assert [e.id for e in results] == ["tree"]


# ---- Selection ----

def test_selection_replace_vs_additive():
    sel = Selection()
    sel.select("a")
    sel.select("b")
    assert sel.ids == ["b"]  # plain select replaces

    sel.select("c", additive=True)
    assert sel.ids == ["b", "c"]
    assert sel.active == "c"


def test_selection_toggle_and_deselect():
    sel = Selection()
    sel.toggle("a")
    assert sel.is_selected("a")
    sel.toggle("a")
    assert not sel.is_selected("a")

    sel.select("x", additive=True)
    sel.select("y", additive=True)
    sel.deselect("x")
    assert sel.ids == ["y"]


def test_selection_additive_reorders_existing_id_to_end():
    sel = Selection()
    sel.select("a", additive=True)
    sel.select("b", additive=True)
    sel.select("a", additive=True)  # re-select -> becomes active again
    assert sel.ids == ["b", "a"]
    assert sel.active == "a"


def test_selection_clear():
    sel = Selection()
    sel.select("a", additive=True)
    sel.clear()
    assert sel.ids == []
    assert sel.active is None


# ---- StudioSession ----

def test_studio_session_select_raises_for_unknown_entity():
    world = _world_with_entities()
    session = StudioSession(world)
    with pytest.raises(KeyError):
        session.select("does-not-exist")


def test_studio_session_active_entity_summary_resolves_selection():
    world = _world_with_entities()
    session = StudioSession(world)
    session.select("wall")
    summary = session.active_entity_summary()
    assert summary.entity.id == "wall"


def test_studio_session_active_entity_summary_none_without_selection():
    world = _world_with_entities()
    session = StudioSession(world)
    assert session.active_entity_summary() is None


def test_studio_session_provenance_panel_aggregates_inspector_data():
    world = _world_with_entities()
    session = StudioSession(world)
    panel = session.provenance_panel("tree")

    assert panel["provenance"] == "INFERRED"
    assert panel["confidence"] == 0.4
    assert panel["is_canonical"] is True  # INFERRED is canonical; only GENERATED/UNKNOWN/CONFLICT aren't
    assert panel["observations"] == []
    assert panel["materials"] == []


def test_studio_session_provenance_panel_flags_conflict_as_non_canonical():
    world = _world_with_entities()
    world.entities["wall"].provenance = Provenance.CONFLICT
    session = StudioSession(world)
    panel = session.provenance_panel("wall")
    assert panel["is_canonical"] is False


def test_studio_session_provenance_panel_none_for_missing_entity():
    world = _world_with_entities()
    session = StudioSession(world)
    assert session.provenance_panel("nope") is None


def test_studio_session_visible_entities_omits_entities_without_transform():
    world = _world_with_entities()
    world.entities["wall"].transform = {"position": {"x": 0.0, "y": 5.0, "z": 0.0}}
    session = StudioSession(world)
    visible = session.visible_entities()
    assert [v["entity_id"] for v in visible] == ["wall"]
