"""Tests for the Studio foundation: Outliner, Selection, StudioSession (Goal G)."""

import pytest

from engine.commands import CommandNotFoundError, PermissionDeniedError, SetEntityTransformCommand
from engine.studio import Outliner, Selection, StudioSession
from events.types import ENTITY_CREATED_EVENT, ENTITY_DELETED_EVENT, ENTITY_TRANSFORM_SET_EVENT
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


# ---- StudioSession editing: wired through the command pipeline ----

def test_create_entity_goes_through_command_pipeline():
    world = _world_with_entities()
    session = StudioSession(world, actor_id="user-1")

    result = session.create_entity("new_wall", "structure", name="New Wall")

    assert "new_wall" in world.entities
    assert world.entities["new_wall"].provenance == Provenance.GENERATED
    events = session.event_bus.events_of_type(ENTITY_CREATED_EVENT)
    assert len(events) == 1
    assert events[0].actor_id == "user-1"
    assert events[0].event_id == result.event_id


def test_move_selected_moves_the_active_selection():
    world = _world_with_entities()
    session = StudioSession(world)
    session.select("wall")

    session.move_selected((1.0, 2.0, 3.0))

    assert world.entities["wall"].transform == {"position": {"x": 1.0, "y": 2.0, "z": 3.0}}
    assert len(session.event_bus.events_of_type(ENTITY_TRANSFORM_SET_EVENT)) == 1


def test_move_selected_raises_when_nothing_selected():
    world = _world_with_entities()
    session = StudioSession(world)
    with pytest.raises(ValueError):
        session.move_selected((0.0, 0.0, 0.0))


def test_delete_selected_removes_entity_and_clears_selection():
    world = _world_with_entities()
    session = StudioSession(world)
    session.select("tree")

    session.delete_selected()

    assert "tree" not in world.entities
    assert session.selection.active is None
    assert len(session.event_bus.events_of_type(ENTITY_DELETED_EVENT)) == 1


def test_delete_selected_raises_when_nothing_selected():
    world = _world_with_entities()
    session = StudioSession(world)
    with pytest.raises(ValueError):
        session.delete_selected()


def test_studio_editing_respects_permission_policy():
    class DenyAll:
        def check(self, command, world):
            raise PermissionDeniedError("denied")

    world = _world_with_entities()
    session = StudioSession(world, permission_policy=DenyAll())
    session.select("wall")

    with pytest.raises(PermissionDeniedError):
        session.move_selected((1.0, 1.0, 1.0))
    assert world.entities["wall"].transform is None


def test_studio_editing_surfaces_validation_errors():
    world = _world_with_entities()
    session = StudioSession(world)
    with pytest.raises(CommandNotFoundError):
        session.commands.execute(SetEntityTransformCommand(entity_id="does-not-exist", position=(0, 0, 0)))
