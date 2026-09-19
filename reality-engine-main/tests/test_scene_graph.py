"""Tests for the Scene Graph query engine (spec sec 22/59)."""

from engine.scene_graph import SceneGraph
from world_ir import Entity, Relationship, RelationshipKind, WorldIR


def _world():
    """Room contains a chair (CONTAINS) and a window is PART_OF the wall,
    which RESTS_ON the foundation. One entity carries an UNKNOWN edge."""
    world = WorldIR(id="w1")
    room = Entity(id="room", name="Room 4", relationships=[
        Relationship(kind=RelationshipKind.CONTAINS, target_id="chair"),
    ])
    chair = Entity(id="chair", name="Chair")
    wall = Entity(id="wall", name="Wall", relationships=[
        Relationship(kind=RelationshipKind.RESTS_ON, target_id="foundation"),
    ])
    window = Entity(id="window", name="Window", relationships=[
        Relationship(kind=RelationshipKind.PART_OF, target_id="wall"),
    ])
    foundation = Entity(id="foundation", name="Foundation")
    mystery = Entity(id="mystery", name="Mystery", relationships=[
        Relationship(kind=RelationshipKind.UNKNOWN, target_id="wall"),
    ])
    world.entities = {
        "room": room, "chair": chair, "wall": wall,
        "window": window, "foundation": foundation, "mystery": mystery,
    }
    return world


def test_edges_from_and_to():
    graph = SceneGraph(_world())
    assert [e.target_id for e in graph.edges_from("room")] == ["chair"]
    assert {e.source_id for e in graph.edges_to("wall")} == {"window", "mystery"}


def test_edges_filtered_by_kind():
    graph = SceneGraph(_world())
    assert graph.edges_from("wall", RelationshipKind.RESTS_ON)
    assert not graph.edges_from("wall", RelationshipKind.PART_OF)


def test_contents_of_finds_contains_and_reverse_part_of():
    graph = SceneGraph(_world())
    contents = graph.contents_of("room")
    assert {e.id for e in contents} == {"chair"}

    contents = graph.contents_of("wall")
    assert {e.id for e in contents} == {"window"}


def test_container_of_is_inverse_of_contents_of():
    graph = SceneGraph(_world())
    assert graph.container_of("window").id == "wall"
    assert graph.container_of("chair").id == "room"
    assert graph.container_of("room") is None


def test_supporters_of_generalizes_inspector_find_supporting():
    graph = SceneGraph(_world())
    supporters = graph.supporters_of("foundation")
    assert {e.id for e in supporters} == {"wall"}


def test_unknown_relationship_entities():
    graph = SceneGraph(_world())
    assert {e.id for e in graph.unknown_relationship_entities()} == {"mystery"}


def test_query_by_kind_across_whole_world():
    graph = SceneGraph(_world())
    part_of_edges = graph.query_by_kind(RelationshipKind.PART_OF)
    assert [(e.source_id, e.target_id) for e in part_of_edges] == [("window", "wall")]


def test_path_exists_follows_multi_hop_chain():
    graph = SceneGraph(_world())
    # window -PART_OF-> wall -RESTS_ON-> foundation
    assert graph.path_exists("window", "foundation")
    assert not graph.path_exists("foundation", "window")  # edges are directed


def test_path_exists_respects_kind_filter():
    graph = SceneGraph(_world())
    assert graph.path_exists("window", "foundation", kinds=[RelationshipKind.PART_OF, RelationshipKind.RESTS_ON])
    assert not graph.path_exists("window", "foundation", kinds=[RelationshipKind.PART_OF])


def test_path_exists_same_entity_is_trivially_true():
    graph = SceneGraph(_world())
    assert graph.path_exists("wall", "wall")


def test_path_exists_false_for_unknown_entity():
    graph = SceneGraph(_world())
    assert not graph.path_exists("wall", "does-not-exist")


def test_all_edges_returns_every_relationship_in_the_world():
    graph = SceneGraph(_world())
    edges = graph.all_edges()
    assert len(edges) == 4  # room->chair, wall->foundation, window->wall, mystery->wall
