"""Tests for engine/geometry/adjacency.py -- AABB overlap/adjacency inference."""
from world_ir import Entity, RelationshipKind, WorldIR


def make_entity(entity_id: str, x: float, y: float, z: float, radius: float = 0.0) -> Entity:
    return Entity(id=entity_id, transform={"position": {"x": x, "y": y, "z": z}, "radius": radius})


def make_world(*entities: Entity) -> WorldIR:
    world = WorldIR(id="w1")
    for e in entities:
        world.entities[e.id] = e
    return world


from engine.geometry.adjacency import aabb_overlap, compute_aabb, infer_geometric_relationships


def test_aabb_overlap_touching_boundary_counts_as_overlap():
    # a: [0,1]^3, b: [1,2]^3 -- share the x=1 plane exactly.
    assert aabb_overlap((0, 0, 0), (1, 1, 1), (1, 0, 0), (2, 1, 1)) is True


def test_aabb_overlap_clearly_separated():
    assert aabb_overlap((0, 0, 0), (1, 1, 1), (5, 5, 5), (6, 6, 6)) is False


def test_aabb_overlap_one_box_inside_another():
    outer_min, outer_max = (0, 0, 0), (10, 10, 10)
    inner_min, inner_max = (2, 2, 2), (3, 3, 3)
    assert aabb_overlap(outer_min, outer_max, inner_min, inner_max) is True


def test_two_entities_overlap():
    a = make_entity("a", 0, 0, 0, radius=2.0)
    b = make_entity("b", 1, 0, 0, radius=2.0)
    world = make_world(a, b)
    result = infer_geometric_relationships(world)
    assert result == [("a", "b", RelationshipKind.OVERLAPS)]


def test_two_entities_not_touching_even_with_margin():
    a = make_entity("a", 0, 0, 0, radius=1.0)
    b = make_entity("b", 100, 0, 0, radius=1.0)
    result = infer_geometric_relationships(make_world(a, b), adjacency_margin=2.0)
    assert result == []


def test_adjacency_requires_nonzero_margin():
    # Unexpanded AABBs: a=[-1,1], b=[3,5] on x -- gap of 2 units.
    a = make_entity("a", 0, 0, 0, radius=1.0)
    b = make_entity("b", 4, 0, 0, radius=1.0)
    world = make_world(a, b)

    assert infer_geometric_relationships(world, adjacency_margin=0.0) == []

    result = infer_geometric_relationships(world, adjacency_margin=1.0)
    assert result == [("a", "b", RelationshipKind.ADJACENT_TO)]


def test_entity_without_transform_is_skipped_not_defaulted_to_origin():
    no_transform = Entity(id="floating", transform=None)
    at_origin = make_entity("origin", 0, 0, 0, radius=5.0)
    world = make_world(no_transform, at_origin)
    assert infer_geometric_relationships(world) == []


def test_three_entities_exact_pairs_and_kinds_no_duplicates():
    # a,b overlap; b,c overlap; a,c do not touch even with margin.
    a = make_entity("a", 0, 0, 0, radius=1.0)
    b = make_entity("b", 1.5, 0, 0, radius=1.0)
    c = make_entity("c", 3.0, 0, 0, radius=1.0)
    world = make_world(a, b, c)

    result = infer_geometric_relationships(world, adjacency_margin=0.0)
    assert set(result) == {
        ("a", "b", RelationshipKind.OVERLAPS),
        ("b", "c", RelationshipKind.OVERLAPS),
    }
    # each unordered pair appears exactly once
    pairs = [(r[0], r[1]) for r in result]
    assert len(pairs) == len(set(pairs))
    assert ("c", "a") not in [(r[0], r[1]) for r in result]
    assert ("a", "c") not in [(r[0], r[1]) for r in result]


def test_compute_aabb():
    min_c, max_c = compute_aabb({"x": 1, "y": 2, "z": 3}, 0.5)
    assert min_c == (0.5, 1.5, 2.5)
    assert max_c == (1.5, 2.5, 3.5)
