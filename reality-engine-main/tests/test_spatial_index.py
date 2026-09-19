"""Tests for the geometric spatial index/query engine
(engine/scene_graph/spatial_index.py)."""

from __future__ import annotations

import math

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR

from engine.scene_graph.spatial_index import SpatialIndex


def _positioned(entity_id, x, y, z, entity_type=EntityType.DEBRIS, **kwargs):
    return Entity(id=entity_id, type=entity_type, transform={"position": {"x": x, "y": y, "z": z}}, **kwargs)


def _world_with(*entities, geometries=()) -> WorldIR:
    world = WorldIR()
    for e in entities:
        world.entities[e.id] = e
    for g in geometries:
        world.geometries[g.id] = g
    return world


def test_position_resolved_from_transform():
    world = _world_with(_positioned("ent-1", 1.0, 2.0, 3.0))
    index = SpatialIndex(world)
    assert len(index) == 1
    assert index.unlocalized == ()


def test_position_resolved_from_geometry_bounds_centroid_when_no_transform():
    geom = Geometry(id="geom-1", type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(2, 2, 2))
    entity = Entity(id="ent-1", geometry_ids=[geom.id])
    world = _world_with(entity, geometries=[geom])

    index = SpatialIndex(world)
    assert len(index) == 1
    nearest = index.nearest((0.0, 0.0, 0.0), k=1)
    entity_found, distance = nearest[0]
    assert entity_found.id == "ent-1"
    assert math.isclose(distance, 3 ** 0.5)  # centroid (1,1,1)


def test_entity_with_no_position_source_is_unlocalized_not_fabricated():
    entity = Entity(id="ent-bare")
    world = _world_with(entity)

    index = SpatialIndex(world)
    assert len(index) == 0
    assert len(index.unlocalized) == 1
    assert index.unlocalized[0].entity_id == "ent-bare"
    assert "no transform.position" in index.unlocalized[0].reason


def test_nearest_returns_closest_first_deterministically():
    world = _world_with(
        _positioned("far", 10.0, 0.0, 0.0),
        _positioned("near", 1.0, 0.0, 0.0),
        _positioned("mid", 5.0, 0.0, 0.0),
    )
    index = SpatialIndex(world)
    results = index.nearest((0.0, 0.0, 0.0), k=2)
    assert [e.id for e, _ in results] == ["near", "mid"]
    assert results[0][1] == 1.0
    assert results[1][1] == 5.0


def test_nearest_tie_breaks_by_entity_id():
    world = _world_with(
        _positioned("b-entity", 1.0, 0.0, 0.0),
        _positioned("a-entity", -1.0, 0.0, 0.0),  # same distance from origin
    )
    index = SpatialIndex(world)
    results = index.nearest((0.0, 0.0, 0.0), k=2)
    assert [e.id for e, _ in results] == ["a-entity", "b-entity"]


def test_nearest_respects_predicate():
    world = _world_with(
        _positioned("wall-1", 1.0, 0.0, 0.0, entity_type=EntityType.WALL),
        _positioned("debris-1", 2.0, 0.0, 0.0, entity_type=EntityType.DEBRIS),
    )
    index = SpatialIndex(world)
    results = index.nearest((0.0, 0.0, 0.0), k=5, predicate=lambda e: e.type == EntityType.DEBRIS)
    assert [e.id for e, _ in results] == ["debris-1"]


def test_within_radius_excludes_entities_outside_radius():
    world = _world_with(
        _positioned("close", 1.0, 0.0, 0.0),
        _positioned("far", 100.0, 0.0, 0.0),
    )
    index = SpatialIndex(world)
    results = index.within_radius((0.0, 0.0, 0.0), radius=5.0)
    assert [e.id for e, _ in results] == ["close"]


def test_within_radius_boundary_is_inclusive():
    world = _world_with(_positioned("edge", 5.0, 0.0, 0.0))
    index = SpatialIndex(world)
    results = index.within_radius((0.0, 0.0, 0.0), radius=5.0)
    assert len(results) == 1


def test_within_radius_rejects_negative_radius():
    world = _world_with(_positioned("ent-1", 0.0, 0.0, 0.0))
    index = SpatialIndex(world)
    try:
        index.within_radius((0.0, 0.0, 0.0), radius=-1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_within_region_finds_entity_whose_position_is_inside():
    world = _world_with(_positioned("inside", 1.0, 1.0, 1.0), _positioned("outside", 10.0, 10.0, 10.0))
    index = SpatialIndex(world)
    results = index.within_region((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    assert [e.id for e in results] == ["inside"]


def test_within_region_finds_entity_whose_aabb_overlaps_even_if_centroid_outside():
    # A wall spanning x=[-5, 5] centered at x=0 -- centroid is outside a
    # region that only covers x=[1, 10], but the wall's AABB overlaps it.
    geom = Geometry(id="geom-wall", type=GeometryType.PLANE, bounds_min=Vector3(-5, 0, 0), bounds_max=Vector3(5, 2, 0.2))
    wall = Entity(id="ent-wall", type=EntityType.WALL, geometry_ids=[geom.id],
                  transform={"position": {"x": 0.0, "y": 1.0, "z": 0.1}})
    world = _world_with(wall, geometries=[geom])

    index = SpatialIndex(world)
    results = index.within_region((1.0, 0.0, 0.0), (10.0, 2.0, 0.2))
    assert [e.id for e in results] == ["ent-wall"]


def test_spatial_index_does_not_observe_world_mutations_after_build():
    world = _world_with(_positioned("ent-1", 0.0, 0.0, 0.0))
    index = SpatialIndex(world)
    world.entities["ent-2"] = _positioned("ent-2", 1.0, 0.0, 0.0)

    assert len(index) == 1  # snapshot, same contract as SceneGraph


def test_len_and_empty_world():
    index = SpatialIndex(WorldIR())
    assert len(index) == 0
    assert index.nearest((0.0, 0.0, 0.0)) == []
    assert index.within_radius((0.0, 0.0, 0.0), radius=10.0) == []
    assert index.within_region((-1, -1, -1), (1, 1, 1)) == []
