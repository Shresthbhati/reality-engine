"""Tests for engine/scene_graph/spatial_tiling.py (P13-01 spatial cells).

Acceptance criterion from docs/future/large-world/LARGE_WORLD.md: "global
queries (bbox) return the union correctly" regardless of chunking -- i.e.
SpatialTiling.query_region must agree with SpatialIndex.within_region
(the flat-scan reference implementation) for the same region.
"""

from __future__ import annotations

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR

from engine.scene_graph.spatial_index import SpatialIndex
from engine.scene_graph.spatial_tiling import SpatialTiling


def _positioned(entity_id, x, y, z, entity_type=EntityType.DEBRIS, **kwargs):
    return Entity(id=entity_id, type=entity_type, transform={"position": {"x": x, "y": y, "z": z}}, **kwargs)


def _world_with(*entities, geometries=()) -> WorldIR:
    world = WorldIR()
    for e in entities:
        world.entities[e.id] = e
    for g in geometries:
        world.geometries[g.id] = g
    return world


def test_chunk_size_must_be_positive():
    world = _world_with(_positioned("e1", 0, 0, 0))
    try:
        SpatialTiling(world, chunk_size=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_entities_bucketed_into_distinct_chunks():
    world = _world_with(
        _positioned("near", 1.0, 1.0, 1.0),
        _positioned("far", 500.0, 500.0, 500.0),
    )
    tiling = SpatialTiling(world, chunk_size=10.0)
    assert len(tiling) == 2
    assert tiling.chunk_of("near") != tiling.chunk_of("far")
    assert tiling.chunk_count == 2


def test_unlocalized_entity_excluded():
    world = _world_with(Entity(id="ghost"))  # no transform, no geometry bounds
    tiling = SpatialTiling(world, chunk_size=10.0)
    assert len(tiling) == 0
    assert tiling.chunk_of("ghost") is None


def test_query_region_matches_flat_scan_reference():
    world = _world_with(
        _positioned("a", 0.0, 0.0, 0.0),
        _positioned("b", 5.0, 5.0, 5.0),
        _positioned("c", 50.0, 50.0, 50.0),
        _positioned("d", -20.0, -20.0, -20.0),
    )
    region_min, region_max = (-1.0, -1.0, -1.0), (10.0, 10.0, 10.0)

    flat = SpatialIndex(world).within_region(region_min, region_max)
    tiled = SpatialTiling(world, chunk_size=3.0).query_region(region_min, region_max)

    assert [e.id for e in flat] == [e.id for e in tiled] == ["a", "b"]


def test_query_region_includes_entity_whose_geometry_overlaps_but_centroid_is_outside():
    # A wall spanning a chunk boundary: centroid outside the query region,
    # but its geometry AABB overlaps it -- must still be returned (same
    # rule as SpatialIndex.within_region).
    geom = Geometry(id="g1", type=GeometryType.BOX, bounds_min=Vector3(-5, 0, 0), bounds_max=Vector3(15, 1, 1))
    wall = Entity(id="wall", geometry_ids=[geom.id])  # centroid at x=5
    world = _world_with(wall, geometries=[geom])

    tiling = SpatialTiling(world, chunk_size=2.0)
    # geom AABB is x in [-5, 15]; region x in [-4, -2] overlaps the AABB
    # near its left edge but does not contain the centroid (x=5).
    hits = tiling.query_region((-4.0, -1.0, -1.0), (-2.0, 1.0, 1.0))
    assert [e.id for e in hits] == ["wall"]


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
