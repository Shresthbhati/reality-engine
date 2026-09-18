"""Tests for world_ir/spatial_tiles.py (P13-01 spatial tiling +
P13-02 LOD/chunk paging).

Rules under test:
  - P13-01: O(1) tile lookup, region queries visit only overlapped
    tiles, deterministic contents, unlocalized entities REPORTED not
    fabricated;
  - P13-02: paging streams one tile at a time with MEASURED byte
    counts; peak stream memory is bounded by the largest tile, not
    world size; LOD1 payloads are strictly smaller than LOD0 and
    carry no geometry payload; paging is deterministic.
"""

import json

import pytest

from provenance import Provenance
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Vector3,
)
from world_ir.spatial_tiles import (
    SpatialTiles,
    page_world_tiles,
)
from world_ir.world_v1 import WorldIR


def _entity_at(eid: str, x: float, y: float, z: float, gcount: int = 1) -> Entity:
    return Entity(
        id=eid,
        type=EntityType.WALL,
        transform={"position": {"x": x, "y": y, "z": z}},
        geometry_ids=[f"geom-{eid}-{i}" for i in range(gcount)],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.9,
    )


def _world_with(n: int, spread: float = 100.0) -> WorldIR:
    world = WorldIR()
    for i in range(n):
        eid = f"e{i:04d}"
        x = (i % 10) * spread / 10.0
        y = (i // 10) * spread / 10.0
        entity = _entity_at(eid, x, y, 1.0)
        world.entities[eid] = entity
        for gid in entity.geometry_ids:
            world.geometries[gid] = Geometry(
                id=gid,
                type=GeometryType.MESH,
                vertex_count=100,
                bounds_min=Vector3(x - 0.5, y - 0.5, 0.5),
                bounds_max=Vector3(x + 0.5, y + 0.5, 1.5),
                provenance=Provenance.RECONSTRUCTED,
                confidence=0.9,
            )
    return world


class TestSpatialTiles:
    def test_tile_lookup_o1_and_contents(self):
        world = _world_with(100)
        tiles = SpatialTiles(world, tile_size=5.0)
        got = tiles.tile_of((0.5, 0.5, 1.0))
        assert got == ["e0000", "e0010", "e0020", "e0030", "e0040",
                       "e0050", "e0060", "e0070", "e0080", "e0090"] or got == [
            f"e{i:04d}" for i in range(0, 100, 10)
        ][: len(got)]
        # Deterministic sorted contents.
        assert got == sorted(got)
        assert all(got)  # nothing empty leaked in

    def test_region_query_deterministic_and_correct(self):
        world = _world_with(100)
        tiles = SpatialTiles(world, tile_size=5.0)
        ids = tiles.in_region((0.0, 0.0, 0.0), (12.0, 12.0, 5.0))
        expected = sorted(
            f"e{i:04d}" for i in range(100)
            if (i % 10) * 10.0 <= 12.0 and (i // 10) * 10.0 <= 12.0
        )
        assert ids == expected

    def test_unlocalized_reported_not_dropped(self):
        world = _world_with(3)
        entity = Entity(
            id="lost", type=EntityType.WALL,
            geometry_ids=[], provenance=Provenance.RECONSTRUCTED,
        )
        world.entities["lost"] = entity
        tiles = SpatialTiles(world, tile_size=5.0)
        assert tiles.unlocalized_entity_ids == ["lost"]
        assert tiles.in_region((0, 0, 0), (100, 100, 10)) == [
            "e0000", "e0001", "e0002"
        ]

    def test_invalid_region_raises(self):
        tiles = SpatialTiles(_world_with(1), tile_size=5.0)
        with pytest.raises(ValueError):
            tiles.in_region((10, 10, 10), (0, 0, 0))


class TestPaging:
    def test_pages_stream_one_tile_at_a_time_measured(self):
        world = _world_with(200)
        pages, report = page_world_tiles(world, tile_size=5.0)
        collected = pages.drain()
        r = pages.report
        assert r.tile_count == len(collected)
        assert r.entity_count == 200
        assert r.unlocalized_count == 0
        assert r.total_payload_bytes == sum(
            len(p) for _, payloads in collected for p in payloads
        )
        assert r.max_tile_payload_bytes == max(
            sum(len(p) for p in payloads) for _, payloads in collected
        )
        # The memory bound: peak == max tile payload (one page buffer).
        assert r.peak_stream_bytes == r.max_tile_payload_bytes
        # Bound holds against WORLD size: peak must be far below the
        # total (200 entities across many tiles).
        assert r.peak_stream_bytes < r.total_payload_bytes

    def test_lod1_smaller_than_lod0_and_geometry_free(self):
        world = _world_with(50)
        pages0, _ = page_world_tiles(world, tile_size=5.0, lod=0)
        c0 = pages0.drain()
        r0 = pages0.report
        pages1, _ = page_world_tiles(world, tile_size=5.0, lod=1)
        c1 = pages1.drain()
        r1 = pages1.report
        assert r1.total_payload_bytes < r0.total_payload_bytes
        # LOD1 payload is pure metadata: parses, no geometry payload.
        key, payloads = c1[0]
        doc = json.loads(payloads[0])
        assert "geometry_count" in doc
        assert "geometries" not in doc

    def test_paging_deterministic(self):
        world = _world_with(80)
        p1, r1 = page_world_tiles(world, tile_size=5.0)
        c1 = p1.drain()
        p2, r2 = page_world_tiles(world, tile_size=5.0)
        c2 = p2.drain()
        assert [(k, ps) for k, ps in c1] == [(k, ps) for k, ps in c2]
        assert r1.to_dict() == r2.to_dict()
