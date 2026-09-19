"""Tile membership boundary semantics (world_ir/spatial_tiles.py's
SpatialTiles) -- P0 audit item: entities at tile centers, boundaries,
corners, negative and very large coordinates must map to deterministic,
documented tile keys, never nondeterministically depending on tiny
floating-point noise.

DOCUMENTED SEMANTICS (existing code, not invented here):
SpatialTiles._key_of(position) = (floor(x / tile_size), floor(y / tile_size),
floor(z / tile_size)). This makes each tile a HALF-OPEN interval on every
axis: [tile_index * tile_size, (tile_index + 1) * tile_size). A position
exactly AT a tile's lower boundary belongs to that tile; a position
exactly at the upper boundary belongs to the NEXT tile. This is standard,
well-defined floor-based spatial hashing -- these tests pin that
convention down so it can never silently change.
"""

from __future__ import annotations

import math

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType
from world_ir.spatial_tiles import SpatialTiles
from world_ir.world_v1 import WorldIR


def _entity(eid, x, y, z) -> Entity:
    return Entity(
        id=eid, type=EntityType.STRUCTURE,
        transform={"position": {"x": x, "y": y, "z": z}},
        provenance=Provenance.RECONSTRUCTED,
    )


def _world_with(*entities) -> WorldIR:
    w = WorldIR()
    for e in entities:
        w.entities[e.id] = e
    return w


class TestTileCenterAndBoundary:
    def test_tile_center_is_unambiguous(self):
        world = _world_with(_entity("e", 5.0, 5.0, 5.0))
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_of((5.0, 5.0, 5.0)) == ["e"]

    def test_lower_boundary_belongs_to_this_tile(self):
        # x=10.0 is the LOWER boundary of tile 1, not the upper boundary
        # of tile 0 -- floor(10.0 / 10.0) == 1.
        world = _world_with(_entity("e", 10.0, 0.0, 0.0))
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_ids() == [(1, 0, 0)]

    def test_just_below_boundary_belongs_to_previous_tile(self):
        world = _world_with(_entity("e", 9.999999, 0.0, 0.0))
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_ids() == [(0, 0, 0)]

    def test_exact_zero_is_tile_zero_not_negative(self):
        world = _world_with(_entity("e", 0.0, 0.0, 0.0))
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_ids() == [(0, 0, 0)]


class TestCornerAndNegativeCoordinates:
    def test_corner_of_a_tile(self):
        # The far corner of tile (1,1,1) (just inside it on every axis).
        world = _world_with(_entity("corner", 19.999, 19.999, 19.999))
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_ids() == [(1, 1, 1)]

    def test_negative_coordinates_use_floor_not_truncation(self):
        # floor(-0.1 / 10) == -1, NOT 0 -- truncation-toward-zero would
        # give the wrong (and inconsistent-with-positive-side) answer.
        world = _world_with(_entity("e", -0.1, -0.1, -0.1))
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_ids() == [(-1, -1, -1)]

    def test_negative_boundary_belongs_to_upper_tile(self):
        # x=-10.0 is the lower boundary of tile -1: floor(-10.0/10.0) == -1.
        world = _world_with(_entity("e", -10.0, 0.0, 0.0))
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_ids() == [(-1, 0, 0)]

    def test_negative_and_positive_entities_do_not_collide(self):
        world = _world_with(
            _entity("neg", -5.0, -5.0, -5.0),
            _entity("pos", 5.0, 5.0, 5.0),
        )
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_of((-5.0, -5.0, -5.0)) == ["neg"]
        assert tiles.tile_of((5.0, 5.0, 5.0)) == ["pos"]


class TestVeryLargeCoordinates:
    def test_city_scale_coordinates_do_not_lose_precision(self):
        # City-scale UTM-like coordinates (hundreds of kilometers):
        # floor-based bucketing must still be exact at this magnitude.
        big_x = 500_000.0
        world = _world_with(_entity("e", big_x, 0.0, 0.0))
        tiles = SpatialTiles(world, tile_size=10.0)
        expected = math.floor(big_x / 10.0)
        assert tiles.tile_ids() == [(expected, 0, 0)]

    def test_very_large_coordinates_still_deterministic_across_runs(self):
        big_x = 12_345_678.9
        world = _world_with(_entity("e", big_x, 0.0, 0.0))
        r1 = SpatialTiles(world, tile_size=10.0).tile_ids()
        r2 = SpatialTiles(world, tile_size=10.0).tile_ids()
        assert r1 == r2


class TestFloatingPointStability:
    def test_repeated_construction_from_same_data_is_identical(self):
        """The same entity must not unpredictably move between tiles
        because of tiny numerical differences: rebuilding the tile
        index from IDENTICAL input data must produce the identical
        tile assignment, every time."""
        world = _world_with(
            _entity("a", 9.9999999999, 0.0, 0.0),
            _entity("b", 10.0000000001, 0.0, 0.0),
        )
        for _ in range(20):
            tiles = SpatialTiles(world, tile_size=10.0)
            assert tiles.tile_of((9.9999999999, 0.0, 0.0)) == ["a"]
            assert tiles.tile_of((10.0000000001, 0.0, 0.0)) == ["b"]

    def test_computed_position_equal_to_stored_position_lands_in_same_tile(self):
        """A position recomputed via arithmetic that is mathematically
        equal to a stored position must land in the same tile -- this
        pins down that SpatialTiles does not apply any additional
        rounding/snapping beyond plain floor division."""
        stored = 15.0
        recomputed = 10.0 + 5.0  # same value, different computation path
        assert stored == recomputed  # sanity: no float drift in this case
        world = _world_with(_entity("e", stored, 0.0, 0.0))
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_of((recomputed, 0.0, 0.0)) == ["e"]


class TestGeometryCentroidFallback:
    def test_geometry_bounds_centroid_used_when_no_transform(self):
        from world_ir.schema_v1 import Geometry, GeometryType, Vector3

        world = WorldIR()
        world.geometries["g"] = Geometry(
            id="g", type=GeometryType.BOX,
            bounds_min=Vector3(0.0, 0.0, 0.0), bounds_max=Vector3(20.0, 0.0, 0.0),
        )
        e = Entity(id="e", type=EntityType.STRUCTURE, geometry_ids=["g"], provenance=Provenance.RECONSTRUCTED)
        world.entities["e"] = e
        tiles = SpatialTiles(world, tile_size=10.0)
        # centroid = (10.0, 0.0, 0.0) -> tile 1, not tile 0.
        assert tiles.tile_ids() == [(1, 0, 0)]

    def test_no_transform_and_no_geometry_bounds_is_unlocalized_not_guessed(self):
        e = Entity(id="e", type=EntityType.STRUCTURE, provenance=Provenance.RECONSTRUCTED)
        world = _world_with(e)
        tiles = SpatialTiles(world, tile_size=10.0)
        assert tiles.tile_ids() == []
        assert tiles.unlocalized_entity_ids == ["e"]
