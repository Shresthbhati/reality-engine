"""Streaming/bounded-memory city compilation (P14-01 increment:
"large-extent streaming compile" -- the ledger's named open item).

compile_city_world materializes every feature's Entity in one world at
once. For city-extent imports (hundreds of thousands of OSM ways) that
is unbounded. The streaming path compiles features in deterministic
spatial tiles, YIELDING one (tile_key, WorldIR) fragment per non-empty
tile -- callers process tiles one at a time (export, persist via
WorldStore's own tiling) without ever holding the whole city in
memory. Claude's WorldStore owns persistence/partitioning; this module
only produces honest, tile-scoped world fragments.

Rules under test:
  - Deterministic: same features + tile size -> identical fragment
    sequence (sorted tile keys, stable per-tile entity ids).
  - A feature spanning several tiles is compiled into EVERY tile its
    footprint AABB touches (the same multi-chunk membership rule
    SpatialTiling uses), with that fact recorded on the entity and in
    the report -- cross-tile identity is never silently merged/split.
  - Refusals propagate: unknown kinds refuse in CityFeature's
    constructor exactly like compile_city_world.
"""

import pytest

from engine.compiler.city_compiler import CityFeature



def _grid_buildings(n_per_side=6, spacing=10.0):
    """Deterministic n x n grid of buildings, each 4x4 m footprint."""
    features = []
    for i in range(n_per_side):
        for j in range(n_per_side):
            x0 = i * spacing
            y0 = j * spacing
            features.append(
                CityFeature(
                    kind="building",
                    name=f"b-{i}-{j}",
                    footprint=[(x0, y0), (x0 + 4.0, y0), (x0 + 4.0, y0 + 4.0), (x0, y0 + 4.0)],
                    height=12.0,
                )
            )
    return features


class TestStreamingCompile:
    def test_produces_multiple_tile_fragments(self):
        from engine.compiler.streaming import compile_city_world_streaming

        fragments = list(
            compile_city_world_streaming(_grid_buildings(), tile_size=25.0, name="city")
        )
        # Yields are (tile_key, world) pairs in sorted key order, then a
        # final (None, report) summary yield.
        worlds = [w for k, w in fragments if k is not None]
        assert len(worlds) > 1
        total_entities = sum(len(w.entities) for w in worlds)
        # Every building lands in exactly one tile (4 m footprints in a
        # 25 m grid do not straddle tile boundaries).
        assert total_entities == 36

    def test_deterministic_fragment_order_and_ids(self):
        from engine.compiler.streaming import compile_city_world_streaming

        a = list(compile_city_world_streaming(_grid_buildings(), tile_size=25.0, name="city"))
        b = list(compile_city_world_streaming(_grid_buildings(), tile_size=25.0, name="city"))
        keys_a = [k for k, _ in a]
        keys_b = [k for k, _ in b]
        assert keys_a[:-1] == sorted(keys_a[:-1])  # sorted tile keys, report last
        assert keys_a == keys_b
        for (ka, wa), (kb, wb) in zip(a, b):
            if ka is None:  # the (None, report) summary yield
                assert isinstance(wa, dict) and wa == wb
                continue
            assert ka == kb
            assert list(wa.entities) == list(wb.entities)
            assert wa.to_dict() == wb.to_dict()

    def test_spanning_feature_compiles_into_every_touched_tile(self):
        from engine.compiler.streaming import compile_city_world_streaming

        # A 60 m x 6 m road crosses three 25 m tiles along x (0-24, 24-49,
        # 49-60 clipped) -- its AABB touches exactly three tile cells.
        road = CityFeature(
            kind="road",
            name="main-street",
            footprint=[(0.0, 0.0), (60.0, 0.0), (60.0, 6.0), (0.0, 6.0)],
            height=None,
        )
        fragments = list(compile_city_world_streaming([road], tile_size=25.0, name="city"))
        nonempty = [
            (k, w) for k, w in fragments
            if k is not None and len(w.entities) > 0
        ]
        assert len(nonempty) == 3
        for _, w in nonempty:
            (entity,) = w.entities.values()
            assert entity.custom_properties.get("spans_multiple_tiles") is True

    def test_report_counts_features_and_spans(self):
        from engine.compiler.streaming import compile_city_world_streaming

        fragments = list(
            compile_city_world_streaming(_grid_buildings(3), tile_size=25.0, name="city")
        )
        report = fragments[-1][1]
        assert report["features_compiled"] == 9
        assert report["tiles_emitted"] == len(fragments) - 1  # report yield excluded
        assert report["features_spanning_tiles"] == 0

    def test_report_counts_spanning_features(self):
        from engine.compiler.streaming import compile_city_world_streaming

        road = CityFeature(
            kind="road",
            name="main-street",
            footprint=[(0.0, 0.0), (60.0, 0.0), (60.0, 6.0), (0.0, 6.0)],
        )
        fragments = list(compile_city_world_streaming([road], tile_size=25.0, name="city"))
        assert fragments[-1][1]["features_spanning_tiles"] == 1

    def test_unknown_kind_refuses(self):
        from engine.compiler.streaming import compile_city_world_streaming

        with pytest.raises(ValueError):
            list(compile_city_world_streaming(
                [CityFeature(kind="volcano", name="v", footprint=[(0, 0), (1, 0), (1, 1)])],
                tile_size=10.0,
            ))
