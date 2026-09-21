"""Lazy spatial query correctness (WORLDOS real-data city-scale
checkpoint, Task 2): `worldstore.lazy_query` must give byte-identical
answers to the full-world `SpatialIndex`, while provably touching fewer
tile artifacts than a full load -- otherwise it's not "lazy", it's a
detour to the same cost.
"""

from __future__ import annotations

from engine.scene_graph.spatial_index import SpatialIndex
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR
from worldstore.lazy_query import lazy_nearest, lazy_within_radius, lazy_within_region
from worldstore.store import WorldStore
from worldstore.tiles import open_version, save_version_tiled


def _entity(eid: str, x: float, y: float, z: float) -> Entity:
    return Entity(id=eid, transform={"position": {"x": x, "y": y, "z": z}})


def _grid_world() -> WorldIR:
    # A 5x5 grid of entities spaced 20 units apart -- spans multiple
    # 10-unit tiles on every axis, so lazy queries actually exercise
    # tile-boundary crossing rather than degenerating to a single tile.
    entities = {}
    for i in range(5):
        for j in range(5):
            eid = f"e-{i}-{j}"
            entities[eid] = _entity(eid, i * 20.0, 0.0, j * 20.0)
    return WorldIR(id="w-grid", entities=entities)


def _store_and_version(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    stored = save_version_tiled(store, world, parent=None, tile_size=10.0)
    return store, world, stored.version_id


def test_lazy_within_region_matches_full_index(tmp_path):
    store, world, vid = _store_and_version(tmp_path)
    handle = open_version(store, vid)

    full = SpatialIndex(world, chunk_size=10.0).within_region((0.0, -1.0, 0.0), (45.0, 1.0, 45.0))
    lazy = lazy_within_region(handle, (0.0, -1.0, 0.0), (45.0, 1.0, 45.0))

    assert [e.id for e in full] == [e.id for e in lazy]


def test_lazy_within_radius_matches_full_index(tmp_path):
    store, world, vid = _store_and_version(tmp_path)
    handle = open_version(store, vid)

    full = SpatialIndex(world, chunk_size=10.0).within_radius((40.0, 0.0, 40.0), 25.0)
    lazy = lazy_within_radius(handle, (40.0, 0.0, 40.0), 25.0)

    assert [(e.id, round(d, 9)) for e, d in full] == [(e.id, round(d, 9)) for e, d in lazy]


def test_lazy_nearest_matches_full_index_including_tie_break_order(tmp_path):
    store, world, vid = _store_and_version(tmp_path)
    handle = open_version(store, vid)

    full = SpatialIndex(world, chunk_size=10.0).nearest((40.0, 0.0, 40.0), k=5)
    lazy = lazy_nearest(handle, (40.0, 0.0, 40.0), k=5)

    assert [(e.id, round(d, 9)) for e, d in full] == [(e.id, round(d, 9)) for e, d in lazy]


def test_lazy_nearest_expands_radius_until_correct_even_from_corner(tmp_path):
    # Querying from a far corner forces at least one radius-doubling
    # iteration before the k-th candidate is provably within the
    # searched box -- exercises the ring-expansion loop, not just the
    # single-iteration happy path.
    store, world, vid = _store_and_version(tmp_path)
    handle = open_version(store, vid)

    full = SpatialIndex(world, chunk_size=10.0).nearest((0.0, 0.0, 0.0), k=3)
    lazy = lazy_nearest(handle, (0.0, 0.0, 0.0), k=3)

    assert [(e.id, round(d, 9)) for e, d in full] == [(e.id, round(d, 9)) for e, d in lazy]


def test_lazy_within_region_loads_fewer_tiles_than_the_whole_world(tmp_path, monkeypatch):
    store, world, vid = _store_and_version(tmp_path)
    handle = open_version(store, vid)
    total_tiles = len(handle.manifest.tiles)
    assert total_tiles > 1, "fixture must span multiple tiles for this to be a meaningful check"

    load_count = {"n": 0}
    original_get = store._store.get

    def counting_get(uri):
        load_count["n"] += 1
        return original_get(uri)

    monkeypatch.setattr(store._store, "get", counting_get)
    lazy_within_region(handle, (0.0, -1.0, 0.0), (5.0, 1.0, 5.0))  # one corner tile only

    assert 0 < load_count["n"] < total_tiles


def test_lazy_query_respects_predicate(tmp_path):
    store, world, vid = _store_and_version(tmp_path)
    handle = open_version(store, vid)

    only_row_0 = lambda e: e.id.startswith("e-0-")
    lazy = lazy_within_radius(handle, (0.0, 0.0, 40.0), 100.0, predicate=only_row_0)

    assert lazy and all(e.id.startswith("e-0-") for e, _ in lazy)
