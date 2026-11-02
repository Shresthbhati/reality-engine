"""Locality instrumentation (WORLDOS real-data city-scale mission):
directly proves the two architectural failures the mission calls out by
name never happen, with hard counts, not just passing assertions on
final results:

    "lazy query -> secretly loads entire WorldIR"
    "incremental update -> still recompiles the whole world"

Uses a 100-tile world (10x10 grid, one entity per tile) so "loads
everything" vs "loads a bounded few" is unambiguous.
"""

from __future__ import annotations

from world_ir.incremental import apply_incremental_update
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR
from worldstore.lazy_query import lazy_nearest, lazy_within_radius, lazy_within_region
from worldstore.store import WorldStore
from worldstore.tiles import open_version, save_version_partitioned, save_version_tiled


def _entity(eid: str, x: float, y: float, z: float) -> Entity:
    return Entity(id=eid, transform={"position": {"x": x, "y": y, "z": z}})


def _grid_world(n: int = 10, spacing: float = 20.0) -> WorldIR:
    entities = {}
    for i in range(n):
        for j in range(n):
            eid = f"e-{i}-{j}"
            entities[eid] = _entity(eid, i * spacing, 0.0, j * spacing)
    return WorldIR(id="w-locality", entities=entities)


def _count_artifact_gets(store, fn):
    calls = {"n": 0}
    original_get = store._store.get

    def counting_get(uri):
        calls["n"] += 1
        return original_get(uri)

    store._store.get = counting_get
    try:
        result = fn()
    finally:
        store._store.get = original_get
    return result, calls["n"]


def _count_artifact_puts(store, fn):
    calls = {"n": 0}
    original_put = store._store.put

    def counting_put(data):
        calls["n"] += 1
        return original_put(data)

    store._store.put = counting_put
    try:
        result = fn()
    finally:
        store._store.put = original_put
    return result, calls["n"]


def test_lazy_query_never_touches_more_than_a_bounded_few_tiles(tmp_path):
    """Proves query locality is real: a query over ~1% of the world's
    area must not read anywhere close to all 100 tile artifacts."""
    store = WorldStore(tmp_path / "store")
    world = _grid_world(n=10, spacing=20.0)  # 100 entities, 100 tiles (tile_size=10)
    stored = save_version_tiled(store, world, parent=None, tile_size=10.0)
    total_tiles = len(open_version(store, stored.version_id).manifest.tiles)
    assert total_tiles == 100

    # A FRESH handle per query -- WorldVersionHandle caches loaded
    # tiles per-instance (Mission 5 fix), so this measures each query
    # TYPE's own locality in isolation, not cache reuse across calls.
    _, region_gets = _count_artifact_gets(
        store, lambda: lazy_within_region(open_version(store, stored.version_id), (0.0, -1.0, 0.0), (5.0, 1.0, 5.0))
    )
    _, radius_gets = _count_artifact_gets(
        store, lambda: lazy_within_radius(open_version(store, stored.version_id), (0.0, 0.0, 0.0), 5.0)
    )
    _, nearest_gets = _count_artifact_gets(
        store, lambda: lazy_nearest(open_version(store, stored.version_id), (0.0, 0.0, 0.0), k=1)
    )

    # "Bounded few", not "secretly everything": each query touches a
    # small fraction of the 100 tiles, never the whole manifest.
    assert 0 < region_gets < total_tiles // 4
    assert 0 < radius_gets < total_tiles // 4
    assert 0 < nearest_gets < total_tiles // 4


def test_repeated_query_on_the_same_handle_reuses_cached_tiles(tmp_path):
    """Mission 5 (query amplification): measured that a repeated or
    adjacent query against the SAME open handle re-fetched and
    re-verified identical tile bytes every time -- proven inefficiency,
    fixed with a per-handle tile cache. This proves the fix: a second,
    identical query costs ZERO additional artifact reads."""
    store = WorldStore(tmp_path / "store")
    world = _grid_world(n=10, spacing=20.0)
    stored = save_version_tiled(store, world, parent=None, tile_size=10.0)
    handle = open_version(store, stored.version_id)

    _, first_gets = _count_artifact_gets(
        store, lambda: lazy_within_radius(handle, (0.0, 0.0, 0.0), 5.0)
    )
    assert first_gets > 0

    _, second_gets = _count_artifact_gets(
        store, lambda: lazy_within_radius(handle, (0.0, 0.0, 0.0), 5.0)
    )
    assert second_gets == 0  # fully served from the handle's tile cache

    # An ADJACENT query touching an overlapping tile set also benefits
    # partially -- only the NEW tile(s) it introduces cost a real read.
    _, adjacent_gets = _count_artifact_gets(
        store, lambda: lazy_within_radius(handle, (5.0, 0.0, 0.0), 5.0)
    )
    assert adjacent_gets <= first_gets


def test_incremental_update_never_touches_more_than_the_affected_tiles(tmp_path):
    """Proves compilation locality is real: persisting a 1-entity
    change writes artifacts for only the affected tile(s), never
    rewrites all 100 tiles' content."""
    store = WorldStore(tmp_path / "store")
    world = _grid_world(n=10, spacing=20.0)
    v1 = save_version_tiled(store, world, parent=None, tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)

    _, put_calls = _count_artifact_puts(
        store,
        lambda: save_version_tiled(
            store, result.new_world, parent=v1.version_id, tile_size=10.0, incremental_result=result,
        ),
    )

    # 1 put for the whole-world blob (save_version_tiled still writes
    # it) + at most 2 puts for the rebuilt tile(s) (old/new tile if the
    # entity crossed a boundary) -- NOT 100 puts for every tile.
    assert put_calls <= 3
    assert len(result.rebuilt_tile_ids) <= 2
    assert len(result.changed_entity_ids) == 1


def test_partitioned_incremental_update_writes_only_affected_artifacts(tmp_path):
    """Same proof as above, but for the O(affected) `save_version_partitioned`
    path, which has no whole-world blob to inflate the put count at all."""
    store = WorldStore(tmp_path / "store")
    world = _grid_world(n=10, spacing=20.0)
    v1 = save_version_partitioned(store, world, parent=None, tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)

    _, put_calls = _count_artifact_puts(
        store,
        lambda: save_version_partitioned(
            store, result.new_world, parent=v1.version_id, tile_size=10.0, incremental_result=result,
        ),
    )

    # 1 put for the rebuilt tile(s) (<=2, boundary-crossing case) + 1
    # put for the residual blob -- NOT one put per tile in the world.
    assert put_calls <= 3

    v2 = save_version_partitioned(
        store, result.new_world, parent=v1.version_id, version_id="v-2b", tile_size=10.0, incremental_result=result,
    )
    v1_handle = open_version(store, v1.version_id)
    v2_handle = open_version(store, "v-2b")
    reused = sum(
        1 for a, b in zip(
            sorted(v1_handle.manifest.tiles, key=lambda t: t.tile_key),
            sorted(v2_handle.manifest.tiles, key=lambda t: t.tile_key),
        )
        if a.artifact_uri == b.artifact_uri
    )
    assert reused == 99  # every tile except e-0-0's is byte-identical-reused


def test_locality_instrumentation_summary(tmp_path, capsys):
    """Prints the exact counts the mission asked to have measured:
    tiles loaded, tiles rebuilt, tiles reused, entities rebuilt,
    artifacts reused -- as one consolidated, reproducible report."""
    store = WorldStore(tmp_path / "store")
    world = _grid_world(n=10, spacing=20.0)
    v1 = save_version_partitioned(store, world, parent=None, tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)
    v2 = save_version_partitioned(
        store, result.new_world, parent=v1.version_id, tile_size=10.0, incremental_result=result,
    )

    v2_handle = open_version(store, v2.version_id)
    _, query_tile_gets = _count_artifact_gets(
        store, lambda: lazy_within_radius(v2_handle, (0.0, 0.0, 0.0), 5.0)
    )

    summary = {
        "total_tiles": len(v2_handle.manifest.tiles),
        "tiles_rebuilt": len(result.rebuilt_tile_ids),
        "tiles_reused": len(v2_handle.manifest.tiles) - len(result.rebuilt_tile_ids),
        "entities_changed": len(result.changed_entity_ids),
        "entities_affected": len(result.affected_entity_ids),
        "tile_artifacts_touched_by_one_local_query": query_tile_gets,
    }
    print(summary)
    captured = capsys.readouterr()
    assert "total_tiles" in captured.out

    assert summary["tiles_rebuilt"] == 1
    assert summary["tiles_reused"] == 99
    assert summary["entities_changed"] == 1
    assert summary["tile_artifacts_touched_by_one_local_query"] < 10
