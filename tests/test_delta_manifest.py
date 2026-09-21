"""Delta tile manifests (WORLDOS real-data city-scale mission, item 8):
`save_version_partitioned` still rewrote the FULL tile manifest on every
save -- O(tile count), not O(affected) -- because every tile got a
manifest entry whether reused or not. `save_version_partitioned_delta`
fixes this: it writes ONLY changed/new tile refs and vacated tile keys.
`open_version`/`load_version_partitioned` must resolve a delta chain
transparently and produce results IDENTICAL to a full manifest.
"""

from __future__ import annotations

import pytest

from world_ir.incremental import apply_incremental_update
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore, WorldStoreError
from worldstore.tiles import (
    load_version_partitioned,
    open_version,
    save_version_partitioned,
    save_version_partitioned_delta,
)


def _entity(eid: str, x: float, y: float, z: float) -> Entity:
    return Entity(id=eid, transform={"position": {"x": x, "y": y, "z": z}})


def _grid_world(n: int = 10, spacing: float = 20.0) -> WorldIR:
    entities = {}
    for i in range(n):
        for j in range(n):
            eid = f"e-{i}-{j}"
            entities[eid] = _entity(eid, i * spacing, 0.0, j * spacing)
    return WorldIR(id="w-delta", entities=entities)


def test_delta_save_requires_a_resolvable_parent(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    with pytest.raises(WorldStoreError):
        save_version_partitioned_delta(store, world, parent="nonexistent", tile_size=10.0)


def test_delta_save_resolves_to_the_same_effective_manifest_as_a_full_save(tmp_path):
    store_full = WorldStore(tmp_path / "store_full")
    store_delta = WorldStore(tmp_path / "store_delta")
    world = _grid_world()
    save_version_partitioned(store_full, world, parent=None, version_id="v-1", tile_size=10.0)
    save_version_partitioned(store_delta, world, parent=None, version_id="v-1", tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)

    save_version_partitioned(
        store_full, result.new_world, parent="v-1", version_id="v-2", tile_size=10.0, incremental_result=result,
    )
    save_version_partitioned_delta(
        store_delta, result.new_world, parent="v-1", version_id="v-2", tile_size=10.0, incremental_result=result,
    )

    h_full = open_version(store_full, "v-2")
    h_delta = open_version(store_delta, "v-2")

    full_tiles = {t.tile_key: (t.entity_ids, t.content_hash) for t in h_full.manifest.tiles}
    delta_tiles = {t.tile_key: (t.entity_ids, t.content_hash) for t in h_delta.manifest.tiles}
    assert full_tiles == delta_tiles  # identical effective tile set, different storage mechanism
    assert h_full.manifest.unlocalized_entity_ids == h_delta.manifest.unlocalized_entity_ids


def test_delta_chain_of_three_versions_resolves_correctly(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)

    moved_a = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    r2 = apply_incremental_update(world, [moved_a], tile_size=10.0)
    save_version_partitioned_delta(
        store, r2.new_world, parent="v-1", version_id="v-2", tile_size=10.0, incremental_result=r2,
    )

    moved_b = Entity(id="e-5-5", transform={"position": {"x": 100.5, "y": 0.0, "z": 100.0}})
    r3 = apply_incremental_update(r2.new_world, [moved_b], tile_size=10.0)
    save_version_partitioned_delta(
        store, r3.new_world, parent="v-2", version_id="v-3", tile_size=10.0, incremental_result=r3,
    )

    handle = open_version(store, "v-3")
    assert len(handle.manifest.tiles) == 100  # no tile count drift across the chain

    # Reloading the full world through the chain must match r3.new_world exactly.
    reloaded = load_version_partitioned(store, "v-3")
    assert set(reloaded.entities) == set(r3.new_world.entities)
    for eid in r3.new_world.entities:
        assert reloaded.entities[eid].to_dict() == r3.new_world.entities[eid].to_dict()


def test_delta_reuses_unaffected_tile_artifacts_across_the_chain(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)
    save_version_partitioned_delta(
        store, result.new_world, parent="v-1", version_id="v-2", tile_size=10.0, incremental_result=result,
    )

    v1_handle = open_version(store, "v-1")
    v2_handle = open_version(store, "v-2")
    unaffected_key = next(t.tile_key for t in v1_handle.manifest.tiles if "e-9-9" in t.entity_ids)
    v1_ref = next(t for t in v1_handle.manifest.tiles if t.tile_key == unaffected_key)
    v2_ref = next(t for t in v2_handle.manifest.tiles if t.tile_key == unaffected_key)
    assert v1_ref.artifact_uri == v2_ref.artifact_uri


def test_delta_write_is_far_smaller_than_a_full_manifest_rewrite(tmp_path):
    """The actual point of this feature: at real scale, a delta save's
    bytes-on-disk growth for the MANIFEST/DELTA file itself must be a
    small fraction of a full manifest rewrite's size, since the delta
    only lists changed tiles instead of all of them."""
    store_full = WorldStore(tmp_path / "store_full")
    store_delta = WorldStore(tmp_path / "store_delta")
    world = _grid_world(n=20, spacing=20.0)  # 400 entities, 400 tiles
    save_version_partitioned(store_full, world, parent=None, version_id="v-1", tile_size=10.0)
    save_version_partitioned(store_delta, world, parent=None, version_id="v-1", tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)

    save_version_partitioned(
        store_full, result.new_world, parent="v-1", version_id="v-2", tile_size=10.0, incremental_result=result,
    )
    save_version_partitioned_delta(
        store_delta, result.new_world, parent="v-1", version_id="v-2", tile_size=10.0, incremental_result=result,
    )

    full_manifest_bytes = (tmp_path / "store_full" / "tile_manifests" / "v-2.json").stat().st_size
    delta_bytes = (tmp_path / "store_delta" / "tile_manifest_deltas" / "v-2.json").stat().st_size

    assert delta_bytes < full_manifest_bytes / 10  # delta lists ~1 tile vs. the full manifest's 400


def test_delta_save_crash_leaves_parent_chain_valid_and_new_version_unreferenced(tmp_path, monkeypatch):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)

    original_put = store._store.put
    call_count = {"n": 0}

    def flaky_put(data: bytes):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated disk failure during delta save")
        return original_put(data)

    monkeypatch.setattr(store._store, "put", flaky_put)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_version_partitioned_delta(
            store, result.new_world, parent="v-1", version_id="v-2", tile_size=10.0, incremental_result=result,
        )

    assert set(load_version_partitioned(store, "v-1").entities) == set(world.entities)
    with pytest.raises(WorldStoreError):
        open_version(store, "v-2")
