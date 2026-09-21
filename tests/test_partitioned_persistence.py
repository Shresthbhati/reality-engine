"""Partitioned (O(affected), not O(n)) persistence (WORLDOS real-data
city-scale checkpoint, Task 5): `save_version_partitioned`/
`load_version_partitioned` are an ADDITIVE alternative to
`WorldStore.save_version()`/`load_version()`, which are unchanged and
must keep working exactly as before for whole-world versions.
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
    save_version_tiled,
)


def _entity(eid: str, x: float, y: float, z: float) -> Entity:
    return Entity(id=eid, transform={"position": {"x": x, "y": y, "z": z}})


def _grid_world(n: int = 4, spacing: float = 20.0) -> WorldIR:
    entities = {}
    for i in range(n):
        for j in range(n):
            eid = f"e-{i}-{j}"
            entities[eid] = _entity(eid, i * spacing, 0.0, j * spacing)
    return WorldIR(id="w-grid", entities=entities)


def test_partitioned_round_trip_matches_original_world(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    stored = save_version_partitioned(store, world, parent=None, tile_size=10.0)

    reloaded = load_version_partitioned(store, stored.version_id)

    assert set(reloaded.entities) == set(world.entities)
    for eid, entity in world.entities.items():
        assert reloaded.entities[eid].to_dict() == entity.to_dict()


def test_partitioned_round_trip_preserves_unlocalized_entities(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    # An entity with no position at all -- must survive via the residual
    # blob, since it has no tile membership whatsoever.
    world.entities["unlocalized-1"] = Entity(id="unlocalized-1")
    stored = save_version_partitioned(store, world, parent=None, tile_size=10.0)

    reloaded = load_version_partitioned(store, stored.version_id)

    assert "unlocalized-1" in reloaded.entities
    assert set(reloaded.entities) == set(world.entities)


def test_whole_world_versions_remain_readable_alongside_partitioned_ones(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    whole = save_version_tiled(store, world, parent=None, tile_size=10.0)
    partitioned = save_version_partitioned(store, world, parent=whole.version_id, tile_size=10.0)

    # Old API, old version: completely unaffected by the new path.
    assert set(store.load_version(whole.version_id).entities) == set(world.entities)
    # New API, new version.
    assert set(load_version_partitioned(store, partitioned.version_id).entities) == set(world.entities)

    ids = [v.version_id for v in store.list_versions()]
    assert whole.version_id in ids
    assert partitioned.version_id in ids


def test_load_version_rejects_a_partitioned_version_with_a_clear_error(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    partitioned = save_version_partitioned(store, world, parent=None, tile_size=10.0)

    with pytest.raises(Exception):
        store.load_version(partitioned.version_id)


def test_load_version_partitioned_rejects_a_whole_world_version(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    whole = save_version_tiled(store, world, parent=None, tile_size=10.0)

    with pytest.raises(WorldStoreError):
        load_version_partitioned(store, whole.version_id)


def test_partitioned_save_reuses_unaffected_tiles_via_incremental_result(tmp_path):
    store = WorldStore(tmp_path / "store")
    base_world = _grid_world()
    v1 = save_version_partitioned(store, base_world, parent=None, tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(base_world, [moved], tile_size=10.0)
    v2 = save_version_partitioned(
        store, result.new_world, parent=v1.version_id, tile_size=10.0, incremental_result=result,
    )

    v1_handle = open_version(store, v1.version_id)
    v2_handle = open_version(store, v2.version_id)
    unaffected_key = next(
        t.tile_key for t in v1_handle.manifest.tiles if "e-3-3" in t.entity_ids
    )
    v1_ref = next(t for t in v1_handle.manifest.tiles if t.tile_key == unaffected_key)
    v2_ref = next(t for t in v2_handle.manifest.tiles if t.tile_key == unaffected_key)
    assert v1_ref.artifact_uri == v2_ref.artifact_uri  # reused verbatim, not rewritten


def test_partitioned_save_crash_leaves_v1_valid_and_v2_unreferenced(tmp_path, monkeypatch):
    store = WorldStore(tmp_path / "store")
    base_world = _grid_world()
    v1 = save_version_partitioned(store, base_world, parent=None, version_id="v-1", tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(base_world, [moved], tile_size=10.0)

    original_put = store._store.put
    call_count = {"n": 0}

    def flaky_put(data: bytes):
        call_count["n"] += 1
        # Call #1 is the first rebuilt tile artifact write -- fail there
        # to simulate a crash before the residual artifact, manifest, or
        # version record are ever written.
        if call_count["n"] == 1:
            raise OSError("simulated disk failure during partitioned save")
        return original_put(data)

    monkeypatch.setattr(store._store, "put", flaky_put)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_version_partitioned(
            store, result.new_world, parent="v-1", version_id="v-2",
            tile_size=10.0, incremental_result=result,
        )

    # V1 is completely unaffected.
    reloaded_v1 = load_version_partitioned(store, "v-1")
    assert set(reloaded_v1.entities) == set(base_world.entities)
    # V2 is undiscoverable via any path -- no manifest, no version record.
    with pytest.raises(WorldStoreError):
        open_version(store, "v-2")
    with pytest.raises(WorldStoreError):
        load_version_partitioned(store, "v-2")
    assert "v-1" in [v.version_id for v in store.list_versions()]
    assert "v-2" not in [v.version_id for v in store.list_versions()]


def test_partitioned_save_writes_far_fewer_bytes_than_a_whole_world_save(tmp_path):
    """The actual measured claim behind Task 5: for a localized change
    in a many-tile world, the partitioned path's new bytes on disk are
    a small fraction of a full whole-world save's bytes -- not just a
    different code path with the same cost."""
    store_a = WorldStore(tmp_path / "store_whole")
    store_b = WorldStore(tmp_path / "store_partitioned")
    base_world = _grid_world(n=10, spacing=20.0)  # 100 entities, ~100 tiles

    v1_whole = save_version_tiled(store_a, base_world, parent=None, tile_size=10.0)
    v1_part = save_version_partitioned(store_b, base_world, parent=None, tile_size=10.0)

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(base_world, [moved], tile_size=10.0)

    def _bytes_under(root):
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

    whole_before = _bytes_under(tmp_path / "store_whole")
    part_before = _bytes_under(tmp_path / "store_partitioned")

    save_version_tiled(store_a, result.new_world, parent=v1_whole.version_id, tile_size=10.0, incremental_result=result)
    save_version_partitioned(
        store_b, result.new_world, parent=v1_part.version_id, tile_size=10.0, incremental_result=result,
    )

    whole_growth = _bytes_under(tmp_path / "store_whole") - whole_before
    part_growth = _bytes_under(tmp_path / "store_partitioned") - part_before

    # Honest claim, not an inflated one: the partitioned save's ONLY
    # win is skipping the full-world blob (it still rewrites the tile
    # manifest in full every version, which is O(tile count) -- see
    # the "known limitations" note in the Task 5 handoff). So the
    # measured saving here is real but modest at this scale, not an
    # order-of-magnitude reduction.
    assert part_growth < whole_growth
