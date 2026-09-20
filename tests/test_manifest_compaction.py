"""Delta-chain compaction (WORLDOS real-data city-scale mission
follow-up): `_resolve_manifest`'s chain walk grows unbounded as more
`save_version_partitioned_delta` saves accumulate. `compact_manifest_chain`
bounds it by materializing the resolved manifest back into a full one,
without rewriting history or requiring destructive cleanup.
"""

from __future__ import annotations

from world_ir.incremental import apply_incremental_update
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore
from worldstore.tiles import (
    compact_manifest_chain,
    delta_chain_depth,
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
    return WorldIR(id="w-compact", entities=entities)


def _chain_of_deltas(store, world, length: int):
    """Builds v-1 (full) -> v-2..v-(length+1) (deltas), each moving a
    different entity so every delta actually changes one tile."""
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)
    current_world = world
    current_parent = "v-1"
    for i in range(length):
        moved = Entity(id=f"e-{i}-{i}", transform={"position": {"x": i * 20.0 + 0.5, "y": 0.0, "z": i * 20.0}})
        result = apply_incremental_update(current_world, [moved], tile_size=10.0)
        vid = f"v-{i + 2}"
        save_version_partitioned_delta(
            store, result.new_world, parent=current_parent, version_id=vid,
            tile_size=10.0, incremental_result=result,
        )
        current_world = result.new_world
        current_parent = vid
    return current_parent, current_world


def test_delta_chain_depth_grows_with_each_delta_save(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, _ = _chain_of_deltas(store, world, length=5)
    assert delta_chain_depth(store, "v-1") == 0  # full manifest
    assert delta_chain_depth(store, last_id) == 5


def test_compact_manifest_chain_resets_depth_to_zero(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, _ = _chain_of_deltas(store, world, length=5)
    assert delta_chain_depth(store, last_id) == 5

    collapsed = compact_manifest_chain(store, last_id)

    assert collapsed == 5
    assert delta_chain_depth(store, last_id) == 0


def test_compact_manifest_chain_preserves_effective_tile_content(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, final_world = _chain_of_deltas(store, world, length=5)

    before = open_version(store, last_id).manifest
    compact_manifest_chain(store, last_id)
    after = open_version(store, last_id).manifest

    before_tiles = {t.tile_key: (t.entity_ids, t.content_hash) for t in before.tiles}
    after_tiles = {t.tile_key: (t.entity_ids, t.content_hash) for t in after.tiles}
    assert before_tiles == after_tiles
    assert before.unlocalized_entity_ids == after.unlocalized_entity_ids

    # World reconstruction is still exact after compaction.
    reloaded = load_version_partitioned(store, last_id)
    assert set(reloaded.entities) == set(final_world.entities)


def test_compacting_an_ancestor_shortens_descendant_chains_too(tmp_path):
    """The whole point: compacting v-3 must make v-6's walk stop at
    v-3 instead of continuing back to v-1, since v-3 now has its own
    full manifest."""
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, _ = _chain_of_deltas(store, world, length=5)  # v-1 (full) .. v-6 (delta depth 5)
    assert delta_chain_depth(store, last_id) == 5

    compact_manifest_chain(store, "v-3")  # was 2 hops from v-1
    assert delta_chain_depth(store, "v-3") == 0

    # v-6's chain now only needs to walk back to v-3 (2 hops: v-6->v-5->v-4->v-3),
    # not all the way to v-1.
    assert delta_chain_depth(store, last_id) == 3


def test_compacting_an_already_full_manifest_is_a_harmless_no_op(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)

    collapsed = compact_manifest_chain(store, "v-1")

    assert collapsed == 0
    assert delta_chain_depth(store, "v-1") == 0
    assert set(load_version_partitioned(store, "v-1").entities) == set(world.entities)
