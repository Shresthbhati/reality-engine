"""World version lifecycle policy (WORLDOS_LIFECYCLE_HEALTH mission,
Mission 1): deterministic, configurable, observable, failure-safe
compaction governance over the existing delta-chain mechanism
(worldstore/tiles.py). Required test matrix per the mission: no
compaction below threshold, compaction at threshold, repeated
compaction, failed compaction, large chain, large manifest, reopening
after compaction, descendant versions after ancestor compaction.
"""

from __future__ import annotations

from world_ir.incremental import apply_incremental_update
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR
from worldstore.lifecycle import CompactionPolicy, evaluate_compaction, maybe_compact
from worldstore.store import WorldStore
from worldstore.tiles import (
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
    return WorldIR(id="w-policy", entities=entities)


def _chain_of_deltas(store, world, length, n=10, spacing=20.0):
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)
    current_world, current_parent = world, "v-1"
    for i in range(length):
        row, col = divmod(i, n)
        eid = f"e-{row % n}-{col % n}"
        moved = Entity(id=eid, transform={"position": {"x": (row % n) * spacing + 0.5, "y": 0.0, "z": (col % n) * spacing}})
        result = apply_incremental_update(current_world, [moved], tile_size=10.0)
        vid = f"v-{i + 2}"
        save_version_partitioned_delta(
            store, result.new_world, parent=current_parent, version_id=vid, tile_size=10.0, incremental_result=result,
        )
        current_world, current_parent = result.new_world, vid
    return current_parent, current_world


def test_no_compaction_below_threshold(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, _ = _chain_of_deltas(store, world, length=3)
    policy = CompactionPolicy(max_chain_depth=10, max_accumulated_delta_bytes=10_000_000, max_deltas_before_forced_compaction=100)

    decision = evaluate_compaction(store, last_id, policy)
    assert decision.should_compact is False
    assert decision.reason is None

    outcome = maybe_compact(store, last_id, policy)
    assert outcome.attempted is False
    assert outcome.compacted is False
    assert delta_chain_depth(store, last_id) == 3  # untouched


def test_compaction_at_chain_depth_threshold(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, _ = _chain_of_deltas(store, world, length=5)
    policy = CompactionPolicy(max_chain_depth=5, max_accumulated_delta_bytes=10_000_000, max_deltas_before_forced_compaction=100)

    decision = evaluate_compaction(store, last_id, policy)
    assert decision.should_compact is True
    assert "max_chain_depth" in decision.reason

    outcome = maybe_compact(store, last_id, policy)
    assert outcome.attempted is True
    assert outcome.compacted is True
    assert outcome.depth_after == 0
    assert delta_chain_depth(store, last_id) == 0


def test_compaction_at_accumulated_bytes_threshold(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, _ = _chain_of_deltas(store, world, length=3)
    # A tiny byte threshold trips before depth would.
    policy = CompactionPolicy(max_chain_depth=100, max_accumulated_delta_bytes=1, max_deltas_before_forced_compaction=100)

    decision = evaluate_compaction(store, last_id, policy)
    assert decision.should_compact is True
    assert "max_accumulated_delta_bytes" in decision.reason


def test_repeated_compaction_is_idempotent_and_observable(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, _ = _chain_of_deltas(store, world, length=5)
    policy = CompactionPolicy(max_chain_depth=5)

    first = maybe_compact(store, last_id, policy)
    assert first.compacted is True

    second = maybe_compact(store, last_id, policy)
    assert second.attempted is False  # already depth 0, policy no longer trips
    assert second.decision.depth == 0


def test_failed_compaction_never_invalidates_the_previous_valid_version(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, final_world = _chain_of_deltas(store, world, length=5)
    policy = CompactionPolicy(max_chain_depth=5)

    # Corrupt a delta file mid-chain -- a realistic failure
    # (`_resolve_manifest` reads manifest/delta JSON directly from
    # disk, not through the artifact store, so that's the actual
    # dependency to fail here).
    from worldstore.tiles import _delta_path
    mid_delta = _delta_path(store, "v-3")
    original_bytes = mid_delta.read_bytes()
    mid_delta.write_bytes(b"{not valid json")

    outcome = maybe_compact(store, last_id, policy)
    assert outcome.compacted is False
    assert outcome.error is not None

    mid_delta.write_bytes(original_bytes)  # restore -- simulates the transient failure clearing
    # Nothing was corrupted by the failed attempt: the chain still
    # resolves exactly as before, with its original (unwritten) depth.
    assert delta_chain_depth(store, last_id) == 5
    reloaded = load_version_partitioned(store, last_id)
    assert set(reloaded.entities) == set(final_world.entities)


def test_large_chain_compaction(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, final_world = _chain_of_deltas(store, world, length=60)
    policy = CompactionPolicy(max_chain_depth=20)

    assert delta_chain_depth(store, last_id) == 60
    outcome = maybe_compact(store, last_id, policy)
    assert outcome.compacted is True
    assert delta_chain_depth(store, last_id) == 0
    reloaded = load_version_partitioned(store, last_id)
    assert set(reloaded.entities) == set(final_world.entities)


def test_large_manifest_compaction(tmp_path):
    """A wide (many-tile) world, not just a deep chain."""
    store = WorldStore(tmp_path / "store")
    world = _grid_world(n=20, spacing=20.0)  # 400 tiles
    last_id, final_world = _chain_of_deltas(store, world, length=10, n=20, spacing=20.0)
    policy = CompactionPolicy(max_chain_depth=10)

    outcome = maybe_compact(store, last_id, policy)
    assert outcome.compacted is True
    handle = open_version(store, last_id)
    assert len(handle.manifest.tiles) == 400
    reloaded = load_version_partitioned(store, last_id)
    assert set(reloaded.entities) == set(final_world.entities)


def test_reopening_after_compaction_returns_identical_data(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, final_world = _chain_of_deltas(store, world, length=5)

    before = {t.tile_key: t.content_hash for t in open_version(store, last_id).manifest.tiles}
    maybe_compact(store, last_id, CompactionPolicy(max_chain_depth=5))
    after = {t.tile_key: t.content_hash for t in open_version(store, last_id).manifest.tiles}

    assert before == after
    assert set(load_version_partitioned(store, last_id).entities) == set(final_world.entities)


def test_descendant_versions_after_ancestor_compaction_still_resolve_correctly(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    last_id, final_world = _chain_of_deltas(store, world, length=6)  # v-1..v-7

    maybe_compact(store, "v-4", CompactionPolicy(max_chain_depth=0))  # force-compact an ancestor
    assert delta_chain_depth(store, "v-4") == 0
    assert delta_chain_depth(store, last_id) == 3  # v-7 -> v-6 -> v-5 -> v-4(full)

    reloaded = load_version_partitioned(store, last_id)
    assert set(reloaded.entities) == set(final_world.entities)
    for eid in final_world.entities:
        assert reloaded.entities[eid].to_dict() == final_world.entities[eid].to_dict()
