"""Concurrency safety (WORLDOS_LIFECYCLE_HEALTH mission, Mission 6):
simultaneous readers, a reader racing V2 creation, a failed V2 creation
racing readers, concurrent incremental updates to independent versions,
and compaction racing readers. Required invariants per the mission:
existing V1 remains readable throughout, no partial manifests are ever
observable, no partial artifact references are ever observable.

Scope, stated honestly: these tests exercise SAME-PROCESS thread
concurrency against the real filesystem (real `_atomic_write_text`/
`os.replace`, real `_SequenceLock` file-mkdir mutex) -- they are not a
multi-process test. The mechanisms under test (atomic rename, a real
mkdir-based cross-process lock) are the same ones that would matter
across processes, but true multi-process races were not separately
exercised this session.
"""

from __future__ import annotations

import threading
import time

from world_ir.incremental import apply_incremental_update
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR
from worldstore.lifecycle import CompactionPolicy, maybe_compact
from worldstore.store import WorldStore
from worldstore.tiles import (
    load_version_partitioned,
    open_version,
    save_version_partitioned,
    save_version_partitioned_delta,
    save_version_tiled,
)


def _entity(eid: str, x: float, y: float, z: float) -> Entity:
    return Entity(id=eid, transform={"position": {"x": x, "y": y, "z": z}})


def _grid_world(n: int = 10, spacing: float = 20.0) -> WorldIR:
    entities = {}
    for i in range(n):
        for j in range(n):
            eid = f"e-{i}-{j}"
            entities[eid] = _entity(eid, i * spacing, 0.0, j * spacing)
    return WorldIR(id="w-concurrency", entities=entities)


def test_simultaneous_readers_get_consistent_results(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)

    results = []
    errors = []
    barrier = threading.Barrier(8)

    def reader():
        try:
            barrier.wait(timeout=5)
            handle = open_version(store, "v-1")
            results.append(len(handle.manifest.tiles))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert len(results) == 8
    assert all(r == results[0] for r in results)  # every reader saw the same tile count


def test_reader_during_v2_creation_never_sees_a_broken_v1(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)

    read_results = []
    read_errors = []
    stop = threading.Event()

    def reader_loop():
        while not stop.is_set():
            try:
                reloaded = load_version_partitioned(store, "v-1")
                read_results.append(set(reloaded.entities) == set(world.entities))
            except Exception as exc:  # noqa: BLE001
                read_errors.append(exc)

    reader_thread = threading.Thread(target=reader_loop)
    reader_thread.start()

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)
    for i in range(20):  # several writer iterations to widen the race window
        save_version_partitioned(
            store, result.new_world, parent="v-1", version_id=f"v-writer-{i}", tile_size=10.0,
        )

    stop.set()
    reader_thread.join(timeout=10)

    assert not read_errors
    assert read_results  # the reader actually got to run
    assert all(read_results)  # every single read of v-1 was correct, none partial/broken


def test_failed_v2_creation_never_affects_concurrent_v1_reads(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)

    read_results = []
    read_errors = []
    stop = threading.Event()

    def reader_loop():
        while not stop.is_set():
            try:
                reloaded = load_version_partitioned(store, "v-1")
                read_results.append(set(reloaded.entities) == set(world.entities))
            except Exception as exc:  # noqa: BLE001
                read_errors.append(exc)

    reader_thread = threading.Thread(target=reader_loop)
    reader_thread.start()
    time.sleep(0.01)  # let the reader get going first

    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)
    for i in range(10):
        try:
            save_version_partitioned_delta(
                store, result.new_world, parent="v-nonexistent-parent", version_id=f"v-fail-{i}", tile_size=10.0,
            )
        except Exception:
            pass  # expected -- this write is deliberately doomed to fail

    stop.set()
    reader_thread.join(timeout=10)

    assert not read_errors
    assert all(read_results)


def test_concurrent_incremental_updates_to_independent_versions_both_succeed(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)

    outcomes = {}
    errors = []
    barrier = threading.Barrier(2)

    def writer(entity_id: str, target_version: str, new_x: float, new_z: float):
        try:
            barrier.wait(timeout=5)
            # Small in-tile nudge (not a cross-tile move) so both
            # writers' resulting worlds keep the same 100-tile layout --
            # the point of this test is independent-version isolation,
            # not tile-count arithmetic.
            moved = Entity(id=entity_id, transform={"position": {"x": new_x, "y": 0.0, "z": new_z}})
            result = apply_incremental_update(world, [moved], tile_size=10.0)
            save_version_partitioned(
                store, result.new_world, parent="v-1", version_id=target_version,
                tile_size=10.0, incremental_result=result,
            )
            outcomes[target_version] = True
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=writer, args=("e-0-0", "v-a", 0.5, 0.0))
    t2 = threading.Thread(target=writer, args=("e-9-9", "v-b", 180.5, 180.0))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors
    assert outcomes == {"v-a": True, "v-b": True}
    # Both independently readable and correct -- neither writer's
    # concurrent sequence.json append clobbered the other's.
    handle_a = open_version(store, "v-a")
    handle_b = open_version(store, "v-b")
    assert len(handle_a.manifest.tiles) == len(handle_b.manifest.tiles)
    version_ids = {v.version_id for v in store.list_versions()}
    assert {"v-1", "v-a", "v-b"} <= version_ids


def test_compaction_while_readers_exist_never_breaks_reads(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)
    current_world, current_parent = world, "v-1"
    for i in range(15):
        moved = Entity(id=f"e-{i % 10}-{i % 10}", transform={"position": {"x": (i % 10) * 20.0 + 0.5, "y": 0.0, "z": (i % 10) * 20.0}})
        result = apply_incremental_update(current_world, [moved], tile_size=10.0)
        vid = f"v-{i + 2}"
        save_version_partitioned_delta(
            store, result.new_world, parent=current_parent, version_id=vid, tile_size=10.0, incremental_result=result,
        )
        current_world, current_parent = result.new_world, vid
    last_id, final_world = current_parent, current_world

    read_results = []
    read_errors = []
    stop = threading.Event()

    def reader_loop():
        while not stop.is_set():
            try:
                reloaded = load_version_partitioned(store, last_id)
                read_results.append(set(reloaded.entities) == set(final_world.entities))
            except Exception as exc:  # noqa: BLE001
                read_errors.append(exc)

    reader_thread = threading.Thread(target=reader_loop)
    reader_thread.start()
    time.sleep(0.005)

    outcome = maybe_compact(store, last_id, CompactionPolicy(max_chain_depth=5))

    stop.set()
    reader_thread.join(timeout=10)

    assert outcome.compacted is True
    assert not read_errors
    assert read_results
    assert all(read_results)  # every read during/around compaction was correct, none partial
