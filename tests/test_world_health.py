"""Read-only WorldOS health/diagnostics (WORLDOS_LIFECYCLE_HEALTH
mission, Mission 2): every fact reported must come from existing
WorldStore/tile metadata -- these tests verify both the happy path
(a healthy version reports healthy) and each failure mode the mission
asked for (missing artifact, corrupt artifact, stale reference, broken
lineage) is detected without raising.
"""

from __future__ import annotations

import json

from world_ir.incremental import apply_incremental_update
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR
from worldstore.health import check_world_health
from worldstore.store import WorldStore
from worldstore.tiles import (
    open_version,
    save_version_partitioned,
    save_version_partitioned_delta,
    save_version_tiled,
    _manifest_path,
)


def _entity(eid: str, x: float, y: float, z: float) -> Entity:
    return Entity(id=eid, transform={"position": {"x": x, "y": y, "z": z}})


def _grid_world(n: int = 5, spacing: float = 20.0) -> WorldIR:
    entities = {}
    for i in range(n):
        for j in range(n):
            eid = f"e-{i}-{j}"
            entities[eid] = _entity(eid, i * spacing, 0.0, j * spacing)
    return WorldIR(id="w-health", entities=entities)


def test_healthy_full_manifest_version_reports_healthy(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)

    health = check_world_health(store, "v-1")

    assert health.is_healthy
    assert health.manifest_type == "full"
    assert health.chain_depth == 0
    assert health.tile_count == 25
    assert health.missing_artifacts == ()
    assert health.corrupt_artifacts == ()
    assert health.lineage_ok is True
    assert health.parent is None
    assert health.error is None


def test_healthy_partitioned_version_reports_residual_artifact_counted(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    world.entities["unlocalized-1"] = Entity(id="unlocalized-1")
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)

    health = check_world_health(store, "v-1")

    assert health.is_healthy
    assert health.referenced_artifact_count == health.tile_count + 1  # + residual
    assert health.stale_references == ()


def test_healthy_delta_version_reports_chain_depth(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)
    moved = Entity(id="e-0-0", transform={"position": {"x": 0.0, "y": 0.0, "z": 5.0}})
    result = apply_incremental_update(world, [moved], tile_size=10.0)
    save_version_partitioned_delta(
        store, result.new_world, parent="v-1", version_id="v-2", tile_size=10.0, incremental_result=result,
    )

    health = check_world_health(store, "v-2")
    assert health.is_healthy
    assert health.manifest_type == "delta"
    assert health.chain_depth == 1
    assert health.parent == "v-1"


def test_missing_tile_artifact_detected(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
    ref = open_version(store, "v-1").manifest.tiles[0]
    shard = store._store._path_for(ref.content_hash)
    shard.unlink()

    health = check_world_health(store, "v-1")

    assert not health.is_healthy
    assert f"tile:{ref.tile_key}" in health.missing_artifacts


def test_corrupt_tile_artifact_detected(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
    ref = open_version(store, "v-1").manifest.tiles[0]
    shard = store._store._path_for(ref.content_hash)
    shard.write_bytes(b"tampered bytes")

    health = check_world_health(store, "v-1")

    assert not health.is_healthy
    assert len(health.corrupt_artifacts) == 1


def test_stale_unlocalized_reference_detected(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    world.entities["unlocalized-1"] = Entity(id="unlocalized-1")
    save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)
    manifest_path = _manifest_path(store, "v-1")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["unlocalized_entity_ids"].append("ghost-entity-not-in-residual")
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    health = check_world_health(store, "v-1")

    assert not health.is_healthy
    assert "ghost-entity-not-in-residual" in health.stale_references


def test_broken_lineage_detected(tmp_path):
    store = WorldStore(tmp_path / "store")
    world = _grid_world()
    save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
    # Simulate a corrupted manifest whose recorded parent no longer
    # exists (e.g. the parent version was manually deleted) -- a real
    # save can never produce this via any normal path (both
    # save_version()/save_version_partitioned() diff against/resolve
    # the parent first), so hand-editing the manifest is the only way
    # to exercise this detection.
    manifest_path = _manifest_path(store, "v-1")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["parent"] = "v-nonexistent"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    health = check_world_health(store, "v-1")

    assert not health.is_healthy
    assert health.lineage_ok is False


def test_unknown_version_reports_error_without_raising(tmp_path):
    store = WorldStore(tmp_path / "store")
    health = check_world_health(store, "v-does-not-exist")

    assert not health.is_healthy
    assert health.error is not None
    assert health.manifest_type == "unknown"
