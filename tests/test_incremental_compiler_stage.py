"""Incremental compiler stage wiring (WORLDOS real-data city-scale
checkpoint, Task 3): proves `affected_closure` is actually wired into a
real compiler stage end to end -- new evidence -> changed entities ->
affected closure -> affected tiles -> recompile only the affected
region -> reuse every other tile's exact prior artifact -- not merely
that `apply_incremental_update` computes a diff in memory.

Before this test, nothing in the repo carried a real
`ReconstructionResult` through to `worldstore.tiles.save_version_tiled`,
so tile-artifact reuse was never exercised for the actual evidence path.
"""

from __future__ import annotations

from engine.compiler.incremental_adapter import apply_reconstruction_update_tiled
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore
from worldstore.tiles import open_version, save_version_tiled


def _far_entity(eid: str, x: float, y: float, z: float) -> Entity:
    return Entity(id=eid, transform={"position": {"x": x, "y": y, "z": z}})


def _base_world() -> WorldIR:
    # Two rooms 100 units apart -- comfortably separate 10-unit tiles,
    # so an update to one must never touch the other's tile artifact.
    entities = {
        "room-a": _far_entity("room-a", 0.0, 0.0, 0.0),
        "room-b": _far_entity("room-b", 100.0, 0.0, 0.0),
    }
    return WorldIR(id="w-two-rooms", entities=entities)


def _reconstruction_for(session_id: str) -> ReconstructionResult:
    points = [
        ReconstructedPoint(position=(i * 0.1, 0.0, 0.0), track_id=f"{session_id}-{i}", source_evidence_ids=[f"ev-{session_id}"])
        for i in range(20)
    ]
    camera = ReconstructedCameraPose(evidence_id=f"ev-{session_id}", position=(0.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0))
    return ReconstructionResult(points=points, camera_poses=[camera], registration_status="success")


def test_incremental_compiler_stage_reuses_unaffected_tile_artifacts(tmp_path):
    store = WorldStore(tmp_path / "store")
    base_world = _base_world()
    v1 = save_version_tiled(store, base_world, parent=None, tile_size=10.0)

    # New evidence updates room-a only -- room-b's tile must be reused
    # verbatim, not recompiled, since nothing connects the two rooms.
    reconstruction = _reconstruction_for("sess-1")
    compiled = apply_reconstruction_update_tiled(
        store, base_world, reconstruction,
        parent_version_id=v1.version_id,
        session_id="sess-1",
        target_entity_id="room-a",
        tile_size=10.0,
    )

    assert compiled.tiles_total == 2
    assert compiled.tiles_rebuilt == 1  # only room-a's tile
    assert compiled.tiles_reused == 1  # room-b's tile untouched

    v1_handle = open_version(store, v1.version_id)
    v2_handle = open_version(store, compiled.stored.version_id)
    room_b_key = next(t.tile_key for t in v1_handle.manifest.tiles if "room-b" in t.entity_ids)
    v1_ref = next(t for t in v1_handle.manifest.tiles if t.tile_key == room_b_key)
    v2_ref = next(t for t in v2_handle.manifest.tiles if t.tile_key == room_b_key)

    # Reuse proof: the SAME content-addressed artifact, not merely an
    # equal one recomputed from the same (unchanged) inputs.
    assert v1_ref.artifact_uri == v2_ref.artifact_uri
    assert v1_ref.content_hash == v2_ref.content_hash


def test_incremental_compiler_stage_actually_recompiles_the_changed_room(tmp_path):
    store = WorldStore(tmp_path / "store")
    base_world = _base_world()
    v1 = save_version_tiled(store, base_world, parent=None, tile_size=10.0)

    reconstruction = _reconstruction_for("sess-1")
    compiled = apply_reconstruction_update_tiled(
        store, base_world, reconstruction,
        parent_version_id=v1.version_id,
        session_id="sess-1",
        target_entity_id="room-a",
        tile_size=10.0,
    )

    v1_handle = open_version(store, v1.version_id)
    v2_handle = open_version(store, compiled.stored.version_id)
    room_a_key = next(t.tile_key for t in v1_handle.manifest.tiles if "room-a" in t.entity_ids)
    v1_ref = next(t for t in v1_handle.manifest.tiles if t.tile_key == room_a_key)
    v2_ref = next(t for t in v2_handle.manifest.tiles if t.tile_key == room_a_key)

    # room-a's own entity content changed (new geometry/observations),
    # so its tile's artifact must differ -- reuse would be a bug here.
    assert v1_ref.content_hash != v2_ref.content_hash

    loaded = v2_handle.load_tile(room_a_key)
    assert "room-a" in loaded
    assert loaded["room-a"].custom_properties["point_count"] == 20


def test_second_generation_update_still_reuses_across_two_prior_versions(tmp_path):
    """Sabotage-style check: a THIRD version's reuse must be computed
    against its immediate parent (v2), not accidentally fall back to v1
    or rebuild everything because reuse-chaining broke."""
    store = WorldStore(tmp_path / "store")
    base_world = _base_world()
    v1 = save_version_tiled(store, base_world, parent=None, tile_size=10.0)

    compiled_v2 = apply_reconstruction_update_tiled(
        store, base_world, _reconstruction_for("sess-1"),
        parent_version_id=v1.version_id, session_id="sess-1",
        target_entity_id="room-a", tile_size=10.0,
    )

    compiled_v3 = apply_reconstruction_update_tiled(
        store, compiled_v2.result.new_world, _reconstruction_for("sess-2"),
        parent_version_id=compiled_v2.stored.version_id, session_id="sess-2",
        target_entity_id="room-a", tile_size=10.0,
    )

    assert compiled_v3.tiles_rebuilt == 1
    assert compiled_v3.tiles_reused == 1

    v2_handle = open_version(store, compiled_v2.stored.version_id)
    v3_handle = open_version(store, compiled_v3.stored.version_id)
    room_b_key = next(t.tile_key for t in v2_handle.manifest.tiles if "room-b" in t.entity_ids)
    v2_ref = next(t for t in v2_handle.manifest.tiles if t.tile_key == room_b_key)
    v3_ref = next(t for t in v3_handle.manifest.tiles if t.tile_key == room_b_key)
    assert v2_ref.artifact_uri == v3_ref.artifact_uri
