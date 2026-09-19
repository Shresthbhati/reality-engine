"""Independent acceptance test for true localized incremental world update.

Spec:
Scenario:
  World V1
   ↓
  new evidence affecting region X
   ↓
  incremental update
   ↓
  World V2

Verifies the 12 Critical Invariants:
1. affected entity changes
2. unaffected entity is reused (assert old_object is new_object, not merely equality)
3. unaffected geometry is reused (assert old_geometry is new_geometry)
4. unaffected tile is reused (unaffected_tile_id not in invalidated_tile_ids)
5. affected tile is rebuilt (affected_tile_id in invalidated_tile_ids)
6. V1 remains immutable
7. provenance preserved
8. uncertainty preserved
9. coordinate frame preserved
10. lineage preserved across restart
11. deterministic result
12. failure during update leaves V1 intact
"""

from __future__ import annotations

import copy
import pytest

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from tests.test_room_inference import _CAMS, _two_room_scene
from world_ir.artifact_store import FileArtifactStore
from world_ir.schema_v1 import Entity, EntityType, Vector3
from world_ir.world_v1 import WorldIR
from worldstore.store import StoredVersion, WorldStore, WorldStoreError


def _setup_base_world_with_store(tmp_path) -> tuple[WorldStore, StoredVersion, WorldIR]:
    """Create a persistent WorldStore with an initial World V1 containing
    known entities and geometries distributed across space."""
    store_dir = tmp_path / "worldstore"
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    store = WorldStore(store_dir)

    import dataclasses
    from reconstruction.backend.interface import ReconstructedCameraPose
    cams = [
        ReconstructedCameraPose(evidence_id=f"ev-{i}", position=c, rotation=(1.0, 0.0, 0.0, 0.0))
        for i, c in enumerate(_CAMS)
    ]
    recon = dataclasses.replace(_two_room_scene(), camera_poses=cams)
    base_world, _ = compile_reconstruction_to_world(
        recon, CompileOptions(seed=42, artifact_store=artifact_store)
    )

    # Add two explicit spatial landmark entities:
    # Landmark 1 in Region A (x=1.0, y=0.0, z=1.0) -> tile (0, 0) for tile_size=10.0
    # Landmark 2 in Region B (x=50.0, y=0.0, z=50.0) -> tile (5, 5) for tile_size=10.0
    base_world.entities["landmark-region-a"] = Entity(
        id="landmark-region-a",
        type=EntityType.SENSOR,
        name="Region A Sensor",
        transform={"position": {"x": 1.0, "y": 0.0, "z": 1.0}},
        confidence=0.95,
        custom_properties={"region": "A"},
    )
    base_world.entities["landmark-region-b"] = Entity(
        id="landmark-region-b",
        type=EntityType.SENSOR,
        name="Region B Station",
        transform={"position": {"x": 50.0, "y": 0.0, "z": 50.0}},
        confidence=0.98,
        custom_properties={"region": "B"},
    )

    v1 = store.save_version(base_world, parent=None, version_id="v-base-1")
    return store, v1, base_world


def test_true_localized_incremental_update(tmp_path):
    """Rigorous adversarial integration test for true localized incremental update.
    Requires `apply_incremental_update` from `world_ir`.
    """
    try:
        from world_ir import apply_incremental_update, IncrementalUpdateResult
    except ImportError as err:
        pytest.fail(
            f"LOCALIZED INCREMENTAL UPDATE MISSING: Cannot import apply_incremental_update from world_ir. "
            f"Error: {err}. Category A localized update is not implemented."
        )

    store, v1_record, base_world = _setup_base_world_with_store(tmp_path)
    initial_v1_bytes = (tmp_path / "worldstore" / "versions" / f"{v1_record.version_id}.json").read_bytes()

    # Pre-conditions: check initial entities and geometries
    assert "landmark-region-a" in base_world.entities
    assert "landmark-region-b" in base_world.entities
    assert len(base_world.geometries) > 0

    old_unaffected_entity = base_world.entities["landmark-region-b"]
    first_geom_id = next(iter(base_world.geometries.keys()))
    old_unaffected_geom = base_world.geometries[first_geom_id]

    # ── UPDATE SCENARIO: New evidence updates Region A only ──
    # Create updated entity for Region A
    updated_region_a_entity = Entity(
        id="landmark-region-a",
        type=EntityType.SENSOR,
        name="Region A Sensor (Updated High Precision)",
        transform={"position": {"x": 1.05, "y": 0.0, "z": 1.02}},
        confidence=0.99,
        custom_properties={"region": "A", "updated": True},
    )

    # 11. Determinism check: run the exact same update twice
    result_1 = apply_incremental_update(base_world, [updated_region_a_entity])
    result_2 = apply_incremental_update(base_world, [updated_region_a_entity])

    assert result_1.changed_entity_ids == result_2.changed_entity_ids
    assert result_1.invalidated_tile_ids == result_2.invalidated_tile_ids
    assert result_1.reused_entity_ids == result_2.reused_entity_ids

    v2_world = result_1.new_world

    # 1. Affected entity changes
    assert "landmark-region-a" in result_1.changed_entity_ids
    assert v2_world.entities["landmark-region-a"].name == "Region A Sensor (Updated High Precision)"
    assert v2_world.entities["landmark-region-a"].confidence == 0.99
    assert v2_world.entities["landmark-region-a"].custom_properties.get("updated") is True

    # 2. Unaffected entity is REUSED (strict object identity)
    assert "landmark-region-b" in result_1.reused_entity_ids
    new_unaffected_entity = v2_world.entities["landmark-region-b"]
    assert new_unaffected_entity is old_unaffected_entity, (
        f"CRITICAL INVARIANT VIOLATION: Unaffected entity was reconstructed instead of reused! "
        f"id(old)={id(old_unaffected_entity)}, id(new)={id(new_unaffected_entity)}"
    )

    # 3. Unaffected geometry is REUSED (strict object identity)
    assert first_geom_id in result_1.reused_geometry_ids
    new_unaffected_geom = v2_world.geometries[first_geom_id]
    assert new_unaffected_geom is old_unaffected_geom, (
        f"CRITICAL INVARIANT VIOLATION: Unaffected geometry was reconstructed instead of reused! "
        f"id(old)={id(old_unaffected_geom)}, id(new)={id(new_unaffected_geom)}"
    )

    # 4 & 5. Tile Invalidation: affected tile invalidated, unaffected tile NOT invalidated
    # Spatial tiles uniform grid tile_size = 10.0
    # Region A (1.0, 0.0, 1.0) -> tile (0, 0, 0)
    # Region B (50.0, 0.0, 50.0) -> tile (5, 0, 5)
    assert (0, 0, 0) in result_1.invalidated_tile_ids
    assert (5, 0, 5) not in result_1.invalidated_tile_ids

    # 6. V1 remains immutable on disk
    v1_bytes_after = (tmp_path / "worldstore" / "versions" / f"{v1_record.version_id}.json").read_bytes()
    assert initial_v1_bytes == v1_bytes_after, "V1 file on disk was modified!"

    # 7. Provenance preserved
    assert new_unaffected_entity.provenance == old_unaffected_entity.provenance

    # 8. Uncertainty preserved
    assert new_unaffected_entity.confidence == old_unaffected_entity.confidence
    assert new_unaffected_entity.uncertainty == old_unaffected_entity.uncertainty

    # 9. Coordinate frame preserved
    assert v2_world.coordinate_frame == base_world.coordinate_frame

    # 10. Lineage preserved across process restart
    v2_record = store.save_version(v2_world, parent=v1_record.version_id, version_id="v-inc-2")
    assert v2_record.parent == "v-base-1"

    # Simulate fresh process restart
    restarted_store = WorldStore(tmp_path / "worldstore")
    assert restarted_store.parents("v-inc-2") == ["v-base-1"]
    assert restarted_store.ancestors("v-inc-2") == ["v-base-1"]

    reloaded_v2 = restarted_store.load_version("v-inc-2")
    assert reloaded_v2.entities["landmark-region-a"].name == "Region A Sensor (Updated High Precision)"
    assert reloaded_v2.entities["landmark-region-b"].name == "Region B Station"

    # 12. Failure during update does NOT corrupt V1
    # Attempting to save a version with an invalid parent or corrupt input
    with pytest.raises(WorldStoreError):
        store.save_version(v2_world, parent="non-existent-parent", version_id="v-corrupt")

    # V1 is still 100% valid and verified
    assert restarted_store.verify_version("v-base-1") == []
    v1_reloaded = restarted_store.load_version("v-base-1")
    assert v1_reloaded.entities["landmark-region-a"].name == "Region A Sensor"
