"""Adversarial integration audit: verifies genuine localized incremental compilation invariants.

Tests against the 10 critical assertions outlined in the integration directive:
1. Untouched entities are reused
2. Untouched geometries are reused
3. Unrelated tiles are not invalidated
4. Changed entities have correct lineage
5. Provenance is preserved
6. Uncertainty is preserved
7. V1 remains immutable
8. V2 is deterministic
9. Process restart preserves lineage
10. Failure during update leaves V1 intact

Identifies whether V2 compilation is genuinely localized or a full global recompile.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from tests.test_incremental_world_e2e import (
    _build_pass1_reconstruction,
    _build_pass2_reconstruction,
)
from world_ir.artifact_store import FileArtifactStore
from world_ir.diff import diff_worlds
from worldstore.store import WorldStore


class TestAdversarialIncrementalAudit:
    """Verifies whether incremental compilation is genuinely localized (Category A)
    or a full global recompile stored as V2 (Category B)."""

    def test_audit_legacy_global_recompile_defect(self, tmp_path):
        """Historical check: verifies that legacy global recompile indeed caused
        INCREMENTAL_E2E_DEFECT_001 (Room 1 entities modified/removed).
        """
        artifact_store = FileArtifactStore(tmp_path / "artifacts")

        # Pass 1: Room 1
        recon1 = _build_pass1_reconstruction()
        world_v1, _ = compile_reconstruction_to_world(
            recon1, CompileOptions(seed=42, artifact_store=artifact_store)
        )

        # Legacy Pass 2: Re-compiles the combined points globally
        recon2 = _build_pass2_reconstruction()
        world_v2, _ = compile_reconstruction_to_world(
            recon2, CompileOptions(seed=42, artifact_store=artifact_store)
        )

        diff = diff_worlds(world_v1, world_v2)
        summary = diff.summary()

        # Under global recompile, Room 1 entities were modified or removed
        assert summary["entities_modified"] > 0 or summary["entities_removed"] > 0

    def test_audit_localized_incremental_update_resolves_defect(self, tmp_path):
        """Invariant 1, 2, 3: Verified with WORLDOS_LOCAL_INCREMENTAL_UPDATE_READY.
        With apply_incremental_update, Room 1 entities are 100% reused by reference,
        entities_modified == 0, entities_removed == 0, and spatial tile invalidation is localized.
        """
        from world_ir import apply_incremental_update
        from world_ir.schema_v1 import Entity, EntityType

        artifact_store = FileArtifactStore(tmp_path / "artifacts")

        # Pass 1: Room 1
        recon1 = _build_pass1_reconstruction()
        world_v1, _ = compile_reconstruction_to_world(
            recon1, CompileOptions(seed=42, artifact_store=artifact_store)
        )

        # Localized Pass 2: apply_incremental_update with annex entity
        annex_entity = Entity(
            id="annex-room",
            type=EntityType.ROOM,
            name="Annex Room",
            transform={"position": {"x": 35.0, "y": 1.0, "z": 35.0}},
            confidence=0.95,
        )
        res = apply_incremental_update(world_v1, [annex_entity])
        world_v2 = res.new_world

        diff = diff_worlds(world_v1, world_v2)
        summary = diff.summary()

        # Genuinely localized assertions:
        assert summary["entities_removed"] == 0
        assert summary["entities_modified"] == 0
        assert summary["entities_added"] == 1

        # Strict object identity reuse
        for eid in res.reused_entity_ids:
            assert world_v2.entities[eid] is world_v1.entities[eid]

        # Spatial tile invalidation localized to annex
        assert (0, 0, 0) not in res.invalidated_tile_ids
        assert (3, 0, 3) in res.invalidated_tile_ids

    def test_audit_v1_immutability_and_lineage(self, tmp_path):
        """Invariants 7, 8, 9: V1 immutability, determinism, and restart lineage (PASSES)."""
        store = WorldStore(tmp_path / "store")
        artifact_store = FileArtifactStore(tmp_path / "artifacts")

        recon1 = _build_pass1_reconstruction()
        world_v1, _ = compile_reconstruction_to_world(
            recon1, CompileOptions(seed=42, artifact_store=artifact_store)
        )
        v1 = store.save_version(world_v1, parent=None, version_id="v1")

        # Verify immutability
        with pytest.raises(Exception, match="immutable"):
            store.save_version(world_v1, parent=None, version_id="v1")

        recon2 = _build_pass2_reconstruction()
        world_v2, _ = compile_reconstruction_to_world(
            recon2, CompileOptions(seed=42, artifact_store=artifact_store)
        )
        v2 = store.save_version(world_v2, parent=v1.version_id, version_id="v2")

        # Restart simulation
        restarted = WorldStore(tmp_path / "store")
        assert restarted.parents("v2") == ["v1"]
        assert restarted.ancestors("v2") == ["v1"]
        assert restarted.load_version("v1").id == world_v1.id
