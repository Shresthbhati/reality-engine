"""Tests for WorldStore's atomic-write / corruption-handling contract
(worldstore/store.py): "no partially committed world state" on crash
during save, and explicit (not raw-parser) errors on a corrupted
on-disk record.

These are additive to tests/test_world_store.py's normal-path coverage
-- this file only covers the failure/atomicity contract.
"""

from __future__ import annotations

import json

import pytest

from provenance import Provenance
from world_ir import Entity, EntityType
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore, WorldStoreError


def _world(world_id="w-1"):
    w = WorldIR(id=world_id)
    w.entities["entity-a"] = Entity(
        id="entity-a", type=EntityType.STRUCTURE, name="Shed",
        provenance=Provenance.RECONSTRUCTED, confidence=0.8,
    )
    return w


class TestAtomicWrite:
    def test_save_leaves_no_temp_files_behind(self, tmp_path):
        root = tmp_path / "worldstore"
        store = WorldStore(root)
        store.save_version(_world(), parent=None, version_id="v-1")
        leftovers = list((root / "versions").glob("*.tmp-*"))
        assert leftovers == [], "atomic write must clean up its temp file"
        leftovers_seq = list(root.glob("*.tmp-*"))
        assert leftovers_seq == []

    def test_version_record_and_sequence_are_complete_json(self, tmp_path):
        root = tmp_path / "worldstore"
        store = WorldStore(root)
        store.save_version(_world(), parent=None, version_id="v-1")
        # A real crash mid-write can't be simulated without OS-level
        # fault injection, but the atomic-write contract guarantees any
        # file that exists on disk is either the previous complete
        # version or the new complete version -- never a fragment. This
        # asserts the files that DO exist parse cleanly.
        record_path = root / "versions" / "v-1.json"
        json.loads(record_path.read_text(encoding="utf-8"))
        json.loads((root / "sequence.json").read_text(encoding="utf-8"))


class TestCorruptionIsExplicit:
    def test_corrupted_version_record_raises_worldstoreerror(self, tmp_path):
        root = tmp_path / "worldstore"
        store = WorldStore(root)
        store.save_version(_world(), parent=None, version_id="v-1")
        # Simulate on-disk corruption (e.g. a crash outside this
        # class's own atomic write path, or external tampering).
        (root / "versions" / "v-1.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(WorldStoreError, match="corrupted"):
            store.load_version("v-1")
        with pytest.raises(WorldStoreError, match="corrupted"):
            store.list_versions()

    def test_corrupted_sequence_index_raises_worldstoreerror(self, tmp_path):
        root = tmp_path / "worldstore"
        store = WorldStore(root)
        store.save_version(_world(), parent=None, version_id="v-1")
        (root / "sequence.json").write_text("[not valid json", encoding="utf-8")
        with pytest.raises(WorldStoreError, match="corrupted"):
            store.list_versions()
        with pytest.raises(WorldStoreError, match="corrupted"):
            store.save_version(_world(world_id="w-2"), parent=None, version_id="v-2")

    def test_unknown_version_still_raises_unknown_not_corrupted(self, tmp_path):
        # A missing file and a corrupted file must be distinguishable
        # failure modes -- both are explicit, but a caller retrying a
        # version id typo shouldn't be told the store is corrupted.
        root = tmp_path / "worldstore"
        store = WorldStore(root)
        with pytest.raises(WorldStoreError, match="unknown version"):
            store.load_version("v-missing")


class TestNoPartialCommitOnDiffFailure:
    def test_failed_save_does_not_register_orphan_sequence_entry(self, tmp_path):
        """If diffing against a bad parent raises before the version
        record is written, the sequence index must not gain an entry
        for a version that was never actually saved."""
        root = tmp_path / "worldstore"
        store = WorldStore(root)
        with pytest.raises(WorldStoreError, match="unknown version"):
            store.save_version(_world(), parent="v-does-not-exist", version_id="v-1")
        assert store.list_versions() == []
        assert not (root / "versions" / "v-1.json").exists()
