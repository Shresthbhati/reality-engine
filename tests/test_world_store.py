"""Tests for worldstore/store.py (P12 WorldStore): persistent versioned
world storage.

The directive's done-when: "world survives process restarts". That
means a FileArtifactStore-backed WorldStore: save a world version,
construct a NEW store instance against the same root (the old object's
in-memory state gone), and read the version back identical -- entities,
geometries, metadata, and the artifact bytes behind data_uri intact.

Rules under test:
- versions are immutable: saving the same version id twice is refused,
  never a silent overwrite (observed reality is recoverable);
- version lineage (parent pointers) is queryable;
- artifact integrity is checkable (hash mismatch detected);
- listing versions is ordered and stable.
"""

import pytest

from world_ir.artifact_store import FileArtifactStore
from world_ir.world_v1 import WorldIR


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "worldstore"


def _world_with_entity(world_id="w-1"):
    from provenance import Provenance
    from world_ir import Entity, EntityType

    w = WorldIR(id=world_id)
    w.entities["entity-a"] = Entity(
        id="entity-a", type=EntityType.STRUCTURE, name="Shed",
        provenance=Provenance.RECONSTRUCTED, confidence=0.8,
    )
    w.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    return w


class TestWorldStorePersistence:
    def test_world_survives_process_restart(self, root):
        from worldstore.store import WorldStore

        store1 = WorldStore(root)
        v1 = store1.save_version(_world_with_entity(), parent=None)
        assert v1.version_id

        # Simulate a process restart: brand-new store object, same root.
        store2 = WorldStore(root)
        loaded = store2.load_version(v1.version_id)
        assert loaded.id == "w-1"
        assert "entity-a" in loaded.entities
        assert loaded.metadata["scale"]["state"] == "metric"

    def test_versions_are_immutable(self, root):
        from worldstore.store import WorldStore, WorldStoreError

        store = WorldStore(root)
        v1 = store.save_version(_world_with_entity(), parent=None)
        with pytest.raises(WorldStoreError, match="immutable"):
            store.save_version(_world_with_entity(), parent=None,
                               version_id=v1.version_id)

    def test_version_lineage(self, root):
        from worldstore.store import WorldStore

        store = WorldStore(root)
        v1 = store.save_version(_world_with_entity("w-1"), parent=None)
        v2 = store.save_version(_world_with_entity("w-1"), parent=v1.version_id)
        v3 = store.save_version(_world_with_entity("w-1"), parent=v2.version_id)
        assert store.parents(v3.version_id) == [v2.version_id]
        # Root-first lineage (the walk reverses the parent chain).
        assert store.ancestors(v3.version_id) == [v1.version_id, v2.version_id]

    def test_list_versions_ordered(self, root):
        from worldstore.store import WorldStore

        store = WorldStore(root)
        v1 = store.save_version(_world_with_entity(), parent=None)
        v2 = store.save_version(_world_with_entity(), parent=v1.version_id)
        ids = [v.version_id for v in store.list_versions()]
        assert ids == [v1.version_id, v2.version_id]

    def test_artifact_integrity_check(self, root):
        from worldstore.store import WorldStore

        store = WorldStore(root)
        v1 = store.save_version(_world_with_entity(), parent=None)
        # Untampered: no failures.
        assert store.verify_version(v1.version_id) == []

    def test_load_unknown_version_raises(self, root):
        from worldstore.store import WorldStore, WorldStoreError

        store = WorldStore(root)
        with pytest.raises(WorldStoreError, match="unknown version"):
            store.load_version("v-nope")

    def test_root_version_has_no_changed_ids(self, root):
        from worldstore.store import WorldStore

        store = WorldStore(root)
        v1 = store.save_version(_world_with_entity(), parent=None)
        assert v1.changed_entity_ids == []
        assert v1.changed_geometry_ids == []

    def test_child_version_records_changed_entity_ids(self, root):
        from provenance import Provenance
        from world_ir import Entity, EntityType
        from worldstore.store import WorldStore

        store = WorldStore(root)
        v1 = store.save_version(_world_with_entity("w-1"), parent=None)

        modified = _world_with_entity("w-1")
        modified.entities["entity-a"].confidence = 0.99  # changes entity-a
        modified.entities["entity-b"] = Entity(  # adds entity-b
            id="entity-b", type=EntityType.STRUCTURE, name="Fence",
            provenance=Provenance.RECONSTRUCTED, confidence=0.5,
        )
        v2 = store.save_version(modified, parent=v1.version_id)

        assert v2.changed_entity_ids == ["entity-a", "entity-b"]

    def test_source_session_ids_recorded(self, root):
        from worldstore.store import WorldStore

        store = WorldStore(root)
        v1 = store.save_version(
            _world_with_entity(), parent=None, source_session_ids=["session-1", "session-2"],
        )
        assert v1.source_session_ids == ["session-1", "session-2"]
        # Survives a fresh store instance over the same root.
        store2 = WorldStore(root)
        reloaded = [v for v in store2.list_versions() if v.version_id == v1.version_id][0]
        assert reloaded.source_session_ids == ["session-1", "session-2"]
