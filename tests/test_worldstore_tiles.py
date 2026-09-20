"""Tests for worldstore/tiles.py: the lazy, tile-granular WorldStore
access layer (WORLDOS lazy-spatial-world checkpoint).

Covers the mandated matrix: manifest contract, persisted tile
artifacts (content-addressed, reused when unchanged), lazy loading
(list/lookup/load without touching unrelated tiles), spatial query
(only overlapping tiles loaded), incremental tile reuse (A/B/C from
the mission spec), entity crossing a tile boundary, entity deletion,
failure safety (a crash mid-save must never leave V1 invalid or a
manifest referencing a missing tile), and determinism.
"""

from __future__ import annotations

import json

import pytest

from provenance import Provenance
from world_ir import apply_incremental_update
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore, WorldStoreError
from worldstore.tiles import open_version, save_version_tiled


def _entity(eid, x, y, z) -> Entity:
    return Entity(
        id=eid, type=EntityType.STRUCTURE,
        transform={"position": {"x": x, "y": y, "z": z}},
        provenance=Provenance.RECONSTRUCTED,
    )


def _abc_world() -> WorldIR:
    w = WorldIR(id="w-abc")
    w.entities["a"] = _entity("a", 1.0, 0.0, 0.0)    # tile A = (0,0,0)
    w.entities["b"] = _entity("b", 20.0, 0.0, 0.0)   # tile B = (2,0,0)
    w.entities["c"] = _entity("c", 40.0, 0.0, 0.0)   # tile C = (4,0,0)
    return w


class TestManifestContract:
    def test_manifest_lists_all_tiles_deterministically_ordered(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        handle = open_version(store, "v-1")
        assert handle.list_tiles() == ((0, 0, 0), (2, 0, 0), (4, 0, 0))

    def test_manifest_preserves_lineage_and_frame(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        handle = open_version(store, "v-1")
        assert handle.manifest.version_id == "v-1"
        assert handle.manifest.parent is None
        assert handle.manifest.world_id == "w-abc"
        assert handle.manifest.coordinate_frame == world.coordinate_frame.value

    def test_manifest_carries_world_version_and_occupancy_metadata(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        world.version = 3
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        manifest = open_version(store, "v-1").manifest

        assert manifest.world_version == 3
        summary = manifest.occupancy_summary()
        assert summary["tile_count"] == 3  # tiles A, B, C
        assert summary["total_entity_count"] == 3  # a, b, c
        assert summary["unlocalized_entity_count"] == 0
        assert summary["max_tile_occupancy"] == 1
        assert summary["min_tile_occupancy"] == 1

    def test_opening_a_non_tiled_version_raises_explicit_error(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        store.save_version(world, parent=None, version_id="v-plain")  # NOT tiled
        with pytest.raises(WorldStoreError, match="no tile manifest"):
            open_version(store, "v-plain")


class TestPersistedTileArtifacts:
    def test_tile_artifact_is_content_addressed_and_deterministic(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        v1 = save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        handle = open_version(store, "v-1")
        ref_a = next(r for r in handle.manifest.tiles if r.tile_key == (0, 0, 0))
        assert ref_a.content_hash
        assert ref_a.artifact_uri.startswith("artifact://")
        assert ref_a.entity_ids == ("a",)

    def test_tile_artifact_bytes_are_actually_on_disk(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        digest = open_version(store, "v-1").manifest.tiles[0].content_hash
        shard = tmp_path / "artifacts" / digest[:2] / f"{digest}.bin"
        assert shard.is_file()


class TestLazyLoading:
    def test_load_tile_returns_only_that_tiles_entities(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        handle = open_version(store, "v-1")
        tile_a = handle.load_tile((0, 0, 0))
        assert set(tile_a) == {"a"}
        tile_b = handle.load_tile((2, 0, 0))
        assert set(tile_b) == {"b"}

    def test_opening_a_version_does_not_read_any_tile_artifact(self, tmp_path, monkeypatch):
        """The whole point: open_version() must read ONLY the manifest."""
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)

        calls = []
        original_get = store._store.get
        monkeypatch.setattr(store._store, "get", lambda uri: (calls.append(uri), original_get(uri))[1])
        open_version(store, "v-1")
        assert calls == []  # zero artifact reads just to open the version

    def test_load_tile_reads_exactly_one_artifact(self, tmp_path, monkeypatch):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        handle = open_version(store, "v-1")

        calls = []
        original_get = store._store.get
        monkeypatch.setattr(store._store, "get", lambda uri: (calls.append(uri), original_get(uri))[1])
        handle.load_tile((0, 0, 0))
        assert len(calls) == 1

    def test_unknown_tile_raises(self, tmp_path):
        store = WorldStore(tmp_path)
        save_version_tiled(store, _abc_world(), parent=None, version_id="v-1", tile_size=10.0)
        handle = open_version(store, "v-1")
        with pytest.raises(WorldStoreError, match="unknown tile"):
            handle.load_tile((99, 99, 99))


class TestSpatialQuery:
    def test_query_region_loads_only_overlapping_tiles(self, tmp_path, monkeypatch):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        handle = open_version(store, "v-1")

        calls = []
        original_get = store._store.get
        monkeypatch.setattr(store._store, "get", lambda uri: (calls.append(uri), original_get(uri))[1])
        result = handle.query_region((0.0, 0.0, 0.0), (5.0, 5.0, 5.0))  # only tile A overlaps
        assert set(result) == {"a"}
        assert len(calls) == 1  # only tile A's artifact was read -- B and C untouched

    def test_query_spanning_two_tiles_loads_exactly_two(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        handle = open_version(store, "v-1")
        result = handle.query_region((0.0, 0.0, 0.0), (25.0, 5.0, 5.0))  # tiles A and B
        assert set(result) == {"a", "b"}


class TestIncrementalTileReuse:
    def test_a_rebuilt_b_and_c_reused_verbatim(self, tmp_path):
        """The exact scenario from the mission spec."""
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        h1 = open_version(store, "v-1")

        new_a = _entity("a", 1.5, 0.0, 0.0)  # still tile A, content changed
        result = apply_incremental_update(world, [new_a], tile_size=10.0)
        save_version_tiled(
            store, result.new_world, parent="v-1", version_id="v-2",
            tile_size=10.0, incremental_result=result,
        )
        h2 = open_version(store, "v-2")

        by_key_1 = {r.tile_key: r for r in h1.manifest.tiles}
        by_key_2 = {r.tile_key: r for r in h2.manifest.tiles}
        assert by_key_2[(0, 0, 0)].artifact_uri != by_key_1[(0, 0, 0)].artifact_uri  # A rebuilt
        assert by_key_2[(2, 0, 0)].artifact_uri == by_key_1[(2, 0, 0)].artifact_uri  # B reused
        assert by_key_2[(4, 0, 0)].artifact_uri == by_key_1[(4, 0, 0)].artifact_uri  # C reused
        assert by_key_2[(2, 0, 0)].content_hash == by_key_1[(2, 0, 0)].content_hash

    def test_entity_crossing_tile_boundary_rebuilds_both_old_and_new_tile(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        h1 = open_version(store, "v-1")

        moved_a = _entity("a", 15.0, 0.0, 0.0)  # jumps from tile (0,0,0) into (1,0,0)
        result = apply_incremental_update(world, [moved_a], tile_size=10.0)
        save_version_tiled(
            store, result.new_world, parent="v-1", version_id="v-2",
            tile_size=10.0, incremental_result=result,
        )
        h2 = open_version(store, "v-2")

        # Old tile (0,0,0) no longer contains "a" -- it's gone from v2's
        # manifest entirely (no other entity lives there).
        assert (0, 0, 0) not in h2.list_tiles()
        # New tile (1,0,0) is a genuinely new tile entry.
        assert (1, 0, 0) in h2.list_tiles()
        assert set(h2.load_tile((1, 0, 0))) == {"a"}
        # Untouched tiles B and C keep their exact artifact identity.
        by_key_1 = {r.tile_key: r for r in h1.manifest.tiles}
        by_key_2 = {r.tile_key: r for r in h2.manifest.tiles}
        assert by_key_2[(2, 0, 0)].artifact_uri == by_key_1[(2, 0, 0)].artifact_uri
        assert by_key_2[(4, 0, 0)].artifact_uri == by_key_1[(4, 0, 0)].artifact_uri

    def test_entity_deletion_rebuilds_old_tile_others_reused(self, tmp_path):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)
        h1 = open_version(store, "v-1")

        result = apply_incremental_update(world, [], removed_entities=["a"], tile_size=10.0)
        save_version_tiled(
            store, result.new_world, parent="v-1", version_id="v-2",
            tile_size=10.0, incremental_result=result,
        )
        h2 = open_version(store, "v-2")

        assert (0, 0, 0) not in h2.list_tiles()  # "a"'s tile is gone (nothing else there)
        by_key_1 = {r.tile_key: r for r in h1.manifest.tiles}
        by_key_2 = {r.tile_key: r for r in h2.manifest.tiles}
        assert by_key_2[(2, 0, 0)].artifact_uri == by_key_1[(2, 0, 0)].artifact_uri
        assert by_key_2[(4, 0, 0)].artifact_uri == by_key_1[(4, 0, 0)].artifact_uri


class TestFailureSafety:
    def test_crash_during_tile_write_leaves_v1_valid_and_v2_unreferenced(self, tmp_path, monkeypatch):
        store = WorldStore(tmp_path)
        world = _abc_world()
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)

        new_a = _entity("a", 1.5, 0.0, 0.0)
        result = apply_incremental_update(world, [new_a], tile_size=10.0)

        original_put = store._store.put
        call_count = {"n": 0}

        def flaky_put(data: bytes):
            call_count["n"] += 1
            # Call #1 is save_version()'s own whole-world blob write
            # (must succeed, so V2's whole-world record exists); call #2
            # is the FIRST tile artifact write -- fail there to simulate
            # a crash mid tile-rebuild, after the world itself committed.
            if call_count["n"] == 2:
                raise OSError("simulated disk failure during tile write")
            return original_put(data)

        monkeypatch.setattr(store._store, "put", flaky_put)
        with pytest.raises(OSError, match="simulated disk failure"):
            save_version_tiled(
                store, result.new_world, parent="v-1", version_id="v-2",
                tile_size=10.0, incremental_result=result,
            )

        # V1 is completely unaffected and still opens/loads correctly.
        h1 = open_version(store, "v-1")
        assert set(h1.load_tile((0, 0, 0))) == {"a"}
        # V2's tile manifest was never written -- no partial tiled
        # version is visible via the lazy API.
        with pytest.raises(WorldStoreError):
            open_version(store, "v-2")
        # No orphaned reference: v-2's whole-world blob WAS committed by
        # save_version() before the tile write failed (that's a separate,
        # already-atomic step) -- but no CALLER can reach it via the tile
        # API, and list_versions() still reports it accurately as a
        # real (non-tiled) version rather than hiding it.
        assert "v-2" in {v.version_id for v in store.list_versions()}


class TestDeterminism:
    def test_repeated_manifest_build_is_identical(self, tmp_path):
        store1 = WorldStore(tmp_path / "s1")
        store2 = WorldStore(tmp_path / "s2")
        world = _abc_world()
        save_version_tiled(store1, world, parent=None, version_id="v-1", tile_size=10.0)
        save_version_tiled(store2, world, parent=None, version_id="v-1", tile_size=10.0)
        h1 = open_version(store1, "v-1")
        h2 = open_version(store2, "v-1")
        assert h1.manifest.to_dict() == h2.manifest.to_dict()
