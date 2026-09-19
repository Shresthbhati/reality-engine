"""Tests for world_ir.incremental.apply_incremental_update -- the
localized WorldIR update primitive (World V1 + changed entities/
geometries -> World V2, without rebuilding anything untouched).

The critical semantic requirement: unchanged entities are NOT rebuilt.
Identity assertions (`is`, not `==`) are used everywhere the contract
promises object reuse -- equality after a full rebuild would pass
these tests too, which is exactly what they must NOT tolerate.
"""

from __future__ import annotations

from provenance import Provenance, Uncertainty
from world_ir import apply_incremental_update
from world_ir.schema_v1 import (
    Entity, EntityType, Geometry, GeometryType, Relationship,
    RelationshipKind, Vector3,
)
from world_ir.world_v1 import WorldIR
from world_ir.diff import diff_worlds
from worldstore.store import WorldStore, WorldStoreError


def _entity(eid, x=0.0, y=0.0, z=0.0, **kwargs) -> Entity:
    return Entity(
        id=eid, type=kwargs.pop("type", EntityType.STRUCTURE),
        transform={"position": {"x": x, "y": y, "z": z}},
        provenance=kwargs.pop("provenance", Provenance.RECONSTRUCTED),
        **kwargs,
    )


def _base_world() -> WorldIR:
    world = WorldIR(id="w-city")
    world.entities["wall-1"] = _entity(
        "wall-1", 1.0, 0.0, 0.0,
        relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-1")],
    )
    world.entities["room-1"] = _entity("room-1", 1.0, 0.0, 0.0, type=EntityType.ROOM)
    world.entities["far-tree"] = _entity("far-tree", 50.0, 0.0, 0.0, type=EntityType.VEGETATION)
    world.geometries["geom-wall-1"] = Geometry(
        id="geom-wall-1", type=GeometryType.PLANE,
        bounds_min=Vector3(0.0, 0.0, 0.0), bounds_max=Vector3(1.0, 3.0, 0.2),
        provenance=Provenance.OBSERVED,
    )
    world.entities["wall-1"].geometry_ids = ["geom-wall-1"]
    return world


class TestSingleChangedEntity:
    def test_changed_entity_is_a_new_object(self):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])
        assert result.new_world.entities["wall-1"] is new_wall
        assert result.changed_entity_ids == frozenset({"wall-1"})


class TestRelationshipDependentEntity:
    def test_container_is_affected_but_not_mutated(self):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])
        # room-1 is PART_OF-connected to wall-1 -> affected, but this
        # function has no derivation logic, so it is NOT replaced.
        assert "room-1" in result.affected_entity_ids
        assert result.new_world.entities["room-1"] is base.entities["room-1"]


class TestGeometryOnlyChange:
    def test_geometry_change_marks_owning_entity_affected_without_mutating_it(self):
        base = _base_world()
        new_geom = Geometry(
            id="geom-wall-1", type=GeometryType.PLANE,
            bounds_min=Vector3(0.0, 0.0, 0.0), bounds_max=Vector3(1.0, 3.0, 0.3),
            provenance=Provenance.OBSERVED,
        )
        result = apply_incremental_update(base, [], updated_geometries=[new_geom])
        assert result.new_world.geometries["geom-wall-1"] is new_geom
        assert result.changed_geometry_ids == frozenset({"geom-wall-1"})
        # wall-1 references the changed geometry -> affected (its own
        # object is untouched -- no entity-level change was supplied).
        assert "wall-1" in result.affected_entity_ids
        assert result.new_world.entities["wall-1"] is base.entities["wall-1"]


class TestUnrelatedEntityIdentity:
    def test_unrelated_entity_is_the_same_object(self):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])
        assert result.new_world.entities["far-tree"] is base.entities["far-tree"]
        assert "far-tree" in result.reused_entity_ids
        assert "far-tree" not in result.affected_entity_ids

    def test_unrelated_geometry_is_the_same_object(self):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])
        assert result.new_world.geometries["geom-wall-1"] is base.geometries["geom-wall-1"]
        assert "geom-wall-1" in result.reused_geometry_ids


class TestUnrelatedTileReusable:
    def test_far_tile_is_not_invalidated(self):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)  # stays in the same 10-unit tile
        result = apply_incremental_update(base, [new_wall], tile_size=10.0)
        # far-tree at x=50 is a different tile from wall-1/room-1 at x=1.
        far_tile = (5, 0, 0)
        assert far_tile not in result.invalidated_tile_ids
        assert (0, 0, 0) in result.invalidated_tile_ids


class TestDeterministicInvalidation:
    def test_same_update_produces_same_invalidated_tiles(self):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        r1 = apply_incremental_update(base, [new_wall])
        r2 = apply_incremental_update(base, [new_wall])
        assert r1.invalidated_tile_ids == r2.invalidated_tile_ids
        assert r1.affected_entity_ids == r2.affected_entity_ids

    def test_moved_entity_invalidates_both_old_and_new_tile(self):
        base = _base_world()
        # wall-1 jumps to a distant tile.
        moved_wall = _entity("wall-1", 40.0, 0.0, 0.0)
        result = apply_incremental_update(base, [moved_wall], tile_size=10.0)
        assert (0, 0, 0) in result.invalidated_tile_ids  # old tile
        assert (4, 0, 0) in result.invalidated_tile_ids  # new tile


class TestProvenancePreserved:
    def test_unchanged_entity_provenance_intact(self):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])
        assert result.new_world.entities["far-tree"].provenance == Provenance.RECONSTRUCTED

    def test_changed_entity_carries_its_own_supplied_provenance(self):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0, provenance=Provenance.OBSERVED)
        result = apply_incremental_update(base, [new_wall])
        assert result.new_world.entities["wall-1"].provenance == Provenance.OBSERVED


class TestUncertaintyPreserved:
    def test_unchanged_geometry_uncertainty_untouched(self):
        base = _base_world()
        base.geometries["geom-wall-1"].observations = []  # baseline, no mutation expected
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])
        # Same object -> whatever measured state it carried survives untouched.
        assert result.new_world.geometries["geom-wall-1"] is base.geometries["geom-wall-1"]


class TestVersionLineage:
    def test_version_increments(self):
        base = _base_world()
        base.version = 3
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])
        assert result.new_world.version == 4
        assert result.new_world.id == base.id  # same world, next version

    def test_worldstore_diff_agrees_with_reported_changed_ids(self, tmp_path):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])

        store = WorldStore(tmp_path)
        v1 = store.save_version(base, parent=None, version_id="v-1")
        v2 = store.save_version(result.new_world, parent="v-1", version_id="v-2")
        # diff_worlds (WorldStore's own independent mechanism) must
        # agree with apply_incremental_update's own report -- two
        # independent computations of "what changed", cross-checked.
        assert set(v2.changed_entity_ids) == set(result.changed_entity_ids)

        world_diff = diff_worlds(base, result.new_world)
        assert {d.entity_id for d in world_diff.entity_diffs} == set(result.changed_entity_ids)


class TestProcessRestart:
    def test_v2_reloads_identical_after_fresh_store_instance(self, tmp_path):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])

        store = WorldStore(tmp_path)
        store.save_version(base, parent=None, version_id="v-1")
        store.save_version(result.new_world, parent="v-1", version_id="v-2")

        # A brand new WorldStore instance (simulating a process restart)
        # over the same root reads back the same world.
        fresh_store = WorldStore(tmp_path)
        reloaded = fresh_store.load_version("v-2")
        assert reloaded.entities["wall-1"].transform["position"]["x"] == 1.5
        assert reloaded.entities["far-tree"].transform["position"]["x"] == 50.0
        assert reloaded.version == 2


class TestFailureLeavesV1Untouched:
    def test_failed_v2_save_does_not_corrupt_or_alter_v1(self, tmp_path):
        base = _base_world()
        new_wall = _entity("wall-1", 1.5, 0.0, 0.0)
        result = apply_incremental_update(base, [new_wall])

        store = WorldStore(tmp_path)
        store.save_version(base, parent=None, version_id="v-1")

        # Force a failure: save V2 against a nonexistent parent id.
        import pytest
        with pytest.raises(WorldStoreError, match="unknown version"):
            store.save_version(result.new_world, parent="v-does-not-exist", version_id="v-2")

        # V1 is untouched and still loads exactly as saved.
        v1_reloaded = store.load_version("v-1")
        assert v1_reloaded.entities["wall-1"].transform["position"]["x"] == 1.0
        assert "v-2" not in {v.version_id for v in store.list_versions()}
