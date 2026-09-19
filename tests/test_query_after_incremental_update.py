"""Query correctness after an incremental update: V1 query -> apply
apply_incremental_update() -> V2 query must reflect the change,
never leak stale (pre-update) data.

Audit finding this test documents: engine.scene_graph.spatial_index.
SpatialIndex is a snapshot built once over a WorldIR ("Does not observe
later mutations to `world`" -- its own docstring). There is no lazy or
incremental spatial-index update path today -- a fresh SpatialIndex
must be built over `result.new_world` after every apply_incremental_update()
call. This file proves that rebuilding on new_world gives correct
results; it does NOT add incremental index maintenance (out of scope,
documented as a limitation in the checkpoint handoff).
"""

from __future__ import annotations

from engine.scene_graph.spatial_index import SpatialIndex
from provenance import Provenance
from world_ir import apply_incremental_update
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def _entity(eid, x, y, z) -> Entity:
    return Entity(
        id=eid, type=EntityType.STRUCTURE,
        transform={"position": {"x": x, "y": y, "z": z}},
        provenance=Provenance.RECONSTRUCTED,
    )


class TestQueryReflectsMovedEntity:
    def test_v1_query_finds_old_position_v2_query_finds_new_position(self):
        base = WorldIR(id="w1")
        base.entities["a"] = _entity("a", 0.0, 0.0, 0.0)
        base.entities["b"] = _entity("b", 100.0, 0.0, 0.0)

        v1_index = SpatialIndex(base)
        nearest_v1 = v1_index.nearest((0.0, 0.0, 0.0), k=1)
        assert nearest_v1[0][0].id == "a"

        moved_a = _entity("a", 90.0, 0.0, 0.0)  # a moves near b
        result = apply_incremental_update(base, [moved_a])

        v2_index = SpatialIndex(result.new_world)
        nearest_v2 = v2_index.nearest((90.0, 0.0, 0.0), k=1)
        assert nearest_v2[0][0].id == "a"  # a's new position, exact match
        assert nearest_v2[0][1] == 0.0

        # The V1 index, built before the update, still reports the OLD
        # world -- proving V1 and V2 queries are genuinely independent
        # snapshots, not a shared mutable structure that could leak
        # V2 state backward into a V1 query.
        assert v1_index.nearest((0.0, 0.0, 0.0), k=1)[0][0].id == "a"
        assert v1_index.nearest((0.0, 0.0, 0.0), k=1)[0][1] == 0.0


class TestQueryReflectsDeletion:
    def test_deleted_entity_is_unreachable_in_v2_query(self):
        base = WorldIR(id="w1")
        base.entities["a"] = _entity("a", 0.0, 0.0, 0.0)
        base.entities["b"] = _entity("b", 5.0, 0.0, 0.0)

        result = apply_incremental_update(base, [], removed_entities=["a"])
        v2_index = SpatialIndex(result.new_world)

        nearest = v2_index.nearest((0.0, 0.0, 0.0), k=2)
        found_ids = {e.id for e, _ in nearest}
        assert "a" not in found_ids
        assert found_ids == {"b"}


class TestQueryReflectsAddition:
    def test_new_entity_is_reachable_in_v2_query(self):
        base = WorldIR(id="w1")
        base.entities["a"] = _entity("a", 0.0, 0.0, 0.0)

        new_entity = _entity("c", 1.0, 0.0, 0.0)
        result = apply_incremental_update(base, [new_entity])
        v2_index = SpatialIndex(result.new_world)

        nearest = v2_index.nearest((1.0, 0.0, 0.0), k=1)
        assert nearest[0][0].id == "c"


class TestUnrelatedEntityQueryUnaffected:
    def test_far_unrelated_entity_query_result_identical_before_and_after(self):
        base = WorldIR(id="w1")
        base.entities["a"] = _entity("a", 0.0, 0.0, 0.0)
        base.entities["far"] = _entity("far", 1000.0, 0.0, 0.0)

        moved_a = _entity("a", 1.0, 0.0, 0.0)
        result = apply_incremental_update(base, [moved_a])
        v2_index = SpatialIndex(result.new_world)

        nearest = v2_index.nearest((1000.0, 0.0, 0.0), k=1)
        assert nearest[0][0].id == "far"
        assert nearest[0][1] == 0.0
        # The object itself is untouched -- the same Entity the V1
        # index would have resolved to.
        assert nearest[0][0] is base.entities["far"]
