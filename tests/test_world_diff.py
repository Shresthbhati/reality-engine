"""Tests for the deterministic WorldIR diff (world_ir/diff.py)."""

import copy

from provenance import Provenance
from world_ir.diff import ChangeKind, diff_worlds
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR


def _world_with(entities=(), geometries=()) -> WorldIR:
    world = WorldIR(id="w-fixed")
    for e in entities:
        world.entities[e.id] = e
    for g in geometries:
        world.geometries[g.id] = g
    return world


def test_identical_worlds_produce_empty_diff():
    box = Geometry(id="geom-1", type=GeometryType.BOX)
    entity = Entity(id="ent-1", name="Crate", type=EntityType.DEBRIS, geometry_ids=["geom-1"])
    a = _world_with([entity], [box])
    b = copy.deepcopy(a)

    result = diff_worlds(a, b)
    assert result.is_empty()
    assert result.summary() == {
        "entities_added": 0, "entities_removed": 0, "entities_modified": 0,
        "geometries_added": 0, "geometries_removed": 0, "geometries_modified": 0,
    }


def test_added_and_removed_entities():
    kept = Entity(id="ent-kept", name="Kept")
    removed = Entity(id="ent-removed", name="Removed")
    added = Entity(id="ent-added", name="Added")

    a = _world_with([kept, removed])
    b = _world_with([kept, added])

    result = diff_worlds(a, b)
    assert result.added_entity_ids == ("ent-added",)
    assert result.removed_entity_ids == ("ent-removed",)
    assert result.modified_entity_ids == ()


def test_transform_move_is_reported_as_a_field_change():
    before = Entity(id="ent-1", name="Wall", transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}})
    after = Entity(id="ent-1", name="Wall", transform={"position": {"x": 1.0, "y": 0.0, "z": 0.0}})

    a = _world_with([before])
    b = _world_with([after])

    result = diff_worlds(a, b)
    assert result.modified_entity_ids == ("ent-1",)
    diff = result.entity_diffs[0]
    fields = {c.field for c in diff.changes}
    assert fields == {"transform"}
    assert diff.changes[0].old == before.transform
    assert diff.changes[0].new == after.transform


def test_confidence_and_provenance_changes_reported_separately():
    before = Entity(id="ent-1", provenance=Provenance.ESTIMATED, confidence=0.5)
    after = Entity(id="ent-1", provenance=Provenance.OBSERVED, confidence=0.9)

    a = _world_with([before])
    b = _world_with([after])

    result = diff_worlds(a, b)
    fields = {c.field: (c.old, c.new) for c in result.entity_diffs[0].changes}
    assert fields["provenance"] == (Provenance.ESTIMATED, Provenance.OBSERVED)
    assert fields["confidence"] == (0.5, 0.9)


def test_custom_property_measurement_change_is_reported():
    before = Entity(id="ent-1", custom_properties={"thickness_m": 0.2})
    after = Entity(id="ent-1", custom_properties={"thickness_m": 0.25})

    a = _world_with([before])
    b = _world_with([after])

    result = diff_worlds(a, b)
    diff = result.entity_diffs[0]
    assert len(diff.changes) == 1
    assert diff.changes[0].field == "custom_properties.thickness_m"
    assert diff.changes[0].old == 0.2
    assert diff.changes[0].new == 0.25


def test_geometry_id_reordering_alone_is_not_a_diff():
    entity_a = Entity(id="ent-1", geometry_ids=["geom-1", "geom-2"])
    entity_b = Entity(id="ent-1", geometry_ids=["geom-2", "geom-1"])

    a = _world_with([entity_a])
    b = _world_with([entity_b])

    assert diff_worlds(a, b).is_empty()


def test_geometry_bounds_change_reported():
    before = Geometry(id="geom-1", type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(1, 1, 1))
    after = Geometry(id="geom-1", type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(2, 1, 1))

    a = _world_with([], [before])
    b = _world_with([], [after])

    result = diff_worlds(a, b)
    assert result.summary()["geometries_modified"] == 1
    diff = result.geometry_diffs[0]
    assert diff.kind is ChangeKind.MODIFIED
    fields = {c.field for c in diff.changes}
    assert fields == {"bounds_max"}


def test_added_and_removed_geometry():
    kept = Geometry(id="geom-kept", type=GeometryType.BOX)
    removed = Geometry(id="geom-removed", type=GeometryType.BOX)
    added = Geometry(id="geom-added", type=GeometryType.PLANE)

    a = _world_with([], [kept, removed])
    b = _world_with([], [kept, added])

    result = diff_worlds(a, b)
    kinds = {d.geometry_id: d.kind for d in result.geometry_diffs}
    assert kinds == {"geom-added": ChangeKind.ADDED, "geom-removed": ChangeKind.REMOVED}


def test_diff_is_order_independent_of_dict_insertion_order():
    e1 = Entity(id="ent-1", name="A")
    e2 = Entity(id="ent-2", name="B")

    world_forward = WorldIR(id="w1")
    world_forward.entities["ent-1"] = e1
    world_forward.entities["ent-2"] = e2

    world_backward = WorldIR(id="w1")
    world_backward.entities["ent-2"] = copy.deepcopy(e2)
    world_backward.entities["ent-1"] = copy.deepcopy(e1)

    result = diff_worlds(world_forward, world_backward)
    assert result.is_empty()


def test_to_dict_is_json_serializable_shape():
    before = Entity(id="ent-1", name="Old")
    after = Entity(id="ent-1", name="New")
    a = _world_with([before])
    b = _world_with([after])

    result = diff_worlds(a, b)
    payload = result.to_dict()
    assert payload["from_world_id"] == "w-fixed"
    assert payload["to_world_id"] == "w-fixed"
    assert payload["summary"]["entities_modified"] == 1
    assert payload["entity_diffs"][0]["changes"][0]["field"] == "name"
    assert payload["entity_diffs"][0]["changes"][0]["old"] == "Old"
    assert payload["entity_diffs"][0]["changes"][0]["new"] == "New"


def test_diff_never_mutates_inputs():
    before = Entity(id="ent-1", name="Old", custom_properties={"a": 1})
    after = Entity(id="ent-1", name="New", custom_properties={"a": 2})
    a = _world_with([before])
    b = _world_with([after])

    a_snapshot = copy.deepcopy(a)
    b_snapshot = copy.deepcopy(b)
    diff_worlds(a, b)

    assert a.entities["ent-1"].to_dict() == a_snapshot.entities["ent-1"].to_dict()
    assert b.entities["ent-1"].to_dict() == b_snapshot.entities["ent-1"].to_dict()


def test_deterministic_repeated_runs_produce_identical_result():
    before = Entity(id="ent-1", name="Old")
    after = Entity(id="ent-1", name="New")
    a = _world_with([before])
    b = _world_with([after])

    r1 = diff_worlds(a, b).to_dict()
    r2 = diff_worlds(a, b).to_dict()
    assert r1 == r2
