"""T-4.2 (rule 18): a late reconstruction result never overwrites an accepted manual edit.

Edits (rename, reclassify, move, property) are recorded as they are made and replayed onto a freshly rebuilt world: exact id first,
else the nearest same-type element within a tolerance, else skipped with the reason. The user's value always wins.
"""
import pytest

from world_ir import Entity, EntityType, WorldIR
from world_ir.edit_carry import CARRY_TOL_M, carry_edits, record_edit


def _wall(eid, x, name="Inferred wall"):
    return Entity(id=eid, type=EntityType.WALL, name=name, transform={"position": {"x": x, "y": 0.0, "z": 0.0}},
                  custom_properties={"wall_slab": {"start_xyz_m": [x - 1, 0, 0], "end_xyz_m": [x + 1, 0, 0]}})


def _sofa(eid, x):
    return Entity(id=eid, type=EntityType.FURNITURE, name="Sofa 1", transform={"position": {"x": x, "y": 0.0, "z": 2.0}},
                  custom_properties={"category": "sofa", "entity_class": "furniture.sofa"}, semantic_labels=["furniture_candidate"])


def _world(*entities):
    w = WorldIR(id="w", name="w")
    for e in entities:
        w.entities[e.id] = e
    return w


def test_a_rename_and_a_property_survive_a_rebuild_with_the_same_ids():
    old = _world(_wall("w1", 0.0), _sofa("s1", 3.0))
    record_edit(old, old.entities["w1"], "name", "Party wall")
    record_edit(old, old.entities["w1"], "custom_properties.load_bearing", True)
    new = _world(_wall("w1", 0.0), _sofa("s1", 3.0))                      # the rebuild: automatic name, no user property
    rep = carry_edits(new, old.metadata["user_edits"])
    assert new.entities["w1"].name == "Party wall" and new.entities["w1"].custom_properties["load_bearing"] is True
    assert len(rep["applied"]) == 2 and rep["skipped"] == []


def test_the_nearest_same_type_element_takes_the_edit_when_ids_changed_and_a_far_one_does_not():
    old = _world(_wall("w1", 0.0))
    record_edit(old, old.entities["w1"], "name", "Party wall")
    near = _world(_wall("other-id", CARRY_TOL_M * 0.5))
    assert carry_edits(near, old.metadata["user_edits"])["applied"][0]["matched_by"].startswith("nearest")
    assert near.entities["other-id"].name == "Party wall"
    far = _world(_wall("other-id", CARRY_TOL_M * 3))
    rep = carry_edits(far, old.metadata["user_edits"])
    assert far.entities["other-id"].name == "Inferred wall" and "no wall within" in rep["skipped"][0]["reason"]


def test_an_edit_is_never_applied_to_an_element_of_another_type():
    old = _world(_sofa("s1", 3.0))
    record_edit(old, old.entities["s1"], "name", "Chesterfield")
    new = _world(_wall("w9", 3.0))                                          # a wall exactly where the sofa was
    rep = carry_edits(new, old.metadata["user_edits"])
    assert new.entities["w9"].name == "Inferred wall" and rep["applied"] == []


def test_a_move_and_a_reclassification_survive_and_the_last_edit_per_field_wins():
    old = _world(_sofa("s1", 3.0))
    s = old.entities["s1"]
    record_edit(old, s, "transform", {"position": {"x": 3.5, "y": 0.0, "z": 2.0}})
    record_edit(old, s, "entity_class", "furniture.armchair")
    record_edit(old, s, "name", "First")
    record_edit(old, s, "name", "Second")
    assert len([r for r in old.metadata["user_edits"] if r["field"] == "name"]) == 1
    new = _world(_sofa("s1", 3.0))
    carry_edits(new, old.metadata["user_edits"])
    e = new.entities["s1"]
    assert e.name == "Second" and e.transform["position"]["x"] == 3.5 and e.custom_properties["entity_class"] == "furniture.armchair"
    assert e.custom_properties["classification_basis"] == "user"


def test_carried_edits_stay_in_the_registry_so_they_survive_the_next_rebuild_too():
    old = _world(_wall("w1", 0.0))
    record_edit(old, old.entities["w1"], "name", "Party wall")
    once = _world(_wall("w1", 0.0))
    carry_edits(once, old.metadata["user_edits"])
    twice = _world(_wall("w1", 0.0))
    carry_edits(twice, once.metadata["user_edits"])
    assert twice.entities["w1"].name == "Party wall"


def test_pipeline_owned_properties_cannot_be_recorded_as_edits():
    w = _world(_wall("w1", 0.0))
    with pytest.raises(ValueError, match="pipeline-owned"):
        record_edit(w, w.entities["w1"], "custom_properties.reconstruction", {"x": 1})
