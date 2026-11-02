"""Blueprint 3.4: WorldIR contract section 3 invariants 1-7, one passing and one failing case each."""
from __future__ import annotations

import pytest

from provenance import Provenance
from world_ir import WorldIR
from world_ir.observations import (
    AssociationDecision, AssociationState, DepthSource, Observation, attach_ledger,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Relationship, RelationshipKind, Surface, Vector3
from world_ir.validation import validate_world_ir


def _codes(world: WorldIR) -> set:
    return {i.code for i in validate_world_ir(world).errors}


def _inv(world: WorldIR, n: int) -> list:
    return [c for c in _codes(world) if c.startswith(f"invariant_{n}_")]


def _box(world: WorldIR, gid: str, lo, hi) -> str:
    world.geometries[gid] = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(*lo), bounds_max=Vector3(*hi))
    return gid


def _ent(world: WorldIR, eid: str, etype=EntityType.STRUCTURE, **kw) -> Entity:
    world.entities[eid] = Entity(id=eid, type=etype, **kw)
    return world.entities[eid]


def _obs(oid="o1") -> Observation:
    return Observation(oid, "img1", "chair", (0.0, 0.0, 0.0), 0.1, DepthSource.MVS)


def test_clean_world_has_no_invariant_errors():
    w = WorldIR(id="w", name="w")
    _ent(w, "e1", provenance=Provenance.RECONSTRUCTED)
    assert validate_world_ir(w).is_valid()


# 1 ---------------------------------------------------------------------------------------------
def test_inv1_pass_every_entity_and_surface_has_an_evidence_state():
    w = WorldIR(id="w", name="w")
    _ent(w, "e1", provenance=Provenance.UNKNOWN)
    w.surfaces["s1"] = Surface(id="s1", custom_attributes={"evidence_state": "INFERRED"})
    assert _inv(w, 1) == []


def test_inv1_fail_surface_without_state_and_entity_with_bogus_state():
    w = WorldIR(id="w", name="w")
    w.surfaces["s1"] = Surface(id="s1")
    _ent(w, "e1", custom_properties={"evidence_state": "SORT_OF_SEEN"})
    r = validate_world_ir(w)
    assert not r.is_valid()
    assert sum(i.code == "invariant_1_no_evidence_state" for i in r.errors) == 2


# 2 ---------------------------------------------------------------------------------------------
def test_inv2_pass_observed_entity_cites_an_observation():
    w = WorldIR(id="w", name="w")
    attach_ledger(w, [_obs()], [])
    _ent(w, "e1", provenance=Provenance.OBSERVED, custom_properties={"observation_ids": ["o1"]})
    assert _inv(w, 2) == [] and _inv(w, 5) == []


def test_inv2_fail_observed_entity_without_observation():
    w = WorldIR(id="w", name="w")
    _ent(w, "e1", provenance=Provenance.OBSERVED)
    assert _inv(w, 2) == ["invariant_2_observed_without_observation"]
    _ent(w, "e2", custom_properties={"evidence_state": "PARTIALLY_OBSERVED"})
    assert len([c for c in validate_world_ir(w).errors if c.code.startswith("invariant_2_")]) == 2


# 3 ---------------------------------------------------------------------------------------------
def test_inv3_pass_inferred_stays_inferred():
    w = WorldIR(id="w", name="w")
    _ent(w, "e1", provenance=Provenance.INFERRED, custom_properties={"evidence_state": "INFERRED"})
    assert _inv(w, 3) == []


def test_inv3_fail_inferred_also_labelled_observed():
    w = WorldIR(id="w", name="w")
    attach_ledger(w, [_obs()], [])
    _ent(w, "e1", provenance=Provenance.INFERRED,
         custom_properties={"evidence_state": "OBSERVED", "observation_ids": ["o1"]})
    assert _inv(w, 3) == ["invariant_3_inferred_labelled_observed"]


# 4 ---------------------------------------------------------------------------------------------
def test_inv4_pass_model_units_when_relative_and_metres_when_metric():
    rel = WorldIR(id="w", name="w", metadata={"scale": {"state": "relative"}})
    _ent(rel, "e1", custom_properties={"width": {"value": 2.0, "unit": "model_units"}})
    assert _inv(rel, 4) == []
    met = WorldIR(id="w", name="w", metadata={"scale": {"state": "metric"}})
    _ent(met, "e1", custom_properties={"width": {"value": 2.0, "unit": "m"}})
    assert _inv(met, 4) == []


def test_inv4_fail_metric_unit_in_relative_or_undeclared_world():
    for meta in ({"scale": {"state": "relative"}}, {}):
        w = WorldIR(id="w", name="w", metadata=meta)
        _ent(w, "e1", custom_properties={"dims": [{"value": 2.0, "unit": "m"}]})
        assert _inv(w, 4) == ["invariant_4_metric_unit_in_relative_world"]


# 5 ---------------------------------------------------------------------------------------------
def test_inv5_pass_ids_unique_and_references_exist():
    w = WorldIR(id="w", name="w")
    attach_ledger(w, [_obs()], [AssociationDecision("o1", AssociationState.DIFFERENT_ENTITY, "e1", 0.0, "new")])
    _ent(w, "e1", custom_properties={"observation_ids": ["o1"]})
    assert _inv(w, 5) == [] and validate_world_ir(w).is_valid()


def test_inv5_fail_key_id_mismatch_dangling_observation_and_dangling_relationship():
    w = WorldIR(id="w", name="w")
    w.entities["a"] = Entity(id="b")
    _ent(w, "e1", custom_properties={"observation_ids": ["nope"]},
         relationships=[Relationship(kind=RelationshipKind.ADJACENT_TO, target_id="ghost")])
    attach_ledger(w, [_obs()], [AssociationDecision("missing", AssociationState.AMBIGUOUS, None, 0.0, "x")])
    codes = _codes(w)
    assert {"invariant_5_id_mismatch", "invariant_5_dangling_reference", "structural"} <= codes


# 6 ---------------------------------------------------------------------------------------------
def _two_overlapping_chairs(**second) -> WorldIR:
    w = WorldIR(id="w", name="w")
    _ent(w, "c1", geometry_ids=[_box(w, "g1", (0, 0, 0), (1, 1, 1))], semantic_labels=["chair"])
    _ent(w, "c2", geometry_ids=[_box(w, "g2", (0.05, 0, 0), (1.05, 1, 1))], semantic_labels=["chair"], **second)
    return w


def test_inv6_pass_distinct_decision_or_other_class_or_apart():
    w = _two_overlapping_chairs(custom_properties={"distinct_from": ["c1"]})
    assert _inv(w, 6) == []
    w_led = _two_overlapping_chairs()
    assert _inv(w_led, 6) != []  # sanity: without the exemption it fails
    attach_ledger(w_led, [_obs()], [AssociationDecision("o1", AssociationState.DIFFERENT_ENTITY, "c2", 0.0, "x",
                                                        candidates=(("c1", 0.1),))])
    assert _inv(w_led, 6) == []  # a DIFFERENT_ENTITY decision in the ledger exempts the pair
    w2 = WorldIR(id="w", name="w")
    _ent(w2, "c1", geometry_ids=[_box(w2, "g1", (0, 0, 0), (1, 1, 1))], semantic_labels=["chair"])
    _ent(w2, "t1", geometry_ids=[_box(w2, "g2", (0, 0, 0), (1, 1, 1))], semantic_labels=["table"])
    _ent(w2, "c3", geometry_ids=[_box(w2, "g3", (5, 0, 0), (6, 1, 1))], semantic_labels=["chair"])
    assert _inv(w2, 6) == []


def test_inv6_fail_same_class_overlapping_more_than_half():
    assert _inv(_two_overlapping_chairs(), 6) == ["invariant_6_duplicate_entity"]


def test_inv6_overlapping_floor_patches_warn_by_default_and_block_when_enabled(monkeypatch):
    w = WorldIR(id="w", name="w")
    _ent(w, "f1", EntityType.FLOOR, geometry_ids=[_box(w, "g1", (0, 0, 0), (1, 0.01, 1))])
    _ent(w, "f2", EntityType.FLOOR, geometry_ids=[_box(w, "g2", (0, 0, 0), (1, 0.01, 1))])
    r = validate_world_ir(w)
    assert r.is_valid() and [i.code for i in r.warnings] == ["invariant_6_duplicate_entity"]
    monkeypatch.setattr("world_ir.validation.SURFACE_DUPLICATE_BLOCKS", True)
    assert _inv(w, 6) == ["invariant_6_duplicate_entity"]


# 7 ---------------------------------------------------------------------------------------------
def _room_wall_door() -> WorldIR:
    w = WorldIR(id="w", name="w")
    _ent(w, "wall1", EntityType.WALL)
    _ent(w, "door1", EntityType.DOOR, custom_properties={"host_wall_id": "wall1"})
    _ent(w, "room1", EntityType.ROOM, custom_properties={"boundary_element_ids": ["wall1"]})
    return w


def test_inv7_pass_opening_on_surface_and_surface_in_space():
    assert _inv(_room_wall_door(), 7) == []


def test_inv7_fail_orphan_opening_and_orphan_surface(monkeypatch):
    monkeypatch.setattr("world_ir.validation.SURFACE_WITHOUT_SPACE_BLOCKS", True)
    w = WorldIR(id="w", name="w")
    _ent(w, "wall1", EntityType.WALL)
    _ent(w, "door1", EntityType.DOOR)
    _ent(w, "room1", EntityType.ROOM)
    assert sorted(_inv(w, 7)) == ["invariant_7_opening_without_surface", "invariant_7_surface_without_space"]


def test_inv7_orphan_surface_is_a_warning_by_default():
    w = WorldIR(id="w", name="w")
    _ent(w, "wall1", EntityType.WALL)
    _ent(w, "room1", EntityType.ROOM)
    r = validate_world_ir(w)
    assert r.is_valid() and [i.code for i in r.warnings] == ["invariant_7_surface_without_space"]


def test_inv7_not_enforced_before_the_world_models_those_levels():
    w = WorldIR(id="w", name="w")  # planes compiled, no room yet (partial capture): not blocked
    _ent(w, "wall1", EntityType.WALL)
    assert _inv(w, 7) == []
    w2 = WorldIR(id="w", name="w")  # an opening in a world with no surfaces at all: not blocked either
    _ent(w2, "door1", EntityType.DOOR)
    assert _inv(w2, 7) == []


def test_blocked_commit_report_is_readable():
    w = WorldIR(id="w", name="w")
    _ent(w, "e1", provenance=Provenance.OBSERVED)
    text = validate_world_ir(w).render()
    assert text.startswith("world validation: BLOCKED, 1 error(s)")
    assert "invariant_2_observed_without_observation" in text and "e1" in text
