"""Stable entity identity across versions: the same wall refined is MODIFIED, never removed + added.

Identity comes from measured spatial continuity (one-to-one preserved/refined, same type) -- never from random
ids, and never forced: splits, merges and ambiguous overlaps keep their own ids and record lineage instead.
"""

from __future__ import annotations

import numpy as np

from engine.pipeline import world_delta as wd
from tests.test_spatial_continuity import WALL, lattice, rec
from tests.test_world_delta import ring, snap
from world_ir import WorldIR
from world_ir.diff import ChangeKind, diff_worlds
from world_ir.entity_reid import carry_identity, rekey_entities
from world_ir.schema_v1 import Entity, EntityType, Provenance, Relationship, RelationshipKind

CAMS = ring()


def _snap(entities):
    s = snap(list(CAMS), cams=CAMS, types={"wall": len(entities)})
    s["entities"] = entities
    return s


def _world(records):
    w = WorldIR(id="w")
    for r in records:
        w.entities[r["id"]] = Entity(id=r["id"], type=EntityType(r["type"]), name=r["id"],
                                     provenance=Provenance.RECONSTRUCTED, confidence=0.8)
    return w


def _relations(prev_recs, cand_recs):
    d = wd.compute_delta(_snap(prev_recs), _snap(cand_recs))
    return d["entities"]["relations"]


def _other(i):                       # a distinct wall elsewhere so the scene has structure beyond the subject
    return lattice(0, 3, 0, 2, origin=(0, 0, 8 + 8 * i))


def _diff_kinds(prev_w, cand_w):
    return {d.entity_id: d.kind for d in diff_worlds(prev_w, cand_w).entity_diffs}


def test_the_same_wall_refined_keeps_its_id_and_the_diff_says_modified_not_removed_plus_added():
    prev = [rec("wall-A", "wall", WALL), rec("o1", "wall", _other(0)), rec("o2", "wall", _other(1))]
    cand = [rec("zz-9", "wall", WALL + np.array([0, 0, 0.15])), rec("o1x", "wall", _other(0)),
            rec("o2x", "wall", _other(1))]
    rels = _relations(prev, cand)
    before = _diff_kinds(_world(prev), _world(cand))
    assert before["wall-A"] == ChangeKind.REMOVED and before["zz-9"] == ChangeKind.ADDED      # the old behaviour

    cw = _world(cand)
    out = carry_identity({r["id"] for r in prev}, rels, cw)
    assert out["carried"] == {"zz-9": "wall-A", "o1x": "o1", "o2x": "o2"}
    assert set(cw.entities) == {"wall-A", "o1", "o2"} and all(e.id == k for k, e in cw.entities.items())
    assert cw.entities["wall-A"].custom_properties["identity"]["by"] == "spatial_continuity"
    kinds = _diff_kinds(_world(prev), cw)
    assert ChangeKind.REMOVED not in kinds.values() and ChangeKind.ADDED not in kinds.values(), kinds


def test_real_removal_and_real_addition_are_not_disguised_as_identity():
    prev = [rec("a", "wall", WALL), rec("gone", "wall", _other(0))]
    cand = [rec("a2", "wall", WALL), rec("brand-new", "wall", _other(3))]
    cw = _world(cand)
    out = carry_identity({"a", "gone"}, _relations(prev, cand), cw)
    assert out["carried"] == {"a2": "a"}
    kinds = _diff_kinds(_world(prev), cw)
    assert kinds["gone"] == ChangeKind.REMOVED and kinds["brand-new"] == ChangeKind.ADDED


def test_a_split_never_forces_identity_and_records_lineage():
    a1, a2 = lattice(0, 3, 0, 2), lattice(3.05, 6, 0, 2)
    prev = [rec("w", "wall", WALL), rec("o1", "wall", _other(0)), rec("o2", "wall", _other(1))]
    cand = [rec("f1", "wall", a1), rec("f2", "wall", a2), rec("o1x", "wall", _other(0)), rec("o2x", "wall", _other(1))]
    cw = _world(cand)
    out = carry_identity({"w", "o1", "o2"}, _relations(prev, cand), cw)
    assert "f1" not in out["carried"] and "f2" not in out["carried"] and "w" not in cw.entities
    assert [x["relation"] for x in out["lineage"]] == ["split"]
    assert cw.entities["f1"].custom_properties["continuity"] == {"relation": "split", "previous": ["w"]}


def test_a_merge_never_forces_identity_and_records_lineage():
    a1, a2 = lattice(0, 3, 0, 2), lattice(3.05, 6, 0, 2)
    prev = [rec("p1", "wall", a1), rec("p2", "wall", a2), rec("o1", "wall", _other(0)), rec("o2", "wall", _other(1))]
    cand = [rec("whole", "wall", WALL), rec("o1x", "wall", _other(0)), rec("o2x", "wall", _other(1))]
    cw = _world(cand)
    out = carry_identity({"p1", "p2", "o1", "o2"}, _relations(prev, cand), cw)
    assert "whole" in cw.entities and "p1" not in cw.entities and "p2" not in cw.entities
    assert [x["relation"] for x in out["lineage"]] == ["merge"]


def test_ambiguous_overlap_stays_ambiguous_ids_untouched():
    prev = [rec("p", "wall", lattice(0, 4, 0, 2)), rec("q", "wall", lattice(3, 7, 0, 2)),
            rec("o1", "wall", _other(0)), rec("o2", "wall", _other(1))]
    cand = [rec("c", "wall", lattice(2, 5, 0, 2)), rec("o1x", "wall", _other(0)), rec("o2x", "wall", _other(1))]
    cw = _world(cand)
    out = carry_identity({"p", "q", "o1", "o2"}, _relations(prev, cand), cw)
    assert "c" in cw.entities and "p" not in cw.entities and "q" not in cw.entities
    assert "c" not in out["carried"]


def test_an_id_already_in_use_is_never_overwritten():
    w = _world([rec("x", "wall", WALL), rec("y", "wall", _other(0))])
    assert rekey_entities(w, {"x": "y"}) == {}                   # target id exists: refused
    assert set(w.entities) == {"x", "y"}


def test_rekey_updates_every_reference_to_the_renamed_entity():
    w = _world([rec("new", "wall", WALL), rec("other", "wall", _other(0))])
    w.entities["other"].relationships.append(
        Relationship(kind=RelationshipKind.ADJACENT_TO, target_id="new", confidence=0.7, provenance=Provenance.INFERRED))
    assert rekey_entities(w, {"new": "old"}) == {"new": "old"}
    assert w.entities["other"].relationships[0].target_id == "old" and "new" not in w.entities
    assert w.entities["old"].id == "old"
    assert w.entities["other"].relationships[0].confidence == 0.7


def test_confidence_defaults_never_read_as_observed_certainty():
    from world_ir.schema_v1 import CausalRelation, Observation

    r = Relationship(kind=RelationshipKind.PART_OF, target_id="t")
    assert r.confidence == 0.5 and r.provenance == Provenance.UNKNOWN
    assert Observation().confidence == 0.5 and CausalRelation().confidence == 0.5
    # a legacy record without confidence is "not measured", and measured values round-trip untouched
    assert Relationship.from_dict({"kind": "part_of", "target_id": "t"}).confidence == 0.5
    measured = Relationship(kind=RelationshipKind.PART_OF, target_id="t", confidence=0.83,
                            provenance=Provenance.INFERRED)
    back = Relationship.from_dict(measured.to_dict())
    assert back.confidence == 0.83 and back.provenance == Provenance.INFERRED
    assert Observation.from_dict(Observation(confidence=0.91).to_dict()).confidence == 0.91


def test_promoted_topology_relationships_are_inferred_with_their_elements_confidence():
    from perception.architecture import promotion

    rel = promotion._inferred_rel(RelationshipKind.ADJACENT_TO, "room-2", 0.62)
    assert rel.provenance == Provenance.INFERRED and rel.confidence == 0.62


def test_scale_references_are_validated_persisted_and_replace_the_same_pair(tmp_path, monkeypatch):
    import pytest

    from apps.api import worldstore_service as wss
    from reconstruction.scale import ScaleAnchoringError

    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path))
    assert wss.list_scale_references("w1") == []
    with pytest.raises(ScaleAnchoringError):
        wss.add_scale_reference("w1", "a", "a", 1.0)             # same photograph twice
    with pytest.raises(ScaleAnchoringError):
        wss.add_scale_reference("w1", "a", "b", -2.0)            # not a distance
    with pytest.raises(ScaleAnchoringError):
        wss.add_scale_reference("w1", "a", "b", float("nan"))
    assert wss.list_scale_references("w1") == []                 # nothing was stored for any refused input
    wss.add_scale_reference("w1", "a", "b", 1.2, "tape measure")
    wss.add_scale_reference("w1", "b", "c", 3.0)
    wss.add_scale_reference("w1", "b", "a", 1.25, "tape measure, remeasured")   # same pair, corrected
    refs = wss.list_scale_references("w1")
    assert [(r["evidence_id_a"], r["evidence_id_b"], r["distance_m"]) for r in refs] == [("b", "c", 3.0), ("b", "a", 1.25)]
    assert refs[-1]["method"] == "tape measure, remeasured" and wss.list_scale_references("other") == []
