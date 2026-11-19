"""Canonical entity -> WorldIR bridge. SYNTHETIC chair scenario: proves conversion rules (no fabrication, determinism, parts,
traceability), not any model's accuracy."""

from __future__ import annotations

import json

import pytest

from engine.multiengine.association import Candidate, PairScore, associate
from engine.multiengine.canonical import DIMENSIONS, POSITION, build_canonical_entity
from engine.multiengine.evidence_matrix import EvidenceContext, EvidenceMatrix
from engine.multiengine.observation import ObservationDraft, make_observation
from engine.multiengine.registry import default_registry
from engine.multiengine.worldir_bridge import (
    NOT_MEASURED,
    BridgeError,
    SpatialRelation,
    canonical_summary,
    convert,
    trace,
)
from world_ir.schema_v1 import EntityType, GeometryType, RelationshipKind
from world_ir.world_v1 import WorldIR

TOL = {POSITION: 0.1, DIMENSIONS: 0.05}
ENGINES = ("owlvit", "mask_rcnn", "depth_anything")  # three different correlation families in the default registry


class Builder:
    def __init__(self):
        self.m = EvidenceMatrix()
        self.n = 0

    def obs(self, engine, quantity, predicted, *, image=None, frame="world", conf=0.8, calibrated=None, artifact=None):
        self.n += 1
        o = make_observation(
            ObservationDraft(task="detection", quantity=quantity, source_id=image or f"img{self.n}.jpg",
                             coordinate_system=frame, predicted=predicted, confidence=conf, image_region=(1, 1, 9, 9)),
            engine_id=engine, model_version="v1", execution_config={}, processed_at="2026-01-01T00:00:00+00:00",
        )
        refs = {"run_output": artifact} if artifact else {}
        self.m.add(o, EvidenceContext(
            o.observation_id, run_id="run-1", architecture_id="arch", checkpoint_id="ck", artifact_refs=refs,
            calibrated_confidence=calibrated, calibration_basis="synthetic fixture" if calibrated is not None else None,
            validation="synthetic",
        ))
        return o.observation_id

    def entity(self, ids, label):
        cands = [Candidate(i, f"image{k}", label) for k, i in enumerate(ids)]
        pairs = [PairScore(ids[a], ids[b], 0.9) for a in range(len(ids)) for b in range(a + 1, len(ids))]
        res = associate(cands, pairs)
        assert len(res.entities) == 1
        return res.entities[0], res, {i: label for i in ids}


def _chair(b, *, material=("wood", "wood"), with_dims=True, artifact=None):
    ids = [
        b.obs(ENGINES[0], POSITION, {"value": [2.0, 1.0, 0.45]}, artifact=artifact),
        b.obs(ENGINES[1], POSITION, {"value": [2.02, 1.01, 0.45]}, artifact=artifact),
    ]
    if with_dims:
        ids += [b.obs(ENGINES[0], DIMENSIONS, {"value": [0.5, 0.5, 0.9]}, artifact=artifact),
                b.obs(ENGINES[1], DIMENSIONS, {"value": [0.5, 0.52, 0.9]}, artifact=artifact)]
    for eng, lab in zip(ENGINES, material, strict=False):
        ids.append(b.obs(eng, "material", {"label": lab}))
    return ids


def _canon(b, ids, label, **kw):
    ent, res, labels = b.entity(ids, label)
    return build_canonical_entity(ent, b.m, target_frame="world", tolerances=TOL, labels=labels,
                                  registry=default_registry(), association=res, **kw)


def _full():
    b = Builder()
    uri, _ = b.m.artifacts.put(b"raw model output bytes")
    chair = _canon(b, _chair(b, artifact=uri), "chair")
    seat_ids = [
        b.obs(ENGINES[0], POSITION, {"value": [2.0, 1.0, 0.45]}),
        b.obs(ENGINES[1], POSITION, {"value": [2.0, 1.0, 0.46]}),
        b.obs(ENGINES[0], DIMENSIONS, {"value": [0.45, 0.45, 0.05]}),
        b.obs(ENGINES[1], DIMENSIONS, {"value": [0.45, 0.46, 0.05]}),
    ]
    seat = _canon(b, seat_ids, "seat", parent_entity_id=chair.entity_id)
    return b, chair, seat, uri


def test_chair_converts_with_identity_ontology_geometry_material_and_evidence_refs():
    b, chair, seat, uri = _full()
    res = convert([chair, seat], b.m)
    e = res.entities[f"ent-{chair.entity_id}"]
    assert e.type is EntityType.FURNITURE and e.name == "Chair"
    assert e.semantic_labels[0] == "furniture/seating/chair" and "furniture" in e.semantic_labels
    ev = e.custom_properties["evidence"]
    assert ev["ontology"]["category"] == "furniture" and ev["ontology"]["path"] == "furniture/seating/chair"
    assert e.transform["position"]["x"] == pytest.approx(2.01, abs=0.02) and e.transform["frame_id"] == "world"
    (g,) = [res.geometries[i] for i in e.geometry_ids]
    assert g.type is GeometryType.BOX and g.provenance.value == "ESTIMATED"
    assert g.bounds_max.x - g.bounds_min.x == pytest.approx(0.5, abs=0.02)  # the box is the resolved dimensions, nothing else
    (mat,) = [res.materials[i] for i in e.material_ids]
    assert mat.class_name == "wood"
    assert len(e.observations) == len(chair.members)  # EVERY member observation is referenced, supporting or not
    kinds = {(r.kind, r.role) for r in e.external_references}
    assert ("image", "supporting_evidence") in kinds and ("model_output", "raw_model_output") in kinds
    out_ref = next(r for r in e.external_references if r.kind == "model_output")
    assert out_ref.uri == uri and out_ref.checksum == "sha256:" + uri.removeprefix("artifact://")


def test_confidence_is_never_fabricated():
    b = Builder()
    ids = _chair(b)
    ids_cal = b.obs(ENGINES[2], POSITION, {"value": [2.0, 1.0, 0.45]}, calibrated=0.7)
    ce = _canon(b, ids + [ids_cal], "chair")
    e = next(iter(convert([ce], b.m).entities.values()))
    assert e.confidence == NOT_MEASURED
    assert all(v is None for v in e.confidence_vector.to_dict().values())  # nothing was measured per dimension
    assert e.uncertainty.confidence == 0.0 and e.uncertainty.calibrated is False and "not measured" in e.uncertainty.note
    by_id = {o.id: o for o in e.observations}
    plain = by_id[ids[0]]
    assert plain.confidence == NOT_MEASURED and plain.metadata["raw_confidence"] == 0.8
    assert plain.metadata["calibrated_confidence"] is None  # raw and calibrated are separate fields
    cal = by_id[ids_cal]
    assert cal.confidence == 0.7 and cal.metadata["calibrated_confidence"] == 0.7 and cal.metadata["raw_confidence"] == 0.8


def test_part_is_its_own_entity_linked_part_of_and_unobserved_parts_are_not_called_absent():
    b, chair, seat, _ = _full()
    res = convert([chair, seat], b.m)
    s = res.entities[f"ent-{seat.entity_id}"]
    (rel,) = s.relationships
    assert rel.kind is RelationshipKind.PART_OF and rel.target_id == f"ent-{chair.entity_id}"
    assert rel.metadata["listed_in_parent_ontology_components"] is True
    assert s.custom_properties["evidence"]["parent_entity_id"] == chair.entity_id
    ev = res.entities[f"ent-{chair.entity_id}"].custom_properties["evidence"]
    assert ev["unobserved_parts"] == ["backrest", "leg"]  # only the seat was observed
    assert "NOT observed" in ev["unobserved_parts_note"] and "not claimed" in ev["unobserved_parts_note"]


def test_no_geometry_without_dimensions_and_no_transform_without_position():
    b = Builder()
    no_dims = _canon(b, _chair(b, with_dims=False), "chair")
    r1 = convert([no_dims], b.m)
    e = next(iter(r1.entities.values()))
    assert e.geometry_ids == [] and r1.geometries == {} and e.transform is not None
    assert any("no geometry created" in n for n in r1.notes)
    only_dims = _canon(b, [b.obs(ENGINES[0], DIMENSIONS, {"value": [0.5, 0.5, 0.9]})], "chair")
    e2 = next(iter(convert([only_dims], b.m).entities.values()))
    assert e2.transform is None and e2.geometry_ids == []  # a position that was never measured is not invented


def test_competing_material_labels_and_estimates_are_kept_not_merged():
    b = Builder()
    ids = _chair(b, material=("wood", "metal"))
    ids.append(b.obs(ENGINES[2], POSITION, {"value": [9.0, 9.0, 0.45]}))
    ce = _canon(b, ids, "chair")
    r = convert([ce], b.m)
    e = next(iter(r.entities.values()))
    assert r.materials == {} and e.material_ids == []  # an unresolved material is not picked
    hyp = {h["quantity"]: h for h in e.custom_properties["evidence"]["competing_hypotheses"]}
    assert {c["label"] for c in hyp["material"]["candidates"]} == {"wood", "metal"}


def _dump(r):
    return json.dumps(
        {
            "e": {k: v.to_dict() for k, v in sorted(r.entities.items())},
            "g": {k: v.to_dict() for k, v in sorted(r.geometries.items())},
            "m": {k: v.to_dict() for k, v in sorted(r.materials.items())},
        },
        sort_keys=True,
    )


def test_conversion_is_deterministic_and_input_order_independent():
    b, chair, seat, _ = _full()
    a = convert([chair, seat], b.m)
    c = convert([seat, chair], b.m)
    assert _dump(a) == _dump(c) == _dump(convert([chair, seat], b.m))


def test_worldir_round_trip_keeps_everything_and_the_canonical_summary():
    b, chair, seat, _ = _full()
    res = convert([chair, seat], b.m)
    w = WorldIR(id="world-test", name="synthetic")
    res.apply_to(w)
    again = WorldIR.from_dict(json.loads(json.dumps(w.to_dict(), sort_keys=True)))
    assert set(again.entities) == set(w.entities) and set(again.geometries) == set(w.geometries)
    for k, ent in w.entities.items():
        assert again.entities[k].to_dict() == ent.to_dict()
        assert canonical_summary(again.entities[k]) == canonical_summary(ent)
    assert canonical_summary(again.entities[f"ent-{chair.entity_id}"])["members"] == sorted(chair.members)


def test_trace_confirms_every_reference_and_detects_a_broken_one():
    b, chair, seat, uri = _full()
    e = convert([chair, seat], b.m).entities[f"ent-{chair.entity_id}"]
    assert trace(e, b.m) == []
    problems = trace(e, EvidenceMatrix())
    assert any("unknown observation" in p for p in problems) and any("model output" in p for p in problems)
    e.observations.pop()
    assert any("do not match the recorded canonical members" in p for p in trace(e, b.m))


def test_apply_to_refuses_to_overwrite_different_content_but_accepts_identical():
    b, chair, seat, _ = _full()
    res = convert([chair, seat], b.m)
    w = WorldIR(id="w")
    res.apply_to(w)
    res.apply_to(w)  # identical content: fine
    w.entities[f"ent-{chair.entity_id}"].name = "tampered"
    with pytest.raises(BridgeError, match="refusing to overwrite"):
        res.apply_to(w)


def test_relationships_need_evidence_basis_and_the_ontology_prior_makes_none():
    b = Builder()
    chair = _canon(b, _chair(b), "chair")
    floor = _canon(b, [b.obs(ENGINES[0], POSITION, {"value": [0.0, 0.0, 0.0]})], "floor")
    plain = convert([chair, floor], b.m)
    assert all(not e.relationships for e in plain.entities.values())  # 'rests_on floor' in the ontology is a prior, not evidence
    basis = "contact analysis of resolved geometry (synthetic)"
    rel = SpatialRelation(chair.entity_id, RelationshipKind.RESTS_ON, floor.entity_id, basis)
    e = convert([chair, floor], b.m, relations=[rel]).entities[f"ent-{chair.entity_id}"]
    (r,) = e.relationships
    assert r.kind is RelationshipKind.RESTS_ON and r.target_id == f"ent-{floor.entity_id}"
    assert r.metadata["basis"].startswith("contact")
    no_basis = SpatialRelation(chair.entity_id, RelationshipKind.RESTS_ON, floor.entity_id, "")
    with pytest.raises(BridgeError, match="state its basis"):
        convert([chair, floor], b.m, relations=[no_basis])
    ghost = SpatialRelation(chair.entity_id, RelationshipKind.RESTS_ON, "ghost", "x")
    with pytest.raises(BridgeError, match="not a canonical entity"):
        convert([chair], b.m, relations=[ghost])
