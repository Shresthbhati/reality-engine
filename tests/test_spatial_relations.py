"""Spatial relations: derived from resolved boxes only, scale-aware, never an up-axis claim. Stub entities (synthetic geometry)."""

from __future__ import annotations

from types import SimpleNamespace as NS

from engine.multiengine.spatial_relations import derive_relations
from world_ir.schema_v1 import RelationshipKind


def _ent(eid, centre, size, state="sfm_gauge", sup=None):
    sup = sup or (f"{eid}-p",)
    det = lambda v: NS(status="resolved", value=v, frame="world", supporting=sup)  # noqa: E731
    return NS(entity_id=eid, ontology_path="furniture/x", determinations={"position": det(centre), "dimensions": det(size)}), {sup[0]: state}


def _matrix(states):
    return NS(store=NS(get=lambda oid: NS(predicted={"scale_state": states[oid]} if states.get(oid) else {})))


def _derive(*ents):
    states = {}
    for _, s in ents:
        states.update(s)
    return derive_relations([e for e, _ in ents], _matrix(states))


def test_touching_boxes_are_adjacent_both_ways_with_a_stated_basis():
    rels = _derive(_ent("a", (0, 0, 0), (1, 1, 1)), _ent("b", (1.02, 0, 0), (1, 1, 1)))  # 0.02 gap on x, tolerance 0.1
    assert {(r.source_entity_id, r.target_entity_id, r.kind) for r in rels} == {("a", "b", RelationshipKind.ADJACENT_TO), ("b", "a", RelationshipKind.ADJACENT_TO)}
    assert all("tolerance" in r.basis and r.observation_ids for r in rels)


def test_deep_interpenetration_is_overlaps_and_distant_boxes_have_no_relation():
    assert {r.kind for r in _derive(_ent("a", (0, 0, 0), (1, 1, 1)), _ent("b", (0.3, 0.3, 0.3), (1, 1, 1)))} == {RelationshipKind.OVERLAPS}
    assert _derive(_ent("a", (0, 0, 0), (1, 1, 1)), _ent("b", (5, 0, 0), (1, 1, 1))) == []


def test_different_or_undeclared_scale_is_never_compared():
    assert _derive(_ent("a", (0, 0, 0), (1, 1, 1), "sfm_gauge"), _ent("b", (1, 0, 0), (1, 1, 1), "metric_anchored")) == []
    assert _derive(_ent("a", (0, 0, 0), (1, 1, 1), None), _ent("b", (1, 0, 0), (1, 1, 1), None)) == []


def test_no_support_relation_is_ever_claimed_without_an_up_axis():
    kinds = {r.kind for r in _derive(_ent("a", (0, 0, 0), (1, 1, 1)), _ent("b", (0, 0, 1.0), (1, 1, 1)))}  # stacked on z
    assert kinds <= {RelationshipKind.ADJACENT_TO, RelationshipKind.OVERLAPS}
