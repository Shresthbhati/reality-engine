"""Geometric / semantic conflict detection (SYNTHETIC boxes; logic only, no real-world accuracy claim)."""

from __future__ import annotations

import pytest

from engine.multiengine.association import Candidate, PairScore, associate, ontology_label_compat
from engine.multiengine.conflicts import (
    Box,
    find_overlaps,
    ontology_contact_allowed,
    record_conflicts,
    semantic_conflicts,
)
from engine.multiengine.evidence_store import EvidenceStore
from engine.multiengine.observation import ObservationDraft, make_observation


def _box(i, path, centre, size):
    return Box(
        i, path,
        tuple(c - s / 2 for c, s in zip(centre, size, strict=True)),
        tuple(c + s / 2 for c, s in zip(centre, size, strict=True)),
    )


def test_overlap_beyond_tolerance_is_reported_with_depth():
    a = _box("a", "furniture/seating/chair", (0, 0, 0.45), (0.5, 0.5, 0.9))
    b = _box("b", "furniture/seating/chair", (0.3, 0, 0.45), (0.5, 0.5, 0.9))  # 0.2 m overlap in x
    (c,) = find_overlaps([a, b], tolerance_m=0.05)
    assert c.kind == "interpenetration" and abs(c.magnitude - 0.2) < 1e-9
    assert "rotation not modelled" in c.basis["model"]


def test_overlap_within_tolerance_or_touching_is_not_a_conflict():
    a = _box("a", "furniture/seating/chair", (0, 0, 0), (1, 1, 1))
    touching = _box("b", "furniture/seating/chair", (1, 0, 0), (1, 1, 1))
    near = _box("d", "furniture/seating/chair", (0.98, 0, 0), (1, 1, 1))
    far = _box("c", "furniture/seating/chair", (0.98, 5, 0), (1, 1, 1))
    assert find_overlaps([a, touching], tolerance_m=0.0) == []
    assert find_overlaps([a, near], tolerance_m=0.05) == []
    assert find_overlaps([a, far], tolerance_m=0.0) == []


def test_tolerance_has_no_default_and_must_be_non_negative():
    with pytest.raises(TypeError):
        find_overlaps([])  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        find_overlaps([], tolerance_m=-1)


def test_ontology_declared_contact_is_not_a_conflict():
    from perception.ontology import get_default_ontology

    onto = get_default_ontology()
    host = next(n for n in onto.nodes.values() if n.hosts)
    a = _box("a", host.path, (0, 0, 0.4), (1, 1, 0.8))
    b = _box("b", host.hosts[0], (0, 0, 0.7), (0.3, 0.3, 0.4))  # overlaps the host's top by 0.1 m
    allowed = ontology_contact_allowed(onto)
    assert allowed(a, b)
    assert find_overlaps([a, b], tolerance_m=0.01, contact_allowed=allowed) == []
    assert len(find_overlaps([a, b], tolerance_m=0.01)) == 1  # the same overlap IS a conflict without the declaration


def test_undeclared_overlap_of_two_unrelated_classes_stays_a_conflict():
    allowed = ontology_contact_allowed()
    a = _box("a", "furniture/seating/chair", (0, 0, 0.45), (0.5, 0.5, 0.9))
    b = _box("b", "furniture/seating/sofa", (0.2, 0, 0.45), (0.5, 0.5, 0.9))
    assert not allowed(a, b)
    assert len(find_overlaps([a, b], tolerance_m=0.05, contact_allowed=allowed)) == 1


def _obs(store, source):
    o = make_observation(
        ObservationDraft(task="detection", quantity="position", source_id=source, coordinate_system="world",
                         predicted={"value": [0, 0, 0]}),
        engine_id="e", model_version="1", execution_config={}, processed_at="2026-01-01T00:00:00+00:00",
    )
    store.add(o)
    return o.observation_id


def test_semantic_conflict_and_recording_in_evidence_store():
    store = EvidenceStore()
    a, b = _obs(store, "img0"), _obs(store, "img1")
    res = associate([Candidate(a, "img0", "chair"), Candidate(b, "img1", "sofa")], [PairScore(a, b, 0.9)],
                    labels_compatible=ontology_label_compat())
    conflicts = semantic_conflicts(res)
    assert [c.kind for c in conflicts] == ["semantic_label"]
    assert record_conflicts(store, conflicts, {}) == 1
    assert record_conflicts(store, conflicts, {}) == 0  # idempotent
    assert (a, b, "conflicts_with") in store.relations_of(a)


def test_interpenetration_links_supporting_observations_and_skips_unsupported_entities():
    store = EvidenceStore()
    a, b = _obs(store, "img0"), _obs(store, "img1")
    c = find_overlaps(
        [_box("e1", "furniture/seating/chair", (0, 0, 0), (1, 1, 1)),
         _box("e2", "furniture/seating/sofa", (0.5, 0, 0), (1, 1, 1))],
        tolerance_m=0.05,
    )
    assert record_conflicts(store, c, {"e1": [a], "e2": [b]}) == 1
    assert record_conflicts(store, c, {"e1": [a]}) == 0
