"""Cross-view association: ledger of every pair decision, pinned to the existing union-find clustering."""

from __future__ import annotations

import random

import pytest

from engine.multiengine.association import (
    AssociationError,
    Candidate,
    PairScore,
    associate,
    ontology_label_compat,
)
from engine.multiengine.evidence_store import EvidenceStore
from engine.multiengine.observation import ObservationDraft, make_observation
from perception.instances.global_association import cluster_observations


def _c(i, img, label="chair", parent=None):
    return Candidate(f"o{i}", img, label, f"o{parent}" if parent is not None else None)


def _partition(result):
    return sorted(e.members for e in result.entities)


def test_matches_cluster_observations_on_random_inputs():
    """The ledger version must reproduce the existing clustering exactly when no label rule is supplied."""
    rng = random.Random(7)
    for _ in range(60):
        n = rng.randint(2, 14)
        imgs = [f"img{rng.randint(0, 4)}" for _ in range(n)]
        parents = [-1] * n
        for i in range(n):
            if i > 0 and rng.random() < 0.25:
                parents[i] = rng.randrange(0, i)
        pairs = [(a, b, round(rng.uniform(-0.2, 1.0), 2)) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.6]
        rng.shuffle(pairs)
        ref = cluster_observations(imgs, pairs, parents)
        cands = [_c(i, imgs[i], parent=None if parents[i] < 0 else parents[i]) for i in range(n)]
        res = associate(cands, [PairScore(f"o{a}", f"o{b}", s) for a, b, s in pairs])
        want = {}
        for i, e in enumerate(ref):
            want.setdefault(int(e), []).append(f"o{i}")
        assert _partition(res) == sorted(tuple(sorted(v)) for v in want.values())


def test_result_does_not_depend_on_pair_order():
    cands = [_c(0, "a"), _c(1, "b"), _c(2, "c")]
    pairs = [PairScore("o0", "o1", 0.5), PairScore("o1", "o2", 0.5), PairScore("o0", "o2", 0.5)]
    r1 = associate(cands, pairs)
    r2 = associate(cands, list(reversed(pairs)))
    assert _partition(r1) == _partition(r2)
    assert [d.outcome for d in r1.decisions] == [d.outcome for d in r2.decisions]


def test_every_pair_gets_a_decision_with_reason():
    cands = [_c(0, "a"), _c(1, "b"), _c(2, "a"), _c(3, "c")]
    pairs = [
        PairScore("o0", "o1", 0.9, {"epipolar_px": 0.8}),  # merged
        PairScore("o2", "o1", 0.7),  # o2 is in image a and o1's entity already has image a -> same image
        PairScore("o0", "o1", 0.6),  # duplicate of an accepted pair
        PairScore("o3", "o0", -0.1),  # non positive
    ]
    res = associate(cands, pairs)
    assert {d.reason for d in res.decisions} == {
        "accepted", "same_image_conflict", "already_same_entity", "non_positive_score",
    }
    accepted = next(d for d in res.decisions if d.outcome == "merged")
    assert accepted.basis == {"epipolar_px": 0.8}  # what justified the merge travels with the decision
    assert len(res.decisions) == len(pairs)


def test_whole_part_rules_are_refused_with_reason():
    cands = [_c(0, "a", "table"), _c(1, "b", "table"), _c(2, "a", "leg", parent=0),
             _c(3, "b", "leg", parent=1), _c(4, "c", "leg")]
    pairs = [
        PairScore("o0", "o1", 0.9),  # wholes merge
        PairScore("o2", "o3", 0.8),  # parts of the merged wholes merge
        PairScore("o2", "o4", 0.7),  # a part with a non-part: refused
    ]
    res = associate(cands, pairs)
    assert res.refused("whole_part_mismatch")
    assert res.entity_of("o2") == res.entity_of("o3")
    assert res.entity_of("o4") != res.entity_of("o2")


def test_part_parent_mismatch_refused():
    cands = [_c(0, "a", "table"), _c(1, "b", "table"), _c(2, "a", "leg", parent=0), _c(3, "b", "leg", parent=1)]
    res = associate(cands, [PairScore("o2", "o3", 0.9)])  # wholes were never merged
    assert [d.reason for d in res.decisions] == ["part_parent_mismatch"]


def test_label_conflict_refusal_uses_ontology():
    compat = ontology_label_compat()
    cands = [_c(0, "a", "chair"), _c(1, "b", "sofa"), _c(2, "c", "chair"), _c(3, "d", "zzzz-unlabeled")]
    pairs = [PairScore("o0", "o1", 0.95), PairScore("o0", "o2", 0.9), PairScore("o0", "o3", 0.5)]
    res = associate(cands, pairs, labels_compatible=compat)
    refused = res.refused("label_conflict")
    assert len(refused) == 1 and refused[0].b == "o1"
    assert res.entity_of("o0") == res.entity_of("o2")
    # an unclassified/unknown label cannot contradict a classified one
    assert res.entity_of("o3") == res.entity_of("o0")
    # without the rule the chair/sofa pair would have been merged (the stronger pair goes first)
    assert not associate(cands, pairs).refused("label_conflict")


def test_accepted_pairs_become_same_entity_relations_in_the_store():
    store = EvidenceStore()
    ids = []
    for i, img in enumerate(["a", "b"]):
        o = make_observation(
            ObservationDraft(task="detection", quantity="bbox", source_id=img, coordinate_system="image",
                             predicted={"value": [i]}),
            engine_id="e", model_version="1", execution_config={}, processed_at="2026-01-01T00:00:00+00:00",
        )
        store.add(o)
        ids.append(o.observation_id)
    res = associate([Candidate(ids[0], "a", "chair"), Candidate(ids[1], "b", "chair")],
                    [PairScore(ids[0], ids[1], 0.8)], store=store)
    assert len(res.entities) == 1
    assert (ids[0], ids[1], "same_entity") in store.relations_of(ids[0])


def test_entity_key_is_stable_and_member_derived():
    cands = [_c(0, "a"), _c(1, "b")]
    k1 = associate(cands, [PairScore("o0", "o1", 1.0)]).entities[0].entity_key
    k2 = associate(list(reversed(cands)), [PairScore("o1", "o0", 1.0)]).entities[0].entity_key
    assert k1 == k2 and k1.startswith("ent_")


def test_bad_inputs_raise():
    with pytest.raises(AssociationError, match="duplicate"):
        associate([_c(0, "a"), _c(0, "b")], [])
    with pytest.raises(AssociationError, match="unknown observation"):
        associate([_c(0, "a")], [PairScore("o0", "ghost", 1.0)])
    with pytest.raises(AssociationError, match="itself"):
        associate([_c(0, "a")], [PairScore("o0", "o0", 1.0)])
    with pytest.raises(AssociationError, match="unknown parent"):
        associate([Candidate("o0", "a", "leg", "nope")], [])


def test_unpaired_detections_stay_their_own_entities():
    res = associate([_c(0, "a"), _c(1, "b")], [])
    assert len(res.entities) == 2 and all(len(e.members) == 1 for e in res.entities)
