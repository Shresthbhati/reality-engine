"""Evidence-backed change proposals over the WorldStore: assess, accept (as a NEW version), override, reject."""

from __future__ import annotations

import copy

import pytest

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Observation, Vector3
from world_ir.world_v1 import WorldIR
from worldstore.proposals import (
    CONFLICT, NEEDS_REVIEW, SUPPORTED, ProposalError, accept_proposal, get_proposal, list_proposals,
    propose_update, reject_proposal,
)
from worldstore.store import WorldStore


def geom(gid, lo=(0, 0, 0), hi=(1, 1, 1), prov=Provenance.OBSERVED):
    return Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(*lo), bounds_max=Vector3(*hi), provenance=prov)


def ent(eid, gid, prov=Provenance.OBSERVED, conf=0.9, obs=(), etype=EntityType.WALL, name=None):
    return Entity(id=eid, name=name or eid, type=etype, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
                  geometry_ids=[gid], provenance=prov, confidence=conf,
                  observations=[Observation(id=o, sensor_type="camera") for o in obs])


def base_world():
    w = WorldIR()
    for eid, hi in (("wall-a", (4, 0.2, 3)), ("wall-b", (0.2, 5, 3)), ("door", (1, 0.2, 2))):
        w.geometries[f"g-{eid}"] = geom(f"g-{eid}", hi=hi)
        w.entities[eid] = ent(eid, f"g-{eid}", obs=(f"obs-{eid}",))
    return w


@pytest.fixture
def store(tmp_path):
    s = WorldStore(tmp_path / "ws")
    s.save_version(base_world(), parent=None, version_id="v-base")
    return s


def candidate(mutate):
    w = copy.deepcopy(base_world())
    mutate(w)
    return w


def verdicts(p):
    return {a.entity_id: a.verdict for a in p.assessments}


def test_each_rule_yields_its_verdict(store):
    def mutate(w):
        w.geometries["g-new"] = geom("g-new", (9, 9, 0), (10, 10, 1))
        w.entities["window"] = ent("window", "g-new", obs=("obs-w",))                          # added + evidence
        w.geometries["g-guess"] = geom("g-guess", (9, 9, 0), (10, 10, 1), Provenance.INFERRED)
        w.entities["ghost"] = ent("ghost", "g-guess", Provenance.INFERRED, 0.3)                # added, no evidence
        del w.entities["door"]                                                                  # removed
        w.geometries["g-wall-a"] = geom("g-wall-a", hi=(4.5, 0.2, 3))                          # moved, no new evidence
        w.entities["wall-a"] = ent("wall-a", "g-wall-a", obs=("obs-wall-a",))
        w.geometries["g-wall-b"] = geom("g-wall-b", hi=(0.2, 5.5, 3), prov=Provenance.INFERRED)
        w.entities["wall-b"] = ent("wall-b", "g-wall-b", Provenance.INFERRED, 0.4, obs=("obs-wall-b", "obs-x"))
    p = propose_update(store, "v-base", candidate(mutate))
    assert verdicts(p) == {"window": SUPPORTED, "ghost": NEEDS_REVIEW, "door": NEEDS_REVIEW,
                           "wall-a": NEEDS_REVIEW, "wall-b": CONFLICT}
    assert "weaker INFERRED" in " ".join(next(a for a in p.assessments if a.entity_id == "wall-b").reasons)
    assert "absence is not evidence" in " ".join(next(a for a in p.assessments if a.entity_id == "door").reasons)
    assert p.summary[SUPPORTED] == 1 and p.summary[CONFLICT] == 1 and p.candidate_validation["valid"]


def test_modification_with_new_evidence_is_supported(store):
    def mutate(w):
        w.geometries["g-wall-a"] = geom("g-wall-a", hi=(4.5, 0.2, 3))
        w.entities["wall-a"] = ent("wall-a", "g-wall-a", obs=("obs-wall-a", "obs-new-scan"), conf=0.95)
    p = propose_update(store, "v-base", candidate(mutate))
    a = p.assessments[0]
    assert a.verdict == SUPPORTED and a.new_evidence == ("obs-new-scan",) and "geometry" in a.categories


def test_sub_tolerance_geometry_noise_is_not_a_geometry_change(store):
    def mutate(w):
        w.geometries["g-wall-a"] = geom("g-wall-a", hi=(4.005, 0.2, 3))
        w.entities["wall-a"] = ent("wall-a", "g-wall-a", obs=("obs-wall-a",), conf=0.95)       # only confidence moved
    a = propose_update(store, "v-base", candidate(mutate)).assessments[0]
    assert "geometry" not in a.categories and a.verdict == SUPPORTED


def test_accept_defaults_to_supported_only_and_creates_new_version_leaving_base_intact(store):
    def mutate(w):
        w.geometries["g-new"] = geom("g-new", (9, 9, 0), (10, 10, 1))
        w.entities["window"] = ent("window", "g-new", obs=("obs-w",))
        del w.entities["door"]
    p = propose_update(store, "v-base", candidate(mutate), source_session_ids=["sess-2"])
    v = accept_proposal(store, p.proposal_id)
    assert v.parent == "v-base" and list(v.source_session_ids) == ["sess-2"] and not store.verify_version(v.version_id)
    new, old = store.load_version(v.version_id), store.load_version("v-base")
    assert "window" in new.entities and "door" in new.entities                 # removal needed review: not applied
    assert "window" not in old.entities and "door" in old.entities             # trusted base untouched
    done = get_proposal(store, p.proposal_id)
    assert done.status == "accepted" and done.decision["version_id"] == v.version_id \
        and done.decision["skipped"] == ["door"]
    with pytest.raises(ProposalError, match="already accepted"):
        accept_proposal(store, p.proposal_id)


def test_needs_review_change_applies_only_when_named(store):
    p = propose_update(store, "v-base", candidate(lambda w: w.entities.pop("door")))
    with pytest.raises(ProposalError, match="nothing to apply"):
        accept_proposal(store, p.proposal_id)
    v = accept_proposal(store, p.proposal_id, ["door"])
    assert "door" not in store.load_version(v.version_id).entities
    assert "g-door" not in store.load_version(v.version_id).geometries           # orphaned geometry went with it


def test_conflict_needs_override_and_reason_and_both_are_recorded(store):
    def mutate(w):
        w.geometries["g-wall-b"] = geom("g-wall-b", hi=(0.2, 5.5, 3), prov=Provenance.INFERRED)
        w.entities["wall-b"] = ent("wall-b", "g-wall-b", Provenance.INFERRED, 0.4, obs=("obs-x",))
    p = propose_update(store, "v-base", candidate(mutate))
    with pytest.raises(ProposalError, match="override_conflicts"):
        accept_proposal(store, p.proposal_id, ["wall-b"])
    with pytest.raises(ProposalError, match="reason"):
        accept_proposal(store, p.proposal_id, ["wall-b"], override_conflicts=["wall-b"])
    v = accept_proposal(store, p.proposal_id, ["wall-b"], override_conflicts=["wall-b"], reason="surveyor confirmed")
    d = get_proposal(store, p.proposal_id).decision
    assert d["overridden_conflicts"] == ["wall-b"] and d["reason"] == "surveyor confirmed"
    assert store.load_version(v.version_id).entities["wall-b"].provenance == Provenance.INFERRED


def test_invalid_merge_is_refused_and_nothing_is_saved(store):
    def mutate(w):
        w.geometries["g-bad"] = geom("g-bad", (5, 5, 5), (1, 1, 1))              # min > max
        w.entities["bad"] = ent("bad", "g-bad", obs=("obs-b",))
    p = propose_update(store, "v-base", candidate(mutate))
    assert not p.candidate_validation["valid"] and p.candidate_validation["errors"]
    before = [v.version_id for v in store.list_versions()]
    with pytest.raises(ProposalError, match="fails validation"):
        accept_proposal(store, p.proposal_id)
    assert [v.version_id for v in store.list_versions()] == before
    assert get_proposal(store, p.proposal_id).status == "pending"


def test_reject_is_recorded_requires_reason_and_is_final(store):
    p = propose_update(store, "v-base", candidate(lambda w: w.entities.pop("door")))
    with pytest.raises(ProposalError, match="reason"):
        reject_proposal(store, p.proposal_id, "  ")
    reject_proposal(store, p.proposal_id, "scan was partial")
    assert get_proposal(store, p.proposal_id).decision == {"reason": "scan was partial"}
    with pytest.raises(ProposalError, match="already rejected"):
        accept_proposal(store, p.proposal_id, ["door"])


def test_proposals_are_deterministic_idempotent_and_persist_across_store_instances(store, tmp_path):
    cand = candidate(lambda w: w.entities.pop("door"))
    p1 = propose_update(store, "v-base", cand)
    reject_proposal(store, p1.proposal_id, "no")
    p2 = propose_update(store, "v-base", copy.deepcopy(cand))
    assert p1.proposal_id == p2.proposal_id and p2.status == "rejected"          # returns the decided record, unchanged
    reopened = WorldStore(tmp_path / "ws")
    assert [p.proposal_id for p in list_proposals(reopened)] == [p1.proposal_id]
    with pytest.raises(ProposalError, match="unknown proposal"):
        get_proposal(reopened, "p-doesnotexist")
    with pytest.raises(ProposalError, match="unknown proposal"):
        get_proposal(reopened, "../versions/v-base")                              # no path traversal
