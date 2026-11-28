"""Evidence-world correction routes: mandatory base version, typed correction record, lineage (Sprint 2, workstream 6)."""

from __future__ import annotations

import pytest

from tests.test_evidence_world_api import A, _ents, _ev, _persisted_run, _url, _world, client  # noqa: F401
from world_ir.correction_log import correction_log, lineage_successors


def _chairs(client, wid):
    return [e["entity_id"] for e in _ents(client, wid) if e["ontology_path"].endswith("chair")]


def test_every_evidence_correction_route_requires_the_base_version(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    a, b = _chairs(client, wid)[:2]
    before = _ev(client, wid)
    calls = [
        ("reclassify", a, {"new_class": "furniture/seating/sofa"}),
        ("merge", a, {"other_entity_id": b}),
        ("split", a, {"detach_observation_ids": ["x"]}),
    ]
    for op, ent, body in calls:
        r = client.post(_url(wid, "run-c", "entities", ent, op), json=body, headers=A)
        assert r.status_code == 428 and "base version is required" in r.json()["detail"] and before in r.json()["detail"], (op, r.text)
        stale = client.post(_url(wid, "run-c", "entities", ent, op), json={**body, "expected_version_id": "v-old"}, headers=A)
        assert stale.status_code == 409 and "stale" in stale.json()["detail"] and before in stale.json()["detail"], (op, stale.text)
    cal = client.post(_url(wid, "run-c", "calibrate"), json={"distance_anchors": [{"anchor_id": "x"}]}, headers=A)
    assert cal.status_code in (422,)                                   # payload shape is judged first
    assert _ev(client, wid) == before                                  # nothing was written by any refused request


def test_a_reclassify_and_a_merge_write_a_complete_record_and_lineage_into_the_new_version(client, tmp_path):
    from worldstore.store import WorldStore

    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    a, b = _chairs(client, wid)[:2]
    v0 = _ev(client, wid)
    r = client.post(_url(wid, "run-c", "entities", a, "reclassify"), json={"new_class": "furniture/seating/sofa", "note": "sofa",
                                                                           "expected_version_id": v0}, headers=A)
    assert r.status_code == 200, r.text
    v1 = r.json()["version_id"]
    m = client.post(_url(wid, "run-c", "entities", a, "merge"), json={"other_entity_id": b, "expected_version_id": v1}, headers=A)
    assert m.status_code == 200, m.text
    v2 = m.json()["version_id"]
    store = WorldStore(tmp_path / "ws" / "evidence-world" / "tenant-a" / wid / "run-c" / "worldstore")
    w2 = store.load_version(v2)
    recs = correction_log(w2)
    assert [(x.operation, x.base_version_id, x.new_version_id) for x in recs] == [("reclassify", v0, v1), ("merge", v1, v2)]
    for rec in recs:
        assert rec.missing_fields() == [] and rec.actor_kind == "user" and rec.changes
    assert recs[0].evidence_untouched is True                          # a reclassify never touches evidence
    assert recs[1].lineage == [{"from_ids": [f"ent-{b}"], "to_id": f"ent-{a}", "relation": "merge"}]
    assert lineage_successors(w2, f"ent-{b}") == [f"ent-{a}"]          # the absorbed id still resolves to the survivor
    assert correction_log(store.load_version(v0)) == []                # the parent version is untouched
