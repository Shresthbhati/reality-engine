"""Evidence-world API: entity -> images/predictions traceability, tenant isolation, traversal safety, and the gate on the legacy
shared-store routes. Engines are SYNTHETIC test doubles; the HTTP, auth and tenancy layers are the real application."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from engine.multiengine.contract import Task  # noqa: E402
from engine.multiengine.evidence_flow import StageSpec, run_evidence_flow  # noqa: E402
from tests.test_multiengine_evidence_flow import SOURCES, _setup  # noqa: E402

A = {"x-tenant-id": "tenant-a"}
B = {"x-tenant-id": "tenant-b"}
ROOT = {"x-tenant-id": "root"}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import importlib

    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _world(client, headers, name="w") -> str:
    r = client.post("/api/worlds", json={"name": name}, headers=headers)
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _seed(tmp_path, team, world_id, run_id="run-1", fail=()):
    from apps.api.evidence_world_stage import evidence_root

    reg, adapters = _setup(fail)
    root = evidence_root(team, world_id, run_id, base=tmp_path / "ws")
    return run_evidence_flow(root, run_id, SOURCES, registry=reg, adapters=adapters, stages=[StageSpec(Task.DETECTION)])


def _url(wid, *parts):
    return "/".join([f"/api/worlds/{wid}/evidence-world", *parts])


def test_entity_traces_back_to_images_and_model_predictions(client, tmp_path):
    wid = _world(client, A)
    _seed(tmp_path, "tenant-a", wid)
    runs = client.get(_url(wid), headers=A).json()["runs"]
    assert [r["run_id"] for r in runs] == ["run-1"] and runs[0]["counts"]["entities"] == 3
    run = client.get(_url(wid, "run-1"), headers=A).json()
    assert run["status"] == "complete" and any("2D" in x for x in run["limitations"])
    ents = client.get(_url(wid, "run-1", "entities"), headers=A).json()["entities"]
    chair = next(e for e in ents if e["ontology_path"] == "furniture/seating/chair")
    detail = client.get(_url(wid, "run-1", "entities", chair["entity_id"]), headers=A).json()
    obs = detail["observations"]
    assert len(obs) == 2 and {o["observation"]["engine_id"] for o in obs} == {"det_a", "det_b"}  # both engines' raw predictions
    assert all(o["observation"]["source_id"] in SOURCES for o in obs)  # the ORIGINAL image
    assert obs[0]["observation"]["engine_id"] in ("det_a", "det_b") and obs[0]["raw_confidence"] == 0.9
    assert obs[0]["calibrated_confidence"] is None and obs[0]["context"]["run_id"] == "run-1"
    assert detail["trace_problems"] == [] and detail["entity"]["geometry_ids"] == []
    assert detail["capture_requests"]  # an entity without measured size has a request, tied to it
    assert client.get(_url(wid, "run-1", "entities", "ent_nope"), headers=A).status_code == 404


def test_rows_runs_and_sections_are_exposed_with_failures_preserved(client, tmp_path):
    wid = _world(client, A)
    _seed(tmp_path, "tenant-a", wid, fail=("det_b",))
    runs = {r["engine_id"]: r for r in client.get(_url(wid, "run-1", "runs"), headers=A).json()["runs"]}
    assert runs["det_a"]["status"] == "succeeded" and runs["det_b"]["status"] == "failed"
    assert "synthetic crash" in runs["det_b"]["error"]
    rows = client.get(_url(wid, "run-1", "rows"), params={"engine_id": "det_a", "limit": 2}, headers=A).json()
    assert rows["total"] == 3 and len(rows["rows"]) == 2 and rows["rows"][0]["raw_confidence"] == 0.9
    assert client.get(_url(wid, "run-1", "rows"), params={"status": "rejected"}, headers=A).json()["total"] == 0
    for section in ("associations", "conflicts", "capture-requests", "hierarchy", "refinement"):
        assert client.get(_url(wid, "run-1", section), headers=A).status_code == 200, section
    assert client.get(_url(wid, "run-1", "hierarchy"), headers=A).json()["hierarchy"]["city_scale_claimable"] is False
    hist = client.get(_url(wid, "run-1", "refinement"), headers=A).json()["refinement_history"]
    assert hist[0]["stop_reason"] in {"target_met", "max_iterations", "no_improvement", "no_engine_available",
                                      "awaiting_evidence", "nothing_to_refine"}  # an explicit reason from the real loop
    assert "iterations" in hist[0] and "rolled_back_observations" in hist[0]


def test_rejected_predictions_are_exposed_raw_with_their_reason(client, tmp_path):
    from apps.api.evidence_world_stage import evidence_root
    from engine.multiengine.evidence_matrix import EvidenceMatrix, RejectedPrediction

    wid = _world(client, A)
    _seed(tmp_path, "tenant-a", wid)
    m = EvidenceMatrix(evidence_root("tenant-a", wid, "run-1", base=tmp_path / "ws"))
    m.record_rejected(RejectedPrediction("run-1", "detection", "det_a", "v1", 3, "non-finite number nan", {"predicted": {"v": "nan"}}))
    rej = client.get(_url(wid, "run-1", "rejected"), headers=A).json()["rejected"]
    assert len(rej) == 1 and rej[0]["reason"] == "non-finite number nan" and rej[0]["draft"]["predicted"] == {"v": "nan"}
    assert client.get(_url(wid, "run-1", "rejected"), headers=B).status_code in (403, 404)


def test_artifacts_are_served_only_when_the_run_references_them(client, tmp_path):
    wid = _world(client, A)
    _seed(tmp_path, "tenant-a", wid)
    run = next(r for r in client.get(_url(wid, "run-1", "runs"), headers=A).json()["runs"] if r["output_artifact"])
    digest = run["output_artifact"].removeprefix("artifact://")
    got = client.get(_url(wid, "run-1", "artifacts", digest), headers=A)
    assert got.status_code == 200 and got.content.startswith(b"[")  # the complete raw output bundle, as stored
    assert client.get(_url(wid, "run-1", "artifacts", "ab" * 32), headers=A).status_code == 404  # not referenced
    for bad in ("zz", "..%2F..%2Fetc%2Fpasswd", digest.upper(), digest[:-1]):
        assert client.get(_url(wid, "run-1", "artifacts", bad), headers=A).status_code == 404


def test_another_tenant_cannot_see_the_world_its_runs_entities_or_artifacts(client, tmp_path):
    wid = _world(client, A)
    _seed(tmp_path, "tenant-a", wid)
    ent = client.get(_url(wid, "run-1", "entities"), headers=A).json()["entities"][0]["entity_id"]
    runs = client.get(_url(wid, "run-1", "runs"), headers=A).json()["runs"]
    digest = next(r for r in runs if r["output_artifact"])["output_artifact"].removeprefix("artifact://")
    paths = ["", "run-1", "run-1/entities", f"run-1/entities/{ent}", "run-1/rows", "run-1/runs", "run-1/rejected", "run-1/associations",
             "run-1/conflicts", "run-1/capture-requests", "run-1/hierarchy", "run-1/refinement", f"run-1/artifacts/{digest}"]
    for path in paths:
        r = client.get(_url(wid, *([path] if path else [])), headers=B)
        assert r.status_code in (403, 404), (path, r.status_code)  # fail closed; never the data
        assert "snapshot" not in r.text and "observation" not in r.text
    wid_b = _world(client, B, "b-world")
    assert client.get(_url(wid_b, "run-1"), headers=B).status_code == 404  # tenant-a's run id does not exist in B's world
    assert client.get(_url(wid_b), headers=B).json()["runs"] == []
    assert client.get(_url(wid, "run-1"), headers=ROOT).status_code == 200  # a global administrator is the exception
    assert client.get(_url("world-does-not-exist", "run-1"), headers=A).status_code == 404


def test_a_damaged_or_missing_snapshot_is_an_error_never_an_empty_successful_world(client, tmp_path):
    from apps.api.evidence_world_stage import evidence_root

    wid = _world(client, A)
    _seed(tmp_path, "tenant-a", wid)  # run-1: healthy
    bad = evidence_root("tenant-a", wid, "run-bad", base=tmp_path / "ws")
    bad.mkdir(parents=True)
    (bad / "snapshot.json").write_text("{ not json", encoding="utf-8")
    for tail in ("entities", "rows", "refinement"):
        assert client.get(_url(wid, "run-bad", tail), headers=A).status_code == 500  # explicit error, not 200 with nothing
    assert client.get(_url(wid, "run-nothing"), headers=A).status_code == 404  # no snapshot at all: not found, not empty
    listing = {r["run_id"]: r for r in client.get(_url(wid), headers=A).json()["runs"]}
    assert listing["run-1"]["status"] == "complete" and listing["run-bad"]["status"] == "unreadable"  # visible, others unaffected
    (bad / "snapshot.json").write_text('{"unrelated": true}', encoding="utf-8")  # valid JSON but not a snapshot
    assert client.get(_url(wid, "run-bad"), headers=A).status_code == 500


def test_readiness_and_the_persisted_world_are_derived_from_persisted_data_and_tenant_scoped(client, tmp_path):
    wid = _world(client, A)
    _seed(tmp_path, "tenant-a", wid)  # runs the flow directly: nothing has persisted a WORLD yet
    assert client.get(_url(wid, "run-1", "world"), headers=A).status_code == 404  # not an empty 200
    rd = client.get(_url(wid, "run-1", "readiness"), headers=A).json()
    assert rd["physical_accuracy"] == "NOT_MEASURED" and rd["world"]["status"] == "not_persisted" and rd["export_ready"] is False
    assert rd["stage_status"] == "complete" and rd["scale"]["status"] == "no_depth_evidence" and rd["evidence_received"]["sources"] == 2
    assert client.get(_url(wid, "run-1", "readiness"), headers=B).status_code in (403, 404)  # tenant isolation
    assert client.get(_url(wid, "run-1", "world"), headers=B).status_code in (403, 404)


def test_a_persisted_world_is_served_verified_and_reflected_in_readiness(client, tmp_path):
    from apps.api import evidence_world_stage as ew
    from tests.test_multiengine_evidence_flow import _setup

    wid = _world(client, A)
    files = []
    for n in ("img1.jpg", "img2.jpg"):
        p = tmp_path / n
        p.write_bytes(b"\xff\xd8\xff\xd9")
        files.append(str(p))

    def inputs():
        reg, adapters = _setup(())
        return reg, adapters, None

    facts = ew.run_for_job(team_id="tenant-a", world_id=wid, job_id="run-2", sources=files, base=tmp_path / "ws", flow_inputs=inputs)
    got = client.get(_url(wid, "run-2", "world"), headers=A).json()
    assert got["version_id"] == facts["world_ir"]["version_id"] and len(got["world"]["entities"]) == 3
    rd = client.get(_url(wid, "run-2", "readiness"), headers=A).json()
    assert rd["world"]["status"] == "persisted" and rd["world"]["entities"] == 3 and rd["world"]["entities_with_position"] == 0
    assert rd["export_ready"] is False and "no entity has a measured position" in rd["export_not_ready_reason"]  # 2D-only: honest


def _persisted_run(client, tmp_path, wid, job="run-c"):
    from apps.api import evidence_world_stage as ew
    from tests.test_multiengine_evidence_flow import _setup

    files = []
    for n in ("img1.jpg", "img2.jpg"):
        p = tmp_path / n
        p.write_bytes(b"\xff\xd8\xff\xd9")
        files.append(str(p))

    def inputs():
        reg, adapters = _setup(())
        return reg, adapters, None

    return ew.run_for_job(team_id="tenant-a", world_id=wid, job_id=job, sources=files, base=tmp_path / "ws", flow_inputs=inputs)


def test_a_correction_is_a_new_version_that_keeps_every_piece_of_original_evidence(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    before = client.get(_url(wid, "run-c", "world"), headers=A).json()
    chair = next(e for e in client.get(_url(wid, "run-c", "entities"), headers=A).json()["entities"] if e["ontology_path"].endswith("chair"))
    r = client.post(_url(wid, "run-c", "entities", chair["entity_id"], "reclassify"), json={"new_class": "furniture/seating/sofa", "note": "it is a sofa"}, headers=A)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["parent_version_id"] == before["version_id"] and out["version_id"] != before["version_id"] and out["original_evidence_preserved"]
    after = client.get(_url(wid, "run-c", "world"), headers=A).json()  # head is the NEW version (store sequence, not file name)
    assert after["version_id"] == out["version_id"]
    old_e, new_e = before["world"]["entities"][f"ent-{chair['entity_id']}"], after["world"]["entities"][f"ent-{chair['entity_id']}"]
    assert new_e["custom_properties"]["evidence"] == old_e["custom_properties"]["evidence"]  # the evidence block is untouched
    assert [o["id"] for o in new_e["observations"]] == [o["id"] for o in old_e["observations"]]
    assert new_e["custom_properties"]["user_decision"]["old_class"] and new_e["custom_properties"]["classification_basis"] == "user"
    # the raw layer is never rewritten: the snapshot and matrix still describe the machine's own claim
    detail = client.get(_url(wid, "run-c", "entities", chair["entity_id"]), headers=A).json()
    assert detail["trace_problems"] == [] and len(detail["observations"]) == 2


def test_a_correction_is_validated_tenant_scoped_and_404s_without_a_world_or_entity(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    ents = client.get(_url(wid, "run-c", "entities"), headers=A).json()["entities"]
    eid = ents[0]["entity_id"]
    assert client.post(_url(wid, "run-c", "entities", "nope", "reclassify"), json={"new_class": "x"}, headers=A).status_code == 404
    assert client.post(_url(wid, "run-c", "entities", eid, "reclassify"), json={"new_class": ""}, headers=A).status_code == 422
    assert client.post(_url(wid, "run-c", "entities", eid, "reclassify"), json={"new_class": "x/y"}, headers=B).status_code in (403, 404)
    assert client.post(_url(wid, "no-run", "entities", eid, "reclassify"), json={"new_class": "x/y"}, headers=A).status_code == 404


def test_export_is_refused_when_nothing_is_positioned(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)  # 2D-only: entities exist, none has a position
    r = client.post(_url(wid, "run-c", "export"), json={"format": "glb", "acknowledge_unscaled": True}, headers=A)
    assert r.status_code == 409 and "no entity has a measured position" in r.json()["detail"]


def test_gauge_scale_geometry_exports_only_after_an_explicit_acknowledgement_and_says_it_is_not_metres(client, tmp_path):
    from PIL import Image

    from apps.api import evidence_world_stage as ew
    import os

    from tests.test_lift_stage_e2e import DX, H, W, _camera, _Depth, _inputs, _per_view, _points

    wid = _world(client, A)
    photos = []
    for n in ("a.png", "b.png"):
        p = tmp_path / n
        Image.new("RGB", (W, H), (120, 120, 120)).save(p)
        photos.append(str(p))
    ew.run_for_job(team_id="tenant-a", world_id=wid, job_id="run-g", sources=photos, base=tmp_path / "ws", flow_inputs=_inputs(_Depth(_per_view())),
                   depth_engines=["fake_depth"], cameras={p: _camera(DX[os.path.basename(p)]) for p in photos}, points3d=_points())
    rd = client.get(_url(wid, "run-g", "readiness"), headers=A).json()
    assert rd["export_ready"] is True and rd["metric_export_ready"] is False
    refused = client.post(_url(wid, "run-g", "export"), json={"format": "glb"}, headers=A)
    assert refused.status_code == 409 and "acknowledge_unscaled" in refused.json()["detail"]
    ok = client.post(_url(wid, "run-g", "export"), json={"format": "glb", "acknowledge_unscaled": True}, headers=A)
    assert ok.status_code == 200 and ok.headers["x-export-scale"] == "sfm_units-NOT-metres" and ok.headers["x-physical-accuracy"] == "NOT_MEASURED"
    assert int(ok.headers["x-entities-exported"]) >= 1 and ok.content[:4] == b"glTF"  # the object plus its observed surfaces
    assert client.post(_url(wid, "run-g", "export"), json={"format": "glb", "acknowledge_unscaled": True}, headers=B).status_code in (403, 404)


def test_the_manifest_route_returns_the_manifest_with_a_fresh_verification_and_is_tenant_scoped(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    r = client.get(_url(wid, "run-c", "manifest"), headers=A)
    assert r.status_code == 200
    body = r.json()
    assert body["manifest"]["run_id"] == "run-c" and body["verification"]["ok"] is True and body["manifest"]["result"]["world"]["status"] == "persisted"
    assert client.get(_url(wid, "run-c", "manifest"), headers=B).status_code in (403, 404)
    assert client.get(_url(wid, "no-run", "manifest"), headers=A).status_code == 404


def _ents(client, wid, run="run-c"):
    return client.get(_url(wid, run, "entities"), headers=A).json()["entities"]


def test_a_merge_correction_rebuilds_the_entity_from_the_raw_evidence_and_keeps_the_original_layer(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    chairs = [e for e in _ents(client, wid) if e["ontology_path"].endswith("chair")]
    assert len(chairs) == 2
    a, b = chairs[0]["entity_id"], chairs[1]["entity_id"]
    before = client.get(_url(wid, "run-c", "world"), headers=A).json()
    r = client.post(_url(wid, "run-c", "entities", a, "merge"), json={"other_entity_id": b, "note": "same chair, two photos"}, headers=A)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["rebuilt_from_evidence"] and out["members"] == len(set(chairs[0]["members"]) | set(chairs[1]["members"])) and out["parent_version_id"] == before["version_id"]
    after = client.get(_url(wid, "run-c", "world"), headers=A).json()["world"]
    assert f"ent-{b}" not in after["entities"] and f"ent-{a}" in after["entities"] and len(after["entities"]) == len(before["world"]["entities"]) - 1
    merged = after["entities"][f"ent-{a}"]
    ev = merged["custom_properties"]["evidence"]
    assert sorted(ev["members"]) == sorted(set(chairs[0]["members"]) | set(chairs[1]["members"])) and {o["id"] for o in merged["observations"]} == set(ev["members"])
    assert merged["custom_properties"]["user_decision"]["kind"] == "merge" and merged["custom_properties"]["merged_from"] == [f"ent-{b}"]
    # the raw layer is untouched: the run's snapshot still lists BOTH machine entities, and the merged one still traces to its evidence
    assert len(_ents(client, wid)) == len(before["world"]["entities"])
    assert client.get(_url(wid, "run-c", "entities", a), headers=A).json()["trace_problems"] == []


def test_merging_entities_from_one_photograph_is_refused_unless_acknowledged_and_conflicting_classes_are_kept(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    ents = _ents(client, wid)
    chair = next(e for e in ents if e["ontology_path"].endswith("chair"))
    socket = next(e for e in ents if "socket" in e["ontology_path"])
    url = _url(wid, "run-c", "entities", chair["entity_id"], "merge")
    refused = client.post(url, json={"other_entity_id": socket["entity_id"]}, headers=A)
    chair_src = {o["observation"]["source_id"] for o in client.get(_url(wid, "run-c", "entities", chair["entity_id"]), headers=A).json()["observations"]}
    socket_src = {o["observation"]["source_id"] for o in client.get(_url(wid, "run-c", "entities", socket["entity_id"]), headers=A).json()["observations"]}
    if chair_src & socket_src:
        assert refused.status_code == 422 and "same photograph" in refused.json()["detail"]
        ok = client.post(url, json={"other_entity_id": socket["entity_id"], "acknowledge_same_image": True}, headers=A)
        assert ok.status_code == 200 and ok.json()["decision"]["acknowledged_same_image"] is True
        merged = client.get(_url(wid, "run-c", "world"), headers=A).json()["world"]["entities"][f"ent-{chair['entity_id']}"]
        assert merged["custom_properties"]["evidence"]["class_status"] in ("conflict", "resolved")  # a class disagreement is kept, never silently resolved
    else:
        assert refused.status_code == 200


def test_merge_validation_and_tenancy(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    e = _ents(client, wid)[0]["entity_id"]
    url = _url(wid, "run-c", "entities", e, "merge")
    assert client.post(url, json={"other_entity_id": e}, headers=A).status_code == 422
    assert client.post(url, json={"other_entity_id": "nope"}, headers=A).status_code == 404
    assert client.post(url, json={"other_entity_id": "x"}, headers=B).status_code in (403, 404)


def test_a_merge_then_a_split_restores_two_entities_each_rebuilt_from_its_own_evidence(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    chairs = [e for e in _ents(client, wid) if e["ontology_path"].endswith("chair")]
    a, b = chairs[0]["entity_id"], chairs[1]["entity_id"]
    assert client.post(_url(wid, "run-c", "entities", a, "merge"), json={"other_entity_id": b}, headers=A).status_code == 200
    merged = client.get(_url(wid, "run-c", "world"), headers=A).json()["world"]["entities"][f"ent-{a}"]
    ids_b = [m for m in chairs[1]["members"] if m in {o["id"] for o in merged["observations"]}]
    r = client.post(_url(wid, "run-c", "entities", a, "split"), json={"detach_observation_ids": ids_b, "note": "actually two chairs"}, headers=A)
    assert r.status_code == 200, r.text
    out = r.json()
    world = client.get(_url(wid, "run-c", "world"), headers=A).json()["world"]["entities"]
    new_id = f"ent-{out['new_entity_id']}"
    assert new_id in world and f"ent-{a}" in world and out["rebuilt_from_evidence"]
    moved = {o["id"] for o in world[new_id]["observations"]}
    kept = {o["id"] for o in world[f"ent-{a}"]["observations"]}
    assert moved and kept and not (moved & kept) and set(ids_b) <= moved  # a partition of the evidence: nothing duplicated, nothing lost
    assert world[new_id]["custom_properties"]["split_from"] == f"ent-{a}" and world[new_id]["relationships"] == []


def test_split_refuses_bad_requests_and_cannot_empty_the_original(client, tmp_path):
    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    e = _ents(client, wid)[0]
    url = _url(wid, "run-c", "entities", e["entity_id"], "split")
    assert client.post(url, json={"detach_observation_ids": ["not-an-observation"]}, headers=A).status_code == 422
    detail = client.get(_url(wid, "run-c", "entities", e["entity_id"]), headers=A).json()
    every_bbox = [o["observation"]["observation_id"] for o in detail["observations"] if o["observation"]["quantity"] == "bbox"]
    r = client.post(url, json={"detach_observation_ids": every_bbox}, headers=A)
    assert r.status_code == 422 and "nothing would stay" in r.json()["detail"]
    assert client.post(_url(wid, "run-c", "entities", "nope", "split"), json={"detach_observation_ids": ["x"]}, headers=A).status_code == 404
    assert client.post(url, json={"detach_observation_ids": every_bbox[:1]}, headers=B).status_code in (403, 404)


def _commit_head(tmp_path, wid, version_id="v-head1"):
    """A committed HEAD for the world (a small wall/door world), written the way the app does: WorldStore version + DB mirror row + HEAD pointer."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from apps.api.models import World, WorldVersion
    from tests.test_world_proposals import base_world
    from worldstore.store import WorldStore

    w = base_world()
    w.id = wid
    stored = WorldStore(tmp_path / "ws").save_version(w, parent=None, version_id=version_id)
    eng = create_engine(f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    with Session(eng) as db:
        db.add(WorldVersion(id=stored.version_id, world_id=wid, artifact_uri=stored.artifact_uri, artifact_hash=stored.artifact_hash,
                            source_session_ids=[], changed_entity_ids=[], changed_geometry_ids=[], label="as-built"))
        db.flush()
        db.get(World, wid).current_version_id = stored.version_id
        db.commit()
    eng.dispose()
    return version_id


def test_propose_is_refused_until_the_frame_is_proven_and_never_commits(client, tmp_path):
    from apps.api import evidence_world_stage as ew

    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    url = _url(wid, "run-c", "propose")
    assert client.post(url, json={}, headers=A).status_code == 409  # no committed version at all
    head = _commit_head(tmp_path, wid)
    r = client.post(url, json={}, headers=A)
    assert r.status_code == 409 and "frame not proven" in r.json()["detail"]  # HEAD exists but nothing ties this run to it
    assert ew.record_committed_version("tenant-a", wid, "run-c", "v-some-other-version", base=tmp_path / "ws")
    assert client.post(url, json={}, headers=A).status_code == 409  # the run accompanied a different version than HEAD
    assert client.get(f"/api/worlds/{wid}/proposals", headers=A).json()["proposals"] == []
    assert client.post(url, json={}, headers=B).status_code in (403, 404)
    assert head


def test_propose_then_accept_adds_the_evidence_entities_and_leaves_heads_own_untouched(client, tmp_path):
    from apps.api import evidence_world_stage as ew
    from worldstore.store import WorldStore

    wid = _world(client, A)
    _persisted_run(client, tmp_path, wid)
    head = _commit_head(tmp_path, wid)
    assert ew.record_committed_version("tenant-a", wid, "run-c", head, base=tmp_path / "ws")
    r = client.post(_url(wid, "run-c", "propose"), json={"note": "promote"}, headers=A)
    assert r.status_code == 201, r.text
    out = r.json()
    ev_ids = set(client.get(_url(wid, "run-c", "world"), headers=A).json()["world"]["entities"])
    prop = out["proposal"]
    assert out["added_entities"] == len(ev_ids) and out["frame_check"]["head_version"] == head
    assert {a["entity_id"] for a in prop["assessments"]} == ev_ids and {a["kind"] for a in prop["assessments"]} == {"added"}
    # nothing is committed by proposing: HEAD is still the original version
    assert [v["version_id"] if isinstance(v, dict) else v for v in [client.get(f"/api/worlds/{wid}", headers=A).json().get("current_version_id")]] == [head]
    acc = client.post(f"/api/worlds/{wid}/proposals/{prop['proposal_id']}/accept", json={"entity_ids": sorted(ev_ids)}, headers=A)
    assert acc.status_code == 200, acc.text
    merged = WorldStore(tmp_path / "ws").load_version(acc.json()["version_id"])
    assert ev_ids <= set(merged.entities) and {"wall-a", "wall-b", "door"} <= set(merged.entities)
    sample = next(iter(ev_ids))
    assert merged.entities[sample].custom_properties["promoted_from_evidence_run"] == "run-c"
    assert merged.entities[sample].custom_properties["evidence_world_version"]  # provenance back to the evidence run


def test_run_ids_cannot_traverse_out_of_the_world_directory(client, tmp_path):
    wid = _world(client, A)
    _seed(tmp_path, "tenant-a", wid)
    for rid in ("a..b", "%2e%2e", "x" * 300, "bad%20id"):  # what survives the HTTP client's own URL normalisation
        assert client.get(_url(wid, rid), headers=A).status_code == 404, rid
    # ".." / "." never reach the handler over HTTP (clients collapse them), so the guard itself is checked directly
    from fastapi import HTTPException

    from apps.api.routes_evidence_world import _run_dir

    base = tmp_path / "ws" / "evidence-world" / "tenant-a" / wid
    assert base.is_dir()
    for rid in ("..", ".", "../..", "a/b", "", "/abs"):
        with pytest.raises(HTTPException) as ei:
            _run_dir(base, rid)
        assert ei.value.status_code == 404


def test_legacy_shared_store_routes_are_restricted_to_global_administrators(client, tmp_path):
    body = {"sources": [str(tmp_path / "nope.jpg")]}
    assert client.post("/api/engines/reconstruct", json=body, headers=A).status_code == 403  # was open to any tenant
    assert client.get("/api/engines/evidence", headers=A).status_code == 403  # was a cross-tenant leak
    assert client.get("/api/engines/evidence", headers=ROOT).status_code == 200
    assert client.get("/api/engines", headers=A).status_code == 200  # the read-only registry view stays open
    assert client.get("/api/engines/stages", headers=A).status_code == 200
