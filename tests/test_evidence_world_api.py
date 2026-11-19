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
    assert "not run" in hist[0]["reprocessing"]


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
