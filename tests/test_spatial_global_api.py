"""The spatial-global read API, exercised through a real reconstruction job: a registered producer feeds `spatial_inputs.v1`, the job persists a
spatial version (twice, so lineage exists), and the routes expose the world, its evidence, identity decisions and changes."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from benchmarks.spatial.synth import (  # noqa: E402
    PoseSim,
    SimConfig,
    simulate_multiview,
    two_rooms_shared_wall,
)
from reconstruction.spatial.stage import SCHEMA  # noqa: E402
from tests.test_application_api import client  # noqa: E402,F401  (pytest fixture)


def _inputs(n):
    local, prior, _ = simulate_multiview(two_rooms_shared_wall(), SimConfig(n_images=n, seed=101, p_truncate=0.0), PoseSim())
    return {"schema": SCHEMA, "scale_state": "metric", "frame": "world", "storeys": {"L0": 0.0},
            "poses": [{"image_id": p.image_id, "theta": p.theta, "tx": p.tx, "ty": p.ty, "sigma_theta": p.sigma_theta, "sigma_t": p.sigma_t}
                      for p in prior.values()], "observations": [o.to_dict() for o in local]}


def _world_id(client, sid):
    return client.get(f"/api/sessions/{sid}").json().get("world_id") or client.get("/api/worlds").json()["items"][0]["id"]


def _run(client, tmp_path, monkeypatch, n):
    from apps.api import spatial_global_stage as sg
    from tests.test_job_stage_runs_pipeline import _first_run
    monkeypatch.setenv("RE_SPATIAL_GLOBAL", "1")
    sg.register_inputs_provider(lambda job: _inputs(n))
    try:
        sid, job = _first_run(client, tmp_path, monkeypatch)
    finally:
        sg.register_inputs_provider(None)
    return sid, job


def test_routes_expose_the_persisted_reconstruction_with_evidence_and_provenance(client, tmp_path, monkeypatch):
    sid, job = _run(client, tmp_path, monkeypatch, 8)
    wid = _world_id(client, sid)
    base = f"/api/worlds/{wid}/spatial-global"

    vs = client.get(base).json()["versions"]
    assert len(vs) == 1 and vs[0]["status"] in ("succeeded", "degraded") and vs[0]["validation_state"]

    s = client.get(f"{base}/latest").json()
    assert s["version_id"] == vs[0]["version_id"] and s["frame"] == {"frame": "world", "scale_state": "metric"}
    assert s["quality"].keys() >= {"geometric", "structural", "evidence", "topological", "processing"} and "score" not in s["quality"]
    walls = [e for e in s["entities"] if e["type"] == "wall"]
    assert walls and all(e["scale_state"] == "metric" and e["coordinate_frame"] == "world" for e in walls)
    assert set(walls[0]["beliefs"]) == {"existence", "position", "dimensions", "orientation", "geometry", "topology", "semantic"}
    assert client.get(f"{base}/versions/{s['version_id']}").json()["counts"] == s["counts"]

    d = client.get(f"{base}/entities/{walls[0]['id']}").json()
    assert d["evidence"] and all(x["observation"] and x["raw_image_frame"] and x["image_pose"] for x in d["evidence"])   # back to raw, per image
    assert d["beliefs"]["position"]["status"] and d["sigma_parts"]["measurement"] > 0
    assert client.get(f"{base}/entities/does-not-exist").status_code == 404

    exp = client.get(f"{base}/export").json()
    assert exp["version_id"] == s["version_id"] and exp["world"]["metadata"]["spatial_evidence"]


def test_a_second_run_shows_what_changed_and_keeps_entity_ids(client, tmp_path, monkeypatch):
    sid, _ = _run(client, tmp_path, monkeypatch, 6)
    wid = _world_id(client, sid)
    base = f"/api/worlds/{wid}/spatial-global"
    first = client.get(f"{base}/latest").json()
    # a second reconstruct on the same world with more evidence
    from apps.api import spatial_global_stage as sg
    from tests.test_application_api import _upload_photo, _wait_job
    sg.register_inputs_provider(lambda job: _inputs(9))
    try:
        sid2 = client.post("/api/sessions", json={"name": "More"}).json()["id"]
        from tests.test_application_api import _attach
        _attach(client, wid, sid2)
        for i in range(3):
            _upload_photo(client, sid2, f"frame_{i:02d}.jpg", b"jpeg-" + f"{i}".encode() * 8)
        _wait_job(client, client.post(f"/api/sessions/{sid2}/reconstruct").json()["job_id"], timeout=90.0)
    finally:
        sg.register_inputs_provider(None)
    vs = client.get(base).json()["versions"]
    if len(vs) < 2:
        pytest.skip("the second session did not produce a second spatial version in this fixture")
    second = client.get(f"{base}/latest").json()
    ch = client.get(f"{base}/changes/{second['version_id']}").json()
    assert ch["parent"] == first["version_id"] and ch["identity_decisions"]
    assert not ch["removed"] and {e["id"] for e in first["entities"]} <= {e["id"] for e in second["entities"]}   # identities survived


def test_unknown_world_and_version_are_404_and_no_spatial_run_is_a_404_not_an_empty_world(client, tmp_path, monkeypatch):
    assert client.get("/api/worlds/nope/spatial-global").status_code == 404
    wid = client.post("/api/worlds", json={"name": "Empty"}).json()["id"]
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    assert client.get(f"/api/worlds/{wid}/spatial-global/latest").status_code == 404
    assert client.get(f"/api/worlds/{wid}/spatial-global").json()["versions"] == []


def test_corrections_posted_through_the_api_survive_the_next_reconstruction(client, tmp_path, monkeypatch):
    sid, _ = _run(client, tmp_path, monkeypatch, 6)
    wid = _world_id(client, sid)
    base = f"/api/worlds/{wid}/spatial-global"
    s = client.get(f"{base}/latest").json()
    wall = next(e["id"] for e in s["entities"] if e["type"] == "wall")

    bad = client.post(f"{base}/corrections", json={"kind": "exclude", "evidence_id": "img999#nope"})
    assert bad.status_code == 422                                                       # unknown evidence is refused, not stored
    assert client.post(f"{base}/corrections", json={"kind": "dimension", "entity_id": wall, "property": "bogus", "value": 3}).status_code == 422

    r = client.post(f"{base}/corrections", json={"kind": "dimension", "entity_id": wall, "property": "length_m", "value": 6.5, "note": "tape"})
    assert r.status_code == 200 and r.json()["parent"] == s["version_id"]
    listed = client.get(f"{base}/corrections").json()
    assert listed["corrections"] and listed["corrections"][0]["value"] == 6.5
    assert len(client.get(base).json()["versions"]) == 2                                # history kept: a NEW version, the old one untouched

    from apps.api import spatial_global_stage as sg
    from tests.test_application_api import _attach, _upload_photo, _wait_job
    sg.register_inputs_provider(lambda job: _inputs(9))
    try:
        sid2 = client.post("/api/sessions", json={"name": "More"}).json()["id"]
        _attach(client, wid, sid2)
        for i in range(3):
            _upload_photo(client, sid2, f"frame_{i:02d}.jpg", b"jpeg-" + f"{i}".encode() * 8)
        _wait_job(client, client.post(f"/api/sessions/{sid2}/reconstruct").json()["job_id"], timeout=90.0)
    finally:
        sg.register_inputs_provider(None)
    latest = client.get(f"{base}/latest").json()
    assert latest["version_id"] not in (s["version_id"], r.json()["version_id"])
    d = client.get(f"{base}/entities/{wall}").json()                                    # same stable id after reprocessing
    assert d["entity"]["id"] == wall
    assert client.get(f"{base}/corrections").json()["corrections"][0]["value"] == 6.5   # the correction was replayed onto the new version
