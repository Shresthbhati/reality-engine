"""RELEASE GATE: real photographs -> real COLMAP -> validated WorldIR -> persisted V1 that survives a restart ->
data the Studio reads == persisted data -> correction -> V2 with the right parent and an exact diff -> honest failure.

Nothing here is mocked except the one failure injection at the end (a backend that returns a NaN pose), which exists
only to prove malformed output can never become a world version. Every other step runs the real engine on the
South Building photographs (datasets/south_building, see MANIFEST.json), through the same HTTP routes the Studio calls.

Run:  python -m pytest -q tests/integration/test_release_gate.py      (about 10-15 minutes on CPU)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import pytest

from tests.integration.test_progressive_product_journey import (
    HARD,
    SIX,
    _OK,
    _WHY,
    IMGS,
    _ply_vertices,
    _post,
    app_client,
)

pytest.importorskip("fastapi")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.e2e,
    pytest.mark.skipif(len(IMGS) < 25, reason="BLOCKED: South Building dataset not present"),
    pytest.mark.skipif(not _OK, reason=f"BLOCKED: real engines required ({_WHY})"),
]

TERMINAL = ("succeeded", "partial", "failed", "cancelled")
#: run-to-run tolerance on the served point count for identical input (measured 0.9% across two clean runs; see
#: docs/implementation/RELEASE_GATE_PLAN.md Phase 4)
POINT_COUNT_TOLERANCE = 0.05


class _NanPoseBackend:
    """Failure injection: a backend that 'finishes' but reports a non-finite camera pose."""

    def reconstruct(self, evidence):
        import dataclasses

        from tests.test_cli_compile import _TwoViewBackend

        res = _TwoViewBackend().reconstruct(evidence)
        bad = dataclasses.replace(res.camera_poses[0], position=(math.nan, 0.0, 0.0))
        return dataclasses.replace(res, camera_poses=[bad, *res.camera_poses[1:]])


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _watch_job(c, job_id: str, timeout: float = 1200.0) -> tuple[dict, list[tuple[str, str | None]]]:
    """Follow a job to a terminal state, recording every distinct (status, stage) observed."""
    seen: list[tuple[str, str | None]] = []
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/api/jobs/{job_id}").json()
        key = (j["status"], j.get("stage"))
        if not seen or seen[-1] != key:
            seen.append(key)
        if j["status"] in TERMINAL:
            return j, seen
        time.sleep(0.4)
    raise AssertionError(f"job {job_id} never reached a terminal state; observed {seen}")


def _versions(c, wid: str) -> list[dict]:
    r = c.get(f"/api/worlds/{wid}/versions")
    assert r.status_code == 200, r.text
    return r.json()["items"]


def _stored_world(root: Path, version_id: str) -> dict:
    """The version straight from disk, bypassing every API route."""
    from worldstore.store import WorldStore

    return WorldStore(root / "ws").load_version(version_id).to_dict()


def _independent_point_count(root: Path, world: dict) -> int:
    """Points the world's geometry artifacts really contain, decoded without the /points route."""
    from apps.api.routes_worlds import _decode_point_payload
    from world_ir.artifact_store import FileArtifactStore

    stores = [FileArtifactStore(root / "ws" / "pipeline-artifacts"), FileArtifactStore(root / "ws" / "artifacts")]
    total = 0
    for g in world["geometries"].values():
        uri = g.get("data_uri")
        if not uri:
            continue
        for st in stores:
            try:
                raw = st.get(uri)
            except Exception:
                continue
            total += len(_decode_point_payload(raw))
            break
    return total


def _assert_canonical(world: dict, evidence_ids: set[str]) -> None:
    from world_ir.validation import validate_world_ir
    from world_ir.world_v1 import WorldIR

    wir = WorldIR.from_dict(world)
    report = validate_world_ir(wir)
    assert report.is_valid(), report.messages()[:5]
    assert wir.validate() == [], wir.validate()[:5]            # dangling entity/geometry/material/surface refs
    assert world["entities"], "a reconstruction with no entities is not a world"
    for eid, e in world["entities"].items():
        for gid in e["geometry_ids"]:
            assert gid in world["geometries"], f"{eid} -> dangling geometry {gid}"
        assert e["provenance"] and e["provenance"] != "OBSERVED", (eid, e["provenance"])   # SfM output is computed
        assert isinstance(e["uncertainty"], dict) and "confidence" in e["uncertainty"], eid
        assert 0.0 <= e["confidence"] <= 1.0, eid
        if e["provenance"] in ("RECONSTRUCTED", "INFERRED"):
            trace = (e.get("custom_properties") or {}).get("reconstruction")
            assert trace and trace["session_id"], f"{eid} has no reconstruction trace"
    for gid, g in world["geometries"].items():
        assert g["provenance"], gid
    assert world["coordinate_frame"] and world["metadata"]["frame"]["up_source"], "no coordinate frame record"
    used = {u["evidence_id"] for u in world["metadata"]["evidence"]["used"]}
    assert used <= evidence_ids and used, "world cites evidence that was never uploaded"


# ----------------------------------------------------------------------------------------------- the journey


def test_release_gate_real_photos_to_corrected_v2_and_honest_failure(tmp_path, monkeypatch):
    root = tmp_path
    inputs = {p.name: p.read_bytes() for p in SIX}

    # ---- 1. upload SIX real photographs; stored bytes and metadata are the input's
    with app_client(root) as c:
        body = _post(c, SIX)
        wid, job_id = body["world_id"], body["job_id"]
        assert body["created_world"] and not body["rejected"] and len(body["evidence"]) == 6
        ev_ids = set()
        for item in body["evidence"]:
            ev = c.get(f"/api/evidence/{item['id']}").json()
            ev_ids.add(ev["id"])
            assert ev["name"] in inputs, ev["name"]                                # original filename kept
            want = inputs[ev["name"]]
            assert ev["checksum"] == hashlib.sha256(want).hexdigest()              # sha256 == the input's
            assert ev["size"] == len(want)
            art = c.get(f"/api/evidence/{ev['id']}/artifact")
            assert art.status_code == 200 and art.content == want                  # stored bytes are byte-identical
            facts = ev["metadata"]["image_facts"]
            assert facts["ok"] and facts["width"] > 0 and facts["height"] > 0 and isinstance(facts["exif"], dict)
        assert len({e["checksum"] for e in (c.get(f"/api/evidence/{i}").json() for i in ev_ids)}) == 6

        # ---- 2. reconstruct: observe the real job lifecycle, then real output
        job, seen = _watch_job(c, job_id)
        assert job["status"] == "succeeded", (job["status"], job.get("error"), seen)
        assert len(seen) >= 2 and seen[-1][0] == "succeeded", seen                 # it was observed running first
        assert any(s[0] in ("queued", "running") for s in seen[:-1]), seen
        pl = job["payload"]
        assert pl["adopted"] is True and pl["validated"] is True and pl["registration_status"] in ("success", "partial")
        st = c.get(f"/api/worlds/{wid}/status").json()
        m = st["model"]
        assert st["state"] in ("READY_TO_INSPECT", "PARTIALLY_COMPLETE") and not st["in_progress"]
        assert m["images_used"] == 6 and m["images_registered"] >= 4, m
        assert m["level"] >= 2, m["attempts"]                                      # real multi-view, not a fallback
        assert pl["cameras_registered"] == m["images_registered"] and pl["points"] > 0

        cams = c.get(f"/api/worlds/{wid}/cameras").json()
        assert len(cams["cameras"]) == m["images_registered"]
        for cam in cams["cameras"]:
            assert cam["evidence_id"] in ev_ids                                    # a camera per uploaded photo, no invented ones
            assert all(math.isfinite(v) for v in (*cam["position_m"], *cam["rotation_wxyz"]))
            assert abs(math.sqrt(sum(q * q for q in cam["rotation_wxyz"])) - 1.0) < 1e-3, "rotation is not a unit quaternion"

        # ---- 3. canonical WorldIR validates; refs resolve; uncertainty + provenance present
        v1 = c.get(f"/api/worlds/{wid}").json()["current_version_id"]
        assert v1 == pl["version_id"]
        v1_api = c.get(f"/api/worlds/{wid}/worldir").json()
        _assert_canonical(v1_api, ev_ids)

        # ---- 4. persisted V1 (lineage) ...
        vs = _versions(c, wid)
        assert [(v["id"], v["parent_version_id"], v["is_current"]) for v in vs] == [(v1, None, True)]

        # ---- 5. ... and the Studio's data source IS the persisted data
        on_disk = _stored_world(root, v1)
        assert _digest(v1_api) == _digest(on_disk), "/worldir differs from what is on disk"
        ply = c.get(f"/api/worlds/{wid}/points").content
        n_ply = _ply_vertices(ply)
        assert n_ply > 0 and n_ply == _independent_point_count(root, on_disk), "/points differs from stored artifacts"
        assert b"placeholder" not in ply[:400].lower()
        v1_digest, v1_points, v1_cams = _digest(on_disk), n_ply, {c_["evidence_id"] for c_ in cams["cameras"]}

    # ---- 4b. restart: a new app process over the same storage returns the same version and content
    with app_client(root) as c:
        assert c.get(f"/api/worlds/{wid}").json()["current_version_id"] == v1
        assert [v["id"] for v in _versions(c, wid)] == [v1]
        assert _digest(c.get(f"/api/worlds/{wid}/worldir").json()) == v1_digest
        assert _ply_vertices(c.get(f"/api/worlds/{wid}/points").content) == v1_points
        assert {x["evidence_id"] for x in c.get(f"/api/worlds/{wid}/cameras").json()["cameras"]} == v1_cams

        # ---- 6. correction: invalid -> 422, HEAD stays; valid -> V2; stale parent -> 409
        target = next(eid for eid, e in v1_api["entities"].items() if e["type"] == "wall")
        old_name = v1_api["entities"][target]["name"]
        bad = c.post(f"/api/worlds/{wid}/commit", json={"entity_id": target, "changes": {"confidence": 7}})
        assert bad.status_code == 422, bad.text
        assert c.get(f"/api/worlds/{wid}").json()["current_version_id"] == v1
        assert [v["id"] for v in _versions(c, wid)] == [v1]

        change = {"entity_id": target, "changes": {"name": "Corrected east facade"}, "parent_version_id": v1,
                  "commit_message": "release gate correction"}
        ok = c.post(f"/api/worlds/{wid}/commit", json=change)
        assert ok.status_code == 200, ok.text
        res = ok.json()
        assert res["duplicate"] is False and res["changed_fields"] == ["name"] and res["entity_id"] == target
        v2 = res["version_id"]
        assert v2 != v1 and c.get(f"/api/worlds/{wid}").json()["current_version_id"] == v2

        stale = c.post(f"/api/worlds/{wid}/commit", json={**change, "changes": {"name": "late edit"}, "parent_version_id": v1})
        assert stale.status_code == 409, stale.text
        assert c.get(f"/api/worlds/{wid}").json()["current_version_id"] == v2

        # ---- 14. a byte-identical retry is reported as a duplicate and mints no version
        retry = c.post(f"/api/worlds/{wid}/commit", json={**change, "parent_version_id": v2})
        assert retry.status_code == 200 and retry.json()["duplicate"] is True and retry.json()["version_id"] == v2
        assert retry.json()["changed_fields"] == []

        # ---- 7. V2's parent is V1; the diff lists exactly the corrected field
        vs = _versions(c, wid)
        assert [(v["id"], v["parent_version_id"], v["is_current"]) for v in vs] == [(v2, v1, True), (v1, None, False)]
        d = c.get(f"/api/worlds/{wid}/diff", params={"base": v1, "head": v2}).json()
        assert [(x["entity_id"], x["kind"]) for x in d["entity_diffs"]] == [(target, "modified")]
        assert [(ch["field"], ch["old"], ch["new"]) for ch in d["entity_diffs"][0]["changes"]] == [
            ("name", old_name, "Corrected east facade")]
        assert d["geometry_diffs"] == []
        s = d["summary"]
        assert (s["entities_added"], s["entities_removed"], s["entities_modified"]) == (0, 0, 1)
        assert (s["geometries_added"], s["geometries_removed"], s["geometries_modified"]) == (0, 0, 0)
        assert sum(d["categories"].values()) == 1 and d["categories"]["semantic"] == 1
        # V1 is immutable: re-read after V2 exists, byte-for-byte what it was
        assert _digest(c.get(f"/api/worlds/{wid}/worldir", params={"version": v1}).json()) == v1_digest

        # ---- 13. malformed reconstruction output can never become a version, and HEAD stays V2
        monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.integration.test_release_gate:_NanPoseBackend")
        extra = _post(c, [IMGS[21]], world_id=wid)
        bad_job, bad_seen = _watch_job(c, extra["job_id"], timeout=600.0)
        monkeypatch.delenv("REALITY_TEST_BACKEND")
        assert bad_job["status"] != "succeeded" or bad_job["payload"].get("adopted") is False, (bad_job, bad_seen)
        assert c.get(f"/api/worlds/{wid}").json()["current_version_id"] == v2
        assert [v["id"] for v in _versions(c, wid)] == [v2, v1], "malformed output minted a version"
        assert _digest(c.get(f"/api/worlds/{wid}/worldir").json()) == _digest(_stored_world(root, v2))

    # ---- 12. restart again: V2, its lineage and its diff survive
    with app_client(root) as c:
        assert c.get(f"/api/worlds/{wid}").json()["current_version_id"] == v2
        assert [(v["id"], v["parent_version_id"]) for v in _versions(c, wid)] == [(v2, v1), (v1, None)]
        d2 = c.get(f"/api/worlds/{wid}/diff", params={"base": v1, "head": v2}).json()
        assert [(x["entity_id"], x["kind"]) for x in d2["entity_diffs"]] == [(target, "modified")]
        assert c.get(f"/api/worlds/{wid}/worldir").json()["entities"][target]["name"] == "Corrected east facade"


def test_release_gate_hard_window_is_never_a_success(tmp_path):
    """Wide-baseline photographs real COLMAP cannot register: a labelled degraded result or a failure, never success."""
    with app_client(tmp_path) as c:
        body = _post(c, HARD)
        wid = body["world_id"]
        job, seen = _watch_job(c, body["job_id"])
        assert job["status"] in ("partial", "failed"), (job["status"], seen)       # NEVER 'succeeded'
        st = c.get(f"/api/worlds/{wid}/status").json()
        vs = _versions(c, wid)
        if job["status"] == "failed":
            assert job.get("error"), "a failure must carry its reason"
            assert vs == [] and st["state"] != "READY_TO_INSPECT"
            return
        # degraded: explicitly labelled, limitations stated, no registered geometry claimed
        assert st["state"] == "PARTIALLY_COMPLETE", st["state"]
        m = st["model"]
        assert m["level"] == 0 and m["model_state"] == "ROUGH" and m["images_registered"] == 0, m
        failed = [a for a in m["attempts"] if a["outcome"] == "failed"]
        assert failed and failed[0]["level"] == 2 and failed[0]["detail"], m["attempts"]
        assert any(g["kind"] == "add_overlap" for g in st["guidance"])
        assert len(vs) == 1 and job["payload"]["adopted"] is True
        w = c.get(f"/api/worlds/{wid}/worldir").json()
        assert all(e["provenance"] != "OBSERVED" for e in w["entities"].values())
        assert len(c.get(f"/api/worlds/{wid}/cameras").json()["cameras"]) <= 1


def test_release_gate_same_input_reproduces(tmp_path):
    """Same photographs, same configuration, two fresh worlds: identical registration, point counts within tolerance."""
    runs = []
    for k in range(2):
        with app_client(tmp_path / f"run{k}") as c:
            body = _post(c, SIX)
            job, _ = _watch_job(c, body["job_id"])
            assert job["status"] == "succeeded", job.get("error")
            m = c.get(f"/api/worlds/{body['world_id']}/status").json()["model"]
            runs.append((m["images_registered"], _ply_vertices(c.get(f"/api/worlds/{body['world_id']}/points").content)))
    assert runs[0][0] == runs[1][0], runs                                           # registered cameras: exactly equal
    assert abs(runs[0][1] - runs[1][1]) <= POINT_COUNT_TOLERANCE * max(runs[0][1], runs[1][1]), runs
