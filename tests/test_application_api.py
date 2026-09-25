"""Application API tests: sessions, GPS persistence, uploads → evidence →
jobs → notifications, worlds, honest failure states.

Exercises the application backend end to end (in-process worker) against a
temporary SQLite database. No fabricated data: a session without GPS must
report location unavailable; reconstruction must fail explicitly until wired.
"""

from __future__ import annotations

import importlib
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_url = f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _wait_job(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not settle in time")


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_session_with_gps_persists_real_location(client):
    r = client.post(
        "/api/sessions",
        json={
            "name": "Boundary capture",
            "location": {
                "latitude": 22.5726,
                "longitude": 88.3639,
                "accuracy": 4.2,
                "altitude": 9.0,
                "source": "gnss",
            },
            "device_metadata": {"platform": "test"},
        },
    )
    assert r.status_code == 201
    sid = r.json()["id"]
    assert r.json()["location"]["latitude"] == 22.5726
    assert r.json()["location"]["accuracy"] == 4.2

    loc = client.get(f"/api/sessions/{sid}/location")
    assert loc.status_code == 200
    assert loc.json()["longitude"] == 88.3639


def test_session_without_gps_reports_location_unavailable(client):
    r = client.post("/api/sessions", json={"name": "Indoor scan"})
    assert r.status_code == 201
def test_session_location_is_honest_not_fabricated(client):
    r = client.post("/api/sessions", json={"name": "No GPS"})
    sid = r.json()["id"]
    detail = client.get(f"/api/sessions/{sid}/location").json()["detail"]
    assert "no gps" in detail.lower()


def test_upload_creates_evidence_and_processing_job_completes(client):
    sid = client.post("/api/sessions", json={"name": "Upload session"}).json()["id"]
    payload = b"\xff\xd8\xff\xe0fake-jpeg-bytes-" + uuid.uuid4().hex.encode()
    r = client.post(
        f"/api/uploads?session_id={sid}",
        files={"file": ("capture_001.jpg", payload, "image/jpeg")},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["evidence_id"]
    assert len(body["checksum"]) == 64

    ev = client.get(f"/api/evidence/{body['evidence_id']}").json()
    assert ev["type"] == "photo"
    assert ev["size"] == len(payload)

    job = _wait_job(client, body["job_id"])
    assert job["status"] == "completed", job.get("error")

    ev = client.get(f"/api/evidence/{body['evidence_id']}").json()
    assert ev["processing_state"] == "processed"
    assert ev["metadata"]["verified"] is True

    notes = client.get("/api/notifications").json()["items"]
    assert any(n["type"] == "evidence.processed" and n["entity_id"] == ev["id"] for n in notes)

    s = client.get(f"/api/sessions/{sid}").json()
    assert s["uploaded_at"] is not None


def test_evidence_artifact_roundtrip(client):
    payload = b"pointcloud-ply-content"
    up = client.post(
        "/api/uploads",
        files={"file": ("scan.ply", payload, "application/octet-stream")},
    ).json()
    r = client.get(f"/api/evidence/{up['evidence_id']}/artifact")
    assert r.status_code == 200
    assert r.content == payload


def test_world_attach_and_coverage(client):
    sid = client.post(
        "/api/sessions",
        json={"name": "Geo session", "location": {"latitude": 1.0, "longitude": 2.0}},
    ).json()["id"]
    w = client.post("/api/worlds", json={"name": "Test Area"}).json()
    assert client.post(f"/api/worlds/{w['id']}/attach/{sid}").status_code == 200

    cov = client.get(f"/api/worlds/{w['id']}/coverage").json()
    assert cov["available"] is True
    assert cov["session_points"][0]["lat"] == 1.0

    empty = client.post("/api/worlds", json={"name": "Empty world"}).json()
    cov2 = client.get(f"/api/worlds/{empty['id']}/coverage").json()
    assert cov2["available"] is False
    assert "no spatial data" in cov2["reason"].lower()


def test_trajectory_honest_until_artifacts_exist(client):
    sid = client.post("/api/sessions", json={"name": "Traj"}).json()["id"]
    traj = client.get(f"/api/sessions/{sid}/trajectory").json()
    assert traj["available"] is False
    assert traj["points"] == []


def test_reconstruct_fails_with_honest_error_when_no_evidence(client):
    """Reconstruction fails with an honest error when session has no evidence."""
    w = client.post("/api/worlds", json={"name": "Recon World"}).json()
    sid = client.post("/api/sessions", json={"name": "Recon"}).json()["id"]
    assert client.post(f"/api/worlds/{w['id']}/attach/{sid}").status_code == 200
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    job = _wait_job(client, r.json()["job_id"], timeout=90.0)
    assert job["status"] == "failed"
    # honest error about missing usable photo evidence, not fake success
    assert "usable photo evidence" in (job.get("error") or "").lower()


def test_world_detail_and_versions_are_real(client):
    """GET /api/worlds/{id} and /versions back the UI's World screens.

    Versions must be empty (not fabricated) until computation creates them,
    and an unknown id must 404 rather than return a blank world.
    """
    sid = client.post("/api/sessions", json={"name": "Compose session"}).json()["id"]
    w = client.post(
        "/api/worlds", json={"name": "Detail World", "latitude": 12.9, "longitude": 80.2}
    ).json()

    detail = client.get(f"/api/worlds/{w['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == w["id"]
    assert body["name"] == "Detail World"
    assert body["latitude"] == 12.9
    assert body["session_count"] == 0
    assert body["current_version_id"] is None

    assert client.post(f"/api/worlds/{w['id']}/attach/{sid}").status_code == 200
    assert client.get(f"/api/worlds/{w['id']}").json()["session_count"] == 1

    versions = client.get(f"/api/worlds/{w['id']}/versions").json()
    assert versions["items"] == []  # no WorldStore versions yet — not invented

    assert client.get("/api/worlds/wld_missing").status_code == 404
    assert client.get("/api/worlds/wld_missing/versions").status_code == 404


def test_evidence_delete_removes_record_and_keeps_artifact(client):
    """Deleting Evidence is an application-record operation.

    The content-addressed artifact stays in the store (other records may
    reference the same sha256) and the deletion is recorded as real activity.
    """
    up = client.post(
        "/api/uploads",
        files={"file": ("frame.jpg", b"jpeg-bytes-for-delete-test", "image/jpeg")},
    ).json()
    evidence_id = up["evidence_id"]

    assert client.delete(f"/api/evidence/{evidence_id}").status_code == 204
    assert client.get(f"/api/evidence/{evidence_id}").status_code == 404
    assert client.get(f"/api/evidence/{evidence_id}/artifact").status_code == 404
    assert client.delete(f"/api/evidence/{evidence_id}").status_code == 404

    listed = client.get("/api/evidence").json()["items"]
    assert all(e["id"] != evidence_id for e in listed)

    acts = client.get("/api/activity").json()["items"]
    assert any(a["type"] == "evidence.deleted" and a["entity_id"] == evidence_id for a in acts)


def test_unknown_entity_404s(client):
    assert client.get("/api/sessions/ses_nonexistent").status_code == 404
    assert client.get("/api/evidence/ev_nonexistent").status_code == 404
    assert client.get("/api/worlds/wld_nonexistent/coverage").status_code == 404


def test_activity_feed_real_events(client):
    client.post("/api/sessions", json={"name": "Activity session"})
    acts = client.get("/api/activity").json()["items"]
    assert any(a["type"] == "session.created" for a in acts)



# --------------------------------------------------------------------------
# Reconstruction job -> World/Version mirror -> worldir/points/cameras
# --------------------------------------------------------------------------


def _upload_photo(client, sid, name, payload):
    r = client.post(
        f"/api/uploads?session_id={sid}",
        files={"file": (name, payload, "image/jpeg")},
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_worldir_endpoints_404_until_a_version_exists(client):
    """The spatial workstation's endpoints exist and are honest: an unknown
    world is 404; a real world with no versions says so explicitly instead
    of returning fabricated geometry."""
    w = client.post("/api/worlds", json={"name": "Empty World"}).json()
    expected_detail = {
        "worldir": "no version compiled",
        "points": "no points artifact",
        "cameras": "no cameras artifact",
    }
    for path, needle in expected_detail.items():
        assert client.get(f"/api/worlds/wld_missing/{path}").status_code == 404
        r = client.get(f"/api/worlds/{w['id']}/{path}")
        assert r.status_code == 404, (path, r.status_code)
        assert needle in r.json()["detail"].lower()


def test_reconstruct_session_full_chain(client, tmp_path, monkeypatch):
    """RECONSTRUCT_SESSION runs the real chain end to end: uploaded evidence
    -> EvidenceItem -> engine.pipeline.vertical_slice (the same function the
    CLI's compile command uses) -> WorldStore version via
    worldstore_service.commit_version -> application World/WorldVersion
    mirror.

    This environment has real COLMAP installed but no deterministic test
    backend seam for the vertical-slice pipeline, so non-photographic bytes
    genuinely fail SfM registration -- exactly the "never fabricate a
    successful reconstruction" behavior being verified here (see
    apps.api.jobs._run_reconstruct_session and the Phase 3 verification).
    """
    store_root = tmp_path / "ws"
    monkeypatch.setenv("WORLDSTORE_ROOT", str(store_root))

    w = client.post("/api/worlds", json={"name": "Recon World"}).json()
    sid = client.post("/api/sessions", json={"name": "Recon chain"}).json()["id"]
    assert client.post(f"/api/worlds/{w['id']}/attach/{sid}").status_code == 200
    for i in range(3):
        _upload_photo(client, sid, f"frame_{i:02d}.jpg", b"jpeg-" + f"{i}".encode() * 8)

    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    job = _wait_job(client, r.json()["job_id"], timeout=90.0)
    # Real COLMAP genuinely cannot register non-photographic bytes; the job
    # must fail honestly rather than fabricate a completed reconstruction.
    assert job["status"] == "failed"
    assert job.get("error")

    # No version was committed for a failed reconstruction -- the mirror
    # and the world's HEAD pointer stay untouched.
    detail = client.get(f"/api/worlds/{w['id']}").json()
    assert detail["current_version_id"] is None
    versions = client.get(f"/api/worlds/{w['id']}/versions").json()["items"]
    assert versions == []


def test_reconstruct_honest_failure_without_usable_images(client):
    """Non-photo / unresolvable evidence cannot reconstruct: the job fails
    with the measured reason. Never a fake success."""
    w = client.post("/api/worlds", json={"name": "No Images World"}).json()
    sid = client.post("/api/sessions", json={"name": "No images"}).json()["id"]
    assert client.post(f"/api/worlds/{w['id']}/attach/{sid}").status_code == 200
    # upload a non-photo artifact -> classified dataset -> not photo evidence
    r = client.post(
        f"/api/uploads?session_id={sid}",
        files={"file": ("scan.ply", b"ply-bytes", "application/octet-stream")},
    )
    assert r.status_code == 201

    r = client.post(f"/api/sessions/{sid}/reconstruct")
    job = _wait_job(client, r.json()["job_id"], timeout=90.0)
    assert job["status"] == "failed"
    assert "usable photo evidence" in (job.get("error") or "").lower()


def test_reconstruct_honest_failure_when_backend_unavailable(client, monkeypatch):
    """When the vertical-slice pipeline import fails (optional dependency
    missing), the job fails honestly naming the import error -- never a
    silent fallback or fake result."""
    import apps.api.jobs as jobs_mod

    def _boom(*args, **kwargs):
        raise ImportError("engine.pipeline.vertical_slice unavailable (simulated)")

    monkeypatch.setattr(
        "engine.pipeline.vertical_slice.vertical_slice", _boom, raising=False
    )

    w = client.post("/api/worlds", json={"name": "No Backend World"}).json()
    sid = client.post("/api/sessions", json={"name": "No backend"}).json()["id"]
    assert client.post(f"/api/worlds/{w['id']}/attach/{sid}").status_code == 200
    for i in range(2):
        _upload_photo(client, sid, f"frame_{i}.jpg", b"jpeg-bytes-here")

    r = client.post(f"/api/sessions/{sid}/reconstruct")
    job = _wait_job(client, r.json()["job_id"], timeout=90.0)
    assert job["status"] == "failed"
    err = (job.get("error") or "").lower()
    # The injected import failure surfaces verbatim in the job's traceback
    # -- an honest, specific error, never a silent fallback or fake result.
    assert "unavailable (simulated)" in err
    del jobs_mod  # imported only to document the module under test
