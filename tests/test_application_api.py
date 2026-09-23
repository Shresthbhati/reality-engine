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
    sid = client.post("/api/sessions", json={"name": "Recon"}).json()["id"]
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "failed"
    # Now fails with honest error about missing evidence, not fake "not wired"
    assert "no evidence items to reconstruct" in (job.get("error") or "").lower()


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

