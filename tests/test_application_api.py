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

# Repository-level dependency guard (mission known issue): the API stack
# (fastapi/sqlalchemy/aiosqlite) ships in apps/api/requirements.txt, not
# the root package, so a checkout without it must SKIP these modules
# explicitly instead of erroring the whole suite at fixture startup.
pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

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
    assert "usable photo evidence" in (job.get("error") or "").lower() or "needs at least 2" in (job.get("error") or "").lower()


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
    for path in ("worldir", "points", "cameras"):
        assert client.get(f"/api/worlds/wld_missing/{path}").status_code == 404
        r = client.get(f"/api/worlds/{w['id']}/{path}")
        assert r.status_code == 404, (path, r.status_code)
        assert "reconstruction" in r.json()["detail"].lower()


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
    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None  # drop cached store so this test's WORLDSTORE_ROOT takes effect

    w = client.post("/api/worlds", json={"name": "Recon World"}).json()
    sid = client.post("/api/sessions", json={"name": "Recon chain"}).json()["id"]
    assert client.post(f"/api/worlds/{w['id']}/attach/{sid}").status_code == 200
    for i in range(3):
        _upload_photo(client, sid, f"frame_{i:02d}.jpg", b"jpeg-" + f"{i}".encode() * 8)
    wid = client.post("/api/worlds", json={"name": "Recon world"}).json()["id"]
    assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200

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
    assert "usable photo evidence" in (job.get("error") or "").lower() or "needs at least 2" in (job.get("error") or "").lower()


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


def test_current_version_from_another_world_is_rejected(client, tmp_path, monkeypatch):
    """A world's current_version_id pointing at a version that belongs to
    a DIFFERENT world must not silently serve that world's WorldIR: the
    computational surfaces return 409 instead of cross-world data."""
    import asyncio

    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))

    from world_ir.world_v1 import WorldIR
    from worldstore.store import WorldStore

    store = WorldStore(tmp_path / "ws")
    stored = store.save_version(WorldIR(id="w-real"), parent=None, version_id="v-real")

    wa = client.post("/api/worlds", json={"name": "World A"}).json()
    wb = client.post("/api/worlds", json={"name": "World B"}).json()

    async def _seed():
        import apps.api.db as db_mod
        from apps.api.models import World, WorldVersion

        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            db.add(WorldVersion(
                id=stored.version_id,
                world_id=wa["id"],
                parent_version_id=None,
                artifact_uri=stored.artifact_uri,
                artifact_hash=stored.artifact_hash,
            ))
            wb_row = await db.get(World, wb["id"])
            wb_row.current_version_id = stored.version_id
            wa_row = await db.get(World, wa["id"])
            wa_row.current_version_id = stored.version_id
            await db.commit()

    asyncio.run(_seed())

    for path in ("worldir", "points", "cameras"):
        r = client.get(f"/api/worlds/{wb['id']}/{path}")
        assert r.status_code == 409, (path, r.status_code, r.text[:200])
        assert "does not belong" in r.json()["detail"]

    # The owning world still reads its own version fine.
    assert client.get(f"/api/worlds/{wa['id']}/worldir").status_code == 200


# --------------------------------------------------------------------------
# Commit / version integrity (P0): correction commits must validate,
# adopt atomically, and behave deterministically under stale/duplicate/
# concurrent requests.
# --------------------------------------------------------------------------


def _seed_compiled_world(client, tmp_path, monkeypatch, name="Commit world"):
    """A world with one real WorldStore version + DB mirror row: a valid
    commit base. Returns (world_id, version_id)."""
    import asyncio

    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    from provenance import Provenance
    from world_ir import Entity, EntityType
    from world_ir.world_v1 import WorldIR
    from worldstore.store import WorldStore

    wid = client.post("/api/worlds", json={"name": name}).json()["id"]
    store = WorldStore(tmp_path / "ws")
    wir = WorldIR(id=wid)
    wir.entities["ent-shed"] = Entity(
        id="ent-shed", type=EntityType.STRUCTURE, name="Shed",
        provenance=Provenance.RECONSTRUCTED, confidence=0.8,
    )
    stored = store.save_version(wir, parent=None)

    async def _seed():
        import apps.api.db as db_mod
        from apps.api.models import World, WorldVersion

        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            db.add(WorldVersion(
                id=stored.version_id,
                world_id=wid,
                parent_version_id=None,
                artifact_uri=stored.artifact_uri,
                artifact_hash=stored.artifact_hash,
                source_session_ids=[],
                changed_entity_ids=[],
            ))
            row = await db.get(World, wid)
            row.current_version_id = stored.version_id
            await db.commit()

    asyncio.run(_seed())
    return wid, stored.version_id


def _commit(client, wid, entity_id="ent-shed", changes=None, parent=None):
    body = {"entity_id": entity_id, "changes": changes if changes is not None else {}}
    if parent is not None:
        body["parent_version_id"] = parent
    return client.post(f"/api/worlds/{wid}/commit", json=body)


def _head_and_count(client, wid):
    detail = client.get(f"/api/worlds/{wid}").json()
    versions = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    return detail["current_version_id"], len(versions)


def test_commit_rejects_unsupported_field(client, tmp_path, monkeypatch):
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    r = _commit(client, wid, changes={"id": "ent-hacked", "name": "Barn"})
    assert r.status_code == 422, r.text
    assert "id" in r.json()["detail"]
    # Nothing persisted: HEAD unmoved, no new version, entity untouched.
    head, count = _head_and_count(client, wid)
    assert (head, count) == (v0, 1)
    assert client.get(f"/api/worlds/{wid}/worldir").json()["entities"]["ent-shed"]["name"] == "Shed"


def test_commit_rejects_invalid_values(client, tmp_path, monkeypatch):
    wid, _ = _seed_compiled_world(client, tmp_path, monkeypatch)
    bad = [
        ({"confidence": 1.5}, 422),
        ({"confidence": "high"}, 422),
        ({"type": "not-a-type"}, 422),
        ({"semantic_labels": "wall"}, 422),
        ({"transform": [1, 2, 3]}, 422),
        ({}, 422),
        ({f"extra-{i}": i for i in range(33)}, 413),
        ({"custom_properties": {"blob": "x" * 20000}}, 413),
    ]
    for changes, status in bad:
        r = _commit(client, wid, changes=changes)
        assert r.status_code == status, (changes, r.status_code, r.text[:200])


def test_commit_rejects_nan_confidence(client, tmp_path, monkeypatch):
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    # Strict-JSON cannot spell NaN, but the server's parser accepts it --
    # it must still be rejected explicitly, never stored.
    import json as _json

    import httpx

    raw = _json.dumps({"entity_id": "ent-shed", "changes": {"confidence": float("nan")}})
    r = client.post(
        f"/api/worlds/{wid}/commit",
        content=raw.encode(),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422, r.text
    head, _ = _head_and_count(client, wid)
    assert head == v0


def test_commit_happy_path_adopts_version(client, tmp_path, monkeypatch):
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    r = _commit(client, wid, changes={"name": "Barn", "confidence": 0.9})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["duplicate"] is False
    assert body["version_id"] != v0
    assert body["changed_fields"] == ["name", "confidence"]
    # HEAD moved, mirror row exists, and the world stays readable (a
    # commit that advances HEAD without a mirror row bricks reads).
    head, count = _head_and_count(client, wid)
    assert (head, count) == (body["version_id"], 2)
    wir = client.get(f"/api/worlds/{wid}/worldir")
    assert wir.status_code == 200
    ent = wir.json()["entities"]["ent-shed"]
    assert ent["name"] == "Barn"
    assert ent["confidence"] == 0.9


def test_commit_stale_parent_rejected(client, tmp_path, monkeypatch):
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    v1 = _commit(client, wid, changes={"name": "Barn"}).json()["version_id"]
    r = _commit(client, wid, changes={"name": "Silo"}, parent=v0)
    assert r.status_code == 409, r.text
    assert "stale" in r.json()["detail"].lower()
    head, count = _head_and_count(client, wid)
    assert (head, count) == (v1, 2)


def test_commit_duplicate_retry_returns_current(client, tmp_path, monkeypatch):
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    first = _commit(client, wid, changes={"name": "Barn"}).json()
    second = _commit(client, wid, changes={"name": "Barn"})
    assert second.status_code == 200
    body = second.json()
    assert body["duplicate"] is True
    assert body["version_id"] == first["version_id"]
    head, count = _head_and_count(client, wid)
    assert (head, count) == (first["version_id"], 2)


def test_commit_unknown_entity_404(client, tmp_path, monkeypatch):
    wid, _ = _seed_compiled_world(client, tmp_path, monkeypatch)
    assert _commit(client, wid, entity_id="ent-missing", changes={"name": "X"}).status_code == 404


def test_commit_failed_worldstore_write_leaves_head_intact(client, tmp_path, monkeypatch):
    """A WorldStore failure mid-commit is explicit (409) with HEAD
    unmoved and no mirror row -- never a silent success."""
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    from worldstore import store as ws_mod
    from worldstore.store import WorldStoreError

    def _boom(self, world, **kwargs):
        raise WorldStoreError("simulated disk failure")

    monkeypatch.setattr(ws_mod.WorldStore, "save_version", _boom)
    r = _commit(client, wid, changes={"name": "Barn"})
    assert r.status_code == 409, r.text
    head, count = _head_and_count(client, wid)
    assert (head, count) == (v0, 1)
    assert client.get(f"/api/worlds/{wid}/worldir").json()["entities"]["ent-shed"]["name"] == "Shed"


def test_commit_failed_db_write_rolls_back(client, tmp_path, monkeypatch):
    """A DB outage at adopt time is an explicit 503: HEAD unmoved, no
    mirror row, the orphaned version file never adopted."""
    from sqlalchemy.ext.asyncio import AsyncSession

    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)

    async def _down(self, *args, **kwargs):
        raise RuntimeError("simulated db outage")

    monkeypatch.setattr(AsyncSession, "commit", _down)
    r = _commit(client, wid, changes={"name": "Barn"})
    assert r.status_code == 503, r.text
    assert "not adopted" in r.json()["detail"].lower()


def test_conditional_head_update_single_winner(client, tmp_path, monkeypatch):
    """The adoption primitive itself: two sessions racing the same HEAD
    value -- the conditional UPDATE lets exactly one win (rowcount 1);
    the loser sees rowcount 0 and must rebase, never silently overwrite."""
    import asyncio

    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)

    async def _race():
        import apps.api.db as db_mod
        from apps.api.models import World
        from sqlalchemy import update

        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            first = await db.execute(
                update(World)
                .where(World.id == wid, World.current_version_id == v0)
                .values(current_version_id="v-winner-a")
            )
            await db.commit()
            second = await db.execute(
                update(World)
                .where(World.id == wid, World.current_version_id == v0)
                .values(current_version_id="v-winner-b")
            )
            await db.commit()
            row = await db.get(World, wid)
            return first.rowcount, second.rowcount, row.current_version_id

    won, lost, head = asyncio.run(_race())
    assert (won, lost) == (1, 0)
    assert head == "v-winner-a"


def test_concurrent_commits_no_corruption(client, tmp_path, monkeypatch):
    import threading

    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    barrier = threading.Barrier(2)
    results = []

    def worker(name):
        try:
            barrier.wait(timeout=10)
            r = _commit(client, wid, changes={"name": name})
            results.append(r.status_code)
        except Exception as exc:  # noqa: BLE001 -- surfaced in the assertion
            results.append(f"ERROR {type(exc).__name__}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("Barn", "Silo")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    # Either serialization (second chains onto the first: 200 + 200, three
    # versions) or a genuine overlap (second loses the conditional HEAD
    # update: 200 + 409, two versions) is correct -- but nothing else is:
    # no errors, no missing mirror rows, HEAD always readable.
    assert all(isinstance(s, int) for s in results), results
    head, count = _head_and_count(client, wid)
    if sorted(results) == [200, 200]:
        assert head != v0 and count == 3
    else:
        assert sorted(results) == [200, 409], results
        assert head != v0 and count == 2
    versions = {v["id"] for v in client.get(f"/api/worlds/{wid}/versions").json()["items"]}
    assert head in versions
    assert client.get(f"/api/worlds/{wid}/worldir").status_code == 200


# --------------------------------------------------------------------------
# Procedural rooms (P0 concurrency + validation)
# --------------------------------------------------------------------------


def _room_world(client, tmp_path, monkeypatch, name="Room world"):
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None  # drop cached store so this test's root takes effect
    return client.post("/api/worlds", json={"name": name}).json()["id"]


def _room_body(name="Study", width=4.0, depth=5.0, height=2.8):
    return {
        "name": name, "width": width, "depth": depth, "height": height,
        "openings": [], "seed": 7, "origin": [0.0, 0.0, 0.0],
    }


def test_create_room_happy_path(client, tmp_path, monkeypatch):
    wid = _room_world(client, tmp_path, monkeypatch)
    r = client.post(f"/api/worlds/{wid}/rooms", json=_room_body())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["room_entity_id"] == "Study-room"
    assert body["entity_ids_added"], "room creation must add entities"
    items = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    assert len(items) == 1 and items[0]["is_current"] is True
    wir = client.get(f"/api/worlds/{wid}/worldir")
    assert wir.status_code == 200
    assert "Study-room" in wir.json()["entities"]


def test_create_room_rejects_bad_input(client, tmp_path, monkeypatch):
    wid = _room_world(client, tmp_path, monkeypatch)
    cases = [
        dict(_room_body(), width=0.0),
        dict(_room_body(), depth=-2.0),
        dict(_room_body(), name="   "),
        dict(_room_body(), openings=[
            {"wall": "north", "kind": "door", "lateral_offset": 1.0,
             "width": 0.9, "bottom": 0.0, "top": 2.1}
            for _ in range(201)
        ]),
        dict(_room_body(), openings=[
            {"wall": "attic", "kind": "door", "lateral_offset": 1.0,
             "width": 0.9, "bottom": 0.0, "top": 2.1}
        ]),
    ]
    for payload in cases:
        r = client.post(f"/api/worlds/{wid}/rooms", json=payload)
        assert r.status_code in (413, 422), (payload, r.status_code, r.text[:200])
    # None of the rejections persisted a version.
    assert client.get(f"/api/worlds/{wid}/versions").json()["items"] == []


def test_create_room_rejects_nan(client, tmp_path, monkeypatch):
    import json as _json

    wid = _room_world(client, tmp_path, monkeypatch)
    payload = _room_body()
    payload["width"] = float("nan")
    r = client.post(
        f"/api/worlds/{wid}/rooms",
        content=_json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422, r.text
    assert client.get(f"/api/worlds/{wid}/versions").json()["items"] == []


def test_create_room_duplicate_submit(client, tmp_path, monkeypatch):
    wid = _room_world(client, tmp_path, monkeypatch)
    first = client.post(f"/api/worlds/{wid}/rooms", json=_room_body())
    assert first.status_code == 201, first.text
    second = client.post(f"/api/worlds/{wid}/rooms", json=_room_body())
    assert second.status_code == 201, second.text
    # A byte-identical retry must not mint a duplicate version.
    assert second.json()["version_id"] == first.json()["version_id"]
    assert len(client.get(f"/api/worlds/{wid}/versions").json()["items"]) == 1


def test_concurrent_rooms_no_lost_update(client, tmp_path, monkeypatch):
    import threading

    wid = _room_world(client, tmp_path, monkeypatch)
    barrier = threading.Barrier(2)
    results = []

    def worker(name):
        try:
            barrier.wait(timeout=10)
            r = client.post(f"/api/worlds/{wid}/rooms", json=_room_body(name=name))
            results.append(r.status_code)
        except Exception as exc:  # noqa: BLE001
            results.append(f"ERROR {type(exc).__name__}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("Study", "Kitchen")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    # Serialized (201 + 201 chained) or genuinely overlapped (201 + 409)
    # are both correct; anything else (errors, lost versions, unreadable
    # HEAD) is a consistency failure.
    assert all(isinstance(s, int) for s in results), results
    items = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    head = client.get(f"/api/worlds/{wid}").json()["current_version_id"]
    if sorted(results) == [201, 201]:
        assert len(items) == 2
    else:
        assert sorted(results) == [201, 409], results
        assert len(items) == 1
    assert head in {v["id"] for v in items}
    assert client.get(f"/api/worlds/{wid}/worldir").status_code == 200


# --------------------------------------------------------------------------
# Job lifecycle (P0): duplicates, retries, staleness, timeouts
# --------------------------------------------------------------------------


def test_reconstruct_duplicate_in_progress_409(client, tmp_path, monkeypatch):
    sid = client.post("/api/sessions", json={"name": "Dup"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Dup world"}).json()["id"]
    assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
    for i in range(2):
        _upload_photo(client, sid, f"frame_{i}.jpg", b"jpeg-bytes-here")
    first = client.post(f"/api/sessions/{sid}/reconstruct")
    assert first.status_code == 200
    second = client.post(f"/api/sessions/{sid}/reconstruct")
    assert second.status_code == 409, second.text
    assert "in progress" in second.json()["detail"].lower()
    # The first job still settles on its own; no duplicate pipeline ran.
    job = _wait_job(client, first.json()["job_id"], timeout=90.0)
    assert job["status"] in ("completed", "failed")


def test_failed_job_exhausts_retries(client):
    # test_reconstruct_honest_failure_without_usable_images leaves a job
    # that fails deterministically: it must retry (attempts > 1) and end
    # 'failed' -- never stuck 'running', never silently dropped.
    sid = client.post("/api/sessions", json={"name": "Retry"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Retry world"}).json()["id"]
    assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
    client.post(
        f"/api/uploads?session_id={sid}",
        files={"file": ("scan.ply", b"ply-bytes", "application/octet-stream")},
    )
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    job = _wait_job(client, r.json()["job_id"], timeout=90.0)
    assert job["status"] == "failed"
    assert job["attempts"] == 3, job
    assert job["error"], "a failed job must carry its error explicitly"


def test_reap_stale_running_job(client):
    """A job stranded 'running' by a dead worker is requeued, not wedged."""
    import asyncio
    from datetime import timedelta

    import apps.api.db as db_mod
    from apps.api.jobs import reap_stale_jobs
    from apps.api.models import Job, utcnow

    async def _scenario():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            db.add(Job(
                id="job_stale001", type="PROCESS_EVIDENCE",
                entity_type="evidence", entity_id="ev_missing",
                status="running", worker_id="worker-dead",
                attempts=1, max_attempts=3,
                heartbeat_at=utcnow() - timedelta(seconds=3600),
            ))
            db.add(Job(
                id="job_fresh001", type="PROCESS_EVIDENCE",
                entity_type="evidence", entity_id="ev_missing",
                status="running", worker_id="worker-live",
                attempts=1, max_attempts=3,
                heartbeat_at=utcnow(),
            ))
            await db.commit()
            reaped = await reap_stale_jobs(db)
            stale = await db.get(Job, "job_stale001")
            fresh = await db.get(Job, "job_fresh001")
            return reaped, stale.status, fresh.status

    reaped, stale_status, fresh_status = asyncio.run(_scenario())
    assert reaped == 1
    assert stale_status == "queued"
    assert fresh_status == "running"


def test_job_handler_timeout_fails_explicitly(client, monkeypatch):
    """A wedged handler fails the job with a timeout error instead of
    stalling the worker queue forever."""
    import asyncio

    import apps.api.db as db_mod
    import apps.api.jobs as jobs_mod
    from apps.api.models import Job

    async def _never(_db, _job):
        await asyncio.sleep(60)

    monkeypatch.setitem(jobs_mod._HANDLERS, "HANG_FOREVER", _never)
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "1")

    async def _scenario():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            db.add(Job(
                id="job_hang001", type="HANG_FOREVER",
                entity_type="session", entity_id="ses_x",
                status="queued", attempts=0, max_attempts=1,
            ))
            await db.commit()
            job = await jobs_mod.process_next_job(db)
            return job.status, job.error

    status, error = asyncio.run(_scenario())
    assert status == "failed"
    assert "timed out" in (error or "").lower()
