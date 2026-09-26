"""Artifact integrity + reconciliaiton surfaces (P0/P1).

Missing, corrupt, and traversal-shaped artifacts must surface as honest
empty states -- never fabricated geometry, never a 500. CLI-style
versions written straight to WorldStore must reconcile into the DB
mirror instead of drifting silently.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

# Same API-stack guard as tests/test_application_api.py: skip explicitly
# when apps/api/requirements.txt was never installed.
pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_url = f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import importlib

    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _seed_world_with_geometry(client, tmp_path, monkeypatch, geometries):
    """World with one WorldStore version + mirror row; geometries control
    the artifact scenarios below. Returns (world_id, version_id)."""
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    from provenance import Provenance
    from world_ir import Entity, EntityType
    from world_ir.schema_v1 import Geometry, GeometryType
    from world_ir.world_v1 import WorldIR
    from worldstore.store import WorldStore

    wid = client.post("/api/worlds", json={"name": "Artifact world"}).json()["id"]
    store = WorldStore(tmp_path / "ws")
    wir = WorldIR(id=wid)
    wir.entities["ent-shed"] = Entity(
        id="ent-shed", type=EntityType.STRUCTURE, name="Shed",
        provenance=Provenance.RECONSTRUCTED, confidence=0.8,
    )
    for gid, data_uri in geometries.items():
        wir.geometries[gid] = Geometry(
            id=gid, type=GeometryType.POINTCLOUD, data_uri=data_uri,
        )
    stored = store.save_version(wir, parent=None)

    async def _seed():
        import apps.api.db as db_mod
        from apps.api.models import World, WorldVersion

        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            db.add(WorldVersion(
                id=stored.version_id, world_id=wid, parent_version_id=None,
                artifact_uri=stored.artifact_uri, artifact_hash=stored.artifact_hash,
                source_session_ids=[], changed_entity_ids=[],
            ))
            row = await db.get(World, wid)
            row.current_version_id = stored.version_id
            await db.commit()

    asyncio.run(_seed())
    return wid, stored.version_id


def test_storage_resolve_rejects_traversal_and_garbage():
    from apps.api.storage import resolve_artifact, store_bytes

    digest, path = store_bytes(b"real-bytes")
    assert resolve_artifact(f"sha256://{digest}") == path
    assert resolve_artifact("sha256://../escape") is None
    assert resolve_artifact("sha256://../../etc/passwd") is None
    assert resolve_artifact("sha256://not-hex!!") is None
    assert resolve_artifact("sha256://") is None
    assert resolve_artifact("file:///etc/passwd") is None
    assert resolve_artifact("") is None
    assert resolve_artifact("sha256://" + "0" * 64) is None  # well-formed, absent


def test_points_missing_artifact_is_honest_404(client, tmp_path, monkeypatch):
    wid, _ = _seed_world_with_geometry(
        client, tmp_path, monkeypatch, {"geom-1": "artifact://" + "0" * 64}
    )
    assert client.get(f"/api/worlds/{wid}/worldir").status_code == 200
    r = client.get(f"/api/worlds/{wid}/points")
    assert r.status_code == 404, r.text
    assert "point geometry" in r.json()["detail"].lower()


def test_points_corrupt_artifact_is_honest_404(client, tmp_path, monkeypatch):
    from world_ir.artifact_store import FileArtifactStore

    uri, _ = FileArtifactStore(tmp_path / "ws" / "artifacts").put(b"not-a-point-cloud")
    wid, _ = _seed_world_with_geometry(client, tmp_path, monkeypatch, {"geom-1": uri})
    assert client.get(f"/api/worlds/{wid}/worldir").status_code == 200
    r = client.get(f"/api/worlds/{wid}/points")
    assert r.status_code == 404, r.text
    assert "point geometry" in r.json()["detail"].lower()


def test_points_traversal_uri_never_escapes(client, tmp_path, monkeypatch):
    wid, _ = _seed_world_with_geometry(
        client, tmp_path, monkeypatch, {"geom-1": "artifact://../../traversal"}
    )
    # A hostile data_uri is skipped like any undecodable payload: honest
    # 404, never a 500, never filesystem access outside the store.
    r = client.get(f"/api/worlds/{wid}/points")
    assert r.status_code == 404, r.text


def test_planted_symlink_never_followed(tmp_path):
    """A symlink planted at a blob path: get() refuses it (surfaced as
    missing, never followed outside the store); put() drops the link
    and writes the real blob instead of writing through it."""
    import os

    from world_ir.artifact_store import ArtifactNotFoundError, FileArtifactStore

    store = FileArtifactStore(tmp_path / "artifacts")
    data = b"real-blob-bytes"
    uri, digest = store.put(data)
    target = store._path_for(digest)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"sensitive-outside-bytes")
    target.unlink()
    try:
        os.symlink(outside, target)
    except OSError:
        pytest.skip("symlinks require privilege on this platform")
    with pytest.raises(ArtifactNotFoundError):
        store.get(uri)
    uri2, digest2 = store.put(data)
    assert (uri2, digest2) == (uri, digest)
    assert not target.is_symlink()
    assert target.read_bytes() == data
    assert outside.read_bytes() == b"sensitive-outside-bytes"


def test_worldstore_tamper_detected_on_read(tmp_path):
    from provenance import Provenance
    from world_ir import Entity, EntityType
    from world_ir.world_v1 import WorldIR
    from worldstore.store import WorldStore, WorldStoreError

    store = WorldStore(tmp_path / "ws")
    wir = WorldIR(id="w-tamper")
    wir.entities["e"] = Entity(
        id="e", type=EntityType.STRUCTURE, provenance=Provenance.RECONSTRUCTED,
    )
    stored = store.save_version(wir, parent=None)
    assert store.verify_version(stored.version_id) == []

    record = store._record(stored.version_id)
    from world_ir.artifact_store import FileArtifactStore

    blob = FileArtifactStore(tmp_path / "ws" / "artifacts")
    digest = FileArtifactStore.digest_of(record["artifact_uri"])
    target = blob._path_for(digest)
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))

    failures = store.verify_version(stored.version_id)
    assert len(failures) == 1 and "mismatch" in failures[0]["reason"]
    with pytest.raises(WorldStoreError, match="mismatch"):
        store.load_version(stored.version_id)


def test_cli_written_version_reconciles_into_mirror(client, tmp_path, monkeypatch):
    """A version saved straight to WorldStore (CLI path, no mirror row)
    appears in the versions listing via resync instead of drifting."""
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    from world_ir.world_v1 import WorldIR
    from worldstore.store import WorldStore

    wid = client.post("/api/worlds", json={"name": "CLI world"}).json()["id"]
    stored = WorldStore(tmp_path / "ws").save_version(WorldIR(id=wid), parent=None)

    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None  # service cache must see this test's root
    # No mirror row was ever written, yet the on-disk version reconciles
    # into the listing instead of drifting silently. HEAD is not adopted
    # by a read path (that would be a silent write), so worldir stays an
    # honest 404 until something explicitly adopts the version.
    items = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    assert [v["id"] for v in items] == [stored.version_id]
    assert items[0]["is_current"] is False
    assert client.get(f"/api/worlds/{wid}/worldir").status_code == 404


def test_safe_filename_strips_structure_breakers():
    from apps.api.routes_misc import _safe_filename

    # Raw hostile bytes (no HTTP layer to encode them away first).
    assert _safe_filename('../../etc/x"\r\nSet-Cookie: pwned=1.jpg') == "xSet-Cookie: pwned=1.jpg"
    assert _safe_filename("C:\\fakepath\\photo.jpg") == "photo.jpg"
    assert _safe_filename("") == "unnamed"
    assert _safe_filename("...") == "unnamed"
    assert len(_safe_filename("n" * 500)) == 200


def test_hostile_filename_sanitized_end_to_end(client):
    evil = '../../etc/x"\r\nSet-Cookie: pwned=1.jpg'
    r = client.post(
        "/api/uploads",
        files={"file": (evil, b"jpeg-bytes", "image/jpeg")},
    )
    assert r.status_code == 201, r.text
    ev_id = r.json()["evidence_id"]
    ev = client.get(f"/api/evidence/{ev_id}").json()
    assert ".." not in ev["name"] and "\r" not in ev["name"] and "\n" not in ev["name"]
    assert '"' not in ev["name"]
    assert ev["name"], "sanitized name must never be empty"
    art = client.get(f"/api/evidence/{ev_id}/artifact")
    assert art.status_code == 200
    assert art.content == b"jpeg-bytes"
    # Structural header safety: Starlette RFC-5987-encodes the filename,
    # so hostile text may survive only as inert percent-escapes -- it must
    # never appear raw (no CR/LF for response splitting, no raw quotes to
    # break out of the quoted-string, no traversal).
    disposition = art.headers.get("content-disposition", "")
    assert "\r" not in disposition and "\n" not in disposition
    assert '"' not in disposition
    assert ".." not in disposition


def test_oversized_upload_rejected(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "10")
    r = client.post(
        "/api/uploads",
        files={"file": ("big.jpg", b"x" * 100, "image/jpeg")},
    )
    assert r.status_code == 413, r.text


def test_tampered_evidence_fails_checksum_explicitly(client, tmp_path, monkeypatch):
    """Bytes swapped under a recorded checksum: processing fails with a
    checksum mismatch -- never registers tampered evidence as verified."""
    import asyncio

    import apps.api.db as db_mod
    import apps.api.jobs as jobs_mod
    from apps.api.models import Evidence, Job
    from apps.api.storage import resolve_artifact, store_bytes

    async def _scenario():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            digest, _ = store_bytes(b"original-bytes")
            db.add(Evidence(
                id="ev_tamper01", name="frame.jpg", type="photo",
                session_id=None, checksum=digest,
                artifact_uri=f"sha256://{digest}",
            ))
            db.add(Job(
                id="job_tamper01", type="PROCESS_EVIDENCE",
                entity_type="evidence", entity_id="ev_tamper01",
                status="queued", attempts=0, max_attempts=1,
            ))
            await db.commit()
            # Tamper after the record, before processing.
            target = resolve_artifact(f"sha256://{digest}")
            target.write_bytes(b"tampered-bytes!!")
            job = await jobs_mod.process_next_job(db)
            return job.status, job.error

    # Direct drive (same pattern as the timeout/reap tests): the row is
    # claimed microseconds after commit, long before the 1s worker poll.
    status, error = asyncio.run(_scenario())
    assert status == "failed"
    assert "checksum mismatch" in (error or "").lower()


def test_create_world_rejects_bad_input(client):
    import json as _json

    assert client.post("/api/worlds", json={"name": "   "}).status_code == 422
    assert client.post("/api/worlds", json={"name": "w" * 201}).status_code == 422
    assert client.post("/api/worlds", json={"name": "W", "latitude": 100.0}).status_code == 422
    assert client.post("/api/worlds", json={"name": "W", "longitude": -200.0}).status_code == 422
    raw = _json.dumps({"name": "W", "latitude": float("nan")})
    r = client.post("/api/worlds", content=raw.encode(),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    big = client.post("/api/worlds", json={"name": "W", "description": "d" * 70000})
    assert big.status_code == 413, big.status_code
    # None rejected cleanly: no world rows leaked.
    assert client.post("/api/worlds", json={"name": "Fine"}).status_code == 201


def test_create_session_rejects_bad_input(client):
    import json as _json

    assert client.post("/api/sessions", json={"name": ""}).status_code == 422
    bad_loc = {"name": "S", "location": {"latitude": 91.0, "longitude": 0.0}}
    assert client.post("/api/sessions", json=bad_loc).status_code == 422
    raw = _json.dumps({"name": "S", "location": {"latitude": 1.0, "longitude": float("inf")}})
    r = client.post("/api/sessions", content=raw.encode(),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    big = client.post("/api/sessions", json={"name": "S", "device_metadata": {"b": "x" * 70000}})
    assert big.status_code == 413, big.status_code


def test_export_download_filename_safe(client):
    from apps.api.storage import store_bytes

    digest, _ = store_bytes(b"export-bytes")
    r = client.get(f"/api/worlds/wld_ok/export/{digest}/download?format=gltf")
    assert r.status_code == 200
    assert "wld_ok.gltf" in r.headers.get("content-disposition", "")
    # Request-controlled segments must never reach headers raw: quotes,
    # CR/LF and traversal survive only as inert text, if at all.
    r = client.get(
        "/api/worlds/wld_x%22%0D%0A_y/export/" + digest + "/download?format=gl%22t%0Af"
    )
    assert r.status_code == 200, r.text
    disposition = r.headers.get("content-disposition", "")
    assert "\r" not in disposition and "\n" not in disposition
    # At most the two RFC delimiters around the filename: nothing
    # attacker-controlled can break out of the quoted string.
    assert disposition.count('"') <= 2 and ".." not in disposition


def test_malformed_ids_404_without_side_effects(client):
    for path in (
        "/api/worlds/wld-../escape",
        "/api/worlds/%2e%2e%2fetc/worldir",
        "/api/worlds/wld_missing/versions",
        "/api/worlds/wld_missing/worldir",
        "/api/worlds/wld_missing/commit",
    ):
        if path.endswith("/commit"):
            r = client.post(path, json={"entity_id": "e", "changes": {"name": "x"}})
        else:
            r = client.get(path)
        assert r.status_code == 404, (path, r.status_code, r.text[:200])
