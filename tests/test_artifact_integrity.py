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
