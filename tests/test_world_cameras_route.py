"""GET /api/worlds/{id}/cameras for a COLMAP-reconstructed world.

Found by the canonical demo journey: the route read poses only from entity observations (where just the single-image
bootstrap writes one), so every real reconstruction returned ZERO cameras and the Studio's camera layer was empty.
A reconstructed world's registered poses live in the version's cameras artifact (written at adoption); the route must
serve them, per version, and never invent any when neither source has them.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
from fastapi.testclient import TestClient

from tests.test_export_pipeline_e2e import _promoted_world

CAMS = {"cameras": [
    {"evidence_id": "img-a", "position_m": [0.0, 1.5, 0.0], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
    {"evidence_id": "img-b", "position_m": [1.0, 1.5, 0.25], "rotation_wxyz": [0.9238795, 0.0, 0.3826834, 0.0]},
], "image_size": [1024, 768]}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws

    ws._store = None
    with TestClient(main_mod.app) as c:
        yield c


def _seed(client, tmp_path, cameras_payload):
    from apps.api.storage import store_bytes
    from worldstore.store import WorldStore

    wid = client.post("/api/worlds", json={"name": "cams"}).json()["id"]
    wir = _promoted_world()
    wir.id = wid
    stored = WorldStore(tmp_path / "ws").save_version(wir, parent=None)
    uri = None
    if cameras_payload is not None:
        digest, _ = store_bytes(json.dumps(cameras_payload).encode("utf-8"))
        uri = f"sha256://{digest}"

    async def _go():
        import apps.api.db as db_mod
        from apps.api.models import World, WorldVersion

        async with db_mod.get_sessionmaker()() as db:
            db.add(WorldVersion(id=stored.version_id, world_id=wid, parent_version_id=None,
                                artifact_uri=stored.artifact_uri, artifact_hash=stored.artifact_hash,
                                source_session_ids=[], changed_entity_ids=[], cameras_artifact_uri=uri))
            (await db.get(World, wid)).current_version_id = stored.version_id
            await db.commit()

    asyncio.run(_go())
    return wid, stored.version_id


def test_a_reconstructed_worlds_poses_come_from_its_cameras_artifact(client, tmp_path):
    wid, vid = _seed(client, tmp_path, CAMS)
    body = client.get(f"/api/worlds/{wid}/cameras").json()
    assert body["version_id"] == vid and body["image_size"] == [1024, 768]
    assert [c["evidence_id"] for c in body["cameras"]] == ["img-a", "img-b"]
    assert body["cameras"][1]["position_m"] == [1.0, 1.5, 0.25]
    assert body["cameras"][1]["rotation_wxyz"] == pytest.approx([0.9238795, 0.0, 0.3826834, 0.0])
    assert client.get(f"/api/worlds/{wid}/cameras", params={"version": vid}).json() == body


def test_no_cameras_anywhere_is_an_empty_list_not_invented_poses(client, tmp_path):
    wid, _ = _seed(client, tmp_path, None)
    assert client.get(f"/api/worlds/{wid}/cameras").json()["cameras"] == []


def test_a_missing_or_corrupt_artifact_degrades_to_empty(client, tmp_path):
    wid, _ = _seed(client, tmp_path, {"cameras": "not a list"})
    assert client.get(f"/api/worlds/{wid}/cameras").json()["cameras"] == []
