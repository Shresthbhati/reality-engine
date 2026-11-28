"""Comprehensive security regression suite for tenant isolation & evidence ownership.

Covers:
1. Evidence upload ownership & tenant/session-scoped deduplication (Phase 1).
2. World API & Export API authorization across all audited endpoints, including
   cross-tenant version_id and export content_hash isolation (Phase 2 & 3).
3. Background job worker ownership validation (Phase 3F).
4. Undo/redo cross-tenant isolation (Phase 4).
5. Fresh-process persistence reload over SQLite + WorldStore (Phase 4).
6. Alembic migration 0007 upgrade/downgrade on a representative DB (Phase 5).
"""
from __future__ import annotations

import asyncio
import importlib
import io
import time

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import Role, issue_scoped_token


def _make_jpeg_bytes(tag: bytes = b"A") -> bytes:
    from PIL import Image, ImageDraw

    buf = io.BytesIO()
    img = Image.new("RGB", (256, 256), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    c1 = (40, 80, 180) if tag == b"A" else (40, 160, 80)
    for x in range(0, 256, 32):
        for y in range(0, 256, 32):
            if ((x // 32) + (y // 32)) % 2 == 0:
                draw.rectangle([x, y, x + 31, y + 31], fill=c1)
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _hdr(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _wait_job(client: TestClient, job_id: str, headers: dict[str, str], timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}", headers=headers)
        assert r.status_code == 200, r.text
        data = r.json()
        if data["status"] in ("succeeded", "partial", "failed", "cancelled"):
            return data
        time.sleep(0.1)
    raise AssertionError(f"Job {job_id} did not finish within {timeout}s")


@pytest.fixture()
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("REALITY_ENV", "test")
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    monkeypatch.setenv("REALITY_API_TOKEN", "master-secret-token")
    monkeypatch.setenv("REALITY_SIGNING_KEY", "signing-secret-key-for-tests")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sec.db'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "worldstore"))

    import apps.api.db as db_mod
    import apps.api.main as main_mod
    import apps.api.worldstore_service as ws_mod

    ws_mod._store = None
    importlib.reload(db_mod)
    importlib.reload(main_mod)
    return main_mod.app, db_mod, ws_mod, tmp_path


def test_evidence_upload_dedup_is_tenant_and_session_scoped(isolated_env):
    app, _, _, _ = isolated_env
    tok_a = issue_scoped_token("tenant-a", "user-a", Role.ADMIN)
    tok_b = issue_scoped_token("tenant-b", "user-b", Role.ADMIN)
    payload = _make_jpeg_bytes(b"A")

    with TestClient(app) as client:
        sess_a1 = client.post("/api/sessions", json={"name": "A1"}, headers=_hdr(tok_a)).json()["id"]
        sess_a2 = client.post("/api/sessions", json={"name": "A2"}, headers=_hdr(tok_a)).json()["id"]
        sess_b1 = client.post("/api/sessions", json={"name": "B1"}, headers=_hdr(tok_b)).json()["id"]

        # 1. Tenant A uploads photo to sess_a1
        up_a1 = client.post(
            f"/api/uploads?session_id={sess_a1}",
            files={"file": ("photo.jpg", payload, "image/jpeg")},
            headers=_hdr(tok_a),
        )
        assert up_a1.status_code == 201, up_a1.text
        d_a1 = up_a1.json()
        assert d_a1["deduplicated"] is False

        # 2. Same tenant, SAME session, identical bytes -> deduplicated within (tenant-a, sess_a1)
        up_a1_dup = client.post(
            f"/api/uploads?session_id={sess_a1}",
            files={"file": ("photo_copy.jpg", payload, "image/jpeg")},
            headers=_hdr(tok_a),
        )
        assert up_a1_dup.status_code == 201, up_a1_dup.text
        assert up_a1_dup.json()["deduplicated"] is True
        assert up_a1_dup.json()["evidence_id"] == d_a1["evidence_id"]

        # 3. Same tenant, DIFFERENT session -> independent Evidence row
        up_a2 = client.post(
            f"/api/uploads?session_id={sess_a2}",
            files={"file": ("photo.jpg", payload, "image/jpeg")},
            headers=_hdr(tok_a),
        )
        assert up_a2.status_code == 201, up_a2.text
        assert up_a2.json()["deduplicated"] is False
        assert up_a2.json()["evidence_id"] != d_a1["evidence_id"]

        # 4. Tenant B uploads BYTE-IDENTICAL image -> must get its own Evidence row, never Tenant A's
        up_b1 = client.post(
            f"/api/uploads?session_id={sess_b1}",
            files={"file": ("photo.jpg", payload, "image/jpeg")},
            headers=_hdr(tok_b),
        )
        assert up_b1.status_code == 201, up_b1.text
        d_b1 = up_b1.json()
        assert d_b1["deduplicated"] is False
        assert d_b1["evidence_id"] != d_a1["evidence_id"]

        # 5. Tenant B cannot retrieve or delete Tenant A's evidence or upload to Tenant A's session
        assert client.get(f"/api/evidence/{d_a1['evidence_id']}", headers=_hdr(tok_b)).status_code == 404
        assert client.get(f"/api/evidence/{d_a1['evidence_id']}/artifact", headers=_hdr(tok_b)).status_code == 404
        assert client.delete(f"/api/evidence/{d_a1['evidence_id']}", headers=_hdr(tok_b)).status_code == 404
        assert client.post(
            f"/api/uploads?session_id={sess_a1}",
            files={"file": ("x.jpg", payload, "image/jpeg")},
            headers=_hdr(tok_b),
        ).status_code == 404

        # 6. Tenant A can still retrieve its own evidence & artifact
        assert client.get(f"/api/evidence/{d_a1['evidence_id']}", headers=_hdr(tok_a)).status_code == 200
        assert client.get(f"/api/evidence/{d_a1['evidence_id']}/artifact", headers=_hdr(tok_a)).status_code == 200


def test_world_and_export_endpoints_enforce_tenant_isolation_and_persist_across_restart(isolated_env):
    app, db_mod, ws_mod, tmp_path = isolated_env
    tok_a = issue_scoped_token("tenant-a", "user-a", Role.ADMIN)
    tok_b = issue_scoped_token("tenant-b", "user-b", Role.ADMIN)
    payload = _make_jpeg_bytes(b"A")

    with TestClient(app) as client:
        # Setup Tenant A world + reconstruction + export + command
        wid_a = client.post("/api/worlds", json={"name": "World A"}, headers=_hdr(tok_a)).json()["id"]
        sid_a = client.post("/api/sessions", json={"name": "Sess A"}, headers=_hdr(tok_a)).json()["id"]
        assert client.post(f"/api/worlds/{wid_a}/attach/{sid_a}", headers=_hdr(tok_a)).status_code == 200
        for i in range(2):
            r = client.post(
                f"/api/uploads?session_id={sid_a}",
                files={"file": (f"a{i}.jpg", payload + bytes([i]), "image/jpeg")},
                headers=_hdr(tok_a),
            )
            assert r.status_code == 201
        job_a = client.post(f"/api/sessions/{sid_a}/reconstruct", headers=_hdr(tok_a)).json()["job_id"]
        res_a = _wait_job(client, job_a, _hdr(tok_a))
        assert res_a["status"] in ("succeeded", "partial"), res_a
        ver_a = res_a["payload"]["version_id"]

        wir_a = client.get(f"/api/worlds/{wid_a}/worldir", headers=_hdr(tok_a)).json()
        ent_a = next(iter(wir_a["entities"]))

        exp_a = client.post(
            f"/api/worlds/{wid_a}/export",
            json={"format": "gltf"},
            headers=_hdr(tok_a),
        )
        assert exp_a.status_code == 200, exp_a.text
        dl_url_a = exp_a.json()["download_url"]
        hash_a = dl_url_a.split("/export/")[1].split("/download")[0]

        # Setup Tenant B world
        wid_b = client.post("/api/worlds", json={"name": "World B"}, headers=_hdr(tok_b)).json()["id"]

        # 1. All audited endpoints must deny Tenant B on wid_a (404) and unauthenticated caller (401)
        read_paths = [
            f"/api/worlds/{wid_a}/worldir",
            f"/api/worlds/{wid_a}/entities/{ent_a}/provenance",
            f"/api/worlds/{wid_a}/entities/{ent_a}/confidence",
            f"/api/worlds/{wid_a}/report",
            f"/api/worlds/{wid_a}/review/coco",
            f"/api/worlds/{wid_a}/points",
            f"/api/worlds/{wid_a}/cameras",
            f"/api/worlds/{wid_a}/scale/suggestion",
            f"/api/worlds/{wid_a}/quality",
            f"/api/worlds/{wid_a}/spatial-intelligence",
            f"/api/worlds/{wid_a}/space-graph",
            f"/api/worlds/{wid_a}/query/spaces",
            f"/api/worlds/{wid_a}/export/package",
            dl_url_a,
        ]
        for path in read_paths:
            assert client.get(path, headers=_hdr(tok_b)).status_code == 404, f"Tenant B accessed {path}"
            assert client.get(path).status_code == 401, f"Unauthenticated accessed {path}"

        # 2. Mutating endpoints must also deny Tenant B
        assert client.post(
            f"/api/worlds/{wid_a}/scale/suggestion/reject",
            json={"source_entity_id": ent_a},
            headers=_hdr(tok_b),
        ).status_code == 404
        assert client.post(
            f"/api/worlds/{wid_a}/export",
            json={"format": "gltf"},
            headers=_hdr(tok_b),
        ).status_code == 404
        assert client.post(
            f"/api/worlds/{wid_a}/commands",
            json={"command_type": "move", "entity_id": ent_a, "params": {"dx": 1.0}},
            headers=_hdr(tok_b),
        ).status_code == 404
        assert client.post(f"/api/worlds/{wid_a}/undo", headers=_hdr(tok_b)).status_code == 404
        assert client.post(f"/api/worlds/{wid_a}/redo", headers=_hdr(tok_b)).status_code == 404

        # 3. Cross-tenant version_id & export content_hash probes using Tenant B's OWN wid_b must fail with 404
        assert client.get(f"/api/worlds/{wid_b}/space-graph?version_id={ver_a}", headers=_hdr(tok_b)).status_code == 404
        assert client.get(f"/api/worlds/{wid_b}/query/spaces?version_id={ver_a}", headers=_hdr(tok_b)).status_code == 404
        assert client.get(f"/api/worlds/{wid_b}/diff?base={ver_a}&head={ver_a}", headers=_hdr(tok_b)).status_code == 404
        assert client.get(
            f"/api/worlds/{wid_b}/export/{hash_a}/download?format=gltf",
            headers=_hdr(tok_b),
        ).status_code == 404

        # 4. Owner succeeds on audited endpoints
        assert client.get(f"/api/worlds/{wid_a}/report", headers=_hdr(tok_a)).status_code == 200
        assert client.get(f"/api/worlds/{wid_a}/cameras", headers=_hdr(tok_a)).status_code == 200
        assert client.get(f"/api/worlds/{wid_a}/quality", headers=_hdr(tok_a)).status_code == 200
        assert client.get(f"/api/worlds/{wid_a}/space-graph", headers=_hdr(tok_a)).status_code == 200
        assert client.get(dl_url_a, headers=_hdr(tok_a)).status_code == 200

    # 5. Fresh-process persistence reload: dispose DB & reload app over same SQLite + WorldStore
    asyncio.run(db_mod.dispose_db())
    ws_mod._store = None
    import apps.api.main as main_mod
    importlib.reload(db_mod)
    importlib.reload(main_mod)

    with TestClient(main_mod.app) as fresh:
        assert fresh.get(f"/api/worlds/{wid_a}/worldir", headers=_hdr(tok_b)).status_code == 404
        assert fresh.get(f"/api/worlds/{wid_a}/report", headers=_hdr(tok_b)).status_code == 404
        assert fresh.get(dl_url_a, headers=_hdr(tok_b)).status_code == 404
        assert fresh.get(f"/api/worlds/{wid_a}/worldir", headers=_hdr(tok_a)).status_code == 200
        assert fresh.get(dl_url_a, headers=_hdr(tok_a)).status_code == 200


def test_background_worker_refuses_cross_tenant_session_world_mismatch(isolated_env):
    app, db_mod, _, _ = isolated_env
    tok_a = issue_scoped_token("tenant-a", "user-a", Role.ADMIN)
    tok_b = issue_scoped_token("tenant-b", "user-b", Role.ADMIN)

    with TestClient(app) as client:
        wid_a = client.post("/api/worlds", json={"name": "World A"}, headers=_hdr(tok_a)).json()["id"]
        sid_b = client.post("/api/sessions", json={"name": "Sess B"}, headers=_hdr(tok_b)).json()["id"]

        # Tamper DB directly to simulate a stale/forged cross-tenant session->world link
        async def _tamper():
            from apps.api import jobs as jobrunner
            from apps.api.models import Session

            async with db_mod.get_sessionmaker()() as db:
                s = await db.get(Session, sid_b)
                s.world_id = wid_a
                job = jobrunner.enqueue_job(db, jobrunner.RECONSTRUCT_SESSION, "session", sid_b, team_id="tenant-b")
                await db.commit()
                return job.id

        job_id = asyncio.run(_tamper())
        res = _wait_job(client, job_id, _hdr(tok_b))
        assert res["status"] == "failed", res
        assert "Cross-tenant" in (res.get("error") or ""), res


def test_alembic_migration_0007_upgrade_and_downgrade(tmp_path):
    import sqlite3

    from alembic import command
    from alembic.config import Config

    db_file = tmp_path / "mig_test.db"
    cfg = Config("apps/api/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_file}")

    command.upgrade(cfg, "head")
    con = sqlite3.connect(db_file)
    indexes = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    con.close()
    assert "ix_evidence_team_session_checksum" in indexes
    assert "uq_evidence_team_session_checksum" in indexes
    assert "ix_uploads_team_session_checksum" in indexes

    command.downgrade(cfg, "0006")
    con = sqlite3.connect(db_file)
    indexes_after = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    con.close()
    assert "uq_evidence_team_session_checksum" not in indexes_after
    assert "ix_evidence_team_session_checksum" not in indexes_after
    assert "ix_uploads_team_session_checksum" not in indexes_after
