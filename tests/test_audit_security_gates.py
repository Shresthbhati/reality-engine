"""Mandatory security regression tests from the live audit (SEC-001 through SEC-005).

Validates:
- SEC-T01: Tenant A admin cannot read Tenant B world (404)
- SEC-T02: Tenant A admin cannot modify Tenant B world (404)
- SEC-T03: NULL-owned legacy record queried by non-default tenant is not exposed (404)
- SEC-T04: Forged token using development fallback rejected in production environment
- SEC-T05: Missing signing key in production halts startup
- SEC-T06: Expired scoped token rejected
- SEC-T08: Revoked share token rejected even across server restarts
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import Role, issue_scoped_token, issue_share_token, revoke_share_token_in_memory, verify_scoped_token
from apps.api.models import World, WorldShareLink

TEST_SECRET = "security-audit-test-secret-32-chars-long"


@pytest.fixture
def auth_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'sec_app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_SIGNING_KEY", TEST_SECRET)
    monkeypatch.setenv("REALITY_API_TOKEN", "system-master-token")
    
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    
    with TestClient(main_mod.app) as c:
        yield c


def test_sec_t01_tenant_a_admin_cannot_read_tenant_b_world(auth_api):
    """SEC-T01: Tenant A administrator receives 404 when querying Tenant B's world."""
    token_b = issue_scoped_token("tenant-b", "admin-b", Role.ADMIN, secret=TEST_SECRET)
    token_a = issue_scoped_token("tenant-a", "admin-a", Role.ADMIN, secret=TEST_SECRET)
    root_token = issue_scoped_token("root", "sysadmin", Role.SYSTEM_ADMIN, secret=TEST_SECRET)

    # 1. Tenant B creates a world
    res = auth_api.post("/api/worlds", json={"name": "Tenant B Secret Room"}, headers={"Authorization": f"Bearer {token_b}"})
    assert res.status_code == 201, res.text
    world_b_id = res.json()["id"]

    # 2. Tenant A admin attempts to read Tenant B's world -> must be 404
    read_res = auth_api.get(f"/api/worlds/{world_b_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert read_res.status_code == 404, f"Tenant A admin could read Tenant B world: {read_res.status_code}"

    # 3. Tenant A admin list worlds -> Tenant B world must not be listed
    list_res = auth_api.get("/api/worlds", headers={"Authorization": f"Bearer {token_a}"})
    assert list_res.status_code == 200
    listed_ids = [w["id"] for w in list_res.json()["items"]]
    assert world_b_id not in listed_ids, "Tenant B world appeared in Tenant A's world list"

    # 4. System admin CAN read Tenant B's world
    sys_res = auth_api.get(f"/api/worlds/{world_b_id}", headers={"Authorization": f"Bearer {root_token}"})
    assert sys_res.status_code == 200, "System admin could not read world"


def test_sec_t02_tenant_a_admin_cannot_modify_tenant_b_world(auth_api):
    """SEC-T02: Tenant A administrator receives 404 when attempting to mutate Tenant B's world."""
    token_b = issue_scoped_token("tenant-b", "admin-b", Role.ADMIN, secret=TEST_SECRET)
    token_a = issue_scoped_token("tenant-a", "admin-a", Role.ADMIN, secret=TEST_SECRET)

    res = auth_api.post("/api/worlds", json={"name": "Tenant B Immutable"}, headers={"Authorization": f"Bearer {token_b}"})
    assert res.status_code == 201
    world_b_id = res.json()["id"]

    # Tenant A admin attempts to rename Tenant B's world
    patch_res = auth_api.patch(f"/api/worlds/{world_b_id}", json={"name": "Hacked Room"}, headers={"Authorization": f"Bearer {token_a}"})
    assert patch_res.status_code == 404, "Tenant A admin was able to modify Tenant B world"


def test_sec_t03_null_owned_legacy_records_not_exposed(auth_api):
    """SEC-T03: Rows where team_id IS NULL are NOT visible to non-default tenant callers."""
    # Create a world directly with no team_id (representing a legacy unowned row)
    from apps.api.db import get_sessionmaker
    import asyncio
    maker = get_sessionmaker()

    async def _insert_legacy():
        async with maker() as db:
            w = World(id="wld_legacy_null_team", name="Legacy Unowned World", status="active", team_id=None)
            db.add(w)
            await db.commit()

    asyncio.run(_insert_legacy())

    token_tenant = issue_scoped_token("tenant-c", "user-c", Role.ADMIN, secret=TEST_SECRET)
    token_default = issue_scoped_token("default", "user-def", Role.ADMIN, secret=TEST_SECRET)

    # Tenant C admin queries legacy world -> must receive 404
    res = auth_api.get("/api/worlds/wld_legacy_null_team", headers={"Authorization": f"Bearer {token_tenant}"})
    assert res.status_code == 404, "Legacy null-owned record was exposed to non-default tenant"

    # Fail closed: not even the "default" tenant sees an unowned row (enforce_project_isolation: unowned/legacy records are
    # hidden from every non-global-admin tenant); only a global admin (root/system tenant) can see and re-assign it.
    res_def = auth_api.get("/api/worlds/wld_legacy_null_team", headers={"Authorization": f"Bearer {token_default}"})
    assert res_def.status_code == 404, "Legacy null-owned record was exposed to the default tenant"
    token_root = issue_scoped_token("root", "root-admin", Role.ADMIN, secret=TEST_SECRET)
    res_root = auth_api.get("/api/worlds/wld_legacy_null_team", headers={"Authorization": f"Bearer {token_root}"})
    assert res_root.status_code == 200


def test_sec_t04_production_forged_dev_token_rejected(monkeypatch):
    """SEC-T04: In production mode, tokens signed with the fallback dev key are rejected."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REALITY_SIGNING_KEY", "real-prod-secret-key-32-bytes-xyz")

    # Attacker crafts token with the dev fallback key
    forged_token = issue_scoped_token("tenant-x", "admin-x", Role.ADMIN, secret="reality-default-dev-secret-key")
    # Verify against server configured with real prod secret
    parsed = verify_scoped_token(forged_token)
    assert parsed is None, "Forged dev token was accepted when production key is configured"


def test_sec_t05_missing_production_signing_key_fails():
    """SEC-T05: In production mode, missing signing key causes a fatal error rather than silent dev fallback."""
    import os
    from apps.api.auth import _get_signing_secret
    old_env = os.environ.get("ENVIRONMENT")
    old_key = os.environ.pop("REALITY_SIGNING_KEY", None)
    old_token = os.environ.pop("REALITY_API_TOKEN", None)
    try:
        os.environ["ENVIRONMENT"] = "production"
        with pytest.raises(RuntimeError, match="FATAL SEC-002"):
            _get_signing_secret()
    finally:
        if old_env is not None:
            os.environ["ENVIRONMENT"] = old_env
        else:
            os.environ.pop("ENVIRONMENT", None)
        if old_key is not None:
            os.environ["REALITY_SIGNING_KEY"] = old_key
        if old_token is not None:
            os.environ["REALITY_API_TOKEN"] = old_token


def test_sec_t06_expired_scoped_token_rejected():
    """SEC-T06: Expired scoped token is rejected."""
    # Issue a token with timestamp 2 days ago
    two_days_ago = int(time.time()) - (2 * 86400)
    expired_token = issue_scoped_token("tenant-x", "user-x", Role.EDITOR, secret=TEST_SECRET, timestamp=two_days_ago)
    assert verify_scoped_token(expired_token, secret=TEST_SECRET) is None, "Expired scoped token was accepted"

    # Token with timestamp in distant future (> 300s clock skew) is also rejected
    future_token = issue_scoped_token("tenant-x", "user-x", Role.EDITOR, secret=TEST_SECRET, timestamp=int(time.time()) + 1000)
    assert verify_scoped_token(future_token, secret=TEST_SECRET) is None, "Future token beyond clock skew was accepted"


def test_sec_t08_revoked_share_token_survives_restart(auth_api, monkeypatch):
    """SEC-T08: Revoked share link remains revoked even if in-memory cache is wiped (simulating server restart)."""
    # 1. Create a world and share link
    token_a = issue_scoped_token("tenant-a", "user-a", Role.ADMIN, secret=TEST_SECRET)
    res = auth_api.post("/api/worlds", json={"name": "Shareable World"}, headers={"Authorization": f"Bearer {token_a}"})
    assert res.status_code == 201
    wid = res.json()["id"]

    share_res = auth_api.post(f"/api/worlds/{wid}/shares", json={"label": "Client Review"}, headers={"Authorization": f"Bearer {token_a}"})
    assert share_res.status_code == 200
    share_id = share_res.json()["id"]
    share_token = share_res.json()["token"]

    # 2. Verify share token works initially
    acc_res = auth_api.get(f"/api/worlds/shared/access/{share_token}")
    assert acc_res.status_code == 200

    # 3. Revoke the share link
    rev_res = auth_api.post(f"/api/worlds/{wid}/shares/{share_id}/revoke", headers={"Authorization": f"Bearer {token_a}"})
    assert rev_res.status_code == 200

    # 4. Simulate server restart by completely clearing in-memory revocation set
    from apps.api.auth import _REVOKED_SHARE_TOKENS
    _REVOKED_SHARE_TOKENS.clear()

    # 5. Token must STILL be rejected through both shared endpoint and generic Bearer middleware
    acc_after = auth_api.get(f"/api/worlds/shared/access/{share_token}")
    assert acc_after.status_code in (401, 403), "Revoked share token was accepted by shared endpoint after cache clear"

    # Also test generic bearer authentication using revoked share token
    generic_res = auth_api.get(f"/api/worlds/{wid}", headers={"Authorization": f"Bearer {share_token}"})
    assert generic_res.status_code in (401, 403), "Revoked share token authenticated through generic Bearer route"
