"""Phase 1 Security & Tenant Isolation Adversarial Tests.

Acceptance: A tenant administrator cannot access another tenant's data,
even with forged, expired or manipulated credentials.
"""
from __future__ import annotations

import importlib
import time

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import (
    Role,
    _get_signing_secret,
    issue_scoped_token,
    rate_limiter,
    verify_scoped_token,
)
from apps.api.models import World, new_id

TEST_SECRET = "phase1-adversarial-test-key-32bytes-secure"


class Api:
    def __init__(self, client: TestClient, secret: str = TEST_SECRET):
        self.c = client
        self.secret = secret

    def token_for(self, tenant: str, role: Role, user: str = "u1", **kw) -> str:
        return issue_scoped_token(tenant, user, role, secret=self.secret, **kw)

    def request(self, token: str, method: str, path: str, **kw):
        headers = kw.pop("headers", {})
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return self.c.request(method, path, headers=headers, **kw)


@pytest.fixture()
def sec_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'sec.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_SIGNING_KEY", TEST_SECRET)
    monkeypatch.delenv("REALITY_API_TOKEN", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    rate_limiter.reset()
    with TestClient(main_mod.app) as c:
        yield Api(c, secret=TEST_SECRET)
    rate_limiter.reset()


def test_hardcoded_signing_secret_fallback_cannot_forge_tokens(monkeypatch):
    """Removing hardcoded signing secret: forged tokens with static keys fail."""
    # An attacker attempts to forge a token using the old default key
    fake_secret = "reality-default-dev-secret-key"
    forged_token = issue_scoped_token("tenant-victim", "admin-user", Role.SYSTEM_ADMIN, secret=fake_secret)

    # In an environment with an ephemeral/configured secret, verification MUST fail
    verified = verify_scoped_token(forged_token, secret=TEST_SECRET)
    assert verified is None, "Forged token signed with default dev key must be rejected"

    # In production without signing key configured, _get_signing_secret must fail closed
    monkeypatch.delenv("REALITY_SIGNING_KEY", raising=False)
    monkeypatch.delenv("REALITY_API_TOKEN", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(RuntimeError, match="FATAL SEC-002"):
        _get_signing_secret()


def test_token_expiry_issuer_and_audience_enforced():
    """Token validation rejects expired tokens, wrong issuer, and wrong audience."""
    # 1. Expired token (issued 2 hours ago, ttl 3600s)
    now = int(time.time())
    expired_token = issue_scoped_token(
        "tenant-a", "user-1", Role.EDITOR, secret=TEST_SECRET,
        timestamp=now - 7200, expires_in_s=3600,
    )
    assert verify_scoped_token(expired_token, secret=TEST_SECRET) is None

    # 2. Token from the future beyond clock skew (e.g. +300s)
    future_token = issue_scoped_token(
        "tenant-a", "user-1", Role.EDITOR, secret=TEST_SECRET,
        timestamp=now + 300, expires_in_s=3600,
    )
    assert verify_scoped_token(future_token, secret=TEST_SECRET) is None

    # 3. Wrong issuer
    bad_iss_token = issue_scoped_token(
        "tenant-a", "user-1", Role.EDITOR, secret=TEST_SECRET,
        issuer="untrusted-authority",
    )
    assert verify_scoped_token(bad_iss_token, secret=TEST_SECRET) is None

    # 4. Wrong audience
    bad_aud_token = issue_scoped_token(
        "tenant-a", "user-1", Role.EDITOR, secret=TEST_SECRET,
        audience="other-app",
    )
    assert verify_scoped_token(bad_aud_token, secret=TEST_SECRET) is None

    # 5. Valid token
    valid_token = issue_scoped_token("tenant-a", "user-1", Role.EDITOR, secret=TEST_SECRET)
    ctx = verify_scoped_token(valid_token, secret=TEST_SECRET)
    assert ctx is not None
    assert ctx.tenant_id == "tenant-a"
    assert ctx.role == Role.EDITOR


def test_tenant_admin_cannot_access_or_modify_other_tenant_data(sec_api):
    """Tenant administrator on tenant-b CANNOT read, modify, or delete tenant-a resources."""
    tok_a = sec_api.token_for("tenant-a", Role.EDITOR)
    tok_b_admin = sec_api.token_for("tenant-b", Role.ADMIN)

    # Tenant A creates a world
    res = sec_api.request(tok_a, "POST", "/api/worlds", json={"name": "A Private World"})
    assert res.status_code in (200, 201), res.text
    world_a_id = res.json()["id"]

    # 1. Tenant B admin tries to GET world A -> 404
    get_res = sec_api.request(tok_b_admin, "GET", f"/api/worlds/{world_a_id}")
    assert get_res.status_code == 404
    assert "not found" in get_res.json().get("detail", "").lower()

    # 2. Tenant B admin tries to PATCH world A -> 404
    patch_res = sec_api.request(tok_b_admin, "PATCH", f"/api/worlds/{world_a_id}", json={"name": "Hacked"})
    assert patch_res.status_code == 404

    # 3. Tenant B admin tries to DELETE world A -> 404
    del_res = sec_api.request(tok_b_admin, "DELETE", f"/api/worlds/{world_a_id}")
    assert del_res.status_code == 404

    # 4. World A remains intact and unchanged
    a_check = sec_api.request(tok_a, "GET", f"/api/worlds/{world_a_id}")
    assert a_check.status_code == 200
    assert a_check.json()["name"] == "A Private World"

    # 5. World A does not appear in tenant B's world listing
    b_list = sec_api.request(tok_b_admin, "GET", "/api/worlds")
    assert b_list.status_code == 200
    world_ids = [w["id"] for w in b_list.json()["items"]]
    assert world_a_id not in world_ids


def test_null_ownership_records_eliminated_from_cross_tenant_visibility(sec_api):
    """Records with NULL team_id in database are strictly hidden from non-global-admin tenants."""
    tok_a_admin = sec_api.token_for("tenant-a", Role.ADMIN)
    tok_b_admin = sec_api.token_for("tenant-b", Role.ADMIN)
    tok_root = sec_api.token_for("root", Role.SYSTEM_ADMIN)

    # Insert a legacy world directly into the DB with team_id = NULL
    import asyncio

    from apps.api.db import get_sessionmaker

    legacy_id = new_id("wld")
    async def create_unowned():
        async with get_sessionmaker()() as db:
            w = World(id=legacy_id, name="Unowned Legacy World", status="active", team_id=None)
            db.add(w)
            await db.commit()

    asyncio.run(create_unowned())

    # Tenant A admin cannot see it
    assert sec_api.request(tok_a_admin, "GET", f"/api/worlds/{legacy_id}").status_code == 404
    # Tenant B admin cannot see it
    assert sec_api.request(tok_b_admin, "GET", f"/api/worlds/{legacy_id}").status_code == 404
    # Root system administrator CAN see it
    assert sec_api.request(tok_root, "GET", f"/api/worlds/{legacy_id}").status_code == 200


def test_persistent_share_token_revocation(sec_api):
    """Revoking a share link persists to the database and cannot be re-used after worker restart."""
    import apps.api.auth as auth_mod
    tok_a = sec_api.token_for("tenant-a", Role.EDITOR)

    # 1. Tenant A creates world and share link
    w_res = sec_api.request(tok_a, "POST", "/api/worlds", json={"name": "Shared World"})
    assert w_res.status_code in (200, 201)
    wid = w_res.json()["id"]

    sh_res = sec_api.request(tok_a, "POST", f"/api/worlds/{wid}/shares", json={"label": "Client review"})
    assert sh_res.status_code == 200
    token = sh_res.json()["token"]
    share_id = sh_res.json()["id"]

    # 2. Access via public share access endpoint -> 200
    acc1 = sec_api.request("", "GET", f"/api/worlds/shared/access/{token}")
    assert acc1.status_code == 200

    # 3. Access via query param share token on standard route -> 200
    qp1 = sec_api.request("", "GET", f"/api/worlds/{wid}?share_token={token}")
    assert qp1.status_code == 200

    # 4. Revoke share link
    rev = sec_api.request(tok_a, "POST", f"/api/worlds/{wid}/shares/{share_id}/revoke")
    assert rev.status_code == 200
    assert rev.json()["status"] == "revoked"

    # 5. Clear in-memory cache to simulate server/worker restart
    auth_mod._REVOKED_SHARE_TOKENS.clear()

    # 6. Persistent DB check: Public share access must be rejected (403)
    acc2 = sec_api.request("", "GET", f"/api/worlds/shared/access/{token}")
    assert acc2.status_code == 403

    # 7. Query param and Bearer header access must be rejected (401)
    qp2 = sec_api.request("", "GET", f"/api/worlds/{wid}?share_token={token}")
    assert qp2.status_code == 401

    bearer_res = sec_api.request(token, "GET", f"/api/worlds/{wid}")
    assert bearer_res.status_code == 401
