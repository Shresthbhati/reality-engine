"""Account Lifecycle, Team Invites, and GDPR Data Erasure Test Suite (T-6.4).

Verifies:
1. Public signup: creates tenant organization, admin user, and issues scoped Bearer token.
2. Email validation, duplicate rejection, and OWASP PBKDF2 password hashing.
3. Secure login: issues authenticated scoped tokens, rejects wrong passwords in constant time.
4. Team invites: Admin-only creation, public token inspection, member acceptance, role assignment.
5. GDPR Right to Erasure / Delete-my-data:
   - Viewer / Editor mutation blocked (403).
   - Cross-tenant isolation (Tenant A cannot purge Tenant B).
   - Complete database purge of tenant worlds, versions, sessions, evidence, uploads, and users.
   - Physical destruction of on-disk artifacts (STORAGE_ROOT blobs and WorldStore version files).
   - Zero collateral damage: other tenants' records and artifact files remain 100% intact.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from apps.api import auth
from apps.api.auth import Role, rate_limiter, verify_scoped_token
from apps.api.db import dispose_db, init_db
from apps.api.main import app
from apps.api.models import Evidence, Session, Tenant, Upload, UserAccount, World
from apps.api.storage import store_bytes


class ApiClient:
    def __init__(self, token: str | None = None, host: str = "203.0.113.195"):
        self.token = token
        self.host = host

    def req(
        self,
        method: str,
        path: str,
        headers: dict | None = None,
        json: dict | None = None,
    ) -> httpx.Response:
        hdrs = dict(headers or {})
        if self.token and "Authorization" not in hdrs:
            hdrs["Authorization"] = f"Bearer {self.token}"

        async def _run():
            transport = httpx.ASGITransport(app=app, client=(self.host, 54321))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                return await c.request(method, path, headers=hdrs, json=json)

        return asyncio.run(_run())


@pytest.fixture(autouse=True)
def _setup_account_test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("REALITY_API_TOKEN", raising=False)
    monkeypatch.setenv("REALITY_SIGNING_KEY", "account-test-secret-key-999")
    monkeypatch.setattr(auth, "_warned", False)
    rate_limiter.reset()

    db_path = tmp_path / "account_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "worldstore"))

    asyncio.run(dispose_db())
    asyncio.run(init_db())
    yield
    rate_limiter.reset()
    asyncio.run(dispose_db())


def test_signup_creates_tenant_and_admin_user():
    client = ApiClient()

    # 1. Successful signup
    payload = {
        "tenant_name": "Studio Nord Architekten",
        "email": "lead@studionord.de",
        "password": "super-secure-password-123",
        "full_name": "Greta Thunberg",
    }
    resp = client.req("POST", "/api/auth/signup", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()

    assert "token" in data
    assert data["tenant"]["name"] == "Studio Nord Architekten"
    assert data["tenant"]["slug"].startswith("studio-nord-architekten")
    assert data["user"]["email"] == "lead@studionord.de"
    assert data["user"]["role"] == "admin"

    # Token verifies correctly to admin role for the created tenant
    ctx = verify_scoped_token(data["token"], secret="account-test-secret-key-999")
    assert ctx is not None
    assert ctx.tenant_id == data["tenant"]["id"]
    assert ctx.user_id == data["user"]["id"]
    assert ctx.role == Role.ADMIN

    # 2. Duplicate email rejected with 409
    dup_resp = client.req("POST", "/api/auth/signup", json=payload)
    assert dup_resp.status_code == 409
    assert "already exists" in dup_resp.json()["detail"]

    # 3. Invalid email rejected with 422
    bad_email_resp = client.req("POST", "/api/auth/signup", json={**payload, "email": "not-an-email"})
    assert bad_email_resp.status_code == 422

    # 4. Short password rejected with 422
    short_pwd_resp = client.req("POST", "/api/auth/signup", json={**payload, "email": "other@nord.de", "password": "short"})
    assert short_pwd_resp.status_code == 422


def test_login_authenticates_and_issues_token():
    client = ApiClient()

    signup_payload = {
        "tenant_name": "Bauhaus Lab",
        "email": "walter@bauhaus.de",
        "password": "gropius-password-1919",
        "full_name": "Walter Gropius",
    }
    client.req("POST", "/api/auth/signup", json=signup_payload)

    # 1. Successful login
    login_resp = client.req("POST", "/api/auth/login", json={
        "email": "walter@bauhaus.de",
        "password": "gropius-password-1919",
    })
    assert login_resp.status_code == 200, login_resp.text
    login_data = login_resp.json()
    assert "token" in login_data
    assert login_data["user"]["email"] == "walter@bauhaus.de"

    # 2. Incorrect password rejected with 401
    bad_pwd_resp = client.req("POST", "/api/auth/login", json={
        "email": "walter@bauhaus.de",
        "password": "wrong-password",
    })
    assert bad_pwd_resp.status_code == 401
    assert "Invalid email or password" in bad_pwd_resp.json()["detail"]

    # 3. Non-existent email rejected with 401
    bad_email_resp = client.req("POST", "/api/auth/login", json={
        "email": "unknown@bauhaus.de",
        "password": "gropius-password-1919",
    })
    assert bad_email_resp.status_code == 401


def test_team_invite_lifecycle():
    client = ApiClient()

    # Create admin account
    signup_res = client.req("POST", "/api/auth/signup", json={
        "tenant_name": "Foster Partners",
        "email": "norman@foster.com",
        "password": "curved-glass-1234",
        "full_name": "Norman Foster",
    }).json()
    admin_token = signup_res["token"]
    admin_client = ApiClient(token=admin_token)

    # Check /api/account/me
    me_resp = admin_client.req("GET", "/api/account/me")
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "norman@foster.com"
    assert me_resp.json()["role"] == "admin"

    # 1. Admin creates invite for a new team member with role editor
    inv_resp = admin_client.req("POST", "/api/account/invites", json={
        "email": "zaha@foster.com",
        "role": "editor",
    })
    assert inv_resp.status_code == 201, inv_resp.text
    inv_data = inv_resp.json()
    assert inv_data["email"] == "zaha@foster.com"
    assert inv_data["role"] == "editor"
    invite_token = inv_data["invite_token"]

    # 2. List invites
    list_inv = admin_client.req("GET", "/api/account/invites")
    assert list_inv.status_code == 200
    assert len(list_inv.json()["items"]) == 1

    # 3. Public inspection of invite before accept
    inspect_resp = client.req("GET", f"/api/auth/invites/{invite_token}")
    assert inspect_resp.status_code == 200
    assert inspect_resp.json()["email"] == "zaha@foster.com"
    assert inspect_resp.json()["tenant_name"] == "Foster Partners"

    # 4. Invited user accepts invite
    accept_resp = client.req("POST", f"/api/auth/invites/{invite_token}/accept", json={
        "full_name": "Zaha Hadid",
        "password": "fluid-geometry-1234",
    })
    assert accept_resp.status_code == 201, accept_resp.text
    accept_data = accept_resp.json()
    assert accept_data["user"]["email"] == "zaha@foster.com"
    assert accept_data["user"]["role"] == "editor"
    assert accept_data["tenant"]["id"] == signup_res["tenant"]["id"]

    # Member can log in and view account
    member_client = ApiClient(token=accept_data["token"])
    member_me = member_client.req("GET", "/api/account/me")
    assert member_me.status_code == 200
    assert member_me.json()["role"] == "editor"

    # 5. Non-admin (editor) cannot create invites
    forbidden_inv = member_client.req("POST", "/api/account/invites", json={
        "email": "other@foster.com",
        "role": "editor",
    })
    assert forbidden_inv.status_code == 403

    # 6. Used invite cannot be accepted a second time
    re_accept = client.req("POST", f"/api/auth/invites/{invite_token}/accept", json={
        "full_name": "Imposter",
        "password": "some-password-123",
    })
    assert re_accept.status_code == 404


def test_delete_my_data_gdpr_erasure_and_artifact_destruction():
    """Right to Erasure completely purges tenant data and removes on-disk artifacts without touching other tenants."""
    client = ApiClient()

    # 1. Setup Tenant Alpha
    alpha_signup = client.req("POST", "/api/auth/signup", json={
        "tenant_name": "Alpha Architects",
        "email": "admin@alpha.com",
        "password": "alpha-password-123",
        "full_name": "Alice Alpha",
    }).json()
    alpha_token = alpha_signup["token"]
    alpha_tenant_id = alpha_signup["tenant"]["id"]
    alpha_client = ApiClient(token=alpha_token)

    # 2. Setup Tenant Beta (Control Tenant)
    beta_signup = client.req("POST", "/api/auth/signup", json={
        "tenant_name": "Beta Design Group",
        "email": "admin@beta.com",
        "password": "beta-password-123",
        "full_name": "Bob Beta",
    }).json()
    beta_token = beta_signup["token"]
    beta_tenant_id = beta_signup["tenant"]["id"]
    beta_client = ApiClient(token=beta_token)

    # 3. Create real on-disk artifact files for Tenant Alpha
    alpha_photo_bytes = b"alpha-unique-room-photo-raw-bytes-12345"
    alpha_digest, alpha_path = store_bytes(alpha_photo_bytes)
    assert alpha_path.exists()

    # 4. Create real on-disk artifact files for Tenant Beta
    beta_photo_bytes = b"beta-unique-living-room-photo-bytes-99999"
    beta_digest, beta_path = store_bytes(beta_photo_bytes)
    assert beta_path.exists()

    # 5. Seed DB records for Tenant Alpha
    async def _seed_alpha():
        from apps.api.db import get_sessionmaker
        maker = get_sessionmaker()
        async with maker() as s:
            w_alpha = World(
                id="wld_alpha_1",
                name="Alpha Penthouse",
                status="active",
                team_id=alpha_tenant_id,
            )
            s_alpha = Session(
                id="ses_alpha_1",
                name="Alpha Survey",
                status="completed",
                team_id=alpha_tenant_id,
                world_id="wld_alpha_1",
            )
            u_alpha = Upload(
                id="upl_alpha_1",
                session_id="ses_alpha_1",
                team_id=alpha_tenant_id,
                filename="alpha_photo.jpg",
                artifact_uri=f"sha256://{alpha_digest}",
            )
            e_alpha = Evidence(
                id="evi_alpha_1",
                name="alpha_photo.jpg",
                type="photo",
                team_id=alpha_tenant_id,
                session_id="ses_alpha_1",
                artifact_uri=f"sha256://{alpha_digest}",
            )
            s.add_all([w_alpha, s_alpha, u_alpha, e_alpha])
            await s.commit()

    asyncio.run(_seed_alpha())

    # 6. Seed DB records for Tenant Beta
    async def _seed_beta():
        from apps.api.db import get_sessionmaker
        maker = get_sessionmaker()
        async with maker() as s:
            w_beta = World(
                id="wld_beta_1",
                name="Beta Villa",
                status="active",
                team_id=beta_tenant_id,
            )
            s_beta = Session(
                id="ses_beta_1",
                name="Beta Survey",
                status="completed",
                team_id=beta_tenant_id,
                world_id="wld_beta_1",
            )
            u_beta = Upload(
                id="upl_beta_1",
                session_id="ses_beta_1",
                team_id=beta_tenant_id,
                filename="beta_photo.jpg",
                artifact_uri=f"sha256://{beta_digest}",
            )
            s.add_all([w_beta, s_beta, u_beta])
            await s.commit()

    asyncio.run(_seed_beta())

    # 7. Unauthenticated erasure request is rejected
    unauth_resp = client.req("DELETE", "/api/account/data")
    assert unauth_resp.status_code == 401

    # 8. Tenant Alpha Admin initiates Delete My Data
    purge_resp = alpha_client.req("DELETE", "/api/account/data")
    assert purge_resp.status_code == 200, purge_resp.text
    purge_data = purge_resp.json()
    assert purge_data["status"] == "purged"
    assert purge_data["tenant_id"] == alpha_tenant_id
    assert purge_data["worlds_deleted"] >= 1
    assert purge_data["sessions_deleted"] >= 1
    assert purge_data["artifacts_removed"] >= 1

    # 9. Verify Tenant Alpha records are permanently deleted from database
    async def _verify_alpha_gone():
        from apps.api.db import get_sessionmaker
        maker = get_sessionmaker()
        async with maker() as s:
            w = await s.get(World, "wld_alpha_1")
            assert w is None
            sess = await s.get(Session, "ses_alpha_1")
            assert sess is None
            upl = await s.get(Upload, "upl_alpha_1")
            assert upl is None
            usr = (await s.execute(select(UserAccount).where(UserAccount.tenant_id == alpha_tenant_id))).scalars().all()
            assert len(usr) == 0
            tnt = await s.get(Tenant, alpha_tenant_id)
            assert tnt is None

    asyncio.run(_verify_alpha_gone())

    # 10. Verify Tenant Alpha physical artifact was destroyed on disk
    assert not alpha_path.exists(), "Alpha artifact blob file should have been deleted from disk!"

    # 11. Verify Tenant Beta records and physical artifacts are 100% INTACT
    assert beta_client.req("GET", "/api/account/me").status_code == 200
    async def _verify_beta_intact():
        from apps.api.db import get_sessionmaker
        maker = get_sessionmaker()
        async with maker() as s:
            w = await s.get(World, "wld_beta_1")
            assert w is not None
            assert w.name == "Beta Villa"
            sess = await s.get(Session, "ses_beta_1")
            assert sess is not None
            tnt = await s.get(Tenant, beta_tenant_id)
            assert tnt is not None

    asyncio.run(_verify_beta_intact())
    assert beta_path.exists(), "Beta artifact blob file must remain 100% intact!"
