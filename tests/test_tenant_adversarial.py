"""Phase 1: two-tenant adversarial sweep.

Tenant A owns a world, a session, an upload + evidence, a job, an asset and a site. Tenant B (an editor), a tenant-B
administrator and an unauthenticated caller then try every route that takes one of those ids, with every HTTP method.
No attempt may succeed (no 2xx) and none may crash the server (no 5xx); list endpoints may not reveal A's ids. Positive
controls prove the ids are real (tenant A reaches its own resources) so a 404 means "isolated", not "wrong id".

Synthetic verification: temp SQLite database, signed test tokens, no real data.
"""
from __future__ import annotations

import importlib
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import Role, issue_scoped_token, rate_limiter
from tests.test_activity_ownership import _insert

SECRET = "adversarial-test-secret"
#: Routes that are public by design (a share/invite link IS the credential) or carry no tenant resource.
PUBLIC_PATH_PARTS = ("/api/shared", "/api/share/", "/healthz", "/api/health", "/docs", "/openapi", "/redoc", "{token}", "/api/auth")
ID_PARAMS = ("world_id", "session_id", "evidence_id", "upload_id", "job_id", "asset_id", "site_id")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db.as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_SIGNING_KEY", SECRET)
    monkeypatch.setenv("REALITY_API_TOKEN", "master-token-for-tests")
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    # The sweep sends hundreds of requests from one host. A throttled request (429) is "rejected" for the wrong reason
    # and would make a breach invisible, so the limiter is lifted here and the sweep asserts it never saw a 429.
    monkeypatch.setattr(rate_limiter, "default_rps", 1_000_000.0)
    monkeypatch.setattr(rate_limiter, "default_burst", 1_000_000.0)
    rate_limiter.reset()

    def bearer(tenant, user, role):
        return {"Authorization": f"Bearer {issue_scoped_token(tenant, user, role, secret=SECRET)}"}

    headers = {
        "a": bearer("tenant-a", "user-a", Role.ADMIN),
        "b": bearer("tenant-b", "user-b", Role.EDITOR),
        "b-admin": bearer("tenant-b", "admin-b", Role.ADMIN),
        "anon": {},
    }
    with TestClient(main_mod.app, raise_server_exceptions=False) as client:
        ids = _seed_tenant_a(client, headers["a"], db)
        yield client, headers, ids, main_mod.app
    rate_limiter.reset()


def _seed_tenant_a(client, ha, db) -> dict:
    wid = client.post("/api/worlds", json={"name": "A world"}, headers=ha).json()["id"]
    sid = client.post("/api/sessions", json={"name": "A session"}, headers=ha).json()["id"]
    up = client.post(f"/api/uploads?session_id={sid}", headers=ha,
                     files={"file": ("a.jpg", b"\xff\xd8\xff\xe0tenant-a-private-bytes", "image/jpeg")})
    assert up.status_code == 201, up.text
    con = sqlite3.connect(db)
    _insert(con, "jobs", id="job_a_private", team_id="tenant-a", type="RECONSTRUCT_SESSION", entity_type="session",
            entity_id=sid, status="succeeded")
    _insert(con, "assets", id="ast-a-private", team_id="tenant-a", name="A asset", category="furniture")
    _insert(con, "sites", id="site_a_private", team_id="tenant-a", name="A site")
    con.commit()
    con.close()
    return {"world_id": wid, "session_id": sid, "evidence_id": up.json()["evidence_id"], "upload_id": up.json()["upload_id"],
            "job_id": "job_a_private", "asset_id": "ast-a-private", "site_id": "site_a_private"}


def _concrete(path: str, ids: dict) -> str:
    """Fill a route template with tenant A's real ids (any other parameter gets a throwaway value)."""
    return re.sub(r"\{([^}]+)\}", lambda m: ids.get(m.group(1).split(":")[0], "unknown-sub-id"), path)


def _targets(app, ids):
    # The OpenAPI schema is the complete route table (app.routes nests included routers in this FastAPI version).
    for path, ops in sorted(app.openapi()["paths"].items()):
        if any(p in path for p in PUBLIC_PATH_PARTS):
            continue
        if not any(f"{{{name}}}" in path for name in ID_PARAMS):
            continue
        for method in sorted(m.upper() for m in ops if m.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}):
            yield method, path, _concrete(path, ids)


def test_positive_controls_tenant_a_reaches_its_own_resources(env):
    client, h, ids, _app = env
    assert client.get(f"/api/worlds/{ids['world_id']}", headers=h["a"]).status_code == 200
    assert client.get(f"/api/sessions/{ids['session_id']}", headers=h["a"]).status_code == 200
    assert client.get(f"/api/evidence/{ids['evidence_id']}", headers=h["a"]).status_code == 200
    assert client.get(f"/api/jobs/{ids['job_id']}", headers=h["a"]).status_code == 200


@pytest.mark.parametrize("who", ["b", "b-admin", "anon"])
def test_no_route_taking_a_tenant_resource_id_succeeds_for_another_tenant(env, who):
    client, h, ids, app = env
    attempts, breaches, crashes, throttled = 0, [], [], []
    for method, template, url in _targets(app, ids):
        kwargs = {"headers": h[who]}
        if method in ("POST", "PUT", "PATCH"):
            kwargs["json"] = {}
        res = client.request(method, url, **kwargs)
        attempts += 1
        if 200 <= res.status_code < 300:
            breaches.append(f"{method} {template} -> {res.status_code}")
        elif res.status_code == 429:
            throttled.append(f"{method} {template}")
        elif res.status_code >= 500:
            crashes.append(f"{method} {template} -> {res.status_code}")
    assert attempts > 40, f"the sweep reached only {attempts} routes; the route discovery is broken"
    assert not throttled, f"{len(throttled)} request(s) were rate-limited, so the sweep proved nothing for them: {throttled[:3]}"
    assert not breaches, "cross-tenant access succeeded:\n" + "\n".join(breaches)
    assert not crashes, "foreign-id requests crashed the server:\n" + "\n".join(crashes)


@pytest.mark.parametrize("who", ["b", "b-admin"])
def test_list_endpoints_do_not_reveal_another_tenants_ids(env, who):
    client, h, ids, _app = env
    secret_ids = set(ids.values())
    for path in ("/api/worlds", "/api/sessions", "/api/evidence", "/api/jobs", "/api/activity", "/api/notifications",
                 "/api/assets", "/api/sites"):
        res = client.get(path, headers=h[who])
        if res.status_code == 404:
            continue
        assert res.status_code == 200, f"{path}: {res.status_code}"
        assert not [i for i in secret_ids if i in res.text], f"{path} reveals tenant A's ids to {who}"


def test_tenant_a_data_is_not_changed_by_the_failed_attacks(env):
    client, h, ids, app = env
    for method, _template, url in _targets(app, ids):
        kwargs = {"headers": h["b-admin"]}
        if method in ("POST", "PUT", "PATCH"):
            kwargs["json"] = {}
        client.request(method, url, **kwargs)
    for path in (f"/api/worlds/{ids['world_id']}", f"/api/sessions/{ids['session_id']}",
                 f"/api/evidence/{ids['evidence_id']}", f"/api/jobs/{ids['job_id']}"):
        assert client.get(path, headers=h["a"]).status_code == 200, f"{path} was destroyed or hidden by a cross-tenant call"
