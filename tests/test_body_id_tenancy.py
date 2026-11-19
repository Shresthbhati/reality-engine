"""Body-supplied ids: tenant B must not reach tenant A's records by naming them in a JSON body or query string.

The path-id sweep in test_tenant_adversarial.py sends ``{}`` as the body, so a route that takes an id in its body fails
validation (422) before any ownership check and looks "rejected" for the wrong reason. Here every body schema in the
OpenAPI document is filled with a *valid-shaped* body that names tenant A's real ids, so a 2xx means a breach.

Synthetic verification: temp SQLite database, signed test tokens, no real data. SQLite only.
"""
from __future__ import annotations

import importlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import Role, issue_scoped_token, rate_limiter
from tests.test_activity_ownership import _insert
from tests.test_tenant_adversarial import PUBLIC_PATH_PARTS, SECRET, _concrete

ID_FIELDS = {
    "world_id": "world_id", "design_world_id": "world_id", "site_world_id": "world_id", "target_world_id": "world_id",
    "session_id": "session_id", "evidence_id": "evidence_id", "evidence_ids": "evidence_id", "upload_id": "upload_id",
    "site_id": "site_id", "job_id": "job_id", "asset_id": "asset_id", "entity_id": "evidence_id",
}


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
    monkeypatch.setattr(rate_limiter, "default_rps", 1_000_000.0)
    monkeypatch.setattr(rate_limiter, "default_burst", 1_000_000.0)
    rate_limiter.reset()

    def bearer(tenant, user, role):
        return {"Authorization": f"Bearer {issue_scoped_token(tenant, user, role, secret=SECRET)}"}

    h = {"a": bearer("tenant-a", "user-a", Role.ADMIN), "b": bearer("tenant-b", "user-b", Role.EDITOR),
         "b-admin": bearer("tenant-b", "admin-b", Role.ADMIN), "sys": bearer("root", "sys", Role.SYSTEM_ADMIN)}
    with TestClient(main_mod.app, raise_server_exceptions=False) as client:
        ids = _seed(client, h["a"], db)
        yield client, h, ids, main_mod.app, db
    rate_limiter.reset()


def _seed(client, ha, db) -> dict:
    wid = client.post("/api/worlds", json={"name": "A world"}, headers=ha).json()["id"]
    sid = client.post("/api/sessions", json={"name": "A session"}, headers=ha).json()["id"]
    up = client.post(f"/api/uploads?session_id={sid}", headers=ha,
                     files={"file": ("a.jpg", b"\xff\xd8\xff\xe0tenant-a-private-bytes", "image/jpeg")})
    assert up.status_code == 201, up.text
    eid = up.json()["evidence_id"]
    con = sqlite3.connect(db)
    _insert(con, "locations", id="loc_a_private", entity_type="evidence", entity_id=eid, latitude=12.3456, longitude=65.4321,
            source="exif", coordinate_reference_system="EPSG:4326")
    _insert(con, "sites", id="site_a_private", team_id="tenant-a", name="A site")
    con.commit()
    con.close()
    return {"world_id": wid, "session_id": sid, "evidence_id": eid, "upload_id": up.json()["upload_id"],
            "site_id": "site_a_private", "job_id": "job_a", "asset_id": "ast_a"}


# ---------------------------------------------------------------- /api/sessions/group (the reported finding)

def _group(client, headers, evidence_ids):
    return client.post("/api/sessions/group", json={"evidence_ids": evidence_ids}, headers=headers)


def test_owner_can_group_their_own_evidence(env):
    client, h, ids, *_ = env
    res = _group(client, h["a"], [ids["evidence_id"]])
    assert res.status_code == 200 and res.json()["total_items"] == 1


@pytest.mark.parametrize("who", ["b", "b-admin"])
def test_foreign_evidence_cannot_be_grouped_and_leaks_nothing(env, who):
    client, h, ids, *_ = env
    foreign = _group(client, h[who], [ids["evidence_id"]])
    unknown = _group(client, h[who], ["ev_does_not_exist"])
    # Same status and same body shape as an unknown id: existence is not disclosed, and no GPS/timestamp is derived.
    assert foreign.status_code == unknown.status_code
    assert foreign.json() == unknown.json()
    assert "12.3456" not in foreign.text and ids["evidence_id"] not in foreign.text


def test_mixed_batch_returns_nothing_partial(env):
    client, h, ids, *_ = env
    mine = client.post("/api/sessions", json={"name": "B session"}, headers=h["b"]).json()["id"]
    up = client.post(f"/api/uploads?session_id={mine}", headers=h["b"],
                     files={"file": ("b.jpg", b"\xff\xd8\xff\xe0tenant-b", "image/jpeg")})
    assert up.status_code == 201, up.text
    res = _group(client, h["b"], [up.json()["evidence_id"], ids["evidence_id"]])
    # A batch naming a foreign id is rejected as a whole, exactly like one naming an unknown id: no partial grouping.
    unknown = _group(client, h["b"], [up.json()["evidence_id"], "ev_does_not_exist"])
    assert res.status_code == unknown.status_code and res.json() == unknown.json()
    assert ids["evidence_id"] not in res.text and "12.3456" not in res.text


def test_system_admin_can_group_any_tenants_evidence(env):
    client, h, ids, *_ = env
    res = _group(client, h["sys"], [ids["evidence_id"]])
    assert res.status_code == 200 and res.json()["total_items"] == 1


def test_unowned_legacy_evidence_is_hidden_from_tenants(env):
    client, h, ids, _app, db = env
    con = sqlite3.connect(db)
    con.execute("UPDATE evidence SET team_id = NULL WHERE id = ?", (ids["evidence_id"],))
    con.commit()
    con.close()
    assert _group(client, h["a"], [ids["evidence_id"]]).status_code == _group(client, h["a"], ["ev_nope"]).status_code
    assert _group(client, h["sys"], [ids["evidence_id"]]).status_code == 200


# ---------------------------------------------------------------- generic: every body/query id field, valid-shaped bodies

def _resolve(schema, spec):
    while "$ref" in schema:
        schema = spec["components"]["schemas"][schema["$ref"].split("/")[-1]]
    return schema


def _filler(schema, spec, ids, name=None, depth=0):
    schema = _resolve(schema, spec)
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return _filler(non_null[0], spec, ids, name, depth) if non_null else None
    kind = schema.get("type")
    key = ID_FIELDS.get(name or "")
    if key:
        return [ids[key]] if kind == "array" else ids[key]
    if "enum" in schema:
        return schema["enum"][0]
    if kind == "object" or "properties" in schema:
        if depth > 3:
            return {}
        required = set(schema.get("required", []))
        return {k: _filler(v, spec, ids, k, depth + 1) for k, v in schema.get("properties", {}).items() if k in required or k in ID_FIELDS}
    if kind == "array":
        return []
    if kind == "string":
        return "x"
    if kind in ("integer", "number"):
        return schema.get("minimum", 1)
    if kind == "boolean":
        return False
    return None


def _body_targets(app, ids):
    spec = app.openapi()
    for path, ops in sorted(spec["paths"].items()):
        if any(p in path for p in PUBLIC_PATH_PARTS):
            continue
        for method, op in sorted(ops.items()):
            if method.upper() not in {"POST", "PUT", "PATCH"}:
                continue
            body_schema = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            if body_schema is None:
                continue
            resolved = _resolve(body_schema, spec)
            if not (set(resolved.get("properties", {})) & set(ID_FIELDS)):
                continue
            yield method.upper(), path, _concrete(path, ids), _filler(body_schema, spec, ids)


ADOPTION_BODY = {"session_id": "", "min_confidence": 0.0}


def test_foreign_world_adoption_is_not_found_not_conflict(env):
    client, h, ids, *_ = env
    body = {**ADOPTION_BODY, "session_id": ids["session_id"]}
    foreign = client.post(f"/api/worlds/{ids['world_id']}/adoptions", json=body, headers=h["b"])
    unknown = client.post("/api/worlds/world_does_not_exist/adoptions", json=body, headers=h["b"])
    assert foreign.status_code == unknown.status_code == 404  # a 409 "no HEAD" would reveal that the world exists


def test_adoption_cannot_name_another_tenants_session(env):
    client, h, ids, *_ = env
    mine = client.post("/api/worlds", json={"name": "B world"}, headers=h["b"]).json()["id"]
    res = client.post(f"/api/worlds/{mine}/adoptions", json={**ADOPTION_BODY, "session_id": ids["session_id"]}, headers=h["b"])
    assert res.status_code == 404 and ids["session_id"] not in res.text


def test_proposal_cannot_cite_another_tenants_session(env):
    client, h, ids, *_ = env
    mine = client.post("/api/worlds", json={"name": "B world"}, headers=h["b"]).json()["id"]
    res = client.post(f"/api/worlds/{mine}/proposals", json={"candidate": {}, "source_session_ids": [ids["session_id"]]},
                      headers=h["b"])
    assert res.status_code == 404 and ids["session_id"] not in res.text


@pytest.mark.parametrize("who", ["b", "b-admin"])
def test_no_body_id_route_succeeds_for_another_tenant(env, who):
    client, h, ids, app, _db = env
    breaches, attempts = [], 0
    for method, template, url, body in _body_targets(app, ids):
        res = client.request(method, url, json=body, headers=h[who])
        attempts += 1
        assert res.status_code != 429, f"{method} {template} was rate-limited; the check proved nothing"
        if 200 <= res.status_code < 300:
            breaches.append(f"{method} {template} body={sorted(body)} -> {res.status_code}")
    assert attempts >= 5, f"only {attempts} body-id routes were discovered; the discovery is broken"
    assert not breaches, "a body-supplied id reached another tenant's record:\n" + "\n".join(breaches)
