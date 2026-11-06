"""Cross-tenant isolation on every route module that touches tenant-owned data (repair roadmap P0 item 2).

Tenant A creates a world, a session, an upload + its evidence + job, and a site. Tenant B (same role) then tries every
route of every module that reads or writes those objects. Each must answer 404 with the handler's own "<thing> not
found" text, so a foreign id cannot be told apart from an unknown one. The same call by tenant A must NOT get that 404
(positive control: the guard is not simply blocking everyone). An ADMIN keeps access.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")
from fastapi.testclient import TestClient

from apps.api.auth import Role, issue_scoped_token, rate_limiter

SECRET = "tenant-isolation-test-secret"
NOT_FOUND = {"world": "World not found", "session": "Session not found", "evidence": "Evidence not found",
             "upload": "Upload not found", "site": "Site not found", "job": "Job not found"}


class Api:
    def __init__(self, client: TestClient):
        self.c = client
        self.h = {t: {"Authorization": "Bearer " + issue_scoped_token(t, f"user-{t}", role, secret=SECRET)}
                  for t, role in (("tenant-a", Role.EDITOR), ("tenant-b", Role.EDITOR), ("root", Role.ADMIN))}

    def call(self, who: str, method: str, path: str, **kw):
        return self.c.request(method, path, headers=self.h[who], **kw)


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_SIGNING_KEY", SECRET)
    monkeypatch.setenv("REALITY_API_TOKEN", "master-token-for-tests")
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    rate_limiter.reset()
    with TestClient(main_mod.app) as c:
        yield Api(c)
    rate_limiter.reset()


@pytest.fixture()
def ids(api):
    """Everything below belongs to tenant A."""
    world = api.call("tenant-a", "POST", "/api/worlds", json={"name": "A world"}).json()["id"]
    session = api.call("tenant-a", "POST", "/api/sessions", json={"name": "A session"}).json()["id"]
    up = api.call("tenant-a", "POST", f"/api/uploads?session_id={session}",
                  files={"file": ("a.csv", b"t,v\n1,2\n", "text/csv")})
    assert up.status_code == 201, up.text
    pending = api.call("tenant-a", "POST", "/api/uploads/resumable/start",
                       data={"filename": "p.csv", "total_size": "8", "session_id": session}).json()["upload_id"]
    site = api.call("tenant-a", "POST", "/api/sites", json={"name": "A site"}).json()["id"]
    return {"world": world, "session": session, "upload": up.json()["upload_id"], "pending": pending,
            "evidence": up.json()["evidence_id"], "job": up.json()["job_id"], "site": site}


def _world_cases(w):
    base = f"/api/worlds/{w}"
    return [
        ("artifacts", "GET", f"{base}/artifacts/{'0' * 64}", {}),
        ("assistant", "POST", f"{base}/assistant/ask", {"json": {"query": "hi"}}),
        ("assistant", "POST", f"{base}/assistant/call", {"json": {"tool": "x", "args": {}}}),
        ("assistant", "POST", f"{base}/assistant/capture-plan", {}),
        ("assistant", "POST", f"{base}/assistant/confirm", {"json": {"plan_id": "plan_x"}}),
        ("completion", "GET", f"{base}/completion/proposals", {}),
        ("completion", "POST", f"{base}/completion/proposals/cmp_x/accept", {}),
        ("export", "GET", f"{base}/export/formats", {}),
        ("export", "GET", f"{base}/export/package", {}),
        ("export", "POST", f"{base}/export", {"json": {}}),
        ("intent", "GET", f"{base}/intent", {}),
        ("intent", "PUT", f"{base}/intent", {"json": {"scale": "room", "capabilities": []}}),
        ("measurements", "GET", f"{base}/measurements", {}),
        ("measurements", "POST", f"{base}/measurements",
         {"json": {"kind": "distance", "points": [[0, 0, 0], [1, 0, 0]], "scale_state": "authored", "method": "t"}}),
        ("measurements", "DELETE", f"{base}/measurements/msr_x", {}),
        ("procedural", "GET", f"{base}/generated", {}),
        ("procedural", "POST", f"{base}/rooms", {"json": {}}),
        ("proposals", "GET", f"{base}/proposals", {}),
        ("query", "GET", f"{base}/query/nearest?x=0&y=0&z=0", {}),
        ("reconstructions", "GET", f"{base}/status", {}),
        ("reconstructions", "GET", f"{base}/capture-coverage", {}),
    ]


def _other_cases(i):
    s, e, u, j, st = i["session"], i["evidence"], i["upload"], i["job"], i["site"]
    return [
        ("session", "sessions", "GET", f"/api/sessions/{s}", {}),
        ("session", "sessions", "GET", f"/api/sessions/{s}/location", {}),
        ("session", "sessions", "GET", f"/api/sessions/{s}/trajectory", {}),
        ("session", "sessions", "POST", f"/api/sessions/{s}/reconstruct", {}),
        ("session", "uploads", "POST", f"/api/uploads?session_id={s}", {"files": {"file": ("b.csv", b"t,v\n1,2\n", "text/csv")}}),
        ("session", "uploads", "POST", "/api/uploads/resumable/start",
         {"data": {"filename": "b.csv", "total_size": "8", "session_id": s}}),
        ("evidence", "evidence", "GET", f"/api/evidence/{e}", {}),
        ("evidence", "evidence", "GET", f"/api/evidence/{e}/artifact", {}),
        ("evidence", "evidence", "DELETE", f"/api/evidence/{e}", {}),
        ("evidence", "evidence-quality", "GET", f"/api/evidence/{e}/quality", {}),
        ("evidence", "evidence-quality", "PUT", f"/api/evidence/{e}/decision", {"json": {"decision": "rejected"}}),
        ("upload", "uploads", "POST", f"/api/uploads/resumable/{u}/finalize", {}),
        ("upload", "uploads", "POST", "/api/uploads/resumable/chunk",
         {"data": {"upload_id": i["pending"], "offset": "0", "total_size": "8"}, "files": {"file": ("c", b"x", "text/plain")}}),
        ("job", "jobs", "GET", f"/api/jobs/{j}", {}),
        ("job", "jobs", "GET", f"/api/jobs/{j}/events", {}),
        ("job", "jobs", "POST", f"/api/jobs/{j}/cancel", {}),
        ("site", "sites", "GET", f"/api/sites/{st}", {}),
        ("site", "sites", "DELETE", f"/api/sites/{st}", {}),
        ("site", "sites", "GET", f"/api/sites/{st}/anchor", {}),
        ("site", "sites", "PUT", f"/api/sites/{st}/anchor", {"json": {"lat": 1, "lon": 2}}),
        ("site", "sites", "POST", f"/api/sites/{st}/compose", {"json": {"name": "c", "kind": "house"}}),
        ("site", "sites", "POST", f"/api/sites/{st}/placements", {"json": {"world_id": "wld_x", "cell_x": 0, "cell_z": 0}}),
    ]


def _is_hidden(r, kind) -> bool:
    return r.status_code == 404 and r.json().get("detail") == NOT_FOUND[kind]


def test_tenant_b_cannot_reach_tenant_a_worlds_on_any_world_scoped_module(api, ids):
    for module, method, path, kw in _world_cases(ids["world"]):
        b = api.call("tenant-b", method, path, **kw)
        assert _is_hidden(b, "world"), f"{module} {method} {path}: tenant B got {b.status_code} {b.text[:200]}"
        a = api.call("tenant-a", method, path, **kw)
        assert not _is_hidden(a, "world"), f"{module} {method} {path}: owner was blocked ({a.status_code} {a.text[:200]})"
        assert not _is_hidden(api.call("root", method, path, **kw), "world"), f"{module} {path}: admin blocked"


def test_tenant_b_cannot_reach_tenant_a_sessions_evidence_uploads_jobs_sites(api, ids):
    for kind, module, method, path, kw in _other_cases(ids):
        b = api.call("tenant-b", method, path, **kw)
        assert _is_hidden(b, kind), f"{module} {method} {path}: tenant B got {b.status_code} {b.text[:200]}"
    # Owner positive controls (read-only, so the loop above is not order dependent for A's data).
    for kind, module, method, path, kw in _other_cases(ids):
        if method == "GET":
            a = api.call("tenant-a", method, path, **kw)
            assert not _is_hidden(a, kind), f"{module} {path}: owner was blocked ({a.status_code})"


def test_a_write_by_tenant_b_leaves_tenant_a_data_intact(api, ids):
    api.call("tenant-b", "DELETE", f"/api/evidence/{ids['evidence']}")
    api.call("tenant-b", "DELETE", f"/api/sites/{ids['site']}")
    api.call("tenant-b", "POST", f"/api/jobs/{ids['job']}/cancel")
    assert api.call("tenant-a", "GET", f"/api/evidence/{ids['evidence']}").status_code == 200
    assert api.call("tenant-a", "GET", f"/api/sites/{ids['site']}").status_code == 200
    assert api.call("tenant-a", "GET", f"/api/jobs/{ids['job']}").json()["status"] != "cancelled"


def test_lists_only_show_the_callers_own_rows(api, ids):
    b_session = api.call("tenant-b", "POST", "/api/sessions", json={"name": "B session"}).json()["id"]
    b_site = api.call("tenant-b", "POST", "/api/sites", json={"name": "B site"}).json()["id"]
    sessions_b = {s["id"] for s in api.call("tenant-b", "GET", "/api/sessions").json()["items"]}
    assert sessions_b == {b_session}
    assert {s["id"] for s in api.call("tenant-a", "GET", "/api/sessions").json()["items"]} == {ids["session"]}
    assert {s["id"] for s in api.call("tenant-b", "GET", "/api/sites").json()} == {b_site}
    assert {s["id"] for s in api.call("tenant-a", "GET", "/api/sites").json()} == {ids["site"]}
    assert api.call("tenant-b", "GET", "/api/evidence").json()["items"] == []
    assert [e["id"] for e in api.call("tenant-a", "GET", "/api/evidence").json()["items"]] == [ids["evidence"]]
    assert api.call("tenant-b", "GET", "/api/evidence/quality").json()["counts"]["total"] == 0
    assert api.call("tenant-b", "GET", "/api/jobs").json()["items"] == []
    assert ids["job"] in {j["id"] for j in api.call("tenant-a", "GET", "/api/jobs").json()["items"]}
    assert ids["job"] in {j["id"] for j in api.call("root", "GET", "/api/jobs").json()["items"]}
    for feed in ("notifications", "activity"):
        mine = api.call("tenant-a", "GET", f"/api/{feed}").json()["items"]
        theirs = api.call("tenant-b", "GET", f"/api/{feed}").json()["items"]
        assert all(i["entity_id"] != ids["evidence"] for i in theirs), feed
        assert mine or feed == "notifications", feed


def test_tenant_b_cannot_place_or_compose_tenant_a_worlds_on_its_own_site(api, ids):
    b_site = api.call("tenant-b", "POST", "/api/sites", json={"name": "B site"}).json()["id"]
    r = api.call("tenant-b", "POST", f"/api/sites/{b_site}/placements",
                 json={"world_id": ids["world"], "cell_x": 0, "cell_z": 0})
    assert _is_hidden(r, "world"), r.text
    assert api.call("tenant-a", "POST", f"/api/sites/{ids['site']}/placements",
                    json={"world_id": ids["world"], "cell_x": 0, "cell_z": 0}).status_code == 201


def test_reconstruction_into_a_foreign_world_is_refused_before_any_file_is_read(api, ids):
    r = api.call("tenant-b", "POST", "/api/reconstructions", data={"world_id": ids["world"]},
                 files=[("files", ("x.jpg", b"\xff\xd8\xff", "image/jpeg"))])
    assert _is_hidden(r, "world"), r.text


def test_new_rows_are_stamped_with_the_creators_tenant(api, ids):
    import asyncio

    from sqlalchemy import select

    from apps.api.db import get_sessionmaker
    from apps.api.models import Evidence, Session, Site, Upload

    async def teams():
        async with get_sessionmaker()() as db:
            return [(await db.execute(select(m.team_id).where(m.id == ids[k]))).scalar_one()
                    for m, k in ((Session, "session"), (Evidence, "evidence"), (Upload, "upload"), (Site, "site"))]

    assert asyncio.run(teams()) == ["tenant-a"] * 4


def test_pipeline_proposals_are_private_to_the_submitting_tenant(api):
    pid = api.call("tenant-a", "POST", "/proposals", json={"kind": "k", "payload": {"secret": 1}}).json()["proposal_id"]
    assert api.call("tenant-a", "GET", f"/proposals/{pid}").status_code == 200
    assert api.call("tenant-b", "GET", f"/proposals/{pid}").status_code == 404
    assert api.call("tenant-b", "POST", f"/proposals/{pid}/approve", json={}).status_code == 404
    assert api.call("tenant-b", "POST", f"/proposals/{pid}/reject", json={}).status_code == 404
    assert pid not in {p["proposal_id"] for p in api.call("tenant-b", "GET", "/proposals").json()["proposals"]}
    assert pid in {p["proposal_id"] for p in api.call("tenant-a", "GET", "/proposals").json()["proposals"]}
    assert api.call("tenant-a", "GET", f"/proposals/{pid}").json()["status"] == "pending"


def test_geo_import_never_writes_into_another_tenants_world_of_the_same_name(api, monkeypatch):
    import json
    from pathlib import Path

    import httpx

    import apps.api.routes_geo as rg

    fx = json.loads((Path(__file__).parent / "fixtures" / "osm" / "synthetic_street_furniture.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(rg, "_make_client", lambda: httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=fx))))
    body = {"lat": 22.5757, "lon": 88.4312, "radius_m": 300, "name": "Shared name"}
    a = api.call("tenant-a", "POST", "/api/geo/context", json=body)
    b = api.call("tenant-b", "POST", "/api/geo/context", json=body)
    assert a.status_code == b.status_code == 200, (a.text, b.text)
    assert a.json()["world_id"] != b.json()["world_id"]
    assert api.call("tenant-b", "GET", f"/api/worlds/{a.json()['world_id']}").status_code == 404
