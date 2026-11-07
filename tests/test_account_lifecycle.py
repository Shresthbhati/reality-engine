"""T-6.4: signup, invite, and delete-my-data on top of the existing tenant isolation.

Everything goes through the real API with real signed tokens. The security-relevant facts: accounts are never admin, tokens
expire and die with their account, an invite is single-use, credential endpoints are rate-limited and do not reveal which
emails exist, and erasing a team removes its rows AND its files while leaving a file another team shares.
"""
from __future__ import annotations

import asyncio
import importlib
import time
from datetime import timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("aiosqlite")
from fastapi.testclient import TestClient  # noqa: E402

from apps.api.auth import Role, issue_scoped_token, rate_limiter  # noqa: E402

PW = "correct horse battery"


class Api:
    def __init__(self, client, db_mod, root):
        self.c, self.db_mod, self.root = client, db_mod, root

    def call(self, token, method, path, **kw):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return self.c.request(method, path, headers=headers, **kw)

    def signup(self, email, password=PW):
        r = self.c.post("/api/accounts/signup", json={"email": email, "password": password})
        assert r.status_code == 201, r.text
        return r.json()

    def db(self, fn):
        async def go():
            async with self.db_mod.get_sessionmaker()() as db:
                return await fn(db)

        return asyncio.run(go())

    def blobs(self):
        return {p.name for p in (self.root / "artifacts").rglob("*") if p.is_file()}


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_SIGNING_KEY", "account-lifecycle-test-secret")
    monkeypatch.setenv("REALITY_API_TOKEN", "master-token-for-tests")
    monkeypatch.setenv("REALITY_SIGNUP", "open")
    monkeypatch.setenv("REALITY_EMBEDDED_WORKER", "0")        # jobs stay queued; none is mid-run when a team is erased
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    rate_limiter.reset()
    with TestClient(main_mod.app) as c:
        yield Api(c, db_mod, tmp_path)
    rate_limiter.reset()


def test_signup_is_closed_unless_the_deployment_opens_it(api, monkeypatch):
    monkeypatch.delenv("REALITY_SIGNUP")
    r = api.c.post("/api/accounts/signup", json={"email": "a@x.com", "password": PW})
    assert r.status_code == 403


def test_signup_login_and_me(api):
    s = api.signup("Owner@Example.com")
    assert s["account"]["is_owner"] and s["account"]["role"] == "editor" and s["account"]["email"] == "owner@example.com"
    me = api.call(s["token"], "GET", "/api/accounts/me").json()
    assert me["id"] == s["account"]["id"] and me["tenant_id"] == s["account"]["tenant_id"]
    login = api.c.post("/api/accounts/login", json={"email": "owner@example.com", "password": PW})
    assert login.status_code == 200 and api.call(login.json()["token"], "GET", "/api/accounts/me").status_code == 200
    # a token is never admin: ADMIN bypasses tenant isolation
    assert Role.ADMIN.value not in s["token"].split(":")


def test_bad_input_duplicates_and_no_email_enumeration(api):
    api.signup("dup@x.com")
    assert api.c.post("/api/accounts/signup", json={"email": "dup@x.com", "password": PW}).status_code == 409
    assert api.c.post("/api/accounts/signup", json={"email": "not-an-email", "password": PW}).status_code == 422
    assert api.c.post("/api/accounts/signup", json={"email": "ok@x.com", "password": "short"}).status_code == 422
    wrong = api.c.post("/api/accounts/login", json={"email": "dup@x.com", "password": "wrong password!"})
    unknown = api.c.post("/api/accounts/login", json={"email": "nobody@x.com", "password": "wrong password!"})
    assert wrong.status_code == unknown.status_code == 401 and wrong.json() == unknown.json()


def test_repeated_wrong_logins_are_rate_limited(api):
    api.signup("target@x.com")
    codes = [api.c.post("/api/accounts/login", json={"email": "target@x.com", "password": "guess number one"}).status_code
             for _ in range(8)]
    assert 429 in codes and codes[0] == 401


def test_two_signed_up_teams_cannot_see_each_other(api):
    a, b = api.signup("a@x.com"), api.signup("b@x.com")
    wid = api.call(a["token"], "POST", "/api/worlds", json={"name": "A world"}).json()["id"]
    assert api.call(a["token"], "GET", f"/api/worlds/{wid}").status_code == 200
    assert api.call(b["token"], "GET", f"/api/worlds/{wid}").status_code == 404


def test_invite_flow_is_owner_only_single_use_and_expires(api):
    owner = api.signup("owner@x.com")
    wid = api.call(owner["token"], "POST", "/api/worlds", json={"name": "Team world"}).json()["id"]
    inv = api.call(owner["token"], "POST", "/api/accounts/invites", json={"email": "viewer@x.com", "role": "viewer"})
    assert inv.status_code == 201, inv.text
    token = inv.json()["token"]
    assert api.call(owner["token"], "POST", "/api/accounts/invites", json={"email": "x@x.com", "role": "admin"}).status_code == 422

    joined = api.c.post("/api/accounts/invites/accept", json={"token": token, "password": PW})
    assert joined.status_code == 201, joined.text
    member = joined.json()
    assert member["account"]["tenant_id"] == owner["account"]["tenant_id"] and not member["account"]["is_owner"]
    assert api.call(member["token"], "GET", f"/api/worlds/{wid}").status_code == 200          # sees the team's data
    assert api.call(member["token"], "POST", "/api/worlds", json={"name": "nope"}).status_code == 403   # viewer: read-only
    assert api.call(member["token"], "POST", "/api/accounts/invites", json={"email": "y@x.com"}).status_code == 403   # not the owner

    assert api.c.post("/api/accounts/invites/accept", json={"token": token, "password": PW}).status_code == 410   # single use
    stale = api.call(owner["token"], "POST", "/api/accounts/invites", json={"email": "late@x.com"}).json()["token"]

    async def expire(db):
        from sqlalchemy import update

        from apps.api.models import Invite, utcnow

        await db.execute(update(Invite).where(Invite.email == "late@x.com").values(expires_at=utcnow() - timedelta(days=1)))
        await db.commit()

    api.db(expire)
    assert api.c.post("/api/accounts/invites/accept", json={"token": stale, "password": PW}).status_code == 410
    assert api.c.post("/api/accounts/invites/accept", json={"token": "garbage", "password": PW}).status_code == 410


def test_a_removed_member_loses_access_on_the_next_request_and_anyone_may_delete_their_own_account(api):
    owner = api.signup("owner@x.com")
    token = api.call(owner["token"], "POST", "/api/accounts/invites", json={"email": "m@x.com", "role": "editor"}).json()["token"]
    m = api.c.post("/api/accounts/invites/accept", json={"token": token, "password": PW}).json()
    assert api.call(m["token"], "GET", "/api/accounts/me").status_code == 200
    members = api.call(owner["token"], "GET", "/api/accounts/members").json()["items"]
    assert {x["email"] for x in members} == {"owner@x.com", "m@x.com"}
    assert api.call(owner["token"], "DELETE", f"/api/accounts/members/{owner['account']['id']}").status_code == 409
    assert api.call(owner["token"], "DELETE", f"/api/accounts/members/{m['account']['id']}").status_code == 200
    assert api.call(m["token"], "GET", "/api/accounts/me").status_code == 401                 # token died with the account

    v_tok = api.call(owner["token"], "POST", "/api/accounts/invites", json={"email": "v@x.com", "role": "viewer"}).json()["token"]
    v = api.c.post("/api/accounts/invites/accept", json={"token": v_tok, "password": PW}).json()
    r = api.call(v["token"], "DELETE", "/api/accounts/me", json={"confirm": "delete my data"})
    assert r.status_code == 200 and r.json()["deleted"] == "account"                           # a viewer may leave
    assert api.call(owner["token"], "GET", "/api/accounts/me").status_code == 200              # the team is untouched


def test_scoped_tokens_expire(api, monkeypatch):
    s = api.signup("e@x.com")
    old = issue_scoped_token(s["account"]["tenant_id"], s["account"]["id"], Role.EDITOR, timestamp=int(time.time()) - 40 * 86400)
    assert api.call(old, "GET", "/api/accounts/me").status_code == 401
    monkeypatch.setenv("REALITY_TOKEN_TTL_SECONDS", "0")                                       # operator opts out of expiry
    assert api.call(old, "GET", "/api/accounts/me").status_code == 200


def test_owner_deleting_their_data_removes_rows_and_files_but_not_a_blob_another_team_shares(api):
    a, b = api.signup("a@x.com"), api.signup("b@x.com")

    def populate(who, files):
        wid = api.call(who["token"], "POST", "/api/worlds", json={"name": "W"}).json()["id"]
        sid = api.call(who["token"], "POST", "/api/sessions", json={"name": "S"}).json()["id"]
        for i, content in enumerate(files):
            up = api.call(who["token"], "POST", f"/api/uploads?session_id={sid}", files={"file": (f"f{i}.csv", content, "text/csv")})
            assert up.status_code == 201, up.text
        site = api.call(who["token"], "POST", "/api/sites", json={"name": "Site"}).json()["id"]
        return wid, sid, site

    shared = b"t,v\nshared,1\n"
    a_wid, a_sid, a_site = populate(a, [shared, b"t,v\nonly-a,2\n"])
    b_wid, b_sid, _ = populate(b, [shared])
    before = api.blobs()
    assert len(before) == 2                                   # shared + only-a (content-addressed)

    assert api.call(a["token"], "DELETE", "/api/accounts/me", json={}).status_code == 422             # confirmation required
    assert api.call(a["token"], "GET", f"/api/worlds/{a_wid}").status_code == 200                     # nothing happened yet
    r = api.call(a["token"], "DELETE", "/api/accounts/me", json={"confirm": "Delete my data"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["deleted"] == "team" and out["worlds"] == 1 and out["sessions"] == 1 and out["uploads"] == 2 and out["sites"] == 1

    assert api.call(a["token"], "GET", "/api/accounts/me").status_code == 401                         # the account is gone
    assert len(api.blobs()) == 1, "A's private file is deleted; the file B also holds survives"

    async def left(db):
        from sqlalchemy import func, select

        from apps.api.models import Account, Evidence, Session, Site, Upload, World

        counts = {}
        for name, model in (("worlds", World), ("sessions", Session), ("uploads", Upload), ("evidence", Evidence),
                            ("sites", Site), ("accounts", Account)):
            col = model.tenant_id if model is Account else model.team_id
            counts[name] = {t: (await db.execute(select(func.count()).select_from(model).where(col == t))).scalar_one()
                            for t in (a["account"]["tenant_id"], b["account"]["tenant_id"])}
        return counts

    counts = api.db(left)
    ta, tb = a["account"]["tenant_id"], b["account"]["tenant_id"]
    assert all(c[ta] == 0 for c in counts.values()), counts
    assert counts["worlds"][tb] == counts["sessions"][tb] == counts["uploads"][tb] == counts["accounts"][tb] == 1
    assert api.call(b["token"], "GET", f"/api/worlds/{b_wid}").status_code == 200
