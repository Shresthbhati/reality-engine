"""Bearer-token auth on apps/api."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
import asyncio

import httpx

from apps.api import auth
from apps.api.main import app

PROTECTED = "/api/system/verification"          # read-only, no DB
TOKEN = "t0ken-for-tests"


class client:
    """Sync wrapper over an ASGI transport so the peer address can be set (TestClient can't). No app startup runs."""

    def __init__(self, host: str = "127.0.0.1"):
        self.host = host

    def _req(self, method: str, url: str, **kw) -> httpx.Response:
        async def go():
            transport = httpx.ASGITransport(app=app, client=(self.host, 50000))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                return await c.request(method, url, **kw)
        return asyncio.run(go())

    def get(self, url, **kw): return self._req("GET", url, **kw)
    def post(self, url, **kw): return self._req("POST", url, **kw)
    def options(self, url, **kw): return self._req("OPTIONS", url, **kw)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("REALITY_API_TOKEN", raising=False)
    monkeypatch.setattr(auth, "_warned", False)


def test_token_set_requires_correct_bearer(monkeypatch):
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    c = client()
    assert c.get(PROTECTED).status_code == 401
    assert c.get(PROTECTED, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get(PROTECTED, headers={"Authorization": TOKEN}).status_code == 401          # no scheme
    assert c.post("/api/jobs/ingest", json={}).status_code == 401                         # mutating route, no token
    r = c.get(PROTECTED, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_health_and_preflight_stay_open(monkeypatch):
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    c = client("10.0.0.5")
    assert c.get("/api/health").status_code == 200
    r = c.options(PROTECTED, headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    assert r.status_code == 200


def test_unset_token_allows_loopback_with_warning(caplog):
    with caplog.at_level("WARNING", logger="reality.api.auth"):
        assert client("127.0.0.1").get(PROTECTED).status_code == 200
        assert client("::1").get(PROTECTED).status_code == 200
    warnings = [r for r in caplog.records if "REALITY_API_TOKEN is not set" in r.getMessage()]
    assert len(warnings) == 1                                                             # logged once


def test_unset_token_rejects_non_loopback():
    r = client("192.168.1.20").get(PROTECTED)
    assert r.status_code == 401
    assert client("192.168.1.20").get("/api/health").status_code == 200


def test_token_never_logged(monkeypatch, caplog):
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    with caplog.at_level("DEBUG"):
        client().get(PROTECTED, headers={"Authorization": "Bearer wrong-secret"})
        client().get(PROTECTED, headers={"Authorization": f"Bearer {TOKEN}"})
    assert TOKEN not in caplog.text and "wrong-secret" not in caplog.text
