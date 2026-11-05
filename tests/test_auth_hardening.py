"""Auth hardening: rate limiting must not be keyed on client-supplied headers, bad roles fail closed,
and the bucket table is bounded."""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.auth import TokenBucketRateLimiter, auth_middleware, rate_limiter


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("REALITY_API_TOKEN", raising=False)
    app = FastAPI()
    app.middleware("http")(auth_middleware)

    @app.get("/x")
    def x():
        return {"ok": True}

    rate_limiter.reset()
    yield TestClient(app)
    rate_limiter.reset()


def _exhaust(host: str) -> None:
    bucket = rate_limiter.get_bucket(host, rps=0.01, burst=1.0)
    bucket.tokens = 0.0
    bucket.last_refill = time.monotonic()


def test_rotating_tenant_header_does_not_evade_rate_limit(client):
    _exhaust("testclient")
    for tenant in ("a", "b", "c", "d"):
        resp = client.get("/x", headers={"X-Tenant-Id": tenant})
        assert resp.status_code == 429, f"tenant {tenant!r} got a fresh bucket"


def test_invalid_role_header_is_rejected_not_promoted_to_admin(client):
    resp = client.get("/x", headers={"X-Role": "superuser"})
    assert resp.status_code == 400


def test_valid_role_header_still_works(client):
    assert client.get("/x", headers={"X-Role": "viewer"}).status_code == 200


def test_bucket_table_is_bounded():
    limiter = TokenBucketRateLimiter()
    limiter.MAX_BUCKETS = 5
    for i in range(50):
        limiter.check(f"client-{i}")
    assert len(limiter._buckets) <= 5
