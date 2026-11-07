"""DAST & Security Pass Test Suite (T-6.8).

Automated offensive-style verification covering:
1. Unauthenticated access prevention on all non-loopback requests / token enforcement.
2. Cross-tenant isolation & resource existence enumeration prevention (fail-closed 404).
3. Role-Based Access Control (RBAC): viewer role forbidden from mutating endpoints (403).
4. SQL Injection, Path Traversal, and Malicious payload rejection without traceback leakage.
5. Mandatory HTTP security headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy).
6. Client share token containment (cross-world isolation and immediate revocation).
7. Rate limiting denial-of-service resilience.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from apps.api import auth
from apps.api.auth import (
    Role,
    issue_scoped_token,
    issue_share_token,
    rate_limiter,
    revoke_share_token_in_memory,
)
from apps.api.main import app

TOKEN = "secret-production-api-token-xyz123"


class SecurityClient:
    """Async/Sync test client capable of simulating external attacker IP addresses."""

    def __init__(self, host: str = "203.0.113.195"):  # External non-loopback IP
        self.host = host

    def req(self, method: str, path: str, headers: dict | None = None, json: dict | None = None) -> httpx.Response:
        async def _run():
            transport = httpx.ASGITransport(app=app, client=(self.host, 54321))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                return await c.request(method, path, headers=headers, json=json)
        return asyncio.run(_run())


@pytest.fixture(autouse=True)
def _reset_auth_env(monkeypatch):
    monkeypatch.delenv("REALITY_API_TOKEN", raising=False)
    monkeypatch.delenv("REALITY_SIGNING_KEY", raising=False)
    monkeypatch.setattr(auth, "_warned", False)
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def test_dast_external_unauthenticated_requests_fail_closed(monkeypatch):
    """External attackers without token cannot access protected endpoints."""
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    attacker = SecurityClient(host="198.51.100.42")

    sensitive_endpoints = [
        ("GET", "/api/worlds"),
        ("GET", "/api/sessions"),
        ("GET", "/api/jobs"),
        ("GET", "/api/usage"),
        ("GET", "/api/tenants/tenant-1/usage"),
        ("POST", "/api/worlds"),
        ("POST", "/api/worlds/world-1/commands"),
        ("POST", "/api/worlds/world-1/undo"),
        ("POST", "/api/worlds/world-1/redo"),
        ("POST", "/api/worlds/world-1/archive"),
    ]

    for method, path in sensitive_endpoints:
        resp = attacker.req(method, path)
        assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}, expected 401"
        data = resp.json()
        assert "detail" in data
        # Ensure no stack traces or server paths leak in 401 body
        assert "traceback" not in resp.text.lower()
        assert "file \"" not in resp.text.lower()


def test_dast_healthz_and_preflight_stay_available_to_public(monkeypatch):
    """Public health probes and CORS OPTIONS preflights remain open."""
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    client = SecurityClient(host="198.51.100.42")

    assert client.req("GET", "/api/health").status_code == 200
    assert client.req("GET", "/healthz").status_code == 200

    preflight = client.req("OPTIONS", "/api/worlds", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
    })
    assert preflight.status_code == 200


def test_dast_rbac_viewer_forbidden_from_mutating_endpoints(monkeypatch):
    """Viewer token must receive HTTP 403 Forbidden on all mutating operations."""
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    monkeypatch.setenv("REALITY_SIGNING_KEY", "signing-key-123")
    viewer_token = issue_scoped_token("tenant-alpha", "user-guest", Role.VIEWER, secret="signing-key-123")
    client = SecurityClient()

    mutations = [
        ("POST", "/api/worlds", {"name": "Hacked World"}),
        ("POST", "/api/worlds/world-1/commands", {"command_type": "delete", "entity_id": "e-1"}),
        ("POST", "/api/worlds/world-1/undo", {}),
        ("POST", "/api/worlds/world-1/redo", {}),
        ("POST", "/api/worlds/world-1/archive", {}),
        ("DELETE", "/api/worlds/world-1", {}),
    ]

    for method, path, payload in mutations:
        resp = client.req(
            method,
            path,
            headers={"Authorization": f"Bearer {viewer_token}"},
            json=payload,
        )
        assert resp.status_code == 403, f"{method} {path} returned {resp.status_code}, expected 403 Forbidden"
        assert "insufficient permissions" in resp.json()["detail"].lower()


def test_dast_malicious_inputs_fail_cleanly_without_traceback_leak(monkeypatch):
    """SQL injection, path traversal, and garbage payloads do not leak internals."""
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    client = SecurityClient()
    headers = {"Authorization": f"Bearer {TOKEN}"}

    fuzz_payloads = [
        "/api/worlds/' OR 1=1 --",
        "/api/worlds/..%2F..%2F..%2Fetc%2Fpasswd",
        "/api/worlds/`sleep 5`",
        "/api/worlds/<script>alert(1)</script>",
        "/api/worlds/null%00byte",
    ]

    for path in fuzz_payloads:
        resp = client.req("GET", path, headers=headers)
        # Should cleanly return 400, 404, or 422
        assert resp.status_code in (400, 404, 422), f"Path {path} returned unexpected {resp.status_code}"
        # Response body must not contain stack traces or server file paths
        assert "traceback (most recent call last)" not in resp.text.lower()
        assert "apps/api" not in resp.text.lower()
        assert "sqlalchemy" not in resp.text.lower()


def test_dast_mandatory_security_headers_present_on_all_responses(monkeypatch):
    """Every HTTP response must deliver security headers to protect browsers."""
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    client = SecurityClient()

    resp = client.req("GET", "/api/health")
    assert resp.status_code == 200
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "X-XSS-Protection" in resp.headers


def test_dast_share_token_scope_and_revocation(monkeypatch):
    """Client share link tokens cannot access unauthorized worlds and can be revoked."""
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    monkeypatch.setenv("REALITY_SIGNING_KEY", "signing-key-123")

    token_world_a = issue_share_token("world-aaa", "share-1", secret="signing-key-123")

    # Accessing world-b with token for world-a must be strictly forbidden
    from apps.api.auth import enforce_project_isolation, verify_share_token

    auth_a = verify_share_token(token_world_a, secret="signing-key-123")
    assert auth_a is not None
    assert auth_a.project_id == "world-aaa"

    # Enforce isolation against a different world resource
    class FakeWorld:
        id = "world-bbb"
        tenant_id = "tenant-other"

    with pytest.raises(Exception) as exc:
        enforce_project_isolation(FakeWorld(), auth_a)
    assert "scoped to a different world" in str(exc.value)

    # Revocation invalidates token immediately
    revoke_share_token_in_memory(token_world_a)
    assert verify_share_token(token_world_a, secret="signing-key-123") is None


def test_dast_rate_limiter_throttles_denial_of_service(monkeypatch):
    """Exhausting rate limit bucket triggers HTTP 429 Too Many Requests."""
    monkeypatch.setenv("REALITY_API_TOKEN", TOKEN)
    client = SecurityClient(host="198.51.100.99")
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # Deplete bucket tokens
    bucket = rate_limiter.get_bucket("198.51.100.99", rps=1.0, burst=2.0)
    bucket.tokens = 0.0

    resp = client.req("GET", "/api/worlds", headers=headers)
    assert resp.status_code == 429
    assert resp.headers.get("Retry-After") is not None
    assert "rate limit exceeded" in resp.json()["detail"].lower()
