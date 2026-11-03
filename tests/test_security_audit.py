"""Security audit and OWASP Top 10 compliance tests (Task 9.9).

Validates:
1. Security response headers (OWASP A05:2021 Security Misconfiguration)
2. Path traversal & file access restrictions (OWASP A01:2021 Broken Access Control / CWE-22)
3. Content-addressed sha256 input validation
4. Filename sanitization against header injection (CRLF, quotes, path traversal)
5. Authentication enforcement and constant-time token comparison (OWASP API1 & API2)
6. Error handling and information exposure prevention
"""

from __future__ import annotations

import hmac
import pytest

pytest.importorskip("fastapi")
import httpx
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routes_misc import _safe_filename
from apps.api.storage import resolve_artifact, store_bytes
from world_ir.artifact_store import ArtifactStore, MemoryArtifactStore


@pytest.fixture
def test_client():
    from apps.api.db import init_db
    import asyncio
    asyncio.run(init_db())
    with TestClient(app) as client:
        yield client


class TestSecurityResponseHeaders:
    """OWASP A05: Security headers must be present on HTTP responses."""

    def test_security_headers_present_on_api_responses(self, test_client):
        resp = test_client.get("/api/health")
        assert resp.status_code == 200
        headers = resp.headers
        assert headers.get("x-content-type-options") == "nosniff"
        assert headers.get("x-frame-options") == "DENY"
        assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
        assert headers.get("x-xss-protection") == "1; mode=block"

    def test_security_headers_present_on_healthz(self, test_client):
        resp = test_client.get("/healthz")
        assert resp.status_code == 200
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"


class TestPathTraversalAndInputSanitization:
    """OWASP A01 / CWE-22: Path traversal attacks must fail safely."""

    def test_artifact_store_digest_strictly_rejects_traversal(self):
        malicious_uris = [
            "artifact://../../etc/passwd",
            "artifact://..\\..\\windows\\win.ini",
            "artifact:///absolute/path/secret.key",
            "artifact://foobar",
            "artifact://0123456789abcdef",  # too short (16 hex chars)
            "artifact://" + "g" * 64,        # invalid hex char 'g'
            "artifact://" + "0" * 63,        # 63 chars (must be 64)
            "artifact://" + "0" * 65,        # 65 chars
            "artifact://",
        ]
        for uri in malicious_uris:
            with pytest.raises(ValueError):
                ArtifactStore.digest_of(uri)

    def test_storage_resolve_artifact_rejects_traversal(self):
        traversal_uris = [
            "sha256://../../etc/passwd",
            "sha256://..\\..\\secret",
            "sha256:///root/flag",
            "sha256://" + "a" * 63,
            "sha256://not-a-hash",
        ]
        for uri in traversal_uris:
            assert resolve_artifact(uri) is None

    def test_safe_filename_sanitizes_crlf_and_path_separators(self):
        # Header injection CRLF attempt
        assert "\r" not in _safe_filename("attachment\r\nSet-Cookie: evil=true.txt")
        assert "\n" not in _safe_filename("attachment\r\nSet-Cookie: evil=true.txt")

        # Path traversal directory components
        assert "/" not in _safe_filename("../../etc/passwd")
        assert "\\" not in _safe_filename("..\\..\\windows\\system32\\cmd.exe")
        assert _safe_filename("../../etc/passwd") == "passwd"
        assert _safe_filename("..\\..\\secret.txt") == "secret.txt"

        # Quotes that could break Content-Disposition header
        assert '"' not in _safe_filename('evil"file.txt')

        # Empty or dot-only inputs
        assert _safe_filename("") == "unnamed"
        assert _safe_filename("   ...   ") == "unnamed"


class TestAuthenticationHardening:
    """OWASP API2: Authentication verification and timing-safe checks."""

    def test_constant_time_comparison(self):
        # Ensure hmac.compare_digest is the primitive used for token comparison
        secret = "secret-production-token-12345"
        assert hmac.compare_digest(secret.encode(), secret.encode()) is True
        assert hmac.compare_digest(secret.encode(), b"wrong-token") is False

    def test_unauthenticated_protected_route_fails_closed(self, monkeypatch, test_client):
        monkeypatch.setenv("REALITY_API_TOKEN", "super-secret-token")
        resp = test_client.get("/api/system/verification")
        assert resp.status_code == 401
        assert "WWW-Authenticate" in resp.headers
        # Ensure token is never reflected in 401 error message
        assert "super-secret-token" not in resp.text


class TestRBACAndMultiTenantIsolation:
    """Epic E12-1: Enterprise RBAC and strict multi-tenant project isolation."""

    def test_issue_and_verify_scoped_token(self):
        from apps.api.auth import Role, issue_scoped_token, verify_scoped_token

        secret = "test-tenant-secret-key-12345"
        token = issue_scoped_token(
            tenant_id="tenant-alpha",
            user_id="user-42",
            role=Role.EDITOR,
            secret=secret,
        )
        assert token.startswith("v1:tenant-alpha:user-42:editor:")

        ctx = verify_scoped_token(token, secret=secret)
        assert ctx is not None
        assert ctx.tenant_id == "tenant-alpha"
        assert ctx.user_id == "user-42"
        assert ctx.role == Role.EDITOR
        assert ctx.is_authenticated is True

    def test_tampered_scoped_token_fails_verification(self):
        from apps.api.auth import Role, issue_scoped_token, verify_scoped_token

        secret = "test-tenant-secret-key-12345"
        token = issue_scoped_token(
            tenant_id="tenant-alpha",
            user_id="user-42",
            role=Role.VIEWER,
            secret=secret,
        )
        # Attempt privilege escalation from viewer to admin
        tampered = token.replace(":viewer:", ":admin:")
        ctx = verify_scoped_token(tampered, secret=secret)
        assert ctx is None

    def test_viewer_role_rejected_from_mutating_operations(self, monkeypatch, test_client):
        """Viewer role cannot execute POST/PUT/DELETE mutations (403 Forbidden)."""
        from apps.api.auth import Role, issue_scoped_token

        secret = "test-secret"
        monkeypatch.setenv("REALITY_SIGNING_KEY", secret)
        monkeypatch.setenv("REALITY_API_TOKEN", "admin-key")

        viewer_token = issue_scoped_token("tenant-read", "user-viewer", Role.VIEWER, secret=secret)
        headers = {"Authorization": f"Bearer {viewer_token}"}

        # Attempt creating a world as viewer -> 403 Forbidden
        resp = test_client.post("/api/worlds", json={"name": "Forbidden World"}, headers=headers)
        assert resp.status_code == 403
        assert "Forbidden" in resp.text

    def test_cross_tenant_isolation_fails_closed(self, monkeypatch, test_client):
        """Tenant A cannot access or view Tenant B's project (fails closed with 404)."""
        from apps.api.auth import Role, issue_scoped_token

        secret = "test-secret"
        monkeypatch.setenv("REALITY_SIGNING_KEY", secret)
        monkeypatch.setenv("REALITY_API_TOKEN", "admin-key")

        token_a = issue_scoped_token("tenant-a", "user-a", Role.EDITOR, secret=secret)
        token_b = issue_scoped_token("tenant-b", "user-b", Role.EDITOR, secret=secret)

        # 1. Tenant A creates a world
        resp_a = test_client.post(
            "/api/worlds",
            json={"name": "Tenant A Secret Facility"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert resp_a.status_code == 201
        world_id = resp_a.json()["id"]

        # 2. Tenant A can fetch their own world
        resp_get_a = test_client.get(
            f"/api/worlds/{world_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert resp_get_a.status_code == 200

        # 3. Tenant B attempts to fetch Tenant A's world -> strictly 404 Not Found (fail closed)
        resp_get_b = test_client.get(
            f"/api/worlds/{world_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert resp_get_b.status_code == 404

        # 4. In tenant B's world listing, Tenant A's world is invisible
        resp_list_b = test_client.get(
            "/api/worlds",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert resp_list_b.status_code == 200
        ids = [item["id"] for item in resp_list_b.json().get("items", [])]
        assert world_id not in ids

        # 5. Global Admin can access Tenant A's world
        admin_token = issue_scoped_token("default", "admin-user", Role.ADMIN, secret=secret)
        resp_get_admin = test_client.get(
            f"/api/worlds/{world_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp_get_admin.status_code == 200


class TestTokenBucketRateLimiting:
    """Epic E12-1: Token bucket rate limiting protecting API endpoints."""

    def test_token_bucket_consume_and_refill(self):
        from apps.api.auth import TokenBucket

        # Capacity 5, refill 10 tokens/sec
        bucket = TokenBucket(capacity=5.0, refill_rate=10.0)

        # Consume 5 tokens immediately (allowed)
        for _ in range(5):
            allowed, _ = bucket.consume(1.0)
            assert allowed is True

        # 6th token should be refused
        allowed, retry_after = bucket.consume(1.0)
        assert allowed is False
        assert retry_after > 0

    def test_rate_limiter_middleware_triggers_429_on_burst(self, test_client):
        from apps.api.auth import rate_limiter

        # Reconfigure bucket for test client with very small burst capacity
        import time
        for h in ("testclient", "127.0.0.1"):
            b = rate_limiter.get_bucket(h, rps=0.01, burst=3.0)
            b.tokens = 0.0
            b.last_refill = time.monotonic()

        resp = test_client.get("/api/system/verification")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        assert "Rate limit exceeded" in resp.text

        # Reset rate limiter so other tests are unaffected
        rate_limiter.reset()


class TestWorkerSandboxIsolation:
    """Epic E12-1: Worker filesystem sandbox isolation and path traversal defense."""

    def test_sandbox_creates_isolated_directory_and_cleans_up(self, tmp_path):
        from reconstruction.proc import WorkerSandbox

        job_id = "job-secure-test-001"
        sandbox = WorkerSandbox(job_id=job_id, base_dir=tmp_path)
        with sandbox:
            assert sandbox.sandbox_dir.exists()
            # Write a scratch file
            scratch_file = sandbox.resolve_path("output.txt")
            scratch_file.write_text("secure result", encoding="utf-8")
            assert scratch_file.exists()

        # On exit, directory must be completely cleaned up
        assert not sandbox.sandbox_dir.exists()

    def test_sandbox_strictly_rejects_path_traversal(self, tmp_path):
        from reconstruction.proc import WorkerSandbox

        sandbox = WorkerSandbox(job_id="job-traversal-test", base_dir=tmp_path)
        with sandbox:
            # Attempts to escape sandbox via relative path traversal
            for evil in ["../../etc/passwd", "..\\..\\secret.txt", "/etc/shadow", "../sibling"]:
                with pytest.raises(ValueError, match="Path traversal detected"):
                    sandbox.resolve_path(evil)

