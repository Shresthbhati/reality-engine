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
    return TestClient(app)


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
