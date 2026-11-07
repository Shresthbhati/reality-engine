"""Authentication, Multi-Tenant Project Isolation, and Rate Limiting for Reality Engine.

Features:
* Role-Based Access Control (RBAC): Admin, Editor, Viewer hierarchy.
* Scoped Bearer Tokens: Cryptographically signed tenant/user/role tokens.
* Multi-Tenant Project Isolation: Strict tenant boundary enforcement preventing cross-tenant data leakage.
* Token Bucket Rate Limiting: High-performance sliding token bucket preventing API abuse and denial of service.
* Loopback Fallback: Safe local-first development without credentials, failing closed when exposed.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("reality.api.auth")

OPEN_PATHS = frozenset({"/api/health", "/healthz"})
# "testclient" is Starlette's TestClient peer name; it is not a routable address, so real clients can't spoof it.
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_warned = False


class Role(str, Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

    @property
    def rank(self) -> int:
        ranks = {Role.VIEWER: 1, Role.EDITOR: 2, Role.ADMIN: 3}
        return ranks.get(self, 0)

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, Role):
            return NotImplemented
        return self.rank >= other.rank

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Role):
            return NotImplemented
        return self.rank > other.rank

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, Role):
            return NotImplemented
        return self.rank <= other.rank

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Role):
            return NotImplemented
        return self.rank < other.rank


@dataclass
class AuthContext:
    user_id: str
    tenant_id: str
    role: Role
    is_authenticated: bool = True
    is_loopback: bool = False
    project_id: str | None = None


# -------------------------------------------------------------------------------------------------
# Scoped Token Signing & Verification
# -------------------------------------------------------------------------------------------------

def _get_signing_secret() -> str:
    return os.environ.get("REALITY_SIGNING_KEY") or os.environ.get("REALITY_API_TOKEN") or "reality-default-dev-secret-key"


def issue_scoped_token(
    tenant_id: str,
    user_id: str,
    role: Role | str,
    secret: str | None = None,
    timestamp: int | None = None,
) -> str:
    """Issue a cryptographically signed scoped bearer token."""
    role_str = role.value if isinstance(role, Role) else str(role).lower()
    ts = timestamp if timestamp is not None else int(time.time())
    sec = secret or _get_signing_secret()
    payload = f"{tenant_id}:{user_id}:{role_str}:{ts}"
    sig = hmac.new(sec.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"v1:{payload}:{sig}"


def verify_scoped_token(token: str, secret: str | None = None) -> AuthContext | None:
    """Verify and parse a scoped bearer token in constant time."""
    parts = token.strip().split(":")
    if len(parts) != 6 or parts[0] != "v1":
        return None
    _, tenant_id, user_id, role_str, ts_str, sig = parts
    sec = secret or _get_signing_secret()
    payload = f"{tenant_id}:{user_id}:{role_str}:{ts_str}"
    expected_sig = hmac.new(sec.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig.encode("utf-8"), expected_sig.encode("utf-8")):
        return None
    try:
        role = Role(role_str)
    except ValueError:
        return None
    return AuthContext(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        is_authenticated=True,
    )


# -------------------------------------------------------------------------------------------------
# Read-Only Client Share Link Tokens (T-4b.4)
# -------------------------------------------------------------------------------------------------

_REVOKED_SHARE_TOKENS: set[str] = set()


def revoke_share_token_in_memory(token: str) -> None:
    """Mark a share token revoked in memory."""
    _REVOKED_SHARE_TOKENS.add(token.strip())


def issue_share_token(
    world_id: str,
    share_id: str,
    expires_at: int | None = None,
    secret: str | None = None,
) -> str:
    """Issue a token-scoped, read-only share token for a specific world (T-4b.4)."""
    sec = secret or _get_signing_secret()
    exp_str = str(int(expires_at)) if expires_at is not None else "0"
    payload = f"{world_id}:{share_id}:{exp_str}"
    sig = hmac.new(sec.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"sh1:{payload}:{sig}"


def verify_share_token(token: str, secret: str | None = None) -> AuthContext | None:
    """Verify a read-only client share token in constant time. Returns scoped viewer context or None."""
    token = token.strip()
    if token in _REVOKED_SHARE_TOKENS:
        return None
    parts = token.split(":")
    if len(parts) != 5 or parts[0] != "sh1":
        return None
    _, world_id, share_id, exp_str, sig = parts
    sec = secret or _get_signing_secret()
    payload = f"{world_id}:{share_id}:{exp_str}"
    expected_sig = hmac.new(sec.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig.encode("utf-8"), expected_sig.encode("utf-8")):
        return None
    try:
        exp_ts = int(exp_str)
        if exp_ts > 0 and time.time() > exp_ts:
            return None  # expired
    except ValueError:
        return None
    return AuthContext(
        user_id=f"client:{share_id}",
        tenant_id="shared",
        role=Role.VIEWER,
        is_authenticated=True,
        project_id=world_id,
    )


# -------------------------------------------------------------------------------------------------
# Token Bucket Rate Limiting
# -------------------------------------------------------------------------------------------------

class TokenBucket:
    """Thread-safe token bucket algorithm."""

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> tuple[bool, float]:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.last_refill = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            if self.tokens >= amount:
                self.tokens -= amount
                return True, 0.0
            else:
                missing = amount - self.tokens
                retry_after = missing / self.refill_rate if self.refill_rate > 0 else 1.0
                return False, retry_after


class TokenBucketRateLimiter:
    """Sliding rate limiter keyed by client host or tenant ID."""

    def __init__(self, default_rps: float = 100.0, default_burst: float = 150.0):
        self.default_rps = default_rps
        self.default_burst = default_burst
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    #: Hard cap on tracked keys so a flood of distinct keys cannot grow memory without bound.
    MAX_BUCKETS = 10_000

    def get_bucket(self, key: str, rps: float | None = None, burst: float | None = None) -> TokenBucket:
        with self._lock:
            if key not in self._buckets:
                if len(self._buckets) >= self.MAX_BUCKETS:
                    # ponytail: evict oldest-inserted; an LRU only matters if legit clients exceed the cap
                    self._buckets.pop(next(iter(self._buckets)))
                self._buckets[key] = TokenBucket(
                    capacity=burst if burst is not None else self.default_burst,
                    refill_rate=rps if rps is not None else self.default_rps,
                )
            return self._buckets[key]

    def check(self, key: str, cost: float = 1.0, rps: float | None = None, burst: float | None = None) -> tuple[bool, float]:
        bucket = self.get_bucket(key, rps=rps, burst=burst)
        return bucket.consume(cost)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


# Global API rate limiter instance
_DEFAULT_RPS = float(os.environ.get("REALITY_RATE_LIMIT_RPS", "100.0"))
_DEFAULT_BURST = float(os.environ.get("REALITY_RATE_LIMIT_BURST", "150.0"))
rate_limiter = TokenBucketRateLimiter(default_rps=_DEFAULT_RPS, default_burst=_DEFAULT_BURST)


# -------------------------------------------------------------------------------------------------
# Middleware & Helpers
# -------------------------------------------------------------------------------------------------

def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=401, headers={"WWW-Authenticate": "Bearer"})


def _forbidden(detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=403)


def _rate_limited(retry_after: float) -> JSONResponse:
    return JSONResponse(
        {"detail": "Rate limit exceeded. Too many requests."},
        status_code=429,
        headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
    )


async def auth_middleware(request: Request, call_next):
    global _warned
    if request.method == "OPTIONS" or request.url.path in OPEN_PATHS:
        return await call_next(request)

    # 1. Rate Limiting Check
    client_host = request.client.host if request.client else "127.0.0.1"
    # Key on the network peer only: X-Tenant-Id is client-supplied and unauthenticated at this point,
    # so keying on it would let any caller mint a fresh bucket per request by rotating the header.
    allowed, retry_after = rate_limiter.check(client_host)
    if not allowed:
        return _rate_limited(retry_after)

    role_header = request.headers.get("x-role", "admin").lower()
    try:
        requested_role = Role(role_header)
    except ValueError:
        # Fail closed: an unrecognised role must never silently become ADMIN.
        return JSONResponse({"detail": f"Invalid X-Role {role_header!r}; expected admin, editor or viewer"}, status_code=400)

    # 2. Token & Authentication Parsing
    auth_header = request.headers.get("authorization", "")
    scheme, _, supplied = auth_header.partition(" ")
    supplied = supplied.strip()

    global_token = os.environ.get("REALITY_API_TOKEN", "")
    auth_ctx: AuthContext | None = None

    if supplied:
        if scheme.lower() != "bearer":
            return _unauthorized("Invalid authorization scheme; Bearer required")

        # First check client share token (T-4b.4)
        if supplied.startswith("sh1:"):
            share_ctx = verify_share_token(supplied)
            if share_ctx is not None:
                auth_ctx = share_ctx
            else:
                return _unauthorized("Missing, expired, or invalid share token")
        # Next check global master API token if configured
        elif global_token and hmac.compare_digest(supplied.encode("utf-8"), global_token.encode("utf-8")):
            auth_ctx = AuthContext(
                user_id=request.headers.get("x-user-id", "root"),
                tenant_id=request.headers.get("x-tenant-id", "default"),
                role=requested_role,
                is_authenticated=True,
            )
        else:
            # Check cryptographic scoped token
            scoped_ctx = verify_scoped_token(supplied)
            if scoped_ctx is not None:
                auth_ctx = scoped_ctx
            elif global_token:
                return _unauthorized("Missing or invalid bearer token")
    else:
        # Check query parameter share token for browser direct links
        share_param = request.query_params.get("share_token")
        if share_param and share_param.startswith("sh1:"):
            share_ctx = verify_share_token(share_param)
            if share_ctx is not None:
                auth_ctx = share_ctx
            else:
                return _unauthorized("Missing, expired, or invalid share token")
        elif global_token:
            return _unauthorized("Missing or invalid bearer token")
        else:
            # No REALITY_API_TOKEN set: allow loopback clients only
            if client_host not in _LOOPBACK:
                return _unauthorized("REALITY_API_TOKEN is not set; only loopback clients are allowed")
            if not _warned:
                _warned = True
                log.warning("REALITY_API_TOKEN is not set: API is unauthenticated (loopback clients only). "
                            "Set it before exposing the API beyond this machine.")
            auth_ctx = AuthContext(
                user_id=request.headers.get("x-user-id", "local"),
                tenant_id=request.headers.get("x-tenant-id", "default"),
                role=requested_role,
                is_authenticated=True,
                is_loopback=True,
            )

    if auth_ctx is None:
        return _unauthorized("Missing or invalid bearer token")

    # 3. RBAC Check on Mutating Operations
    # Viewer role is strictly read-only: mutating methods (POST, PUT, PATCH, DELETE) are forbidden
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        if auth_ctx.role == Role.VIEWER:
            return _forbidden("Forbidden: insufficient permissions for mutating operations")

    request.state.auth = auth_ctx
    return await call_next(request)


def get_auth_context(request: Request) -> AuthContext:
    """Dependency to inject the authenticated AuthContext into endpoint handlers."""
    auth = getattr(request.state, "auth", None)
    if auth is not None:
        return auth
    return AuthContext(
        user_id="anonymous",
        tenant_id="default",
        role=Role.VIEWER,
        is_authenticated=False,
    )


def enforce_project_isolation(resource: Any, auth: AuthContext) -> None:
    """Strict project isolation: non-admin users can only access resources belonging to their tenant."""
    if auth.project_id is not None:
        resource_id = getattr(resource, "id", None) or getattr(resource, "world_id", None)
        if resource_id and resource_id != auth.project_id:
            raise HTTPException(status_code=403, detail="Access forbidden: token is scoped to a different world")
    if auth.role == Role.ADMIN:
        return
    if auth.tenant_id == "shared" and auth.project_id:
        return
    team_id = getattr(resource, "team_id", None) or getattr(resource, "tenant_id", None)
    if team_id and team_id != auth.tenant_id:
        # Fail closed with 404 to prevent resource existence enumeration across tenants
        raise HTTPException(status_code=404, detail="Resource not found")
