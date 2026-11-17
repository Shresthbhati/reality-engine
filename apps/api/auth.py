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


def is_open_path(path: str) -> bool:
    """Paths accessible without bearer authentication (health probes, public signup/login, invite acceptance)."""
    if path in OPEN_PATHS:
        return True
    if path in ("/api/auth/signup", "/api/auth/login"):
        return True
    if path.startswith("/api/auth/invites/"):
        return True
    if path in ("/api/billing/tiers", "/api/billing/webhook"):
        return True
    if path.startswith("/api/worlds/shared/access/"):
        return True
    return False


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with 100,000 iterations (OWASP recommendation)."""
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"pbkdf2:sha256:100000${salt.hex()}${key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored PBKDF2 hash using constant-time comparison."""
    try:
        parts = stored_hash.split("$")
        if len(parts) != 3:
            return False
        algo_iter, salt_hex, key_hex = parts
        _, _, iters_str = algo_iter.split(":")
        iters = int(iters_str)
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        computed_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(computed_key, expected_key)
    except Exception:
        return False


class Role(str, Enum):
    SYSTEM_ADMIN = "system_admin"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

    @property
    def rank(self) -> int:
        ranks = {Role.VIEWER: 1, Role.EDITOR: 2, Role.ADMIN: 3, Role.SYSTEM_ADMIN: 4}
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
# Scoped Token Signing & Verification (Phase 1 Security Hardening)
# -------------------------------------------------------------------------------------------------

_DEV_FALLBACK_SECRET: str | None = None
DEFAULT_TOKEN_ISSUER = "reality-engine"
DEFAULT_TOKEN_AUDIENCE = "reality-engine-api"
DEFAULT_TOKEN_TTL_S = 86400


def _get_signing_secret() -> str:
    key = os.environ.get("REALITY_SIGNING_KEY") or os.environ.get("REALITY_API_TOKEN")
    if key:
        return key
    env = (os.environ.get("ENVIRONMENT") or os.environ.get("APP_ENV") or "").lower()
    if env in ("production", "prod", "staging"):
        raise RuntimeError("FATAL SEC-002: REALITY_SIGNING_KEY or REALITY_API_TOKEN must be configured in production mode")
    # In development/test mode: generate a secure ephemeral secret per process run
    # instead of a static hardcoded string ("reality-default-dev-secret-key") that anyone can forge tokens with.
    global _DEV_FALLBACK_SECRET
    if _DEV_FALLBACK_SECRET is None:
        import secrets
        _DEV_FALLBACK_SECRET = secrets.token_hex(32)
    return _DEV_FALLBACK_SECRET


def issue_scoped_token(
    tenant_id: str,
    user_id: str,
    role: Role | str,
    secret: str | None = None,
    timestamp: int | None = None,
    expires_in_s: int = DEFAULT_TOKEN_TTL_S,
    issuer: str = DEFAULT_TOKEN_ISSUER,
    audience: str = DEFAULT_TOKEN_AUDIENCE,
) -> str:
    """Issue a cryptographically signed scoped bearer token with expiry, issuer, and audience (Phase 1)."""
    role_str = role.value if isinstance(role, Role) else str(role).lower()
    ts = timestamp if timestamp is not None else int(time.time())
    exp = ts + expires_in_s
    sec = secret or _get_signing_secret()
    payload = f"{tenant_id}:{user_id}:{role_str}:{ts}:{exp}:{issuer}:{audience}"
    sig = hmac.new(sec.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"v2:{payload}:{sig}"


def verify_scoped_token(
    token: str,
    secret: str | None = None,
    expected_issuer: str | None = None,
    expected_audience: str | None = None,
) -> AuthContext | None:
    """Verify and parse a scoped bearer token in constant time.

    Enforces HMAC signature, expiration timestamp, clock skew (<= 60s),
    issuer validation, and audience validation (Phase 1).
    """
    token_str = token.strip()
    parts = token_str.split(":")
    if len(parts) < 6:
        return None
    version = parts[0]
    sec = secret or _get_signing_secret()
    now = int(time.time())
    clock_skew_s = 60

    if version == "v2":
        # v2:{tenant_id}:{user_id}:{role_str}:{ts}:{exp}:{iss}:{aud}:{sig}
        if len(parts) != 9:
            return None
        _, tenant_id, user_id, role_str, ts_str, exp_str, iss, aud, sig = parts
        payload = f"{tenant_id}:{user_id}:{role_str}:{ts_str}:{exp_str}:{iss}:{aud}"
        expected_sig = hmac.new(sec.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig.encode("utf-8"), expected_sig.encode("utf-8")):
            return None
        try:
            ts = int(ts_str)
            exp = int(exp_str)
        except ValueError:
            return None

        # Expiry check: reject expired tokens
        if now > exp + clock_skew_s:
            return None
        # Future token check (clock skew)
        if ts > now + clock_skew_s:
            return None

        # Issuer check
        req_iss = expected_issuer or os.environ.get("REALITY_TOKEN_ISSUER", DEFAULT_TOKEN_ISSUER)
        if iss != req_iss:
            return None

        # Audience check
        req_aud = expected_audience or os.environ.get("REALITY_TOKEN_AUDIENCE", DEFAULT_TOKEN_AUDIENCE)
        if aud != req_aud:
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

    elif version == "v1":
        # v1:{tenant_id}:{user_id}:{role_str}:{ts}:{sig} (backward compatibility)
        if len(parts) != 6:
            return None
        _, tenant_id, user_id, role_str, ts_str, sig = parts
        payload = f"{tenant_id}:{user_id}:{role_str}:{ts_str}"
        expected_sig = hmac.new(sec.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig.encode("utf-8"), expected_sig.encode("utf-8")):
            return None
        try:
            ts = int(ts_str)
        except ValueError:
            return None
        max_age_s = int(os.environ.get("REALITY_TOKEN_MAX_AGE_S", 86400))
        if max_age_s > 0 and (ts < (now - max_age_s) or ts > (now + clock_skew_s)):
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

    return None


# -------------------------------------------------------------------------------------------------
# Read-Only Client Share Link Tokens (T-4b.4)
# -------------------------------------------------------------------------------------------------

_REVOKED_SHARE_TOKENS: set[str] = set()



async def check_share_token_not_revoked(token: str) -> bool:
    token = token.strip()
    if token in _REVOKED_SHARE_TOKENS:
        return False
    try:
        from sqlalchemy import select

        from apps.api.db import get_sessionmaker
        from apps.api.models import WorldShareLink
        maker = get_sessionmaker()
        async with maker() as db:
            res = await db.execute(select(WorldShareLink).where(WorldShareLink.token == token))
            link = res.scalars().first()
            if link is None or link.revoked_at is not None:
                _REVOKED_SHARE_TOKENS.add(token)
                return False
            return True
    except Exception:
        return token not in _REVOKED_SHARE_TOKENS

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
    if request.method == "OPTIONS":
        return await call_next(request)

    # 1. Rate Limiting Check
    client_host = request.client.host if request.client else "127.0.0.1"
    # Key on the network peer only: X-Tenant-Id is client-supplied and unauthenticated at this point,
    # so keying on it would let any caller mint a fresh bucket per request by rotating the header.
    allowed, retry_after = rate_limiter.check(client_host)
    if not allowed:
        return _rate_limited(retry_after)

    # Open paths (health probes, public signup/login, invite acceptance) proceed without bearer token
    if is_open_path(request.url.path):
        return await call_next(request)

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
            if share_ctx is not None and await check_share_token_not_revoked(supplied):
                auth_ctx = share_ctx
            else:
                return _unauthorized("Missing, expired, or invalid share token")
        # Next check global master API token if configured
        elif global_token and hmac.compare_digest(supplied.encode("utf-8"), global_token.encode("utf-8")):
            req_tenant = request.headers.get("x-tenant-id", "default")
            effective_role = Role.SYSTEM_ADMIN if (requested_role == Role.SYSTEM_ADMIN or req_tenant in ("root", "system")) else requested_role
            auth_ctx = AuthContext(
                user_id=request.headers.get("x-user-id", "root"),
                tenant_id=req_tenant,
                role=effective_role,
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
            if share_ctx is not None and await check_share_token_not_revoked(share_param):
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
            req_tenant = request.headers.get("x-tenant-id", "default")
            effective_role = Role.SYSTEM_ADMIN if (requested_role == Role.SYSTEM_ADMIN or req_tenant in ("root", "system")) else requested_role
            auth_ctx = AuthContext(
                user_id=request.headers.get("x-user-id", "local"),
                tenant_id=req_tenant,
                role=effective_role,
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


def is_global_admin(auth: AuthContext) -> bool:
    """Check if the context has cross-tenant / global administrator authority.

    Only SYSTEM_ADMIN (or root/system tenant admins) has cross-tenant authority.
    A tenant administrator on tenant-b or tenant-a has Role.ADMIN and is strictly
    isolated to their own tenant.
    """
    return auth.role == Role.SYSTEM_ADMIN or (auth.role >= Role.ADMIN and auth.tenant_id in ("root", "system"))


def enforce_project_isolation(resource: Any, auth: AuthContext) -> None:
    """Strict project isolation: non-admin users can only access resources belonging to their tenant."""
    if auth.project_id is not None:
        resource_id = getattr(resource, "id", None) or getattr(resource, "world_id", None)
        if resource_id and resource_id != auth.project_id:
            raise HTTPException(status_code=403, detail="Access forbidden: token is scoped to a different world")
    if is_global_admin(auth):
        return
    if auth.tenant_id == "shared" and auth.project_id:
        return
    team_id = getattr(resource, "team_id", None) or getattr(resource, "tenant_id", None)
    if team_id is None:
        # Cross-tenant visibility of records with NULL ownership eliminated (Phase 1):
        # Unowned / legacy records are strictly hidden from non-global-admin tenants.
        raise HTTPException(status_code=404, detail="Resource not found")
    if team_id != auth.tenant_id:
        # Fail closed with 404 to prevent resource existence enumeration across tenants
        raise HTTPException(status_code=404, detail="Resource not found")
