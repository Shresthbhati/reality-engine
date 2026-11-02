"""Bearer-token auth for the application API (local-first design).

* ``REALITY_API_TOKEN`` set  -> every request except ``GET /api/health`` (and CORS preflight) needs
  ``Authorization: Bearer <token>``; compared in constant time. The token is never logged.
* ``REALITY_API_TOKEN`` unset -> unauthenticated access is allowed ONLY for loopback clients (a warning is logged
  once); any non-loopback client gets 401. This fails closed when the server is bound to 0.0.0.0 without a token,
  without needing to know the bind host.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Request
from fastapi.responses import JSONResponse

log = logging.getLogger("reality.api.auth")

OPEN_PATHS = frozenset({"/api/health"})
# "testclient" is Starlette's TestClient peer name; it is not a routable address, so real clients can't spoof it.
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_warned = False


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=401, headers={"WWW-Authenticate": "Bearer"})


async def auth_middleware(request: Request, call_next):
    global _warned
    if request.method == "OPTIONS" or request.url.path in OPEN_PATHS:
        return await call_next(request)
    token = os.environ.get("REALITY_API_TOKEN", "")
    if token:
        scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.strip().encode(), token.encode()):
            return _unauthorized("Missing or invalid bearer token")
    else:
        # No peer address means an in-process or unix-socket transport (e.g. Starlette TestClient): local.
        host = request.client.host if request.client else "127.0.0.1"
        if host not in _LOOPBACK:
            return _unauthorized("REALITY_API_TOKEN is not set; only loopback clients are allowed")
        if not _warned:
            _warned = True
            log.warning("REALITY_API_TOKEN is not set: API is unauthenticated (loopback clients only). "
                        "Set it before exposing the API beyond this machine.")
    return await call_next(request)
