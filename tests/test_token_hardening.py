"""Phase 1 item 11: scoped-token expiry, issuer, audience, legacy-format policy, delimiter safety, share-link revocation.

Synthetic verification: tokens are signed with a test key; no real credentials.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import time

import pytest

from apps.api import auth
from apps.api.auth import Role, issue_scoped_token, verify_scoped_token

KEY = "token-hardening-test-key"


def _v1(tenant: str = "t1", user: str = "u1", role: str = "editor", ts: int | None = None) -> str:
    ts = int(time.time()) if ts is None else ts
    payload = f"{tenant}:{user}:{role}:{ts}"
    return f"v1:{payload}:{hmac.new(KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]}"


def test_valid_v2_token_round_trips():
    ctx = verify_scoped_token(issue_scoped_token("t1", "u1", Role.EDITOR, secret=KEY), secret=KEY)
    assert ctx is not None and (ctx.tenant_id, ctx.user_id, ctx.role) == ("t1", "u1", Role.EDITOR)


def test_expired_token_is_rejected_beyond_the_clock_skew():
    old = int(time.time()) - 7200
    token = issue_scoped_token("t1", "u1", Role.EDITOR, secret=KEY, timestamp=old, expires_in_s=3600)
    assert verify_scoped_token(token, secret=KEY) is None


def test_token_from_the_future_is_rejected():
    token = issue_scoped_token("t1", "u1", Role.EDITOR, secret=KEY, timestamp=int(time.time()) + 3600)
    assert verify_scoped_token(token, secret=KEY) is None


def test_wrong_issuer_and_wrong_audience_are_rejected():
    assert verify_scoped_token(issue_scoped_token("t1", "u1", Role.EDITOR, secret=KEY, issuer="someone-else"), secret=KEY) is None
    assert verify_scoped_token(issue_scoped_token("t1", "u1", Role.EDITOR, secret=KEY, audience="other-api"), secret=KEY) is None


def test_token_signed_with_another_key_is_rejected():
    assert verify_scoped_token(issue_scoped_token("t1", "u1", Role.EDITOR, secret="not-the-key"), secret=KEY) is None


def test_changing_the_tenant_in_a_valid_token_invalidates_it():
    token = issue_scoped_token("tenant-a", "u1", Role.EDITOR, secret=KEY)
    assert verify_scoped_token(token.replace("tenant-a", "tenant-b", 1), secret=KEY) is None


@pytest.mark.parametrize("field", ["tenant_id", "user_id", "issuer", "audience"])
def test_issuing_refuses_the_field_delimiter(field):
    args = {"tenant_id": "t1", "user_id": "u1", "issuer": "iss", "audience": "aud"}
    args[field] = "a:b"
    with pytest.raises(ValueError):
        issue_scoped_token(args.pop("tenant_id"), args.pop("user_id"), Role.EDITOR, secret=KEY, **args)


def test_legacy_v1_tokens_are_accepted_in_development_and_refused_in_production(monkeypatch):
    for var in ("ENVIRONMENT", "APP_ENV", "REALITY_ACCEPT_V1_TOKENS"):
        monkeypatch.delenv(var, raising=False)
    assert verify_scoped_token(_v1(), secret=KEY) is not None

    monkeypatch.setenv("ENVIRONMENT", "production")
    assert verify_scoped_token(_v1(), secret=KEY) is None

    monkeypatch.setenv("REALITY_ACCEPT_V1_TOKENS", "1")  # an explicit, deliberate opt-in
    assert verify_scoped_token(_v1(), secret=KEY) is not None


def test_share_link_revocation_check_fails_closed_when_the_database_is_unreachable(monkeypatch):
    import apps.api.db as db_mod

    def _boom():
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(db_mod, "get_sessionmaker", _boom)
    assert asyncio.run(auth.check_share_token_not_revoked("some-share-token-not-seen-before")) is False
