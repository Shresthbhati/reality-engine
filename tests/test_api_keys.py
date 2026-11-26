"""Tests for apps.api.api_keys (Programme 5 API keys)."""

from __future__ import annotations

from apps.api.api_keys import APIKeyStore, has_scope
from apps.api.auth import Role


def test_create_returns_plaintext_once_and_stores_only_hash():
    s = APIKeyStore()
    ck = s.create("tenantA", role=Role.EDITOR, name="ci")
    assert ck.secret.startswith("rek_") and "." in ck.secret
    rec = s.get(ck.key_id)
    assert rec is not None
    assert ck.secret.split(".", 1)[1] not in rec.hashed_secret  # plaintext not stored
    assert rec.hashed_secret.startswith("pbkdf2:sha256:")
    assert rec.tenant_id == "tenantA" and rec.role is Role.EDITOR and not rec.revoked
    # public listing has no secret/hash
    assert "hashed_secret" not in rec.public()


def test_verify_valid_token_returns_authcontext():
    s = APIKeyStore()
    ck = s.create("tenantB", role=Role.ADMIN)
    ctx = s.verify(ck.secret)
    assert ctx is not None and ctx.is_authenticated
    assert ctx.tenant_id == "tenantB" and ctx.role is Role.ADMIN
    assert ctx.user_id == f"apikey:{ck.key_id}"


def test_verify_rejects_bad_malformed_unknown_and_wrong_secret():
    s = APIKeyStore()
    ck = s.create("t")
    assert s.verify("") is None
    assert s.verify("not-a-key") is None
    assert s.verify("rek_unknown.abc") is None
    assert s.verify(ck.secret + "x") is None            # tampered secret
    assert s.verify(ck.key_id) is None                   # missing prefix/parts
    key_id, _, secret = ck.secret[len("rek_"):].partition(".")
    assert s.verify(f"rek_{key_id}.wrongsecret") is None


def test_revocation_blocks_verification():
    s = APIKeyStore()
    ck = s.create("t")
    assert s.verify(ck.secret) is not None
    assert s.revoke(ck.key_id) is True
    assert s.verify(ck.secret) is None
    assert s.revoke(ck.key_id) is False  # already revoked


def test_expiry_blocks_verification():
    s = APIKeyStore()
    ck = s.create("t", ttl_s=100, now=1000.0)
    assert s.verify(ck.secret, now=1050.0) is not None   # before expiry
    assert s.verify(ck.secret, now=1100.0) is None       # at/after expiry
    assert s.verify(ck.secret, now=2000.0) is None


def test_scopes():
    s = APIKeyStore()
    ck = s.create("t", scopes=["worlds:read", "jobs:write"])
    assert s.has_scope(ck.secret, "worlds:read") is True
    assert s.has_scope(ck.secret, "jobs:write") is True
    assert s.has_scope(ck.secret, "admin") is False
    assert s.has_scope(ck.secret + "x", "worlds:read") is False  # invalid token
    assert has_scope(["*"], "anything") is True
    assert has_scope([], "anything") is False


def test_list_for_tenant_isolation_and_no_secrets():
    s = APIKeyStore()
    a = s.create("t1"); s.create("t2"); s.create("t1")
    got = s.list_for_tenant("t1")
    assert len(got) == 2 and all(r.tenant_id == "t1" for r in got)
    assert all("hashed_secret" not in r.public() for r in got)
    assert s.list_for_tenant("nope") == []
    assert len(s) == 3


def test_durable_store_persists_hash_not_plaintext(tmp_path):
    p = tmp_path / "keys.json"
    s1 = APIKeyStore(p)
    ck = s1.create("t", scopes=["x"])
    raw = p.read_text(encoding="utf-8")
    assert ck.secret.split(".", 1)[1] not in raw      # plaintext secret never written
    assert "hashed_secret" in raw                       # only the hash is
    s2 = APIKeyStore(p)                                 # restart over same file
    assert s2.verify(ck.secret) is not None
    assert len(s2) == 1


def test_verify_never_raises():
    s = APIKeyStore()
    for bad in [None, "", "rek_", "rek_a.", ".x", "rek_a.b.c", 12345]:  # type: ignore[list-item]
        assert s.verify(bad) is None  # type: ignore[arg-type]
