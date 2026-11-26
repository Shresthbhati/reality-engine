"""API-key issuance and verification for Reality Engine backend security.

Programme 5 (Security & Access Control) requires API keys. ``apps.api.auth``
provides RBAC + scoped bearer tokens + rate limiting, but no API-key
mechanism. This is an additive, self-contained service; it is NOT wired
into routing or ``auth_middleware`` (that integration is owned separately),
and it does NOT store or return plaintext secrets.

Security properties
-------------------
* **High-entropy secrets**: ``secrets.token_urlsafe``. The plaintext secret
  is returned to the caller EXACTLY ONCE at creation; only a salted
  PBKDF2-HMAC-SHA256 hash is stored (reusing ``apps.api.auth.hash_password``).
* **Constant-time verification**: ``apps.api.auth.verify_password``.
* **Non-enumerating format**: ``rek_<key_id>.<secret>``; an unknown key_id
  still runs verification against a dummy hash to reduce timing oracles.
* **Revocation + expiry**: revoked or expired keys never verify.
* **Scopes**: optional fine-grained scope list in addition to ``Role``;
  ``has_scope`` checks membership; ``"*"`` grants all.
* **No secret in listings**: ``list_*`` returns records without secrets.
* **Optional durable store**: JSON file; secrets/hashes are persisted
  (hash only), never the plaintext secret.

Composes with ``apps.api.auth.Role`` and ``AuthContext``; verification
returns an ``AuthContext`` a caller can pass to existing authorization
helpers.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from apps.api.auth import AuthContext, Role, hash_password, verify_password

__all__ = ["APIKeyRecord", "CreatedKey", "APIKeyStore", "KEY_PREFIX", "has_scope"]

KEY_PREFIX = "rek_"
_SECRET_BYTES = 32
# A fixed dummy hash used to equalize work when key_id is unknown.
_DUMMY_HASH = hash_password("__unknown_api_key__", salt=b"\x00" * 16)


def has_scope(scopes: list[str] | tuple[str, ...] | frozenset[str], required: str) -> bool:
    """True if ``required`` is granted. ``"*"`` grants everything; empty denies."""
    s = set(scopes or ())
    return "*" in s or required in s


@dataclass(frozen=True)
class APIKeyRecord:
    key_id: str
    hashed_secret: str
    tenant_id: str
    role: Role = Role.VIEWER
    scopes: tuple[str, ...] = ()
    name: str = ""
    created_at: float = 0.0
    expires_at: float | None = None
    revoked: bool = False

    def public(self) -> dict:
        """Serializable view WITHOUT the secret hash (safe to return/list)."""
        return {
            "key_id": self.key_id, "tenant_id": self.tenant_id, "role": self.role.value,
            "scopes": list(self.scopes), "name": self.name, "created_at": self.created_at,
            "expires_at": self.expires_at, "revoked": self.revoked,
        }


@dataclass(frozen=True)
class CreatedKey:
    key_id: str
    secret: str  # full token 'rek_<key_id>.<secret>'; shown once, never stored
    record: APIKeyRecord


def _split(token: str) -> tuple[str, str] | None:
    if not token or not token.startswith(KEY_PREFIX):
        return None
    body = token[len(KEY_PREFIX):]
    if "." not in body:
        return None
    key_id, _, secret = body.partition(".")
    if not key_id or not secret:
        return None
    return key_id, secret


class APIKeyStore:
    """Issues and verifies API keys. Optional JSON persistence (hash only)."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = Path(path) if path else None
        self._records: dict[str, APIKeyRecord] = {}
        if self._path and self._path.exists():
            self._load()

    # -- persistence ----------------------------------------------------
    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        except (OSError, ValueError):
            return
        for obj in data.get("keys", []):
            try:
                self._records[obj["key_id"]] = APIKeyRecord(
                    key_id=obj["key_id"], hashed_secret=obj["hashed_secret"],
                    tenant_id=obj["tenant_id"], role=Role(obj.get("role", "viewer")),
                    scopes=tuple(obj.get("scopes", [])), name=obj.get("name", ""),
                    created_at=obj.get("created_at", 0.0), expires_at=obj.get("expires_at"),
                    revoked=obj.get("revoked", False),
                )
            except (KeyError, ValueError):
                continue

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"keys": [
            {"key_id": r.key_id, "hashed_secret": r.hashed_secret, "tenant_id": r.tenant_id,
             "role": r.role.value, "scopes": list(r.scopes), "name": r.name,
             "created_at": r.created_at, "expires_at": r.expires_at, "revoked": r.revoked}
            for r in self._records.values()
        ]}
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._path)

    # -- issuance -------------------------------------------------------
    def create(
        self,
        tenant_id: str,
        *,
        role: Role | str = Role.VIEWER,
        scopes: list[str] | tuple[str, ...] = (),
        name: str = "",
        ttl_s: float | None = None,
        now: float | None = None,
    ) -> CreatedKey:
        """Issue a new key. Returns plaintext secret ONCE; only hash stored."""
        if isinstance(role, str):
            role = Role(role)
        key_id = secrets.token_hex(8)
        secret = secrets.token_urlsafe(_SECRET_BYTES)
        ts = time.time() if now is None else now
        record = APIKeyRecord(
            key_id=key_id, hashed_secret=hash_password(secret), tenant_id=tenant_id,
            role=role, scopes=tuple(scopes), name=name, created_at=ts,
            expires_at=(ts + ttl_s) if ttl_s is not None else None, revoked=False,
        )
        self._records[key_id] = record
        self._save()
        return CreatedKey(key_id=key_id, secret=f"{KEY_PREFIX}{key_id}.{secret}", record=record)

    # -- verification ---------------------------------------------------
    def verify(self, token: str, *, now: float | None = None) -> AuthContext | None:
        """Verify a presented token. Returns AuthContext or None (never raises)."""
        try:
            parsed = _split(token or "")
            if parsed is None:
                verify_password("x", _DUMMY_HASH)  # equalize work
                return None
            key_id, secret = parsed
            record = self._records.get(key_id)
            if record is None:
                verify_password(secret, _DUMMY_HASH)  # reduce key-id timing oracle
                return None
            if not verify_password(secret, record.hashed_secret):
                return None
            if record.revoked:
                return None
            ts = time.time() if now is None else now
            if record.expires_at is not None and ts >= record.expires_at:
                return None
            return AuthContext(
                user_id=f"apikey:{key_id}", tenant_id=record.tenant_id,
                role=record.role, is_authenticated=True, is_loopback=False,
                project_id=None,
            )
        except Exception:  # noqa: BLE001 -- verification must never raise
            return None

    def has_scope(self, token: str, required: str, *, now: float | None = None) -> bool:
        ctx = self.verify(token, now=now)
        if ctx is None:
            return False
        rec = self._records.get(ctx.user_id.removeprefix("apikey:"))
        return rec is not None and has_scope(list(rec.scopes), required)

    # -- management -----------------------------------------------------
    def revoke(self, key_id: str) -> bool:
        rec = self._records.get(key_id)
        if rec is None or rec.revoked:
            return False
        self._records[key_id] = APIKeyRecord(
            key_id=rec.key_id, hashed_secret=rec.hashed_secret, tenant_id=rec.tenant_id,
            role=rec.role, scopes=rec.scopes, name=rec.name, created_at=rec.created_at,
            expires_at=rec.expires_at, revoked=True,
        )
        self._save()
        return True

    def get(self, key_id: str) -> APIKeyRecord | None:
        return self._records.get(key_id)

    def list_for_tenant(self, tenant_id: str) -> list[APIKeyRecord]:
        return [r for r in self._records.values() if r.tenant_id == tenant_id]

    def __len__(self) -> int:
        return len(self._records)
