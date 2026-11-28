"""Feature-flag service for Reality Engine backend infrastructure.

Programme 1 (Backend Architecture) requires feature flags. No feature-flag
system existed in this repository; this is an additive, self-contained
service with NO changes to routing, middleware, or other agents' modules.

Design goals
------------
* **Fail-safe**: an unknown flag, a malformed source, or an evaluation
  error returns the caller-supplied ``default`` (else ``False``). A flag
  is never able to crash a request path.
* **Deterministic rollout**: percentage rollout buckets a stable key
  (tenant / project / explicit key) via SHA-256, so the same subject is
  consistently in or out across processes and restarts -- no flapping.
* **Explicit overrides**: per-tenant allow/deny lists win over rollout.
* **Auditable**: flags are plain data; ``list_flags`` returns exactly
  what is configured. No hidden global mutation of request handlers.

Sources (lowest precedence first; later sources override earlier):
  1. explicit mapping passed to the constructor,
  2. a JSON file path (``REALITY_FEATURE_FLAGS_FILE`` or ``path=``),
  3. environment variable ``REALITY_FEATURE_FLAGS_JSON`` (a JSON object),
  4. ``set()`` calls at runtime.

This module has no third-party dependencies and performs no I/O at
import time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

__all__ = ["FlagState", "Flag", "FeatureFlagStore", "ENV_FLAGS_JSON", "ENV_FLAGS_FILE"]

ENV_FLAGS_JSON = "REALITY_FEATURE_FLAGS_JSON"
ENV_FLAGS_FILE = "REALITY_FEATURE_FLAGS_FILE"

_FROZENSET_EMPTY: frozenset[str] = frozenset()


class FlagState(str, Enum):
    ON = "on"
    OFF = "off"
    CONDITIONAL = "conditional"  # decided by rollout_percent + overrides


def _as_frozenset(value: Any) -> frozenset[str]:
    if value is None:
        return _FROZENSET_EMPTY
    if isinstance(value, str):
        return frozenset({value})
    try:
        return frozenset(str(v) for v in value)
    except TypeError:
        return _FROZENSET_EMPTY


def _coerce_state(value: Any) -> FlagState:
    if isinstance(value, FlagState):
        return value
    if isinstance(value, bool):
        return FlagState.ON if value else FlagState.OFF
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("on", "true", "1", "enabled", "enable"):
            return FlagState.ON
        if v in ("off", "false", "0", "disabled", "disable"):
            return FlagState.OFF
        if v in ("conditional", "rollout", "partial"):
            return FlagState.CONDITIONAL
    return FlagState.OFF


def _clamp_percent(value: Any) -> int:
    try:
        pct = int(value)
    except (TypeError, ValueError):
        return 100
    return max(0, min(100, pct))


@dataclass(frozen=True)
class Flag:
    """A single feature flag. Immutable; produced by ``FeatureFlagStore``."""

    name: str
    state: FlagState = FlagState.OFF
    description: str = ""
    rollout_percent: int = 100
    tenants_on: frozenset[str] = _FROZENSET_EMPTY
    tenants_off: frozenset[str] = _FROZENSET_EMPTY
    default: bool = False

    def _bucket(self, key: str) -> int:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest, 16) % 100

    def evaluate(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        key: str | None = None,
    ) -> bool:
        """Resolve this flag for a subject. Never raises."""
        try:
            # Explicit per-tenant overrides have highest precedence.
            if tenant_id is not None:
                if tenant_id in self.tenants_off:
                    return False
                if tenant_id in self.tenants_on:
                    return True
            if self.state is FlagState.ON:
                return True
            if self.state is FlagState.OFF:
                return False
            # CONDITIONAL: deterministic percentage rollout over a stable key.
            if self.rollout_percent <= 0:
                return False
            if self.rollout_percent >= 100:
                return True
            bucket_key = key or tenant_id or project_id or self.name
            return self._bucket(str(bucket_key)) < self.rollout_percent
        except Exception:  # noqa: BLE001 -- flag evaluation must never break a caller
            return self.default


def _flag_from_raw(name: str, raw: Any) -> Flag:
    """Build a Flag from loosely-typed config. Never raises."""
    try:
        if isinstance(raw, bool):
            return Flag(name=name, state=FlagState.ON if raw else FlagState.OFF)
        if isinstance(raw, Flag):
            return raw if raw.name == name else Flag(
                name=name, state=raw.state, description=raw.description,
                rollout_percent=raw.rollout_percent, tenants_on=raw.tenants_on,
                tenants_off=raw.tenants_off, default=raw.default,
            )
        if isinstance(raw, Mapping):
            return Flag(
                name=name,
                state=_coerce_state(raw.get("state", raw.get("enabled", FlagState.OFF))),
                description=str(raw.get("description", "") or ""),
                rollout_percent=_clamp_percent(raw.get("rollout_percent", raw.get("percent", 100))),
                tenants_on=_as_frozenset(raw.get("tenants_on", raw.get("allow", raw.get("allowlist")))),
                tenants_off=_as_frozenset(raw.get("tenants_off", raw.get("deny", raw.get("denylist")))),
                default=bool(raw.get("default", False)),
            )
        return Flag(name=name, state=_coerce_state(raw))
    except Exception:  # noqa: BLE001
        return Flag(name=name, state=FlagState.OFF)


class FeatureFlagStore:
    """Holds feature-flag configuration and evaluates flags. Thread-safe enough
    for read-heavy use; ``set`` mutates a private dict copy."""

    def __init__(
        self,
        source: Mapping[str, Any] | None = None,
        *,
        path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._flags: dict[str, Flag] = {}
        # 1. explicit mapping
        if source:
            self._ingest(source)
        # 2. JSON file
        file_path = path or os.environ.get(ENV_FLAGS_FILE)
        if file_path:
            try:
                data = json.loads(Path(file_path).read_text(encoding="utf-8"))
                if isinstance(data, Mapping):
                    self._ingest(data.get("flags", data) if isinstance(data, Mapping) else {})
            except (OSError, ValueError):
                pass  # fail-safe: ignore unreadable/malformed file
        # 3. env JSON
        env_json = os.environ.get(ENV_FLAGS_JSON)
        if env_json:
            try:
                data = json.loads(env_json)
                if isinstance(data, Mapping):
                    self._ingest(data.get("flags", data))
            except ValueError:
                pass

    def _ingest(self, mapping: Mapping[str, Any]) -> None:
        for name, raw in mapping.items():
            if name in ("flags",):
                continue
            self._flags[str(name)] = _flag_from_raw(str(name), raw)

    # -- mutation -------------------------------------------------------
    def set(
        self,
        name: str,
        *,
        state: FlagState | str | bool | None = None,
        enabled: bool | None = None,
        rollout_percent: int | None = None,
        tenants_on: Any = None,
        tenants_off: Any = None,
        description: str | None = None,
        default: bool | None = None,
    ) -> Flag:
        existing = self._flags.get(name, Flag(name=name))
        resolved_state = existing.state
        if state is not None:
            resolved_state = _coerce_state(state)
        elif enabled is not None:
            resolved_state = FlagState.ON if enabled else FlagState.OFF
        flag = Flag(
            name=name,
            state=resolved_state,
            description=existing.description if description is None else description,
            rollout_percent=existing.rollout_percent if rollout_percent is None else _clamp_percent(rollout_percent),
            tenants_on=existing.tenants_on if tenants_on is None else _as_frozenset(tenants_on),
            tenants_off=existing.tenants_off if tenants_off is None else _as_frozenset(tenants_off),
            default=existing.default if default is None else bool(default),
        )
        self._flags[name] = flag
        return flag

    def clear(self, name: str) -> None:
        self._flags.pop(name, None)

    # -- query ----------------------------------------------------------
    def get(self, name: str) -> Flag | None:
        return self._flags.get(name)

    def list_flags(self) -> list[Flag]:
        return [self._flags[k] for k in sorted(self._flags)]

    def is_enabled(
        self,
        name: str,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        key: str | None = None,
        default: bool | None = None,
    ) -> bool:
        """Evaluate ``name``. Unknown flag -> ``default`` (else flag/False)."""
        flag = self._flags.get(name)
        if flag is None:
            return bool(default) if default is not None else False
        try:
            return flag.evaluate(tenant_id=tenant_id, project_id=project_id, key=key)
        except Exception:  # noqa: BLE001
            if default is not None:
                return bool(default)
            return flag.default
