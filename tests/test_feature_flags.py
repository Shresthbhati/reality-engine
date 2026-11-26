"""Tests for apps.api.feature_flags (Programme 1 feature flags)."""

from __future__ import annotations

import json

from apps.api.feature_flags import FeatureFlagStore, FlagState


def test_unknown_flag_returns_default_false():
    s = FeatureFlagStore()
    assert s.is_enabled("nope") is False
    assert s.is_enabled("nope", default=True) is True


def test_on_off_states():
    s = FeatureFlagStore({"a": {"state": "on"}, "b": {"state": "off"}})
    assert s.is_enabled("a") is True
    assert s.is_enabled("b") is False


def test_boolean_and_simple_shorthand_sources():
    s = FeatureFlagStore({"x": True, "y": False, "z": "on"})
    assert s.is_enabled("x") and not s.is_enabled("y") and s.is_enabled("z")


def test_tenant_overrides_beat_rollout():
    s = FeatureFlagStore({
        "f": {"state": "conditional", "rollout_percent": 0, "tenants_on": ["t1"], "tenants_off": ["t2"]}
    })
    assert s.is_enabled("f", tenant_id="t1") is True      # allowlist wins over 0% rollout
    assert s.is_enabled("f", tenant_id="t2") is False     # denylist
    assert s.is_enabled("f", tenant_id="t3") is False     # 0% rollout
    s2 = FeatureFlagStore({"f": {"state": "on", "tenants_off": ["t2"]}})
    assert s2.is_enabled("f", tenant_id="t2") is False    # denylist wins over ON
    assert s2.is_enabled("f", tenant_id="t9") is True


def test_deterministic_percentage_rollout_is_stable():
    s = FeatureFlagStore({"roll": {"state": "conditional", "rollout_percent": 50}})
    # stable: same key -> same answer every time, across a fresh store instance
    s2 = FeatureFlagStore({"roll": {"state": "conditional", "rollout_percent": 50}})
    for i in range(200):
        key = f"tenant-{i}"
        assert s.is_enabled("roll", tenant_id=key) == s2.is_enabled("roll", tenant_id=key)
    # boundaries
    assert FeatureFlagStore({"r": {"state": "conditional", "rollout_percent": 100}}).is_enabled("r", key="x") is True
    assert FeatureFlagStore({"r": {"state": "conditional", "rollout_percent": 0}}).is_enabled("r", key="x") is False


def test_rollout_percent_clamped():
    s = FeatureFlagStore({"r": {"state": "conditional", "rollout_percent": 999}})
    assert s.is_enabled("r", key="x") is True
    s2 = FeatureFlagStore({"r": {"state": "conditional", "rollout_percent": -5}})
    assert s2.is_enabled("r", key="x") is False


def test_runtime_set_and_clear():
    s = FeatureFlagStore()
    s.set("feat", enabled=True)
    assert s.is_enabled("feat") is True
    s.set("feat", enabled=False)
    assert s.is_enabled("feat") is False
    s.set("feat", state=FlagState.CONDITIONAL, rollout_percent=100)
    assert s.is_enabled("feat", tenant_id="anyone") is True
    s.clear("feat")
    assert s.get("feat") is None and s.is_enabled("feat") is False


def test_list_flags_sorted():
    s = FeatureFlagStore({"b": True, "a": True, "c": False})
    assert [f.name for f in s.list_flags()] == ["a", "b", "c"]


def test_json_file_source(tmp_path):
    p = tmp_path / "flags.json"
    p.write_text(json.dumps({"flags": {"k": {"state": "on"}, "j": True}}), encoding="utf-8")
    s = FeatureFlagStore(path=p)
    assert s.is_enabled("k") and s.is_enabled("j")


def test_malformed_source_is_failsafe(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    s = FeatureFlagStore(source={"ok": True}, path=p)  # bad file ignored, explicit kept
    assert s.is_enabled("ok") is True
    assert s.is_enabled("k") is False


def test_env_json_source(monkeypatch):
    monkeypatch.setenv("REALITY_FEATURE_FLAGS_JSON", json.dumps({"envflag": {"state": "on"}}))
    s = FeatureFlagStore()
    assert s.is_enabled("envflag") is True


def test_evaluation_never_raises_on_weird_types():
    s = FeatureFlagStore({"f": {"state": "conditional", "rollout_percent": "abc"}})
    assert s.is_enabled("f", tenant_id="t") is True  # bad percent -> clamped to 100
    assert s.is_enabled("f") is True
