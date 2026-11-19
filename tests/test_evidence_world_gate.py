"""The stage is opt-in and the production-default gate is honest: not ready while any criterion is unmet."""

from apps.api import evidence_world_stage as ew


def test_stage_is_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv(ew.ENV_FLAG, raising=False)
    assert ew.enabled() is False
    for v in ("0", "", "no", "off"):
        monkeypatch.setenv(ew.ENV_FLAG, v)
        assert ew.enabled() is False
    monkeypatch.setenv(ew.ENV_FLAG, "1")
    assert ew.enabled() is True


def test_gate_is_not_ready_while_any_criterion_is_unmet():
    g = ew.enablement_gate()
    assert g["ready"] is False and g["unmet"]
    assert {c["id"] for c in g["criteria"]} >= {"real_multi_photo_job_run", "detector_accuracy_benchmark", "confidence_calibration"}
    assert all(c["evidence"] for c in g["criteria"])
