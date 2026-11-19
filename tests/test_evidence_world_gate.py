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
    assert all(c["evidence"] or c["note"] for c in g["criteria"])  # every criterion says what proves it, or why it is unmet
    assert "metric3d_producers_emit_contract" in g["unmet"]  # the contract existing is not the same as producers emitting it


def test_a_claimed_criterion_is_demoted_when_its_evidence_file_is_missing(tmp_path):
    g = ew.enablement_gate(repo_root=tmp_path)  # an empty checkout: none of the evidence tests exist there
    by_id = {c["id"]: c for c in g["criteria"]}
    assert by_id["tenant_isolation"]["met"] is False and by_id["tenant_isolation"]["missing_evidence"]
    real = ew.enablement_gate()
    assert {c["id"]: c for c in real["criteria"]}["tenant_isolation"]["met"] is True  # present in the real checkout


def test_the_gate_is_ready_only_when_every_criterion_is_met_and_never_flips_the_flag(monkeypatch):
    monkeypatch.delenv(ew.ENV_FLAG, raising=False)
    all_met = tuple((c, True, ("tests/test_evidence_world_gate.py",), "") for c, *_ in ew.ENABLEMENT_CRITERIA)
    monkeypatch.setattr(ew, "ENABLEMENT_CRITERIA", all_met)
    assert ew.enablement_gate()["ready"] is True  # synthetic table: proves the logic, not that the real criteria are met
    assert ew.enabled() is False  # a ready gate still does not turn the stage on by itself
