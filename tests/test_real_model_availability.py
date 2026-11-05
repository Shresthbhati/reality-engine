"""Tests for the unified real-model availability report
(perception/availability.py -- real-model paths audit).

Mission requirement: "Real models must not silently fail. Optional
dependencies should use explicit skip reason / blocked state /
availability report. Do not silently return empty results."

Before this module each backend individually raises honest errors at
CONSTRUCTION time -- but nothing answers, without constructing a model
(and possibly downloading weights), the operator question: which real
models can this machine run right now, and for each one that cannot
run, WHY (missing dependency vs missing checkpoint vs missing binary)?

The report must:
  - check dependencies/checkpoints WITHOUT importing torch-heavy modules
    or downloading anything (probe-only, side-effect free);
  - classify every unavailable model with an explicit status and reason
    (never "available" as a guess, never a silent empty result);
  - mark a model "available" only when its dependency imports, its
    checkpoint file resolves, and (for binaries) the executable exists;
  - carry per-model remediation (how to make it available) so the skip
    reason is actionable, not just descriptive.
"""

from __future__ import annotations

import pytest

from perception.availability import (
    REAL_MODEL_SPECS,
    ModelAvailability,
    real_models_by_name,
    report_real_models,
)


def test_report_covers_the_real_model_paths():
    names = {m.name for m in report_real_models()}
    assert {"colmap", "midas", "sam", "mask_rcnn"} <= names


def test_every_model_has_explicit_status_and_reason():
    for model in report_real_models():
        assert model.status in ("available", "blocked")
        assert isinstance(model.reason, str) and model.reason
        if model.status == "blocked":
            assert model.remediation, f"{model.name} blocked without remediation"


def test_midas_blocked_reason_names_the_missing_piece():
    """In the CI/unit environment torch may or may not be present; the
    invariant is the CLASSIFICATION, not a particular status."""
    m = real_models_by_name()["midas"]
    if m.status == "blocked":
        assert (
            "torch" in m.reason.lower()
            or "checkpoint" in m.reason.lower()
            or "cache" in m.reason.lower()
        )


def test_checkpoint_only_models_do_not_download():
    """Probing twice must be cheap and deterministic -- no network
    fetch, no model construction. (If this test hangs, the report is
    downloading; that's the failure this test exists to catch.)"""
    r1 = report_real_models()
    r2 = report_real_models()
    assert {m.name: m.status for m in r1} == {m.name: m.status for m in r2}


def test_model_availability_to_dict():
    m = ModelAvailability(
        name="x", status="blocked", reason="r", remediation="fix",
    )
    d = m.to_dict()
    assert d["name"] == "x" and d["status"] == "blocked"
    assert d["remediation"] == "fix"


def test_specs_registry_is_descriptive_not_loaded():
    """The spec table names the integration path for each real model
    without importing it -- the audit trail for 'which backend behind
    which adapter'."""
    assert set(REAL_MODEL_SPECS) >= {"colmap", "midas", "sam", "mask_rcnn"}
    for spec in REAL_MODEL_SPECS.values():
        assert spec["adapter"], spec


def test_probe_checks_actual_runtime_deps(monkeypatch):
    """A cached checkpoint is not enough: a missing runtime module must block with a precise reason."""
    import perception.availability as av

    present = {"torch", "torchvision"}  # timm / transformers / segment_anything absent
    monkeypatch.setattr(av, "_dependency_ok", lambda m: m in present)
    monkeypatch.setattr(av, "_first_existing", lambda names: av.Path("fake.pt"))  # weights "cached"
    monkeypatch.setattr(av, "_hf_cached", lambda *a, **k: True)
    got = real_models_by_name()
    assert got["midas"].status == "blocked" and "timm" in got["midas"].reason
    assert got["sam"].status == "blocked" and "segment_anything" in got["sam"].reason
    for n in ("depth_anything", "owlvit"):
        assert got[n].status == "blocked" and "transformers" in got[n].reason
        assert got[n].remediation


def test_hf_models_blocked_on_missing_checkpoint_when_deps_present(monkeypatch):
    import perception.availability as av

    monkeypatch.setattr(av, "_dependency_ok", lambda m: True)
    monkeypatch.setattr(av, "_hf_cached", lambda *a, **k: False)
    got = real_models_by_name()
    for n in ("depth_anything", "owlvit"):
        assert got[n].status == "blocked" and "checkpoint" in got[n].reason
