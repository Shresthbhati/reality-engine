"""Calibration records: dataset + method + version are required, raw and calibrated confidence stay separate, and records written
before the structured form existed (a free-text basis) still load. SYNTHETIC values; no real calibration has been performed."""

from __future__ import annotations

import json

import pytest

from engine.multiengine.canonical import POSITION, explain
from engine.multiengine.evidence_matrix import (
    LEGACY_UNSPECIFIED,
    CalibrationRef,
    EvidenceContext,
    EvidenceMatrix,
    MatrixError,
)
from engine.multiengine.worldir_bridge import convert
from tests.test_multiengine_worldir_bridge import ENGINES, Builder, _canon

CAL = CalibrationRef(dataset="synthetic-annotated-chairs-v1", method="isotonic-regression", version="1.0")


def test_a_calibrated_confidence_must_name_dataset_method_and_version():
    ctx = EvidenceContext("obs_x", "run-1", calibrated_confidence=0.7, calibration=CAL)
    assert ctx.calibration.complete and ctx.calibrated_confidence == 0.7
    for kw in ({"dataset": "", "method": "m", "version": "1"}, {"dataset": "d", "method": "", "version": "1"},
               {"dataset": "d", "method": "m", "version": ""}):
        with pytest.raises(MatrixError, match="dataset, method and version"):
            CalibrationRef(**kw)
    with pytest.raises(MatrixError, match="calibration=CalibrationRef"):
        EvidenceContext("obs_x", "run-1", calibrated_confidence=0.7)
    with pytest.raises(MatrixError, match="without a calibrated_confidence"):
        EvidenceContext("obs_x", "run-1", calibration=CAL)
    with pytest.raises(MatrixError, match="without a calibrated_confidence"):
        EvidenceContext("obs_x", "run-1", calibration_basis="x")


def test_a_legacy_basis_is_converted_and_marked_incomplete_not_complete():
    ctx = EvidenceContext("obs_x", "run-1", calibrated_confidence=0.6, calibration_basis="held-out set A")
    assert ctx.calibration == CalibrationRef("held-out set A", LEGACY_UNSPECIFIED, LEGACY_UNSPECIFIED)
    assert ctx.calibration.complete is False and ctx.to_dict()["calibration"]["complete"] is False


def test_structured_calibration_survives_persistence_and_a_restart(tmp_path):
    b = Builder()
    oid = b.obs(ENGINES[0], POSITION, {"value": [1.0, 1.0, 0.0]})
    o = b.m.store.get(oid)
    m2 = EvidenceMatrix(tmp_path / "second")
    m2.add(o, EvidenceContext(oid, "run-1", calibrated_confidence=0.7, calibration=CAL))
    again = EvidenceMatrix(tmp_path / "second")
    row = again.row(oid)
    assert row["raw_confidence"] == 0.8 and row["calibrated_confidence"] == 0.7  # raw and calibrated stay separate
    expected = {"dataset": CAL.dataset, "method": CAL.method, "version": CAL.version, "complete": True}
    assert row["context"]["calibration"] == expected


def test_records_written_before_the_structured_form_still_load(tmp_path):
    """MIGRATION: a context line exactly as PR #340 wrote it (free-text calibration_basis, no 'calibration' key) must load."""
    b = Builder()
    oid = b.obs(ENGINES[0], POSITION, {"value": [1.0, 1.0, 0.0]})
    o = b.m.store.get(oid)
    m = EvidenceMatrix(tmp_path)
    m.store.add(o)  # the observation file exists; now write the context line by hand in the OLD format
    old = {"type": "context", "observation_id": oid, "run_id": "run-old", "architecture_id": None, "checkpoint_id": None,
           "camera_ref": None, "world_transform": None, "artifact_refs": {}, "calibrated_confidence": 0.55,
           "calibration_basis": "isotonic fit on 40 annotated chairs"}
    (tmp_path / "context.jsonl").write_text(json.dumps(old, sort_keys=True) + "\n", encoding="utf-8")
    ctx = EvidenceMatrix(tmp_path).context(oid)
    assert ctx.calibrated_confidence == 0.55 and ctx.validation is None
    assert ctx.calibration.dataset == "isotonic fit on 40 annotated chairs" and ctx.calibration.complete is False


def test_explain_and_the_bridge_carry_the_calibration_and_flag_an_incomplete_one():
    b = Builder()
    ids = [b.obs(ENGINES[0], POSITION, {"value": [2.0, 1.0, 0.45]}), b.obs(ENGINES[1], POSITION, {"value": [2.02, 1.01, 0.45]})]
    # two more observations whose contexts carry a STRUCTURED and a LEGACY calibration
    for engine, calib in ((ENGINES[2], {"calibration": CAL}), ("owlvit", {"calibration_basis": "old set"})):
        o = b.obs(engine, POSITION, {"value": [2.01, 1.0, 0.45]})
        b.m._ctx[o] = EvidenceContext(o, b.m.context(o).run_id, calibrated_confidence=0.7, **calib)  # noqa: SLF001 -- fixture
        ids.append(o)
    ce = _canon(b, ids, "chair")
    exp = explain(ce, b.m)
    assert any("calibration of" in u and "incomplete" in u and "old set" in u for u in exp["uncertain"])
    ent = next(iter(convert([ce], b.m).entities.values()))
    cals = {o.id: o.metadata["calibration"] for o in ent.observations}
    assert any(c and c["dataset"] == CAL.dataset and c["complete"] for c in cals.values())
    assert any(c and c["dataset"] == "old set" and not c["complete"] for c in cals.values())
    assert any(c is None for c in cals.values())  # uncalibrated observations stay None, never a default
