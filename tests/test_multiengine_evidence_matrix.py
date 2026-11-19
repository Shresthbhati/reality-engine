"""Evidence Matrix v2: context sidecar, artifact-backed raw outputs, separate calibrated confidence, append-only status."""

from __future__ import annotations

import json

import pytest

from engine.multiengine.evidence_matrix import (
    EvidenceContext,
    EvidenceMatrix,
    MatrixError,
    artifact_value,
)
from engine.multiengine.observation import ObservationDraft, make_observation


def _obs(value=(1.0, 2.0, 3.0), engine="depth_a", confidence=0.7, **kw):
    draft = ObservationDraft(
        task="detection", quantity="position", source_id="img_1", coordinate_system="world",
        predicted=kw.pop("predicted", {"value": list(value)}), confidence=confidence, **kw,
    )
    return make_observation(
        draft, engine_id=engine, model_version="v1", execution_config={"device": "cpu"},
        processed_at="2026-01-01T00:00:00+00:00",
    )


def _ctx(o, **kw):
    return EvidenceContext(observation_id=o.observation_id, run_id="run-1", **kw)


def test_row_keeps_raw_and_calibrated_confidence_separate():
    m = EvidenceMatrix()
    o = _obs(confidence=0.9)
    m.add(o, _ctx(o, calibrated_confidence=0.6, calibration_basis="isotonic fit on 40 annotated chairs"))
    row = m.row(o.observation_id)
    assert row["raw_confidence"] == 0.9 and row["calibrated_confidence"] == 0.6
    assert row["observation"]["confidence"] == 0.9  # the raw record is untouched
    assert row["status"] == "raw" and row["schema_version"] == 2


def test_unknown_context_fields_stay_none_not_invented():
    m = EvidenceMatrix()
    o = _obs()
    m.add(o, _ctx(o))
    c = m.row(o.observation_id)["context"]
    assert c["architecture_id"] is None and c["checkpoint_id"] is None and c["camera_ref"] is None


def test_calibrated_confidence_requires_basis_and_vice_versa():
    o = _obs()
    with pytest.raises(MatrixError, match="calibration_basis"):
        _ctx(o, calibrated_confidence=0.5)
    with pytest.raises(MatrixError, match="without a calibrated_confidence"):
        _ctx(o, calibration_basis="x")
    with pytest.raises(MatrixError, match="outside"):
        _ctx(o, calibrated_confidence=1.5, calibration_basis="x")
    with pytest.raises(MatrixError, match="run"):
        EvidenceContext(observation_id=o.observation_id, run_id="")


def test_context_is_immutable_once_set_but_identical_readd_is_noop():
    m = EvidenceMatrix()
    o = _obs()
    assert m.add(o, _ctx(o)) is True
    assert m.add(o, _ctx(o)) is False
    with pytest.raises(MatrixError, match="different context"):
        m.add(o, _ctx(o, checkpoint_id="ckpt-2"))
    with pytest.raises(MatrixError, match="does not belong"):
        m.add(o, EvidenceContext(observation_id="obs_other", run_id="r"))


def test_large_output_lives_in_artifact_store_and_is_verified():
    m = EvidenceMatrix()
    uri, digest = m.artifacts.put(b"\x00" * 4096)
    o = _obs(predicted={"depth": artifact_value(uri, shape=[64, 64])})
    m.add(o, _ctx(o, artifact_refs={"mask": uri}))
    assert m.verify() == []
    assert m.row(o.observation_id)["observation"]["predicted"]["depth"]["$artifact"] == uri
    assert digest in uri


def test_reference_to_missing_artifact_is_refused():
    m = EvidenceMatrix()
    o = _obs(predicted={"depth": artifact_value("artifact://" + "ab" * 32)})
    with pytest.raises(MatrixError, match="not in the artifact store"):
        m.add(o, _ctx(o))
    assert len(m) == 0 and len(m.store) == 0


def test_verify_detects_corrupted_artifact_on_disk(tmp_path):
    m = EvidenceMatrix(tmp_path)
    uri, digest = m.artifacts.put(b"original bytes")
    o = _obs(predicted={"depth": artifact_value(uri)})
    m.add(o, _ctx(o))
    assert m.verify() == []
    blob = tmp_path / "artifacts" / digest[:2] / f"{digest}.bin"
    blob.write_bytes(b"tampered")
    problems = m.verify()
    assert len(problems) == 1 and "content hash" in problems[0]


def test_status_history_is_append_only_and_rejected_rows_remain():
    m = EvidenceMatrix()
    o = _obs()
    m.add(o, _ctx(o))
    m.set_status(o.observation_id, "validated", reason="matches tape measure", by="annotator-1")
    m.set_status(o.observation_id, "rejected", reason="later contradicted by 3 views", by="consensus")
    assert m.status(o.observation_id) == "rejected"
    assert [e.status for e in m.status_history(o.observation_id)] == ["validated", "rejected"]
    assert len(m.rows()) == 1  # a rejected row is still in the matrix
    with pytest.raises(MatrixError):
        m.set_status(o.observation_id, "validated", reason="", by="x")
    with pytest.raises(MatrixError, match="unknown status"):
        m.set_status(o.observation_id, "approved", reason="r", by="x")
    with pytest.raises(MatrixError, match="no evidence row"):
        m.set_status("obs_missing", "validated", reason="r", by="x")


def test_persistence_round_trip(tmp_path):
    m = EvidenceMatrix(tmp_path)
    uri, _ = m.artifacts.put(b"blob")
    o = _obs(predicted={"depth": artifact_value(uri)})
    m.add(o, _ctx(o, architecture_id="arch-x", checkpoint_id="ck-7", camera_ref="cam-3",
                  calibrated_confidence=0.4, calibration_basis="held-out set A"))
    m.set_status(o.observation_id, "validated", reason="ok", by="me")
    again = EvidenceMatrix(tmp_path)
    assert again.row(o.observation_id) == m.row(o.observation_id)
    assert again.verify() == []


def test_corrupt_sidecar_line_is_named_not_skipped(tmp_path):
    m = EvidenceMatrix(tmp_path)
    o = _obs()
    m.add(o, _ctx(o))
    log = tmp_path / "context.jsonl"
    log.write_text(log.read_text() + "{not json}\n")
    with pytest.raises(MatrixError, match=r"context\.jsonl:2"):
        EvidenceMatrix(tmp_path)


def test_sidecar_status_for_unknown_row_is_rejected_on_load(tmp_path):
    m = EvidenceMatrix(tmp_path)
    o = _obs()
    m.add(o, _ctx(o))
    log = tmp_path / "context.jsonl"
    bad = json.dumps({"type": "status", "observation_id": "obs_nope", "status": "validated", "reason": "r", "by": "b"})
    log.write_text(log.read_text() + bad + "\n")
    with pytest.raises(MatrixError, match="unknown row"):
        EvidenceMatrix(tmp_path)


def test_verify_reports_observation_added_to_store_without_context():
    m = EvidenceMatrix()
    o = _obs()
    m.store.add(o)  # bypassing the matrix
    assert m.verify() == [f"observation {o.observation_id} has no evidence context"]
