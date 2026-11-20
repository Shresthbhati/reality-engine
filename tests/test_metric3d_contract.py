"""The metric-3D observation contract (metric3d.v1): relative depth is never metric, gauge units are never metres, refused
records stay in the evidence store, and the record survives persistence and reload. SYNTHETIC records -- these prove the contract's
rules, not any depth model's accuracy."""

from __future__ import annotations

import dataclasses

import pytest

from engine.multiengine.canonical import POSITION, determine
from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.metric3d import (
    SCHEMA_VERSION,
    Metric3DError,
    Metric3DObservation,
    ScaleState,
    ingest,
)

SHA = "a" * 64
RAW = "artifact://" + "b" * 64


def rec(**kw) -> Metric3DObservation:
    base = dict(
        engine_id="depth_x", architecture_id="arch-x", checkpoint_id="ck-1", checkpoint_sha256=SHA, source_id="img1.jpg",
        quantity="position", value=(1.0, 2.0, 3.0), units="sfm_units", scale_state=ScaleState.SFM_GAUGE, frame="colmap_world",
        camera_id="cam1", image_size=(640, 480), image_region=(10.0, 10.0, 100.0, 100.0), raw_artifact_ref=RAW,
        confidence=0.7, confidence_basis="raw_model",
    )
    base.update(kw)
    return Metric3DObservation(**base)


def test_a_complete_gauge_observation_is_valid():
    assert rec().problems() == []


@pytest.mark.parametrize("kw,needle", [
    (dict(units="m"), "contradict"),  # gauge units cannot be relabelled metres
    (dict(scale_state=ScaleState.METRIC_ANCHORED, units="m"), "metric_anchor_ref"),  # metres without an anchor
    (dict(quantity="depth_metric"), "METRIC_ANCHORED"),  # a 'metric depth' that is not anchored
    (dict(quantity="depth_relative"), "RELATIVE_DEPTH"),
    (dict(metric_anchor_ref="anchor://x"), "not METRIC_ANCHORED"),
    (dict(frame="unknown"), "coordinate frame"),
    (dict(frame=""), "coordinate frame"),
    (dict(value=(1.0, float("nan"), 3.0)), "finite"),
    (dict(value=()), "non-empty value"),  # a valid observation must carry a value
    (dict(value=(), validation_status="rejected", rejection_reason="r", raw_artifact_ref=None, covariance=((1.0,),)), "0x0"),
    (dict(confidence=1.4), "outside"),
    (dict(confidence_basis=None), "basis"),
    (dict(confidence=None), "without a confidence"),
    (dict(checkpoint_sha256="xyz"), "64"),
    (dict(image_region=(50.0, 50.0, 40.0, 40.0)), "inverted"),
    (dict(image_region=(0.0, 0.0, 9999.0, 10.0)), "outside the original"),
    (dict(raw_artifact_ref=None), "raw_artifact_ref"),
    (dict(raw_artifact_ref="file:///tmp/x"), "artifact://"),
    (dict(covariance=((1.0, 0.0), (0.0, 1.0))), "3x3"),
    (dict(covariance=((1.0, 2.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))), "symmetric"),
    (dict(covariance=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))), "positive semi-definite"),
    (dict(covariance=((float("inf"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))), "non-finite"),
    (dict(engine_id=""), "engine_id"),
    (dict(validation_status="rejected"), "rejection_reason"),
    (dict(rejection_reason="why"), "valid observation cannot"),
])
def test_every_contract_violation_is_reported(kw, needle):
    probs = rec(**kw).problems()
    assert probs and any(needle in p for p in probs), probs


def test_relative_depth_is_its_own_kind_and_never_metres():
    r = rec(quantity="depth_relative", scale_state=ScaleState.RELATIVE_DEPTH, units="relative", value=(0.42,), frame="cam1_image")
    assert r.problems() == []
    assert dataclasses.replace(r, units="m").problems()  # relative depth cannot claim metres


def test_a_rejected_record_may_carry_no_value_and_no_placeholder_is_invented():
    r = rec(value=(), validation_status="rejected", rejection_reason="no pixels", raw_artifact_ref=None)
    assert r.problems() == [] and r.value == ()


def test_anchored_metres_need_the_anchor_reference():
    ok = rec(scale_state=ScaleState.METRIC_ANCHORED, units="m", metric_anchor_ref="anchor://survey-1")
    assert ok.problems() == []


def test_round_trip_through_dict_and_unknown_versions_are_refused():
    r = rec(covariance=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)))
    assert Metric3DObservation.from_dict(r.to_dict()) == r
    with pytest.raises(Metric3DError, match="unsupported"):
        Metric3DObservation.from_dict({**r.to_dict(), "schema_version": "metric3d.v2"})
    assert r.to_dict()["schema_version"] == SCHEMA_VERSION


def _matrix():
    """A matrix whose artifact store really holds the raw output the records point at."""
    m = EvidenceMatrix()
    uri, _ = m.artifacts.put(b"raw depth output bytes")
    return m, uri


def test_a_missing_raw_artifact_is_a_recorded_rejection_not_a_crash_or_a_silent_success():
    m = EvidenceMatrix()  # the artifact referenced by rec() was never stored
    out = ingest(m, [rec()], run_id="run-1", model_version="v1")
    assert out["accepted"] == [] and len(out["rejected"]) == 1 and "artifact" in out["rejected"][0]
    assert len(m) == 0 and len(m.rejected()) == 1


def test_ingest_persists_valid_records_and_keeps_refused_ones_with_their_reason():
    m, raw = _matrix()
    good = rec(raw_artifact_ref=raw)
    bad = rec(units="m", raw_artifact_ref=raw)  # gauge relabelled as metres
    producer_rejected = rec(validation_status="rejected", rejection_reason="depth map had 80% invalid pixels", raw_artifact_ref=raw)
    out = ingest(m, [good, bad, producer_rejected], run_id="run-1", model_version="v1")
    assert len(out["accepted"]) == 1 and len(out["rejected"]) == 2
    rej = m.rejected()
    assert len(rej) == 2 and any("contradict" in r.reason for r in rej) and any("producer rejected" in r.reason for r in rej)
    assert all("metric3d" in r.draft for r in rej)  # the raw prediction is preserved, not just the reason


def test_an_ingested_observation_reloads_losslessly_and_feeds_canonical_fusion():
    m, raw = _matrix()
    r = rec(covariance=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)), raw_artifact_ref=raw)
    (oid,) = ingest(m, [r], run_id="run-1", model_version="v1")["accepted"]
    obs = m.store.get(oid)
    back = Metric3DObservation.from_observation(obs)
    assert dataclasses.replace(back, observation_id="") == r  # nothing lost between producer and store
    assert obs.coordinate_system == "colmap_world" and obs.predicted["scale_state"] == "sfm_gauge"
    assert obs.execution_config["checkpoint_sha256"] == SHA and m.context(oid).checkpoint_id == "ck-1"
    assert m.context(oid).artifact_refs["raw_output"] == raw  # raw output recoverable
    det = determine(POSITION, [oid], m, target_frame="colmap_world")  # existing consumers read predicted['value'] unchanged
    assert det.value == (1.0, 2.0, 3.0) and det.frame == "colmap_world"


def test_ingest_is_idempotent_for_the_same_record():
    m, raw = _matrix()
    r = rec(raw_artifact_ref=raw)
    t = "2026-01-01T00:00:00+00:00"  # the producer's own timestamp; without it the id (which hashes the time) differs per call
    a = ingest(m, [r], run_id="run-1", model_version="v1", processed_at=t)["accepted"]
    b = ingest(m, [r], run_id="run-1", model_version="v1", processed_at=t)["accepted"]
    assert a == b and len(m) == 1


def test_synthetic_scale_aligned_is_preserved_as_distinct_state():
    r = rec(
        quantity="position",
        scale_state=ScaleState.SYNTHETIC_SCALE_ALIGNED,
        units="synthetic_m",
        value=(2.5, 1.0, 2.0),
        frame="world:canonical",
    )
    assert r.problems() == []
    assert r.scale_state == ScaleState.SYNTHETIC_SCALE_ALIGNED
    assert r.units == "synthetic_m"

    # Serialization round-trip
    d = r.to_dict()
    assert d["scale_state"] == "synthetic_scale_aligned"
    assert d["units"] == "synthetic_m"
    reloaded = Metric3DObservation.from_dict(d)
    assert reloaded == r
