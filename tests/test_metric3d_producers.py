"""metric3d.v1 producers: raw depth arrays and lift supports are persisted as recoverable artifacts, scale is never over-claimed, and
refused output stays in the evidence store. SYNTHETIC arrays and points -- these prove the producers' rules and the producer/consumer
agreement, not any model's accuracy (the real-model run is reported separately)."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.metric3d import Metric3DObservation, ScaleState, ingest
from engine.multiengine.metric3d_producers import (
    MIN_LIFT_POINTS,
    checkpoint_sha256,
    depth_records_from_drafts,
    npy_bytes,
    sparse_lift_records,
)
from engine.multiengine.observation import ObservationDraft

SHA = "c" * 64


def _draft(arr, scale="RELATIVE_UNSCALED", **extra):
    pred = {"array": arr, "scale_state": scale, **extra}
    return ObservationDraft(task="depth", quantity="depth_map", source_id="img1.jpg", coordinate_system="cam:img1", predicted=pred)


def _depth(m, drafts):
    return depth_records_from_drafts(drafts, matrix=m, engine_id="midas", architecture_id="dpt-hybrid",
                                     checkpoint_id="dpt_hybrid_384.pt", checkpoint_hash=SHA)


def _stored(m, ref):
    return m.artifacts.get(ref)


def test_the_raw_validator_refuses_an_ndarray_which_is_why_producers_must_externalise_it():
    from engine.multiengine.observation import ObservationError, make_observation

    with pytest.raises(ObservationError, match="ndarray"):
        make_observation(_draft(np.zeros((2, 2), dtype=np.float32)), engine_id="e", model_version="v", execution_config={})


def test_relative_depth_is_recorded_relative_with_the_raw_array_recoverable_byte_exact():
    m = EvidenceMatrix()
    arr = np.random.default_rng(0).uniform(0.1, 9.0, (6, 8)).astype(np.float32)
    (r,) = _depth(m, [_draft(arr)])
    assert r.problems() == [] and r.scale_state is ScaleState.RELATIVE_DEPTH and r.units == "relative" and r.quantity == "depth_relative"
    assert r.image_size == (8, 6) and r.value == (float(arr.min()), float(np.median(arr)), float(arr.max()))
    out = ingest(m, [r], run_id="run-1", model_version="v1")
    assert len(out["accepted"]) == 1 and out["rejected"] == []
    stored = _stored(m, r.raw_artifact_ref)
    assert stored == npy_bytes(arr)  # the original array, byte for byte
    assert np.array_equal(np.load(io.BytesIO(stored)), arr)


def test_a_model_claiming_metres_without_an_anchor_is_unknown_scale_never_metres():
    m = EvidenceMatrix()
    (r,) = _depth(m, [_draft(np.full((4, 4), 2.0, dtype=np.float32), scale="METRIC_CALIBRATED")])
    assert r.scale_state is ScaleState.UNKNOWN and r.units == "unknown" and r.quantity == "depth_unanchored" and r.problems() == []


def test_test_double_output_is_rejected_but_kept():
    m = EvidenceMatrix()
    (r,) = _depth(m, [_draft(np.ones((4, 4), dtype=np.float32), is_test_double=True)])
    assert r.validation_status == "rejected" and "test double" in r.rejection_reason and r.value == ()
    out = ingest(m, [r], run_id="run-1", model_version="v1")
    assert out["accepted"] == [] and len(m.rejected()) == 1 and len(m) == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_depth_is_rejected_with_the_raw_map_still_stored(bad):
    m = EvidenceMatrix()
    arr = np.ones((4, 4), dtype=np.float32)
    arr[1, 2] = bad
    (r,) = _depth(m, [_draft(arr)])
    assert r.validation_status == "rejected" and "non-finite" in r.rejection_reason and r.raw_artifact_ref
    assert np.array_equal(np.load(io.BytesIO(_stored(m, r.raw_artifact_ref))), arr, equal_nan=True)  # evidence kept


def test_a_wrong_shape_or_missing_array_is_rejected_not_crashed():
    m = EvidenceMatrix()
    empty = ObservationDraft(task="depth", quantity="depth_map", source_id="i", coordinate_system="cam:i", predicted={})
    recs = _depth(m, [_draft(np.ones((3,), dtype=np.float32)), empty])
    assert [r.validation_status for r in recs] == ["rejected", "rejected"]


def test_checkpoint_hash_is_a_real_sha256_or_none(tmp_path):
    import hashlib

    f = tmp_path / "w.bin"
    f.write_bytes(b"weights")
    assert checkpoint_sha256(f) == hashlib.sha256(b"weights").hexdigest()
    assert checkpoint_sha256(tmp_path / "missing.bin") is None and checkpoint_sha256(None) is None


def _cloud(n=40, centre=(1.0, 2.0, 3.0), seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform((100, 100), (200, 200), (n, 2)), rng.normal(centre, 0.05, (n, 3))


def test_sparse_lift_gives_a_gauge_centroid_with_a_valid_covariance_and_recoverable_support():
    m = EvidenceMatrix()
    p2, p3 = _cloud()
    far2, far3 = np.array([[500.0, 500.0]] * 10), np.full((10, 3), 99.0)  # points outside the region must not leak in
    (r,) = sparse_lift_records([{"box_2d": (90, 90, 210, 210), "label": "window", "confidence": 0.8}],
                               points2d=np.vstack([p2, far2]), points3d=np.vstack([p3, far3]), matrix=m, source_id="img1.jpg",
                               camera_id="cam1", image_size=(640, 480), frame="colmap_world")
    assert r.problems() == []
    assert r.scale_state is ScaleState.SFM_GAUGE and r.units == "sfm_units" and r.quantity == "visible_surface_centroid"
    assert np.allclose(r.value, (1.0, 2.0, 3.0), atol=0.05)
    cov = np.asarray(r.covariance)
    assert cov.shape == (3, 3) and np.all(np.linalg.eigvalsh(cov) >= 0) and np.sqrt(np.diag(cov)).max() < 0.05  # SE of the mean
    out = ingest(m, [r], run_id="run-1", model_version="v1")
    assert len(out["accepted"]) == 1
    support = json.loads(_stored(m, r.raw_artifact_ref))
    assert support["n_points_in_region"] == 40 and support["label"] == "window" and len(support["points3d"]) == 40


def test_too_few_sparse_points_is_a_rejected_record_never_a_made_up_position():
    m = EvidenceMatrix()
    p2, p3 = _cloud(n=MIN_LIFT_POINTS - 1)
    (r,) = sparse_lift_records([{"box_2d": (90, 90, 210, 210)}], points2d=p2, points3d=p3, matrix=m, source_id="i.jpg",
                               camera_id=None, image_size=None, frame="colmap_world")
    assert r.validation_status == "rejected" and r.value == () and "not determinable" in r.rejection_reason
    assert r.camera_id is None  # unknown camera stays unknown


def test_the_lift_is_never_confusable_with_a_position_or_metres():
    from engine.multiengine.canonical import POSITION

    m = EvidenceMatrix()
    p2, p3 = _cloud()
    (r,) = sparse_lift_records([{"box_2d": (90, 90, 210, 210)}], points2d=p2, points3d=p3, matrix=m, source_id="i.jpg",
                               camera_id="c", image_size=(640, 480), frame="colmap_world")
    assert r.quantity != POSITION and r.units != "m" and r.scale_state is not ScaleState.METRIC_ANCHORED


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError):
        sparse_lift_records([], points2d=np.zeros((3, 2)), points3d=np.zeros((4, 3)), matrix=EvidenceMatrix(), source_id="i",
                            camera_id=None, image_size=None, frame="f")


class _FakeDepthAdapter:
    """SYNTHETIC adapter: proves the orchestration (always unloads, reports failures, never substitutes another model)."""

    def __init__(self, arr=None, fail=False):
        self.spec = type("S", (), {"engine_id": "fake_depth", "name": "fake", "version": "v9"})()
        self.arr, self.fail, self.loaded, self.unloaded = arr, fail, 0, 0

    def validate_input(self, request):
        pass

    def load(self, device):
        self.loaded += 1

    def unload(self):
        self.unloaded += 1

    def infer(self, request):
        if self.fail:
            raise RuntimeError("synthetic crash")
        return [_draft(self.arr)]


def test_produce_depth_runs_one_adapter_persists_its_output_and_always_unloads():
    from engine.multiengine.metric3d_producers import produce_depth

    m = EvidenceMatrix()
    a = _FakeDepthAdapter(np.ones((4, 4), dtype=np.float32))
    res = produce_depth(a, ["img1.jpg"], matrix=m, run_id="run-1", checkpoint_id="ck", checkpoint_hash=SHA)
    assert res["status"] == "succeeded" and len(res["accepted"]) == 1 and a.loaded == 1 and a.unloaded == 1
    assert res["checkpoint_sha256"] == SHA and res["wall_s"] >= 0


def test_produce_depth_reports_a_failure_still_unloads_and_persists_nothing():
    from engine.multiengine.metric3d_producers import produce_depth

    m = EvidenceMatrix()
    a = _FakeDepthAdapter(fail=True)
    res = produce_depth(a, ["img1.jpg"], matrix=m, run_id="run-1", checkpoint_id="ck", checkpoint_hash=None)
    assert res["status"] == "failed" and "synthetic crash" in res["error"] and a.unloaded == 1
    assert res["accepted"] == [] and len(m) == 0 and res["checkpoint_sha256"] is None


def test_the_lifter_adapter_reads_the_real_request_field_and_refuses_missing_inputs_explicitly():
    """Regression: a LOADED Metric3DLifterAdapter used to crash with AttributeError (EngineRequest has no ``options``)."""
    from engine.multiengine.adapters.metric_3d_lifter import Metric3DLifterAdapter
    from engine.multiengine.contract import Device, EngineError, EngineErrorKind, EngineRequest
    from engine.multiengine.registry import maximum_fidelity_registry

    a = Metric3DLifterAdapter(maximum_fidelity_registry().get("sam"))
    a.load(Device("cpu"))
    with pytest.raises(EngineError) as no_camera:
        a.infer(EngineRequest(sources=("photo.jpg",)))
    assert no_camera.value.kind is EngineErrorKind.INVALID_INPUT and "camera" in str(no_camera.value)
    with pytest.raises(EngineError) as relative:
        a.infer(EngineRequest(sources=("photo.jpg",), payload={"camera": {"intrinsics": {}}, "is_metric_depth": False}))
    assert relative.value.kind is EngineErrorKind.INVALID_INPUT  # relative depth can never be unprojected as metres
    a.unload()


def test_producers_and_consumers_agree_a_produced_record_survives_from_observation():
    m = EvidenceMatrix()
    arr = np.linspace(0.2, 3.0, 24, dtype=np.float32).reshape(4, 6)
    (r,) = _depth(m, [_draft(arr)])
    (oid,) = ingest(m, [r], run_id="run-1", model_version="v1")["accepted"]
    back = Metric3DObservation.from_observation(m.store.get(oid))
    assert back.raw_artifact_ref == r.raw_artifact_ref and back.scale_state is r.scale_state and back.checkpoint_sha256 == SHA
