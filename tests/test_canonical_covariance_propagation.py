"""Covariance through frame transforms: Sigma' = J Sigma J^T, never silently dropped. SYNTHETIC observations and frame graph;
proves the propagation mathematics and the refusal paths, not any model's accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from engine.multiengine.alignment import Frame, FrameGraph, Sim3
from engine.multiengine.canonical import DIMENSIONS, POSITION, determine
from engine.multiengine.evidence_matrix import EvidenceContext, EvidenceMatrix
from engine.multiengine.observation import ObservationDraft, make_observation

THETA = 0.7
R = np.array([[np.cos(THETA), -np.sin(THETA), 0.0], [np.sin(THETA), np.cos(THETA), 0.0], [0.0, 0.0, 1.0]])
SCALE = 2.5


def _graph(src_scale="metric"):
    g = FrameGraph()
    g.add_frame(Frame("cam", kind="camera", scale_state=src_scale))
    g.add_frame(Frame("world", kind="world", scale_state="metric"))
    g.add_transform("cam", "world", Sim3.from_arrays(SCALE, R, np.array([1.0, -2.0, 3.0])), source="synthetic", scale_anchored=True)
    return g


def _add(m, quantity, value, cov, frame="cam"):
    unc = {"covariance": cov} if cov is not None else None
    o = make_observation(
        ObservationDraft(task="detection", quantity=quantity, source_id="img.jpg", coordinate_system=frame,
                         predicted={"value": value}, confidence=0.9, uncertainty=unc, image_region=(1, 1, 9, 9)),
        engine_id="eng_a", model_version="v1", execution_config={}, processed_at="2026-01-01T00:00:00+00:00",
    )
    m.add(o, EvidenceContext(o.observation_id, run_id="r", architecture_id="a", checkpoint_id="c", validation="synthetic"))
    return o.observation_id


def test_position_covariance_is_propagated_exactly_through_a_similarity():
    m = EvidenceMatrix()
    cov = np.diag([0.01, 0.04, 0.09])
    oid = _add(m, POSITION, [1.0, 2.0, 3.0], cov.tolist())
    det = determine(POSITION, [oid], m, target_frame="world", graph=_graph())
    J = SCALE * R
    expected = J @ cov @ J.T
    assert det.value is not None
    assert np.allclose(det.value, SCALE * R @ [1.0, 2.0, 3.0] + [1.0, -2.0, 3.0])
    assert any("propagated" in n for n in det.notes) and not any("dropped" in n for n in det.notes)
    assert det.uncertainty is not None
    got = np.asarray(det.uncertainty["covariance"], dtype=float)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-9)


def test_length_covariance_scales_by_the_path_scale_squared():
    m = EvidenceMatrix()
    cov = np.diag([0.01, 0.01, 0.01])
    oid = _add(m, DIMENSIONS, [1.0, 2.0, 3.0], cov.tolist())
    det = determine(DIMENSIONS, [oid], m, target_frame="world", graph=_graph())
    assert np.allclose(det.value, SCALE * np.array([1.0, 2.0, 3.0]))
    got = np.asarray(det.uncertainty["covariance"], dtype=float)
    assert np.allclose(got, SCALE**2 * cov)


def test_malformed_covariance_is_marked_unavailable_not_guessed():
    m = EvidenceMatrix()
    oid = _add(m, POSITION, [1.0, 2.0, 3.0], [[0.01, 0.0], [0.0, 0.01]])  # 2x2 for a 3-vector
    det = determine(POSITION, [oid], m, target_frame="world", graph=_graph())
    assert any("unavailable" in n for n in det.notes)
    assert not (det.uncertainty or {}).get("covariance")


def test_no_covariance_in_means_none_out_and_no_note_claiming_propagation():
    m = EvidenceMatrix()
    oid = _add(m, POSITION, [1.0, 2.0, 3.0], None)
    det = determine(POSITION, [oid], m, target_frame="world", graph=_graph())
    assert not any("propagated" in n for n in det.notes)


def test_incompatible_frames_are_refused_and_the_observation_kept_as_excluded():
    m = EvidenceMatrix()
    oid = _add(m, POSITION, [1.0, 2.0, 3.0], np.eye(3).tolist(), frame="no-such-frame")
    det = determine(POSITION, [oid], m, target_frame="world", graph=_graph())
    assert det.status == "unmeasured" and oid in [e[0] for e in det.excluded]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_covariance_cannot_be_stored_as_evidence(bad):
    from engine.multiengine.observation import ObservationError

    with pytest.raises(ObservationError):
        _add(EvidenceMatrix(), POSITION, [1.0, 2.0, 3.0], [[bad, 0, 0], [0, 1, 0], [0, 0, 1]])
