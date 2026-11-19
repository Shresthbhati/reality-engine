"""Canonical entity + explain(): SYNTHETIC chair scenario (proves the logic, says nothing about real-world accuracy)."""

from __future__ import annotations

import numpy as np

from engine.multiengine.alignment import Frame, FrameGraph, Sim3
from engine.multiengine.association import Candidate, PairScore, associate
from engine.multiengine.canonical import (
    DIMENSIONS,
    POSITION,
    build_canonical_entity,
    explain,
    render_explanation,
)
from engine.multiengine.evidence_matrix import EvidenceContext, EvidenceMatrix
from engine.multiengine.observation import ObservationDraft, make_observation
from engine.multiengine.registry import default_registry

TOL = {POSITION: 0.1, DIMENSIONS: 0.05}


def _add(m, engine, source, quantity, value, frame="world", fallback=None, calibrated=None):
    o = make_observation(
        ObservationDraft(task="detection", quantity=quantity, source_id=source, coordinate_system=frame,
                         predicted={"value": list(value)}, confidence=0.8, image_region=(10, 10, 50, 50)),
        engine_id=engine, model_version="v1", execution_config={}, fallback=fallback,
        processed_at="2026-01-01T00:00:00+00:00",
    )
    ctx = EvidenceContext(
        o.observation_id, run_id="run-1",
        calibrated_confidence=calibrated, calibration_basis="synthetic fixture" if calibrated is not None else None,
    )
    m.add(o, ctx)
    return o.observation_id


def _entity(m, ids, labels=("chair",) * 3):
    cands = [Candidate(i, f"img{k}", labels[k % len(labels)]) for k, i in enumerate(ids)]
    pairs = [PairScore(ids[a], ids[b], 0.9) for a in range(len(ids)) for b in range(a + 1, len(ids))]
    res = associate(cands, pairs)
    assert len(res.entities) == 1
    return res.entities[0], res, {c.observation_id: c.label for c in cands}


def _chair_matrix():
    m = EvidenceMatrix()
    p1 = _add(m, "owlvit", "img0", POSITION, (2.00, 1.00, 0.0))
    p2 = _add(m, "mask_rcnn", "img1", POSITION, (2.02, 1.01, 0.0))
    p3 = _add(m, "depth_anything", "img2", POSITION, (4.0, 3.0, 0.0))  # a wild depth estimate
    return m, (p1, p2, p3)


def test_position_support_outlier_and_uncertainty_are_explained():
    m, (p1, p2, p3) = _chair_matrix()
    ent, res, labels = _entity(m, [p1, p2, p3])
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels=labels,
                                registry=default_registry(), association=res)
    exp = explain(ce, m)
    pos = exp[POSITION]
    assert pos["status"] == "resolved"
    assert {r["observation_id"] for r in pos["supported_by"]} == {p1, p2}
    assert [e["observation_id"] for e in pos["excluded"]] == [p3]  # the disagreeing raw row is kept visible, with its reason
    assert abs(pos["value"][0] - 2.01) < 0.02
    assert any("not verified" in u for u in exp["uncertain"])
    assert any("uncalibrated" in u for u in exp["uncertain"])
    assert exp["ontology_path"] == "furniture/seating/chair" and exp["class_status"] == "agreed_unverified"
    # the raw observation behind every excluded row is retrievable from the matrix untouched
    assert m.store.get(p3).predicted == {"value": [4.0, 3.0, 0.0]}


def test_dimensions_come_from_named_measurements_and_scale_is_applied_through_anchored_frame():
    m = EvidenceMatrix()
    g = FrameGraph()
    g.add_frame(Frame("world", scale_state="metric"))
    g.add_frame(Frame("model", scale_state="relative"))
    g.add_transform("model", "world", Sim3(2.0, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0)),
                    source="tape-measure scale anchor (synthetic)", scale_anchored=True)
    d1 = _add(m, "owlvit", "img0", DIMENSIONS, (0.45, 0.48, 0.90))
    d2 = _add(m, "sam", "img1", DIMENSIONS, (0.225, 0.24, 0.45), frame="model")  # relative units; x2 -> metres
    ent, res, labels = _entity(m, [d1, d2], labels=("chair",))
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels=labels, graph=g,
                                registry=default_registry(), quantities=(DIMENSIONS,))
    dims = explain(ce, m)[DIMENSIONS]
    assert dims["status"] == "resolved"
    assert {r["observation_id"] for r in dims["supported_by"]} == {d1, d2}
    assert np.allclose(dims["value"], [0.45, 0.48, 0.90], atol=0.01)


def test_relative_frame_without_scale_anchor_is_excluded_not_averaged():
    m = EvidenceMatrix()
    g = FrameGraph()
    g.add_frame(Frame("world", scale_state="metric"))
    g.add_frame(Frame("model", scale_state="relative"))
    g.add_transform("model", "world", Sim3(1.0, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0)), source="registration")
    d1 = _add(m, "owlvit", "img0", DIMENSIONS, (0.45, 0.48, 0.90))
    d2 = _add(m, "sam", "img1", DIMENSIONS, (0.225, 0.24, 0.45), frame="model")
    ent, res, labels = _entity(m, [d1, d2], labels=("chair",))
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels=labels, graph=g,
                                registry=default_registry(), quantities=(DIMENSIONS,))
    d = ce.determinations[DIMENSIONS]
    assert d.supporting == (d1,)
    assert d2 in {i for i, _ in d.excluded} and "scale" in dict(d.excluded)[d2]


def test_frame_mismatch_without_graph_is_excluded_with_reason():
    m = EvidenceMatrix()
    d1 = _add(m, "owlvit", "img0", POSITION, (1.0, 1.0, 0.0))
    d2 = _add(m, "sam", "img1", POSITION, (9.0, 9.0, 9.0), frame="cam3")
    ent, _, labels = _entity(m, [d1, d2], labels=("chair",))
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels=labels,
                                quantities=(POSITION,))
    assert dict(ce.determinations[POSITION].excluded)[d2].startswith("frame 'cam3' differs")


def test_unmeasured_dimension_and_single_source_are_reported():
    m = EvidenceMatrix()
    p1 = _add(m, "owlvit", "img0", POSITION, (1.0, 1.0, 0.0))
    ent, res, labels = _entity(m, [p1], labels=("chair",))
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels=labels, registry=default_registry())
    exp = explain(ce, m)
    assert exp[DIMENSIONS]["status"] == "unmeasured" and exp[DIMENSIONS]["supported_by"] == []
    assert any(u.startswith("dimensions: not measured") for u in exp["uncertain"])
    assert any("single observation" in u for u in exp["uncertain"])


def test_rejected_row_is_excluded_from_fusion_but_listed():
    m, (p1, p2, p3) = _chair_matrix()
    m.set_status(p3, "rejected", reason="annotator: depth artefact", by="annotator-1")
    ent, res, labels = _entity(m, [p1, p2, p3])
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels=labels,
                                registry=default_registry(), quantities=(POSITION,))
    d = ce.determinations[POSITION]
    assert dict(d.excluded)[p3] == "rejected in the evidence matrix"
    assert set(d.supporting) == {p1, p2}


def test_dimension_outside_ontology_typical_range_is_flagged_but_value_kept():
    m = EvidenceMatrix()
    d1 = _add(m, "owlvit", "img0", DIMENSIONS, (1.50, 0.48, 0.90))  # a 1.5 m wide "chair"
    d2 = _add(m, "sam", "img1", DIMENSIONS, (1.51, 0.48, 0.90))
    ent, res, labels = _entity(m, [d1, d2], labels=("chair",))
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels=labels,
                                registry=default_registry(), quantities=(DIMENSIONS,))
    exp = explain(ce, m)
    assert abs(exp[DIMENSIONS]["value"][0] - 1.505) < 0.01  # the measurement is not clamped to the prior
    assert any("outside the ontology's typical range" in u and "dimensions.w" in u for u in exp["uncertain"])


def test_class_disagreement_is_kept_not_voted():
    m = EvidenceMatrix()
    a = _add(m, "owlvit", "img0", POSITION, (1.0, 1.0, 0.0))
    b = _add(m, "mask_rcnn", "img1", POSITION, (1.0, 1.0, 0.0))
    ent, res, _ = _entity(m, [a, b], labels=("chair",))
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels={a: "chair", b: "sofa"},
                                registry=default_registry(), quantities=(POSITION,))
    assert ce.class_status == "conflict" and ce.ontology_path == "fallback/unknown_object"
    assert set(ce.class_candidates) == {"furniture/seating/chair", "furniture/seating/sofa"}
    assert any(u.startswith("class: conflict") for u in explain(ce, m)["uncertain"])


def test_fallback_run_is_surfaced():
    m = EvidenceMatrix()
    a = _add(m, "owlvit", "img0", POSITION, (1.0, 1.0, 0.0), fallback={"from": "gpu", "to": "cpu", "why": "oom"})
    ent, _, labels = _entity(m, [a], labels=("chair",))
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels=labels, quantities=(POSITION,))
    assert any("fallback run" in u for u in explain(ce, m)["uncertain"])


def test_render_explanation_is_human_readable():
    m, (p1, p2, p3) = _chair_matrix()
    ent, res, labels = _entity(m, [p1, p2, p3])
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances=TOL, labels=labels,
                                registry=default_registry(), association=res)
    text = render_explanation(explain(ce, m))
    assert "supported by" in text and "excluded" in text and "uncertain:" in text and p1 in text
