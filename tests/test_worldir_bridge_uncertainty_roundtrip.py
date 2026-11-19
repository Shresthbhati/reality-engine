"""Evidence -> frame transform -> canonical -> WorldIR -> WorldStore -> reload: the propagated uncertainty, the unmeasured confidence
and the evidence references must all survive persistence with nothing lost or invented. SYNTHETIC observations; this proves the
plumbing and the mathematics, not any model's accuracy."""

from __future__ import annotations

import json

import numpy as np

from engine.multiengine.alignment import Frame, FrameGraph, Sim3
from engine.multiengine.association import Candidate, PairScore, associate
from engine.multiengine.canonical import DIMENSIONS, POSITION, build_canonical_entity
from engine.multiengine.evidence_matrix import EvidenceContext, EvidenceMatrix
from engine.multiengine.observation import ObservationDraft, make_observation
from engine.multiengine.registry import default_registry
from engine.multiengine.worldir_bridge import convert
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore

SCALE = 2.0
THETA = 0.5
R = np.array([[np.cos(THETA), -np.sin(THETA), 0.0], [np.sin(THETA), np.cos(THETA), 0.0], [0.0, 0.0, 1.0]])
ENGINES = ("owlvit", "mask_rcnn")  # different correlation families in the default registry
POS_COV = np.diag([0.01, 0.02, 0.03])
DIM_COV = np.diag([0.001, 0.001, 0.002])


def _graph() -> FrameGraph:
    g = FrameGraph()
    g.add_frame(Frame("cam", kind="camera", scale_state="metric"))
    g.add_frame(Frame("world", kind="world", scale_state="metric"))
    g.add_transform("cam", "world", Sim3.from_arrays(SCALE, R, np.array([0.0, 0.0, 0.0])), source="synthetic", scale_anchored=True)
    return g


def _entity(frame, with_cov, graph=None):
    m = EvidenceMatrix()
    uri, _ = m.artifacts.put(b"raw output")
    ids = []
    for k, eng in enumerate(ENGINES):
        for q, val, cov in ((POSITION, [1.0, 1.0, 0.5], POS_COV), (DIMENSIONS, [0.25, 0.25, 0.45], DIM_COV)):
            unc = {"covariance": cov.tolist()} if with_cov else None
            o = make_observation(
                ObservationDraft(task="detection", quantity=q, source_id=f"img{k}.jpg", coordinate_system=frame,
                                 predicted={"value": val}, confidence=0.8, uncertainty=unc, image_region=(1, 1, 9, 9)),
                engine_id=eng, model_version="v1", execution_config={}, processed_at="2026-01-01T00:00:00+00:00",
            )
            m.add(o, EvidenceContext(o.observation_id, run_id="run-1", architecture_id="a", checkpoint_id="c",
                                     artifact_refs={"run_output": uri}, validation="synthetic"))
            ids.append(o.observation_id)
    cands = [Candidate(i, f"image{n}", "chair") for n, i in enumerate(ids)]
    pairs = [PairScore(ids[a], ids[b], 0.9) for a in range(len(ids)) for b in range(a + 1, len(ids))]
    res = associate(cands, pairs)
    (ent,) = res.entities
    ce = build_canonical_entity(ent, m, target_frame="world", tolerances={POSITION: 5.0, DIMENSIONS: 5.0},
                                labels={i: "chair" for i in ids}, registry=default_registry(), graph=graph, association=res)
    return m, ce, uri


def test_propagated_uncertainty_reaches_worldir_and_survives_the_world_store(tmp_path):
    m, ce, _ = _entity("cam", True, _graph())
    pos = ce.determinations[POSITION]
    assert pos.status == "resolved" and pos.uncertainty is not None
    w = WorldIR(id="w-unc", name="unc")
    convert([ce], m).apply_to(w)
    (ent,) = w.entities.values()
    assert ent.confidence is None  # unmeasured stays unmeasured
    (g,) = [w.geometries[i] for i in ent.geometry_ids]
    assert g.quality_metrics["uncertainty"]["position"] is not None and g.quality_metrics["uncertainty"]["dimensions"] is not None

    store = WorldStore(tmp_path)
    v = store.save_version(w, parent=None)
    assert store.verify_version(v.version_id) == []
    back = store.load_version(v.version_id)
    assert back.to_dict() == w.to_dict()  # lossless reload
    (e2,) = back.entities.values()
    (g2,) = [back.geometries[i] for i in e2.geometry_ids]
    cov = np.asarray(g2.quality_metrics["uncertainty"]["position"]["covariance"], dtype=float)
    assert cov.shape == (3, 3) and np.isfinite(cov).all() and np.all(np.linalg.eigvalsh(0.5 * (cov + cov.T)) > 0)
    json.dumps(back.to_dict())  # the whole world stays JSON-serialisable


def test_the_reloaded_world_keeps_every_evidence_reference(tmp_path):
    m, ce, uri = _entity("world", True)
    w = WorldIR(id="w-trace", name="trace")
    convert([ce], m).apply_to(w)
    v = WorldStore(tmp_path).save_version(w, parent=None)
    back = WorldStore(tmp_path).load_version(v.version_id)
    (ent,) = back.entities.values()
    assert len(ent.observations) == 4
    assert uri in {r.uri for r in ent.external_references}  # the raw model output stays referenced


def test_without_any_covariance_no_uncertainty_is_invented():
    m, ce, _ = _entity("world", False)
    w = WorldIR(id="w2", name="n")
    convert([ce], m).apply_to(w)
    (e,) = w.entities.values()
    (g,) = [w.geometries[i] for i in e.geometry_ids]
    cov_pos = (g.quality_metrics["uncertainty"]["position"] or {}).get("covariance")
    assert cov_pos is None  # no covariance was observed, so none appears
    assert e.confidence is None
