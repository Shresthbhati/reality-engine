"""spatial_intelligence in the production stage: scale contract, frame transform, engine selection, evidence written back.
Synthetic geometry (proves the integration rules, not any model's accuracy)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from apps.api import evidence_world_stage as ew
from engine.math import Quat, Vec3
from engine.multiengine.metric3d import ScaleState
from engine.multiengine.spatial_intelligence.integration import SpatialSceneResult
from engine.multiengine.spatial_stage import R_UP, _quat_to_R, _R_to_quat, to_working_camera
from tests.test_lift_stage_e2e import DX, F, H, W, _camera, _chair, _Depth, _inputs, _per_view, _points, photos  # noqa: F401
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


def _result(state, units="COLMAP_SFM_UNITS", metric=False):
    return SpatialSceneResult((), (), {}, (), (), {}, scale_units=units, is_physically_metric=metric, scale_state=state)


@pytest.mark.parametrize("state,units,metric,expect", [
    ("SFM_ALIGNED_UNMETRIC", "COLMAP_SFM_UNITS", False, (ScaleState.SFM_GAUGE, "sfm_units")),
    ("SFM_GAUGE", "COLMAP_SFM_UNITS", False, (ScaleState.SFM_GAUGE, "sfm_units")),
    ("RELATIVE_DEPTH", "RELATIVE", False, (ScaleState.RELATIVE_DEPTH, "relative")),
    ("SYNTHETIC_SCALE_ALIGNED", "SYNTHETIC_METRES", False, (ScaleState.SYNTHETIC_SCALE_ALIGNED, "synthetic_m")),
    ("PHYSICALLY_ANCHORED", "METRES", True, (ScaleState.METRIC_ANCHORED, "m")),
    ("SOMETHING_ELSE", "UNSPECIFIED", False, (ScaleState.UNKNOWN, "unknown")),
])
def test_the_bridge_maps_the_real_scale_state_and_never_calls_gauge_or_relative_synthetic(state, units, metric, expect):
    assert _result(state, units, metric).metric3d_scale() == expect


def test_a_gauge_result_emits_gauge_metric3d_records_not_synthetic_metres():
    from engine.multiengine.spatial_intelligence.association import PhysicalMergedEntity

    ent = PhysicalMergedEntity("e1", "chair", ("o1",), ("s",), (0, 0, 1), (1, 1, 1), (1, 0, 0, 0), None, 0.9, 1.0, 0.01, False,
                               scale_state="SFM_ALIGNED_UNMETRIC", scale_units="COLMAP_SFM_UNITS")
    res = SpatialSceneResult((ent,), (), {}, (), (), {}, scale_units="COLMAP_SFM_UNITS", scale_state="SFM_ALIGNED_UNMETRIC")
    recs = res.to_metric3d_observations()
    assert recs and all(r.scale_state is ScaleState.SFM_GAUGE and r.units == "sfm_units" for r in recs)
    # NOTE for the package owner: these records carry no raw_artifact_ref, so metric3d.v1 would refuse them at ingest; the production
    # stage writes its own contract-valid records (with the raw artifact) instead of using this method.


def test_quaternion_matrix_round_trip():
    q = (0.5, 0.5, -0.5, 0.5)
    assert np.allclose(_quat_to_R(_R_to_quat(_quat_to_R(q))), _quat_to_R(q))


def test_the_working_camera_sees_the_same_pixels_as_the_job_camera():
    cam = PinholeCamera(CameraIntrinsics(fx=F, fy=F, cx=W / 2, cy=H / 2, width=W, height=H),
                        CameraExtrinsics(Vec3(0.4, -0.2, 0.7), Quat(*_R_to_quat(_quat_to_R((0.9, 0.1, 0.3, 0.2))))))
    work = to_working_camera(cam)
    for p in ([0.5, 0.1, 3.0], [1.0, -0.3, 4.0], [0.2, 0.2, 2.5]):
        a = cam.project(Vec3(*p))
        b = work.project(Vec3(*(R_UP @ np.array(p))))  # the same world point, expressed in the Z-up working frame
        assert a is not None and b is not None and np.allclose(a, b, atol=1e-6)
    assert np.linalg.det(R_UP) == pytest.approx(1.0) and np.allclose(R_UP @ np.array([0, 1, 0]), [0, 0, 1])  # job +Y becomes working +Z


def _run(tmp_path, photo_paths):
    cams = {p: _camera(DX[os.path.basename(p)]) for p in photo_paths}
    return ew.run_for_job(team_id="t1", world_id="w1", job_id="job-1", sources=photo_paths, base=tmp_path / "ws",
                          flow_inputs=_inputs(_Depth(_per_view())), depth_engines=["fake_depth"], cameras=cams, points3d=_points())


def test_the_production_stage_uses_spatial_intelligence_and_writes_its_results_into_the_evidence(tmp_path, photos, monkeypatch):  # noqa: F811
    monkeypatch.delenv(ew.ENV_LIFT, raising=False)
    r = _run(tmp_path, photos)
    lift = r["lifting"]
    assert lift["engine"] == "spatial_intelligence.reconstruct_spatial_scene" and lift["lifted"] == 2
    assert lift["merged_entities"] == 1 and lift["multiview_entities"] == 1 and "spatial_error" not in lift
    assert lift["max_ray_distance"] > 0 and "median aligned scene depth" in lift["ray_threshold_basis"]
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    root = tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-1"
    m = EvidenceMatrix(root)
    pos = [o for o in m.store.query(quantity="position") if o.engine_id == "lift:spatial_intelligence"]
    assert len(pos) == 2 and all(o.predicted["scale_state"] == "sfm_gauge" and o.uncertainty and o.uncertainty["covariance"] for o in pos)  # covariance kept
    support = m.artifacts.get(m.context(pos[0].observation_id).artifact_refs["raw_output"])
    assert b"lifted_working_frame" in support and b"oriented_extents_w_d_h_working" in support  # orientation / normal / oriented extents stay on the record
    # one physical object seen in two photographs is ONE entity, merged by the package's triangulation (its basis is in the decision)
    snap = json.loads((root / "snapshot.json").read_text())
    assert snap["counts"]["entities"] == 1 and any("ray triangulation" in str(d["basis"]) for d in snap["associations"] if d["outcome"] == "merged")


def test_depth_fusion_runs_in_the_production_stage_and_any_surface_it_records_is_a_real_stored_artifact(tmp_path, photos, monkeypatch):  # noqa: F811
    import hashlib

    monkeypatch.delenv(ew.ENV_LIFT, raising=False)
    f = _run(tmp_path, photos)["lifting"]["fusion"]
    assert f["status"] == "ran", f  # two registered views with aligned depth: fusion executed (synthetic depth: proves wiring, not accuracy)
    assert f["scale_state"] == "SFM_GAUGE" and f["physically_metric"] is False and f["frame"].startswith("job")
    assert {p["source"] for p in f["per_view"]} == set(photos) and all("contributed_to_surface" in p for p in f["per_view"])
    if "mesh_artifact" in f:  # a recorded artifact reference must resolve to exactly the bytes that were fused
        from engine.multiengine.evidence_matrix import EvidenceMatrix

        m = EvidenceMatrix(tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-1")
        data = m.artifacts.get(f["mesh_artifact"]["uri"])
        assert hashlib.sha256(data).hexdigest() == f["mesh_artifact"]["sha256"] and data[:3] == b"ply" and len(data) == f["mesh_artifact"]["bytes"]
    else:
        assert f.get("reason")  # no surface: the reason is on the record, and no artifact is claimed


def test_the_legacy_lifter_remains_selectable_and_says_so(tmp_path, photos, monkeypatch):  # noqa: F811
    monkeypatch.setenv(ew.ENV_LIFT, "legacy")
    assert _run(tmp_path, photos)["lifting"]["engine"] == "legacy box lifter"


def _rel(a, b, kind, conf=0.8):
    from engine.multiengine.spatial_intelligence.spatial_lifting import SpatialRelationKind, SpatialRelationship

    return SpatialRelationship(a, b, SpatialRelationKind(kind), conf, 0.5, "test basis")


def test_relations_keep_only_structural_kinds_and_never_form_a_support_cycle():
    from types import SimpleNamespace as NS

    from engine.multiengine.spatial_stage import make_scene_relations_fn
    from world_ir.schema_v1 import RelationshipKind

    cans = [NS(entity_id=e, members=(f"d{e}",)) for e in "ABC"]
    state = {"relationships": [_rel("dA", "dB", "adjacent_to"), _rel("dA", "dB", "facing"),            # fixed-metre kinds: dropped
                               _rel("dA", "dB", "supported_by", 0.9),                                  # A rests on B
                               _rel("dB", "dC", "supported_by", 0.8),                                  # B rests on C
                               _rel("dC", "dA", "supported_by", 0.7),                                  # C rests on A: would close a cycle
                               _rel("dA", "dC", "mounted_on", 0.6)]}
    out = make_scene_relations_fn(state)(cans, None)
    kinds = {(r.source_entity_id, r.kind.value, r.target_entity_id) for r in out}
    assert kinds == {("A", "supported_by", "B"), ("B", "supported_by", "C"), ("A", "mounted_on", "C")}
    assert not any(r.kind is RelationshipKind.ADJACENT_TO for r in out) and all("spatial_intelligence relation" in r.basis for r in out)


def test_gauge_unit_relations_state_their_real_units_and_never_crash_on_a_missing_metre_distance():
    # REGRESSION (found by the real office run): distance_m is None for sfm_units / synthetic_m, and the bridge formatted it unconditionally
    from types import SimpleNamespace as NS

    from engine.multiengine.spatial_intelligence.spatial_lifting import SpatialRelationKind, SpatialRelationship
    from engine.multiengine.spatial_stage import make_scene_relations_fn

    cans = [NS(entity_id=e, members=(f"d{e}",)) for e in "AB"]
    for units, expected in (("sfm_units", "SfM gauge units"), ("synthetic_m", "synthetic metres")):
        rel = SpatialRelationship("dA", "dB", SpatialRelationKind("supported_by"), 0.9, 0.25, "test basis", units)
        assert rel.distance_m is None
        (out,) = make_scene_relations_fn({"relationships": [rel]})(cans, None)
        assert expected in out.basis and "0.25" in out.basis


def test_persistence_drops_an_impossible_support_cycle_keeps_both_entities_and_records_it():
    from provenance import Provenance
    from world_ir.schema_v1 import Relationship, RelationshipKind
    from world_ir.validation import validate_world_ir
    from world_ir.world_v1 import WorldIR

    w = WorldIR(id="t", name="t")
    for i, (lo, hi) in enumerate([((0, 0, 0), (1, 1, 1)), ((5, 5, 5), (6, 6, 6))]):
        e, g = _chair(i, lo, hi, 1)
        w.entities[e.id], w.geometries[g.id] = e, g
    for a, b in (("ent-0", "ent-1"), ("ent-1", "ent-0")):
        w.entities[a].relationships.append(Relationship(RelationshipKind.SUPPORTED_BY, b, confidence=None, provenance=Provenance.INFERRED))
    if not any("cyclic_support" in str(x) for x in validate_world_ir(w).errors):
        pytest.skip("this schema version does not flag the cycle")
    ew._withhold_unresolved_duplicates(w)
    assert set(w.entities) == {"ent-0", "ent-1"} and validate_world_ir(w).errors == []
    assert all(not r.kind.value.startswith("support") for e in w.entities.values() for r in e.relationships)
    assert w.metadata["withheld_relationships"][0]["entities"] == ["ent-0", "ent-1"] or w.metadata["withheld_relationships"][0]["entities"] == ["ent-1", "ent-0"]


def test_dimension_uncertainty_is_bootstrapped_from_the_detections_own_points_and_withheld_when_there_are_too_few():
    """Synthetic points with a known extent: the estimator recovers it, its covariance is positive semi-definite and shrinks with more points,
    and fewer than MIN_EXTENT_POINTS yields None (unknown), never a stand-in value."""
    from engine.multiengine import spatial_stage as ss

    rng = np.random.default_rng(1)

    def box(n):  # a 2 x 1 x 0.5 box of points
        return rng.uniform([0, 0, 0], [2.0, 1.0, 0.5], (n, 3))

    ext, cov = ss._extent_covariance(box(2000), seed=1)
    assert ext == pytest.approx([1.8, 0.9, 0.45], abs=0.06)  # 5-95 percentile of a uniform edge: 90 % of the side
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-12) and np.all(np.diag(cov) > 0)
    _, cov_small = ss._extent_covariance(box(400), seed=1)
    assert np.trace(cov_small) > np.trace(cov)  # more points, tighter estimator
    assert ss._extent_covariance(box(ss.MIN_EXTENT_POINTS - 1), seed=1) is None


def _flat_camera():
    from engine.math import Quat, Vec3
    from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

    return PinholeCamera(CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100), CameraExtrinsics(Vec3(0, 0, 0), Quat(1, 0, 0, 0)))


def _det(box, mask=None):
    from types import SimpleNamespace

    return SimpleNamespace(box_2d=box, mask_2d=mask)


def test_detection_points_respect_the_mask_drop_depth_outliers_and_refuse_sparse_or_empty_sets():
    from engine.multiengine import spatial_stage as ss

    cam = _flat_camera()
    depth = np.full((100, 100), 2.0, np.float32)
    box = (10, 10, 90, 90)
    full = ss._detection_points_job(_det(box), depth, cam, max_points=10**6)
    half = np.zeros((100, 100), bool)
    half[:, :50] = True
    left = ss._detection_points_job(_det(box, half), depth, cam, max_points=10**6)  # a mask restricts the points to ITS instance, not the whole box
    assert len(left) < len(full) and left[:, 0].max() < full[:, 0].max()
    spiky = depth.copy()
    spiky[20:24, 20:24] = 9.0  # stray far depths (mask edge / flying pixels)
    with_spikes = ss._detection_points_job(_det(box), spiky, cam, max_points=10**6)
    assert len(with_spikes) == len(full) - 16  # robust filter drops exactly the outliers
    empty_mask = np.zeros((100, 100), bool)
    assert len(ss._detection_points_job(_det(box, empty_mask), depth, cam)) == 0
    nan_depth = np.full((100, 100), np.nan, np.float32)
    nan_depth[40:42, 40:42] = 2.0
    sparse = ss._detection_points_job(_det(box), nan_depth, cam)
    assert len(sparse) == 4 and ss._extent_covariance(sparse, seed=0) is None  # occluded / invalid depth: too few points, so no extent
    assert len(ss._detection_points_job(_det((200, 200, 300, 300)), depth, cam)) == 0  # a box entirely outside the depth map


def test_a_flat_or_single_depth_detection_gives_a_finite_psd_covariance_not_nan():
    from engine.multiengine import spatial_stage as ss

    flat = np.column_stack([np.random.default_rng(2).uniform(0, 1, 500), np.random.default_rng(3).uniform(0, 1, 500), np.full(500, 2.0)])
    ext, cov = ss._extent_covariance(flat, seed=0)
    assert ext[2] == pytest.approx(0.0, abs=1e-12) and np.isfinite(cov).all() and np.all(np.linalg.eigvalsh(cov) >= -1e-12)


def test_an_invalid_covariance_is_refused_by_the_observation_contract():
    from engine.multiengine.metric3d import Metric3DObservation, ScaleState

    base = dict(quantity="dimensions", value=(1.0, 1.0, 1.0), engine_id="e", architecture_id="a", checkpoint_id="c", source_id="s", units="sfm_units",
                scale_state=ScaleState.SFM_GAUGE, frame="world", camera_id="s")
    for bad in (((1, 0, 0), (0, 1, 0), (0, 0, float("nan"))), ((1, 0, 0), (0, -1, 0), (0, 0, 1)), ((1, 2, 0), (0, 1, 0), (0, 0, 1))):
        assert Metric3DObservation(covariance=bad, **base).problems(), bad
