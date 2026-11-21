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
from tests.test_lift_stage_e2e import DX, F, H, W, _camera, _Depth, _inputs, _per_view, _points, photos  # noqa: F401
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

    ent = PhysicalMergedEntity("e1", "chair", ("o1",), ("s",), (0, 0, 1), (1, 1, 1), (1, 0, 0, 0), None, 0.9, 1.0, 0.01, False)
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


def test_the_legacy_lifter_remains_selectable_and_says_so(tmp_path, photos, monkeypatch):  # noqa: F811
    monkeypatch.setenv(ew.ENV_LIFT, "legacy")
    assert _run(tmp_path, photos)["lifting"]["engine"] == "legacy box lifter"
