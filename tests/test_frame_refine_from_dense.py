"""Dense monocular depth is tilted against the sparse-derived frame; up is re-fitted to the dense cloud's own planes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from engine.pipeline.stages.base import StageContext
from engine.pipeline.stages.s06_depth import refine_frame_from_dense
from engine.pipeline.vertical_slice import FrameOutcome
from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult
from reconstruction.frame import (
    FrameCanonicalization,
    refine_up_from_planes,
    rotate_in_place,
)

IDENTITY = tuple(tuple(float(v) for v in row) for row in np.eye(3))


def _tilt(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])   # about z


def _plane(normal, n):
    return SimpleNamespace(normal=tuple(normal), inlier_count=n)


def test_up_is_refitted_to_the_tilted_floor_and_ceiling_normals():
    R = _tilt(14.0)
    planes = [_plane(R @ [0, 1, 0], 900), _plane(R @ [0, -1, 0], 700),          # floor and ceiling, tilted 14 degrees
              _plane([1, 0, 0], 800), _plane([0, 0, 1], 600)]                     # walls are ignored
    found = refine_up_from_planes(planes)
    assert found is not None
    R_fix, angle = found
    assert abs(angle - 14.0) < 0.5
    assert np.allclose(R_fix @ (R @ [0, 1, 0]), [0, 1, 0], atol=0.01)            # the tilted floor normal is +Y again


def test_noise_level_and_implausible_corrections_are_refused():
    assert refine_up_from_planes([_plane([0.0, 1.0, 0.0], 500), _plane(_tilt(0.5) @ [0, 1, 0], 400)]) is None   # < 1.5 deg
    assert refine_up_from_planes([_plane(_tilt(50.0) @ [0, 1, 0], 900)]) is None    # beyond the plane tolerance: not a floor
    assert refine_up_from_planes([_plane([1, 0, 0], 900)]) is None                  # walls only: no evidence for up


def test_rotate_in_place_keeps_the_list_objects_callers_hold():
    pts = [ReconstructedPoint(position=(1.0, 0.0, 0.0), track_id="a", source_evidence_ids=("e",))]
    cams = [ReconstructedCameraPose(evidence_id="e", position=(0.0, 1.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0))]
    result = ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")
    rotate_in_place(result, _tilt(90.0))
    assert result.points is pts and result.camera_poses is cams
    assert np.allclose(pts[0].position, [0.0, 1.0, 0.0], atol=1e-9) and np.allclose(cams[0].position, [-1.0, 0.0, 0.0], atol=1e-9)


def _tilted_room(deg):
    R = _tilt(deg)
    pts, k = [], 0
    for y in (0.0, 2.5):                                                           # floor and ceiling, a 4 x 4 grid each
        for x in np.arange(0.0, 4.01, 0.1):
            for z in np.arange(0.0, 4.01, 0.1):
                pts.append(ReconstructedPoint(position=tuple(R @ [x, y, z]), track_id=f"t{k:05d}",
                                              source_evidence_ids=("ev0",)))
                k += 1
    cams = [ReconstructedCameraPose(evidence_id=f"ev{i}", position=tuple(R @ [1.0 + i, 1.2, 1.0]), rotation=(1.0, 0.0, 0.0, 0.0))
            for i in range(3)]
    return ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")


def test_stage_helper_levels_the_cloud_and_composes_the_frame_record(tmp_path):
    result = _tilted_room(14.0)
    record = FrameCanonicalization(rotation=IDENTITY, up_source="dominant_plane", source_plane_id="p", source_inliers=1, note="n")
    ctx = StageContext(job_id="j", workdir=tmp_path, artifacts={}, params={}, state={
        "frame": FrameOutcome(result=None, frame_record=record, unit_k=1.0, unit_facts=None, scale_note="")})

    facts = refine_frame_from_dense(ctx, result)

    assert facts is not None and abs(facts["correction_deg"] - 14.0) < 1.0
    ys = np.array([p.position[1] for p in result.points])
    assert set(np.round(ys[ys < 1.0], 1)) == {0.0} and set(np.round(ys[ys > 1.0], 1)) == {2.5}     # floor and ceiling flat again
    new = ctx.state["frame"].frame_record
    assert new.up_source == "dominant_plane+dense_planes" and "re-fitted" in new.note
    assert abs(np.degrees(np.arccos((np.trace(np.array(new.rotation)) - 1) / 2)) - 14.0) < 1.0   # composed rotation = the fix


def test_an_already_level_cloud_is_left_alone(tmp_path):
    result = _tilted_room(0.0)
    record = FrameCanonicalization(rotation=IDENTITY, up_source="dominant_plane", source_plane_id="p", source_inliers=1, note="n")
    ctx = StageContext(job_id="j", workdir=tmp_path, artifacts={}, params={}, state={
        "frame": FrameOutcome(result=None, frame_record=record, unit_k=1.0, unit_facts=None, scale_note="")})
    before = [p.position for p in result.points]
    assert refine_frame_from_dense(ctx, result) is None
    assert [p.position for p in result.points] == before and ctx.state["frame"].frame_record is record
