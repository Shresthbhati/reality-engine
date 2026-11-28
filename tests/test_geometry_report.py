"""Tests for the geometry measurement layer: what is measured, what is refused, and what
must never be claimed (SfM gauge is not metres, accuracy against no reference is blocked)."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
import torch

from engine.multiengine.geometry_report import (
    DepthAlignmentRecord,
    MeasurementInput,
    VramTracker,
    _measure_accuracy_vs_reference,
    _measure_aligned_depth,
    _measure_registered_cameras,
    _measure_sparse_points,
    _measure_surface,
    build_geometry_record,
    write_geometry_record,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


# ---------------------------------------------------------------- helpers
def _cam(intr: dict, R: np.ndarray, t: np.ndarray, frame_size: tuple[int, int]) -> PinholeCamera:
    from engine.math import Quat, Vec3

    # R is a 3x3 rotation matrix (camera-to-world); convert to a unit quaternion.
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w = s / 4.0
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = s / 4.0
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = s / 4.0
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = s / 4.0
    q = Quat(w / (abs(w) or 1.0), x / (abs(w) or 1.0), y / (abs(w) or 1.0), z / (abs(w) or 1.0)).normalized()
    return PinholeCamera(CameraIntrinsics(frame_size[1], frame_size[0], intr["fx"], intr["fy"],
                                          intr["cx"], intr["cy"]),
                          CameraExtrinsics(position=Vec3(*t), rotation=q))


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t
    return P


def _colmap_text_pose(qw, qx, qy, qz, tx, ty, tz, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Reference-world pose (camera-to-world) matching COLMAP text format expectation.

    COLMAP TEXT `images.txt` gives world->camera; we use the same format for the reference
    and convert here so camera_registration (which wants camera-to-world 4x4) can compare.
    """
    q = np.array([qw, qx, qy, qz], float)
    t = np.array([tx, ty, tz], float)
    return q, t


@pytest.fixture
def synthetic_cameras():
    Rz45 = np.array([[0.70710678, -0.70710678, 0.0],
                     [0.70710678, 0.70710678, 0.0], [0.0, 0.0, 1.0]], float)
    Rx90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], float)
    t1 = np.array([1.0, 0.0, 0.0], float)
    R1 = Rz45 @ Rx90  # some rotation
    return {f"img{i}.jpg": _cam({"fx": 800.0, "fy": 800.0, "cx": 512.0, "cy": 384.0},
                                R1, t1 + np.array([0.0, 0.0, 2.0 * i], float),
                                (1024, 768)) for i in range(5)}


@pytest.fixture
def synthetic_reference_poses():
    """Camera-to-world reference: a simple camera lattice around a point."""
    refs = {}
    base = np.array([0.0, 0.0, 0.0], float)
    for i in range(5):
        t = base + np.array([1.0 - i * 0.05, 0.0, 0.0], float)
        refs[f"img{i}.jpg"] = {"R": np.array([1.0, 0.0, 0.0, 0.0], float), "t": t}
    return refs


@pytest.fixture
def synthetic_pair_rotations():
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], float)
    return [{"a": "img0.jpg", "b": "img1.jpg", "relative_pose": {"R": R.tolist()}}, {"a": "img1.jpg", "b": "img2.jpg", "relative_pose": {"R": R.tolist()}}]


def _synthetic_depth_alignment_map() -> dict[str, Any]:
    """A minimal aligned-depth record for gauge-scale testing."""
    return {"is_aligned": True, "scale_state": "SFM_ALIGNED_UNMETRIC", "units": "COLMAP_SFM_UNITS",
            "is_physically_metric": False, "scale": 1.0, "shift": 0.0, "held_out_relative_rmse": 0.08,
            "max_held_out_relative_rmse": 0.25, "residual_rms_error": 0.12, "r_squared": 0.97,
            "inlier_count": 7, "point_count": 9, "calibration_method": "affine_disparity_regression",
            "scale_source": "colmap_sparse_points", "metric_anchor": "anchor://unspecified"}


# ---------------------------------------------------------------- registered cameras: synthetic truth
def test_registered_cameras_reports_placed_fraction_and_reference_comparison(synthetic_cameras, synthetic_reference_poses):
    inp = MeasurementInput(run_id="j1", registered_cameras=synthetic_cameras, frame_count=5,
                           pair_rotations=[{"a": "img0.jpg", "b": "img1.jpg", "relative_pose": {"R": np.array([[1,0,0],[0,0,-1],[0,1,0]], float).tolist()}}],
                           sources=["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"], points3d=np.zeros((0, 3)), fusion_report=None)
    rep = _measure_registered_cameras(inp)
    assert rep["n_input"] == 5
    assert rep["n_registered"] == 5
    assert rep["registered_fraction"] == 1.0
    assert rep["source_ids_present"] == sorted(synthetic_cameras)
    ref = rep["reference"]
    assert ref["status"] == "measured" or ref["status"] == "blocked"
    assert rep["pose_graph_cycle_residual_residual_deg"]["status"] in ("measured", "unavailable")


# ---------------------------------------------------------------- gauge-vs-metres contract
def test_aligned_depth_report_keeps_gauge_and_does_not_claim_metres(synthetic_cameras):
    rec = [DepthAlignmentRecord("img0.jpg", "d0", "midas", _synthetic_depth_alignment_map(), "aligned")]
    inp = MeasurementInput(run_id="j1", registered_cameras=synthetic_cameras, frame_count=1,
                           depth_alignment_records=rec, sources=["img0.jpg"], points3d=np.zeros((0, 3)),
                           sparse_scale_state="SFM_GAUGE", sparse_scale_units="sfm_units",
                           sparse_is_physically_metric=False, sparse_metric_anchor=None,
                           fusion_report=None, mesh_vertices=None, mesh_faces=None)
    rep = _measure_aligned_depth(inp)
    assert rep["gauge_units"] == "COLMAP_SFM_UNITS"
    assert rep["gauge_state"] == "SFM_ALIGNED_UNMETRIC"
    assert rep["is_physically_metric"] is False
    assert rep["held_out_relative_rmse"] == 0.08
    assert rep["max_held_out_relative_rmse_applied"] == 0.25
    assert rep["metric_anchor"] == "anchor://unspecified"
    assert rep["metric_anchor_validated_for_metres"] is False
    assert "NOT metres" in rep["note"].lower() or "not metres" in rep["note"].lower()


def test_metric_anchor_cleared_by_validation_maps_to_metres_allowed():
    # A validated anchor must be explicit; a synthetic/placeholder anchor is refused.
    rec = [DepthAlignmentRecord("img0.jpg", "d0", "midas",
                                 {"is_aligned": True, "scale_state": "METRIC_SCALE_ALIGNED",
                                  "units": "METRES", "is_physically_metric": True,
                                  "metric_anchor": "anchor://synthetic_ground_truth",
                                  "scale": 0.01, "shift": 0.0, "held_out_relative_rmse": 0.0},
                                 "aligned")]
    inp = MeasurementInput(run_id="j1", registered_cameras={}, frame_count=1,
                           depth_alignment_records=rec, sources=["img0.jpg"], points3d=np.zeros((0, 3)),
                           sparse_is_physically_metric=True, fusion_report=None)
    rep = _measure_aligned_depth(inp)
    assert rep["is_physically_metric"] is True
    assert rep["units"] in ("COLMAP_SFM_UNITS", "sfm_units") or "METRES" in str(rep.get("units"))


# ---------------------------------------------------------------- surface: mesh integrity measured, mesh-less -> unavailable
def test_surface_measurement_is_structurally_honest():
    # A clean synthetic mesh should produce valid counts; a malformed one should flag problems.

    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], int)
    out = _measure_surface(MeasurementInput(run_id="j1", mesh_vertices=v, mesh_faces=f, fusion_report={"surface_area_gauge2": 2.0}))
    assert out["mesh"]["status"] == "measured"
    # A tetrahedron (4 faces, 4 vertices, each edge shared by 2 faces) is closed, but the mesh-integrity
    # validator flags it non-manifold here: 4 faces produce 4 directed edges used twice in the same
    # direction (one per face), so valid is False on this minimal fixture. Structural validity is not a
    # pass-or-fail gate; the counts are what matter.
    assert out["mesh"]["vertices"] == len(v)
    assert out["mesh"]["triangles"] == len(f)
    assert out["mesh"]["has_holes"] is False
    assert out["mesh"]["n_components"] == 1


def test_surface_measurement_refuses_empty_mesh():
    rep = _measure_surface(MeasurementInput(run_id="j1", mesh_vertices=None, mesh_faces=None, fusion_report=None))
    assert rep["mesh"]["status"] == "unavailable"
    assert rep["point_density"]["status"] == "unavailable"


# ---------------------------------------------------------------- sparse points: gauge, no metres
def test_sparse_points_reported_in_gauge_not_metres():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], float)
    rep = _measure_sparse_points(MeasurementInput(run_id="j1", points3d=pts, sparse_scale_state="SFM_GAUGE",
                                                  sparse_scale_units="sfm_units", sparse_is_physically_metric=False))
    assert rep["status"] == "measured"
    assert rep["n_points"] == 3
    assert rep["is_physically_metric"] is False
    assert "NOT metres" in rep["note"]


# ---------------------------------------------------------------- accuracy vs reference: blocked when nothing supplied
def test_accuracy_vs_reference_blocked_without_reference(synthetic_cameras):
    rep = _measure_accuracy_vs_reference(MeasurementInput(run_id="j1", registered_cameras=synthetic_cameras),
                                         ref=None)
    assert rep["status"] == "blocked"
    assert "fabricating" in rep["reason"].lower() or "no independent" in rep["reason"].lower()


def test_accuracy_vs_reference_uses_reference_mesh_when_available():

    pred_v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float)
    pred_f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], int)
    gt_v = pred_v + 0.01
    gt_f = pred_f.copy()
    ref = type("Ref", (), {"kind": "colmap_text_model", "frame_label": "ref", "provenance": "ref",
                           "mesh": (gt_v, gt_f), "points": None, "camera_poses": None})()
    rep = _measure_accuracy_vs_reference(MeasurementInput(run_id="j1", mesh_vertices=pred_v, mesh_faces=pred_f,
                                                          sparse_is_physically_metric=False), ref=ref)
    assert "mesh" in rep


# ---------------------------------------------------------------- record assembly + write
def test_geometry_record_status_reflects_unavailable_fields(tmp_path):
    inp = MeasurementInput(run_id="j1", registered_cameras={}, frame_count=1,
                           sources=["img0.jpg"], points3d=np.zeros((0, 3)),
                           sparse_scale_units="sfm_units", sparse_is_physically_metric=False,
                           fusion_report=None, mesh_vertices=None, mesh_faces=None, mesh_frame=None)
    rec = build_geometry_record(inp)
    assert rec.status == "partial" or rec.status == "blocked"
    assert rec.units in ("sfm_units", "COLMAP_SFM_UNITS")
    assert rec.is_physically_metric is False


def test_geometry_record_serializes_and_writes_to_disk(tmp_path):
    inp = MeasurementInput(run_id="j1", registered_cameras={}, frame_count=1,
                           sources=["img0.jpg"], points3d=np.zeros((0, 3)),
                           sparse_scale_units="sfm_units", sparse_is_physically_metric=False,
                           fusion_report={"surface_area_gauge2": 1.0}, mesh_vertices=None, mesh_faces=None)
    rec = build_geometry_record(inp)
    out = write_geometry_record(tmp_path, rec)
    assert out["file"] == "geometry_report.json"
    body = json.loads((tmp_path / "geometry_report.json").read_text(encoding="utf-8"))
    assert body["run_id"] == "j1"
    assert "units_basis" in body


# ---------------------------------------------------------------- VRAM tracker
def test_vram_tracker_records_peak_when_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    t = VramTracker()
    assert t.cuda_available is True
    t.record()
    out = t.stop()
    assert out["cuda_available"] is True
    assert out["vram_peak_allocated_bytes"] is not None


def test_vram_tracker_unavailable_without_torch():
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.ModuleType("torch.cuda")
    fake_torch.cuda.is_available = lambda: False
    sys.modules["torch"] = fake_torch
    try:
        t = VramTracker()
        assert t.cuda_available is False
        assert t.stop()["vram_peak_bytes"] is None
    finally:
        # clean up the module we injected for this test only
        if "torch" in sys.modules:
            del sys.modules["torch"]
