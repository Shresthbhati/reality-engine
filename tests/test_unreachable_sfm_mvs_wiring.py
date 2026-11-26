"""Tests verifying that reconstruction/sfm/core_pipeline.py and
reconstruction/mvs/dense_cloud.py are reachable and properly wired
(RE-GEO-000014 and RE-GEO-000015).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.math.math3 import Quat, Vec3
from evidence.session import EvidenceItem, EvidenceKind
from perception.depth.interface import DepthMap
from reconstruction.backend.interface import IReconstructionBackend, ReconstructedCameraPose
from reconstruction.backend.python_backend import (
    PythonSfMReconstructionBackend,
    _compose_camera_to_world,
    _projection_matrix,
    _quat_from_R,
)
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from reconstruction.dense_pipeline import run_multiview_dense_cloud
from reconstruction.sfm.core_pipeline import CameraModel, TriangulationEngine


def test_python_sfm_backend_is_reachable_and_implements_interface():
    backend = PythonSfMReconstructionBackend()
    assert isinstance(backend, IReconstructionBackend)

    ev1 = EvidenceItem(id="img1", source_uri="file:///img1.jpg", kind=EvidenceKind.PHOTO)
    ev2 = EvidenceItem(id="img2", source_uri="file:///img2.jpg", kind=EvidenceKind.PHOTO)

    result = backend.reconstruct([ev1, ev2])
    assert result.registration_status == "partial"
    assert len(result.camera_poses) == 2
    assert result.camera_poses[0].evidence_id == "img1"
    assert result.camera_poses[1].evidence_id == "img2"
    assert result.registration_stats["n_good_pairs"] == 0


def test_python_sfm_triangulation_uses_paired_camera_to_world_poses():
    camera = CameraModel(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)

    def rotation_y(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    R0 = np.eye(3)
    C0 = np.zeros(3)
    R_rel1 = rotation_y(0.06)
    t_rel1 = np.array([-0.5, 0.0, 0.0])
    R1, C1 = _compose_camera_to_world(R0, C0, R_rel1, t_rel1)

    R_rel2 = rotation_y(-0.04)
    C2_expected = np.array([0.95, 0.04, 0.02])
    t_rel2 = -R_rel2 @ R1.T @ (C2_expected - C1)
    R2, C2 = _compose_camera_to_world(R1, C1, R_rel2, t_rel2)

    pose1 = ReconstructedCameraPose("cam1", tuple(C1), _quat_from_R(R1))
    pose2 = ReconstructedCameraPose("cam2", tuple(C2), _quat_from_R(R2))
    point_world = C1 + R1 @ np.array([0.15, -0.1, 4.0])
    P1 = _projection_matrix(camera, pose1)
    P2 = _projection_matrix(camera, pose2)

    def project(P: np.ndarray) -> np.ndarray:
        pixel_h = P @ np.append(point_world, 1.0)
        return pixel_h[:2] / pixel_h[2]

    recovered = TriangulationEngine.triangulate_dlt_point(
        P1, P2, project(P1), project(P2)
    )

    assert tuple(C2) == pytest.approx(tuple(C2_expected))
    assert recovered is not None
    assert np.allclose(recovered, point_world, atol=1e-8)


def test_multiview_dense_cloud_pipeline_is_reachable():
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=160.0, cy=120.0, width=320, height=240)
    cam1 = PinholeCamera(
        intrinsics=intrinsics,
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )
    cam2 = PinholeCamera(
        intrinsics=intrinsics,
        extrinsics=CameraExtrinsics(position=Vec3(0.3, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )

    depth1_arr = np.full((240, 320), 2.5, dtype=np.float64)
    depth2_arr = np.full((240, 320), 2.5, dtype=np.float64)

    d1 = DepthMap(evidence_id="cam1", width=320, height=240, values=depth1_arr.tolist(), unit="meters")
    d2 = DepthMap(evidence_id="cam2", width=320, height=240, values=depth2_arr.tolist(), unit="meters")

    cloud_res = run_multiview_dense_cloud(
        cameras={"cam1": cam1, "cam2": cam2},
        depth_maps={"cam1": d1, "cam2": d2},
        stride=16,
        min_consistent_views=1,
        voxel_size_m=0.05,
    )
    assert cloud_res.point_count > 0
    assert cloud_res.points.shape[1] == 3
