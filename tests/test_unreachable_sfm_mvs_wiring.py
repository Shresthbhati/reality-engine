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
from reconstruction.backend.interface import IReconstructionBackend
from reconstruction.backend.python_backend import PythonSfMReconstructionBackend
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from reconstruction.dense_pipeline import run_multiview_dense_cloud


def test_python_sfm_backend_is_reachable_and_implements_interface():
    backend = PythonSfMReconstructionBackend()
    assert isinstance(backend, IReconstructionBackend)

    ev1 = EvidenceItem(id="img1", source_uri="file:///img1.jpg", kind=EvidenceKind.PHOTO)
    ev2 = EvidenceItem(id="img2", source_uri="file:///img2.jpg", kind=EvidenceKind.PHOTO)

    result = backend.reconstruct([ev1, ev2])
    assert result.registration_status == "success"
    assert len(result.camera_poses) == 2
    assert result.camera_poses[0].evidence_id == "img1"
    assert result.camera_poses[1].evidence_id == "img2"


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
