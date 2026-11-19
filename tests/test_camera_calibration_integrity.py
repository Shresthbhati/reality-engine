"""Tests for Camera Calibration Integrity (P0: Remove invented camera measurements)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from engine.multiengine.universal_fusion import project_3d_to_pixel
from engine.multiengine.visibility_reprojection import (
    ValidatedCameraMetadata,
    VisibilityReprojectionValidator,
)


def test_missing_intrinsics_returns_uncalibrated() -> None:
    res = VisibilityReprojectionValidator.validate_camera_metadata({})
    assert res.is_calibrated is False
    assert res.camera is None
    assert res.rejection_reason == "missing_intrinsics"
    assert VisibilityReprojectionValidator.camera_from_metadata({}) is None


def test_missing_focal_or_principal_point_returns_uncalibrated() -> None:
    # Missing fy
    meta_incomplete = {
        "intrinsics": {"fx": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        "is_canonical_frame": True,
    }
    res = VisibilityReprojectionValidator.validate_camera_metadata(meta_incomplete)
    assert res.is_calibrated is False
    assert res.camera is None
    assert res.rejection_reason == "missing_focal_or_principal_point"


def test_missing_image_dimensions_returns_uncalibrated() -> None:
    # Missing width and height
    meta_no_dims = {
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
        "is_canonical_frame": True,
    }
    res = VisibilityReprojectionValidator.validate_camera_metadata(meta_no_dims)
    assert res.is_calibrated is False
    assert res.camera is None
    assert res.rejection_reason == "missing_image_dimensions"

    # Tuple without width / height
    meta_tuple_no_dims = {
        "intrinsics": [500.0, 500.0, 320.0, 240.0],
        "is_canonical_frame": True,
    }
    res2 = VisibilityReprojectionValidator.validate_camera_metadata(meta_tuple_no_dims)
    assert res2.is_calibrated is False
    assert res2.rejection_reason == "missing_image_dimensions"


def test_non_finite_and_non_positive_intrinsics() -> None:
    # NaN focal length
    meta_nan = {
        "intrinsics": {"fx": float("nan"), "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        "is_canonical_frame": True,
    }
    assert VisibilityReprojectionValidator.camera_from_metadata(meta_nan) is None

    # Negative focal length
    meta_neg_fx = {
        "intrinsics": {"fx": -500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        "is_canonical_frame": True,
    }
    res_neg = VisibilityReprojectionValidator.validate_camera_metadata(meta_neg_fx)
    assert res_neg.is_calibrated is False
    assert res_neg.rejection_reason == "non_positive_focal_length"

    # Non-finite distortion
    meta_inf_dist = {
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480, "k1": float("inf")},
        "is_canonical_frame": True,
    }
    assert VisibilityReprojectionValidator.camera_from_metadata(meta_inf_dist) is None


def test_extrinsics_validation_rigidity_and_determinant() -> None:
    valid_intr = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}

    # Missing extrinsics without is_canonical_frame must fail
    meta_missing_extr = {"intrinsics": valid_intr}
    res_missing = VisibilityReprojectionValidator.validate_camera_metadata(meta_missing_extr)
    assert res_missing.is_calibrated is False
    assert res_missing.rejection_reason == "missing_extrinsics_and_not_canonical_frame"

    # Canonical frame identity allowed when explicitly declared
    meta_canonical = {"intrinsics": valid_intr, "is_canonical_frame": True}
    res_can = VisibilityReprojectionValidator.validate_camera_metadata(meta_canonical)
    assert res_can.is_calibrated is True
    assert res_can.camera is not None
    assert res_can.camera.extrinsics.position.x == 0.0

    # Malformed shape (3x3 instead of 3x4 or 4x4)
    meta_bad_shape = {"intrinsics": valid_intr, "extrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    assert VisibilityReprojectionValidator.camera_from_metadata(meta_bad_shape) is None

    # Reflection matrix (det = -1.0) must be rejected
    reflection_extr = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    meta_reflection = {"intrinsics": valid_intr, "extrinsics": reflection_extr}
    res_refl = VisibilityReprojectionValidator.validate_camera_metadata(meta_reflection)
    assert res_refl.is_calibrated is False
    assert "non_rigid_rotation_determinant" in (res_refl.rejection_reason or "")

    # Non-orthogonal matrix (sheared / scaled) must be rejected
    sheared_extr = [
        [1.5, 0.2, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    meta_sheared = {"intrinsics": valid_intr, "extrinsics": sheared_extr}
    assert VisibilityReprojectionValidator.camera_from_metadata(meta_sheared) is None


def test_metadata_provenance_preservation() -> None:
    valid_intr = {"fx": 800.0, "fy": 800.0, "cx": 400.0, "cy": 300.0, "width": 800, "height": 600}
    meta = {
        "intrinsics": valid_intr,
        "is_canonical_frame": True,
        "calibration_source": "colmap_v3.11",
        "calibration_method": "bundle_adjustment_ceres",
        "calibration_version": "3.11.1",
        "coordinate_frame": "world:colmap",
        "uncertainty": {"reprojection_error_px": 0.42, "variance_focal": 0.05},
    }
    validated = VisibilityReprojectionValidator.validate_camera_metadata(meta)
    assert validated.is_calibrated is True
    assert validated.source == "colmap_v3.11"
    assert validated.method == "bundle_adjustment_ceres"
    assert validated.version == "3.11.1"
    assert validated.coordinate_frame == "world:colmap"
    assert validated.uncertainty is not None
    assert validated.uncertainty["reprojection_error_px"] == 0.42


def test_project_3d_to_pixel_rejects_uncalibrated_and_malformed() -> None:
    pt = [0.0, 0.0, 2.0]
    # Empty meta
    assert project_3d_to_pixel(pt, {}) is None
    # Meta with missing width/height
    assert project_3d_to_pixel(pt, {"intrinsics": [500, 500, 320, 240]}) is None
    # Meta with non-rigid extrinsics
    assert project_3d_to_pixel(pt, {
        "intrinsics": [500, 500, 320, 240], "width": 640, "height": 480,
        "extrinsics": [[2.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
    }) is None
    # Valid calibrated projection
    valid_meta = {
        "intrinsics": [500.0, 500.0, 320.0, 240.0],
        "width": 640,
        "height": 480,
        "is_canonical_frame": True,
    }
    proj = project_3d_to_pixel(pt, valid_meta)
    assert proj is not None
    assert pytest.approx(proj[0], abs=1e-3) == 320.0
    assert pytest.approx(proj[1], abs=1e-3) == 240.0
