"""Rigorous validation tests for Metric 3D Lifting.

Tests:
1. Mathematical unprojection accuracy on known 3D synthetic cuboid geometry.
2. Rejection of unscaled / relative monocular depth maps (is_metric_depth=False).
3. Rejection of mismatched or all-zero masks.
4. Correct detection of rotational symmetry / circular planar distributions (orientation=None).
5. Order-statistic quantile variance estimation for N >= 20 and explanation for N < 20.
6. Proper provenance preservation and serialization in LiftedEntity3D.to_dict().
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from engine.math import Quat, Vec3
from engine.multiengine.metric_lifting import LiftedEntity3D, Metric3DLifter
from world.ontology.universal_object_ontology import UniversalObjectOntology
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


@pytest.fixture
def calibrated_camera() -> PinholeCamera:
    """Standard calibrated camera at origin looking along +Z."""
    intr = CameraIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    extr = CameraExtrinsics(
        position=Vec3(0.0, 0.0, 0.0),
        rotation=Quat.identity(),
    )
    return PinholeCamera(intrinsics=intr, extrinsics=extr)


def test_reconstructs_synthetic_3d_cuboid_with_high_accuracy(calibrated_camera: PinholeCamera) -> None:
    """Projects a known 3D cuboid of (0.8m x 0.6m x 0.4m) at position (0.3m, -0.1m, 2.0m)

    into a depth buffer and lifts it back, asserting precision on position and dimensions.
    """
    camera = calibrated_camera
    lifter = Metric3DLifter(UniversalObjectOntology())

    # Create synthetic 3D point cloud on surface of cuboid
    # Cuboid center: (0.3, -0.1, 2.0)
    # Extents: X in [0.3 - 0.4, 0.3 + 0.4] = [-0.1, 0.7] (width 0.8)
    #          Y in [-0.1 - 0.3, -0.1 + 0.3] = [-0.4, 0.2] (height 0.6)
    #          Z in [2.0 - 0.2, 2.0 + 0.2] = [1.8, 2.2] (depth 0.4)
    depth_map = np.full((480, 640), np.nan, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=bool)

    # Sample points on visible front and top surfaces
    xs = np.linspace(-0.1, 0.7, 40)
    ys = np.linspace(-0.4, 0.2, 30)
    zs = np.linspace(1.8, 2.2, 20)

    for x in xs:
        for y in ys:
            # Front face at z = 1.8
            u, v = camera.project(Vec3(x, y, 1.8))
            iu, iv = int(round(u)), int(round(v))
            if 0 <= iu < 640 and 0 <= iv < 480:
                depth_map[iv, iu] = 1.8
                mask[iv, iu] = True

    # Bounding box in 2D
    active_vs, active_us = np.where(mask)
    box_2d = (float(active_us.min()), float(active_vs.min()), float(active_us.max()), float(active_vs.max()))

    lifted = lifter.lift_detection(
        entity_id="cuboid_01",
        class_name="table",
        box_2d=box_2d,
        depth_map=depth_map,
        camera=camera,
        mask_2d=mask,
        is_metric_depth=True,
        depth_source="calibrated_sensor_depth",
        calibration_source="bundle_adjustment",
        confidence=0.95,
    )

    assert lifted is not None
    assert lifted.entity_id == "cuboid_01"
    assert lifted.class_name == "table"
    assert lifted.depth_source == "calibrated_sensor_depth"
    assert lifted.calibration_source == "bundle_adjustment"
    assert lifted.confidence == 0.95

    # Position check (X, Y should match median center within ~2 cm)
    assert pytest.approx(lifted.position[0], abs=0.03) == 0.3
    assert pytest.approx(lifted.position[1], abs=0.03) == -0.1

    # Dimensions check: width and height
    assert pytest.approx(lifted.dimensions[0], abs=0.08) == 0.8
    assert pytest.approx(lifted.dimensions[1], abs=0.08) == 0.6


def test_rejects_relative_unscaled_depth(calibrated_camera: PinholeCamera) -> None:
    """Non-metric relative depth (is_metric_depth=False) MUST return None to prevent fake meters."""
    camera = calibrated_camera
    lifter = Metric3DLifter()

    depth_map = np.full((480, 640), 100.0, dtype=np.float32)

    lifted = lifter.lift_detection(
        entity_id="obj_relative",
        class_name="chair",
        box_2d=(100.0, 100.0, 200.0, 200.0),
        depth_map=depth_map,
        camera=camera,
        is_metric_depth=False,  # Explicit unscaled relative depth
    )

    assert lifted is None


def test_rejects_incompatible_or_empty_masks(calibrated_camera: PinholeCamera) -> None:
    """Mismatched mask dimensions or completely empty masks are rejected."""
    camera = calibrated_camera
    lifter = Metric3DLifter()

    depth_map = np.full((480, 640), 2.0, dtype=np.float32)

    # 1. Incompatible shape
    bad_mask = np.ones((240, 320), dtype=bool)
    lifted_bad_shape = lifter.lift_detection(
        entity_id="bad_mask_shape",
        class_name="door",
        box_2d=(100.0, 100.0, 200.0, 200.0),
        depth_map=depth_map,
        camera=camera,
        mask_2d=bad_mask,
    )
    assert lifted_bad_shape is None

    # 2. All-zero mask
    empty_mask = np.zeros((480, 640), dtype=bool)
    lifted_empty = lifter.lift_detection(
        entity_id="empty_mask",
        class_name="door",
        box_2d=(100.0, 100.0, 200.0, 200.0),
        depth_map=depth_map,
        camera=camera,
        mask_2d=empty_mask,
    )
    assert lifted_empty is None


def test_rotational_symmetry_reports_unknown_orientation(calibrated_camera: PinholeCamera) -> None:
    """Circular/radially symmetric distributions return orientation=None and is_orientation_known=False."""
    camera = calibrated_camera
    lifter = Metric3DLifter()

    # Generate a circular cylinder disc at depth 2.0m centered at (0, 0, 2)
    depth_map = np.full((480, 640), np.nan, dtype=np.float32)
    radius_m = 0.30

    angles = np.linspace(0, 2 * math.pi, 200)
    rs = np.linspace(0.01, radius_m, 20)

    for r in rs:
        for theta in angles:
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            u, v = camera.project(Vec3(x, y, 2.0))
            iu, iv = int(round(u)), int(round(v))
            if 0 <= iu < 640 and 0 <= iv < 480:
                depth_map[iv, iu] = 2.0

    lifted = lifter.lift_detection(
        entity_id="bollard_01",
        class_name="bollard",
        box_2d=(200.0, 120.0, 440.0, 360.0),
        depth_map=depth_map,
        camera=camera,
        is_metric_depth=True,
    )

    assert lifted is not None
    # For a circle, planar eigenvalues are equal -> orientation is None
    assert lifted.orientation is None
    assert lifted.is_orientation_known is False
    assert lifted.orientation_reason == "planar_symmetry_circular_eigenvalues"
    assert lifted.orientation_quat is None


def test_order_statistic_quantile_variance(calibrated_camera: PinholeCamera) -> None:
    """Tests that N >= 20 calculates covariance_dimensions with order-statistic quantile variance,

    while N < 20 sets covariance_dimensions=None with an explanation.
    """
    camera = calibrated_camera
    lifter = Metric3DLifter()

    depth_map = np.full((480, 640), 2.5, dtype=np.float32)

    # 1. High density box (N > 100 points)
    lifted_dense = lifter.lift_detection(
        entity_id="dense_box",
        class_name="bench",
        box_2d=(280.0, 200.0, 360.0, 280.0),
        depth_map=depth_map,
        camera=camera,
    )
    assert lifted_dense is not None
    assert lifted_dense.covariance_dimensions is not None
    cov_d = np.array(lifted_dense.covariance_dimensions)
    assert cov_d.shape == (3, 3)
    assert cov_d[0, 0] > 0
    assert cov_d[1, 1] > 0
    assert cov_d[2, 2] > 0
    assert lifted_dense.dimension_uncertainty_explanation == "order_statistic_quantile_variance_plus_sensor_noise"

    # 2. Sparse points: min_points <= N < 20
    sparse_depth = np.full((480, 640), np.nan, dtype=np.float32)
    # Place a 3x3 grid of 9 points with non-zero X and Y extents
    for dv in [-4, 0, 4]:
        for du in [-15, 0, 15]:
            sparse_depth[240 + dv, 320 + du] = 2.5

    lifted_sparse = lifter.lift_detection(
        entity_id="sparse_box",
        class_name="pipe",
        box_2d=(300.0, 235.0, 340.0, 245.0),
        depth_map=sparse_depth,
        camera=camera,
        min_points=5,
    )
    assert lifted_sparse is not None
    assert lifted_sparse.point_count < 20
    assert lifted_sparse.covariance_dimensions is None
    assert "insufficient_point_density" in str(lifted_sparse.dimension_uncertainty_explanation)


def test_lifted_entity_serialization_dict_contract() -> None:
    """Tests that to_dict() produces the expected schema with explicit quaternion keys and provenance."""
    entity = LiftedEntity3D(
        entity_id="door_01",
        class_name="door",
        position=(1.2, 0.4, 3.1),
        dimensions=(0.9, 0.1, 2.1),
        orientation=(1.0, 0.0, 0.0, 0.0),  # (w, x, y, z)
        coordinate_frame="world:canonical",
        covariance_position=((0.01, 0, 0), (0, 0.01, 0), (0, 0, 0.01)),
        covariance_dimensions=((0.02, 0, 0), (0, 0.02, 0), (0, 0, 0.02)),
        point_count=500,
        rejected_point_count=12,
        depth_source="colmap_mvs_depth",
        calibration_source="colmap_reconstruction",
        fitting_method="pca_horizontal_quantile",
        is_orientation_known=True,
        confidence=0.91,
        confidence_basis="calibrated_empirical",
    )

    d = entity.to_dict()
    assert d["entity_id"] == "door_01"
    assert d["position"] == {"x": 1.2, "y": 0.4, "z": 3.1}
    assert d["dimensions"] == {"width": 0.9, "depth": 0.1, "height": 2.1}
    assert d["orientation"] == {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    assert d["is_orientation_known"] is True
    assert d["rejected_point_count"] == 12
    assert d["depth_source"] == "colmap_mvs_depth"
    assert d["confidence"] == 0.91
    assert d["confidence_basis"] == "calibrated_empirical"
