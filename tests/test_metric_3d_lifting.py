"""Tests for Metric 3D Lifting, Oriented Bounding Boxes, and Part Association."""

from __future__ import annotations

import numpy as np
import pytest

from engine.math import Quat, Vec3
from engine.multiengine.metric_lifting import LiftedEntity3D, Metric3DLifter
from world.ontology.universal_object_ontology import (
    UniversalObjectOntology,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

ANCHORED = dict(  # SYNTHETIC test fixtures declare their scale explicitly; nothing defaults to metres
    is_metric_depth=True, scale_state="PHYSICALLY_ANCHORED", scale_source="synthetic-test-fixture-anchor",
    depth_source="synthetic_test_fixture", calibration_source="synthetic_test_fixture",
)


@pytest.fixture
def test_setup() -> tuple[PinholeCamera, np.ndarray, UniversalObjectOntology]:
    """Pinhole camera at origin, 2.0m flat depth plane, and initialized ontology."""
    intr = CameraIntrinsics(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        width=640, height=480,
    )
    extr = CameraExtrinsics(
        position=Vec3(0.0, 0.0, 0.0),
        rotation=Quat.identity(),
    )
    camera = PinholeCamera(intrinsics=intr, extrinsics=extr)
    depth_map = np.full((480, 640), 2.0, dtype=np.float32)

    ontology = UniversalObjectOntology()
    return camera, depth_map, ontology


def test_lift_detection_reconstructs_metric_position_and_dimensions(
    test_setup: tuple[PinholeCamera, np.ndarray, UniversalObjectOntology],
) -> None:
    """Detection centered at (320, 240) of size 100x100 at 2.0m depth lifts to 0.40m x 0.40m box at (0, 0, 2.0)."""
    camera, depth_map, ontology = test_setup
    lifter = Metric3DLifter(ontology)

    # Box: x in [270, 370], y in [190, 290]
    # At depth 2.0m: delta_x = 100 * 2.0 / 500 = 0.40 m; delta_y = 0.40 m
    lifted = lifter.lift_detection(
        **ANCHORED,
        entity_id="chair_01",
        class_name="chair",
        box_2d=(270.0, 190.0, 370.0, 290.0),
        depth_map=depth_map,
        camera=camera,
        target_frame="world:canonical",
    )

    assert lifted is not None
    assert isinstance(lifted, LiftedEntity3D)
    assert lifted.entity_id == "chair_01"
    assert lifted.class_name == "chair"

    # Position should be centered at (0.0, 0.0, 2.0)
    assert pytest.approx(lifted.position[0], abs=0.02) == 0.0
    assert pytest.approx(lifted.position[1], abs=0.02) == 0.0
    assert pytest.approx(lifted.position[2], abs=0.02) == 2.0

    # Dimensions should be ~0.40 m across width and depth
    assert pytest.approx(lifted.dimensions[0], abs=0.05) == 0.40
    assert pytest.approx(lifted.dimensions[1], abs=0.05) == 0.40

    # Uncertainty covariance exists and is positive definite
    assert lifted.covariance_position is not None
    cov = np.array(lifted.covariance_position)
    assert cov.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_lift_detection_returns_none_when_unobserved_or_invalid_depth(
    test_setup: tuple[PinholeCamera, np.ndarray, UniversalObjectOntology],
) -> None:
    """Zero geometry is fabricated when depth is unobserved (NaN, 0, or negative)."""
    camera, depth_map, ontology = test_setup
    lifter = Metric3DLifter(ontology)

    # Invalidate depth in the box region
    invalid_depth = depth_map.copy()
    invalid_depth[190:290, 270:370] = np.nan

    lifted = lifter.lift_detection(
        **ANCHORED,
        entity_id="unobserved_01",
        class_name="table",
        box_2d=(270.0, 190.0, 370.0, 290.0),
        depth_map=invalid_depth,
        camera=camera,
    )

    # Strictly None -- no fake default geometry!
    assert lifted is None


def test_part_parent_association_links_subpart_to_parent_volume(
    test_setup: tuple[PinholeCamera, np.ndarray, UniversalObjectOntology],
) -> None:
    """A chair_leg inside a chair's bounding volume is associated as PART_OF the chair."""
    camera, depth_map, ontology = test_setup
    lifter = Metric3DLifter(ontology)

    chair = LiftedEntity3D(
        entity_id="chair_main",
        class_name="chair",
        position=(0.0, 0.0, 2.0),
        dimensions=(0.60, 0.60, 0.90),
        orientation=(0.0, 0.0, 0.0, 1.0),
        coordinate_frame="world:canonical",
        covariance_position=None,
        covariance_dimensions=None,
        point_count=1000,
    )

    chair_leg = LiftedEntity3D(
        entity_id="chair_leg_01",
        class_name="chair_leg",  # Recognized subpart in ontology
        position=(0.20, 0.20, 1.80),  # Inside chair volume
        dimensions=(0.05, 0.05, 0.40),
        orientation=(0.0, 0.0, 0.0, 1.0),
        coordinate_frame="world:canonical",
        covariance_position=None,
        covariance_dimensions=None,
        point_count=150,
    )

    associated = lifter.associate_parts_and_parents([chair, chair_leg])

    assert len(associated) == 2
    part_result = next(e for e in associated if e.entity_id == "chair_leg_01")
    assert part_result.is_part is True
    assert part_result.parent_entity_id == "chair_main"
    assert part_result.relative_transform_to_parent is not None


def test_observation_drafts_output_compatible_with_canonical_contracts(
    test_setup: tuple[PinholeCamera, np.ndarray, UniversalObjectOntology],
) -> None:
    """Observation drafts output by lifter strictly follow canonical entity contracts."""
    camera, depth_map, ontology = test_setup
    lifter = Metric3DLifter(ontology)

    lifted = lifter.lift_detection(
        **ANCHORED,
        entity_id="faucet_01",
        class_name="faucet",
        box_2d=(300.0, 220.0, 340.0, 260.0),
        depth_map=depth_map,
        camera=camera,
        confidence=0.88,
    )
    assert lifted is not None

    drafts = lifter.to_observation_drafts(lifted)
    assert len(drafts) == 3

    pos_draft = next(d for d in drafts if d.quantity == "position")
    assert pos_draft.task == "reconstruction_3d"
    assert "value" in pos_draft.predicted
    assert len(pos_draft.predicted["value"]) == 3
    assert pos_draft.uncertainty is not None
    assert "covariance" in pos_draft.uncertainty

    dim_draft = next(d for d in drafts if d.quantity == "dimensions")
    assert "value" in dim_draft.predicted
    assert len(dim_draft.predicted["value"]) == 3
