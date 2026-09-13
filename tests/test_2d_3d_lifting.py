"""Tests for 2D->3D lifting (perception/instances/lifting.py).

Fixtures are hand-built SegmentedRegion/DepthMap pairs (not a real
segmentation/depth backend) -- same pattern
tests/test_geometric_reasoning.py uses for synthetic reconstruction
fixtures: the lifting *algorithm* is what's under test, and it must
behave identically whether its inputs came from a real backend or a
hand-built fixture with known, hand-checkable values.
"""

from __future__ import annotations

import math

import pytest

from engine.physics.math3 import Quat, Vec3
from perception.depth.interface import DepthMap
from perception.instances.lifting import LiftingError, lift_region_to_3d
from perception.segmentation.interface import SegmentedRegion
from provenance import Provenance
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

_WIDTH, _HEIGHT = 8, 6


def _camera() -> PinholeCamera:
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=_WIDTH / 2, cy=_HEIGHT / 2, width=_WIDTH, height=_HEIGHT)
    extrinsics = CameraExtrinsics(position=Vec3.zero(), rotation=Quat.identity())
    return PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)


def _flat_depth(evidence_id="ev-1", depth_value=5.0, unit="meters") -> DepthMap:
    return DepthMap(
        evidence_id=evidence_id, width=_WIDTH, height=_HEIGHT,
        values=[[depth_value for _ in range(_WIDTH)] for _ in range(_HEIGHT)],
        unit=unit,
    )


def _rect_mask(row_lo, row_hi, col_lo, col_hi) -> list[list[bool]]:
    return [[row_lo <= r < row_hi and col_lo <= c < col_hi for c in range(_WIDTH)] for r in range(_HEIGHT)]


def _region(mask, evidence_id="ev-1", confidence=0.9, label="chair") -> SegmentedRegion:
    return SegmentedRegion(region_id="reg-1", evidence_id=evidence_id, label=label, mask=mask, confidence=confidence)


def test_flat_object_lifts_to_a_planar_hypothesis_at_known_depth():
    camera = _camera()
    depth = _flat_depth(depth_value=5.0)
    region = _region(_rect_mask(1, 5, 2, 6))  # a rectangle covering the mid-frame

    hyp = lift_region_to_3d(region, depth, camera)
    assert hyp is not None
    assert hyp.label == "chair"
    assert hyp.point_count == 4 * 4  # (5-1) rows x (6-2) cols
    assert math.isclose(hyp.position.z, 5.0, abs_tol=1e-9)
    assert hyp.provenance is Provenance.INFERRED


def test_full_valid_depth_gives_confidence_equal_to_segmentation_confidence():
    camera = _camera()
    depth = _flat_depth()
    region = _region(_rect_mask(0, _HEIGHT, 0, _WIDTH), confidence=0.8)

    hyp = lift_region_to_3d(region, depth, camera)
    assert hyp is not None
    assert math.isclose(hyp.confidence, 0.8, abs_tol=1e-9)  # 100% valid depth -> full confidence


def test_partial_valid_depth_lowers_confidence_proportionally():
    camera = _camera()
    values = [[5.0 for _ in range(_WIDTH)] for _ in range(_HEIGHT)]
    # Half the masked region has invalid (zero) depth.
    for r in range(_HEIGHT):
        for c in range(_WIDTH // 2):
            values[r][c] = 0.0
    depth = DepthMap(evidence_id="ev-1", width=_WIDTH, height=_HEIGHT, values=values, unit="meters")
    region = _region(_rect_mask(0, _HEIGHT, 0, _WIDTH), confidence=1.0)

    hyp = lift_region_to_3d(region, depth, camera)
    assert hyp is not None
    assert math.isclose(hyp.confidence, 0.5, abs_tol=1e-9)


def test_insufficient_valid_depth_returns_none_not_a_hypothesis():
    camera = _camera()
    values = [[0.0 for _ in range(_WIDTH)] for _ in range(_HEIGHT)]
    values[0][0] = 5.0  # a single valid pixel out of the whole masked region
    depth = DepthMap(evidence_id="ev-1", width=_WIDTH, height=_HEIGHT, values=values, unit="meters")
    region = _region(_rect_mask(0, _HEIGHT, 0, _WIDTH))

    assert lift_region_to_3d(region, depth, camera) is None


def test_empty_mask_returns_none():
    camera = _camera()
    depth = _flat_depth()
    region = _region([[False] * _WIDTH for _ in range(_HEIGHT)])

    assert lift_region_to_3d(region, depth, camera) is None


def test_mismatched_evidence_id_raises():
    camera = _camera()
    depth = _flat_depth(evidence_id="ev-2")
    region = _region(_rect_mask(0, 2, 0, 2), evidence_id="ev-1")

    with pytest.raises(LiftingError):
        lift_region_to_3d(region, depth, camera)


def test_relative_depth_map_is_refused_not_silently_treated_as_metric():
    camera = _camera()
    depth = _flat_depth(unit="relative")
    region = _region(_rect_mask(0, 2, 0, 2))

    with pytest.raises(LiftingError):
        lift_region_to_3d(region, depth, camera)


def test_mismatched_mask_dimensions_raises():
    camera = _camera()
    depth = _flat_depth()
    bad_mask = [[True] * (_WIDTH + 1) for _ in range(_HEIGHT)]
    region = SegmentedRegion(region_id="reg-1", evidence_id="ev-1", label="x", mask=bad_mask, confidence=0.5)

    with pytest.raises(LiftingError):
        lift_region_to_3d(region, depth, camera)


def test_bounds_are_tight_around_the_unprojected_points():
    camera = _camera()
    depth = _flat_depth(depth_value=10.0)
    region = _region(_rect_mask(2, 4, 3, 5))

    hyp = lift_region_to_3d(region, depth, camera)
    assert hyp is not None
    assert hyp.bounds_min.z == hyp.bounds_max.z == 10.0
    assert hyp.bounds_min.x < hyp.bounds_max.x
    assert hyp.bounds_min.y < hyp.bounds_max.y


def test_to_dict_is_plain_data():
    camera = _camera()
    depth = _flat_depth()
    region = _region(_rect_mask(0, 2, 0, 2))

    hyp = lift_region_to_3d(region, depth, camera)
    payload = hyp.to_dict()
    assert payload["label"] == "chair"
    assert payload["provenance"] == "INFERRED"
