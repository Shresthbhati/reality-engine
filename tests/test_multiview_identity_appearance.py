"""Tests for P7-01 multi-view object identity strengthening: appearance
histograms (perception/instances/appearance.py), epipolar consistency
(perception/instances/epipolar.py), and their wiring into
`object_resolution.merge_hypotheses`.

Uses deterministic rendered-view fixtures: real `PinholeCamera`s and
hand-built RGB images, exactly like tests/test_room_inference.py's
synthetic-projection pattern -- no real image/dataset dependency.
"""

from __future__ import annotations

import math

import pytest

from engine.physics.math3 import Quat, Vec3
from perception.instances.appearance import (
    appearance_similarity,
    compute_color_histogram,
)
from perception.instances.epipolar import epipolar_consistent, epipolar_residual_px
from perception.instances.lifting import ObjectHypothesis3D
from perception.instances.object_resolution import merge_hypotheses
from provenance import Uncertainty
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


def _camera(position: Vec3, rotation: Quat = Quat.identity()) -> PinholeCamera:
    return PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=position, rotation=rotation),
    )


def _solid_image(width: int, height: int, color) -> list:
    return [[color for _ in range(width)] for _ in range(height)]


def _full_mask(width: int, height: int) -> list:
    return [[True for _ in range(width)] for _ in range(height)]


def _hyp(region_id, evidence_id, label, position: Vec3, confidence=0.8) -> ObjectHypothesis3D:
    return ObjectHypothesis3D(
        region_id=region_id, evidence_id=evidence_id, label=label,
        position=position,
        bounds_min=Vec3(position.x - 0.1, position.y - 0.1, position.z - 0.1),
        bounds_max=Vec3(position.x + 0.1, position.y + 0.1, position.z + 0.1),
        point_count=100, mask_pixel_count=120, confidence=confidence,
        uncertainty=Uncertainty(confidence=confidence),
    )


# ---------------------------------------------------------------------------
# appearance.py
# ---------------------------------------------------------------------------

def test_color_histogram_of_solid_region_is_a_single_bin():
    image = _solid_image(4, 4, (250, 10, 10))
    mask = _full_mask(4, 4)
    desc = compute_color_histogram(image, mask)
    assert desc is not None
    assert desc.pixel_count == 16
    assert sum(desc.histogram) == pytest.approx(1.0)
    assert max(desc.histogram) == pytest.approx(1.0)


def test_empty_mask_returns_none_not_a_fabricated_histogram():
    image = _solid_image(4, 4, (250, 10, 10))
    mask = [[False for _ in range(4)] for _ in range(4)]
    assert compute_color_histogram(image, mask) is None


def test_identical_color_descriptors_are_fully_similar():
    image = _solid_image(4, 4, (10, 200, 10))
    mask = _full_mask(4, 4)
    a = compute_color_histogram(image, mask)
    b = compute_color_histogram(image, mask)
    assert appearance_similarity(a, b) == pytest.approx(1.0)


def test_distinct_solid_colors_have_low_similarity():
    red = compute_color_histogram(_solid_image(4, 4, (250, 5, 5)), _full_mask(4, 4))
    blue = compute_color_histogram(_solid_image(4, 4, (5, 5, 250)), _full_mask(4, 4))
    assert appearance_similarity(red, blue) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# epipolar.py
# ---------------------------------------------------------------------------

def test_true_correspondence_has_near_zero_epipolar_residual():
    camera_a = _camera(Vec3(0.0, 0.0, 0.0))
    camera_b = _camera(Vec3(1.0, 0.0, 0.0))
    point = Vec3(0.3, 0.1, 4.0)

    pixel_a = camera_a.project(point)
    pixel_b = camera_b.project(point)
    assert pixel_a is not None and pixel_b is not None

    residual = epipolar_residual_px(camera_a, pixel_a, camera_b, pixel_b)
    assert residual < 1e-6
    assert epipolar_consistent(camera_a, pixel_a, camera_b, pixel_b)


def test_unrelated_points_fail_epipolar_consistency():
    camera_a = _camera(Vec3(0.0, 0.0, 0.0))
    camera_b = _camera(Vec3(1.0, 0.0, 0.0))
    point_a = Vec3(0.3, 0.1, 4.0)
    # Off the epipolar plane defined by the baseline + point_a: a real
    # correspondence for pixel_a could never produce this pixel_b.
    point_b = Vec3(0.3, 1.5, 4.0)

    pixel_a = camera_a.project(point_a)
    pixel_b = camera_b.project(point_b)
    assert pixel_a is not None and pixel_b is not None

    residual = epipolar_residual_px(camera_a, pixel_a, camera_b, pixel_b)
    assert residual > 50.0
    assert not epipolar_consistent(camera_a, pixel_a, camera_b, pixel_b)


# ---------------------------------------------------------------------------
# object_resolution.merge_hypotheses wiring
# ---------------------------------------------------------------------------

def test_backward_compatible_without_new_signals():
    # No cameras/appearance supplied -> old label+proximity behavior.
    hyps = [
        _hyp("r1", "ev-1", "chair", Vec3(1.0, 0.0, 0.0)),
        _hyp("r2", "ev-2", "chair", Vec3(1.1, 0.0, 0.0)),
    ]
    candidates = merge_hypotheses(hyps)
    assert len(candidates) == 1


def test_consistent_multi_view_observation_of_one_real_object_merges():
    camera_a = _camera(Vec3(0.0, 0.0, 0.0))
    camera_b = _camera(Vec3(1.0, 0.0, 0.0))
    real_point = Vec3(0.3, 0.1, 4.0)

    hyp_a = _hyp("r1", "ev-a", "chair", real_point)
    hyp_b = _hyp("r2", "ev-b", "chair", Vec3(0.32, 0.09, 4.02))  # tiny reconstruction noise

    color = (200, 60, 60)
    desc_a = compute_color_histogram(_solid_image(4, 4, color), _full_mask(4, 4))
    desc_b = compute_color_histogram(_solid_image(4, 4, color), _full_mask(4, 4))

    candidates = merge_hypotheses(
        [hyp_a, hyp_b],
        cameras={"ev-a": camera_a, "ev-b": camera_b},
        appearance={"r1": desc_a, "r2": desc_b},
    )
    assert len(candidates) == 1
    assert candidates[0].observation_count == 2


def test_wrong_merge_rejected_by_epipolar_inconsistency():
    """Two DIFFERENT physical objects whose 3D reconstructions happen to
    land within the proximity threshold (the naive label+proximity merge
    this repo used to accept) -- the epipolar check must reject them
    even though they share a label and are close in 3D."""
    camera_a = _camera(Vec3(0.0, 0.0, 0.0))
    camera_b = _camera(Vec3(1.0, 0.0, 0.0))

    real_point_a = Vec3(0.3, 0.1, 4.0)
    real_point_b = Vec3(0.3, 1.5, 4.0)  # off camera_a-camera_b's epipolar plane

    hyp_a = _hyp("r1", "ev-a", "chair", real_point_a)
    # Reconstructed position deliberately placed close to hyp_a's (within
    # the default 0.5m merge distance) even though it comes from a real
    # detection of a genuinely different point -- the failure mode this
    # gate exists to catch (bad depth/pose making two different objects'
    # 3D centroids coincide).
    hyp_b = _hyp("r2", "ev-b", "chair", Vec3(0.35, 0.15, 4.0))

    candidates = merge_hypotheses(
        [hyp_a, hyp_b],
        cameras={"ev-a": camera_a, "ev-b": camera_b},
    )
    # Without the epipolar gate these would merge (same label, within
    # 0.5m). With it, the pair must stay separate.
    assert len(candidates) == 2


def test_wrong_merge_rejected_by_appearance_dissimilarity():
    """Same label, same 3D neighborhood (symmetric-scene case from the
    spec: e.g. two identical-shaped but differently-colored objects),
    appearance must still block the merge when colors disagree."""
    hyp_a = _hyp("r1", "ev-1", "chair", Vec3(1.0, 0.0, 0.0))
    hyp_b = _hyp("r2", "ev-2", "chair", Vec3(1.1, 0.0, 0.0))

    desc_red = compute_color_histogram(_solid_image(4, 4, (250, 5, 5)), _full_mask(4, 4))
    desc_blue = compute_color_histogram(_solid_image(4, 4, (5, 5, 250)), _full_mask(4, 4))

    candidates = merge_hypotheses(
        [hyp_a, hyp_b],
        appearance={"r1": desc_red, "r2": desc_blue},
    )
    assert len(candidates) == 2


def test_missing_signal_for_one_hypothesis_falls_back_to_base_gate():
    # Only one side has appearance data -> that pair isn't gated by
    # appearance (honest opt-in degrade, not a silent full block).
    hyp_a = _hyp("r1", "ev-1", "chair", Vec3(1.0, 0.0, 0.0))
    hyp_b = _hyp("r2", "ev-2", "chair", Vec3(1.1, 0.0, 0.0))
    desc_red = compute_color_histogram(_solid_image(4, 4, (250, 5, 5)), _full_mask(4, 4))

    candidates = merge_hypotheses([hyp_a, hyp_b], appearance={"r1": desc_red})
    assert len(candidates) == 1
