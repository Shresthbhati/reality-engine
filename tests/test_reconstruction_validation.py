"""Tests for Goal F: reconstruction validation surfaces disagreement."""

import pytest

from provenance import Uncertainty
from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult
from reconstruction.validation import (
    UnvalidatableReconstructionError,
    validate_reconstructions,
)


def _point(track_id, position):
    return ReconstructedPoint(position=position, track_id=track_id, source_evidence_ids=["p1"])


def test_matching_points_within_threshold_count_as_agreement():
    a = ReconstructionResult(points=[_point("a1", (0, 0, 0))], camera_poses=[], registration_status="success")
    b = ReconstructionResult(points=[_point("b1", (0.01, 0, 0))], camera_poses=[], registration_status="success")

    report = validate_reconstructions(a, b, distance_threshold=0.1)
    assert report.agreements == 1
    assert not report.has_disagreements()


def test_points_beyond_threshold_are_flagged_as_disagreement():
    a = ReconstructionResult(points=[_point("a1", (0, 0, 0))], camera_poses=[], registration_status="success")
    b = ReconstructionResult(points=[_point("b1", (5, 0, 0))], camera_poses=[], registration_status="success")

    report = validate_reconstructions(a, b, distance_threshold=0.1)
    assert report.agreements == 0
    assert report.has_disagreements()
    assert report.disagreements[0].point_a_id == "a1"
    assert report.disagreements[0].point_b_id == "b1"
    assert report.disagreements[0].distance == pytest.approx(5.0)


def test_empty_reconstruction_b_counts_all_of_a_as_unmatched():
    a = ReconstructionResult(points=[_point("a1", (0, 0, 0))], camera_poses=[], registration_status="success")
    b = ReconstructionResult(points=[], camera_poses=[], registration_status="success")

    report = validate_reconstructions(a, b, distance_threshold=0.1)
    assert report.agreements == 0
    assert not report.has_disagreements()
    assert report.unmatched_a == 1


def test_refuses_to_validate_a_failed_reconstruction():
    a = ReconstructionResult(points=[], camera_poses=[], registration_status="failed")
    b = ReconstructionResult(points=[_point("b1", (0, 0, 0))], camera_poses=[], registration_status="success")

    with pytest.raises(UnvalidatableReconstructionError):
        validate_reconstructions(a, b, distance_threshold=0.1)


def test_multiple_points_partition_into_agreements_and_disagreements():
    a = ReconstructionResult(
        points=[_point("a1", (0, 0, 0)), _point("a2", (10, 10, 10))],
        camera_poses=[], registration_status="success",
    )
    b = ReconstructionResult(
        points=[_point("b1", (0.05, 0, 0)), _point("b2", (99, 99, 99))],
        camera_poses=[], registration_status="success",
    )

    report = validate_reconstructions(a, b, distance_threshold=0.5)
    assert report.agreements == 1
    assert len(report.disagreements) == 1
    assert report.disagreements[0].point_a_id == "a2"
