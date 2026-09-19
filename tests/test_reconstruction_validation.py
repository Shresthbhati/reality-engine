"""Tests for Goal F: reconstruction validation surfaces disagreement."""

import math
import random
import time

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


def _brute_force_oracle(a_points, b_points, threshold):
    """Test oracle: the original O(N*M) nearest-match semantics.

    Mirrors the pre-optimization implementation exactly (first-minimal
    tie rule, track_id SET collapse for unmatched_b, discard on both
    agreement and disagreement) so the accelerated implementation is
    pinned to measured behavior, not to a re-derivation.
    """
    matched_b_ids = set()
    agreements = 0
    disagreements = []
    for pa in a_points:
        nearest = min(b_points, key=lambda pb: math.dist(pa.position, pb.position))
        d = math.dist(pa.position, nearest.position)
        if d <= threshold:
            agreements += 1
        else:
            disagreements.append((pa.track_id, nearest.track_id, d))
        matched_b_ids.add(nearest.track_id)
    unmatched_b = len({pb.track_id for pb in b_points} - matched_b_ids)
    return agreements, disagreements, unmatched_b


def test_large_shuffled_sets_match_reference_semantics():
    """Accelerated matching must equal the brute-force oracle on
    shuffled, non-trivial data: jittered near-coincidences (agreements),
    displaced outliers (disagreements, in a-point order), b-only points,
    and duplicate track_ids (unmatched_b collapses to a SET)."""
    rng = random.Random(42)
    a_points, b_points = [], []
    for i in range(300):
        base = (rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-10, 10))
        a_points.append(_point(f"a{i % 250}", base))  # duplicate track_ids on a-side too
        roll = rng.random()
        if roll < 0.7:  # agreement: same surface, sub-threshold jitter
            jitter = 0.01
            b_points.append(_point(f"b{i % 200}", tuple(c + rng.uniform(-jitter, jitter) for c in base)))
        elif roll < 0.9:  # disagreement: displaced beyond the threshold
            b_points.append(_point(f"b{i % 200}", tuple(c + 3.0 for c in base)))
        else:  # b-only structure a never observed
            b_points.append(_point(f"bonly{i}", (rng.uniform(50, 60), rng.uniform(50, 60), rng.uniform(50, 60))))
    rng.shuffle(a_points)
    rng.shuffle(b_points)

    result_a = ReconstructionResult(points=a_points, camera_poses=[], registration_status="success")
    result_b = ReconstructionResult(points=b_points, camera_poses=[], registration_status="success")

    report = validate_reconstructions(result_a, result_b, distance_threshold=0.1)
    exp_agreements, exp_disagreements, exp_unmatched_b = _brute_force_oracle(a_points, b_points, 0.1)

    assert report.agreements == exp_agreements
    assert report.unmatched_b == exp_unmatched_b
    assert report.unmatched_a == 0
    assert len(report.disagreements) == len(exp_disagreements)
    for got, (exp_a_id, exp_b_id, exp_d) in zip(report.disagreements, exp_disagreements):
        assert got.point_a_id == exp_a_id
        assert got.point_b_id == exp_b_id
        assert got.distance == pytest.approx(exp_d)


def test_large_reconstruction_completes_within_bounded_time():
    """Scale gate (large image-set robustness): 20k-vs-20k point
    validation must complete in bounded time. The O(N*M) scan is
    ~4e8 Python distance evals (minutes); an indexed match is not.
    1,000 exactly-coincident pairs must land as agreements."""
    n = 60_000
    coincident = [(float(i) * 0.5, float(i % 97) * 0.25, float(i % 13)) for i in range(1_000)]
    spread_a = [(float(i) * 3.7 + 1000.0, float(i % 891) * 1.3, float(i % 271) * 0.7) for i in range(n - 1_000)]
    spread_b = [(x + 2.5, y, z) for (x, y, z) in spread_a[: n - 1_000]]  # displaced: disagreements
    a_points = [_point(f"a{i}", p) for i, p in enumerate(coincident + spread_a)]
    b_points = [_point(f"b{i}", p) for i, p in enumerate(coincident + spread_b)]

    result_a = ReconstructionResult(points=a_points, camera_poses=[], registration_status="success")
    result_b = ReconstructionResult(points=b_points, camera_poses=[], registration_status="success")

    start = time.perf_counter()
    report = validate_reconstructions(result_a, result_b, distance_threshold=0.1)
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, f"validation took {elapsed:.1f}s -- not large-set tractable"
    assert report.agreements >= 1_000  # every exact coincidence must agree
    assert report.agreements + len(report.disagreements) == n  # every a-point classified
    # Deterministic stragglers: in the displaced chain each a-point's nearest
    # is its PREDECESSOR's twin, so chain-boundary b-points (and coincident
    # twins stolen by a spacing overlap) are never anybody's nearest. Pin the
    # exact measured count -- it is a property of this deterministic fixture,
    # and it exercises the unmatched_b set-collapse at scale.
    assert report.unmatched_b == 283


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
