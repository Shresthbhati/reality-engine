"""Tests for perception/instances/measurement.py."""

from __future__ import annotations

import math

import pytest

from engine.physics.math3 import Vec3
from perception.instances.measurement import measure_dimensions, measure_distance
from perception.instances.object_resolution import MergedObjectCandidate
from provenance import Provenance


def _candidate(candidate_id, x, y, z, half_extents=(0.5, 0.5, 0.5), confidence=0.8) -> MergedObjectCandidate:
    hx, hy, hz = half_extents
    return MergedObjectCandidate(
        candidate_id=candidate_id, label="chair",
        position=Vec3(x, y, z),
        bounds_min=Vec3(x - hx, y - hy, z - hz),
        bounds_max=Vec3(x + hx, y + hy, z + hz),
        source_hypotheses=(), confidence=confidence,
    )


def test_dimensions_match_the_aabb_extents():
    candidate = _candidate("obj-1", 0, 0, 0, half_extents=(0.3, 0.6, 0.9))
    measurements = measure_dimensions(candidate)

    assert math.isclose(measurements["width_m"].value, 0.6, abs_tol=1e-9)
    assert math.isclose(measurements["height_m"].value, 1.2, abs_tol=1e-9)
    assert math.isclose(measurements["depth_m"].value, 1.8, abs_tol=1e-9)


def test_volume_is_the_product_of_extents():
    candidate = _candidate("obj-1", 0, 0, 0, half_extents=(0.5, 1.0, 2.0))
    measurements = measure_dimensions(candidate)
    # extents: 1.0 x 2.0 x 4.0
    assert math.isclose(measurements["volume_m3"].value, 8.0, abs_tol=1e-9)
    assert measurements["volume_m3"].unit == "meter^3"


def test_measurements_are_estimated_provenance_with_candidate_confidence():
    candidate = _candidate("obj-1", 0, 0, 0, confidence=0.65)
    measurements = measure_dimensions(candidate)
    for m in measurements.values():
        assert m.provenance is Provenance.ESTIMATED
        assert m.confidence == 0.65


def test_higher_confidence_yields_tighter_precision():
    low_conf = _candidate("obj-1", 0, 0, 0, confidence=0.3)
    high_conf = _candidate("obj-2", 0, 0, 0, confidence=0.95)

    low_precision = measure_dimensions(low_conf)["width_m"].precision
    high_precision = measure_dimensions(high_conf)["width_m"].precision
    assert high_precision < low_precision


def test_precision_never_goes_below_the_floor():
    candidate = _candidate("obj-1", 0, 0, 0, confidence=1.0)  # would compute to 0 precision otherwise
    precision = measure_dimensions(candidate)["width_m"].precision
    assert precision >= 0.01


def test_distance_between_candidates_is_euclidean():
    a = _candidate("obj-a", 0.0, 0.0, 0.0)
    b = _candidate("obj-b", 3.0, 4.0, 0.0)  # 3-4-5 triangle
    measurement = measure_distance(a, b)
    assert math.isclose(measurement.value, 5.0, abs_tol=1e-9)
    assert measurement.unit == "meter"


def test_distance_confidence_is_the_weaker_endpoint():
    a = _candidate("obj-a", 0.0, 0.0, 0.0, confidence=0.9)
    b = _candidate("obj-b", 1.0, 0.0, 0.0, confidence=0.4)
    measurement = measure_distance(a, b)
    assert measurement.confidence == 0.4


def test_distance_is_symmetric():
    a = _candidate("obj-a", 0.0, 0.0, 0.0)
    b = _candidate("obj-b", 2.0, 2.0, 2.0)
    assert math.isclose(measure_distance(a, b).value, measure_distance(b, a).value, abs_tol=1e-9)


# ------------------------------------------------------------------
# Measured-spread precision (P11: uncertainty propagation into
# measurements). The candidate's contributing hypotheses are
# independent observations of the same physical object; their
# disagreement is MEASURED evidence, not a heuristic.
# ------------------------------------------------------------------

from perception.instances.lifting import ObjectHypothesis3D


def _hypothesis(region_id, position, half=(0.4, 0.4, 0.4), confidence=0.8) -> ObjectHypothesis3D:
    x, y, z = position
    hx, hy, hz = half
    return ObjectHypothesis3D(
        region_id=region_id, evidence_id=f"ev-{region_id}", label="chair",
        position=Vec3(x, y, z),
        bounds_min=Vec3(x - hx, y - hy, z - hz),
        bounds_max=Vec3(x + hx, y + hy, z + hz),
        point_count=50, mask_pixel_count=100, confidence=confidence,
    )


def _merged(hypotheses) -> MergedObjectCandidate:
    from perception.instances.object_resolution import merge_hypotheses

    return merge_hypotheses(list(hypotheses))[0]


def test_single_view_falls_back_to_documented_heuristic():
    # One hypothesis: no independent observations exist, so there is no
    # measured spread -- the documented confidence heuristic is the
    # only available (and honestly labeled) precision source.
    candidate = _merged([_hypothesis("r1", (0.0, 0.0, 0.0), half=(0.5, 0.5, 0.5))])
    m = measure_dimensions(candidate)["width_m"]
    expected = max(0.01, 1.0 * (1.0 - candidate.confidence))  # width 1.0 m
    assert math.isclose(m.precision, expected, abs_tol=1e-9)


def test_multi_view_precision_uses_measured_spread_not_heuristic():
    # Three views, positions agreeing within ~2 cm. The heuristic would
    # give 0.8*(1-0.9) = 0.08 m; the measured spread is ~0.02 m.
    candidate = _merged([
        _hypothesis("r1", (0.00, 0.00, 0.00)),
        _hypothesis("r2", (0.02, 0.00, 0.00)),
        _hypothesis("r3", (0.00, 0.02, 0.00)),
    ])
    m = measure_dimensions(candidate)["width_m"]
    assert m.precision < 0.05  # measured, so far tighter than the heuristic
    assert "measured spread" in m.precision_note


def test_multi_view_spread_is_the_rms_about_the_centroid():
    # Two views at x=0 and x=0.03: the centroid sits at x=0.015, so
    # each view's 3D distance to it is 0.015 -- RMS = 0.015.
    candidate = _merged([
        _hypothesis("r1", (0.0, 0.0, 0.0)),
        _hypothesis("r2", (0.03, 0.0, 0.0)),
    ])
    m = measure_dimensions(candidate)["width_m"]
    assert math.isclose(m.precision, 0.015, abs_tol=1e-9)


def test_disagreeing_views_get_wider_precision_than_agreeing():
    agree = _merged([
        _hypothesis("r1", (0.0, 0.0, 0.0)),
        _hypothesis("r2", (0.01, 0.0, 0.0)),
    ])
    disagree = _merged([
        _hypothesis("r1", (0.0, 0.0, 0.0)),
        _hypothesis("r2", (0.20, 0.0, 0.0)),
    ])
    p_agree = measure_dimensions(agree)["width_m"].precision
    p_dis = measure_dimensions(disagree)["width_m"].precision
    assert p_dis > p_agree


def test_distance_uses_weaker_endpoint_spread():
    a = _merged([_hypothesis("r1", (0.0, 0.0, 0.0)), _hypothesis("r2", (0.04, 0.0, 0.0))])
    b = _merged([_hypothesis("r3", (5.0, 0.0, 0.0)), _hypothesis("r4", (5.005, 0.0, 0.0))])
    m = measure_distance(a, b)
    assert m.precision == pytest.approx(0.02)  # weaker endpoint's spread


def test_conflict_flagged_when_a_view_disagrees_with_its_peers():
    # Three tightly-agreeing views plus one 0.5 m outlier: the outlier
    # is statistically decidable against its peers (n>=3, leave-one-out)
    # -- the measurement must carry CONFLICT provenance, and the note
    # names the disagreement instead of hiding it.
    candidate = _merged([
        _hypothesis("r1", (0.0, 0.0, 0.0)),
        _hypothesis("r2", (0.01, 0.0, 0.0)),
        _hypothesis("r3", (0.02, 0.0, 0.0)),
        _hypothesis("r4", (0.5, 0.0, 0.0)),
    ])
    m = measure_dimensions(candidate)["width_m"]
    assert m.provenance is Provenance.CONFLICT
    assert "disagree" in m.precision_note.lower()


def test_agreeing_multi_view_stays_estimated_not_conflict():
    candidate = _merged([
        _hypothesis("r1", (0.0, 0.0, 0.0)),
        _hypothesis("r2", (0.01, 0.0, 0.0)),
        _hypothesis("r3", (0.02, 0.0, 0.0)),
    ])
    m = measure_dimensions(candidate)["width_m"]
    assert m.provenance is Provenance.ESTIMATED


def test_two_views_never_flagged_as_conflict():
    # Statistical honesty: with n=2 there is no majority to outlier
    # against -- noise and disagreement are indistinguishable, so a
    # conflict is never claimed that cannot be decided.
    candidate = _merged([
        _hypothesis("r1", (0.0, 0.0, 0.0)),
        _hypothesis("r2", (0.4, 0.0, 0.0)),
    ])
    m = measure_dimensions(candidate)["width_m"]
    assert m.provenance is Provenance.ESTIMATED
