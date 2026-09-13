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
