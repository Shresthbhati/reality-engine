"""Tests for multi-view object entity resolution
(perception/instances/object_resolution.py)."""

from __future__ import annotations

import math

import pytest

from engine.math import Vec3
from perception.instances.lifting import ObjectHypothesis3D
from perception.instances.object_resolution import merge_hypotheses
from provenance import Uncertainty


def _hyp(region_id, evidence_id, label, x, y, z, confidence=0.8, half_extent=0.2) -> ObjectHypothesis3D:
    return ObjectHypothesis3D(
        region_id=region_id, evidence_id=evidence_id, label=label,
        position=Vec3(x, y, z),
        bounds_min=Vec3(x - half_extent, y - half_extent, z - half_extent),
        bounds_max=Vec3(x + half_extent, y + half_extent, z + half_extent),
        point_count=100, mask_pixel_count=120, confidence=confidence,
        uncertainty=Uncertainty(confidence=confidence),
    )


def test_close_same_label_hypotheses_merge_into_one_candidate():
    hyps = [
        _hyp("r1", "ev-1", "chair", 1.0, 0.0, 0.0),
        _hyp("r2", "ev-2", "chair", 1.1, 0.0, 0.0),
    ]
    candidates = merge_hypotheses(hyps)
    assert len(candidates) == 1
    assert candidates[0].observation_count == 2
    assert candidates[0].evidence_ids == ("ev-1", "ev-2")


def test_far_apart_same_label_hypotheses_stay_separate():
    hyps = [
        _hyp("r1", "ev-1", "chair", 0.0, 0.0, 0.0),
        _hyp("r2", "ev-2", "chair", 10.0, 0.0, 0.0),
    ]
    candidates = merge_hypotheses(hyps)
    assert len(candidates) == 2


def test_different_labels_never_merge_even_at_the_same_position():
    hyps = [
        _hyp("r1", "ev-1", "chair", 0.0, 0.0, 0.0),
        _hyp("r2", "ev-2", "table", 0.0, 0.0, 0.0),
    ]
    candidates = merge_hypotheses(hyps)
    assert len(candidates) == 2
    assert {c.label for c in candidates} == {"chair", "table"}


def test_transitive_merge_across_a_chain():
    # A-B within threshold, B-C within threshold, A-C is NOT within
    # threshold directly -- must still all merge into one candidate.
    hyps = [
        _hyp("r1", "ev-1", "chair", 0.0, 0.0, 0.0),
        _hyp("r2", "ev-2", "chair", 0.4, 0.0, 0.0),
        _hyp("r3", "ev-3", "chair", 0.8, 0.0, 0.0),
    ]
    candidates = merge_hypotheses(hyps, distance_threshold_m=0.5)
    assert len(candidates) == 1
    assert candidates[0].observation_count == 3


def test_no_source_hypothesis_is_discarded():
    hyps = [
        _hyp("r1", "ev-1", "chair", 0.0, 0.0, 0.0),
        _hyp("r2", "ev-2", "chair", 0.1, 0.0, 0.0),
    ]
    candidates = merge_hypotheses(hyps)
    assert set(h.region_id for h in candidates[0].source_hypotheses) == {"r1", "r2"}


def test_merged_bounds_are_the_union_never_smaller_than_any_single_view():
    hyps = [
        _hyp("r1", "ev-1", "chair", 0.0, 0.0, 0.0, half_extent=0.1),
        _hyp("r2", "ev-2", "chair", 0.2, 0.0, 0.0, half_extent=0.1),
    ]
    candidates = merge_hypotheses(hyps)
    c = candidates[0]
    assert c.bounds_min.x <= -0.1
    assert c.bounds_max.x >= 0.3


def test_agreement_across_multiple_views_raises_confidence():
    hyps = [
        _hyp("r1", "ev-1", "chair", 0.0, 0.0, 0.0, confidence=0.6),
        _hyp("r2", "ev-2", "chair", 0.1, 0.0, 0.0, confidence=0.6),
        _hyp("r3", "ev-3", "chair", 0.1, 0.1, 0.0, confidence=0.6),
    ]
    candidates = merge_hypotheses(hyps)
    assert candidates[0].confidence > 0.6


def test_single_hypothesis_confidence_unchanged():
    hyps = [_hyp("r1", "ev-1", "chair", 0.0, 0.0, 0.0, confidence=0.73)]
    candidates = merge_hypotheses(hyps)
    assert math.isclose(candidates[0].confidence, 0.73, abs_tol=1e-9)


def test_empty_input_returns_empty_list():
    assert merge_hypotheses([]) == []


def test_rejects_negative_distance_threshold():
    with pytest.raises(ValueError):
        merge_hypotheses([_hyp("r1", "ev-1", "chair", 0, 0, 0)], distance_threshold_m=-1.0)


def test_deterministic_regardless_of_input_order():
    a = _hyp("r1", "ev-1", "chair", 0.0, 0.0, 0.0)
    b = _hyp("r2", "ev-2", "chair", 0.1, 0.0, 0.0)
    forward = merge_hypotheses([a, b])
    backward = merge_hypotheses([b, a])
    assert [c.to_dict() for c in forward] == [c.to_dict() for c in backward]


def test_to_dict_is_plain_data():
    hyps = [_hyp("r1", "ev-1", "chair", 0.0, 0.0, 0.0)]
    payload = merge_hypotheses(hyps)[0].to_dict()
    assert payload["label"] == "chair"
    assert payload["observation_count"] == 1
