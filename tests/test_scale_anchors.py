"""Scale anchoring maths on SYNTHETIC points with a known truth: proves the estimator, rejection and held-out logic, not any reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from engine.multiengine.scale_anchors import AnchorError, DistanceAnchor, PointAnchor, calibrate_distances, calibrate_points

RNG = np.random.default_rng(3)
TS = "2026-11-22T10:00:00+00:00"
TRUE_S = 0.37  # metres per gauge unit


def _d(i, a, b, *, noise=0.0, sigma=0.003, held=False, scale=TRUE_S):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = scale * np.linalg.norm(a - b) * (1 + noise)
    return DistanceAnchor(f"d{i}", tuple(a), tuple(b), m, sigma, "m", "laser_distance_meter", TS, "synthetic test", held_out=held)


def _pairs(n):
    out = []
    for _ in range(n):
        a = RNG.uniform(-5, 5, 3)
        v = RNG.normal(size=3)
        out.append((a, a + 4 * v / np.linalg.norm(v)))
    return out


def test_consistent_anchors_recover_the_scale_and_held_out_validates():
    ps = _pairs(6)
    cal = calibrate_distances([_d(i, a, b, noise=RNG.normal(0, 0.002), held=i >= 4) for i, (a, b) in enumerate(ps)])
    assert cal.status == "validated" and cal.applicable
    assert abs(cal.scale_m_per_gauge - TRUE_S) < 3 * cal.scale_sigma + 1e-3
    assert len(cal.held_out) == 2 and all(h["within_3_sigma"] for h in cal.held_out)


def test_an_inconsistent_anchor_is_rejected_by_name_and_the_rest_still_fit():
    ps = _pairs(6)
    anchors = [_d(i, a, b, noise=RNG.normal(0, 0.002)) for i, (a, b) in enumerate(ps)]
    anchors[2] = _d(2, *ps[2], noise=0.4)  # a mis-read tape: 40 % long
    cal = calibrate_distances(anchors)
    assert [r["anchor_id"] for r in cal.rejected] == ["d2"] and cal.status == "unvalidated" and cal.applicable
    assert abs(cal.scale_m_per_gauge - TRUE_S) < 3e-3


def test_single_anchor_and_two_disagreeing_anchors_are_not_applicable():
    a, b = _pairs(2)
    one = calibrate_distances([_d(0, *a)])
    assert one.status == "insufficient" and not one.applicable  # evidence reported, never applied
    two = calibrate_distances([_d(0, *a), _d(1, *b, noise=0.5)])
    assert two.status == "failed" and not two.applicable and two.scale_m_per_gauge is None


def test_a_held_out_anchor_that_is_not_predicted_fails_the_whole_calibration():
    ps = _pairs(5)
    anchors = [_d(i, a, b, noise=RNG.normal(0, 0.002)) for i, (a, b) in enumerate(ps[:4])] + [_d(9, *ps[4], noise=0.2, held=True)]
    cal = calibrate_distances(anchors)
    assert cal.status == "failed" and not cal.applicable and not cal.held_out[0]["within_3_sigma"]


def test_one_direction_only_anchors_warn_that_anisotropy_is_undetectable():
    anchors = [_d(i, [i, 0, 0], [i + 3, 0, 0]) for i in range(3)]
    assert any("anisotropy" in w for w in calibrate_distances(anchors).warnings)


def test_malformed_anchors_are_refused_not_repaired():
    base = dict(point_a=(0, 0, 0), point_b=(1, 0, 0), measured=1.0, sigma=0.01, unit="m", source="tape", measured_at=TS, provenance="x")
    for bad in ({"sigma": 0.0}, {"unit": "ft"}, {"measured_at": "yesterday"}, {"measured": -1.0}, {"point_b": (0, 0, 0)}, {"provenance": ""}):
        with pytest.raises(AnchorError):
            DistanceAnchor("x", **{**base, **bad})


def test_units_are_converted_and_duplicate_ids_refused():
    a = DistanceAnchor("m", (0, 0, 0), (2, 0, 0), 740.0, 3.0, "mm", "tape", TS, "x")
    assert a.metres == pytest.approx(0.74) and a.sigma_m == pytest.approx(0.003)
    with pytest.raises(AnchorError):
        calibrate_distances([a, a])


def _rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _pts(n, held_from, outlier=None, noise=0.005):
    R, t = _rot(0.8), np.array([2.0, -1.0, 0.5])
    out = []
    for i in range(n):
        p = RNG.uniform(-4, 4, 3)
        q = TRUE_S * R @ p + t + RNG.normal(0, noise, 3)
        if i == outlier:
            q += np.array([1.5, 0, 0])
        out.append(PointAnchor(f"p{i}", tuple(p), tuple(q), 0.01, "kinectfusion_pose", TS, "synthetic test", held_out=i >= held_from))
    return out


def test_point_anchors_recover_the_full_similarity_and_validate_on_held_out():
    cal = calibrate_points(_pts(10, held_from=7))
    assert cal.status == "validated" and cal.applicable
    assert abs(cal.scale_m_per_gauge - TRUE_S) < 5e-3 and cal.scale_sigma is not None
    assert np.allclose(np.asarray(cal.transform.R), _rot(0.8), atol=2e-2)
    assert all(h["within_tolerance"] for h in cal.held_out)


def test_point_outlier_is_rejected_and_too_few_points_are_insufficient():
    cal = calibrate_points(_pts(12, held_from=10, outlier=3))
    assert [r["anchor_id"] for r in cal.rejected] == ["p3"] and cal.status == "validated"
    few = calibrate_points(_pts(3, held_from=3))
    assert few.status == "insufficient" and not few.applicable and few.transform is None


def test_residuals_at_the_stated_noise_floor_are_not_rejected_as_outliers():
    """Regression (found on real data): with few points the MAD of tiny residuals collapses and a 1 cm residual looked like a 3-sigma outlier."""
    pts = _pts(8, held_from=8, noise=0.0005)
    cal = calibrate_points(pts)
    assert cal.rejected == [] and len(cal.fit_ids) == 8
