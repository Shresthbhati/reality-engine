"""Regression: relative / SfM-gauge coordinates must never be compared as metres (defect B1)."""

from engine.multiengine.multi_view_association import (
    MultiViewEntityObservation,
    compute_cross_view_pair_score,
)


def _obs(oid, src, pos, units=None, dims=(1.0, 1.0, 1.0)):
    kw = {} if units is None else {"scale_units": units}
    return MultiViewEntityObservation(
        observation_id=oid, source_id=src, class_name="window", position=pos, dimensions=dims, **kw
    )


def test_default_units_are_not_metres():
    assert _obs("a", "i1", (0, 0, 0)).scale_units == "UNSPECIFIED"


def test_mixed_units_are_refused_not_compared():
    a = _obs("a", "i1", (0, 0, 0), "METRES")
    b = _obs("b", "i2", (0.01, 0, 0), "COLMAP_SFM_UNITS")
    s = compute_cross_view_pair_score(a, b)
    assert s.score == 0.0 and s.basis["reason"] == "mixed_scale_units"


def test_gauge_tolerance_scales_with_extent_not_metre_constants():
    # Tiny gauge model: a 0.3-unit gap is far larger than the 0.017-unit objects.
    # The old absolute 0.5 floor would have merged them as the 'same' entity.
    a = _obs("a", "i1", (0.0, 0, 0), "COLMAP_SFM_UNITS", dims=(0.01, 0.01, 0.01))
    b = _obs("b", "i2", (0.3, 0, 0), "COLMAP_SFM_UNITS", dims=(0.01, 0.01, 0.01))
    assert compute_cross_view_pair_score(a, b).score == 0.0
    c = _obs("c", "i2", (0.005, 0, 0), "COLMAP_SFM_UNITS", dims=(0.01, 0.01, 0.01))
    assert compute_cross_view_pair_score(a, c).score > 0.0


def test_metres_keep_physical_tolerance():
    a = _obs("a", "i1", (0.0, 0, 0), "METRES")
    b = _obs("b", "i2", (0.2, 0, 0), "METRES")
    assert compute_cross_view_pair_score(a, b).score > 0.0
