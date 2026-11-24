"""MiDaS inverse-depth normalisation: the shift must make the map usable without changing its ordering.

Why this test exists: MiDaS DPT returns unnormalised inverse depth, whose far tail is routinely negative. The
scale-alignment stage rejects negative maps (correctly, for depth), so every MiDaS map on a real run was refused
with DEPTH_MAP_NEGATIVE before any fitting happened. The fix is an exact affine reparametrisation, and these
assertions pin that it stays exact: positive, strictly ordered the same way, no information invented.
"""

from __future__ import annotations

import numpy as np

from engine.multiengine.adapters.depth_adapters import midas_inverse_depth_to_relative


def test_negative_inverse_depth_becomes_strictly_positive_and_keeps_its_ordering():
    raw = np.array([[-0.4, 0.2], [1.0, 3.5]], dtype=np.float32)
    out, facts = midas_inverse_depth_to_relative(raw)
    assert np.isfinite(out).all() and (out > 0).all(), "every value must be usable as a relative depth sample"
    assert facts["transform"] == "inverse_depth_shifted_to_positive"
    assert facts["raw_negative_fraction"] == 0.25
    # the transform is a constant offset: differences (and therefore the fitted slope) are unchanged
    assert np.allclose(np.diff(out.ravel().astype(np.float64)), np.diff(raw.astype(np.float64).ravel()), atol=1e-4)


def test_non_finite_pixels_are_zeroed_and_do_not_poison_the_range():
    raw = np.array([[np.nan, 1.0], [2.0, np.inf]], dtype=np.float32)
    out, facts = midas_inverse_depth_to_relative(raw)
    assert out[0, 0] == 0.0 and out[1, 1] == 0.0
    assert out[0, 1] > 0 and out[1, 0] > out[0, 1]
    assert facts["raw_min"] == 1.0 and facts["raw_max"] == 2.0


def test_an_all_non_finite_map_is_reported_rather_than_turned_into_fake_depth():
    out, facts = midas_inverse_depth_to_relative(np.full((4, 4), np.nan, dtype=np.float32))
    assert facts["status"] == "all_non_finite"
    assert not out.any(), "no value may be invented for a map with no measurement"
