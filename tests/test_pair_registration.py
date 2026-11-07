"""Photo-to-photo registration from metric depth + matched features (so the engine can fuse photos with no SfM).

Ground truth is a textured ray-cast room seen from known poses (tests/layout_synth.py), so errors are exact.
"""
import numpy as np
import pytest

from perception.layout.registration import register_pair
from tests.layout_synth import relative_pose, render, rotation_error_deg


def _register(a, b):
    return register_pair(a.rgb, a.depth, b.rgb, b.depth, a.hfov)


@pytest.mark.parametrize("src", [(0.8, -0.3, 25.0), (-0.6, 0.5, -30.0), (0.2, 0.2, 12.0)])
def test_recovers_the_relative_pose_between_two_views(src):
    a = render(0.0, 0.0, 0.0, 5.0)
    b = render(src[0], src[1], src[2], 5.0)
    reg = _register(a, b)
    assert reg is not None, "two overlapping views of a textured room must register"
    R, t = relative_pose(a, b)
    assert rotation_error_deg(reg.R, R) < 3.0
    assert np.linalg.norm(reg.t - t) < 0.15
    assert reg.inliers >= 25 and reg.rms_m < 0.1


def test_a_pitch_difference_is_recovered_too():
    a = render(0.0, 0.0, 0.0, 0.0)
    b = render(0.5, 0.0, 15.0, 10.0)
    reg = _register(a, b)
    assert reg is not None
    R, t = relative_pose(a, b)
    assert rotation_error_deg(reg.R, R) < 3.5 and np.linalg.norm(reg.t - t) < 0.2


def test_views_that_share_nothing_are_refused_not_guessed():
    a = render(0.0, 0.0, 0.0, 5.0)
    b = render(0.0, 0.0, 180.0, 5.0)  # looking the opposite way: no common surface
    assert _register(a, b) is None


def test_featureless_images_are_refused():
    a = render(0.0, 0.0, 0.0, 5.0)
    flat = np.full_like(a.rgb, 120)
    assert register_pair(a.rgb, a.depth, flat, a.depth, a.hfov) is None


def test_a_wrong_depth_map_does_not_produce_a_confident_pose():
    a = render(0.0, 0.0, 0.0, 5.0)
    b = render(0.8, -0.3, 25.0, 5.0)
    rng = np.random.default_rng(0)
    junk = rng.uniform(0.5, 9.0, b.depth.shape)  # scrambled geometry: matches cannot agree on one rigid motion
    reg = register_pair(a.rgb, a.depth, b.rgb, junk, a.hfov)
    assert reg is None or reg.inliers < 25
