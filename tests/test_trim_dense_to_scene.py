"""Dense monocular-depth points far outside the SfM scene are dropped before plane detection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from reconstruction.depth_to_points import trim_to_scene


def _pts(arr):
    return [SimpleNamespace(position=tuple(map(float, p))) for p in arr]


def test_far_depth_is_dropped_and_room_points_survive():
    rng = np.random.default_rng(1)
    sparse = _pts(rng.uniform(-1.5, 1.5, size=(300, 3)))                  # the room: ~3 units across
    room = _pts(rng.uniform(-1.6, 1.6, size=(200, 3)))                     # dense points on the room's surfaces
    window_and_sky = _pts(rng.uniform(20, 200, size=(50, 3)))              # hallucinated range beyond the window
    kept, dropped = trim_to_scene(room + window_and_sky, sparse)
    assert dropped == 50 and len(kept) == 200
    assert all(abs(c) < 3 for p in kept for c in p.position)


def test_too_few_reference_points_means_no_scene_and_no_trimming():
    sparse = _pts(np.zeros((10, 3)))
    dense = _pts([(100.0, 100.0, 100.0)])
    kept, dropped = trim_to_scene(dense, sparse)
    assert dropped == 0 and kept == dense
    assert trim_to_scene([], _pts(np.zeros((100, 3)))) == ([], 0)
