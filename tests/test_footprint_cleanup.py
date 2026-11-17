"""B10: footprints are cleaned without forcing right angles, and an open outline is reported partial."""
import numpy as np

from perception.architecture.footprint_cleanup import dominant_directions, footprint, snap


def _rect(w=5.0, h=4.0, jitter=0.01, seed=0):
    r = np.random.default_rng(seed)
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    seg = np.array([[c[i], c[(i + 1) % 4]] for i in range(4)])
    return seg + r.normal(0, jitter, seg.shape)


def test_noisy_rectangle_closes_with_four_edges():
    fp = footprint(_rect())
    assert fp.status == "closed" and len(fp.polygon) == 4


def test_missing_wall_is_partial_not_invented():
    seg = _rect()[:3]
    fp = footprint(seg)
    assert fp.status == "partial" and fp.closure_gap_m > 3.0


def test_a_100_degree_corner_is_not_squared():
    a = np.array([[0.0, 0.0], [4.0, 0.0]]); t = np.radians(100.0)
    b = np.array([[0.0, 0.0], [4 * np.cos(t), 4 * np.sin(t)]])
    seg = np.array([a, b])
    out = snap(seg, dominant_directions(seg))
    v = out[1, 1] - out[1, 0]
    assert abs(np.degrees(np.arctan2(v[1], v[0])) - 100.0) < 0.5


def test_small_misalignment_snaps_to_the_dominant_direction():
    seg = np.array([[[0, 0], [5, 0.0]], [[0, 1], [5, 1.15]], [[0, 2], [5, 2.0]]], float)   # middle one ~1.7 deg off
    out = snap(seg, dominant_directions(seg))
    slopes = (out[:, 1, 1] - out[:, 0, 1]) / (out[:, 1, 0] - out[:, 0, 0])
    assert np.ptp(slopes) < 1e-6          # all three now share the cluster's mean direction (parallel)
