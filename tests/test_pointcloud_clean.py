"""Statistical outlier removal: both backends must agree on what an outlier is, and say which one ran."""
from __future__ import annotations

import numpy as np
import pytest

from reconstruction.pointcloud_clean import remove_statistical_outliers


def _cloud(seed=0):
    rng = np.random.default_rng(seed)
    blob = rng.normal(0.0, 0.2, (400, 3))
    far = np.array([[8.0, 8.0, 8.0], [-9.0, 7.0, 6.0], [10.0, -10.0, 5.0]])
    return np.vstack([blob, far])


@pytest.mark.parametrize("prefer", ["scipy", "open3d"])
def test_far_points_are_removed_and_the_blob_survives(prefer):
    keep, facts = remove_statistical_outliers(_cloud(), prefer=prefer)
    assert not keep[-3:].any(), "the three far points must be flagged"
    assert keep[:400].mean() > 0.9, "almost the whole blob must survive"
    assert facts["n_in"] == 403 and facts["n_kept"] == int(keep.sum())
    assert facts["backend"] in ("scipy", "open3d")


def test_the_two_backends_agree_on_the_outliers():
    pytest.importorskip("open3d")
    a, fa = remove_statistical_outliers(_cloud(1), prefer="scipy")
    b, fb = remove_statistical_outliers(_cloud(1), prefer="open3d")
    assert fa["backend"] == "scipy" and fb["backend"] == "open3d"
    assert (a != b).sum() <= 3, "the same definition: at most a boundary point or two may differ"


def test_too_few_points_are_not_filtered_by_a_noise_statistic():
    keep, facts = remove_statistical_outliers(np.zeros((5, 3)))
    assert keep.all() and facts["backend"] == "none" and "too few" in facts["note"]


def test_non_finite_points_are_always_removed():
    pts = _cloud()
    pts[3] = [np.nan, 0.0, 0.0]
    keep, facts = remove_statistical_outliers(pts)
    assert not keep[3] and facts["n_kept"] == int(keep.sum())
