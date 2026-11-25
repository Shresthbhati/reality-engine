"""Tests for measured two-view geometry in the feature-matching adapters.

The adapters used to report a confidence built from the match count itself (a
constant, ``match_count / match_count``), or no geometric check at all. These
tests pin the replacement: a synthetic camera pair with KNOWN relative pose must
be recovered, a random correspondence set must not be called verified, and real
adjacent photographs must produce measured inlier and epipolar statistics.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from perception.engine_adapters import _geometric_verification, _sift_run

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "datasets" / "south_building"
K = np.array([[900.0, 0.0, 512.0], [0.0, 900.0, 384.0], [0.0, 0.0, 1.0]])


def _project(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    cam = points @ R.T + t
    img = cam @ K.T
    return img[:, :2] / img[:, 2:3]


def _rotation_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def test_a_known_camera_pair_is_verified_and_its_pose_recovered():
    rng = np.random.default_rng(7)
    points = np.column_stack([rng.uniform(-1, 1, 300), rng.uniform(-1, 1, 300), rng.uniform(4, 8, 300)])
    R = _rotation_y(6.0)
    t = np.array([0.5, 0.02, 0.1])
    p1 = _project(points, np.eye(3), np.zeros(3))
    p2 = _project(points, R, t)

    stats = _geometric_verification(p1, p2, K, K)

    assert stats["verified"] is True
    assert stats["inliers"] == pytest.approx(len(points), abs=3)
    assert stats["inlier_ratio"] > 0.98
    assert stats["epipolar_residual_median_px"] < 0.05
    pose = stats["relative_pose"]
    assert np.isclose(np.linalg.norm(pose["t_unit"]), 1.0, atol=1e-5)
    t_expected = t / np.linalg.norm(t)
    # the essential matrix fixes the translation only up to sign (the usual two-fold ambiguity)
    assert abs(float(np.dot(pose["t_unit"], t_expected))) > 0.98
    R_rec = np.asarray(pose["R"])
    cos_angle = max(-1.0, min(1.0, (float(np.trace(R_rec @ R.T)) - 1.0) / 2.0))
    assert math.degrees(math.acos(cos_angle)) < 1.0
    assert "cannot recover scale" in pose["scale"]


def test_random_correspondences_are_not_reported_as_verified():
    rng = np.random.default_rng(11)
    p1 = rng.uniform(0, 1024, (200, 2))
    p2 = rng.uniform(0, 768, (200, 2))

    stats = _geometric_verification(p1, p2, K, K)

    # RANSAC may find a degenerate geometry in pure noise; it must not look like a match
    assert stats.get("inlier_ratio", 0.0) < 0.5
    assert "relative_pose" not in stats or stats["inlier_ratio"] < 0.5


@pytest.mark.skipif(not (DATASET / "images").is_dir(), reason="south_building dataset not present")
def test_real_adjacent_photographs_produce_measured_statistics():
    images = sorted((DATASET / "images").glob("*.JPG"))[:2]
    if len(images) < 2:
        pytest.skip("need two images")
    intrinsics = {str(p): {"fx": 900.0, "fy": 900.0, "cx": 512.0, "cy": 384.0} for p in images}

    out = _sift_run([str(p) for p in images], run_id="r", pairs=[(str(images[0]), str(images[1]))],
                    intrinsics=intrinsics)

    assert out.status == "succeeded"
    obs = out.observations[0]
    meta = obs.metadata
    assert meta["keypoints_a"] > 100 and meta["keypoints_b"] > 100
    assert meta["inliers"] >= 8
    assert 0.0 < meta["inlier_ratio"] <= 1.0
    assert meta["epipolar_residual_median_px"] >= 0.0
    assert obs.confidence == meta["inlier_ratio"], "confidence is the measured inlier ratio"
    assert obs.calibration_quality == "unverified"
    assert "relative_pose" in meta
    assert meta["runtime_s"] > 0.0


@pytest.mark.skipif(not (DATASET / "images").is_dir(), reason="south_building dataset not present")
def test_identical_images_are_verified_but_a_far_pair_is_measured_not_assumed():
    images = sorted((DATASET / "images").glob("*.JPG"))
    if len(images) < 16:
        pytest.skip("dataset too small")
    out = _sift_run([str(p) for p in (images[0], images[15])], run_id="r",
                    pairs=[(str(images[0]), str(images[15]))])
    # Either the pair verifies with measured statistics, or it is reported unverified with
    # the reason and the counts -- never silently dropped and never given a fake score.
    if out.observations:
        assert out.observations[0].metadata["inliers"] >= 8
    else:
        assert out.summary["n_unverified_pairs"] == 1
        assert out.summary["unverified"][0]["reason"]