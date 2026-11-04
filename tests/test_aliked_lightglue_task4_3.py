"""Tests for Blueprint Task 4.3: ALIKED + LightGlue Matcher.

Verifies:
- ALIKED + LightGlue operates behind IMatcher interface.
- Name, license (Apache-2.0), and commercial suitability (non_commercial=False) are verified.
- End-to-end matching on image pairs with fundamental matrix epipolar RANSAC verification.
- Benchmark comparison comparing registered ratio and inlier ratio against classical SIFT.
"""

import json
from pathlib import Path
import numpy as np
import cv2
import pytest

from reconstruction.matching import get_matcher, available_matchers, IMatcher, MatchResult


@pytest.fixture
def synthetic_image_pair(tmp_path: Path):
    """Creates a pair of synthetic test images with rich natural features and geometric transformation."""
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"

    # Generate a rich patterned image with text and geometric elements
    np.random.seed(42)
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(80):
        color = [int(x) for x in np.random.randint(50, 255, 3)]
        pt1 = (int(np.random.randint(10, 600)), int(np.random.randint(10, 440)))
        pt2 = (pt1[0] + int(np.random.randint(20, 80)), pt1[1] + int(np.random.randint(20, 80)))
        cv2.rectangle(img1, pt1, pt2, color, -1)
        cv2.putText(img1, f"FEAT_{i}", pt1, cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

    # Affine transformation: 1.5 deg rotation + translation
    M = cv2.getRotationMatrix2D((320, 240), 1.5, 1.0)
    M[0, 2] += 6.0
    M[1, 2] += 4.0
    img2 = cv2.warpAffine(img1, M, (640, 480))

    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)

    return {"im1": p1, "im2": p2}


def test_aliked_lightglue_imatcher_contract(synthetic_image_pair):
    matcher = get_matcher("aliked", synthetic_image_pair)
    assert isinstance(matcher, IMatcher)
    assert matcher.name == "lightglue@aliked"
    assert matcher.license == "Apache-2.0"
    assert matcher.non_commercial is False


def test_aliked_lightglue_matching_execution(synthetic_image_pair):
    matcher = get_matcher("aliked", synthetic_image_pair)
    res = matcher.match("im1", "im2")

    assert isinstance(res, MatchResult)
    assert res.matcher == "lightglue@aliked"
    assert res.matches > 50, f"Expected > 50 matches, got {res.matches}"
    assert res.inliers > 40, f"Expected > 40 inliers, got {res.inliers}"
    assert res.inlier_ratio > 0.60, f"Expected inlier ratio > 60%, got {res.inlier_ratio:.2f}"
    assert res.points_a.shape == (res.inliers, 2)
    assert res.points_b.shape == (res.inliers, 2)
    assert res.runtime_s > 0


def test_matcher_benchmark_diff_sift_vs_aliked(synthetic_image_pair):
    """Benchmarks SIFT vs ALIKED+LightGlue recording registered / inlier ratio comparison."""
    sift = get_matcher("sift", synthetic_image_pair)
    aliked = get_matcher("aliked", synthetic_image_pair)

    res_sift = sift.match("im1", "im2")
    res_aliked = aliked.match("im1", "im2")

    diff_report = {
        "benchmark": "sift_vs_aliked_lightglue",
        "sift": res_sift.to_dict(),
        "aliked_lightglue": res_aliked.to_dict(),
        "inlier_ratio_delta": round(res_aliked.inlier_ratio - res_sift.inlier_ratio, 4),
        "inlier_count_delta": res_aliked.inliers - res_sift.inliers,
    }

    # Verify both produced valid inliers
    assert res_sift.inliers > 0
    assert res_aliked.inliers > 0
    assert res_aliked.inlier_ratio >= 0.60

    # Write benchmark diff to reports
    out_dir = Path("docs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "matching_sift_vs_aliked.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diff_report, f, indent=2)
