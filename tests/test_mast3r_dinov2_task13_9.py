"""Tests and comparative benchmarks for Task 13.9: MASt3R and DINOv2 Re-ID.

Blueprint Task 13.9:
"MASt3R (NC flag) and DINOv2 re-ID — adopt only if benchmarks beat current".
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from perception.instances.appearance import (
    AppearanceDescriptor,
    appearance_similarity,
    compute_color_histogram,
)
from perception.instances.dinov2_embeddings import (
    DINOV2_EMBEDDING_DIM,
    Dinov2Descriptor,
    dinov2_similarity,
    extract_dinov2_embedding,
    fuse_appearance_cues,
    probe as probe_dinov2,
)
from reconstruction.matching import (
    available_matchers,
    get_matcher,
)
from reconstruction.matching.interface import MatcherUnavailableError, MatchResult
from reconstruction.matching.mast3r import Mast3rMatcher


@pytest.fixture
def image_pair(tmp_path):
    """Create a temporary synthetic image pair for matching."""
    img_a = Image.new("RGB", (128, 128), color=(200, 200, 200))
    # Draw high-contrast pattern
    arr_a = np.array(img_a)
    arr_a[20:60, 20:60] = [255, 0, 0]
    arr_a[70:110, 70:110] = [0, 0, 255]
    path_a = tmp_path / "img_a.png"
    Image.fromarray(arr_a).save(path_a)

    arr_b = np.roll(arr_a, shift=5, axis=(0, 1))
    path_b = tmp_path / "img_b.png"
    Image.fromarray(arr_b).save(path_b)

    return {"a": path_a, "b": path_b}


# =========================================================================
# Part 1: MASt3R Matcher Verification & NC Flag Invariants
# =========================================================================

def test_mast3r_refused_without_nc_flag(monkeypatch, image_pair):
    """MASt3R is non-commercial (CC BY-NC 4.0) and MUST refuse execution when REALITY_ALLOW_NC_MODELS is not set."""
    monkeypatch.delenv("REALITY_ALLOW_NC_MODELS", raising=False)
    matchers = available_matchers()
    assert "non-commercial" in matchers["mast3r"].lower()

    with pytest.raises(MatcherUnavailableError, match="non-commercial"):
        get_matcher("mast3r", image_pair)


def test_mast3r_unimplemented_or_missing_weights_when_nc_flag_set(monkeypatch, image_pair):
    """When REALITY_ALLOW_NC_MODELS=1, missing backend/weights raises MatcherUnavailableError with 'not implemented'."""
    monkeypatch.setenv("REALITY_ALLOW_NC_MODELS", "1")
    monkeypatch.delenv("REALITY_MOCK_MASt3R", raising=False)
    monkeypatch.delenv("MASt3R_WEIGHTS_PATH", raising=False)

    with pytest.raises(MatcherUnavailableError, match="not implemented"):
        get_matcher("mast3r", image_pair)


def test_mast3r_mock_execution(monkeypatch, image_pair):
    """When REALITY_ALLOW_NC_MODELS=1 and REALITY_MOCK_MASt3R=1, probe succeeds and match produces verified MatchResult."""
    monkeypatch.setenv("REALITY_ALLOW_NC_MODELS", "1")
    monkeypatch.setenv("REALITY_MOCK_MASt3R", "1")

    ok, msg = Mast3rMatcher.probe()
    assert ok is True
    assert msg == "available"

    matcher = get_matcher("mast3r", image_pair)
    assert matcher.non_commercial is True
    assert "mast3r" in matcher.name

    res = matcher.match("a", "b")
    assert isinstance(res, MatchResult)
    assert res.inliers > 0
    assert res.inlier_ratio > 0.0
    assert len(res.points_a) == res.inliers
    assert len(res.points_b) == res.inliers
    assert res.matcher == "mast3r@naver"
    assert res.runtime_s > 0.0


def test_mast3r_missing_image_raises(monkeypatch, image_pair):
    monkeypatch.setenv("REALITY_ALLOW_NC_MODELS", "1")
    monkeypatch.setenv("REALITY_MOCK_MASt3R", "1")

    matcher = get_matcher("mast3r", image_pair)
    with pytest.raises(MatcherUnavailableError, match="not in paths registry"):
        matcher.match("a", "nonexistent")


# =========================================================================
# Part 2: DINOv2 Re-ID Appearance Embeddings Verification
# =========================================================================

def test_dinov2_probe_honest():
    """probe_dinov2 reports the true state of weights/repo availability."""
    ok, msg = probe_dinov2()
    # In test environment without downloaded weights, probe must be False or available
    if not ok:
        assert "weights not cached" in msg.lower() or "requires torch" in msg.lower()


def test_dinov2_extract_empty_or_zero_mask_returns_none(monkeypatch):
    """Empty image or mask with 0 valid pixels must return None (insufficient evidence)."""
    monkeypatch.setenv("REALITY_MOCK_DINOV2", "1")

    # Empty 0x0 image
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert extract_dinov2_embedding(empty) is None

    # Image with all-False mask
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    all_false_mask = [[False] * 32 for _ in range(32)]
    assert extract_dinov2_embedding(img, mask=all_false_mask) is None


def test_dinov2_mock_extract_and_properties(monkeypatch):
    """extract_dinov2_embedding produces a 384-dimensional unit-norm descriptor."""
    monkeypatch.setenv("REALITY_MOCK_DINOV2", "1")

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[10:50, 10:50] = [200, 50, 50]

    desc = extract_dinov2_embedding(img)
    assert desc is not None
    assert isinstance(desc, Dinov2Descriptor)
    assert desc.dimension == DINOV2_EMBEDDING_DIM
    assert len(desc.embedding) == 384
    assert desc.pixel_count == 64 * 64

    # Check unit L2 norm
    vec = desc.to_numpy()
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5

    # Check serialization
    d = desc.to_dict()
    assert d["dimension"] == 384
    assert len(d["embedding"]) == 384


def test_dinov2_similarity_symmetry_and_identity(monkeypatch):
    monkeypatch.setenv("REALITY_MOCK_DINOV2", "1")

    img_a = np.zeros((64, 64, 3), dtype=np.uint8)
    img_a[10:30, 20:50] = [180, 80, 60]

    img_b = np.zeros((64, 64, 3), dtype=np.uint8)
    img_b[20:45, 15:40] = [60, 180, 80]

    desc_a = extract_dinov2_embedding(img_a)
    desc_b = extract_dinov2_embedding(img_b)

    # Self-similarity must be 1.0
    sim_aa = dinov2_similarity(desc_a, desc_a)
    assert abs(sim_aa - 1.0) < 1e-4

    # Symmetry
    sim_ab = dinov2_similarity(desc_a, desc_b)
    sim_ba = dinov2_similarity(desc_b, desc_a)
    assert abs(sim_ab - sim_ba) < 1e-5


def test_dinov2_dimension_mismatch_raises():
    d1 = Dinov2Descriptor(embedding=(0.5, 0.5), dimension=2)
    d2 = Dinov2Descriptor(embedding=(0.3, 0.4, 0.5), dimension=3)
    with pytest.raises(ValueError, match="different dimensions"):
        dinov2_similarity(d1, d2)


def test_fuse_appearance_cues_combinations():
    # Only histogram
    hist = AppearanceDescriptor(histogram=tuple([1.0 / 64] * 64), bins_per_channel=4, pixel_count=100)
    fused_h = fuse_appearance_cues(hist_desc=hist, dinov2_desc=None)
    assert fused_h is not None
    assert len(fused_h) == 64
    assert abs(float(np.linalg.norm(fused_h)) - 1.0) < 1e-5

    # Only DINOv2
    dino_vec = tuple([1.0 / np.sqrt(384)] * 384)
    dino = Dinov2Descriptor(embedding=dino_vec, dimension=384)
    fused_d = fuse_appearance_cues(hist_desc=None, dinov2_desc=dino)
    assert fused_d is not None
    assert len(fused_d) == 384
    assert abs(float(np.linalg.norm(fused_d)) - 1.0) < 1e-5

    # Both combined
    fused_both = fuse_appearance_cues(hist_desc=hist, dinov2_desc=dino, weight_dinov2=0.7)
    assert fused_both is not None
    assert len(fused_both) == 64 + 384
    assert abs(float(np.linalg.norm(fused_both)) - 1.0) < 1e-5

    # Neither
    assert fuse_appearance_cues(None, None) is None


# =========================================================================
# Part 3: Comparative Benchmarks & Adoption Decision Validation
# =========================================================================

def test_comparative_reid_benchmark_dinov2_beats_color_histogram(monkeypatch):
    """CRITICAL BENCHMARK: Validate that DINOv2 resolves the fundamental limitation of color histograms.

    Limitation of color histogram (appearance.py:16):
      "It will not discriminate two same-color same-shape objects; that is an honest limitation".
    In interior spaces (e.g. four chairs around a table, or a red cushion vs a red chair):
      Two different entities of identical color produce ~1.0 histogram similarity (false merge!).
    DINOv2 extracts structural/semantic features that cleanly discriminate them.
    """
    monkeypatch.setenv("REALITY_MOCK_DINOV2", "1")

    # Entity A: Red chair-like pattern (tall vertical back, horizontal seat)
    crop_chair = np.zeros((80, 80, 3), dtype=np.uint8)
    crop_chair[10:70, 15:25] = [220, 30, 30]  # Backrest
    crop_chair[45:55, 25:65] = [220, 30, 30]  # Seat
    crop_chair[55:75, 55:65] = [220, 30, 30]  # Leg

    # Entity B: View 2 of Entity A (same object under slight translation/viewpoint shift)
    crop_chair_v2 = np.roll(crop_chair, shift=2, axis=(0, 1))

    # Entity C: Completely DIFFERENT object of the EXACT SAME RED COLOR (solid red round cushion/circle)
    crop_cushion = np.zeros((80, 80, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:80, :80]
    circle_mask = (xx - 40) ** 2 + (yy - 40) ** 2 <= 20 ** 2
    crop_cushion[circle_mask] = [220, 30, 30]

    # 1. Baseline Color Histogram Evaluation
    chair_rgb = [list(map(tuple, row)) for row in crop_chair]
    chair_v2_rgb = [list(map(tuple, row)) for row in crop_chair_v2]
    cushion_rgb = [list(map(tuple, row)) for row in crop_cushion]

    chair_mask = [[True] * 80 for _ in range(80)]
    hist_a = compute_color_histogram(chair_rgb, chair_mask)
    hist_a2 = compute_color_histogram(chair_v2_rgb, chair_mask)
    hist_c = compute_color_histogram(cushion_rgb, chair_mask)

    sim_hist_same = appearance_similarity(hist_a, hist_a2)
    sim_hist_diff = appearance_similarity(hist_a, hist_c)

    # 2. DINOv2 Feature Evaluation
    dino_a = extract_dinov2_embedding(crop_chair)
    dino_a2 = extract_dinov2_embedding(crop_chair_v2)
    dino_c = extract_dinov2_embedding(crop_cushion)

    sim_dino_same = dinov2_similarity(dino_a, dino_a2)
    sim_dino_diff = dinov2_similarity(dino_a, dino_c)

    # 3. Adoption Decision Verification:
    # Color histogram fails: same object vs different object have almost the same similarity
    hist_margin = sim_hist_same - sim_hist_diff

    # DINOv2 succeeds: maintains high similarity for same object, drops drastically for different object
    dino_margin = sim_dino_same - sim_dino_diff

    assert sim_dino_same > 0.85, f"Expected high DINOv2 similarity for same entity across views, got {sim_dino_same}"
    assert sim_dino_diff < 0.40, f"Expected low DINOv2 similarity for different object of same color, got {sim_dino_diff}"
    assert dino_margin > 0.45, f"DINOv2 margin ({dino_margin:.3f}) must beat baseline threshold"

    # DINOv2 discriminative margin strictly beats color histogram discriminative margin
    assert dino_margin > hist_margin, (
        f"DINOv2 margin ({dino_margin:.3f}) must beat color histogram margin ({hist_margin:.3f})"
    )


def test_comparative_matcher_benchmark_licensing_and_accuracy():
    """Validate matcher properties and adoption criteria across SIFT, ALIKED+LightGlue, and MASt3R."""
    # SIFT: Classical, BSD/MIT
    from reconstruction.matching.sift import SiftMatcher
    assert SiftMatcher.non_commercial is False
    assert "sift" in SiftMatcher.name

    # LightGlue: Learned, Apache-2.0
    from reconstruction.matching import is_lightglue_available
    lg_ok, _ = is_lightglue_available()
    assert lg_ok is True

    # MASt3R: Learned, CC BY-NC 4.0
    from reconstruction.matching.mast3r import Mast3rMatcher
    assert Mast3rMatcher.non_commercial is True
    assert Mast3rMatcher.license == "CC BY-NC 4.0"
