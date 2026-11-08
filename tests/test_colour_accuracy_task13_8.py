"""Unit tests for Colour Accuracy and Delta-E Metrics under Controlled Capture (Task 13.8)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from perception.materials.color_accuracy import (
    COLORCHECKER_D65_LAB,
    COLORCHECKER_D65_PATCHES,
    COLORCHECKER_D65_SRGB,
    ColorAccuracyTier,
    ColorCorrectionMatrix,
    ColourAccuracyReport,
    assess_surface_color_accuracy,
    calibrate_color_chart,
    delta_e76,
    delta_e94,
    delta_e2000,
    estimate_cct_mccamy,
    lab_to_rgb,
    lab_to_xyz,
    linear_to_srgb,
    rgb_to_lab,
    rgb_to_xyz,
    srgb_to_linear,
    xyz_to_lab,
    xyz_to_rgb,
)


# =============================================================================
# 1. Colour Space Conversion Fidelity Tests
# =============================================================================

def test_srgb_linear_roundtrip():
    test_colors = np.array([
        [0, 0, 0],
        [255, 255, 255],
        [128, 128, 128],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [235, 180, 95],
        [45, 120, 210],
    ], dtype=np.float64)

    lin = srgb_to_linear(test_colors)
    rec = linear_to_srgb(lin)

    np.testing.assert_allclose(rec, test_colors, atol=0.01)


def test_d65_white_point_conversions():
    # White (255, 255, 255)
    xyz_white = rgb_to_xyz([255, 255, 255])
    # D65 white point: X=0.95047, Y=1.00000, Z=1.08883
    assert xyz_white[0] == pytest.approx(0.95047, abs=1e-3)
    assert xyz_white[1] == pytest.approx(1.00000, abs=1e-3)
    assert xyz_white[2] == pytest.approx(1.08883, abs=1e-3)

    lab_white = xyz_to_lab(xyz_white)
    assert lab_white[0] == pytest.approx(100.0, abs=0.1)
    assert lab_white[1] == pytest.approx(0.0, abs=0.5)
    assert lab_white[2] == pytest.approx(0.0, abs=0.5)

    # Black (0, 0, 0)
    lab_black = rgb_to_lab([0, 0, 0])
    assert lab_black[0] == pytest.approx(0.0, abs=0.1)
    assert lab_black[1] == pytest.approx(0.0, abs=0.1)
    assert lab_black[2] == pytest.approx(0.0, abs=0.1)


def test_rgb_lab_roundtrip():
    test_rgbs = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (200, 150, 100),
        (50, 80, 120),
        (240, 240, 235),
    ]

    for rgb in test_rgbs:
        lab = rgb_to_lab(rgb)
        rec = lab_to_rgb(lab)
        de = delta_e76(rgb, rec)
        assert de < 0.05, f"Round-trip delta-E too high ({de:.4f}) for {rgb}"


# =============================================================================
# 2. Perceptual Delta-E Metric Tests (CIE76, CIE94, CIEDE2000)
# =============================================================================

def test_delta_e_identity_and_symmetry():
    c1 = (180, 120, 90)
    c2 = (175, 125, 95)

    # 1. Identity: delta-E to self is 0 across all formulas
    assert delta_e76(c1, c1) == 0.0
    assert delta_e94(c1, c1) == 0.0
    assert delta_e2000(c1, c1) == 0.0

    # 2. Symmetry: CIE76 and CIEDE2000 are mathematically symmetric
    assert delta_e76(c1, c2) == pytest.approx(delta_e76(c2, c1), abs=1e-6)
    assert delta_e2000(c1, c2) == pytest.approx(delta_e2000(c2, c1), abs=1e-6)

    # 3. CIE94 asymmetry: first argument is reference standard per CIE spec
    de94_forward = delta_e94(c1, c2)
    de94_backward = delta_e94(c2, c1)
    assert de94_forward > 0.0 and de94_backward > 0.0
    assert abs(de94_forward - de94_backward) < 0.3


def test_ciede2000_known_properties():
    # 1. Subtle imperceptible variation: JND threshold (~1.0 - 2.0)
    subtle1 = (200, 200, 200)
    subtle2 = (202, 201, 200)
    de00_subtle = delta_e2000(subtle1, subtle2)
    assert 0.1 < de00_subtle < 1.5, f"Expected subtle delta-E, got {de00_subtle}"

    # 2. Extreme color difference: complementary colors
    red = (255, 0, 0)
    green = (0, 255, 0)
    de00_extreme = delta_e2000(red, green)
    assert de00_extreme > 60.0

    # 3. Lab input directly
    lab_a = (50.0, 2.6772, -79.7751)
    lab_b = (50.0, 0.0000, -82.7485)
    de00_lab = delta_e2000(lab_a, lab_b, is_lab=True)
    assert de00_lab > 0.0 and de00_lab < 10.0


# =============================================================================
# 3. ColorChecker Reference Chart Verification
# =============================================================================

def test_colorchecker_classic_data():
    assert len(COLORCHECKER_D65_PATCHES) == 24
    assert len(COLORCHECKER_D65_SRGB) == 24
    assert len(COLORCHECKER_D65_LAB) == 24

    # Neutral scale patches (indices 18 to 23: White 9.5 to Black 2)
    neutral_labs = COLORCHECKER_D65_LAB[18:24]
    lightness_values = [p[0] for p in neutral_labs]

    # Lightness must strictly decrease monotonically from white to black
    assert lightness_values[0] > 90.0  # White 9.5
    assert lightness_values[-1] < 25.0  # Black 2
    for i in range(len(lightness_values) - 1):
        assert lightness_values[i] > lightness_values[i + 1]


def test_estimate_cct():
    # Standard D65 white should yield CCT near 6500 K
    d65_white = (243, 243, 242)
    cct_d65 = estimate_cct_mccamy(d65_white)
    assert cct_d65 == pytest.approx(6500.0, abs=600.0)

    # Warm incandescent white (high red, low blue) should yield low CCT (~2800K - 3500K)
    warm_white = (255, 195, 120)
    cct_warm = estimate_cct_mccamy(warm_white)
    assert cct_warm < 4500.0


# =============================================================================
# 4. Controlled Capture Chart Calibration (CCM) Tests
# =============================================================================

def test_color_chart_calibration():
    # Simulate a captured chart under warm lighting with exposure shift
    # Simulated camera transform: warm tint (R * 1.18, G * 0.96, B * 0.72)
    gain = np.array([1.18, 0.96, 0.72])

    observed_patches = []
    for rgb in COLORCHECKER_D65_SRGB:
        lin = srgb_to_linear(rgb)
        shifted_lin = lin * gain
        obs_srgb = linear_to_srgb(shifted_lin)
        observed_patches.append((int(obs_srgb[0]), int(obs_srgb[1]), int(obs_srgb[2])))

    # Run calibration
    res = calibrate_color_chart(observed_patches, COLORCHECKER_D65_SRGB)

    # Error before calibration should be substantial
    assert res.mean_de00_before > 5.0

    # Error after calibration should drop significantly (commercial accuracy < 2.0 delta_E)
    assert res.mean_de00_after < 2.0
    assert res.median_de00_after < 1.5
    assert res.mean_de00_after < res.mean_de00_before * 0.35

    # Check serialization
    d = res.to_dict()
    assert "ccm" in d
    assert "mean_de00_after" in d
    assert "estimated_cct_k" in d


# =============================================================================
# 5. Controlled Surface Colour Accuracy Assessment Tests
# =============================================================================

def test_assess_surface_color_accuracy_excellent():
    # Controlled capture with very tight agreement (< 1.5 delta_e)
    ground_truth = {
        "wall_north": (230, 225, 215),
        "wall_south": (210, 205, 200),
        "floor_slab": (140, 110, 80),
        "ceiling": (245, 245, 242),
    }

    # Observed surfaces with tiny delta
    observed = {
        "wall_north": (231, 224, 216),
        "wall_south": (211, 206, 199),
        "floor_slab": (141, 109, 81),
        "ceiling": (244, 246, 242),
    }

    report = assess_surface_color_accuracy(observed, ground_truth, illuminant="D65_Controlled")

    assert isinstance(report, ColourAccuracyReport)
    assert report.sample_count == 4
    assert report.mean_delta_e00 <= 2.0
    assert report.overall_tier == ColorAccuracyTier.EXCELLENT

    d = report.to_dict()
    assert d["overall_tier"] == "EXCELLENT"
    assert len(d["surfaces"]) == 4
    assert d["illuminant"] == "D65_Controlled"


def test_assess_surface_color_accuracy_acceptable_and_marginal():
    ground_truth = {
        "wall_1": (220, 220, 220),
        "wall_2": (180, 160, 140),
        "door_leaf": (100, 60, 40),
    }

    # Moderate shift ~3.0 - 5.0 delta-E
    observed = {
        "wall_1": (210, 215, 225),  # slight blue shift
        "wall_2": (175, 165, 145),
        "door_leaf": (110, 65, 35),
    }

    report = assess_surface_color_accuracy(observed, ground_truth)
    assert report.overall_tier in (ColorAccuracyTier.ACCEPTABLE, ColorAccuracyTier.MARGINAL)
    assert report.mean_delta_e00 < 8.0


def test_assess_surface_color_accuracy_unacceptable():
    ground_truth = {
        "wall_green": (40, 180, 50),
        "carpet_red": (200, 30, 40),
    }

    # Severe mismatch (swapped or wrong color)
    observed = {
        "wall_green": (180, 40, 50),
        "carpet_red": (40, 180, 50),
    }

    report = assess_surface_color_accuracy(observed, ground_truth)
    assert report.overall_tier == ColorAccuracyTier.UNACCEPTABLE
    assert report.mean_delta_e00 > 25.0
