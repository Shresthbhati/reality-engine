"""Colour science, delta-E metrics, and controlled-capture calibration (Task 13.8).

Provides:
1. Standard Colour Space Conversions:
   - sRGB <-> Linear RGB <-> CIEXYZ <-> CIELAB (D65 illuminant, 2 deg observer).
2. Perceptual Difference Metrics:
   - delta_e76: CIE 1976 Euclidean distance in CIELAB space.
   - delta_e94: CIE 1994 formula with lightness, chroma, and hue weighting.
   - delta_e2000: Full CIEDE2000 standard formula (ISO/CIE 11664-6).
3. Controlled Capture Reference Targets:
   - ColorChecker Classic 24-patch standard reference values (sRGB & CIELAB D65).
4. Color Calibration & Correction:
   - ColorCorrectionMatrix (CCM): 3x3 regularised linear transform fitting.
   - calibrate_color_chart: Evaluates pre/post calibration errors and estimated CCT.
5. Controlled Surface Colour Accuracy Assessment:
   - ColourAccuracyReport: Multi-surface evaluation against ground-truth spectrophotometer
     or calibrated color chart measurements under controlled illumination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
# Standard D65 Reference Constants (CIE 1931 2° Standard Observer)
# -----------------------------------------------------------------------------
# D65 White point normalised to Y = 1.00000
D65_WHITE_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

# sRGB to XYZ Matrix (IEC 61966-2-1)
_M_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# XYZ to sRGB Matrix
_M_XYZ_TO_SRGB = np.linalg.inv(_M_SRGB_TO_XYZ)


# -----------------------------------------------------------------------------
# Colour Space Conversions
# -----------------------------------------------------------------------------

def srgb_to_linear(rgb: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert sRGB in [0, 255] (or [0, 1]) to linear RGB [0, 1]."""
    arr = np.asarray(rgb, dtype=np.float64)
    if np.max(arr) > 1.0 + 1e-6:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: Union[Sequence[float], np.ndarray], out_255: bool = True) -> np.ndarray:
    """Convert linear RGB [0, 1] to sRGB in [0, 255] or [0, 1]."""
    c = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)
    return srgb * 255.0 if out_255 else srgb


def rgb_to_xyz(rgb: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert sRGB in [0, 255] to CIEXYZ under D65 white point."""
    lin = srgb_to_linear(rgb)
    return lin @ _M_SRGB_TO_XYZ.T


def xyz_to_rgb(xyz: Union[Sequence[float], np.ndarray], out_255: bool = True) -> np.ndarray:
    """Convert CIEXYZ under D65 to sRGB."""
    lin = np.asarray(xyz, dtype=np.float64) @ _M_XYZ_TO_SRGB.T
    return linear_to_srgb(lin, out_255=out_255)


def xyz_to_lab(xyz: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert CIEXYZ under D65 to CIELAB (L*, a*, b*)."""
    norm_xyz = np.asarray(xyz, dtype=np.float64) / D65_WHITE_XYZ
    eps = 216.0 / 24389.0  # (6/29)^3 approx 0.008856
    kappa = 24389.0 / 27.0  # (29/3)^3 approx 903.3
    f = np.where(norm_xyz > eps, np.cbrt(norm_xyz), (kappa * norm_xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert CIELAB (L*, a*, b*) under D65 to CIEXYZ."""
    arr = np.asarray(lab, dtype=np.float64)
    L = arr[..., 0]
    a = arr[..., 1]
    b = arr[..., 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    eps = 6.0 / 29.0  # approx 0.20689
    kappa = 24389.0 / 27.0

    xr = np.where(fx > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > (kappa * (eps ** 3)), ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)

    return np.stack([xr, yr, zr], axis=-1) * D65_WHITE_XYZ


def rgb_to_lab(rgb: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert sRGB [0, 255] directly to CIELAB (L*, a*, b*)."""
    xyz = rgb_to_xyz(rgb)
    return xyz_to_lab(xyz)


def lab_to_rgb(lab: Union[Sequence[float], np.ndarray], out_255: bool = True) -> np.ndarray:
    """Convert CIELAB (L*, a*, b*) directly to sRGB."""
    xyz = lab_to_xyz(lab)
    return xyz_to_rgb(xyz, out_255=out_255)


# -----------------------------------------------------------------------------
# Perceptual Difference Metrics: Delta E (CIE 1976, CIE 1994, CIEDE2000)
# -----------------------------------------------------------------------------

def delta_e76(
    color1: Union[Sequence[float], np.ndarray],
    color2: Union[Sequence[float], np.ndarray],
    is_lab: bool = False,
) -> float:
    """CIE 1976 colour difference (Euclidean distance in CIELAB).

    If is_lab is False (default), inputs are assumed to be sRGB in [0, 255].
    """
    lab1 = np.asarray(color1, dtype=np.float64) if is_lab else rgb_to_lab(color1)
    lab2 = np.asarray(color2, dtype=np.float64) if is_lab else rgb_to_lab(color2)
    return float(np.linalg.norm(lab1 - lab2))


def delta_e94(
    color1: Union[Sequence[float], np.ndarray],
    color2: Union[Sequence[float], np.ndarray],
    is_lab: bool = False,
    textile: bool = False,
) -> float:
    """CIE 1994 colour difference with application-specific weighting.

    Default is graphic arts weighting (kL=1, k1=0.045, k2=0.015).
    """
    lab1 = np.asarray(color1, dtype=np.float64) if is_lab else rgb_to_lab(color1)
    lab2 = np.asarray(color2, dtype=np.float64) if is_lab else rgb_to_lab(color2)

    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    dL = L1 - L2
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    dC = C1 - C2

    da = a1 - a2
    db = b1 - b2
    dH_sq = max(0.0, (da ** 2 + db ** 2) - (dC ** 2))
    dH = math.sqrt(dH_sq)

    if textile:
        kL, kC, kH = 2.0, 1.0, 1.0
        K1, K2 = 0.048, 0.014
    else:  # Graphic arts default
        kL, kC, kH = 1.0, 1.0, 1.0
        K1, K2 = 0.045, 0.015

    sL = 1.0
    sC = 1.0 + K1 * C1
    sH = 1.0 + K2 * C1

    return float(math.sqrt((dL / (kL * sL)) ** 2 + (dC / (kC * sC)) ** 2 + (dH / (kH * sH)) ** 2))


def delta_e2000(
    color1: Union[Sequence[float], np.ndarray],
    color2: Union[Sequence[float], np.ndarray],
    is_lab: bool = False,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> float:
    """CIEDE2000 colour difference (ISO/CIE 11664-6 standard).

    Complete implementation handling all hue angles, trigonometric wrapping,
    chroma non-linearities, and blue rotation factor R_T.
    """
    lab1 = np.asarray(color1, dtype=np.float64) if is_lab else rgb_to_lab(color1)
    lab2 = np.asarray(color2, dtype=np.float64) if is_lab else rgb_to_lab(color2)

    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    # 1. Calculate C_ab and G
    C1_ab = math.hypot(a1, b1)
    C2_ab = math.hypot(a2, b2)
    mean_C_ab = (C1_ab + C2_ab) / 2.0
    mean_C_ab_pow7 = mean_C_ab ** 7
    G = 0.5 * (1.0 - math.sqrt(mean_C_ab_pow7 / (mean_C_ab_pow7 + 25.0 ** 7)))

    # 2. Adjusted a', C', h'
    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = math.hypot(a1_prime, b1)
    C2_prime = math.hypot(a2_prime, b2)

    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360.0
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360.0

    # 3. Differences
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    if C1_prime * C2_prime < 1e-12:
        dh_prime = 0.0
    elif abs(h2_prime - h1_prime) <= 180.0:
        dh_prime = h2_prime - h1_prime
    elif (h2_prime - h1_prime) > 180.0:
        dh_prime = (h2_prime - h1_prime) - 360.0
    else:
        dh_prime = (h2_prime - h1_prime) + 360.0

    dH_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(dh_prime / 2.0))

    # 4. Means and weighting factors
    mean_L_prime = (L1 + L2) / 2.0
    mean_C_prime = (C1_prime + C2_prime) / 2.0

    if C1_prime * C2_prime < 1e-12:
        mean_h_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180.0:
        mean_h_prime = (h1_prime + h2_prime) / 2.0
    elif (h1_prime + h2_prime) < 360.0:
        mean_h_prime = (h1_prime + h2_prime + 360.0) / 2.0
    else:
        mean_h_prime = (h1_prime + h2_prime - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(mean_h_prime - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * mean_h_prime))
        + 0.32 * math.cos(math.radians(3.0 * mean_h_prime + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * mean_h_prime - 63.0))
    )

    d_theta = 30.0 * math.exp(-(((mean_h_prime - 275.0) / 25.0) ** 2))
    mean_C_prime_pow7 = mean_C_prime ** 7
    R_C = 2.0 * math.sqrt(mean_C_prime_pow7 / (mean_C_prime_pow7 + 25.0 ** 7))

    S_L = 1.0 + (0.015 * ((mean_L_prime - 50.0) ** 2)) / math.sqrt(20.0 + (mean_L_prime - 50.0) ** 2)
    S_C = 1.0 + 0.045 * mean_C_prime
    S_H = 1.0 + 0.015 * mean_C_prime * T

    R_T = -math.sin(math.radians(2.0 * d_theta)) * R_C

    # 5. Composite CIEDE2000 result
    term_L = dL_prime / (kL * S_L)
    term_C = dC_prime / (kC * S_C)
    term_H = dH_prime / (kH * S_H)

    de00_sq = term_L ** 2 + term_C ** 2 + term_H ** 2 + R_T * term_C * term_H
    return float(math.sqrt(max(0.0, de00_sq)))


# -----------------------------------------------------------------------------
# ColorChecker Classic (24 Standard Reference Patches under D65)
# -----------------------------------------------------------------------------

COLORCHECKER_D65_PATCHES: Dict[str, Dict[str, Any]] = {
    "patch_01_dark_skin": {"name": "Dark skin", "rgb": (115, 82, 68), "lab": (37.99, 13.56, 14.06)},
    "patch_02_light_skin": {"name": "Light skin", "rgb": (194, 150, 130), "lab": (65.71, 18.13, 17.81)},
    "patch_03_blue_sky": {"name": "Blue sky", "rgb": (98, 122, 157), "lab": (49.93, -4.88, -21.93)},
    "patch_04_foliage": {"name": "Foliage", "rgb": (87, 108, 67), "lab": (43.14, -13.10, 21.91)},
    "patch_05_blue_flower": {"name": "Blue flower", "rgb": (133, 128, 177), "lab": (55.60, 8.88, -25.39)},
    "patch_06_bluish_green": {"name": "Bluish green", "rgb": (103, 189, 170), "lab": (70.72, -33.37, -0.20)},
    "patch_07_orange": {"name": "Orange", "rgb": (214, 126, 44), "lab": (62.66, 36.07, 57.10)},
    "patch_08_purplish_blue": {"name": "Purplish blue", "rgb": (80, 91, 166), "lab": (40.02, 10.41, -45.96)},
    "patch_09_moderate_red": {"name": "Moderate red", "rgb": (193, 90, 99), "lab": (51.12, 48.24, 16.21)},
    "patch_10_purple": {"name": "Purple", "rgb": (94, 60, 108), "lab": (30.33, 22.98, -21.59)},
    "patch_11_yellow_green": {"name": "Yellow green", "rgb": (157, 188, 64), "lab": (72.53, -23.71, 57.26)},
    "patch_12_orange_yellow": {"name": "Orange yellow", "rgb": (224, 163, 46), "lab": (71.60, 18.79, 67.36)},
    "patch_13_blue": {"name": "Blue", "rgb": (56, 61, 150), "lab": (29.51, 18.48, -50.00)},
    "patch_14_green": {"name": "Green", "rgb": (70, 148, 73), "lab": (55.26, -38.34, 31.37)},
    "patch_15_red": {"name": "Red", "rgb": (175, 54, 60), "lab": (40.97, 51.78, 25.47)},
    "patch_16_yellow": {"name": "Yellow", "rgb": (231, 199, 31), "lab": (81.73, 4.04, 79.82)},
    "patch_17_magenta": {"name": "Magenta", "rgb": (187, 86, 149), "lab": (51.94, 49.99, -14.57)},
    "patch_18_cyan": {"name": "Cyan", "rgb": (8, 133, 161), "lab": (51.04, -28.63, -28.64)},
    "patch_19_white_95": {"name": "White 9.5", "rgb": (243, 243, 242), "lab": (96.54, -0.43, 1.19)},
    "patch_20_neutral_8": {"name": "Neutral 8", "rgb": (200, 200, 200), "lab": (81.22, -0.64, -0.34)},
    "patch_21_neutral_65": {"name": "Neutral 6.5", "rgb": (160, 160, 160), "lab": (66.77, -0.73, -0.50)},
    "patch_22_neutral_5": {"name": "Neutral 5", "rgb": (122, 122, 121), "lab": (50.87, -0.15, -0.27)},
    "patch_23_neutral_35": {"name": "Neutral 3.5", "rgb": (85, 85, 85), "lab": (35.66, -0.42, -1.23)},
    "patch_24_black_2": {"name": "Black 2", "rgb": (52, 52, 52), "lab": (20.46, -0.08, -0.97)},
}

COLORCHECKER_D65_SRGB: List[Tuple[int, int, int]] = [
    p["rgb"] for p in COLORCHECKER_D65_PATCHES.values()
]

COLORCHECKER_D65_LAB: List[Tuple[float, float, float]] = [
    p["lab"] for p in COLORCHECKER_D65_PATCHES.values()
]


# -----------------------------------------------------------------------------
# Color Correction Matrix (CCM) & Chart Calibration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ColorCorrectionMatrix:
    """A 3x3 regularised linear colour correction matrix."""

    matrix: np.ndarray  # Shape (3, 3)

    def apply(self, rgb: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Apply color correction to an sRGB triple or array of triples."""
        arr = np.asarray(rgb, dtype=np.float64)
        is_flat = arr.ndim == 1
        lin = srgb_to_linear(arr)
        cal_lin = lin @ self.matrix.T
        cal_srgb = linear_to_srgb(cal_lin, out_255=True)
        return cal_srgb if not is_flat else cal_srgb.reshape(-1)

    def to_list(self) -> List[List[float]]:
        return [[round(float(v), 5) for v in row] for row in self.matrix]


def estimate_cct_mccamy(rgb_white: Union[Sequence[float], np.ndarray]) -> float:
    """Estimate Correlated Colour Temperature (CCT) in Kelvin using McCamy's formula."""
    xyz = rgb_to_xyz(rgb_white)
    total = np.sum(xyz)
    if total < 1e-9:
        return 6500.0
    x = xyz[0] / total
    y = xyz[1] / total
    # Epicentre of isotemperature lines
    xe, ye = 0.3320, 0.1858
    n = (x - xe) / (ye - y)
    cct = 449.0 * (n ** 3) + 3525.0 * (n ** 2) + 6823.3 * n + 5520.33
    return float(np.clip(cct, 1500.0, 25000.0))


@dataclass
class ChartCalibrationResult:
    """Outcome of calibrating a captured colour chart against reference ground truth."""

    ccm: ColorCorrectionMatrix
    mean_de00_before: float
    mean_de00_after: float
    max_de00_before: float
    max_de00_after: float
    median_de00_after: float
    p95_de00_after: float
    mean_de76_after: float
    estimated_cct_k: float
    patch_errors_de00: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ccm": self.ccm.to_list(),
            "mean_de00_before": round(self.mean_de00_before, 2),
            "mean_de00_after": round(self.mean_de00_after, 2),
            "max_de00_before": round(self.max_de00_before, 2),
            "max_de00_after": round(self.max_de00_after, 2),
            "median_de00_after": round(self.median_de00_after, 2),
            "p95_de00_after": round(self.p95_de00_after, 2),
            "mean_de76_after": round(self.mean_de76_after, 2),
            "estimated_cct_k": round(self.estimated_cct_k, 0),
        }


def calibrate_color_chart(
    observed_patches: Sequence[Tuple[int, int, int]],
    reference_patches: Sequence[Tuple[int, int, int]] = COLORCHECKER_D65_SRGB,
    regularisation: float = 1e-3,
) -> ChartCalibrationResult:
    """Fit a 3x3 colour correction matrix (CCM) mapping observed RGB to reference ground truth.

    Inputs are 24-patch sRGB triples [0, 255]. Uses Ridge regression in linear RGB.
    """
    obs_arr = np.asarray(observed_patches, dtype=np.float64)
    ref_arr = np.asarray(reference_patches, dtype=np.float64)

    if len(obs_arr) != len(ref_arr):
        raise ValueError(f"Patch count mismatch: {len(obs_arr)} observed vs {len(ref_arr)} reference")

    # Initial errors before calibration
    errors_before = [delta_e2000(o, r) for o, r in zip(obs_arr, ref_arr)]

    # Convert to linear RGB
    obs_lin = srgb_to_linear(obs_arr)  # (N, 3)
    ref_lin = srgb_to_linear(ref_arr)  # (N, 3)

    # Solve M @ obs_lin.T = ref_lin.T via Ridge regression:
    # M = (ref_lin.T @ obs_lin) @ inv(obs_lin.T @ obs_lin + lambda * I)
    XtX = obs_lin.T @ obs_lin
    XtY = obs_lin.T @ ref_lin
    reg_I = regularisation * np.eye(3)
    M_T = np.linalg.solve(XtX + reg_I, XtY)
    M = M_T.T

    ccm = ColorCorrectionMatrix(matrix=M)

    # Evaluate calibrated patches
    cal_srgb = ccm.apply(obs_arr)
    errors_after_00 = [delta_e2000(c, r) for c, r in zip(cal_srgb, ref_arr)]
    errors_after_76 = [delta_e76(c, r) for c, r in zip(cal_srgb, ref_arr)]

    # Estimated CCT from white patch (index 18: patch 19 white)
    white_obs = obs_arr[18] if len(obs_arr) > 18 else (255, 255, 255)
    cct = estimate_cct_mccamy(white_obs)

    return ChartCalibrationResult(
        ccm=ccm,
        mean_de00_before=float(np.mean(errors_before)),
        mean_de00_after=float(np.mean(errors_after_00)),
        max_de00_before=float(np.max(errors_before)),
        max_de00_after=float(np.max(errors_after_00)),
        median_de00_after=float(np.median(errors_after_00)),
        p95_de00_after=float(np.percentile(errors_after_00, 95)),
        mean_de76_after=float(np.mean(errors_after_76)),
        estimated_cct_k=cct,
        patch_errors_de00=errors_after_00,
    )


# -----------------------------------------------------------------------------
# Controlled Surface Colour Accuracy Assessment
# -----------------------------------------------------------------------------

class ColorAccuracyTier(str, Enum):
    """Grading of reconstructed colour under controlled capture standards."""

    EXCELLENT = "EXCELLENT"        # delta_e00 <= 2.0 (near/below JND)
    ACCEPTABLE = "ACCEPTABLE"      # delta_e00 <= 4.0 (commercial tolerance)
    MARGINAL = "MARGINAL"          # delta_e00 <= 8.0 (noticeable shift)
    UNACCEPTABLE = "UNACCEPTABLE"  # delta_e00 > 8.0


@dataclass
class SurfaceColorComparison:
    """Individual surface color comparison against ground truth."""

    surface_id: str
    observed_rgb: Tuple[int, int, int]
    ground_truth_rgb: Tuple[int, int, int]
    delta_e00: float
    delta_e76: float
    tier: ColorAccuracyTier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "observed_rgb": list(self.observed_rgb),
            "ground_truth_rgb": list(self.ground_truth_rgb),
            "delta_e00": round(self.delta_e00, 2),
            "delta_e76": round(self.delta_e76, 2),
            "tier": self.tier.value,
        }


@dataclass
class ColourAccuracyReport:
    """Comprehensive colour accuracy scorecard for controlled captures."""

    sample_count: int
    mean_delta_e00: float
    median_delta_e00: float
    max_delta_e00: float
    mean_delta_e76: float
    overall_tier: ColorAccuracyTier
    illuminant: str  # e.g. "D65", "Controlled"
    surfaces: List[SurfaceColorComparison] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean_delta_e00": round(self.mean_delta_e00, 2),
            "median_delta_e00": round(self.median_delta_e00, 2),
            "max_delta_e00": round(self.max_delta_e00, 2),
            "mean_delta_e76": round(self.mean_delta_e76, 2),
            "overall_tier": self.overall_tier.value,
            "illuminant": self.illuminant,
            "surfaces": [s.to_dict() for s in self.surfaces],
        }


def assess_surface_color_accuracy(
    observed_surfaces: Dict[str, Tuple[int, int, int]],
    ground_truth_surfaces: Dict[str, Tuple[int, int, int]],
    illuminant: str = "D65",
) -> ColourAccuracyReport:
    """Assess colour accuracy across reconstructed architectural surfaces or objects

    against ground truth reference spectrophotometer or chart measurements.
    """
    comparisons: List[SurfaceColorComparison] = []

    common_keys = [k for k in observed_surfaces if k in ground_truth_surfaces]
    if not common_keys:
        return ColourAccuracyReport(
            sample_count=0,
            mean_delta_e00=0.0,
            median_delta_e00=0.0,
            max_delta_e00=0.0,
            mean_delta_e76=0.0,
            overall_tier=ColorAccuracyTier.UNACCEPTABLE,
            illuminant=illuminant,
            surfaces=[],
        )

    for sid in common_keys:
        obs_rgb = observed_surfaces[sid]
        gt_rgb = ground_truth_surfaces[sid]

        de00 = delta_e2000(obs_rgb, gt_rgb)
        de76 = delta_e76(obs_rgb, gt_rgb)

        if de00 <= 2.0:
            tier = ColorAccuracyTier.EXCELLENT
        elif de00 <= 4.0:
            tier = ColorAccuracyTier.ACCEPTABLE
        elif de00 <= 8.0:
            tier = ColorAccuracyTier.MARGINAL
        else:
            tier = ColorAccuracyTier.UNACCEPTABLE

        comparisons.append(
            SurfaceColorComparison(
                surface_id=sid,
                observed_rgb=obs_rgb,
                ground_truth_rgb=gt_rgb,
                delta_e00=de00,
                delta_e76=de76,
                tier=tier,
            )
        )

    de00_values = [c.delta_e00 for c in comparisons]
    de76_values = [c.delta_e76 for c in comparisons]

    mean_de00 = float(np.mean(de00_values))
    median_de00 = float(np.median(de00_values))
    max_de00 = float(np.max(de00_values))
    mean_de76 = float(np.mean(de76_values))

    if mean_de00 <= 2.0:
        overall_tier = ColorAccuracyTier.EXCELLENT
    elif mean_de00 <= 4.0:
        overall_tier = ColorAccuracyTier.ACCEPTABLE
    elif mean_de00 <= 8.0:
        overall_tier = ColorAccuracyTier.MARGINAL
    else:
        overall_tier = ColorAccuracyTier.UNACCEPTABLE

    return ColourAccuracyReport(
        sample_count=len(comparisons),
        mean_delta_e00=mean_de00,
        median_delta_e00=median_de00,
        max_delta_e00=max_de00,
        mean_delta_e76=mean_de76,
        overall_tier=overall_tier,
        illuminant=illuminant,
        surfaces=comparisons,
    )
