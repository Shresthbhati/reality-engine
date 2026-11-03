"""Lighting & Environmental Illuminance Engine (Block 36).

Implements 9-coefficient Spherical Harmonics (L <= 2) ambient irradiance estimation,
dominant directional light vector recovery, and de-lighting / albedo recovery from
shaded surface observations and normals.

Based on Ramamoorthi & Hanrahan (2001) "An Efficient Representation for Irradiance Environment Maps":
- 9 real Spherical Harmonics basis functions (order l=0, 1, 2)
- Lambertian cosine convolution kernel factors (A_0 = pi, A_1 = 2pi/3, A_2 = pi/4)
- Tikhonov-regularized least-squares solver across RGB channels
- Decomposition into ambient omnidirectional irradiance, primary directional light, and color temperature
- Texture de-lighting: computes intrinsic diffuse albedo a = c / (E(n) + epsilon)
- Canonical grounding into WorldIR.environment["lighting"]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from world_ir.world_v1 import WorldIR


# Spherical Harmonics constants
C0 = 0.28209479177387814  # 0.5 * sqrt(1/pi)
C1 = 0.4886025119029199   # 0.5 * sqrt(3/pi)
C2_0 = 1.0925484305920792  # 0.5 * sqrt(15/pi)
C2_1 = 0.31539156525252005 # 0.25 * sqrt(5/pi)
C2_2 = 0.5462742152960396  # 0.25 * sqrt(15/pi)

# Lambertian diffuse reflection convolution factors A_l
# For irradiance E(n) = sum A_l * c_lm * Y_lm(n)
A0 = math.pi
A1 = (2.0 * math.pi) / 3.0
A2 = math.pi / 4.0


def compute_sh_basis_9(normals: np.ndarray) -> np.ndarray:
    """Computes the 9 real Spherical Harmonics basis functions for order L <= 2.

    Args:
        normals: (N, 3) unit normal vectors [x, y, z].

    Returns:
        (N, 9) basis matrix evaluated at each normal vector.
    """
    n = np.asarray(normals, dtype=np.float64)
    if n.ndim == 1:
        n = n.reshape(1, 3)

    # Normalize vectors
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    u = n / norms

    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]

    num_pts = len(u)
    basis = np.zeros((num_pts, 9), dtype=np.float64)

    # l=0, m=0
    basis[:, 0] = C0

    # l=1
    basis[:, 1] = C1 * y  # m = -1
    basis[:, 2] = C1 * z  # m = 0
    basis[:, 3] = C1 * x  # m = +1

    # l=2
    basis[:, 4] = C2_0 * (x * y)          # m = -2
    basis[:, 5] = C2_0 * (y * z)          # m = -1
    basis[:, 6] = C2_1 * (3.0 * z**2 - 1.0)  # m = 0
    basis[:, 7] = C2_0 * (x * z)          # m = +1
    basis[:, 8] = C2_2 * (x**2 - y**2)    # m = +2

    return basis


def evaluate_sh_irradiance(
    sh_coeffs_9x3: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Evaluates diffuse irradiance E(n) for a set of unit normals given 9x3 RGB SH coefficients.

    Multiplies basis functions with Lambertian convolution factors [A0, A1, A1, A1, A2, A2, A2, A2, A2].

    Args:
        sh_coeffs_9x3: (9, 3) RGB spherical harmonics coefficients.
        normals: (N, 3) unit normal vectors.

    Returns:
        (N, 3) RGB irradiance values (in linear photometric units).
    """
    basis = compute_sh_basis_9(normals)  # (N, 9)

    # Scale basis by Lambertian convolution kernel factors
    kernel = np.array([A0, A1, A1, A1, A2, A2, A2, A2, A2], dtype=np.float64)
    scaled_basis = basis * kernel  # (N, 9)

    irradiance = scaled_basis @ sh_coeffs_9x3  # (N, 3)
    return np.maximum(irradiance, 0.0)


@dataclass
class LightingDiagnostics:
    """Quantitative evaluation of lighting model fit."""

    rmse_rgb: Tuple[float, float, float]
    total_rmse: float
    sample_count: int
    condition_number: float
    dominant_direction_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rmse_rgb": [round(float(c), 4) for c in self.rmse_rgb],
            "total_rmse": round(float(self.total_rmse), 4),
            "sample_count": self.sample_count,
            "condition_number": round(float(self.condition_number), 2),
            "dominant_direction_confidence": round(float(self.dominant_direction_confidence), 3),
        }


@dataclass
class LightingEstimate:
    """Complete ambient and directional lighting representation for a scene."""

    sh_coefficients: np.ndarray  # (9, 3)
    ambient_irradiance_rgb: Tuple[float, float, float]
    dominant_light_direction: Tuple[float, float, float]  # Unit vector pointing towards light
    directional_intensity_rgb: Tuple[float, float, float]
    diagnostics: LightingDiagnostics

    def evaluate(self, normals: np.ndarray) -> np.ndarray:
        """Evaluates RGB irradiance at given surface normals."""
        return evaluate_sh_irradiance(self.sh_coefficients, normals)

    def delight(
        self,
        normals: np.ndarray,
        observed_linear_rgb: np.ndarray,
        epsilon: float = 0.05,
    ) -> np.ndarray:
        """Recovers intrinsic diffuse albedo by dividing observed color by computed irradiance.

        Args:
            normals: (N, 3) surface normal vectors.
            observed_linear_rgb: (N, 3) linear RGB surface colors.
            epsilon: Non-zero stabilizer to prevent division by zero in deep shadow.

        Returns:
            (N, 3) diffuse albedo in range [0.0, 1.0].
        """
        irradiance = self.evaluate(normals)  # (N, 3)
        ambient_mean = np.mean(self.ambient_irradiance_rgb)
        denom = irradiance + (epsilon * max(1e-3, ambient_mean))
        albedo = observed_linear_rgb / denom

        # Normalize relative to reference scale if needed
        return np.clip(albedo, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sh_coefficients": self.sh_coefficients.tolist(),
            "ambient_irradiance_rgb": [round(float(c), 4) for c in self.ambient_irradiance_rgb],
            "dominant_light_direction": [round(float(d), 4) for d in self.dominant_light_direction],
            "directional_intensity_rgb": [round(float(i), 4) for i in self.directional_intensity_rgb],
            "diagnostics": self.diagnostics.to_dict(),
        }


def fit_spherical_harmonics_lighting(
    normals: np.ndarray,
    observed_linear_rgb: np.ndarray,
    regularization_lambda: float = 1e-3,
) -> LightingEstimate:
    """Solves for 9-coefficient RGB spherical harmonics lighting from surface normals and observed colors.

    Args:
        normals: (N, 3) surface normal vectors.
        observed_linear_rgb: (N, 3) observed linear RGB color values.
        regularization_lambda: Ridge regression parameter to ensure numerical stability.

    Returns:
        LightingEstimate containing fitted SH coefficients, dominant light direction, and diagnostics.
    """
    n = np.asarray(normals, dtype=np.float64)
    c = np.asarray(observed_linear_rgb, dtype=np.float64)

    if len(n) < 9 or len(c) < 9:
        raise ValueError(f"Spherical harmonics fitting requires at least 9 samples; got {len(n)}")

    basis = compute_sh_basis_9(n)  # (N, 9)
    kernel = np.array([A0, A1, A1, A1, A2, A2, A2, A2, A2], dtype=np.float64)
    scaled_basis = basis * kernel  # (N, 9)

    # Ridge regression: (A^T A + lambda * I)^-1 A^T C
    ata = scaled_basis.T @ scaled_basis
    reg = regularization_lambda * np.eye(9)
    reg[0, 0] = 0.0  # Do not regularize ambient DC component

    cond = float(np.linalg.cond(ata + reg))
    sh_coeffs = np.linalg.solve(ata + reg, scaled_basis.T @ c)  # (9, 3)

    # Evaluate fitted irradiance and compute residuals
    fitted_irradiance = evaluate_sh_irradiance(sh_coeffs, n)
    residuals = c - fitted_irradiance
    rmse_per_channel = np.sqrt(np.mean(residuals**2, axis=0))
    total_rmse = float(np.sqrt(np.mean(residuals**2)))

    # Omnidirectional ambient irradiance (order 0)
    # E_ambient = A0 * c_00 * Y_00 = pi * c_00 * C0
    ambient_rgb = tuple(float(max(0.0, A0 * C0 * sh_coeffs[0, k])) for k in range(3))

    # Dominant directional vector from order 1 gradient
    # Basis: Y_1,-1 (y), Y_1,0 (z), Y_1,1 (x)
    # The gradient direction in world space is (c_1,1, c_1,-1, c_1,0)
    dir_x = float(np.mean(sh_coeffs[3, :]))
    dir_y = float(np.mean(sh_coeffs[1, :]))
    dir_z = float(np.mean(sh_coeffs[2, :]))
    dir_vec = np.array([dir_x, dir_y, dir_z], dtype=np.float64)
    dir_norm = float(np.linalg.norm(dir_vec))

    if dir_norm > 1e-6:
        dominant_direction = tuple(float(d) for d in (dir_vec / dir_norm))
        # Confidence scales with ratio of directional gradient to ambient
        amb_mean = max(1e-3, float(np.mean(ambient_rgb)))
        confidence = min(1.0, dir_norm / amb_mean)
    else:
        dominant_direction = (0.0, 0.0, 1.0)
        confidence = 0.0

    # Directional intensity per channel
    dir_intensity = tuple(
        float(np.sqrt(sh_coeffs[3, k] ** 2 + sh_coeffs[1, k] ** 2 + sh_coeffs[2, k] ** 2))
        for k in range(3)
    )

    diag = LightingDiagnostics(
        rmse_rgb=(float(rmse_per_channel[0]), float(rmse_per_channel[1]), float(rmse_per_channel[2])),
        total_rmse=total_rmse,
        sample_count=len(n),
        condition_number=cond,
        dominant_direction_confidence=confidence,
    )

    return LightingEstimate(
        sh_coefficients=sh_coeffs,
        ambient_irradiance_rgb=(ambient_rgb[0], ambient_rgb[1], ambient_rgb[2]),
        dominant_light_direction=(dominant_direction[0], dominant_direction[1], dominant_direction[2]),
        directional_intensity_rgb=dir_intensity,
        diagnostics=diag,
    )


def estimate_and_attach_world_lighting(
    world: WorldIR,
    normals: np.ndarray,
    observed_linear_rgb: np.ndarray,
) -> LightingEstimate:
    """Estimates lighting from observations and attaches the estimate to WorldIR.environment["lighting"]."""
    estimate = fit_spherical_harmonics_lighting(normals, observed_linear_rgb)

    world.environment["lighting"] = estimate.to_dict()
    return estimate
