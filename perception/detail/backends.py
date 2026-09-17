"""Local-refinement fit backends (universal-perception directive
sections 11/13): the geometry attempts a refinement executor may
spend on an ROI's evidence.

Design rules:
  - NO duplicated geometry math: sphere/cylinder are thin adapters
    over perception/architecture/parametric.py's measured-residual
    fits (Kasa/Newton sphere, seed+coordinate-descent cylinder),
    reused UNCHANGED. Only the plane backend is new here, and it
    reuses parametric.py's Jacobi eigen solver and documented
    confidence map.
  - Every fit reports MEASURED residuals (rms/max over the real
    points) and a documented-map confidence. Degenerate support
    raises FitRefused -- never best-effort garbage.
  - Deterministic: no RNG anywhere; identical input -> identical fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from perception.architecture.parametric import (
    CylinderFit,
    FitRefused,
    SphereFit,
    _confidence_from_residual,
    _dot,
    _eigen_symmetric_3x3,
    _measure_residuals,
    _norm,
    fit_cylinder,
    fit_sphere,
)

__all__ = [
    "FitRefused",
    "PlaneFit",
    "fit_plane",
    "fit_sphere_on",
    "fit_cylinder_on",
    "MIN_PLANE_POINTS",
]

#: Documented tolerance scale for the plane backend's confidence map,
#: matching parametric.py's measurement-tolerance convention: at this
#: rms the fit quality maps to 0.5. Not tuned against real datasets
#: (the same deferral as every threshold in this repo).
_PLANE_CONFIDENCE_TOLERANCE_M = 0.1

#: A plane is only determined when the point support spans 2D: the
#: second-largest covariance eigenvalue must be non-degenerate
#: relative to the largest. Collinear points and tiny clusters refuse.
_PLANE_SPREAD_RATIO = 1e-3

#: Minimum support for a plane fit. parametric.py's smallest minimum
#: is 6 (sphere); a plane is cheaper to determine, so 5 is documented.
MIN_PLANE_POINTS = 5


@dataclass(frozen=True)
class PlaneFit:
    """A fitted plane through real points: unit normal, a point on
    the plane, and MEASURED residuals of the support."""

    normal: Tuple[float, float, float]
    point: Tuple[float, float, float]
    rms_residual_m: float
    max_residual_m: float
    n_points: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "normal": list(self.normal),
            "point": list(self.point),
            "rms_residual_m": self.rms_residual_m,
            "max_residual_m": self.max_residual_m,
            "n_points": self.n_points,
            "confidence": self.confidence,
        }


def fit_plane(points: Sequence[Tuple[float, float, float]]) -> PlaneFit:
    """Least-squares plane through real points.

    The normal is the covariance's smallest-eigenvalue eigenvector
    (parametric.py's Jacobi solver -- robust for repeated/near-zero
    eigenvalues). Refuses (FitRefused) when support is too small or
    degenerate-collinear: a plane is not determined by a line.
    """
    pts = [(float(x), float(y), float(z)) for x, y, z in points]
    n = len(pts)
    if n < MIN_PLANE_POINTS:
        raise FitRefused(
            f"plane fit needs at least {MIN_PLANE_POINTS} points, got {n}"
        )

    # Covariance about the centroid (parametric._covariance convention:
    # divide by n, deterministic).
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    cz = sum(p[2] for p in pts) / n
    cov = [[0.0] * 3 for _ in range(3)]
    for p in pts:
        d = (p[0] - cx, p[1] - cy, p[2] - cz)
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j] / n

    eigs = _eigen_symmetric_3x3(cov)
    lam = [eigs[i][0] for i in range(3)]
    if lam[2] <= 0.0 or lam[1] <= _PLANE_SPREAD_RATIO * lam[2]:
        raise FitRefused(
            "plane support is degenerate-collinear (second eigenvalue "
            f"{lam[1]:.3e} vs largest {lam[2]:.3e})"
        )

    # Smallest-eigenvalue eigenvector is the unit normal.
    normal = _norm(eigs[0][1])
    point = (cx, cy, cz)
    signed = [
        _dot((p[0] - cx, p[1] - cy, p[2] - cz), normal)
        for p in pts
    ]
    rms, mx = _measure_residuals(pts, signed)
    return PlaneFit(
        normal=normal,
        point=point,
        rms_residual_m=rms,
        max_residual_m=mx,
        n_points=n,
        confidence=_confidence_from_residual(
            rms, _PLANE_CONFIDENCE_TOLERANCE_M
        ),
    )


def fit_sphere_on(
    points: Sequence[Tuple[float, float, float]],
) -> SphereFit:
    """Sphere backend: parametric.fit_sphere unchanged."""
    return fit_sphere(points)


def fit_cylinder_on(
    points: Sequence[Tuple[float, float, float]],
    up: Tuple[float, float, float],
) -> CylinderFit:
    """Cylinder backend: parametric.fit_cylinder unchanged."""
    return fit_cylinder(points, up)
