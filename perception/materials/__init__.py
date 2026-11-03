"""Materials, surface appearance, and environmental lighting perception."""

from .classifier import MaterialClassifier, MaterialClassificationResult, MaterialMatch
from .lighting_estimator import (
    LightingDiagnostics,
    LightingEstimate,
    compute_sh_basis_9,
    estimate_and_attach_world_lighting,
    evaluate_sh_irradiance,
    fit_spherical_harmonics_lighting,
)
from .surface_appearance import (
    SurfaceAppearance,
    SurfacePolygon,
    SurfaceView,
    compute_surface_appearance,
)

__all__ = [
    "LightingDiagnostics",
    "LightingEstimate",
    "MaterialClassificationResult",
    "MaterialClassifier",
    "MaterialMatch",
    "SurfaceAppearance",
    "SurfacePolygon",
    "SurfaceView",
    "compute_sh_basis_9",
    "compute_surface_appearance",
    "estimate_and_attach_world_lighting",
    "evaluate_sh_irradiance",
    "fit_spherical_harmonics_lighting",
]
