"""Materials, surface appearance, and environmental lighting perception."""

from .classifier import (
    MaterialCategory,
    MaterialClassification,
    MaterialEvidence,
    PBRParameters,
    assign_pbr_material,
    classify_material,
)
from .lighting_estimator import (
    LightingDiagnostics,
    LightingEstimate,
    compute_sh_basis_9,
    estimate_and_attach_world_lighting,
    evaluate_sh_irradiance,
    fit_spherical_harmonics_lighting,
)
from .surface_appearance import (
    AppearanceReport,
    PlanarSurface,
    SurfaceAppearance,
    SurfaceView,
    attach_appearance,
    measure_surface_appearance,
    views_from_poses,
)

__all__ = [
    "AppearanceReport",
    "LightingDiagnostics",
    "LightingEstimate",
    "MaterialCategory",
    "MaterialClassification",
    "MaterialEvidence",
    "PBRParameters",
    "PlanarSurface",
    "SurfaceAppearance",
    "SurfaceView",
    "assign_pbr_material",
    "attach_appearance",
    "classify_material",
    "compute_sh_basis_9",
    "estimate_and_attach_world_lighting",
    "evaluate_sh_irradiance",
    "fit_spherical_harmonics_lighting",
    "measure_surface_appearance",
    "views_from_poses",
]
