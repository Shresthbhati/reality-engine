"""Reconstruction Intelligence and Multi-View Spatial Understanding Package."""

from engine.multiengine.spatial_intelligence.association import (
    AssociationOutcome,
    CrossViewRayCandidate,
    MultiViewIntelligenceEngine,
    PairScore,
    PhysicalMergedEntity,
    build_camera_ray,
    compute_ray_cpa,
    evaluate_cross_view_pair,
)
from engine.multiengine.spatial_intelligence.attributes import (
    EntityVisualAttributes,
    MaterialType,
    ObjectState,
    SurfaceFinish,
    extract_visual_attributes,
)
from engine.multiengine.spatial_intelligence.integration import (
    DetectionInput,
    SpatialSceneResult,
    reconstruct_spatial_scene,
)
from engine.multiengine.spatial_intelligence.spatial_lifting import (
    SpatialEntity3D,
    SpatialRelationKind,
    SpatialRelationship,
    infer_spatial_relationships,
    lift_region_to_spatial_entity,
    propagate_unprojection_covariance,
)
from engine.multiengine.spatial_intelligence.taxonomy import (
    DimensionMeasurability,
    Domain,
    GeometryType,
    TaxonomyClass,
    are_semantically_compatible,
    get_all_taxonomy_classes,
    get_detection_queries_by_domain,
    resolve_taxonomy_class,
)

__all__ = [
    "DimensionMeasurability",
    "Domain",
    "GeometryType",
    "TaxonomyClass",
    "are_semantically_compatible",
    "get_all_taxonomy_classes",
    "get_detection_queries_by_domain",
    "resolve_taxonomy_class",
    "EntityVisualAttributes",
    "MaterialType",
    "ObjectState",
    "SurfaceFinish",
    "extract_visual_attributes",
    "SpatialEntity3D",
    "SpatialRelationKind",
    "SpatialRelationship",
    "infer_spatial_relationships",
    "lift_region_to_spatial_entity",
    "propagate_unprojection_covariance",
    "AssociationOutcome",
    "CrossViewRayCandidate",
    "MultiViewIntelligenceEngine",
    "PairScore",
    "PhysicalMergedEntity",
    "build_camera_ray",
    "compute_ray_cpa",
    "evaluate_cross_view_pair",
    "DetectionInput",
    "SpatialSceneResult",
    "reconstruct_spatial_scene",
]
