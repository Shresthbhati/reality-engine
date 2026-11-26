"""Reality Engine Perception Intelligence Subsystem (A3)."""

from __future__ import annotations

from perception.intelligence.architectural import (
    ArchitecturalAnalysisResult,
    analyze_architecture,
)
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    AppearanceDescriptor,
    ArchitecturalObservation,
    CoordinateSystemSpec,
    CrossViewIdentityHypothesis,
    DetectionProvenance,
    FurnitureObservation,
    MaterialHypothesisRecord,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ObservationLifecycle,
    ObservationRevision,
    PerceptionBundle,
    ScoreSemantics,
    SegmentationObservation,
    SpatialRelationshipObservation,
    TemporalObservation,
    ValidationError,
    stable_observation_id,
)
from perception.intelligence.diagnostics import (
    DiagnosticFinding,
    DiagnosticReport,
    run_perception_diagnostics,
)
from perception.intelligence.furniture import (
    FurnitureAnalysisResult,
    analyze_furniture,
    classify_furniture_hierarchy,
)
from perception.intelligence.identity import (
    CameraPoseConstraint,
    IdentityMergeProposal,
    IdentityResolutionReport,
    IdentitySplitProposal,
    resolve_cross_view_identities,
)
from perception.intelligence.image_adaptive import (
    ImageQualityProfile,
    RegionProposal,
    analyze_image_quality,
    plan_adaptive_crops,
    reconcile_crop_duplicates,
    transform_crop_box_to_image,
)
from perception.intelligence.materials_intel import (
    SUPPORTED_MATERIALS,
    CrossViewMaterialConsensus,
    aggregate_cross_view_materials,
    extract_appearance_and_materials,
)
from perception.intelligence.orchestration import (
    EngineHealthState,
    OrchestratedDetectionResult,
    PerceptionEngineScheduler,
    RoutingPlan,
)
from perception.intelligence.refinement import (
    RefinementAction,
    RefinementPlan,
    plan_active_refinement,
)
from perception.intelligence.relationships import (
    RELATIONSHIP_ONTOLOGY,
    SceneRelationshipReport,
    infer_scene_relationships,
)
from perception.intelligence.segmentation_intel import (
    EvaluatedMask,
    classify_mask_kind,
    evaluate_instance_mask,
    separate_overlapping_masks,
)
from perception.intelligence.world_integration import (
    build_intelligence_bundle,
    observation_from_bbox_draft,
    to_evidence_matrix_drafts,
)

__all__ = [
    "AppearanceDescriptor",
    "ArchitecturalAnalysisResult",
    "ArchitecturalObservation",
    "CameraPoseConstraint",
    "CoordinateSystemSpec",
    "CrossViewIdentityHypothesis",
    "CrossViewMaterialConsensus",
    "DetectionProvenance",
    "DiagnosticFinding",
    "DiagnosticReport",
    "EngineHealthState",
    "EvaluatedMask",
    "FurnitureAnalysisResult",
    "FurnitureObservation",
    "IdentityMergeProposal",
    "IdentityResolutionReport",
    "IdentitySplitProposal",
    "ImageQualityProfile",
    "MaterialHypothesisRecord",
    "ModelExecutionProvenance",
    "NormalizedPerceptionObservation",
    "ObservationLifecycle",
    "ObservationRevision",
    "OrchestratedDetectionResult",
    "PerceptionBundle",
    "PerceptionEngineScheduler",
    "RELATIONSHIP_ONTOLOGY",
    "RefinementAction",
    "RefinementPlan",
    "RegionProposal",
    "RoutingPlan",
    "SCHEMA_VERSION",
    "SUPPORTED_MATERIALS",
    "SceneRelationshipReport",
    "ScoreSemantics",
    "SegmentationObservation",
    "SpatialRelationshipObservation",
    "TemporalObservation",
    "ValidationError",
    "aggregate_cross_view_materials",
    "analyze_architecture",
    "analyze_furniture",
    "analyze_image_quality",
    "build_intelligence_bundle",
    "classify_furniture_hierarchy",
    "classify_mask_kind",
    "evaluate_instance_mask",
    "extract_appearance_and_materials",
    "infer_scene_relationships",
    "observation_from_bbox_draft",
    "plan_active_refinement",
    "plan_adaptive_crops",
    "reconcile_crop_duplicates",
    "resolve_cross_view_identities",
    "run_perception_diagnostics",
    "separate_overlapping_masks",
    "stable_observation_id",
    "to_evidence_matrix_drafts",
    "transform_crop_box_to_image",
]
