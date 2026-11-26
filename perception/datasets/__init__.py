"""Perception datasets & human-reviewed ground-truth package (A4.1 - A4.2)."""

from perception.datasets.ground_truth import (
    AnnotationProvenance,
    GroundTruthAnnotation,
    ReviewStatus,
    build_human_review_manifest,
    compute_inter_annotator_agreement,
    export_ground_truth_json,
    import_ground_truth_json,
    score_ground_truth_dataset_quality,
)
from perception.datasets.indoor_registry import (
    INDOOR_DATASET_CATALOG,
    DatasetIntegrityReport,
    DatasetSpec,
    NormalizedDatasetSample,
    create_scene_disjoint_splits,
    normalize_dataset_sample,
    validate_dataset_integrity,
)

__all__ = [
    "AnnotationProvenance",
    "DatasetIntegrityReport",
    "DatasetSpec",
    "GroundTruthAnnotation",
    "INDOOR_DATASET_CATALOG",
    "NormalizedDatasetSample",
    "ReviewStatus",
    "build_human_review_manifest",
    "compute_inter_annotator_agreement",
    "create_scene_disjoint_splits",
    "export_ground_truth_json",
    "import_ground_truth_json",
    "normalize_dataset_sample",
    "score_ground_truth_dataset_quality",
    "validate_dataset_integrity",
]
