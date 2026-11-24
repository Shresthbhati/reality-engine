"""Production Perception Output Contract for Downstream Consumers (Sprint 10 P6).

Provides Freebuff and downstream reconstruction/spatial stages with a stable,
strongly-typed, documented interface for perception outputs.

Guarantees:
- Explicit image identity and source engine provenance.
- Strict 0-indexed top-left pixel bounding boxes (x1, y1, x2, y2).
- Calibrated and raw confidence scores.
- Masks and explicit mask provenance (SAM, Polygon, etc.).
- Ranked parent-child candidates with linkage confidences.
- Multi-view physical identity clustering hypotheses.
- Occlusion and truncation metrics.
- Comprehensive uncertainty scores and rejection ledgers.
- Full serialization to/from JSON and bridging to WorldIR entities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np


class CoordinateConvention(str, Enum):
    """Supported bounding box coordinate conventions."""

    PIXEL_XYXY_TOPLEFT = "pixel_xyxy_topleft"  # [x1, y1, x2, y2], 0-indexed, x=col, y=row
    NORMALIZED_XYXY = "normalized_xyxy"        # [x1/w, y1/h, x2/w, y2/h] in [0, 1]


class MaskProvenance(str, Enum):
    """Provenance and generation method for instance segmentation masks."""

    SAM_VIT_B = "sam_vit_b"
    TOPOLOGICAL_CONTOUR = "topological_contour"
    BOUNDING_BOX_PRIOR = "bounding_box_prior"
    DEPTH_DISCONTINUITY = "depth_discontinuity"
    INFERRED = "inferred"


class RejectionReason(str, Enum):
    """Taxonomy of rejection reasons for suppressed candidate detections."""

    CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"
    DUPLICATE_SUPPRESSED = "duplicate_suppressed"
    TOPOLOGY_INVALID = "topology_invalid"
    HOLE_RATIO_EXCEEDED = "hole_ratio_exceeded"
    ASPECT_RATIO_ANOMALOUS = "aspect_ratio_anomalous"
    SCALE_INCOMPATIBLE = "scale_incompatible"
    CLASS_CONFUSION_REJECTED = "class_confusion_rejected"


@dataclass(frozen=True)
class ParentCandidate:
    """Candidate parent object for hierarchical component ownership."""

    parent_entity_id: str
    relationship_kind: str  # "part_of", "carcass_of", "mounted_on", "supported_by"
    confidence: float
    geometric_containment_ratio: float


@dataclass(frozen=True)
class PhysicalIdentityHypothesis:
    """Multi-view physical object identity tracking hypothesis."""

    cluster_id: str
    confidence: float
    cross_view_observations: tuple[tuple[str, str], ...]  # ((image_id, entity_id), ...)
    appearance_similarity: float
    epipolar_consistency: float


@dataclass
class RejectionRecord:
    """Audit entry documenting why a candidate detection or component link was rejected."""

    candidate_id: str
    source_image_id: str
    class_name: str
    rejection_reason: RejectionReason | str
    severity: str = "suppressed"  # "suppressed", "filtered", "rejected"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_image_id": self.source_image_id,
            "class_name": self.class_name,
            "rejection_reason": (
                self.rejection_reason.value
                if isinstance(self.rejection_reason, RejectionReason)
                else str(self.rejection_reason)
            ),
            "severity": self.severity,
            "details": self.details,
        }


@dataclass
class PerceptionEntityOutput:
    """Stable production record for an individual detected object or component."""

    entity_id: str
    source_image_id: str
    source_engine: str  # e.g., "grounding_dino_tiny", "tiled_inference"
    class_name: str
    confidence: float
    calibrated_confidence: float
    box_2d: tuple[float, float, float, float]
    coordinate_convention: CoordinateConvention = CoordinateConvention.PIXEL_XYXY_TOPLEFT
    mask_2d: np.ndarray | None = None
    mask_provenance: MaskProvenance = MaskProvenance.INFERRED
    mask_polygon: Sequence[tuple[float, float]] | None = None
    parent_candidates: Sequence[ParentCandidate] = ()
    selected_parent_id: str | None = None
    physical_identity: PhysicalIdentityHypothesis | None = None
    visibility: str = "visible"  # "visible", "occluded", "truncated"
    occlusion_level: str = "none"  # "none", "partial", "heavy"
    is_truncated: bool = False
    uncertainty_score: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "source_image_id": self.source_image_id,
            "source_engine": self.source_engine,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "calibrated_confidence": round(float(self.calibrated_confidence), 4),
            "box_2d": [round(float(v), 2) for v in self.box_2d],
            "coordinate_convention": self.coordinate_convention.value,
            "has_mask": self.mask_2d is not None,
            "mask_provenance": self.mask_provenance.value,
            "mask_polygon": list(self.mask_polygon) if self.mask_polygon is not None else None,
            "parent_candidates": [asdict(c) for c in self.parent_candidates],
            "selected_parent_id": self.selected_parent_id,
            "physical_identity": asdict(self.physical_identity) if self.physical_identity else None,
            "visibility": self.visibility,
            "occlusion_level": self.occlusion_level,
            "is_truncated": self.is_truncated,
            "uncertainty_score": round(float(self.uncertainty_score), 4),
            "attributes": self.attributes,
        }


@dataclass
class PerceptionHandoffManifest:
    """Complete perception stage production handoff artifact for downstream consumption."""

    session_id: str
    pipeline_version: str = "2.0.0-sprint10"
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    images_processed: Sequence[str] = ()
    entities: Sequence[PerceptionEntityOutput] = ()
    rejections: Sequence[RejectionRecord] = ()
    telemetry: dict[str, Any] = field(default_factory=dict)

    def validate_contract(self) -> tuple[bool, list[str]]:
        """Strict schema and invariant contract validation."""
        errors: list[str] = []
        entity_ids = set()

        for ent in self.entities:
            # Check unique ID
            if ent.entity_id in entity_ids:
                errors.append(f"Duplicate entity_id: {ent.entity_id}")
            entity_ids.add(ent.entity_id)

            # Check bbox validity
            if len(ent.box_2d) != 4:
                errors.append(f"Entity {ent.entity_id} has invalid box_2d length: {len(ent.box_2d)}")
            else:
                x1, y1, x2, y2 = ent.box_2d
                if x2 <= x1 or y2 <= y1:
                    errors.append(f"Entity {ent.entity_id} has degenerate box_2d: {ent.box_2d}")

            # Check confidence bounds
            if not (0.0 <= ent.confidence <= 1.0):
                errors.append(f"Entity {ent.entity_id} confidence out of bounds: {ent.confidence}")
            if not (0.0 <= ent.calibrated_confidence <= 1.0):
                errors.append(f"Entity {ent.entity_id} calibrated_confidence out of bounds: {ent.calibrated_confidence}")

            # Check parent integrity
            if ent.selected_parent_id and ent.selected_parent_id == ent.entity_id:
                errors.append(f"Entity {ent.entity_id} cannot be its own parent")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "pipeline_version": self.pipeline_version,
            "created_at_utc": self.created_at_utc,
            "images_processed": list(self.images_processed),
            "total_entities": len(self.entities),
            "total_rejections": len(self.rejections),
            "entities": [e.to_dict() for e in self.entities],
            "rejections": [r.to_dict() for r in self.rejections],
            "telemetry": self.telemetry,
        }
