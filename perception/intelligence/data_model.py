"""Comprehensive versioned perception data architecture (A3.1).

Provides strict, serializable, provenance-preserving data models for:
  - Normalized object, architectural, furniture, and component observations
  - Segmentation observations and mask provenance
  - Material hypotheses and appearance descriptors
  - Spatial relationship observations and cross-view identity hypotheses
  - Model execution provenance, uncertainty, temporal state, and revision history

All coordinates, image dimensions, score semantics, and missing-data states
are explicit so downstream EvidenceMatrix, SpatialInputsV1, and WorldIR
consumers never conflate raw scores with calibrated probabilities or image-space
boxes with metric 3D geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "perception.intelligence.v1"


class ValidationError(ValueError):
    """Raised when a perception data record violates schema or physical invariants."""


class ObservationLifecycle(str, Enum):
    ACTIVE = "active"
    REVISED = "revised"
    SUPERSEDED = "superseded"
    WITHHELD = "withheld"
    REJECTED = "rejected"


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


def _validate_box_xyxy(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValidationError(f"bbox_px must have 4 coordinates, got {len(box)}")
    vals = tuple(float(v) for v in box)
    if any(not math.isfinite(v) for v in vals):
        raise ValidationError(f"bbox_px contains non-finite coordinate: {vals!r}")
    x1, y1, x2, y2 = vals
    if x1 < -1.0 or y1 < -1.0 or x2 > width + 1.0 or y2 > height + 1.0:
        raise ValidationError(
            f"bbox_px {vals!r} out of bounds for image ({width}, {height})"
        )
    if x2 <= x1 or y2 <= y1:
        raise ValidationError(f"degenerate bbox_px {vals!r}")
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def stable_observation_id(
    prefix: str,
    image_id: str,
    engine_id: str,
    label: str,
    bbox_px: tuple[float, float, float, float] | None = None,
) -> str:
    """Deterministic content-addressed observation identifier."""
    payload = {
        "prefix": prefix,
        "image_id": image_id,
        "engine_id": engine_id,
        "label": label,
        "bbox_px": [round(v, 1) for v in bbox_px] if bbox_px else None,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class CoordinateSystemSpec:
    frame_id: str  # e.g. "image:P1180141", "world_metric", "sfm_gauge"
    units: str  # "pixels" | "normalized" | "meters" | "sfm_units"
    image_width: int
    image_height: int
    origin: str = "top_left"

    def __post_init__(self) -> None:
        _require_non_empty(self.frame_id, "frame_id")
        _require_non_empty(self.units, "units")
        if int(self.image_width) <= 0 or int(self.image_height) <= 0:
            raise ValidationError(
                f"image dimensions must be positive, got ({self.image_width}, {self.image_height})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoordinateSystemSpec:
        return cls(
            frame_id=str(data["frame_id"]),
            units=str(data["units"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            origin=str(data.get("origin", "top_left")),
        )


@dataclass(frozen=True)
class ScoreSemantics:
    """Explicit separation between raw detector score and calibrated probability."""

    raw_score: float | None
    score_emitted: bool
    calibrated_probability: float | None = None
    is_calibrated: bool = False
    calibration_dataset: str | None = None
    calibration_model_version: str | None = None
    uncertainty: float | None = None
    calibration_note: str = "uncalibrated"

    def __post_init__(self) -> None:
        if self.raw_score is not None:
            r = float(self.raw_score)
            if not math.isfinite(r) or not (0.0 <= r <= 1.0):
                raise ValidationError(f"raw_score must be in [0, 1] or None, got {self.raw_score!r}")
        if self.calibrated_probability is not None:
            c = float(self.calibrated_probability)
            if not math.isfinite(c) or not (0.0 <= c <= 1.0):
                raise ValidationError(
                    f"calibrated_probability must be in [0, 1] or None, got {self.calibrated_probability!r}"
                )
            if not self.is_calibrated:
                raise ValidationError("calibrated_probability cannot be non-None when is_calibrated=False")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoreSemantics:
        return cls(**data)


@dataclass(frozen=True)
class ModelExecutionProvenance:
    engine_id: str
    architecture_family: str
    checkpoint_id: str
    model_version: str = "unknown"
    device: str = "cpu"
    wall_time_ms: float | None = None
    peak_vram_mb: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.engine_id, "engine_id")
        _require_non_empty(self.architecture_family, "architecture_family")
        _require_non_empty(self.checkpoint_id, "checkpoint_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelExecutionProvenance:
        return cls(**data)


@dataclass(frozen=True)
class DetectionProvenance:
    source_image_id: str
    execution: ModelExecutionProvenance
    inference_mode: str = "full_frame"  # "full_frame" | "tiled" | "host_crop" | "multi_scale"
    tile_id: str | None = None
    crop_origin_px: tuple[int, int] | None = None
    prompt_query: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.source_image_id, "source_image_id")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.crop_origin_px is not None:
            d["crop_origin_px"] = list(self.crop_origin_px)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectionProvenance:
        crop = tuple(int(v) for v in data["crop_origin_px"]) if data.get("crop_origin_px") else None
        return cls(
            source_image_id=str(data["source_image_id"]),
            execution=ModelExecutionProvenance.from_dict(data["execution"]),
            inference_mode=str(data.get("inference_mode", "full_frame")),
            tile_id=data.get("tile_id"),
            crop_origin_px=crop,
            prompt_query=data.get("prompt_query"),
        )


@dataclass(frozen=True)
class ObservationRevision:
    revision_number: int
    reason: str
    lifecycle: ObservationLifecycle
    previous_observation_id: str | None = None
    timestamp_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lifecycle"] = self.lifecycle.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationRevision:
        return cls(
            revision_number=int(data["revision_number"]),
            reason=str(data["reason"]),
            lifecycle=ObservationLifecycle(data["lifecycle"]),
            previous_observation_id=data.get("previous_observation_id"),
            timestamp_utc=str(data.get("timestamp_utc", "")),
        )


@dataclass(frozen=True)
class AppearanceDescriptor:
    color_histogram_24: tuple[float, ...] = ()
    embedding_vector: tuple[float, ...] = ()
    embedding_model: str | None = None
    dominant_color_name: str = "unknown"
    mean_rgb: tuple[float, float, float] | None = None
    texture_roughness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "color_histogram_24": list(self.color_histogram_24),
            "embedding_vector": list(self.embedding_vector),
            "embedding_model": self.embedding_model,
            "dominant_color_name": self.dominant_color_name,
            "mean_rgb": list(self.mean_rgb) if self.mean_rgb is not None else None,
            "texture_roughness": self.texture_roughness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppearanceDescriptor:
        mr = tuple(float(v) for v in data["mean_rgb"]) if data.get("mean_rgb") is not None else None
        return cls(
            color_histogram_24=tuple(float(v) for v in data.get("color_histogram_24", ())),
            embedding_vector=tuple(float(v) for v in data.get("embedding_vector", ())),
            embedding_model=data.get("embedding_model"),
            dominant_color_name=str(data.get("dominant_color_name", "unknown")),
            mean_rgb=mr,
            texture_roughness=data.get("texture_roughness"),
        )


@dataclass(frozen=True)
class MaterialHypothesisRecord:
    material_category: str
    material_name: str
    hypothesis_score: float
    evidence_basis: str
    is_physically_verified: bool = False  # Visual appearance is never physical proof

    def __post_init__(self) -> None:
        if self.is_physically_verified:
            raise ValidationError("Visual material hypotheses cannot claim physical verification")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterialHypothesisRecord:
        return cls(**data)


@dataclass(frozen=True)
class SegmentationObservation:
    mask_id: str
    observation_id: str
    mask_kind: str  # "visible_object_mask" | "occluded_object_hypothesis" | "architectural_surface_mask" | "structural_boundary"
    area_px: int
    box_iou: float
    solidity: float
    hole_count: int
    boundary_uncertainty: float
    quality_score: float
    rle_mask: dict[str, Any] | None = None
    polygon_px: tuple[tuple[float, float], ...] = ()
    depth_mean_relative: float | None = None
    model_id: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "observation_id": self.observation_id,
            "mask_kind": self.mask_kind,
            "area_px": self.area_px,
            "box_iou": self.box_iou,
            "solidity": self.solidity,
            "hole_count": self.hole_count,
            "boundary_uncertainty": self.boundary_uncertainty,
            "quality_score": self.quality_score,
            "rle_mask": self.rle_mask,
            "polygon_px": [list(pt) for pt in self.polygon_px],
            "depth_mean_relative": self.depth_mean_relative,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentationObservation:
        return cls(
            mask_id=str(data["mask_id"]),
            observation_id=str(data["observation_id"]),
            mask_kind=str(data["mask_kind"]),
            area_px=int(data["area_px"]),
            box_iou=float(data["box_iou"]),
            solidity=float(data["solidity"]),
            hole_count=int(data["hole_count"]),
            boundary_uncertainty=float(data["boundary_uncertainty"]),
            quality_score=float(data["quality_score"]),
            rle_mask=data.get("rle_mask"),
            polygon_px=tuple((float(p[0]), float(p[1])) for p in data.get("polygon_px", ())),
            depth_mean_relative=data.get("depth_mean_relative"),
            model_id=str(data.get("model_id", "unknown")),
        )


@dataclass(frozen=True)
class TemporalObservation:
    frame_index: int | None = None
    timestamp_s: float | None = None
    shot_id: int | None = None
    track_id: str | None = None
    motion_state: str = "unknown"  # "static" | "moved" | "appeared" | "disappeared" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalObservation:
        return cls(**data)


@dataclass(frozen=True)
class NormalizedPerceptionObservation:
    """Core versioned observation emitted by the perception intelligence subsystem."""

    observation_id: str
    schema_version: str
    raw_label: str
    canonical_label: str
    taxonomy_id: str
    category: str  # "architecture" | "furniture" | "fixture" | "component" | "object"
    bbox_px: tuple[float, float, float, float]
    coords: CoordinateSystemSpec
    scores: ScoreSemantics
    provenance: DetectionProvenance
    parent_observation_id: str | None = None
    child_observation_ids: tuple[str, ...] = ()
    competing_labels: tuple[dict[str, Any], ...] = ()
    is_partial: bool = False
    is_truncated: bool = False
    occlusion_level: str = "none"
    appearance: AppearanceDescriptor = field(default_factory=AppearanceDescriptor)
    material_hypotheses: tuple[MaterialHypothesisRecord, ...] = ()
    segmentation: SegmentationObservation | None = None
    temporal: TemporalObservation | None = None
    revision: ObservationRevision = field(
        default_factory=lambda: ObservationRevision(1, "initial_detection", ObservationLifecycle.ACTIVE)
    )
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.observation_id, "observation_id")
        _require_non_empty(self.canonical_label, "canonical_label")
        _validate_box_xyxy(self.bbox_px, self.coords.image_width, self.coords.image_height)

    @property
    def bbox_norm(self) -> tuple[float, float, float, float]:
        w, h = float(self.coords.image_width), float(self.coords.image_height)
        return (self.bbox_px[0] / w, self.bbox_px[1] / h, self.bbox_px[2] / w, self.bbox_px[3] / h)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "schema_version": self.schema_version,
            "raw_label": self.raw_label,
            "canonical_label": self.canonical_label,
            "taxonomy_id": self.taxonomy_id,
            "category": self.category,
            "bbox_px": list(self.bbox_px),
            "bbox_norm": list(self.bbox_norm),
            "coords": self.coords.to_dict(),
            "scores": self.scores.to_dict(),
            "provenance": self.provenance.to_dict(),
            "parent_observation_id": self.parent_observation_id,
            "child_observation_ids": list(self.child_observation_ids),
            "competing_labels": list(self.competing_labels),
            "is_partial": self.is_partial,
            "is_truncated": self.is_truncated,
            "occlusion_level": self.occlusion_level,
            "appearance": self.appearance.to_dict(),
            "material_hypotheses": [m.to_dict() for m in self.material_hypotheses],
            "segmentation": self.segmentation.to_dict() if self.segmentation else None,
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "revision": self.revision.to_dict(),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedPerceptionObservation:
        return cls(
            observation_id=str(data["observation_id"]),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            raw_label=str(data["raw_label"]),
            canonical_label=str(data["canonical_label"]),
            taxonomy_id=str(data["taxonomy_id"]),
            category=str(data["category"]),
            bbox_px=tuple(float(v) for v in data["bbox_px"]),
            coords=CoordinateSystemSpec.from_dict(data["coords"]),
            scores=ScoreSemantics.from_dict(data["scores"]),
            provenance=DetectionProvenance.from_dict(data["provenance"]),
            parent_observation_id=data.get("parent_observation_id"),
            child_observation_ids=tuple(data.get("child_observation_ids", ())),
            competing_labels=tuple(data.get("competing_labels", ())),
            is_partial=bool(data.get("is_partial", False)),
            is_truncated=bool(data.get("is_truncated", False)),
            occlusion_level=str(data.get("occlusion_level", "none")),
            appearance=AppearanceDescriptor.from_dict(data["appearance"])
            if data.get("appearance")
            else AppearanceDescriptor(),
            material_hypotheses=tuple(
                MaterialHypothesisRecord.from_dict(m) for m in data.get("material_hypotheses", ())
            ),
            segmentation=SegmentationObservation.from_dict(data["segmentation"])
            if data.get("segmentation")
            else None,
            temporal=TemporalObservation.from_dict(data["temporal"])
            if data.get("temporal")
            else None,
            revision=ObservationRevision.from_dict(data["revision"])
            if data.get("revision")
            else ObservationRevision(1, "deserialized", ObservationLifecycle.ACTIVE),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class ArchitecturalObservation:
    observation: NormalizedPerceptionObservation
    structural_role: str  # "primary_surface" | "structural_support" | "opening_candidate" | "verified_opening" | "frame" | "boundary"
    host_wall_id: str | None = None
    host_wall_hypothesis_state: str = "unknown"  # "observed" | "hypothesized" | "unknown"
    boundary_continuity_score: float | None = None
    is_verified_opening: bool = False
    competing_structural_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "structural_role": self.structural_role,
            "host_wall_id": self.host_wall_id,
            "host_wall_hypothesis_state": self.host_wall_hypothesis_state,
            "boundary_continuity_score": self.boundary_continuity_score,
            "is_verified_opening": self.is_verified_opening,
            "competing_structural_labels": list(self.competing_structural_labels),
        }


@dataclass(frozen=True)
class FurnitureObservation:
    observation: NormalizedPerceptionObservation
    hierarchy_path: tuple[str, ...]
    component_ids: tuple[str, ...] = ()
    supporting_surface_id: str | None = None
    mobility: str = "movable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "hierarchy_path": list(self.hierarchy_path),
            "component_ids": list(self.component_ids),
            "supporting_surface_id": self.supporting_surface_id,
            "mobility": self.mobility,
        }


@dataclass(frozen=True)
class SpatialRelationshipObservation:
    relationship_id: str
    subject_id: str
    predicate: str
    object_id: str
    coordinate_frame: str  # "image_space" | "world_metric"
    support_state: str  # "visually_observed" | "geometrically_supported" | "semantically_inferred" | "contradictory"
    confidence_score: float
    supporting_images: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "coordinate_frame": self.coordinate_frame,
            "support_state": self.support_state,
            "confidence_score": self.confidence_score,
            "supporting_images": list(self.supporting_images),
            "contradictions": list(self.contradictions),
        }


@dataclass(frozen=True)
class CrossViewIdentityHypothesis:
    identity_id: str
    member_observation_ids: tuple[str, ...]
    canonical_label: str
    status: str  # "confirmed_match" | "ambiguous" | "singleton" | "split_proposed" | "merge_proposed"
    identity_score: float
    competing_identity_ids: tuple[str, ...] = ()
    evidence_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "member_observation_ids": list(self.member_observation_ids),
            "canonical_label": self.canonical_label,
            "status": self.status,
            "identity_score": self.identity_score,
            "competing_identity_ids": list(self.competing_identity_ids),
            "evidence_reasons": list(self.evidence_reasons),
        }


@dataclass
class PerceptionBundle:
    """Top-level serializable container for one perception run."""

    run_id: str
    schema_version: str = SCHEMA_VERSION
    observations: list[NormalizedPerceptionObservation] = field(default_factory=list)
    architectural: list[ArchitecturalObservation] = field(default_factory=list)
    furniture: list[FurnitureObservation] = field(default_factory=list)
    relationships: list[SpatialRelationshipObservation] = field(default_factory=list)
    identities: list[CrossViewIdentityHypothesis] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "observations": [o.to_dict() for o in self.observations],
            "architectural": [a.to_dict() for a in self.architectural],
            "furniture": [f.to_dict() for f in self.furniture],
            "relationships": [r.to_dict() for r in self.relationships],
            "identities": [i.to_dict() for i in self.identities],
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerceptionBundle:
        bundle = cls(
            run_id=str(data["run_id"]),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            observations=[
                NormalizedPerceptionObservation.from_dict(o) for o in data.get("observations", ())
            ],
            diagnostics=dict(data.get("diagnostics", {})),
        )
        return bundle
