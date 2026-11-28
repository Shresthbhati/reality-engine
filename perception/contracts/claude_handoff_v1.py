"""Versioned Claude Handoff Contract: ``perception_handoff.v1`` (A4.12).

Defines the stable, schema-validated boundary between Gemini's perception
subsystem and Claude's geometric reconstruction pipeline without requiring
Claude to import Gemini's internal modules or vice-versa.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from perception.intelligence.data_model import (
    NormalizedPerceptionObservation,
    PerceptionBundle,
)

HANDOFF_SCHEMA_VERSION = "perception_handoff.v1"
SUPPORTED_VERSIONS = frozenset({HANDOFF_SCHEMA_VERSION})


class HandoffValidationError(ValueError):
    """Raised when a perception handoff payload fails schema or contract validation."""


@dataclass(frozen=True)
class HandoffObservation:
    observation_id: str
    source_image_id: str
    camera_id: str | None
    image_width: int
    image_height: int
    coordinate_frame: str
    bbox_px: tuple[float, float, float, float]
    mask_ref: str | None
    semantic_class: str
    taxonomy_id: str
    component_hierarchy: tuple[str, ...]
    parent_observation_id: str | None
    relationship_hypotheses: tuple[dict[str, Any], ...]
    depth_ref: str | None
    cross_view_identity_id: str | None
    raw_scores: dict[str, float | None]
    calibrated_probability: float | None
    is_calibrated: bool
    uncertainty: float | None
    provenance: dict[str, Any]
    revision_number: int
    lifecycle_status: str

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise HandoffValidationError("image dimensions must be positive")
        if len(self.bbox_px) != 4 or any(not math.isfinite(v) for v in self.bbox_px):
            raise HandoffValidationError("bbox_px must contain 4 finite coordinates")
        if self.bbox_px[2] <= self.bbox_px[0] or self.bbox_px[3] <= self.bbox_px[1]:
            raise HandoffValidationError("degenerate bbox_px")
        if self.calibrated_probability is not None and not self.is_calibrated:
            raise HandoffValidationError("calibrated_probability must be None when is_calibrated=False")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox_px"] = list(self.bbox_px)
        d["component_hierarchy"] = list(self.component_hierarchy)
        d["relationship_hypotheses"] = list(self.relationship_hypotheses)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffObservation:
        return cls(
            observation_id=str(data["observation_id"]),
            source_image_id=str(data["source_image_id"]),
            camera_id=data.get("camera_id"),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            coordinate_frame=str(data["coordinate_frame"]),
            bbox_px=tuple(float(v) for v in data["bbox_px"]),
            mask_ref=data.get("mask_ref"),
            semantic_class=str(data["semantic_class"]),
            taxonomy_id=str(data["taxonomy_id"]),
            component_hierarchy=tuple(data.get("component_hierarchy", ())),
            parent_observation_id=data.get("parent_observation_id"),
            relationship_hypotheses=tuple(data.get("relationship_hypotheses", ())),
            depth_ref=data.get("depth_ref"),
            cross_view_identity_id=data.get("cross_view_identity_id"),
            raw_scores=dict(data.get("raw_scores", {})),
            calibrated_probability=data.get("calibrated_probability"),
            is_calibrated=bool(data.get("is_calibrated", False)),
            uncertainty=data.get("uncertainty"),
            provenance=dict(data.get("provenance", {})),
            revision_number=int(data.get("revision_number", 1)),
            lifecycle_status=str(data.get("lifecycle_status", "active")),
        )


@dataclass(frozen=True)
class HandoffBundle:
    run_id: str
    schema_version: str = HANDOFF_SCHEMA_VERSION
    observations: tuple[HandoffObservation, ...] = ()
    rejected: tuple[dict[str, str], ...] = ()

    def to_json(self) -> str:
        return json.dumps({
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "observations": [o.to_dict() for o in self.observations],
            "rejected": list(self.rejected),
        }, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> HandoffBundle:
        data = json.loads(text)
        ver = str(data.get("schema_version", ""))
        if ver not in SUPPORTED_VERSIONS:
            raise HandoffValidationError(f"unsupported handoff schema_version: {ver!r}")
        return cls(
            run_id=str(data["run_id"]),
            schema_version=ver,
            observations=tuple(HandoffObservation.from_dict(o) for o in data.get("observations", ())),
            rejected=tuple(data.get("rejected", ())),
        )


def export_claude_handoff_bundle(
    bundle: PerceptionBundle,
    *,
    camera_ids_by_image: dict[str, str] | None = None,
    depth_refs_by_image: dict[str, str] | None = None,
) -> HandoffBundle:
    """Convert a ``PerceptionBundle`` into the versioned ``perception_handoff.v1`` contract."""
    cam_map = camera_ids_by_image or {}
    depth_map = depth_refs_by_image or {}

    id_by_obs: dict[str, str] = {}
    for ident in bundle.identities:
        for m in ident.member_observation_ids:
            id_by_obs[m] = ident.identity_id

    rels_by_subj: dict[str, list[dict[str, Any]]] = {}
    for r in bundle.relationships:
        rels_by_subj.setdefault(r.subject_id, []).append(r.to_dict())

    accepted: list[HandoffObservation] = []
    rejected: list[dict[str, str]] = []

    for o in bundle.observations:
        try:
            src_img = o.provenance.source_image_id
            ho = HandoffObservation(
                observation_id=o.observation_id,
                source_image_id=src_img,
                camera_id=cam_map.get(src_img),
                image_width=o.coords.image_width,
                image_height=o.coords.image_height,
                coordinate_frame=o.coords.frame_id,
                bbox_px=o.bbox_px,
                mask_ref=o.segmentation.mask_id if o.segmentation else None,
                semantic_class=o.canonical_label,
                taxonomy_id=o.taxonomy_id,
                component_hierarchy=tuple(o.taxonomy_id.split(".")),
                parent_observation_id=o.parent_observation_id,
                relationship_hypotheses=tuple(rels_by_subj.get(o.observation_id, ())),
                depth_ref=depth_map.get(src_img),
                cross_view_identity_id=id_by_obs.get(o.observation_id),
                raw_scores={o.provenance.execution.engine_id: o.scores.raw_score},
                calibrated_probability=o.scores.calibrated_probability,
                is_calibrated=o.scores.is_calibrated,
                uncertainty=o.scores.uncertainty,
                provenance=o.provenance.to_dict(),
                revision_number=o.revision.revision_number,
                lifecycle_status=o.revision.lifecycle.value,
            )
            accepted.append(ho)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"observation_id": o.observation_id, "reason": str(exc)})

    return HandoffBundle(
        run_id=bundle.run_id,
        schema_version=HANDOFF_SCHEMA_VERSION,
        observations=tuple(accepted),
        rejected=tuple(rejected),
    )


__all__ = [
    "HANDOFF_SCHEMA_VERSION",
    "HandoffBundle",
    "HandoffObservation",
    "HandoffValidationError",
    "export_claude_handoff_bundle",
]
