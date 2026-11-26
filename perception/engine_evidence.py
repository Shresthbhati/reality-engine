"""Unified engine evidence representation — the common output format for ALL
Reality Engine perception engines.

Every engine (detection, segmentation, depth, matching, camera, fusion, etc.)
emits its results as ``UnifiedEngineOutput``. This gives every engine the
same contract:

  * Evidence identity — source image ID, evidence ID, coordinate frame
  * Geometry — 3-D point samples, 2-D bounding boxes, segmentation masks
  * Attributes — label, category, material, color
  * Uncertainty — measurement uncertainty, calibration state, confidence
  * Model identity — engine ID, architecture, checkpoint, version, license
  * Provenance — how the observation was made
  * Contradictions — other engines' conflicting claims on the same target

Observations must be non-destructive (record, not overwrite) and carry
provenance for every field.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelIdentity:
    """Engine/model identity for provenance tracking."""
    engine_id: str          # "detection", "segmentation", "depth", "matching", ...
    architecture_id: str    # "grounding_dino", "sam2", "mi_da_s", "lightglue_aliked", ...
    checkpoint_id: str      # "none", model name, or HF repo id
    version: str = "unknown"
    license: str = "unknown"
    non_commercial: bool = False
    hardware_class: str = "gpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "architecture_id": self.architecture_id,
            "checkpoint_id": self.checkpoint_id,
            "version": self.version,
            "license": self.license,
            "non_commercial": self.non_commercial,
            "hardware_class": self.hardware_class,
        }


@dataclass(frozen=True)
class CoordinateFrame:
    """Coordinate frame and scale state of an observation.

    metric: the values are in metres.
    sfm_gauge: SfM gauge units (relative).
    relative: fully relative (Monodepth-style inverse depth).
    image: pixel coordinates of the source image.
    """
    frame_id: str = "world"
    scale_state: str = "relative"  # "metric" | "sfm_gauge" | "relative" | "image"
    units: str = "relative"

    def to_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "scale_state": self.scale_state, "units": self.units}


@dataclass(frozen=True)
class ObservationRecord:
    """One engine's observation of one target — the atomic unit of evidence."""
    evidence_id: str
    source_image_id: str
    coordinate_frame: CoordinateFrame
    model_id: ModelIdentity
    label: str
    category: str
    # Geometry (various projections, all from the SAME source image)
    bbox_px: tuple[float, float, float, float] | None = None
    world_pos: tuple[float, float, float] | None = None
    depth_map_ref: str | None = None
    point_samples: tuple[tuple[float, float, float], ...] | None = None
    # Attributes
    material_hypothesis: str | None = None
    color_rgb: tuple[int, int, int] | None = None
    surface_normal: tuple[float, float, float] | None = None
    # Uncertainty
    confidence: float | None = None
    sigma_m: float | None = None
    calibration_quality: str = "unverified"
    # Provenance
    timestamp_s: float | None = None
    raw_output_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_image_id": self.source_image_id,
            "coordinate_frame": self.coordinate_frame.to_dict(),
            "model_id": self.model_id.to_dict(),
            "label": self.label,
            "category": self.category,
            "bbox_px": list(self.bbox_px) if self.bbox_px else None,
            "world_pos": list(self.world_pos) if self.world_pos else None,
            "depth_map_ref": self.depth_map_ref,
            "point_samples": [list(p) for p in self.point_samples] if self.point_samples else None,
            "material_hypothesis": self.material_hypothesis,
            "color_rgb": list(self.color_rgb) if self.color_rgb else None,
            "surface_normal": list(self.surface_normal) if self.surface_normal else None,
            "confidence": self.confidence,
            "sigma_m": self.sigma_m,
            "calibration_quality": self.calibration_quality,
            "timestamp_s": self.timestamp_s,
            "raw_output_ref": self.raw_output_ref,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class UnifiedEngineOutput:
    """Top-level result from any engine: observations + summary + contradictions.

    Every engine returns this shape. The orchestrator merges these across
    engines via ``IntelligenceMerger`` (engine_intelligence.py).
    """
    engine_id: str
    run_id: str
    observations: tuple[ObservationRecord, ...]
    status: str               # "succeeded" | "partial" | "failed" | "unavailable"
    error_message: str = ""
    wall_time_s: float = 0.0
    peak_vram_mb: float | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    # Cross-engine contradiction slots: populated by the orchestrator's
    # IntelligenceMerger (engine_intelligence.py)
    contradictions: tuple[dict[str, Any], ...] = ()

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "run_id": self.run_id,
            "status": self.status,
            "error_message": self.error_message,
            "wall_time_s": self.wall_time_s,
            "peak_vram_mb": self.peak_vram_mb,
            "n_observations": self.n_observations,
            "summary": dict(self.summary),
            "contradictions": [dict(c) for c in self.contradictions],
            "observations": [o.to_dict() for o in self.observations],
        }


def compute_observation_hash(obs: ObservationRecord) -> str:
    """Deterministic hash of one observation — used for cross-engine dedup."""
    d = obs.to_dict()
    d.pop("evidence_id", None)  # dedup key is content, not ID
    d.pop("raw_output_ref", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


#: Scale states whose positions are comparable with each other inside one frame id.
_COMPARABLE_SCALE_STATES = frozenset({"metric", "sfm_gauge"})
#: Position tolerance per scale state. A gauge unit is not a metre, so the two are
#: compared on their own terms and the message names the unit that was used.
_POSITION_TOLERANCE = {"metric": 0.1, "sfm_gauge": 0.1}
_UNIT_LABEL = {"metric": "m", "sfm_gauge": "gauge units"}


def bbox_iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Intersection-over-union of two ``(x0, y0, x1, y1)`` boxes; 0.0 when either is absent or degenerate."""
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx0, by0, bx1, by1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def compare_observations(obs_a: ObservationRecord, obs_b: ObservationRecord) -> list[str]:
    """List of disagreements between two observations that may be the same target.

    Geometry is only compared when it is comparable: the same coordinate frame id
    and the same scale state, and that state must itself carry a length. Two
    observations in different frames or scales get a ``scale_state_mismatch`` note
    instead of a position contradiction -- calling a difference of relative-depth
    values a distance in metres would be an invented measurement.
    """
    issues: list[str] = []
    if obs_a.label != obs_b.label and obs_a.label != "unknown" and obs_b.label != "unknown":
        issues.append(f"label_mismatch: {obs_a.label!r} vs {obs_b.label!r}")

    fa, fb = obs_a.coordinate_frame, obs_b.coordinate_frame
    if obs_a.world_pos and obs_b.world_pos:
        same_frame = fa.frame_id == fb.frame_id
        same_scale = fa.scale_state == fb.scale_state
        comparable = same_frame and same_scale and fa.scale_state in _COMPARABLE_SCALE_STATES
        if not comparable:
            issues.append(
                "scale_state_mismatch: positions are not comparable "
                f"({fa.frame_id}/{fa.scale_state} {fa.units} vs {fb.frame_id}/{fb.scale_state} {fb.units})"
            )
        else:
            dist = float(np.linalg.norm(np.array(obs_a.world_pos) - np.array(obs_b.world_pos)))
            tolerance = _POSITION_TOLERANCE[fa.scale_state]
            if dist > tolerance:
                issues.append(f"position_mismatch: {dist:.3f} {_UNIT_LABEL[fa.scale_state]}")

    if obs_a.material_hypothesis and obs_b.material_hypothesis:
        if obs_a.material_hypothesis != obs_b.material_hypothesis:
            issues.append(f"material_mismatch: {obs_a.material_hypothesis!r} vs {obs_b.material_hypothesis!r}")
    return issues


__all__ = [
    "ModelIdentity",
    "CoordinateFrame",
    "ObservationRecord",
    "UnifiedEngineOutput",
    "compute_observation_hash",
    "compare_observations",
    "bbox_iou",
]
