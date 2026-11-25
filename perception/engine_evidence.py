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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
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
    bbox_px: Optional[Tuple[float, float, float, float]] = None
    world_pos: Optional[Tuple[float, float, float]] = None
    depth_map_ref: Optional[str] = None
    point_samples: Optional[Tuple[Tuple[float, float, float], ...]] = None
    # Attributes
    material_hypothesis: Optional[str] = None
    color_rgb: Optional[Tuple[int, int, int]] = None
    surface_normal: Optional[Tuple[float, float, float]] = None
    # Uncertainty
    confidence: Optional[float] = None
    sigma_m: Optional[float] = None
    calibration_quality: str = "unverified"
    # Provenance
    timestamp_s: Optional[float] = None
    raw_output_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    observations: Tuple[ObservationRecord, ...]
    status: str               # "succeeded" | "partial" | "failed" | "unavailable"
    error_message: str = ""
    wall_time_s: float = 0.0
    peak_vram_mb: Optional[float] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    # Cross-engine contradiction slots: populated by the orchestrator's
    # IntelligenceMerger (engine_intelligence.py)
    contradictions: Tuple[Dict[str, Any], ...] = ()

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    def to_dict(self) -> Dict[str, Any]:
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


def compare_observations(obs_a: ObservationRecord, obs_b: ObservationRecord) -> List[str]:
    """List of contradictions between two observations of the same target."""
    issues: List[str] = []
    if obs_a.label != obs_b.label and obs_a.label != "unknown":
        issues.append(f"label_mismatch: {obs_a.label!r} vs {obs_b.label!r}")
    if obs_a.world_pos and obs_b.world_pos:
        dist = np.linalg.norm(np.array(obs_a.world_pos) - np.array(obs_b.world_pos))
        if dist > 0.1:  # 10cm tolerance
            issues.append(f"position_mismatch: {dist:.3f} m")
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
]
