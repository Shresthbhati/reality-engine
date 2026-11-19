"""Independent engine observations: the raw, immutable record of what ONE engine said about ONE thing.

An observation is never edited and never merged away. Consensus, refinement and WorldIR all *refer* to observation ids; the
originals stay in the evidence store so any reconstructed value can be traced back to who predicted it, from which pixels,
in which coordinate frame, with what configuration, and whether the producing engine fell back to something weaker.

The id is a content hash of everything that defines the observation, so re-adding the same observation is idempotent and a
changed value can never silently reuse an old id.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Tuple


class ObservationError(ValueError):
    """The observation is malformed (no engine, no coordinate frame, non-finite value, ...)."""


def _canonical(value: Any) -> Any:
    """JSON-stable form: mappings sorted, tuples as lists, floats kept exact (repr round-trips)."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ObservationError(f"non-finite number {value!r} cannot be stored as evidence")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ObservationError(f"unsupported value type {type(value).__name__} in an observation")


@dataclass(frozen=True)
class ObservationDraft:
    """What an adapter returns from ``infer``. The runner adds engine id, version, timestamp and execution metadata, so an
    adapter cannot misreport which engine it is."""

    task: str                                   # e.g. "depth", "detection", "pose"
    quantity: str                               # what is measured: "depth_map", "bbox", "camera_pose", "plane", ...
    source_id: str                              # image / video id
    coordinate_system: str                      # frame id the values are expressed in; REQUIRED (never "unknown")
    predicted: Mapping[str, Any]                # the engine's values, in ``coordinate_system``
    frame: int | None = None                 # video frame index, if any
    image_region: Tuple[float, float, float, float] | None = None  # x0, y0, x1, y1 in source pixels
    original_coordinates: Mapping[str, Any] | None = None  # values before any conversion, in the engine's own convention
    confidence: float | None = None          # the engine's own number, if it emits one; None = not emitted
    uncertainty: Mapping[str, Any] | None = None  # e.g. {"std": ..} or {"covariance": ..}; None = unknown
    calibration: Mapping[str, Any] | None = None  # how the confidence/uncertainty was calibrated; None = UNCALIBRATED
    relations: Tuple[Tuple[str, str], ...] = ()  # (kind, other_observation_id)


@dataclass(frozen=True)
class EngineObservation:
    observation_id: str
    engine_id: str
    model_version: str
    task: str
    quantity: str
    source_id: str
    coordinate_system: str
    predicted: Mapping[str, Any]
    frame: int | None
    image_region: Tuple[float, float, float, float] | None
    original_coordinates: Mapping[str, Any] | None
    confidence: float | None
    uncertainty: Mapping[str, Any] | None
    calibration: Mapping[str, Any] | None
    execution_config: Mapping[str, Any]
    processed_at: str
    fallback: Mapping[str, Any] | None       # None = ran as planned; else what was substituted and why
    relations: Tuple[Tuple[str, str], ...] = ()

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def to_dict(self) -> dict:
        return {
            "observation_id": self.observation_id, "engine_id": self.engine_id, "model_version": self.model_version,
            "task": self.task, "quantity": self.quantity, "source_id": self.source_id,
            "coordinate_system": self.coordinate_system, "predicted": _canonical(self.predicted), "frame": self.frame,
            "image_region": list(self.image_region) if self.image_region else None,
            "original_coordinates": _canonical(self.original_coordinates) if self.original_coordinates is not None else None,
            "confidence": self.confidence,
            "uncertainty": _canonical(self.uncertainty) if self.uncertainty is not None else None,
            "calibration": _canonical(self.calibration) if self.calibration is not None else None,
            "execution_config": _canonical(self.execution_config), "processed_at": self.processed_at,
            "fallback": _canonical(self.fallback) if self.fallback is not None else None,
            "relations": [list(r) for r in self.relations],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EngineObservation:
        region = d.get("image_region")
        return cls(
            observation_id=d["observation_id"], engine_id=d["engine_id"], model_version=d["model_version"], task=d["task"],
            quantity=d["quantity"], source_id=d["source_id"], coordinate_system=d["coordinate_system"],
            predicted=d["predicted"], frame=d.get("frame"), image_region=tuple(region) if region else None,
            original_coordinates=d.get("original_coordinates"), confidence=d.get("confidence"),
            uncertainty=d.get("uncertainty"), calibration=d.get("calibration"), execution_config=d.get("execution_config", {}),
            processed_at=d["processed_at"], fallback=d.get("fallback"),
            relations=tuple(tuple(r) for r in d.get("relations", [])),
        )


def content_id(core: Mapping[str, Any]) -> str:
    blob = json.dumps(_canonical(core), sort_keys=True, separators=(",", ":"))
    return "obs_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def id_core(o: EngineObservation) -> dict:
    """The fields the id is the hash of (everything except the id itself and the later-attachable relations)."""
    return {
        "engine_id": o.engine_id, "model_version": o.model_version, "task": o.task, "quantity": o.quantity,
        "source_id": o.source_id, "coordinate_system": o.coordinate_system, "predicted": o.predicted, "frame": o.frame,
        "image_region": list(o.image_region) if o.image_region else None, "original_coordinates": o.original_coordinates,
        "confidence": o.confidence, "uncertainty": o.uncertainty, "calibration": o.calibration,
        "execution_config": o.execution_config, "processed_at": o.processed_at, "fallback": o.fallback,
    }


def make_observation(
    draft: ObservationDraft, *, engine_id: str, model_version: str, execution_config: Mapping[str, Any],
    fallback: Mapping[str, Any] | None = None, processed_at: str | None = None,
) -> EngineObservation:
    """Validate a draft and stamp it with the runner's facts. Rejects what would make the evidence unusable later."""
    if not engine_id or not model_version:
        raise ObservationError("an observation needs the engine id and model version that produced it")
    if not draft.coordinate_system or draft.coordinate_system.strip().lower() in ("unknown", "none", "?"):
        raise ObservationError("an observation must state its coordinate system; spatial values in an unknown frame cannot be compared")
    if not draft.source_id:
        raise ObservationError("an observation must name its source image/video")
    if draft.confidence is not None and not (0.0 <= float(draft.confidence) <= 1.0):
        raise ObservationError(f"confidence {draft.confidence!r} is outside [0, 1]")
    if draft.image_region is not None:
        x0, y0, x1, y1 = draft.image_region
        if not (x1 > x0 and y1 > y0):
            raise ObservationError(f"image_region {draft.image_region!r} is empty or inverted")
    obs = EngineObservation(
        observation_id="", engine_id=engine_id, model_version=model_version, task=draft.task, quantity=draft.quantity,
        source_id=draft.source_id, coordinate_system=draft.coordinate_system, predicted=dict(draft.predicted),
        frame=draft.frame, image_region=draft.image_region,
        original_coordinates=dict(draft.original_coordinates) if draft.original_coordinates is not None else None,
        confidence=draft.confidence, uncertainty=dict(draft.uncertainty) if draft.uncertainty is not None else None,
        calibration=dict(draft.calibration) if draft.calibration is not None else None,
        execution_config=dict(execution_config), processed_at=processed_at or datetime.now(timezone.utc).isoformat(),
        fallback=dict(fallback) if fallback is not None else None, relations=tuple(draft.relations),
    )
    return EngineObservation(**{**obs.__dict__, "observation_id": content_id(id_core(obs))})
