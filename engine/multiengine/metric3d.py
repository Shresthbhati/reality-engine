"""The shared metric-3D observation contract (``metric3d.v1``).

One record shape for everything that produces a spatial value -- depth, 3D position, dimensions, camera-frame points -- so the
producers (depth, lifting, detection) and the consumers (evidence matrix, canonical fusion, WorldIR bridge) cannot disagree about
what a number means. It is ADDITIVE: the record rides inside ``EngineObservation.predicted["metric3d"]`` while ``predicted["value"]``
keeps the vector every existing consumer already reads, so the observation id hash, the evidence store and old worlds are unchanged.

Hard rules (each enforced by ``problems`` and refused at ``ingest``, never silently repaired):

* RELATIVE depth is not metric depth. ``scale_state`` is one of RELATIVE_DEPTH / SFM_GAUGE / METRIC_ANCHORED / UNKNOWN and the
  ``units`` must agree with it (relative|unitless, sfm_units, m, unknown). "m" is only legal with a METRIC_ANCHORED state AND a
  ``metric_anchor_ref`` naming the independent calibration that established the scale. COLMAP gauge units stay ``sfm_units``.
* A value without a coordinate frame, or in frame "unknown", is refused. A covariance must be finite, symmetric, positive
  semi-definite and match the value's dimension. A confidence needs a stated basis; the raw detector number is never relabelled
  "calibrated".
* A valid observation must carry a ``raw_artifact_ref`` so the engine's raw output stays recoverable.
* Refused records are NOT discarded: ``ingest`` writes them to the evidence matrix's rejected-prediction log with the reason.
* Versioning: ``from_dict`` reads ``metric3d.v1`` only and refuses an unknown version loudly; fields may be added in a later minor
  version only as optional, so a v1 reader keeps working.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from engine.multiengine.evidence_matrix import (
    EvidenceContext,
    EvidenceMatrix,
    MatrixError,
    RejectedPrediction,
)
from engine.multiengine.observation import (
    EngineObservation,
    ObservationDraft,
    ObservationError,
    make_observation,
)

SCHEMA_VERSION = "metric3d.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONFIDENCE_BASES = ("raw_model", "calibrated", "heuristic")
EXECUTION_STATUSES = ("succeeded", "partial")
VALIDATION_STATUSES = ("valid", "rejected")


class ScaleState(str, Enum):
    RELATIVE_DEPTH = "relative_depth"  # up to an unknown per-image scale/shift (e.g. monocular depth); NOT a length
    SFM_GAUGE = "sfm_gauge"  # consistent within one reconstruction, arbitrary global scale (COLMAP units)
    SYNTHETIC_SCALE_ALIGNED = "synthetic_scale_aligned"  # metric scale from synthetic/simulated ground truth; NOT physically surveyed
    METRIC_ANCHORED = "metric_anchored"  # metres, established by an independent calibration (``metric_anchor_ref``)
    UNKNOWN = "unknown"


_UNITS = {
    ScaleState.RELATIVE_DEPTH: ("relative", "unitless"),
    ScaleState.SFM_GAUGE: ("sfm_units",),
    ScaleState.SYNTHETIC_SCALE_ALIGNED: ("synthetic_m", "synthetic_metres", "synthetic_meters", "m"),
    ScaleState.METRIC_ANCHORED: ("m",),
    ScaleState.UNKNOWN: ("unknown",),
}


class Metric3DError(ValueError):
    pass


@dataclass(frozen=True)
class Metric3DObservation:
    engine_id: str
    architecture_id: str
    checkpoint_id: str
    source_id: str  # image / video id
    quantity: str  # "depth_relative" | "depth_metric" | "position" | "dimensions" | ...
    value: tuple[float, ...]
    units: str
    scale_state: ScaleState
    frame: str  # coordinate frame the value is expressed in
    execution_status: str = "succeeded"
    validation_status: str = "valid"
    checkpoint_sha256: str | None = None  # None = not recorded (a gap the report must show, never filled in)
    camera_id: str | None = None
    image_size: tuple[int, int] | None = None  # original (width, height) in pixels
    image_region: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1 in the ORIGINAL image
    mask_ref: str | None = None  # artifact:// of the mask the value refers to
    metric_anchor_ref: str | None = None
    confidence: float | None = None  # the engine's own number
    confidence_basis: str | None = None  # raw_model | calibrated | heuristic
    covariance: tuple[tuple[float, ...], ...] | None = None  # in ``frame`` and ``units``; None = unknown
    raw_artifact_ref: str | None = None
    rejection_reason: str | None = None
    observation_id: str = ""  # assigned by the evidence store at ingest; "" before
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ validation
    def problems(self) -> list[str]:
        p: list[str] = []
        for name in ("engine_id", "architecture_id", "checkpoint_id", "source_id", "quantity"):
            if not getattr(self, name):
                p.append(f"{name} is required")
        if not self.frame or self.frame.strip().lower() in ("unknown", "none", "?"):
            p.append("a coordinate frame is required (a value in an unknown frame cannot be compared)")
        if not isinstance(self.scale_state, ScaleState):
            p.append(f"scale_state {self.scale_state!r} is not a ScaleState")
            return p
        has_value = bool(self.value)
        if has_value and not all(isinstance(v, (int, float)) and math.isfinite(v) for v in self.value):
            p.append("value must be a vector of finite numbers")
        if not has_value and self.validation_status == "valid":
            p.append("a valid observation needs a non-empty value; a rejected one may carry none (no placeholder is invented)")
        if self.units not in _UNITS[self.scale_state]:
            p.append(f"units {self.units!r} contradict scale_state {self.scale_state.value} (allowed: {_UNITS[self.scale_state]})")
        if self.scale_state is ScaleState.METRIC_ANCHORED and not self.metric_anchor_ref:
            p.append("metres need a metric_anchor_ref naming the independent calibration; none was given")
        if self.scale_state not in (ScaleState.METRIC_ANCHORED, ScaleState.SYNTHETIC_SCALE_ALIGNED) and self.metric_anchor_ref:
            p.append("a metric_anchor_ref was given but the scale_state is not METRIC_ANCHORED or SYNTHETIC_SCALE_ALIGNED")
        if self.quantity == "depth_metric" and self.scale_state not in (ScaleState.METRIC_ANCHORED, ScaleState.SYNTHETIC_SCALE_ALIGNED):
            p.append("quantity depth_metric requires METRIC_ANCHORED or SYNTHETIC_SCALE_ALIGNED; relative or gauge depth must be named depth_relative")
        if self.quantity == "depth_relative" and self.scale_state is not ScaleState.RELATIVE_DEPTH:
            p.append("quantity depth_relative requires scale_state RELATIVE_DEPTH")
        if self.confidence is not None:
            if not (isinstance(self.confidence, (int, float)) and 0.0 <= float(self.confidence) <= 1.0):
                p.append(f"confidence {self.confidence!r} is outside [0, 1]")
            if self.confidence_basis not in CONFIDENCE_BASES:
                p.append(f"a confidence needs a basis in {CONFIDENCE_BASES}")
        elif self.confidence_basis is not None:
            p.append("a confidence_basis was given without a confidence")
        if self.checkpoint_sha256 is not None and not _SHA256.match(self.checkpoint_sha256):
            p.append("checkpoint_sha256 must be 64 lowercase hex characters")
        if self.image_size is not None and not (len(self.image_size) == 2 and all(isinstance(v, int) and v > 0 for v in self.image_size)):
            p.append("image_size must be two positive integers (width, height)")
        if self.image_region is not None:
            x0, y0, x1, y1 = self.image_region
            if not (all(math.isfinite(v) for v in self.image_region) and x1 > x0 and y1 > y0):
                p.append("image_region is empty, inverted or non-finite")
            elif self.image_size is not None and (x0 < 0 or y0 < 0 or x1 > self.image_size[0] or y1 > self.image_size[1]):
                p.append("image_region lies outside the original image")
        p.extend(self._covariance_problems())
        for name in ("raw_artifact_ref", "mask_ref", "metric_anchor_ref"):
            ref = getattr(self, name)
            if ref is not None and not str(ref).startswith(("artifact://", "anchor://")):
                p.append(f"{name} must be an artifact:// (or anchor://) reference")
        if self.execution_status not in EXECUTION_STATUSES:
            p.append(f"execution_status must be one of {EXECUTION_STATUSES}")
        if self.validation_status not in VALIDATION_STATUSES:
            p.append(f"validation_status must be one of {VALIDATION_STATUSES}")
        elif self.validation_status == "valid":
            if self.rejection_reason:
                p.append("a valid observation cannot carry a rejection_reason")
            if not self.raw_artifact_ref:
                p.append("a valid observation needs a raw_artifact_ref so the engine's raw output stays recoverable")
        elif not self.rejection_reason:
            p.append("a rejected observation needs a rejection_reason")
        return p

    def _covariance_problems(self) -> list[str]:
        if self.covariance is None:
            return []
        try:
            c = np.asarray(self.covariance, dtype=float)
        except (TypeError, ValueError):
            return ["covariance is not a numeric matrix"]
        n = len(self.value)
        if c.shape != (n, n):
            return [f"covariance must be {n}x{n} to match the value, got {c.shape}"]
        if not np.isfinite(c).all():
            return ["covariance contains non-finite numbers"]
        scale = max(1.0, float(np.abs(c).max()))
        if not np.allclose(c, c.T, atol=1e-9 * scale):
            return ["covariance is not symmetric"]
        if float(np.linalg.eigvalsh(0.5 * (c + c.T)).min()) < -1e-9 * scale:
            return ["covariance is not positive semi-definite"]
        return []

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "observation_id": self.observation_id, "engine_id": self.engine_id,
            "architecture_id": self.architecture_id, "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256, "source_id": self.source_id, "camera_id": self.camera_id,
            "image_size": None if self.image_size is None else list(self.image_size),
            "image_region": None if self.image_region is None else list(self.image_region), "mask_ref": self.mask_ref,
            "quantity": self.quantity, "value": list(self.value), "units": self.units, "frame": self.frame,
            "scale_state": self.scale_state.value, "metric_anchor_ref": self.metric_anchor_ref,
            "confidence": self.confidence, "confidence_basis": self.confidence_basis,
            "covariance": None if self.covariance is None else [list(r) for r in self.covariance],
            "raw_artifact_ref": self.raw_artifact_ref, "execution_status": self.execution_status,
            "validation_status": self.validation_status, "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Metric3DObservation:
        ver = d.get("schema_version")
        if ver != SCHEMA_VERSION:
            raise Metric3DError(f"unsupported metric-3D schema version {ver!r}; this reader understands {SCHEMA_VERSION!r} only")
        try:
            state = ScaleState(d["scale_state"])
        except (KeyError, ValueError) as exc:
            raise Metric3DError(f"invalid scale_state: {d.get('scale_state')!r}") from exc
        region, size, cov = d.get("image_region"), d.get("image_size"), d.get("covariance")
        return cls(
            engine_id=d.get("engine_id", ""), architecture_id=d.get("architecture_id", ""),
            checkpoint_id=d.get("checkpoint_id", ""), source_id=d.get("source_id", ""), quantity=d.get("quantity", ""),
            value=tuple(d.get("value") or ()), units=d.get("units", ""), scale_state=state, frame=d.get("frame", ""),
            execution_status=d.get("execution_status", "succeeded"), validation_status=d.get("validation_status", "valid"),
            checkpoint_sha256=d.get("checkpoint_sha256"), camera_id=d.get("camera_id"),
            image_size=None if size is None else tuple(size), image_region=None if region is None else tuple(region),
            mask_ref=d.get("mask_ref"), metric_anchor_ref=d.get("metric_anchor_ref"), confidence=d.get("confidence"),
            confidence_basis=d.get("confidence_basis"),
            covariance=None if cov is None else tuple(tuple(r) for r in cov),
            raw_artifact_ref=d.get("raw_artifact_ref"), rejection_reason=d.get("rejection_reason"),
            observation_id=d.get("observation_id", ""), schema_version=ver,
        )

    # ------------------------------------------------------------------ bridge to the evidence store
    def to_draft(self) -> ObservationDraft:
        """A draft whose ``predicted["value"]`` is the plain vector existing consumers read, with the full record alongside."""
        probs = self.problems()
        if probs:
            raise Metric3DError("; ".join(probs))
        return ObservationDraft(
            task="metric3d", quantity=self.quantity, source_id=self.source_id, coordinate_system=self.frame,
            predicted={"value": [float(v) for v in self.value], "units": self.units, "scale_state": self.scale_state.value,
                       "metric3d": {k: v for k, v in self.to_dict().items() if k != "observation_id"}},
            image_region=self.image_region, confidence=self.confidence,
            uncertainty=None if self.covariance is None else {"covariance": [list(r) for r in self.covariance]},
        )

    @classmethod
    def from_observation(cls, obs: EngineObservation) -> Metric3DObservation:
        rec = (obs.predicted or {}).get("metric3d")
        if not isinstance(rec, Mapping):
            raise Metric3DError(f"observation {obs.observation_id} carries no metric3d record")
        return cls.from_dict({**rec, "observation_id": obs.observation_id})


def ingest(
    matrix: EvidenceMatrix,
    records: Sequence[Metric3DObservation],
    *,
    run_id: str,
    model_version: str,
    execution_config: Mapping[str, Any] | None = None,
    start_index: int = 0,
    processed_at: str | None = None,
) -> dict[str, list[str]]:
    """Persist records. A valid one becomes an observation (+ context); anything invalid -- or marked rejected by the producer -- goes
    to the rejected-prediction log with its reason. Nothing is dropped and nothing is repaired. Returns ``{"accepted": ids,
    "rejected": reasons}``."""
    accepted: list[str] = []
    rejected: list[str] = []
    for i, rec in enumerate(records, start=start_index):
        probs = rec.problems()
        if rec.validation_status == "rejected" and not probs:
            probs = [f"producer rejected: {rec.rejection_reason}"]
        if probs:
            reason = "; ".join(probs)
            rejected.append(reason)
            matrix.record_rejected(RejectedPrediction(
                run_id, "metric3d", rec.engine_id or "unknown", model_version, i, reason,
                {"metric3d": _json_safe(rec.to_dict())}, (rec.source_id,) if rec.source_id else (),
            ))
            continue
        cfg = dict(execution_config or {})
        cfg.update({"contract": SCHEMA_VERSION, "checkpoint_sha256": rec.checkpoint_sha256, "scale_state": rec.scale_state.value})
        try:
            obs = make_observation(
                rec.to_draft(), engine_id=rec.engine_id, model_version=model_version, execution_config=cfg, processed_at=processed_at
            )
        except ObservationError as exc:
            rejected.append(str(exc))
            matrix.record_rejected(RejectedPrediction(
                run_id, "metric3d", rec.engine_id, model_version, i, str(exc), {"metric3d": _json_safe(rec.to_dict())}, (rec.source_id,),
            ))
            continue
        try:
            matrix.add(obs, EvidenceContext(
                obs.observation_id, run_id=run_id, architecture_id=rec.architecture_id, checkpoint_id=rec.checkpoint_id,
                camera_ref=rec.camera_id,
                artifact_refs={"raw_output": rec.raw_artifact_ref, **({"mask": rec.mask_ref} if rec.mask_ref else {})},
                validation="metric3d_contract_v1",
            ))
        except MatrixError as exc:  # e.g. the referenced raw artifact is not in the store: refused, never an "empty success"
            rejected.append(str(exc))
            matrix.record_rejected(RejectedPrediction(
                run_id, "metric3d", rec.engine_id, model_version, i, str(exc), {"metric3d": _json_safe(rec.to_dict())}, (rec.source_id,),
            ))
            continue
        accepted.append(obs.observation_id)
    return {"accepted": accepted, "rejected": rejected}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
