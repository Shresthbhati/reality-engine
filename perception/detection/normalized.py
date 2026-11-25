"""Single normalized detection contract and normalization boundary.

Every detector backend in Reality Engine (Mask R-CNN, OWL-ViT, Grounding DINO,
Florence-2, torchvision COCO, FakeDetector) emits its own model-specific type
(`Detection`, `ImageDetection`, dict, etc.).  This module is the SINGLE
normalization boundary between model-specific outputs and the rest of the
perception engine:

- Explicit coordinate convention (`PIXEL_XYXY` vs `NORMALIZED_XYXY`)
- Explicit image dimensions `(image_width, image_height)`
- Strict bounding-box validation (rejects negative, out-of-bounds, NaN/Inf,
  and degenerate boxes)
- Explicit confidence semantics (`None` stays `None` when a model emits no
  confidence, e.g. Florence-2 `<OD>`)
- Unknown categories remain supported without being forced into known classes
- Empty detections (`DetectionStatus.EMPTY`) are cleanly distinguished from
  model failures (`DetectionStatus.FAILED`)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import numpy as np

from perception.detection.policy import canonical as policy_canonical
from provenance import Uncertainty


class CoordinateConvention(str, Enum):
    PIXEL_XYXY = "pixel_xyxy"
    NORMALIZED_XYXY = "normalized_xyxy"


class DetectionStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    FAILED = "failed"


class InvalidDetectionError(ValueError):
    """Raised when a detection violates coordinate, confidence, or dimension contracts."""


def _validate_image_dims(image_width: int, image_height: int) -> Tuple[int, int]:
    w = int(image_width)
    h = int(image_height)
    if w <= 0 or h <= 0:
        raise InvalidDetectionError(f"image dimensions must be positive, got ({image_width}, {image_height})")
    return w, h


def convert_and_validate_box(
    box: Sequence[float],
    *,
    convention: CoordinateConvention,
    image_width: int,
    image_height: int,
    clip: bool = False,
    tolerance_px: float = 1e-3,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """Convert `box` to validated `(bbox_px, bbox_norm)` in XYXY order.

    Rejects NaN/Inf, negative coordinates, out-of-image coordinates (beyond
    `tolerance_px`), and degenerate boxes (`x_max <= x_min` or `y_max <= y_min`).
    """
    w, h = _validate_image_dims(image_width, image_height)
    if len(box) != 4:
        raise InvalidDetectionError(f"box must have 4 coordinates, got {len(box)}")
    vals = [float(v) for v in box]
    if any(not math.isfinite(v) for v in vals):
        raise InvalidDetectionError(f"box contains non-finite coordinate(s): {vals!r}")

    if convention is CoordinateConvention.NORMALIZED_XYXY:
        x0, y0, x1, y1 = vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h
    elif convention is CoordinateConvention.PIXEL_XYXY:
        x0, y0, x1, y1 = vals
    else:
        raise InvalidDetectionError(f"unsupported coordinate convention: {convention!r}")

    if clip:
        x0 = min(max(x0, 0.0), float(w))
        y0 = min(max(y0, 0.0), float(h))
        x1 = min(max(x1, 0.0), float(w))
        y1 = min(max(y1, 0.0), float(h))
    else:
        if x0 < -tolerance_px or y0 < -tolerance_px:
            raise InvalidDetectionError(f"negative box coordinates: {(x0, y0, x1, y1)!r}")
        if x1 > w + tolerance_px or y1 > h + tolerance_px:
            raise InvalidDetectionError(
                f"out-of-image box coordinates {(x0, y0, x1, y1)!r} for image ({w}, {h})"
            )
        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(w), x1)
        y1 = min(float(h), y1)

    if x1 <= x0 or y1 <= y0:
        raise InvalidDetectionError(f"degenerate box coordinates: {(x0, y0, x1, y1)!r}")

    bbox_px = (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))
    bbox_norm = (round(x0 / w, 6), round(y0 / h, 6), round(x1 / w, 6), round(y1 / h, 6))
    return bbox_px, bbox_norm


def _validate_mask(mask: Any, image_width: int, image_height: int) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or arr.shape != (image_height, image_width):
        raise InvalidDetectionError(
            f"mask shape {arr.shape} does not match image dimensions ({image_height}, {image_width})"
        )
    return arr


@dataclass(frozen=True)
class NormalizedDetection:
    """Canonical normalized detection contract used throughout perception."""

    detection_id: str
    label: str
    canonical_label: str
    bbox_px: Tuple[float, float, float, float]
    bbox_norm: Tuple[float, float, float, float]
    image_width: int
    image_height: int
    confidence: float | None
    model_name: str
    model_version: str = ""
    device: str = "cpu"
    inference_ms: float | None = None
    image_id: str = ""
    frame_index: int | None = None
    timestamp_s: float | None = None
    coordinate_convention: CoordinateConvention = CoordinateConvention.PIXEL_XYXY
    source_convention: CoordinateConvention = CoordinateConvention.PIXEL_XYXY
    mask: np.ndarray | None = None
    has_mask: bool = False
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    quality_flags: Tuple[str, ...] = ()
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_image_dims(self.image_width, self.image_height)
        if not self.label or not str(self.label).strip():
            raise InvalidDetectionError("label must be non-empty")
        if self.confidence is not None:
            c = float(self.confidence)
            if not math.isfinite(c) or not (0.0 <= c <= 1.0):
                raise InvalidDetectionError(f"confidence must be in [0, 1] or None, got {self.confidence!r}")
        if self.mask is not None:
            _validate_mask(self.mask, self.image_width, self.image_height)
            if not self.has_mask:
                object.__setattr__(self, "has_mask", True)
        elif self.has_mask:
            raise InvalidDetectionError("has_mask=True requires a non-None mask")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "label": self.label,
            "canonical_label": self.canonical_label,
            "bbox_px": list(self.bbox_px),
            "bbox_norm": list(self.bbox_norm),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device": self.device,
            "inference_ms": self.inference_ms,
            "image_id": self.image_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "coordinate_convention": self.coordinate_convention.value,
            "source_convention": self.source_convention.value,
            "has_mask": self.has_mask,
            "quality_flags": list(self.quality_flags),
        }


@dataclass(frozen=True)
class NormalizedDetectionBatch:
    """Explicit result envelope separating OK, genuine EMPTY, and FAILED inference."""

    image_id: str
    image_width: int
    image_height: int
    status: DetectionStatus
    detections: Tuple[NormalizedDetection, ...] = ()
    model_name: str = ""
    model_version: str = ""
    device: str = "cpu"
    inference_ms: float | None = None
    rejected_count: int = 0
    rejection_reasons: Tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_image_dims(self.image_width, self.image_height)
        if self.status is DetectionStatus.FAILED:
            if not self.error:
                raise InvalidDetectionError("FAILED batch must include a non-empty error description")
            if self.detections:
                raise InvalidDetectionError("FAILED batch cannot carry detections")
        else:
            if self.error is not None:
                raise InvalidDetectionError(f"{self.status.value} batch cannot carry an error")
            if self.status is DetectionStatus.EMPTY and self.detections:
                raise InvalidDetectionError("EMPTY batch cannot carry detections")
            if self.status is DetectionStatus.OK and not self.detections:
                raise InvalidDetectionError("OK batch must carry at least one detection")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "status": self.status.value,
            "detections": [d.to_dict() for d in self.detections],
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device": self.device,
            "inference_ms": self.inference_ms,
            "rejected_count": self.rejected_count,
            "rejection_reasons": list(self.rejection_reasons),
            "error": self.error,
        }


def _resolve_canonical_label(raw_label: str) -> str:
    cleaned = str(raw_label).strip().lower()
    if not cleaned:
        return "unknown"
    mapped = policy_canonical(cleaned)
    return mapped if mapped else cleaned


def _extract_raw_fields(raw: Any) -> Tuple[str, Sequence[float], float | None, Any, Dict[str, Any]]:
    """Extract (label, box, confidence, mask, extra_meta) from any backend item."""
    if isinstance(raw, NormalizedDetection):
        return (
            raw.label,
            raw.bbox_px,
            raw.confidence,
            raw.mask,
            dict(raw.raw_metadata),
        )
    if isinstance(raw, Mapping):
        label = raw.get("label") or raw.get("category") or ""
        box = raw.get("bbox_px") if "bbox_px" in raw else raw.get("box_xyxy", raw.get("box", raw.get("bbox")))
        if box is None:
            raise InvalidDetectionError(f"mapping detection missing box field: {list(raw.keys())}")
        conf = raw.get("confidence", raw.get("score"))
        mask = raw.get("mask")
        extra = {k: v for k, v in raw.items() if k not in {"label", "category", "bbox_px", "box_xyxy", "box", "bbox", "confidence", "score", "mask"}}
        return str(label), box, (float(conf) if conf is not None else None), mask, extra

    # Object contracts: Detection (.box, .score), ImageDetection (.box_xyxy, .score)
    label = getattr(raw, "label", None)
    if label is None:
        raise InvalidDetectionError(f"unsupported detection object without label: {type(raw).__name__}")
    if hasattr(raw, "box_xyxy"):
        box = getattr(raw, "box_xyxy")
    elif hasattr(raw, "box"):
        box = getattr(raw, "box")
    elif hasattr(raw, "bbox_px"):
        box = getattr(raw, "bbox_px")
    else:
        raise InvalidDetectionError(f"unsupported detection object without box/box_xyxy: {type(raw).__name__}")

    conf = getattr(raw, "score", getattr(raw, "confidence", None))
    mask = getattr(raw, "mask", None)
    extra: Dict[str, Any] = {}
    for attr in ("detection_id", "evidence_id", "view_id", "source", "model_name", "model_version"):
        if hasattr(raw, attr):
            val = getattr(raw, attr)
            if val not in (None, ""):
                extra[attr] = val
    return str(label), box, (float(conf) if conf is not None else None), mask, extra


def normalize_detection(
    raw: Any,
    *,
    image_width: int,
    image_height: int,
    convention: CoordinateConvention = CoordinateConvention.PIXEL_XYXY,
    model_name: str = "",
    model_version: str = "",
    device: str = "cpu",
    inference_ms: float | None = None,
    image_id: str = "",
    frame_index: int | None = None,
    timestamp_s: float | None = None,
    clip: bool = False,
) -> NormalizedDetection:
    """Single normalization boundary from any backend item to `NormalizedDetection`."""
    if isinstance(raw, NormalizedDetection):
        return raw

    label, raw_box, conf, mask_raw, extra = _extract_raw_fields(raw)
    bbox_px, bbox_norm = convert_and_validate_box(
        raw_box,
        convention=convention,
        image_width=image_width,
        image_height=image_height,
        clip=clip,
    )
    mask = _validate_mask(mask_raw, int(image_width), int(image_height))
    resolved_model = str(extra.get("model_name") or model_name or extra.get("source") or "unknown_detector")
    resolved_ver = str(extra.get("model_version") or model_version or "")
    det_id = str(extra.get("detection_id") or f"det_{uuid4().hex[:12]}")
    resolved_img_id = str(image_id or extra.get("evidence_id") or extra.get("view_id") or "")

    flags: List[str] = []
    if conf is None:
        flags.append("missing_confidence")
    if mask is None:
        flags.append("no_mask")
    canon = _resolve_canonical_label(label)
    if canon == "unknown":
        flags.append("unknown_label")

    uncertainty = Uncertainty(
        confidence=round(float(conf), 4) if conf is not None else 0.0,
        note=("confidence_not_provided_by_model" if conf is None else ("unknown_label" if canon == "unknown" else None)),
        calibrated=False,
    )
    return NormalizedDetection(
        detection_id=det_id,
        label=label.strip(),
        canonical_label=canon,
        bbox_px=bbox_px,
        bbox_norm=bbox_norm,
        image_width=int(image_width),
        image_height=int(image_height),
        confidence=conf,
        model_name=resolved_model,
        model_version=resolved_ver,
        device=device,
        inference_ms=inference_ms,
        image_id=resolved_img_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        coordinate_convention=CoordinateConvention.PIXEL_XYXY,
        source_convention=convention,
        mask=mask,
        has_mask=mask is not None,
        uncertainty=uncertainty,
        quality_flags=tuple(flags),
        raw_metadata=extra,
    )


def normalize_backend_output(
    raw_items: Iterable[Any] | None,
    *,
    image_width: int,
    image_height: int,
    convention: CoordinateConvention = CoordinateConvention.PIXEL_XYXY,
    model_name: str = "",
    model_version: str = "",
    device: str = "cpu",
    inference_ms: float | None = None,
    image_id: str = "",
    frame_index: int | None = None,
    timestamp_s: float | None = None,
    error: str | None = None,
    clip: bool = False,
) -> NormalizedDetectionBatch:
    """Normalize a backend's per-image output into a `NormalizedDetectionBatch`.

    - `error is not None` -> `DetectionStatus.FAILED` (never disguised as empty).
    - `raw_items` yields 0 valid items without error -> `DetectionStatus.EMPTY`.
    - Invalid individual boxes are rejected and counted in `rejected_count` + `rejection_reasons`.
    """
    w, h = _validate_image_dims(image_width, image_height)
    if error is not None:
        return NormalizedDetectionBatch(
            image_id=image_id,
            image_width=w,
            image_height=h,
            status=DetectionStatus.FAILED,
            detections=(),
            model_name=model_name,
            model_version=model_version,
            device=device,
            inference_ms=inference_ms,
            error=str(error),
        )

    normalized: List[NormalizedDetection] = []
    reasons: List[str] = []
    for item in (raw_items or ()):
        try:
            normalized.append(
                normalize_detection(
                    item,
                    image_width=w,
                    image_height=h,
                    convention=convention,
                    model_name=model_name,
                    model_version=model_version,
                    device=device,
                    inference_ms=inference_ms,
                    image_id=image_id,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    clip=clip,
                )
            )
        except InvalidDetectionError as exc:
            reasons.append(str(exc))

    status = DetectionStatus.OK if normalized else DetectionStatus.EMPTY
    return NormalizedDetectionBatch(
        image_id=image_id,
        image_width=w,
        image_height=h,
        status=status,
        detections=tuple(normalized),
        model_name=model_name,
        model_version=model_version,
        device=device,
        inference_ms=inference_ms,
        rejected_count=len(reasons),
        rejection_reasons=tuple(reasons),
        error=None,
    )


def _box_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def suppress_duplicate_normalized_detections(
    detections: Sequence[NormalizedDetection],
    *,
    iou_threshold: float = 0.65,
) -> Tuple[Tuple[NormalizedDetection, ...], int]:
    """Same-model, same-canonical-label NMS keeping highest confidence.

    Cross-model and cross-label detections are NEVER suppressed so model
    disagreements and competing category hypotheses are preserved.
    """
    ordered = sorted(
        detections,
        key=lambda d: (-(d.confidence if d.confidence is not None else 0.5), d.detection_id),
    )
    kept: List[NormalizedDetection] = []
    suppressed = 0
    for cand in ordered:
        is_dup = any(
            k.model_name == cand.model_name
            and k.canonical_label == cand.canonical_label
            and _box_iou_xyxy(k.bbox_px, cand.bbox_px) >= iou_threshold
            for k in kept
        )
        if is_dup:
            suppressed += 1
        else:
            kept.append(cand)
    return tuple(kept), suppressed
