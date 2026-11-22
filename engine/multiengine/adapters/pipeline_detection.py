"""Production detection engine: ``DetectionPipeline`` driven by REAL registered detector / segmenter adapters.

The pipeline (domain prompts, overlapping tiled inference, host-crop sub-part detection, class-aware confidence filtering,
duplicate suppression that keeps parts, mask refinement) lived beside the production job and was never called by it. This module
makes it one engine (``detection_pipeline``, Task.DETECTION) of the execution manager, so the job gets the manager's lifecycle for
free: availability probe, load -> infer -> validate -> persist -> unload, content-addressed run output, run records.

Honesty rules enforced here:
- ``RuleBasedDetectorAdapter`` / ``ThresholdSegmenterAdapter`` (deterministic reference stand-ins) are refused in production
  (``ProductionAdapterError``); a deterministic or virtual (test-double) prediction from an inner adapter is refused too.
- An unavailable detector makes the engine UNAVAILABLE (explicit, never replaced). An unavailable or failing segmenter is recorded
  (``segmentation`` facts, per-image summary) and the boxes are still emitted WITHOUT masks: nothing is invented.
- Every raw candidate the pipeline produced is preserved in a per-image ``detection_pipeline_summary`` observation with its
  disposition (kept / below the class confidence threshold / suppressed as a duplicate); refined boxes become ``bbox``
  observations carrying scale tier, parent/host, inner engine and a lossless RLE mask.
"""

from __future__ import annotations

import itertools
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from engine.multiengine.contract import (
    Availability,
    AvailabilityStatus,
    CheckpointRequirement,
    Device,
    EngineAdapter,
    EngineError,
    EngineErrorKind,
    EngineRequest,
    EngineSpec,
    Task,
)
from engine.multiengine.mask_codec import count_holes, encode_mask
from engine.multiengine.observation import ObservationDraft

ENGINE_ID = "detection_pipeline"
ENV_TILE_SIZE = "RE_DETECTION_TILE_SIZE"
DEFAULT_TILE_SIZE = 800
DEFAULT_THRESHOLD = 0.10  # inner detector score floor (untuned starting value); the pipeline then applies its own class thresholds
PROMPT_CHUNK = 32  # text queries per forward pass of an open-vocabulary detector


class ProductionAdapterError(RuntimeError):
    """A deterministic reference adapter was supplied where a real model is required."""


def assert_production_adapter(adapter: Any, role: str) -> None:
    from engine.multiengine.spatial_intelligence.detection_pipeline import RuleBasedDetectorAdapter, ThresholdSegmenterAdapter

    if isinstance(adapter, (RuleBasedDetectorAdapter, ThresholdSegmenterAdapter)):
        raise ProductionAdapterError(f"{type(adapter).__name__} is a deterministic reference stand-in, not a model; refused as the production {role}")


class AdapterDetector:
    """``DetectorAdapter`` over a registered detector ``EngineAdapter``: images are handed to the real model; nothing is synthesised."""

    def __init__(self, inner: EngineAdapter, *, threshold: float = DEFAULT_THRESHOLD) -> None:
        assert_production_adapter(inner, "detector")
        self.inner, self.threshold = inner, float(threshold)
        self.calls = 0
        self.queries = 0
        self._dir = tempfile.TemporaryDirectory(prefix="re-det-")
        self._ids = itertools.count()

    def close(self) -> None:
        self._dir.cleanup()

    def detect(self, image_rgb: np.ndarray, text_prompts: Sequence[str]) -> list[Any]:
        from PIL import Image

        from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox

        path = Path(self._dir.name) / "input.png"
        Image.fromarray(np.ascontiguousarray(image_rgb)).save(path)
        prompts = [str(p) for p in text_prompts]
        out: list[Any] = []
        for i in range(0, len(prompts), PROMPT_CHUNK):
            chunk = prompts[i:i + PROMPT_CHUNK]
            self.calls += 1
            self.queries += len(chunk)
            for d in self.inner.infer(EngineRequest(sources=(str(path),), payload={"queries": chunk, "threshold": self.threshold}, seed=42)):
                if d.quantity != "bbox" or d.image_region is None:
                    continue
                if d.predicted.get("is_test_double"):
                    raise EngineError(EngineErrorKind.INVALID_INPUT, f"{self.inner.spec.engine_id} returned a test-double prediction; refused in production")
                label = d.predicted.get("label")
                if not isinstance(label, str) or not label.strip():
                    continue
                emitted = d.confidence if d.confidence is not None else d.predicted.get("score")
                out.append(DetectionBox(
                    box_id=f"{self.inner.spec.engine_id}:{next(self._ids)}", class_name=label, box_2d=tuple(float(v) for v in d.image_region),
                    # an engine that emits no score (e.g. Florence-2) enters the pipeline at a neutral 0.5 and is flagged: never a measured score
                    confidence=float(emitted) if emitted is not None else 0.5, source_engine=self.inner.spec.engine_id,
                    attributes={"confidence_emitted": emitted is not None},
                ))
        return out


class AdapterSegmenter:
    """``SegmenterAdapter`` over a loaded SAM-style adapter (its ``_predictor``): the image is embedded once, then each box is prompted."""

    def __init__(self, inner: EngineAdapter) -> None:
        assert_production_adapter(inner, "segmenter")
        self.inner = inner
        self.calls = 0
        self.failures = 0
        self._key: tuple | None = None

    def segment_box(self, image_rgb: np.ndarray, box_2d: tuple[float, float, float, float]) -> np.ndarray | None:
        pred = getattr(self.inner, "_predictor", None)
        if pred is None:
            raise EngineError(EngineErrorKind.RUNTIME_ERROR, f"segmenter {self.inner.spec.engine_id} is not loaded")
        self.calls += 1
        try:
            key = (id(image_rgb), image_rgb.shape)
            if key != self._key:
                pred.set_image(image_rgb)
                self._key = key
            masks, _scores, _ = pred.predict(box=np.asarray(box_2d, dtype=float), multimask_output=False)
            return np.asarray(masks[0], dtype=bool)
        except Exception:  # noqa: BLE001 -- one box failing must not lose the others; counted and reported, the box simply has no mask
            self.failures += 1
            return None


class _NoSegmenter:
    """Explicit 'no segmentation model': yields no mask (the pipeline must never fall back to its threshold stand-in)."""

    def segment_box(self, image_rgb: np.ndarray, box_2d: tuple[float, float, float, float]) -> None:
        return None


def _pin(name: str, path: Path | None) -> CheckpointRequirement:
    from engine.multiengine.metric3d_producers import checkpoint_sha256

    return CheckpointRequirement(name=name, path=str(path) if path else None, sha256=checkpoint_sha256(path))


def _checkpoints(det: EngineAdapter, seg: EngineAdapter | None) -> tuple[CheckpointRequirement, ...]:
    from engine.multiengine.metric3d_producers import hf_checkpoint_file

    out: list[CheckpointRequirement] = []
    ck = getattr(det, "checkpoint", None)
    if isinstance(ck, str):
        out.append(_pin(ck, hf_checkpoint_file(ck) if "/" in ck else (Path(ck) if Path(ck).is_file() else None)))
    if seg is not None:
        ck = getattr(seg, "checkpoint", None)
        cached = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "sam_vit_b_01ec64.pth"
        p = Path(ck) if isinstance(ck, str) and Path(ck).is_file() else (cached if cached.is_file() else None)
        out.append(_pin(str(ck or seg.spec.engine_id), p))
    return tuple(out)


class PipelineDetectionEngineAdapter(EngineAdapter):
    def __init__(self, detector: EngineAdapter, segmenter: EngineAdapter | None = None, *, tile_size: int | None = None,
                 tile_overlap: float = 0.25, threshold: float = DEFAULT_THRESHOLD, domain: Any = None, pin_checkpoints: bool = True) -> None:
        assert_production_adapter(detector, "detector")
        if segmenter is not None:
            assert_production_adapter(segmenter, "segmenter")
        from engine.multiengine.spatial_intelligence.taxonomy import Domain

        self.detector, self.segmenter = detector, segmenter
        self.tile_size = int(tile_size or os.environ.get(ENV_TILE_SIZE) or DEFAULT_TILE_SIZE)
        self.tile_overlap, self.threshold = float(tile_overlap), float(threshold)
        self.domain = domain or Domain.INTERIOR
        self._seg_ok = False
        self._seg_reason = "no segmentation engine was supplied"
        self.last_facts: dict[str, Any] = {}
        d = detector.spec
        self.spec = EngineSpec(
            engine_id=ENGINE_ID, name="Tiled open-vocabulary detection + sub-part crops + instance masks",
            version=f"{d.engine_id}-{d.version}" + (f"+{segmenter.spec.engine_id}-{segmenter.spec.version}" if segmenter is not None else ""),
            tasks=frozenset({Task.DETECTION}), input_schema="RGB photograph", output_schema="bbox (+RLE mask), detection_pipeline_summary",
            probe=self._probe, family=d.family or d.engine_id,  # shares the inner detector's lineage: NOT an independent second detector
            dependencies=d.dependencies, checkpoints=_checkpoints(detector, segmenter) if pin_checkpoints else (), hardware=d.hardware,
            licence=d.licence, commercial_ok=d.commercial_ok,
            config={"tile_size": self.tile_size, "tile_overlap": self.tile_overlap, "inner_score_floor": self.threshold, "domain": str(getattr(self.domain, "value", self.domain)),
                    "detector": d.engine_id, "segmenter": segmenter.spec.engine_id if segmenter is not None else None},
            limitations=("thresholds (inner score floor, class confidence, tiling) are untuned starting values",
                         "images no larger than the tile size are not tiled; sub-part crops run on every host with a taxonomy sub-part",
                         "a detector that emits no score enters the pipeline at a neutral 0.5 and is flagged confidence_emitted=false"),
            priority=5, integration="wired")

    # ---------------------------------------------------------------- availability / lifecycle
    def _probe(self) -> Availability:
        a = self.detector.check_available()
        if not a.ok:
            return Availability(AvailabilityStatus.BLOCKED, f"inner detector {self.detector.spec.engine_id} unavailable: {a.reason}", a.remediation)
        return Availability(AvailabilityStatus.AVAILABLE, f"detector {self.detector.spec.engine_id} available",
                            evidence=tuple(a.evidence) or (f"{self.detector.spec.engine_id} probe: {a.reason}",))

    def validate_input(self, request: EngineRequest) -> None:
        if not request.sources:
            raise EngineError(EngineErrorKind.INVALID_INPUT, "the detection pipeline needs at least one photograph")
        bad = [s for s in request.sources if str(s).startswith("virtual:") or not os.path.isfile(str(s))]
        if bad:
            raise EngineError(EngineErrorKind.INVALID_INPUT, f"not real image files (virtual or missing): {bad[:3]}")

    def load(self, device: Device) -> None:
        self.detector.load(device)
        self._seg_ok, self._seg_reason = False, "no segmentation engine was supplied"
        if self.segmenter is not None:
            a = self.segmenter.check_available()
            if not a.ok:
                self._seg_reason = f"{self.segmenter.spec.engine_id} unavailable: {a.reason}"
            else:
                try:
                    self.segmenter.load(device)
                    self._seg_ok, self._seg_reason = True, "loaded"
                except EngineError as exc:  # explicit and recorded: boxes are still produced, without masks
                    self._seg_reason = f"{self.segmenter.spec.engine_id} failed to load: {exc}"

    def unload(self) -> None:
        for a in (self.segmenter, self.detector):
            if a is not None:
                try:
                    a.unload()
                except Exception:  # noqa: BLE001 -- unloading the other model must still happen
                    pass

    def measured_usage(self) -> tuple[float | None, float | None]:
        return self.detector.measured_usage()

    def reproducibility(self, request: EngineRequest) -> dict[str, Any]:
        return {**super().reproducibility(request), "tile_size": self.tile_size, "inner_score_floor": self.threshold,
                "detector": self.detector.spec.engine_id, "segmenter": self.segmenter.spec.engine_id if self.segmenter is not None else None}

    # ---------------------------------------------------------------- inference
    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]:
        from PIL import Image

        from engine.multiengine.spatial_intelligence.detection_pipeline import (
            DetectionPipeline,
            DetectionScale,
            filter_by_class_confidence,
            generate_domain_prompts,
        )

        prompts = generate_domain_prompts(self.domain)
        n_std, n_micro = len(prompts.get(DetectionScale.STANDARD, [])), len(prompts.get(DetectionScale.MICRO, []))
        det = AdapterDetector(self.detector, threshold=self.threshold)
        seg: Any = AdapterSegmenter(self.segmenter) if (self._seg_ok and self.segmenter is not None) else _NoSegmenter()
        pipe = DetectionPipeline(detector=det, segmenter=seg, tile_size=self.tile_size, tile_overlap=self.tile_overlap)
        drafts: list[ObservationDraft] = []
        images: list[dict[str, Any]] = []
        try:
            for src in request.sources:
                t0 = time.monotonic()
                with Image.open(src) as im:
                    rgb = np.asarray(im.convert("RGB"))
                h, w = rgb.shape[:2]
                res = pipe.run_pipeline(rgb, domain=self.domain, image_id=str(src), run_tiling=True, detect_subparts=True)
                coord = f"image:{Path(str(src)).stem}"
                kept = {b.box_id for b in res.refined_boxes}
                passed = {b.box_id for b in filter_by_class_confidence(list(res.raw_detections))}
                raw_rows = [{**b.to_dict(), "disposition": "kept" if b.box_id in kept else ("suppressed_as_duplicate" if b.box_id in passed else "below_class_confidence"),
                             "confidence_emitted": bool(b.attributes.get("confidence_emitted", True))} for b in res.raw_detections]
                masks = 0
                for b in res.refined_boxes:
                    mask_info = None
                    if b.mask_2d is not None and np.asarray(b.mask_2d).shape[:2] == (h, w) and np.asarray(b.mask_2d).any():
                        mask_info = {**encode_mask(b.mask_2d), "holes": count_holes(b.mask_2d), "model": self.segmenter.spec.engine_id if self.segmenter else None}
                        masks += 1
                    emitted = bool(b.attributes.get("confidence_emitted", True))
                    drafts.append(ObservationDraft(
                        task="detection", quantity="bbox", source_id=str(src), coordinate_system=coord,
                        image_region=tuple(float(v) for v in b.box_2d),
                        predicted={"label": b.class_name, "score": float(b.confidence) if emitted else None, "scale_tier": b.scale.value,
                                   "pipeline_box_id": b.box_id, "parent_pipeline_box_id": b.parent_box_id, "host_class": b.attributes.get("host_class"),
                                   "inner_engine": b.source_engine, "confidence_emitted": emitted, "mask": mask_info,
                                   # DetectionBox fields persisted AS GIVEN (same meaning as the perception contract), never reinterpreted
                                   "is_truncated": bool(getattr(b, "is_truncated", False)), "occlusion_level": getattr(b, "occlusion_level", "none"),
                                   "mask_area_px": int(getattr(b, "mask_area_px", 0) or 0), "child_box_ids": list(getattr(b, "child_box_ids", ()) or ()),
                                   "box_normalized": [b.x1 / w, b.y1 / h, b.x2 / w, b.y2 / h]},
                        original_coordinates={"box_xyxy_px": [float(v) for v in b.box_2d], "image_size_px": [w, h]},
                        confidence=float(b.confidence) if emitted else None))
                stats = {**res.execution_stats, "source": str(src), "seconds": round(time.monotonic() - t0, 2), "masks": masks,
                         "tile_size": self.tile_size, "tiled": res.execution_stats.get("tiles_processed", 0) > 0}
                images.append(stats)
                drafts.append(ObservationDraft(
                    task="detection", quantity="detection_pipeline_summary", source_id=str(src), coordinate_system=coord,
                    predicted={"execution_stats": stats, "parent_part_tree": res.parent_part_tree, "raw_detections": raw_rows,
                               "prompts": {"standard": n_std, "micro": n_micro}, "inner_detector": self.detector.spec.engine_id,
                               "segmentation": {"status": "ran" if self._seg_ok else "unavailable", "reason": self._seg_reason,
                                                "engine": self.segmenter.spec.engine_id if self.segmenter else None,
                                                "boxes": len(res.refined_boxes), "masks": masks}}))
        finally:
            det.close()
        n_boxes = sum(i["refined_box_count"] for i in images)
        n_masks = sum(i["masks"] for i in images)
        self.last_facts = {
            "engine": ENGINE_ID, "images": images, "tile_size": self.tile_size, "detector": self.detector.spec.engine_id,
            "detector_forward_passes": det.calls, "detector_queries": det.queries, "prompts": {"standard": n_std, "micro": n_micro},
            "tiles_processed": sum(i["tiles_processed"] for i in images), "subparts_detected": sum(i["subparts_detected"] for i in images),
            "raw_detections": sum(i["raw_detection_count"] for i in images), "refined_boxes": n_boxes,
            "segmentation": {"requested": self.segmenter is not None, "status": "ran" if self._seg_ok else "unavailable", "reason": self._seg_reason,
                             "engine": self.segmenter.spec.engine_id if self.segmenter else None, "masks": n_masks,
                             "coverage": round(n_masks / n_boxes, 4) if n_boxes else None,
                             "sam_calls": getattr(seg, "calls", 0), "sam_failures": getattr(seg, "failures", 0)},
        }
        return drafts


def install(registry: Any, adapters: dict[str, Any], *, detector_id: str = "owlvit", segmenter_id: str = "sam", **kwargs: Any) -> str:
    """Add the ``detection_pipeline`` engine over the registry's REAL detector (and segmenter when present) and return its id.

    Raises ``ProductionAdapterError``/``KeyError`` rather than substituting anything: no detector adapter = no pipeline."""
    if detector_id not in adapters:
        raise KeyError(f"detector {detector_id!r} is not registered in this deployment; the detection pipeline will not run on a stand-in")
    eng = PipelineDetectionEngineAdapter(adapters[detector_id], adapters.get(segmenter_id), **kwargs)
    registry.register(eng.spec)
    adapters[ENGINE_ID] = eng
    return ENGINE_ID
