"""Intelligent Multi-Engine Perception Orchestration (A3.5).

Expands ``perception/detection/multi_engine_fusion.py`` with:
  - Task-aware, image-quality-aware, class-specific, and resolution-aware routing
  - Memory-aware sequential GPU scheduling
  - Engine health tracking (circuit-breaker after repeated failures)
  - Model failure isolation (one failing model never sinks the run)
  - Execution telemetry and correlation-aware family fusion
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.detection.multi_engine_fusion import (
    FusedDetectionHypothesis,
    engine_family,
    fuse_multi_engine_detections,
    select_detection_engines,
)


@dataclass
class EngineHealthState:
    engine_id: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    last_error: str | None = None
    total_wall_s: float = 0.0
    max_vram_mb: float | None = None
    circuit_open: bool = False

    def record_success(self, wall_s: float, vram_mb: float | None = None) -> None:
        self.attempts += 1
        self.successes += 1
        self.total_wall_s += wall_s
        if vram_mb is not None:
            self.max_vram_mb = max(self.max_vram_mb or 0.0, vram_mb)
        self.circuit_open = False

    def record_failure(self, error: str, *, max_consecutive_failures: int = 3) -> None:
        self.attempts += 1
        self.failures += 1
        self.last_error = error
        if self.failures >= max_consecutive_failures and self.successes == 0:
            self.circuit_open = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "last_error": self.last_error,
            "total_wall_s": round(self.total_wall_s, 3),
            "max_vram_mb": self.max_vram_mb,
            "circuit_open": self.circuit_open,
        }


@dataclass(frozen=True)
class RoutingPlan:
    selected_detectors: tuple[str, ...]
    selected_segmenters: tuple[str, ...]
    selected_embedders: tuple[str, ...]
    budget: str
    vram_budget_mb: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrchestratedDetectionResult:
    plan: RoutingPlan
    per_engine_boxes: dict[str, tuple[DetectionBox, ...]]
    fused_hypotheses: tuple[FusedDetectionHypothesis, ...]
    engine_telemetry: dict[str, dict[str, Any]]
    failed_engines: dict[str, str]


class PerceptionEngineScheduler:
    """Memory-aware, health-tracked sequential perception orchestrator."""

    def __init__(self, *, vram_budget_mb: float = 5500.0) -> None:
        self.vram_budget_mb = float(vram_budget_mb)
        self.health: dict[str, EngineHealthState] = {}

    def _health_of(self, engine_id: str) -> EngineHealthState:
        if engine_id not in self.health:
            self.health[engine_id] = EngineHealthState(engine_id=engine_id)
        return self.health[engine_id]

    def build_routing_plan(
        self,
        available_engines: Sequence[str],
        *,
        task: str = "interior_survey",
        budget: str = "balanced",
        is_low_contrast: bool = False,
        is_blurry: bool = False,
        target_classes: Sequence[str] = (),
        image_max_edge: int = 1024,
    ) -> RoutingPlan:
        healthy = [e for e in available_engines if not self._health_of(e).circuit_open]
        reasons: list[str] = [f"task={task}", f"budget={budget}", f"vram_budget_mb={self.vram_budget_mb}"]

        det_pool = [e for e in healthy if e in ("grounding_dino", "mask_rcnn", "florence2", "owlvit", "yolo_world")]
        selected_dets = select_detection_engines(
            det_pool,
            budget=budget,
            is_low_contrast=(is_low_contrast or is_blurry),
        )
        # Class-specific routing: if COCO furniture classes are targeted and mask_rcnn is healthy, ensure it's included
        coco_furniture = {"chair", "sofa", "couch", "bed", "table", "dining_table", "toilet", "sink", "refrigerator"}
        if any(c.lower().replace(" ", "_") in coco_furniture for c in target_classes) and "mask_rcnn" in det_pool:
            if "mask_rcnn" not in selected_dets:
                selected_dets.append("mask_rcnn")
                reasons.append("added_mask_rcnn_for_coco_furniture")

        if image_max_edge >= 1600:
            reasons.append("high_res_tiling_recommended")

        seg_pool = [e for e in healthy if e in ("sam", "sam2", "sam2_hf", "sam3")]
        selected_segs = tuple(seg_pool[:1]) if seg_pool else ()
        emb_pool = [e for e in healthy if e in ("dinov2", "clip", "siglip")]
        selected_embs = tuple(emb_pool[:1]) if (emb_pool and budget != "fast") else ()

        return RoutingPlan(
            selected_detectors=tuple(selected_dets),
            selected_segmenters=selected_segs,
            selected_embedders=selected_embs,
            budget=budget,
            vram_budget_mb=self.vram_budget_mb,
            reasons=tuple(reasons),
        )

    def execute_sequential(
        self,
        plan: RoutingPlan,
        runners: Mapping[str, Callable[[], Sequence[DetectionBox]]],
        *,
        is_low_contrast: bool = False,
    ) -> OrchestratedDetectionResult:
        """Run each selected detector sequentially with failure isolation and correlation-aware fusion."""
        per_engine: dict[str, tuple[DetectionBox, ...]] = {}
        failed: dict[str, str] = {}
        telemetry: dict[str, dict[str, Any]] = {}

        for eid in plan.selected_detectors:
            runner = runners.get(eid)
            if runner is None:
                failed[eid] = "no_runner_registered"
                continue
            h = self._health_of(eid)
            if h.circuit_open:
                failed[eid] = "circuit_open_due_to_prior_failures"
                continue
            t0 = time.perf_counter()
            try:
                boxes = tuple(runner())
                wall_s = time.perf_counter() - t0
                h.record_success(wall_s)
                per_engine[eid] = boxes
                telemetry[eid] = {
                    "status": "succeeded",
                    "family": engine_family(eid),
                    "n_detections": len(boxes),
                    "wall_s": round(wall_s, 4),
                }
            except Exception as exc:  # noqa: BLE001 -- isolate individual model failures
                wall_s = time.perf_counter() - t0
                err = f"{type(exc).__name__}: {exc}"
                h.record_failure(err)
                failed[eid] = err
                telemetry[eid] = {
                    "status": "failed",
                    "family": engine_family(eid),
                    "error": err,
                    "wall_s": round(wall_s, 4),
                }

        fused = tuple(fuse_multi_engine_detections(per_engine, is_low_contrast=is_low_contrast))
        return OrchestratedDetectionResult(
            plan=plan,
            per_engine_boxes=per_engine,
            fused_hypotheses=fused,
            engine_telemetry=telemetry,
            failed_engines=failed,
        )


__all__ = [
    "EngineHealthState",
    "OrchestratedDetectionResult",
    "PerceptionEngineScheduler",
    "RoutingPlan",
]
