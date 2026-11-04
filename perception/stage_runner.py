"""One-model-at-a-time perception stage runner and scheduler (Blueprint Task 4.2).

Per docs/blueprint/15_ML_MODEL_STRATEGY.md:
- Target: RTX 4050 6GB Laptop GPU (hard ceiling 5,200 MB VRAM, 944 MB OS reserve).
- Constraint: One heavy model resident in VRAM at a time.
- Pipeline order (S7): Segmentation/Detection -> Monocular/Metric Depth -> Appearance/Embeddings.
- Loads model -> processes all evidence items -> explicitly unloads and releases VRAM before next model.
- Dynamic fallback to CPU if VRAM ceiling is exceeded or CUDA is unavailable.
- Collects peak VRAM and runtime per task for benchmark telemetry.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from engine.core.device import DeviceInfo, release_vram, select_device
from perception.model_registry import get as get_model_spec

logger = logging.getLogger(__name__)

# RTX 4050 6GB VRAM Ceiling
RTX4050_MAX_USABLE_VRAM_MB: float = 5200.0


@dataclass
class TaskExecutionRecord:
    task: str
    model_name: str
    device: str
    fp16: bool
    items_processed: int
    runtime_ms: float
    peak_vram_mb: float
    cpu_fallback: bool
    status: str = "completed"
    error: Optional[str] = None


@dataclass
class PerceptionStageReport:
    total_runtime_ms: float
    overall_peak_vram_mb: float
    tasks: List[TaskExecutionRecord] = field(default_factory=list)
    vram_budget_mb: float = RTX4050_MAX_USABLE_VRAM_MB

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ModelResidentScheduler:
    """Enforces that at most one deep-learning model occupies VRAM at any given time.
    
    If another model is requested, any existing resident model is evicted and VRAM
    is collected before the new model is loaded.
    """

    def __init__(self, vram_budget_mb: float = RTX4050_MAX_USABLE_VRAM_MB):
        self.vram_budget_mb = vram_budget_mb
        self._current_model_id: Optional[str] = None
        self._current_model_ref: Any = None
        self._peak_vram_observed_mb: float = 0.0

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model_id

    def evict_current(self) -> None:
        """Evicts any currently resident model and purges GPU allocator caches."""
        if self._current_model_id is not None:
            logger.info("Evicting resident model '%s' from VRAM", self._current_model_id)
            self._current_model_ref = None
            self._current_model_id = None
            release_vram()

    def get_gpu_memory_used_mb(self) -> float:
        """Measure current allocated GPU memory in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return float(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            pass
        return 0.0

    @contextmanager
    def model_scope(
        self,
        model_id: str,
        required_vram_mb: Optional[float] = None,
        force_cpu: bool = False,
    ) -> Iterator[DeviceInfo]:
        """Context manager guaranteeing exclusive VRAM occupancy for a model execution scope.
        
        Yields DeviceInfo specifying whether to run CUDA FP16 or CPU.
        """
        # Evict any other resident model first
        if self._current_model_id is not None and self._current_model_id != model_id:
            self.evict_current()

        spec = get_model_spec(model_id)
        est_vram = required_vram_mb or (spec.measured_vram_mb if spec else 1000.0)

        # Select execution device
        if force_cpu:
            dev_info = DeviceInfo("cpu", False, "cpu", None, None, "forced_cpu")
        else:
            dev_info = select_device(min_free_vram_mb=int(est_vram))
            # Check if estimated VRAM exceeds our RTX 4050 ceiling
            if dev_info.is_gpu and est_vram > self.vram_budget_mb:
                dev_info = DeviceInfo(
                    "cpu",
                    False,
                    dev_info.name,
                    dev_info.free_vram_mb,
                    dev_info.total_vram_mb,
                    f"Required VRAM {est_vram} MB exceeds budget {self.vram_budget_mb} MB",
                )

        self._current_model_id = model_id
        start_vram = self.get_gpu_memory_used_mb()

        try:
            yield dev_info
        finally:
            end_vram = self.get_gpu_memory_used_mb()
            peak = max(start_vram, end_vram)
            if peak > self._peak_vram_observed_mb:
                self._peak_vram_observed_mb = peak
            # Evict immediately to honor single-model residence
            self.evict_current()


class PerceptionStageRunner:
    """Sequentially executes perception stages (Segmentation -> Depth -> Embeddings)
    one model at a time, enforcing the RTX 4050 memory ceiling.
    """

    def __init__(
        self,
        scheduler: Optional[ModelResidentScheduler] = None,
        vram_budget_mb: float = RTX4050_MAX_USABLE_VRAM_MB,
    ):
        self.scheduler = scheduler or ModelResidentScheduler(vram_budget_mb=vram_budget_mb)
        self.vram_budget_mb = vram_budget_mb

    def run_pipeline(
        self,
        evidence_items: Sequence[Any],
        segmentation_fn: Optional[Callable[[DeviceInfo, Sequence[Any]], Any]] = None,
        depth_fn: Optional[Callable[[DeviceInfo, Sequence[Any]], Any]] = None,
        embeddings_fn: Optional[Callable[[DeviceInfo, Sequence[Any]], Any]] = None,
        segmentation_model: str = "maskrcnn_resnet50_fpn",
        depth_model: str = "midas_dpt_hybrid",
        embeddings_model: str = "xfeat",
        force_cpu: bool = False,
    ) -> Tuple[Dict[str, Any], PerceptionStageReport]:
        """Runs the 3-stage perception pipeline sequentially with single-model VRAM residence.
        
        Returns:
            Tuple of (stage_outputs_dict, PerceptionStageReport)
        """
        results: Dict[str, Any] = {}
        task_records: List[TaskExecutionRecord] = []
        overall_start_time = time.perf_counter()
        overall_peak_vram = 0.0

        stages = [
            ("segmentation", segmentation_model, segmentation_fn),
            ("depth", depth_model, depth_fn),
            ("embeddings", embeddings_model, embeddings_fn),
        ]

        for task_name, model_id, fn in stages:
            if fn is None:
                continue

            t0 = time.perf_counter()
            with self.scheduler.model_scope(model_id, force_cpu=force_cpu) as dev:
                cpu_fallback = (dev.device == "cpu")
                vram_before = self.scheduler.get_gpu_memory_used_mb()
                error_msg: Optional[str] = None
                output = None

                try:
                    output = fn(dev, evidence_items)
                    results[task_name] = output
                    status = "completed"
                except Exception as exc:
                    logger.error("Error in perception stage %s with model %s: %s", task_name, model_id, exc)
                    status = "failed"
                    error_msg = str(exc)

                t1 = time.perf_counter()
                vram_after = self.scheduler.get_gpu_memory_used_mb()
                peak_task_vram = max(vram_before, vram_after)
                if peak_task_vram > overall_peak_vram:
                    overall_peak_vram = peak_task_vram

                task_records.append(
                    TaskExecutionRecord(
                        task=task_name,
                        model_name=model_id,
                        device=dev.device,
                        fp16=dev.fp16,
                        items_processed=len(evidence_items),
                        runtime_ms=(t1 - t0) * 1000.0,
                        peak_vram_mb=peak_task_vram,
                        cpu_fallback=cpu_fallback,
                        status=status,
                        error=error_msg,
                    )
                )

        total_runtime_ms = (time.perf_counter() - overall_start_time) * 1000.0

        report = PerceptionStageReport(
            total_runtime_ms=total_runtime_ms,
            overall_peak_vram_mb=overall_peak_vram,
            tasks=task_records,
            vram_budget_mb=self.vram_budget_mb,
        )

        return results, report
