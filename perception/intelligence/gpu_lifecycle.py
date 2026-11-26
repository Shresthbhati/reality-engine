"""GPU Lifecycle & Constrained-Hardware Execution Manager (A5 Objective 4).

Manages perception model execution on constrained consumer GPUs (e.g. 6 GB RTX 4050):
  - Sequential GPU execution enforcement (refuses concurrent GPU claims when VRAM is insufficient)
  - Live VRAM telemetry (allocated, reserved, peak, free)
  - Single-resident GPU model cache with automatic eviction & ``torch.cuda.empty_cache()``
  - CPU offloading fallback when VRAM budget is exhausted
  - Adaptive image resolution & batch-size selection
  - Wall-clock timeout & transient failure recovery
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class HardwareProfile:
    cuda_available: bool
    device_name: str
    total_vram_mb: float
    free_vram_mb: float
    safe_vram_ceiling_mb: float
    supports_concurrent_models: bool


@dataclass(frozen=True)
class ModelStageExecutionRecord:
    stage_name: str
    model_id: str
    device_used: str  # "cuda" | "cpu"
    status: str  # "succeeded" | "timed_out" | "oom_recovered_cpu" | "failed"
    wall_time_s: float
    peak_vram_mb: float | None
    batch_size: int
    max_resolution_px: int
    error_message: str | None = None


def probe_hardware_profile(*, safety_margin_mb: float = 600.0) -> HardwareProfile:
    """Inspect live CUDA/CPU hardware and compute safe VRAM ceiling."""
    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            total_mb = round(total_b / (1024.0 * 1024.0), 2)
            free_mb = round(free_b / (1024.0 * 1024.0), 2)
            name = torch.cuda.get_device_name(0)
            ceiling = max(0.0, round(total_mb - safety_margin_mb, 2))
            return HardwareProfile(
                cuda_available=True,
                device_name=name,
                total_vram_mb=total_mb,
                free_vram_mb=free_mb,
                safe_vram_ceiling_mb=ceiling,
                supports_concurrent_models=total_mb >= 14000.0,
            )
    except Exception:
        pass
    return HardwareProfile(
        cuda_available=False,
        device_name="cpu",
        total_vram_mb=0.0,
        free_vram_mb=0.0,
        safe_vram_ceiling_mb=0.0,
        supports_concurrent_models=False,
    )


def select_adaptive_batch_and_resolution(
    num_images: int,
    estimated_model_vram_mb: float,
    hw: HardwareProfile,
) -> tuple[int, int, str]:
    """Choose ``(batch_size, max_resolution_px, target_device)`` for current hardware."""
    if not hw.cuda_available or hw.free_vram_mb < estimated_model_vram_mb * 0.85:
        return (1, 1024, "cpu")
    headroom = hw.free_vram_mb - estimated_model_vram_mb
    if headroom >= 2500.0:
        batch = min(max(1, num_images), 4)
        res = 1280
    elif headroom >= 1000.0:
        batch = min(max(1, num_images), 2)
        res = 1024
    else:
        batch = 1
        res = 800
    return (batch, res, "cuda")


class ModelLifecycleManager:
    """Coordinates sequential GPU model execution, eviction, timeouts, and OOM recovery."""

    def __init__(self, *, vram_ceiling_mb: float | None = None) -> None:
        self.hw = probe_hardware_profile()
        self.vram_ceiling_mb = (
            vram_ceiling_mb
            if vram_ceiling_mb is not None
            else self.hw.safe_vram_ceiling_mb
        )
        self._resident_models: dict[str, Any] = {}
        self.execution_log: list[ModelStageExecutionRecord] = []

    def can_run_concurrently(self, model_vram_list_mb: Sequence[float]) -> bool:
        """Return True ONLY if total required VRAM + margin fits comfortably on the GPU."""
        if not self.hw.cuda_available:
            return False
        return sum(model_vram_list_mb) <= self.vram_ceiling_mb and self.hw.supports_concurrent_models

    def release_all_gpu_models(self) -> None:
        """Evict cached models and reclaim CUDA memory before loading the next stage."""
        self._resident_models.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def run_stage(
        self,
        stage_name: str,
        model_id: str,
        fn: Callable[[str, int, int], T],
        *,
        num_images: int = 1,
        estimated_vram_mb: float = 2000.0,
        timeout_s: float = 120.0,
        allow_cpu_fallback_on_oom: bool = True,
    ) -> tuple[T | None, ModelStageExecutionRecord]:
        """Execute one model stage sequentially with VRAM eviction, timeout, and OOM CPU fallback."""
        # Always evict prior GPU models on constrained GPUs that cannot co-host models
        if not self.hw.supports_concurrent_models:
            self.release_all_gpu_models()

        hw_now = probe_hardware_profile()
        batch_size, max_res, device = select_adaptive_batch_and_resolution(
            num_images, estimated_vram_mb, hw_now
        )
        t0 = time.perf_counter()

        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        def _invoke(dev: str) -> T:
            return fn(dev, batch_size, max_res)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_invoke, device)
                result = fut.result(timeout=timeout_s if timeout_s > 0 else None)
            peak = self._read_peak_vram(device)
            rec = ModelStageExecutionRecord(
                stage_name=stage_name,
                model_id=model_id,
                device_used=device,
                status="succeeded",
                wall_time_s=round(time.perf_counter() - t0, 4),
                peak_vram_mb=peak,
                batch_size=batch_size,
                max_resolution_px=max_res,
            )
            self.execution_log.append(rec)
            return result, rec

        except FuturesTimeoutError:
            rec = ModelStageExecutionRecord(
                stage_name=stage_name,
                model_id=model_id,
                device_used=device,
                status="timed_out",
                wall_time_s=round(time.perf_counter() - t0, 4),
                peak_vram_mb=self._read_peak_vram(device),
                batch_size=batch_size,
                max_resolution_px=max_res,
                error_message=f"stage timed out after {timeout_s}s",
            )
            self.execution_log.append(rec)
            return None, rec

        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower() or "cuda" in str(exc).lower()
            if is_oom and device == "cuda" and allow_cpu_fallback_on_oom:
                self.release_all_gpu_models()
                try:
                    result = _invoke("cpu")
                    rec = ModelStageExecutionRecord(
                        stage_name=stage_name,
                        model_id=model_id,
                        device_used="cpu",
                        status="oom_recovered_cpu",
                        wall_time_s=round(time.perf_counter() - t0, 4),
                        peak_vram_mb=self._read_peak_vram("cuda"),
                        batch_size=1,
                        max_resolution_px=min(max_res, 1024),
                        error_message=f"cuda_oom_fallback:{exc}",
                    )
                    self.execution_log.append(rec)
                    return result, rec
                except Exception as cpu_exc:
                    exc = cpu_exc
            rec = ModelStageExecutionRecord(
                stage_name=stage_name,
                model_id=model_id,
                device_used=device,
                status="failed",
                wall_time_s=round(time.perf_counter() - t0, 4),
                peak_vram_mb=self._read_peak_vram(device),
                batch_size=batch_size,
                max_resolution_px=max_res,
                error_message=f"{type(exc).__name__}: {exc}",
            )
            self.execution_log.append(rec)
            return None, rec

        except Exception as exc:
            rec = ModelStageExecutionRecord(
                stage_name=stage_name,
                model_id=model_id,
                device_used=device,
                status="failed",
                wall_time_s=round(time.perf_counter() - t0, 4),
                peak_vram_mb=self._read_peak_vram(device),
                batch_size=batch_size,
                max_resolution_px=max_res,
                error_message=f"{type(exc).__name__}: {exc}",
            )
            self.execution_log.append(rec)
            return None, rec

    @staticmethod
    def _read_peak_vram(device: str) -> float | None:
        if device != "cuda":
            return None
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
        except Exception:
            pass
        return None


__all__ = [
    "HardwareProfile",
    "ModelLifecycleManager",
    "ModelStageExecutionRecord",
    "probe_hardware_profile",
    "select_adaptive_batch_and_resolution",
]
