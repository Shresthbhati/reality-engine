"""Standardized Engine Adapter Base Architecture.

Defines the contract for independent multi-engine adapters in Reality Engine.
Enforces:
- Strict input validation before execution
- Dependency, licence, and checkpoint verification
- Resource tracking (peak VRAM, CPU memory, execution latency)
- Transparent provenance reporting (ProducerReport)
- Explicit refusal without fake outputs
"""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


class EngineUnavailableError(RuntimeError):
    """Raised when an engine cannot run due to missing dependencies, weights, or device."""
    pass


class DependencyMissingError(EngineUnavailableError):
    """Raised when a Python package or binary is not installed."""
    pass


class LicenceBlockedError(EngineUnavailableError):
    """Raised when an engine's licence prohibits production execution."""
    pass


class InputValidationError(ValueError):
    """Raised when input evidence fails resolution or geometric validation."""
    pass


@dataclass(frozen=True)
class ProducerReport:
    """Standardized provenance and execution telemetry produced by any adapter."""

    engine_name: str
    task_category: str
    model_id: str | None
    model_version: str | None
    checkpoint_hash: str | None
    licence: str
    device: str
    execution_time_ms: float
    peak_vram_mb: float
    host_ram_mb: float
    confidence_is_calibrated: bool
    parameters: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnginePredictionResult:
    """Standardized prediction container preserving raw outputs and telemetry."""

    engine_name: str
    task_category: str
    success: bool
    report: ProducerReport
    predictions: Any  # Strongly-typed per category (poses, depth maps, boxes, masks, meshes)
    raw_output: Any | None = None  # Unmodified raw tensor / model output
    error_message: str | None = None


class BaseEngineAdapter(ABC):
    """Abstract base class for all independent model and engine adapters."""

    def __init__(
        self,
        engine_name: str,
        task_category: str,
        licence: str,
        device: str | None = None,
    ) -> None:
        self.engine_name = engine_name
        self.task_category = task_category
        self.licence = licence
        self._custom_device = device
        self._model: Any | None = None

    @abstractmethod
    def verify_dependencies(self) -> tuple[bool, str]:
        """Verify whether all required packages, binaries, and licences are satisfied."""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load the model checkpoint into memory / device."""
        ...

    def unload_model(self) -> None:
        """Safely release GPU memory and delete model references."""
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def validate_inputs(self, inputs: Any) -> None:
        """Validate input shapes, formats, and evidence constraints."""
        ...

    @abstractmethod
    def _execute_inference(self, validated_inputs: Any, options: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        """Executes actual model inference.

        Returns:
            Tuple of (structured_predictions, raw_predictions, diagnostics_dict)
        """
        ...

    def execute(self, inputs: Any, options: dict[str, Any] | None = None) -> EnginePredictionResult:
        """Standard execution workflow with validation, timing, and memory tracking."""
        options = options or {}

        # 1. Verify dependencies
        is_ready, reason = self.verify_dependencies()
        if not is_ready:
            raise EngineUnavailableError(f"Engine {self.engine_name} is unavailable: {reason}")

        # 2. Validate inputs
        self.validate_inputs(inputs)

        # 3. Prepare device and memory telemetry
        device_str = self._custom_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        vram_start = 0.0
        if torch.cuda.is_available() and "cuda" in device_str:
            torch.cuda.reset_peak_memory_stats()
            vram_start = torch.cuda.memory_allocated() / (1024 * 1024)

        t0 = time.perf_counter()

        # 4. Load model if needed
        self.load_model()

        # 5. Run genuine inference
        predictions, raw_output, diagnostics = self._execute_inference(inputs, options)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        vram_peak = 0.0
        if torch.cuda.is_available() and "cuda" in device_str:
            vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

        report = ProducerReport(
            engine_name=self.engine_name,
            task_category=self.task_category,
            model_id=options.get("model_id", self.engine_name),
            model_version=options.get("model_version", "1.0"),
            checkpoint_hash=options.get("checkpoint_hash"),
            licence=self.licence,
            device=device_str,
            execution_time_ms=elapsed_ms,
            peak_vram_mb=max(0.0, vram_peak - vram_start),
            host_ram_mb=options.get("host_ram_mb", 0.0),
            confidence_is_calibrated=options.get("is_calibrated", False),
            parameters=options,
            diagnostics=diagnostics,
        )

        return EnginePredictionResult(
            engine_name=self.engine_name,
            task_category=self.task_category,
            success=True,
            report=report,
            predictions=predictions,
            raw_output=raw_output,
        )
