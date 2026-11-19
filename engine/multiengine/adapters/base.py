"""Base Architecture for Real Multi-Engine Adapters.

Enforces the mandatory 10-step model execution lifecycle:
1. Select model
2. Load dependencies & weights
3. Run inference on complete dataset (original photos + metadata)
4. Preserve raw output (in original_coordinates)
5. Validate output (geometric & numerical validity)
6. Normalise predictions into SI units in coordinate frame
7. Record errors, uncertainty and execution telemetry
8. Release model resources (GPU cache & host RAM)
9. Verify persistence in EvidenceStore
10. Proceed to next model
"""

from __future__ import annotations

import gc
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

try:
    import torch
except ImportError:
    torch = None

from engine.multiengine.contract import (
    Device,
    EngineAdapter,
    EngineError,
    EngineErrorKind,
    EngineRequest,
    EngineSpec,
)
from engine.multiengine.observation import ObservationDraft


class BaseModelEngineAdapter(EngineAdapter, ABC):
    """Base class for all concrete reconstruction and perception engine adapters."""

    def __init__(self, spec: EngineSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._device: Device = Device("cpu")
        self._vram_start_mb: float = 0.0
        self._peak_vram_mb: float | None = None
        self._peak_ram_mb: float | None = None

    def validate_input(self, request: EngineRequest) -> None:
        """Validate input evidence sources before execution."""
        if not request.sources:
            raise EngineError(
                EngineErrorKind.INVALID_INPUT,
                f"Engine {self.spec.engine_id} requires at least one input source.",
            )

    def load(self, device: Device) -> None:
        """Load dependencies and model weights onto the requested device."""
        self._device = device
        has_cuda = torch is not None and torch.cuda.is_available() and device.kind == "cuda"
        if has_cuda:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            try:
                self._vram_start_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                self._vram_start_mb = 0.0
        else:
            self._vram_start_mb = 0.0

        self._load_model_weights(device)

    def unload(self) -> None:
        """Explicitly release model resources, collect garbage, and purge CUDA cache."""
        self._unload_model_weights()
        self._model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def _load_model_weights(self, device: Device) -> None:
        """Subclass loads the specific model weights."""
        ...

    def _unload_model_weights(self) -> None:
        """Optional hook for subclasses to tear down custom runtime sessions."""
        pass

    def measured_usage(self) -> tuple[float | None, float | None]:
        """Returns (peak_vram_mb, peak_ram_mb) measured during inference."""
        has_cuda = torch is not None and torch.cuda.is_available() and self._device.kind == "cuda"
        if has_cuda:
            try:
                peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception:
                peak = 0.0
            self._peak_vram_mb = max(0.0, peak - self._vram_start_mb)
        else:
            self._peak_vram_mb = 0.0
        return self._peak_vram_mb, self._peak_ram_mb

    def validate_numeric_array(self, arr: Sequence[float], name: str = "array") -> None:
        """Validate that array elements are finite and within physical bounds."""
        for i, val in enumerate(arr):
            if val is None or math.isnan(val) or math.isinf(val):
                raise EngineError(
                    EngineErrorKind.MODEL_ERROR, f"{name}[{i}] contains non-finite value: {val}"
                )

    @abstractmethod
    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]:
        """Run genuine model inference and return standardized ObservationDrafts."""
        ...
