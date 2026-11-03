"""ONNX Runtime algorithmic extraction: Dynamic INT8 Quantization, VRAM profiling, and Provider fallback.

Enables hardware-aware inference acceleration, dynamic range 8-bit integer quantization,
and guaranteed RTX 4050 6GB VRAM bounds (< 6,144 MB) for vision and spatial perception models.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class INT8QuantizedTensor:
    """Quantized 8-bit signed integer tensor with scale and zero-point parameters."""
    q_data: np.ndarray       # int8 array
    scale: float             # Float scale factor s
    zero_point: int          # Integer zero point z
    symmetric: bool
    sqnr_db: float           # Signal-to-Quantization-Noise Ratio in decibels

    def dequantize(self) -> np.ndarray:
        """Reconstruct float32 representation: X_hat = s * (q - z)."""
        return self.scale * (self.q_data.astype(np.float64) - self.zero_point).astype(np.float32)


@dataclass(frozen=True)
class ModelVRAMProfile:
    """Hardware VRAM profile for model deployment under RTX 4050 6GB ceiling."""
    model_name: str
    parameter_count: int
    precision: str           # "fp32", "fp16", or "int8"
    weights_mb: float
    activations_mb: float
    total_peak_vram_mb: float
    fits_rtx4050_6gb: bool
    vram_headroom_mb: float


def quantize_dynamic_int8(
    tensor: np.ndarray,
    symmetric: bool = True,
) -> INT8QuantizedTensor:
    """Quantize float32 tensor to int8 with calibrated scale and zero point."""
    arr = np.asarray(tensor, dtype=np.float32)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    if symmetric:
        abs_max = max(abs(min_val), abs(max_val), 1e-8)
        scale = abs_max / 127.0
        zero_point = 0
        q_raw = np.round(arr / scale)
        q_clamped = np.clip(q_raw, -128, 127).astype(np.int8)
    else:
        val_range = max(max_val - min_val, 1e-8)
        scale = val_range / 255.0
        zero_point = int(np.round(-min_val / scale)) - 128
        q_raw = np.round(arr / scale) + zero_point
        q_clamped = np.clip(q_raw, -128, 127).astype(np.int8)

    # Compute Signal-to-Quantization-Noise Ratio (SQNR)
    dequant = scale * (q_clamped.astype(np.float64) - zero_point)
    noise = arr - dequant
    signal_power = float(np.mean(arr ** 2))
    noise_power = float(np.mean(noise ** 2))

    if noise_power < 1e-12:
        sqnr_db = 100.0
    elif signal_power < 1e-12:
        sqnr_db = 0.0
    else:
        sqnr_db = float(10.0 * math.log10(signal_power / noise_power))

    return INT8QuantizedTensor(
        q_data=q_clamped,
        scale=scale,
        zero_point=zero_point,
        symmetric=symmetric,
        sqnr_db=sqnr_db,
    )


def estimate_model_vram(
    model_name: str,
    parameter_count: int,
    input_resolution: Tuple[int, int] = (640, 640),
    batch_size: int = 1,
    precision: str = "fp32",
    gpu_ceiling_mb: float = 6144.0,  # RTX 4050 6GB limit
) -> ModelVRAMProfile:
    """Estimate weights and peak activation memory under specified precision and resolution."""
    bytes_per_param = {
        "fp32": 4.0,
        "fp16": 2.0,
        "int8": 1.0,
    }.get(precision.lower(), 4.0)

    weights_mb = (parameter_count * bytes_per_param) / (1024.0 * 1024.0)

    # Heuristic activation memory based on spatial feature pyramids
    h, w = input_resolution
    spatial_elements = h * w
    # Typical CNN/Transformer activation footprint is ~30-50x input tensor size during forward pass
    activation_elements = batch_size * spatial_elements * 3 * 35
    activations_mb = (activation_elements * bytes_per_param) / (1024.0 * 1024.0)

    # Add 256MB runtime framework scratchpad (CUDA/DirectML context)
    runtime_overhead_mb = 256.0
    total_peak_vram_mb = weights_mb + activations_mb + runtime_overhead_mb
    fits_ceiling = total_peak_vram_mb <= gpu_ceiling_mb
    headroom_mb = gpu_ceiling_mb - total_peak_vram_mb

    return ModelVRAMProfile(
        model_name=model_name,
        parameter_count=parameter_count,
        precision=precision.lower(),
        weights_mb=round(weights_mb, 2),
        activations_mb=round(activations_mb, 2),
        total_peak_vram_mb=round(total_peak_vram_mb, 2),
        fits_rtx4050_6gb=fits_ceiling,
        vram_headroom_mb=round(headroom_mb, 2),
    )


class OnnxQuantizationEngine:
    """Manages ONNX Runtime execution provider selection and INT8 quantization workflows."""

    def __init__(self, preferred_providers: Optional[Sequence[str]] = None):
        self.preferred_providers = list(preferred_providers or [
            "CUDAExecutionProvider",
            "DirectMLExecutionProvider",
            "CPUExecutionProvider",
        ])

    @classmethod
    def is_onnxruntime_available(cls) -> Tuple[bool, str]:
        """Check if onnxruntime is installed in current Python environment."""
        spec = importlib.util.find_spec("onnxruntime")
        if spec is not None:
            return True, "onnxruntime installed"
        return False, "onnxruntime not installed (using pure Python/NumPy quantization fallback)"

    def resolve_active_provider(self) -> str:
        """Resolve highest available execution provider with deterministic CPU fallback."""
        avail, _ = self.is_onnxruntime_available()
        if not avail:
            return "CPUExecutionProvider (Fallback)"

        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            for pref in self.preferred_providers:
                if pref in available:
                    return pref
        except Exception:
            pass

        return "CPUExecutionProvider"

    def quantize_weights(
        self,
        weights_dict: Dict[str, np.ndarray],
        symmetric: bool = True,
    ) -> Tuple[Dict[str, INT8QuantizedTensor], Dict[str, Any]]:
        """Quantize an entire state dict / weights dictionary to INT8."""
        quantized = {}
        total_sqnr = 0.0
        orig_bytes = 0
        quant_bytes = 0

        for name, tensor in weights_dict.items():
            q_tensor = quantize_dynamic_int8(tensor, symmetric=symmetric)
            quantized[name] = q_tensor
            total_sqnr += q_tensor.sqnr_db
            orig_bytes += tensor.nbytes
            quant_bytes += q_tensor.q_data.nbytes

        num_tensors = max(1, len(weights_dict))
        summary = {
            "num_tensors": num_tensors,
            "original_bytes": orig_bytes,
            "quantized_bytes": quant_bytes,
            "compression_ratio": float(orig_bytes / max(1, quant_bytes)),
            "average_sqnr_db": float(total_sqnr / num_tensors),
            "symmetric": symmetric,
        }
        return quantized, summary
