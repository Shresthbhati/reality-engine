"""Unit tests for ONNX Runtime acceleration and dynamic INT8 quantization."""

import numpy as np
import pytest

from perception.acceleration.onnx_quantizer import (
    OnnxQuantizationEngine,
    estimate_model_vram,
    quantize_dynamic_int8,
)


def test_dynamic_int8_quantization_and_dequantization():
    """Verify symmetric and asymmetric INT8 quantization accurately reconstruct float32 data."""
    rng = np.random.default_rng(42)
    # Generate Gaussian distributed weights
    weights = rng.normal(loc=0.0, scale=0.5, size=(128, 256)).astype(np.float32)

    # 1. Symmetric quantization
    q_sym = quantize_dynamic_int8(weights, symmetric=True)
    assert q_sym.q_data.dtype == np.int8
    assert q_sym.zero_point == 0
    assert q_sym.sqnr_db > 35.0  # High SNR for INT8 quantization

    dequant_sym = q_sym.dequantize()
    # Reconstruction error should be bounded by scale / 2
    max_err = float(np.max(np.abs(weights - dequant_sym)))
    assert max_err <= (q_sym.scale / 2.0) + 1e-5

    # 2. Asymmetric quantization (e.g. ReLU activations >= 0)
    activations = rng.uniform(low=0.0, high=10.0, size=(64, 64)).astype(np.float32)
    q_asym = quantize_dynamic_int8(activations, symmetric=False)
    assert q_asym.q_data.dtype == np.int8
    assert q_asym.sqnr_db > 35.0

    dequant_asym = q_asym.dequantize()
    assert np.allclose(activations, dequant_asym, atol=q_asym.scale)


def test_model_vram_estimation_enforces_rtx4050_bound():
    """Verify VRAM estimation correctly identifies fits and headroom for RTX 4050 (6,144 MB)."""
    # 50M parameter detector (e.g. YOLOX-X) in INT8 vs FP32
    fp32_profile = estimate_model_vram(
        model_name="YOLOX-X",
        parameter_count=50_000_000,
        input_resolution=(640, 640),
        batch_size=1,
        precision="fp32",
    )
    assert fp32_profile.weights_mb == pytest.approx(190.73, rel=0.05)
    assert fp32_profile.fits_rtx4050_6gb is True
    assert fp32_profile.vram_headroom_mb > 0

    int8_profile = estimate_model_vram(
        model_name="YOLOX-X",
        parameter_count=50_000_000,
        input_resolution=(640, 640),
        batch_size=1,
        precision="int8",
    )
    # Weights should be 1/4 the size of FP32
    assert int8_profile.weights_mb == pytest.approx(fp32_profile.weights_mb / 4.0, rel=0.05)
    assert int8_profile.vram_headroom_mb > fp32_profile.vram_headroom_mb


def test_quantization_engine_state_dict_processing():
    """Verify OnnxQuantizationEngine quantizes multi-tensor state dict with compression statistics."""
    engine = OnnxQuantizationEngine()
    state_dict = {
        "backbone.layer1.weight": np.random.randn(64, 64, 3, 3).astype(np.float32),
        "backbone.layer2.weight": np.random.randn(128, 64, 3, 3).astype(np.float32),
        "head.cls.weight": np.random.randn(80, 128).astype(np.float32),
    }

    quantized_dict, summary = engine.quantize_weights(state_dict, symmetric=True)
    assert len(quantized_dict) == 3
    assert summary["compression_ratio"] == pytest.approx(4.0, rel=0.01)
    assert summary["average_sqnr_db"] > 30.0

    # Test provider resolution fallback
    provider = engine.resolve_active_provider()
    assert "ExecutionProvider" in provider
