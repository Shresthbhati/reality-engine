"""Tests for Standardized Multi-Engine Adapters.

Validates:
- BaseEngineAdapter input validation and execution flow
- Open3D TSDF volumetric integration and watertight mesh extraction
- Pycolmap SfM dependency verification and input constraints
- LightGlue feature matcher dependency verification
- Depth Anything V2 and MiDaS adapter verification
- OWL-ViT, Grounding DINO, and Florence-2 open-vocabulary detector verification
- SAM box-prompted instance segmenter verification
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.adapters import (
    DepthAnythingV2Adapter,
    EnginePredictionResult,
    Florence2DetectorAdapter,
    GroundingDinoDetectorAdapter,
    InputValidationError,
    LightGlueMatcherAdapter,
    MidasDPTAdapter,
    Open3DTSDFAdapter,
    OwlVitDetectorAdapter,
    ProducerReport,
    PycolmapSfMAdapter,
    SamPromptedSegmenterAdapter,
)


def test_base_adapter_input_validation():
    # Pycolmap SfM requires at least 2 images
    adapter = PycolmapSfMAdapter()
    is_ready, _ = adapter.verify_dependencies()
    if is_ready:
        with pytest.raises(InputValidationError, match="requires at least 2 images"):
            adapter.validate_inputs(["single_image.jpg"])


def test_open3d_tsdf_adapter_execution():
    adapter = Open3DTSDFAdapter(voxel_length_m=0.05, sdf_trunc_m=0.10)
    is_ready, msg = adapter.verify_dependencies()
    assert is_ready, msg

    # Create synthetic RGB-D frame looking at a planar wall at 2.0 meters
    w, h = 64, 64
    depth = np.ones((h, w), dtype=np.float32) * 2000.0  # 2000 mm = 2.0 m
    color = np.ones((h, w, 3), dtype=np.uint8) * 180

    inputs = {
        "intrinsics": {"width": w, "height": h, "fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 32.0},
        "frames": [
            {
                "depth": depth,
                "color": color,
                "extrinsic_4x4": np.eye(4).tolist(),
                "depth_scale": 1000.0,
                "max_depth": 4.0,
            }
        ],
    }

    result = adapter.execute(inputs)
    assert isinstance(result, EnginePredictionResult)
    assert result.success is True
    assert isinstance(result.report, ProducerReport)
    assert result.report.engine_name == "open3d_tsdf_mesher"
    assert result.report.task_category == "volumetric_surface_reconstruction"
    assert result.report.licence == "MIT"

    preds = result.predictions
    assert preds["integrated_frames"] == 1
    assert "num_vertices" in preds
    assert "num_faces" in preds


def test_matcher_lightglue_adapter_verification():
    adapter = LightGlueMatcherAdapter(feature_type="aliked")
    is_ready, msg = adapter.verify_dependencies()
    assert is_ready, msg
    assert adapter.engine_name == "lightglue_aliked"
    assert adapter.licence == "Apache-2.0"

    # Input validation expects exactly 2 images
    with pytest.raises(InputValidationError):
        adapter.validate_inputs(["single_image.png"])


def test_depth_adapters_verification():
    d_v2 = DepthAnythingV2Adapter()
    ready_v2, msg_v2 = d_v2.verify_dependencies()
    assert ready_v2, msg_v2
    assert d_v2.licence == "Apache-2.0"

    midas = MidasDPTAdapter()
    ready_m, msg_m = midas.verify_dependencies()
    assert ready_m, msg_m
    assert midas.licence == "MIT"


def test_detector_adapters_verification():
    owl = OwlVitDetectorAdapter()
    ready_owl, msg_owl = owl.verify_dependencies()
    assert ready_owl, msg_owl
    assert owl.licence == "Apache-2.0"

    gdino = GroundingDinoDetectorAdapter()
    ready_gdino, msg_gdino = gdino.verify_dependencies()
    assert ready_gdino, msg_gdino
    assert gdino.licence == "Apache-2.0"

    flor = Florence2DetectorAdapter()
    ready_flor, msg_flor = flor.verify_dependencies()
    assert ready_flor, msg_flor
    assert flor.licence == "MIT"


def test_segmenter_sam_adapter_verification():
    sam = SamPromptedSegmenterAdapter()
    ready_sam, msg_sam = sam.verify_dependencies()
    assert ready_sam, msg_sam
    assert sam.licence == "Apache-2.0"

    # Validation expects dict with image and boxes
    with pytest.raises(InputValidationError):
        sam.validate_inputs(["not_a_dict"])
