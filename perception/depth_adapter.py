"""Depth engine adapter for Reality Engine's multi-engine platform.

Wraps Depth Anything V2 (or MiDaS) into the UnifiedEngineOutput contract.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import cv2
import numpy as np

from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.engine_orchestrator import EngineSpec, register_engine

log = logging.getLogger("reality.depth_adapter")


def _depth_available() -> Optional[str]:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        return None
    except ImportError as exc:
        return str(exc)


def _depth_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    t0 = time.perf_counter()
    run_id = ctx.get("run_id", "depth_run")
    obs: List[ObservationRecord] = []
    model_name = ctx.get("depth_model", "depth-anything/Depth-Anything-V2-Small-hf")

    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
    except Exception as exc:
        return UnifiedEngineOutput(
            engine_id="depth_anything", run_id=run_id, observations=(),
            status="failed", error_message=f"model load failed: {exc}",
            wall_time_s=time.perf_counter() - t0,
        )

    for img_path in image_paths:
        try:
            import PIL.Image as PILImage
            img = PILImage.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
            depth = predicted_depth.squeeze().cpu().numpy()
            # Normalize to [0,1] for relative depth
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = (depth - d_min) / (d_max - d_min)
            else:
                depth_norm = depth

            obs.append(ObservationRecord(
                evidence_id=f"depth:{img_path}",
                source_image_id=img_path,
                coordinate_frame=CoordinateFrame(frame_id="image", scale_state="relative", units="relative"),
                model_id=ModelIdentity(engine_id="depth", architecture_id="depth_anything_v2",
                                       checkpoint_id=model_name, license="Apache-2.0", hardware_class="gpu"),
                label="depth_map",
                category="depth",
                confidence=None,  # monocular depth has no calibrated confidence
                sigma_m=None,
                depth_map_ref=img_path,
                metadata={
                    "shape": list(depth.shape),
                    "depth_min": float(d_min),
                    "depth_max": float(d_max),
                    "model": model_name,
                },
            ))
        except Exception as exc:
            log.warning("Depth failed for %s: %s", img_path, exc)

    # Unload model to free VRAM
    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return UnifiedEngineOutput(
        engine_id="depth_anything", run_id=run_id, observations=tuple(obs),
        status="succeeded" if obs else "partial",
        wall_time_s=time.perf_counter() - t0,
        peak_vram_mb=_get_vram_peak(),
        summary={"n_images": len(image_paths), "n_depth_maps": len(obs)},
    )


def _get_vram_peak() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def register_depth_engine() -> None:
    """Register the depth engine with the orchestrator."""
    register_engine(
        EngineSpec(engine_id="depth_anything", architecture_id="depth_anything_v2",
                   checkpoint_id="depth-anything/Depth-Anything-V2-Small-hf",
                   vram_mb=2048, cpu_threads=0, priority=70, license="Apache-2.0",
                   run_condition=_depth_available),
        _depth_run,
    )


__all__ = ["register_depth_engine", "_depth_run"]
