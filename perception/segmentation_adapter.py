"""Segmentation engine adapter for Reality Engine's multi-engine platform.

Wraps SAM 2.1 (or SAM 2.1 Hiera) into the UnifiedEngineOutput contract.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Sequence

import cv2
import numpy as np
import torch

from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.engine_orchestrator import EngineSpec, register_engine

log = logging.getLogger("reality.segmentation_adapter")


def _sam2_available() -> Optional[str]:
    try:
        from transformers import Sam2Model, Sam2Processor
        return None
    except ImportError as exc:
        return str(exc)


def _sam2_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    from transformers import Sam2Model, Sam2Processor

    t0 = time.perf_counter()
    run_id = ctx.get("run_id", "segmentation_run")
    obs: List[ObservationRecord] = []
    model_name = "facebook/sam2.1-hiera-small"
    bboxes = ctx.get("bboxes", {})  # {image_path: [(x1,y1,x2,y2), ...]}

    try:
        processor = Sam2Processor.from_pretrained(model_name)
        model = Sam2Model.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
    except Exception as exc:
        return UnifiedEngineOutput(
            engine_id="sam2", run_id=run_id, observations=(),
            status="failed", error_message=f"model load failed: {exc}",
            wall_time_s=time.perf_counter() - t0,
        )

    for img_path in image_paths:
        try:
            import PIL.Image as PILImage
            image = PILImage.open(img_path).convert("RGB")
            img_bboxes = bboxes.get(img_path, [])
            if not img_bboxes:
                # No boxes provided — skip this image (SAM needs prompts)
                continue
            # SAM2 expects [[[x1,y1,x2,y2], ...]] (image level, box level, box coordinates)
            nested_boxes = [[list(b) for b in img_bboxes]]
            inputs = processor(images=image, input_boxes=nested_boxes, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            masks = processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"]
            )
            for i, mask_list in enumerate(masks):
                for j, mask in enumerate(mask_list):
                    mask_np = mask.squeeze().cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
                    mask_area = int(mask_np.sum()) if mask_np.ndim >= 2 else 0
                    total_pixels = mask_np.shape[-2] * mask_np.shape[-1] if mask_np.ndim >= 2 else 1
                    obs.append(ObservationRecord(
                        evidence_id=f"sam2:{img_path}:{i}:{j}",
                        source_image_id=img_path,
                        coordinate_frame=__import__("perception.engine_evidence", fromlist=["CoordinateFrame"]).CoordinateFrame(
                            frame_id="image", scale_state="image", units="pixels"),
                        model_id=__import__("perception.engine_evidence", fromlist=["ModelIdentity"]).ModelIdentity(
                            engine_id="segmentation", architecture_id="sam2", checkpoint_id=model_name,
                            license="Apache-2.0", hardware_class="gpu"),
                        label="segmentation_mask",
                        category="segmentation",
                        confidence=None,
                        metadata={
                            "mask_area_px": mask_area,
                            "total_pixels": total_pixels,
                            "mask_coverage": round(mask_area / max(1, total_pixels), 4),
                            "model": model_name,
                        },
                    ))
        except Exception as exc:
            log.warning("SAM2 failed for %s: %s", img_path, exc)

    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return UnifiedEngineOutput(
        engine_id="sam2", run_id=run_id, observations=tuple(obs),
        status="succeeded" if obs else "partial",
        wall_time_s=time.perf_counter() - t0,
        peak_vram_mb=_get_vram_peak(),
        summary={"n_images": len(image_paths), "n_masks": len(obs)},
    )


def _get_vram_peak() -> Optional[float]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def register_segmentation_engine() -> None:
    register_engine(
        EngineSpec(engine_id="sam2", architecture_id="sam2", checkpoint_id="facebook/sam2.1-hiera-small",
                   vram_mb=3072, cpu_threads=0, priority=75, license="Apache-2.0",
                   dependencies=("grounding_dino",), run_condition=_sam2_available),
        _sam2_run,
    )


__all__ = ["register_segmentation_engine", "_sam2_run"]
