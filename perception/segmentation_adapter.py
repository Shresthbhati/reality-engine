"""Segmentation engine adapter for Reality Engine's multi-engine platform.

Wraps SAM 2.1 (or SAM 2.1 Hiera) into the UnifiedEngineOutput contract.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from perception.engine_evidence import (
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.engine_orchestrator import EngineSpec, register_engine

log = logging.getLogger("reality.segmentation_adapter")


def _sam2_available() -> str | None:
    try:
        from importlib import import_module

        transformers = import_module("transformers")
        getattr(transformers, "Sam2Model")
        getattr(transformers, "Sam2Processor")
        return None
    except (ImportError, AttributeError) as exc:
        return str(exc)


def _encode_rle_binary(mask_2d: np.ndarray) -> dict[str, Any]:
    """Encode a 2D boolean mask into compact row-major RLE with explicit [H, W] size."""
    h, w = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    flat = np.asarray(mask_2d, dtype=np.uint8).ravel()
    counts: list[int] = []
    cur = 0
    run_len = 0
    for v in flat:
        iv = int(v)
        if iv == cur:
            run_len += 1
        else:
            counts.append(run_len)
            cur = iv
            run_len = 1
    counts.append(run_len)
    return {"size": [h, w], "counts": counts}


def _resolve_image_bboxes(bboxes: dict[str, Any], img_path: str) -> list[tuple[float, float, float, float]]:
    """Resolve box prompts by exact path or basename fallback."""
    from pathlib import Path

    if img_path in bboxes:
        return [tuple(float(v) for v in b[:4]) for b in bboxes[img_path]]
    name = Path(img_path).name
    if name in bboxes:
        return [tuple(float(v) for v in b[:4]) for b in bboxes[name]]
    return []


def _select_best_2d_mask(mask_tensor: Any, iou_scores: Any = None, box_idx: int = 0) -> np.ndarray:
    """Extract a single 2D boolean mask (H, W) by selecting the highest-IoU candidate channel."""
    arr = mask_tensor.detach().cpu().numpy() if hasattr(mask_tensor, "detach") else np.asarray(mask_tensor)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3:
        best_ch = 0
        if iou_scores is not None:
            sc = iou_scores.detach().cpu().numpy() if hasattr(iou_scores, "detach") else np.asarray(iou_scores)
            while sc.ndim > 2:
                sc = sc[0]
            if sc.ndim == 2 and box_idx < sc.shape[0]:
                best_ch = int(np.argmax(sc[box_idx]))
            elif sc.ndim == 1 and sc.size > 0:
                best_ch = int(np.argmax(sc))
        arr = arr[min(best_ch, arr.shape[0] - 1)]
    return np.asarray(arr > 0, dtype=bool)


def _sam2_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    from transformers import Sam2Model, Sam2Processor

    t0 = time.perf_counter()
    run_id = ctx.get("run_id", "segmentation_run")
    obs: list[ObservationRecord] = []
    model_name = "facebook/sam2.1-hiera-small"
    bboxes = ctx.get("bboxes", {})  # {image_path_or_name: [(x1,y1,x2,y2), ...]}
    errors: list[str] = []
    empty_masks = 0

    try:
        try:
            processor = Sam2Processor.from_pretrained(model_name, local_files_only=True)
            model = Sam2Model.from_pretrained(model_name, local_files_only=True)
        except Exception:
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
            img_w, img_h = image.size
            img_bboxes = _resolve_image_bboxes(bboxes, str(img_path))
            if not img_bboxes:
                continue
            nested_boxes = [[list(b) for b in img_bboxes]]
            inputs = processor(images=image, input_boxes=nested_boxes, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            masks = processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"]
            )
            iou_scores = getattr(outputs, "iou_scores", None)
            for i, mask_list in enumerate(masks):
                for j, mask in enumerate(mask_list):
                    mask_2d = _select_best_2d_mask(mask, iou_scores=iou_scores, box_idx=j)
                    if mask_2d.shape != (img_h, img_w):
                        errors.append(f"{img_path}:{j}: shape {mask_2d.shape} != ({img_h}, {img_w})")
                        continue
                    mask_area = int(mask_2d.sum())
                    total_pixels = img_h * img_w
                    if mask_area == 0:
                        empty_masks += 1
                    rle = _encode_rle_binary(mask_2d)
                    box_px = img_bboxes[j] if j < len(img_bboxes) else None
                    obs.append(ObservationRecord(
                        evidence_id=f"sam2:{img_path}:{i}:{j}",
                        source_image_id=str(img_path),
                        coordinate_frame=__import__("perception.engine_evidence", fromlist=["CoordinateFrame"]).CoordinateFrame(
                            frame_id="image", scale_state="image", units="pixels"),
                        model_id=__import__("perception.engine_evidence", fromlist=["ModelIdentity"]).ModelIdentity(
                            engine_id="segmentation", architecture_id="sam2", checkpoint_id=model_name,
                            license="Apache-2.0", hardware_class=device),
                        label="segmentation_mask",
                        category="segmentation",
                        bbox_px=box_px,
                        confidence=None,
                        metadata={
                            "box_index": j,
                            "image_width": img_w,
                            "image_height": img_h,
                            "mask_area_px": mask_area,
                            "total_pixels": total_pixels,
                            "mask_coverage": round(mask_area / max(1, total_pixels), 4),
                            "is_empty": mask_area == 0,
                            "rle_mask": rle,
                            "model": model_name,
                        },
                    ))
        except Exception as exc:
            log.warning("SAM2 failed for %s: %s", img_path, exc)
            errors.append(f"{img_path}: {exc}")

    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return UnifiedEngineOutput(
        engine_id="sam2", run_id=run_id, observations=tuple(obs),
        status="succeeded" if (obs and not errors) else ("partial" if obs else "failed"),
        error_message="; ".join(errors) if errors else None,
        wall_time_s=time.perf_counter() - t0,
        peak_vram_mb=_get_vram_peak(),
        summary={
            "n_images": len(image_paths),
            "n_masks": len(obs),
            "empty_masks": empty_masks,
            "errors": len(errors),
        },
    )


def _get_vram_peak() -> float | None:
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
