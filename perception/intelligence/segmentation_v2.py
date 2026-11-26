"""A5.1 Segmentation Reliability Subsystem.

Provides reliable, provenance-preserving segmentation mask generation and quality
verification for ``NormalizedPerceptionObservation`` records:
  - Supports SAM 2 (``facebook/sam2.1-hiera-small``), SAM 3 (when available),
    and SAM-HQ (``syscv-community/sam-hq-vit-base``) via local cache first.
  - Enforces explicit coordinate system & image-dimension validation.
  - Diagnoses empty masks explicitly and NEVER fabricates masks when
    a segmenter fails or prompts are missing.
  - Measures mask quality (box IoU, solidity, hole count, boundary uncertainty).
"""

from __future__ import annotations

import importlib.util
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from perception.intelligence.data_model import (
    NormalizedPerceptionObservation,
    SegmentationObservation,
)
from perception.segmentation_adapter import _encode_rle_binary, _select_best_2d_mask


def _compute_mask_quality(
    mask_2d: np.ndarray,
    bbox_px: tuple[float, float, float, float],
    *,
    is_architectural: bool = False,
) -> tuple[float, float, int, float, float]:
    """Compute (box_iou, solidity, hole_count, boundary_uncertainty, quality_score)."""
    ih, iw = mask_2d.shape[:2]
    x1 = max(0, min(iw - 1, int(round(bbox_px[0]))))
    y1 = max(0, min(ih - 1, int(round(bbox_px[1]))))
    x2 = max(x1 + 1, min(iw, int(round(bbox_px[2]))))
    y2 = max(y1 + 1, min(ih, int(round(bbox_px[3]))))

    box_mask = np.zeros((ih, iw), dtype=bool)
    box_mask[y1:y2, x1:x2] = True
    inter = int(np.sum(mask_2d & box_mask))
    union = int(np.sum(mask_2d | box_mask))
    box_iou = inter / float(union) if union > 0 else 0.0

    ys, xs = np.where(mask_2d)
    if ys.size == 0:
        return (0.0, 0.0, 0, 1.0, 0.0)
    tight_area = max(1, (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1))
    solidity = min(1.0, float(mask_2d.sum()) / float(tight_area))

    # Boundary uncertainty from edge perimeter ratio
    perim = mask_2d ^ (
        np.roll(mask_2d, 1, axis=0)
        & np.roll(mask_2d, -1, axis=0)
        & np.roll(mask_2d, 1, axis=1)
        & np.roll(mask_2d, -1, axis=1)
    )
    perim_ratio = float(perim.sum()) / float(max(1, int(mask_2d.sum())))
    boundary_unc = min(1.0, round(perim_ratio * 1.5, 4))

    hole_count = 0
    try:
        import cv2

        contours, hierarchy = cv2.findContours(
            mask_2d.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is not None:
            hole_count = int(sum(1 for h in hierarchy[0] if h[3] >= 0))
    except Exception:
        pass

    w_iou = 0.45 if not is_architectural else 0.35
    w_sol = 0.40 if not is_architectural else 0.50
    q_score = max(0.0, min(1.0, w_iou * box_iou + w_sol * solidity + 0.15 * (1.0 - boundary_unc)))
    return (
        round(box_iou, 4),
        round(solidity, 4),
        hole_count,
        round(boundary_unc, 4),
        round(q_score, 4),
    )


@dataclass(frozen=True)
class SegmentationExecutionReport:
    segmenter_id: str
    checkpoint_id: str
    status: str  # "succeeded" | "partial" | "unavailable" | "failed" | "skipped"
    total_prompts: int
    masks_produced: int
    empty_masks: int
    rejected_masks: int
    mean_mask_quality: float
    mean_mask_coverage: float
    wall_time_s: float
    peak_vram_mb: float | None
    diagnostics: tuple[str, ...] = ()


def probe_available_segmenters() -> dict[str, dict[str, Any]]:
    """Probe which segmentation backends (SAM2, SAM3, SAM-HQ) are available."""
    out: dict[str, dict[str, Any]] = {}
    has_tf = importlib.util.find_spec("transformers") is not None
    has_torch = importlib.util.find_spec("torch") is not None
    if not (has_tf and has_torch):
        return {
            "sam2": {"available": False, "reason": "torch/transformers missing"},
            "sam3": {"available": False, "reason": "torch/transformers missing"},
            "sam_hq": {"available": False, "reason": "torch/transformers missing"},
        }
    import transformers

    out["sam2"] = {
        "available": hasattr(transformers, "Sam2Model") and hasattr(transformers, "Sam2Processor"),
        "checkpoint": "facebook/sam2.1-hiera-small",
    }
    out["sam3"] = {
        "available": hasattr(transformers, "Sam3Model") and hasattr(transformers, "Sam3Processor"),
        "checkpoint": "facebook/sam3",
    }
    out["sam_hq"] = {
        "available": hasattr(transformers, "SamModel") and hasattr(transformers, "SamProcessor"),
        "checkpoint": "syscv-community/sam-hq-vit-base",
    }
    return out


def build_segmentation_observation_from_mask(
    obs: NormalizedPerceptionObservation,
    mask_2d: np.ndarray,
    *,
    model_id: str,
    mask_index: int = 0,
) -> tuple[SegmentationObservation | None, str | None]:
    """Validate mask dimensions against observation, score quality, and build ``SegmentationObservation``."""
    arr = np.asarray(mask_2d, dtype=bool)
    expected_shape = (obs.coords.image_height, obs.coords.image_width)
    if arr.ndim != 2 or arr.shape != expected_shape:
        return None, f"dimension_mismatch:{arr.shape}!={expected_shape}"

    area_px = int(arr.sum())
    if area_px == 0:
        return None, f"empty_mask:{obs.observation_id}"

    box_iou, solidity, hole_count, boundary_unc, q_score = _compute_mask_quality(
        arr,
        obs.bbox_px,
        is_architectural=(obs.category == "architecture"),
    )
    rle = _encode_rle_binary(arr)
    mask_kind = "architectural_surface_mask" if obs.category == "architecture" else "visible_object_mask"
    seg = SegmentationObservation(
        mask_id=f"mask_{obs.observation_id}_{mask_index}",
        observation_id=obs.observation_id,
        mask_kind=mask_kind,
        area_px=area_px,
        box_iou=box_iou,
        solidity=solidity,
        hole_count=hole_count,
        boundary_uncertainty=boundary_unc,
        quality_score=q_score,
        rle_mask=rle,
        model_id=model_id,
    )
    return seg, None


def attach_masks_to_observations(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    images_rgb: Mapping[str, np.ndarray] | None = None,
    image_paths: Mapping[str, Path | str] | None = None,
    precomputed_masks: Mapping[str, np.ndarray] | None = None,
    preferred_backend: str = "sam2",
    local_files_only: bool = True,
) -> tuple[list[NormalizedPerceptionObservation], SegmentationExecutionReport]:
    """Run SAM2/SAM3/SAM-HQ (or attach precomputed masks) with strict validation and zero fabrication."""
    t0 = time.perf_counter()
    if not observations:
        return [], SegmentationExecutionReport(
            segmenter_id=preferred_backend,
            checkpoint_id="none",
            status="skipped",
            total_prompts=0,
            masks_produced=0,
            empty_masks=0,
            rejected_masks=0,
            mean_mask_quality=0.0,
            mean_mask_coverage=0.0,
            wall_time_s=0.0,
            peak_vram_mb=None,
            diagnostics=("no_observations_to_segment",),
        )

    # Path 1: Caller supplied precomputed binary masks keyed by observation_id
    if precomputed_masks is not None:
        updated: list[NormalizedPerceptionObservation] = []
        diags: list[str] = []
        empty_cnt = 0
        rejected_cnt = 0
        quals: list[float] = []
        covs: list[float] = []
        for idx, o in enumerate(observations):
            m = precomputed_masks.get(o.observation_id)
            if m is None:
                updated.append(o)
                continue
            seg, err = build_segmentation_observation_from_mask(
                o, m, model_id=f"{preferred_backend}_precomputed", mask_index=idx
            )
            if seg is None:
                if err and err.startswith("empty_mask"):
                    empty_cnt += 1
                else:
                    rejected_cnt += 1
                if err:
                    diags.append(err)
                updated.append(o)
                continue
            quals.append(seg.quality_score)
            covs.append(seg.area_px / float(max(1, o.coords.image_width * o.coords.image_height)))
            updated.append(replace(o, segmentation=seg))
        return updated, SegmentationExecutionReport(
            segmenter_id=preferred_backend,
            checkpoint_id="precomputed",
            status="succeeded" if quals else ("failed" if diags else "partial"),
            total_prompts=len(observations),
            masks_produced=len(quals),
            empty_masks=empty_cnt,
            rejected_masks=rejected_cnt,
            mean_mask_quality=round(float(np.mean(quals)), 4) if quals else 0.0,
            mean_mask_coverage=round(float(np.mean(covs)), 4) if covs else 0.0,
            wall_time_s=round(time.perf_counter() - t0, 4),
            peak_vram_mb=None,
            diagnostics=tuple(diags[:20]),
        )

    # Path 2: Live GPU/CPU execution via SAM2 / SAM3 / SAM-HQ
    probes = probe_available_segmenters()
    selected = preferred_backend if probes.get(preferred_backend, {}).get("available") else None
    if selected is None:
        for cand in ("sam2", "sam_hq", "sam3"):
            if probes.get(cand, {}).get("available"):
                selected = cand
                break

    if selected is None:
        return list(observations), SegmentationExecutionReport(
            segmenter_id=preferred_backend,
            checkpoint_id="none",
            status="unavailable",
            total_prompts=len(observations),
            masks_produced=0,
            empty_masks=0,
            rejected_masks=0,
            mean_mask_quality=0.0,
            mean_mask_coverage=0.0,
            wall_time_s=round(time.perf_counter() - t0, 4),
            peak_vram_mb=None,
            diagnostics=("no_segmentation_backend_available",),
        )

    import PIL.Image as PILImage
    import torch
    import transformers

    ckpt = probes[selected]["checkpoint"]
    diags = []
    model = None
    processor = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if selected == "sam2":
            processor = transformers.Sam2Processor.from_pretrained(ckpt, local_files_only=local_files_only)
            model = transformers.Sam2Model.from_pretrained(ckpt, local_files_only=local_files_only)
        elif selected == "sam3":
            processor = transformers.Sam3Processor.from_pretrained(ckpt, local_files_only=local_files_only)
            model = transformers.Sam3Model.from_pretrained(ckpt, local_files_only=local_files_only)
        else:
            processor = transformers.SamProcessor.from_pretrained(ckpt, local_files_only=local_files_only)
            model = transformers.SamModel.from_pretrained(ckpt, local_files_only=local_files_only)
        model = model.to(device).eval()
    except Exception as exc:
        return list(observations), SegmentationExecutionReport(
            segmenter_id=selected,
            checkpoint_id=ckpt,
            status="failed",
            total_prompts=len(observations),
            masks_produced=0,
            empty_masks=0,
            rejected_masks=0,
            mean_mask_quality=0.0,
            mean_mask_coverage=0.0,
            wall_time_s=round(time.perf_counter() - t0, 4),
            peak_vram_mb=None,
            diagnostics=(f"model_load_failed:{type(exc).__name__}:{exc}",),
        )

    by_img: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for o in observations:
        by_img[o.provenance.source_image_id].append(o)

    seg_by_obs_id: dict[str, SegmentationObservation] = {}
    empty_cnt = 0
    rejected_cnt = 0
    quals = []
    covs = []

    try:
        for img_id, obs_list in by_img.items():
            pil_img: PILImage.Image | None = None
            if images_rgb is not None and img_id in images_rgb:
                pil_img = PILImage.fromarray(np.asarray(images_rgb[img_id], dtype=np.uint8))
            elif image_paths is not None and img_id in image_paths:
                pil_img = PILImage.open(image_paths[img_id]).convert("RGB")
            elif Path(img_id).exists():
                pil_img = PILImage.open(img_id).convert("RGB")

            if pil_img is None:
                diags.append(f"missing_image_source:{img_id}")
                continue

            boxes = [[list(o.bbox_px) for o in obs_list]]
            inputs = processor(images=pil_img, input_boxes=boxes, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])
            iou_scores = getattr(outputs, "iou_scores", None)

            if not masks:
                diags.append(f"no_masks_returned:{img_id}")
                continue
            mask_list = masks[0]
            for j, o in enumerate(obs_list):
                if j >= len(mask_list):
                    diags.append(f"missing_mask_for_box:{o.observation_id}")
                    continue
                mask_2d = _select_best_2d_mask(mask_list[j], iou_scores=iou_scores, box_idx=j)
                seg, err = build_segmentation_observation_from_mask(
                    o, mask_2d, model_id=ckpt, mask_index=j
                )
                if seg is None:
                    if err and err.startswith("empty_mask"):
                        empty_cnt += 1
                    else:
                        rejected_cnt += 1
                    if err:
                        diags.append(err)
                    continue
                seg_by_obs_id[o.observation_id] = seg
                quals.append(seg.quality_score)
                covs.append(seg.area_px / float(max(1, o.coords.image_width * o.coords.image_height)))
    finally:
        peak_vram = (
            round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
            if torch.cuda.is_available()
            else None
        )
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    updated = [
        replace(o, segmentation=seg_by_obs_id[o.observation_id])
        if o.observation_id in seg_by_obs_id
        else o
        for o in observations
    ]
    status = "succeeded" if len(seg_by_obs_id) == len(observations) else ("partial" if seg_by_obs_id else "failed")
    return updated, SegmentationExecutionReport(
        segmenter_id=selected,
        checkpoint_id=ckpt,
        status=status,
        total_prompts=len(observations),
        masks_produced=len(seg_by_obs_id),
        empty_masks=empty_cnt,
        rejected_masks=rejected_cnt,
        mean_mask_quality=round(float(np.mean(quals)), 4) if quals else 0.0,
        mean_mask_coverage=round(float(np.mean(covs)), 4) if covs else 0.0,
        wall_time_s=round(time.perf_counter() - t0, 4),
        peak_vram_mb=peak_vram,
        diagnostics=tuple(diags[:20]),
    )


__all__ = [
    "SegmentationExecutionReport",
    "attach_masks_to_observations",
    "build_segmentation_observation_from_mask",
    "probe_available_segmenters",
]
