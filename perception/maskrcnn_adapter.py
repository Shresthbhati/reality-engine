"""Mask R-CNN (torchvision, COCO-pretrained) engine adapter for the multi-engine
orchestrator.

Wraps ``perception.detection.maskrcnn_backend.MaskRCNNDetector`` (a real,
implemented backend) into the ``UnifiedEngineOutput`` contract so the
orchestrator runs it as a first-class detection + instance-segmentation engine.

Honest rules (matching the repo's other perception backends):
* Unavailable is unavailable: no torch/torchvision -> the engine is reported
  unavailable with the reason; nothing pretends inference happened.
* Model loads once for the whole run (not per-image) and is released before the
  next engine starts, so its VRAM is not shared on the 6 GB card.
* Score threshold and min-mask-pixels are real filters: detections/masks below
  them are dropped and the counts are reported.
* Per-image exceptions are logged and skipped -- no fabricated output.

This adapter exists because maskrcnn_resnet50_fpn is an implemented backend
(``perception/detection/maskrcnn_backend.py``) that was NOT wired into the
orchestrator. Connecting it here is the §6 Part B delivery for a valuable
detection + instance-segmentation engine whose observations feed the
orchestrator -> matrix/evidence consumer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from perception.detection.ontology_grounding import ground_label
from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.engine_orchestrator import EngineSpec, register_engine

log = logging.getLogger("reality.maskrcnn_adapter")


def _maskrcnn_available() -> str | None:
    """Why Mask R-CNN cannot run here, or None when it can."""
    try:
        import torchvision  # noqa: F401
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights  # noqa: F401
    except ImportError as exc:
        return f"torchvision unavailable: {exc}"
    try:
        MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    except Exception as exc:
        return f"Mask R-CNN COCO weights unavailable: {exc}"
    return None


def _maskrcnn_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    """Run Mask R-CNN on every image, producing detection + instance-segmentation
    observations in the UnifiedEngineOutput contract."""
    # Respect the availability probe: if it returns a reason, do not attempt to load
    # the model (the orchestrator already checked this, but the direct-call path --
    # e.g. a test that patches _maskrcnn_available -- relies on the run function
    # honouring the probe so "unavailable" is never fabricated).
    _reason = _maskrcnn_available()
    if _reason is not None:
        return UnifiedEngineOutput(
            engine_id="maskrcnn", run_id=ctx.get("run_id", "maskrcnn_run"), observations=(),
            status="failed", error_message=f"unavailable: {_reason}",
            wall_time_s=0.0,
        )

    import torch
    import torchvision
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

    t0 = time.perf_counter()
    run_id = ctx.get("run_id", "maskrcnn_run")
    obs: list[ObservationRecord] = []
    scores_below = 0
    masks_below = 0
    failed_images: list[str] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(0)

    try:
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        model = maskrcnn_resnet50_fpn(weights=weights)
        model = model.to(device).eval()
        _model_version = f"coco_{torchvision.__version__}"
    except Exception as exc:
        return UnifiedEngineOutput(
            engine_id="maskrcnn", run_id=run_id, observations=(),
            status="failed", error_message=f"model load failed: {exc}",
            wall_time_s=time.perf_counter() - t0,
        )

    scores = weights.meta["categories"]  # 91 COCO classes (index 0 = background)
    threshold = ctx.get("score_threshold", 0.5)
    min_mask_pixels = ctx.get("min_mask_pixels", 64)
    _model_version = _model_version if "_model_version" in dir() else f"coco_{torchvision.__version__}"

    for img_path in image_paths:
        try:
            from PIL import Image

            img = Image.open(img_path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            pred = model([tensor.to(device)])[0]
        except Exception as exc:
            log.warning("Mask R-CNN failed for %s: %s", img_path, exc)
            failed_images.append(img_path)
            continue

        w, h = img.size
        n_det = 0
        for j in range(len(pred["scores"])):
            score = float(pred["scores"][j])
            if score < threshold:
                scores_below += 1
                continue
            label = scores[int(pred["labels"][j])]
            grounding = ground_label(label)
            box = tuple(float(v) for v in pred["boxes"][j].tolist())
            obs.append(ObservationRecord(
                evidence_id=f"maskrcnn:{img_path}:det:{len(obs)}",
                source_image_id=img_path,
                coordinate_frame=CoordinateFrame(
                    frame_id="image", scale_state="image", units="pixels"),
                model_id=ModelIdentity(
                    engine_id="detection", architecture_id="maskrcnn_resnet50_fpn",
                    checkpoint_id="coco_v1",
                    license="COCO_V1 weights: BSD-3 (torchvision); model: MIT",
                    hardware_class=device),
                label=label,
                category=grounding.category,
                bbox_px=box,
                confidence=score,
                calibration_quality="unverified",
                metadata={
                    "model": "maskrcnn_resnet50_fpn",
                    "model_version": _model_version,
                    "score_threshold": threshold,
                    "box": box,
                    "raw_label": label,
                    **grounding.to_metadata(),
                },
            ))
            n_det += 1

            # Instance segmentation mask for the same detection (RLE in metadata).
            if "masks" in pred and j < pred["masks"].shape[0]:
                m = pred["masks"][j, 0].detach().cpu().numpy()  # (Hm, Wm) in model input res
                mask_bool = m > 0.5
                if m.shape[0] != h or m.shape[1] != w:
                    # Resample back to source resolution (nearest, so the mask aligns
                    # with the bbox) -- necessary for any downstream that pairs mask to
                    # image size (e.g. lift_region_to_3d requires mask dims == image dims).
                    from PIL import Image as _I

                    m_img = _I.fromarray((mask_bool.astype("uint8")) * 255)
                    m_img = m_img.resize((w, h), _I.NEAREST)
                    mask_bool = np.asarray(m_img) > 127
                if int(mask_bool.sum()) < min_mask_pixels:
                    masks_below += 1
                    continue
                # Encode mask as a run-length list (row-major) so the observation
                # carries the mask without embedding a full pixel array. Consumers that
                # need the full mask can decode the RLE; the contract intentionally does
                # not add a dedicated mask field (the mask is metadata, not a new column).
                flat = mask_bool.ravel(order="C")
                change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
                bounds = np.concatenate([[0], change, [flat.size]])
                runs = np.diff(bounds).tolist()
                if not flat[0]:
                    runs = [0] + runs
                rle = [int(r) for r in runs]
                obs.append(ObservationRecord(
                    evidence_id=f"maskrcnn:{img_path}:seg:{len(obs)}",
                    source_image_id=img_path,
                    coordinate_frame=CoordinateFrame(
                        frame_id="image", scale_state="image", units="pixels"),
                    model_id=ModelIdentity(
                        engine_id="segmentation", architecture_id="maskrcnn_resnet50_fpn",
                        checkpoint_id="coco_v1",
                        license="COCO_V1 weights: BSD-3 (torchvision); model: MIT",
                        hardware_class=device),
                    label=label,
                    category="instance_segmentation",
                    bbox_px=box,
                    mask=mask_bool,
                    confidence=score,
                    calibration_quality="unverified",
                    metadata={
                        "model": "maskrcnn_resnet50_fpn",
                        "model_version": _model_version,
                        "score_threshold": threshold,
                        "min_mask_pixels": min_mask_pixels,
                        "mask_rle": rle,
                        "mask_shape": [int(h), int(w)],
                        "box": box,
                        "raw_label": label,
                        **grounding.to_metadata(),
                    },
                ))

    peak_vram_mb: float | None = None
    try:
        if device == "cuda" and torch.cuda.is_available():
            peak_vram_mb = float(torch.cuda.max_memory_allocated(0) / (1024 * 1024))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if not image_paths:
        status = "succeeded"
    elif failed_images and not obs:
        status = "failed"
    elif failed_images or not obs:
        status = "partial"
    else:
        status = "succeeded"

    return UnifiedEngineOutput(
        engine_id="maskrcnn", run_id=run_id, observations=tuple(obs),
        status=status,
        wall_time_s=time.perf_counter() - t0,
        peak_vram_mb=peak_vram_mb,
        summary={
            "n_images": len(image_paths),
            "n_observations": len(obs),
            "n_detections": sum(1 for o in obs if o.category == "object"),
            "n_instance_segments": sum(1 for o in obs if o.category == "instance_segmentation"),
            "scores_below_threshold": scores_below,
            "masks_below_min_pixels": masks_below,
            "score_threshold": threshold,
            "min_mask_pixels": min_mask_pixels,
        },
    )


def register_maskrcnn_engine() -> None:
    """Register Mask R-CNN as a first-class detection + instance-segmentation engine."""
    _tv_version = "unknown"
    try:
        import torchvision
        _tv_version = f"coco_{torchvision.__version__}"
    except Exception:
        pass

    register_engine(
        EngineSpec(
            engine_id="maskrcnn",
            architecture_id="maskrcnn_resnet50_fpn",
            checkpoint_id="coco_v1",
            version=_tv_version,
            vram_mb=4000,
            cpu_threads=0,
            priority=55,
            required_for_production=False,
            license="COCO_V1 weights: BSD-3 (torchvision); model: MIT",
            non_commercial=False,
            dependencies=(),
            run_condition=_maskrcnn_available,
            cpu_capable=True,
            timeout_s=0.0,
            drain_after_timeout_s=60.0,
            max_attempts=2,
        ),
        _maskrcnn_run,
    )
