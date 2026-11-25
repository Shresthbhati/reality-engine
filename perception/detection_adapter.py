"""Detection engine adapter for Reality Engine's multi-engine platform.

Wraps Grounding DINO into the UnifiedEngineOutput contract.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Sequence

from PIL import Image
import torch

from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.engine_orchestrator import EngineSpec, register_engine

log = logging.getLogger("reality.detection_adapter")


def _grounding_dino_available() -> Optional[str]:
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        return None
    except ImportError as exc:
        return str(exc)


def _grounding_dino_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    t0 = time.perf_counter()
    run_id = ctx.get("run_id", "detection_run")
    obs: List[ObservationRecord] = []
    model_name = "IDEA-Research/grounding-dino-tiny"
    queries = ctx.get("queries", ["wall", "door", "window", "floor", "ceiling", "furniture"])

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
    except Exception as exc:
        return UnifiedEngineOutput(
            engine_id="grounding_dino", run_id=run_id, observations=(),
            status="failed", error_message=f"model load failed: {exc}",
            wall_time_s=time.perf_counter() - t0,
        )

    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            # Combine queries into a single text prompt
            text = " . ".join(q + "." for q in queries)
            inputs = processor(images=image, text=text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=0.3,
                target_sizes=[image.size[::-1]],
            )
            r = results[0] if results else {}
            boxes = r.get("boxes", [])
            scores = r.get("scores", [])
            text_labels = r.get("text_labels", [])
            for idx in range(len(boxes)):
                box = boxes[idx]
                if len(box) < 4:
                    continue
                label = text_labels[idx] if idx < len(text_labels) else "unknown"
                score = float(scores[idx]) if idx < len(scores) else 0.5
                obs.append(ObservationRecord(
                    evidence_id=f"gdn:{img_path}:{len(obs)}",
                    source_image_id=img_path,
                    coordinate_frame=CoordinateFrame(
                        frame_id="image", scale_state="image", units="pixels"),
                    model_id=ModelIdentity(engine_id="detection", architecture_id="grounding_dino",
                                           checkpoint_id=model_name, license="Apache-2.0", hardware_class="gpu"),
                    label=label,
                    category="architecture" if label in ("wall", "floor", "ceiling", "door", "window") else "object",
                    bbox_px=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=score,
                    metadata={"model": model_name, "box_threshold": 0.3},
                ))
        except Exception as exc:
            log.warning("Grounding DINO failed for %s: %s", img_path, exc)

    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return UnifiedEngineOutput(
        engine_id="grounding_dino", run_id=run_id, observations=tuple(obs),
        status="succeeded" if obs else "partial",
        wall_time_s=time.perf_counter() - t0,
        peak_vram_mb=_get_vram_peak(),
        summary={"n_images": len(image_paths), "n_detections": len(obs)},
    )


def _get_vram_peak() -> Optional[float]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def register_detection_engine() -> None:
    register_engine(
        EngineSpec(engine_id="grounding_dino", architecture_id="grounding_dino",
                   checkpoint_id="IDEA-Research/grounding-dino-tiny",
                   vram_mb=2048, cpu_threads=0, priority=85, license="Apache-2.0",
                   dependencies=(), run_condition=_grounding_dino_available),
        _grounding_dino_run,
    )


__all__ = ["register_detection_engine", "_grounding_dino_run"]
