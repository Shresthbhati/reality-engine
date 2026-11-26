"""Detection engine adapter for Reality Engine's multi-engine platform.

Wraps Grounding DINO into the UnifiedEngineOutput contract.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.engine_orchestrator import EngineSpec, register_engine

log = logging.getLogger("reality.detection_adapter")


def _grounding_dino_available() -> str | None:
    try:
        from importlib import import_module

        transformers = import_module("transformers")
        getattr(transformers, "AutoProcessor")
        getattr(transformers, "AutoModelForZeroShotObjectDetection")
        return None
    except (ImportError, AttributeError) as exc:
        return str(exc)


DEFAULT_SURVEY_QUERIES: tuple[str, ...] = (
    "wall", "floor", "ceiling", "door", "door frame", "window", "window frame",
    "window sill", "opening", "column", "beam", "stairs", "railing",
    "table", "chair", "sofa", "bed", "cabinet", "wardrobe", "shelf",
    "desk", "sink", "toilet", "refrigerator", "lamp",
)

_ARCH_LABELS = frozenset({
    "wall", "floor", "ceiling", "door", "door frame", "window", "window frame",
    "window sill", "opening", "column", "beam", "stairs", "railing", "archway",
    "doorway", "structural opening",
})


def _clean_label(raw_label: str) -> str:
    s = str(raw_label or "").strip().lower()
    if not s or s.startswith("##") or not any(ch.isalpha() for ch in s):
        return ""
    return s


def _grounding_dino_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    t0 = time.perf_counter()
    run_id = ctx.get("run_id", "detection_run")
    obs: list[ObservationRecord] = []
    model_name = "IDEA-Research/grounding-dino-tiny"
    queries = list(ctx.get("queries") or DEFAULT_SURVEY_QUERIES)
    threshold = float(ctx.get("threshold", 0.30))

    try:
        try:
            processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name, local_files_only=True)
        except Exception:
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

    chunk_size = 10
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            for i in range(0, len(queries), chunk_size):
                chunk = queries[i:i + chunk_size]
                text = " . ".join(str(q).strip().rstrip(".") + "." for q in chunk)
                inputs = processor(images=image, text=text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"], threshold=threshold,
                    target_sizes=[image.size[::-1]],
                )
                r = results[0] if results else {}
                boxes = r.get("boxes", [])
                scores = r.get("scores", [])
                text_labels = r.get("text_labels", r.get("labels", []))
                for idx in range(len(boxes)):
                    box = boxes[idx]
                    if len(box) < 4:
                        continue
                    raw_lbl = str(text_labels[idx]) if idx < len(text_labels) else ""
                    label = _clean_label(raw_lbl)
                    if not label:
                        continue
                    score = float(scores[idx]) if idx < len(scores) else None
                    obs.append(ObservationRecord(
                        evidence_id=f"gdn:{img_path}:{len(obs)}",
                        source_image_id=img_path,
                        coordinate_frame=CoordinateFrame(
                            frame_id="image", scale_state="image", units="pixels"),
                        model_id=ModelIdentity(engine_id="detection", architecture_id="grounding_dino",
                                               checkpoint_id=model_name, license="Apache-2.0", hardware_class="gpu"),
                        label=label,
                        category="architecture" if label in _ARCH_LABELS else "object",
                        bbox_px=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        confidence=score,
                        calibration_quality="unverified",
                        metadata={
                            "model": model_name,
                            "box_threshold": threshold,
                            "raw_label": raw_lbl,
                            "raw_score": score,
                            "calibrated_probability": None,
                            "is_calibrated": False,
                            "uncertainty": round(1.0 - score, 4) if score is not None else None,
                        },
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


def _get_vram_peak() -> float | None:
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
