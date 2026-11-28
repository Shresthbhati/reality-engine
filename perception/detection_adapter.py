"""Detection engine adapter for Reality Engine's multi-engine platform.

Wraps Grounding DINO into the UnifiedEngineOutput contract.

Detection is ontology-grounded (``perception.detection.ontology_grounding``):

* the prompt set is derived from the canonical ontology with a reserved budget
  for small classes, so sockets, switches, door handles, hinges and vents are
  actually queried instead of being crowded out by furniture and structure;
* every emitted label is resolved to a canonical ontology path and the
  resolution (or its honest absence) is recorded in observation metadata;
* small objects can be detected at native resolution via tiled inference, which
  is what stops a 40 px fixture becoming sub-pixel when a 4000 px photograph is
  downscaled to the detector's input.

Scores are the model's own; nothing here rescales, calibrates or invents a
confidence. ``calibrated_probability`` stays ``None`` and ``is_calibrated``
stays ``False``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from perception.detection.ontology_grounding import (
    FALLBACK_QUERIES,
    ground_label,
    nms,
    production_queries,
    tile_windows,
)
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


#: Historical hand-written survey list. Retained as the exported fallback that
#: ``perception.intelligence.production_pipeline_a5`` imports and uses when its
#: own survey generator yields nothing; the adapter's *default* is now the
#: ontology-derived :func:`production_queries`.
DEFAULT_SURVEY_QUERIES: tuple[str, ...] = (
    "wall", "floor", "ceiling", "door", "door frame", "window", "window frame",
    "window sill", "opening", "column", "beam", "stairs", "railing",
    "table", "chair", "sofa", "bed", "cabinet", "wardrobe", "shelf",
    "desk", "sink", "toilet", "refrigerator", "lamp",
)

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def _clean_label(raw_label: str) -> str:
    s = str(raw_label or "").strip().lower()
    if not s or s.startswith("##") or not any(ch.isalpha() for ch in s):
        return ""
    return s


def _load_detector(model_name: str = DEFAULT_MODEL) -> tuple[Any, Any, str]:
    """Load processor + model once. Returns ``(processor, model, device)``.

    Isolated so tests can substitute a deterministic stub without a GPU and
    without pretending a real model ran.
    """
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return processor, model, device


def _detect_in_image(
    processor: Any,
    model: Any,
    image: Image.Image,
    queries: Sequence[str],
    threshold: float,
    device: str,
    chunk_size: int = 10,
) -> list[tuple[tuple[float, float, float, float], float | None, str]]:
    """One full-image detection pass over a chunked query list.

    Returns ``(box_px, score, raw_label)`` in the image's own pixel coordinates.
    """
    out: list[tuple[tuple[float, float, float, float], float | None, str]] = []
    queries = [str(q) for q in queries if str(q).strip()]
    for i in range(0, len(queries), chunk_size):
        chunk = queries[i:i + chunk_size]
        text = " . ".join(q.strip().rstrip(".") + "." for q in chunk)
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
            out.append((
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score,
                label,
            ))
    return out


def _detect_tiled(
    processor: Any,
    model: Any,
    image: Image.Image,
    queries: Sequence[str],
    threshold: float,
    device: str,
    tile_scale: int,
    tile_overlap: float,
    chunk_size: int = 10,
) -> tuple[list[tuple[tuple[float, float, float, float], float | None, str]], int]:
    """Detect on native-resolution tiles, map boxes back to full image, then NMS.

    Returns ``(detections, n_tiles)``. Detections carry full-image pixel boxes,
    so the caller treats tiled and untiled output identically.
    """
    w, h = image.size
    windows = tile_windows(w, h, scale=tile_scale, overlap=tile_overlap)
    merged: list[tuple[tuple[float, float, float, float], float | None, str]] = []
    for (x0, y0, x1, y1) in windows:
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image.crop((x0, y0, x1, y1))
        for (bx0, by0, bx1, by1), score, label in _detect_in_image(
            processor, model, crop, queries, threshold, device, chunk_size
        ):
            merged.append(((bx0 + x0, by0 + y0, bx1 + x0, by1 + y0), score, label))

    if not merged:
        return [], len(windows)

    import inspect

    boxes_arg = [m[0] for m in merged]
    scores_arg = [m[1] if m[1] is not None else 0.0 for m in merged]
    if "labels" in inspect.signature(nms).parameters:
        keep = nms(boxes_arg, scores_arg, labels=[m[2] for m in merged])
    else:
        keep = nms(boxes_arg, scores_arg)
    return [merged[i] for i in keep], len(windows)


def _grounding_dino_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    t0 = time.perf_counter()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(0)
        except Exception:
            pass
    run_id = ctx.get("run_id", "detection_run")
    obs: list[ObservationRecord] = []
    model_name = str(ctx.get("model_name") or DEFAULT_MODEL)
    threshold = float(ctx.get("threshold", 0.30))
    domain = str(ctx.get("domain", "all"))
    tile_scale = int(ctx.get("tile_scale", 0) or 0)
    tile_overlap = float(ctx.get("tile_overlap", 0.2))

    explicit_queries = ctx.get("queries")
    if explicit_queries:
        queries = [str(q) for q in explicit_queries]
        query_source = "caller"
    else:
        queries = production_queries(
            domain=domain,
            budget=int(ctx.get("query_budget", 40)),
            small_share=float(ctx.get("small_share", 0.4)),
        )
        query_source = "ontology"
        if not queries:
            queries = list(FALLBACK_QUERIES)
            query_source = "fallback"

    try:
        processor, model, device = _load_detector(model_name)
    except Exception as exc:
        return UnifiedEngineOutput(
            engine_id="grounding_dino", run_id=run_id, observations=(),
            status="failed", error_message=f"model load failed: {exc}",
            wall_time_s=time.perf_counter() - t0,
        )

    skipped_blank: list[str] = []
    failed_images: list[dict[str, str]] = []
    n_unresolved = 0
    tiles_total = 0
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            import numpy as np

            arr = np.asarray(image)
            if arr.size == 0 or float(np.std(arr, axis=(0, 1)).max()) < 2.0:
                skipped_blank.append(img_path)
                continue

            if tile_scale > 1:
                detections, n_tiles = _detect_tiled(
                    processor, model, image, queries, threshold, device,
                    tile_scale, tile_overlap,
                )
                tiles_total += n_tiles
            else:
                detections = _detect_in_image(
                    processor, model, image, queries, threshold, device
                )

            for box, score, label in detections:
                grounding = ground_label(label)
                if not grounding.resolved:
                    n_unresolved += 1
                obs.append(ObservationRecord(
                    evidence_id=f"gdn:{img_path}:{len(obs)}",
                    source_image_id=img_path,
                    coordinate_frame=CoordinateFrame(
                        frame_id="image", scale_state="image", units="pixels"),
                    model_id=ModelIdentity(
                        engine_id="detection", architecture_id="grounding_dino",
                        checkpoint_id=model_name, license="Apache-2.0",
                        hardware_class=device),
                    label=label,
                    category=grounding.category,
                    bbox_px=box,
                    confidence=score,
                    calibration_quality="unverified",
                    metadata={
                        "model": model_name,
                        "box_threshold": threshold,
                        "raw_label": label,
                        "raw_score": score,
                        "calibrated_probability": None,
                        "is_calibrated": False,
                        "uncertainty": round(1.0 - score, 4) if score is not None else None,
                        **grounding.to_metadata(),
                    },
                ))
        except Exception as exc:
            log.warning("Grounding DINO failed for %s: %s", img_path, exc)
            failed_images.append({"image": str(img_path), "error": f"{type(exc).__name__}: {exc}"})

    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    n_small = sum(1 for o in obs if bool(o.metadata.get("is_small")))
    if failed_images and not obs:
        status = "failed"
    elif failed_images or (not obs and skipped_blank):
        status = "partial"
    elif obs:
        status = "succeeded"
    else:
        status = "partial"
    return UnifiedEngineOutput(
        engine_id="grounding_dino", run_id=run_id, observations=tuple(obs),
        status=status,
        wall_time_s=time.perf_counter() - t0,
        peak_vram_mb=_get_vram_peak(),
        summary={
            "n_images": len(image_paths),
            "n_detections": len(obs),
            "n_small_detections": n_small,
            "n_unresolved_labels": n_unresolved,
            "skipped_blank_images": skipped_blank,
            "failed_images": failed_images,
            "n_queries": len(queries),
            "query_source": query_source,
            "domain": domain,
            "tile_scale": tile_scale,
            "tiles_processed": tiles_total,
        },
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


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SURVEY_QUERIES",
    "register_detection_engine",
    "_grounding_dino_run",
]
