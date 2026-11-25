"""Open-vocabulary architectural & indoor object detector via Grounding DINO.

Uses ``IDEA-Research/grounding-dino-tiny`` natively from Hugging Face
``transformers`` (``AutoProcessor`` + ``GroundingDinoForObjectDetection``)
loaded strictly from the local cache (``local_files_only=True``).

All outputs pass through ``perception.detection.normalized.normalize_backend_output``
so downstream scene intelligence consumes a single normalized contract.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from perception.detection.normalized import (
    CoordinateConvention,
    NormalizedDetectionBatch,
    normalize_backend_output,
)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MODEL_VERSION = "tiny_hf_v1"

STRUCTURAL_VOCABULARY: Tuple[str, ...] = (
    "wall",
    "floor",
    "ceiling",
    "column",
    "beam",
    "doorway",
    "door",
    "window",
    "window frame",
    "stairs",
    "railing",
    "balcony",
    "structural opening",
)

ARCHITECTURAL_DETAIL_VOCABULARY: Tuple[str, ...] = (
    "skirting board",
    "cornice",
    "ceiling panel",
    "recess",
    "archway",
    "wall niche",
    "built-in cabinet",
    "built-in shelf",
    "wall panel",
    "ceiling fixture",
)

FURNITURE_VOCABULARY: Tuple[str, ...] = (
    "bed",
    "sofa",
    "chair",
    "table",
    "desk",
    "cabinet",
    "wardrobe",
    "shelf",
    "appliance",
    "lighting fixture",
)

ENVIRONMENTAL_VOCABULARY: Tuple[str, ...] = (
    "plant",
    "curtain",
    "rug",
    "mirror",
    "screen",
    "decorative object",
)

FULL_ARCHITECTURAL_VOCABULARY: Tuple[str, ...] = (
    STRUCTURAL_VOCABULARY
    + ARCHITECTURAL_DETAIL_VOCABULARY
    + FURNITURE_VOCABULARY
    + ENVIRONMENTAL_VOCABULARY
)


class GroundingDinoUnavailable(RuntimeError):
    """Raised when Grounding DINO checkpoint or transformers support is missing."""


class GroundingDinoDetector:
    """Local-cache Grounding DINO detector emitting `NormalizedDetectionBatch`."""

    name = "grounding_dino"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        box_threshold: float = 0.28,
        text_threshold: float = 0.25,
        chunk_size: int = 8,
    ) -> None:
        self.model_id = model_id
        self._device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.chunk_size = max(1, int(chunk_size))
        self._model = None
        self._processor = None
        self.load_time_s: float = 0.0

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        t0 = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
            model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.model_id, local_files_only=True
                )
                .to(dev)
                .eval()
            )
        except Exception as exc:  # noqa: BLE001
            raise GroundingDinoUnavailable(
                f"Grounding DINO is not available locally ({exc}); fetch once with "
                f"python -c \"from huggingface_hub import snapshot_download as d; d('{self.model_id}')\""
            ) from exc
        self._device, self._model, self._processor = dev, model, processor
        self.load_time_s = round(time.perf_counter() - t0, 4)
        return model, processor

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def detect_normalized(
        self,
        image_rgb: np.ndarray,
        queries: Sequence[str] = FULL_ARCHITECTURAL_VOCABULARY,
        *,
        image_id: str = "",
        frame_index: int | None = None,
        timestamp_s: float | None = None,
    ) -> NormalizedDetectionBatch:
        """Run Grounding DINO over `queries` (chunked) and return `NormalizedDetectionBatch`."""
        h, w = image_rgb.shape[:2]
        try:
            import torch

            model, processor = self._load()
            pil = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
            t0 = time.perf_counter()
            raw_items: List[Dict[str, Any]] = []
            cleaned_queries = [str(q).strip().lower() for q in queries if str(q).strip()]
            for i in range(0, len(cleaned_queries), self.chunk_size):
                chunk = cleaned_queries[i : i + self.chunk_size]
                prompt = ". ".join(chunk) + "."
                inputs = processor(images=pil, text=prompt, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[(h, w)],
                )[0]
                labels = results.get("text_labels", results.get("labels", []))
                for box, score, label in zip(results["boxes"], results["scores"], labels):
                    raw_items.append(
                        {
                            "label": str(label).strip() or "unknown",
                            "bbox_px": [float(v) for v in box.tolist()],
                            "score": float(score.item()),
                        }
                    )
            inf_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return normalize_backend_output(
                raw_items,
                image_width=w,
                image_height=h,
                convention=CoordinateConvention.PIXEL_XYXY,
                model_name=self.name,
                model_version=MODEL_VERSION,
                device=str(self._device),
                inference_ms=inf_ms,
                image_id=image_id,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                clip=True,
            )
        except GroundingDinoUnavailable as exc:
            return normalize_backend_output(
                None,
                image_width=w,
                image_height=h,
                model_name=self.name,
                model_version=MODEL_VERSION,
                device=str(self._device or "cpu"),
                image_id=image_id,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return normalize_backend_output(
                None,
                image_width=w,
                image_height=h,
                model_name=self.name,
                model_version=MODEL_VERSION,
                device=str(self._device or "cpu"),
                image_id=image_id,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                error=f"{type(exc).__name__}: {exc}",
            )
