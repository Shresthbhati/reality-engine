"""Grounding DINO Open-Vocabulary Object Detector Adapter.

Uses Grounding DINO (Tiny tier) via HuggingFace transformers for fine-grained
open-vocabulary phrase grounding and bounding box prediction.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image

from engine.adapters.base import (
    BaseEngineAdapter,
    DependencyMissingError,
    InputValidationError,
)
from perception.detection.extensible_prompter import (
    DomainFilter,
    ScaleFilter,
    get_extensible_prompter,
)


class GroundingDinoDetectorAdapter(BaseEngineAdapter):
    """Adapter for Grounding DINO open-vocabulary detector."""

    DEFAULT_CHECKPOINT = "IDEA-Research/grounding-dino-tiny"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        super().__init__(
            engine_name="grounding_dino_detector",
            task_category="open_vocabulary_detection",
            licence="Apache-2.0",
            device=device,
        )
        self.checkpoint = checkpoint
        self._processor = None
        self._prompter = get_extensible_prompter()

    def verify_dependencies(self) -> tuple[bool, str]:
        import importlib.util
        missing = [pkg for pkg in ("torch", "transformers") if importlib.util.find_spec(pkg) is None]
        if missing:
            return False, f"Missing dependencies for Grounding DINO: {', '.join(missing)}"
        return True, "torch and transformers with GroundingDINO available"

    def load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        dev = self._custom_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self._processor = AutoProcessor.from_pretrained(self.checkpoint)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(self.checkpoint)
            self._model = model.to(dev).eval()
        except Exception as e:
            raise DependencyMissingError(f"Failed to load Grounding DINO ({self.checkpoint}): {e}") from e

    def validate_inputs(self, inputs: Any) -> None:
        if not isinstance(inputs, (list, tuple)):
            raise InputValidationError("Inputs must be a list of images or image paths.")
        if len(inputs) == 0:
            raise InputValidationError("Inputs list cannot be empty.")

    def _execute_inference(
        self,
        inputs: Sequence[Union[str, Path, Image.Image, np.ndarray]],
        options: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
        import torch

        dev = self._custom_device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        queries = options.get("queries")
        if not queries:
            domain_filter = DomainFilter(options.get("domain", "all"))
            scale_filter = ScaleFilter(options.get("scale", "all"))
            queries = self._prompter.get_prompts(domain=domain_filter, scale=scale_filter)

        # Grounding DINO expects dot-separated phrases
        text_prompt = ". ".join(queries) + "."
        box_threshold = float(options.get("box_threshold", 0.25))
        text_threshold = float(options.get("text_threshold", 0.20))

        pil_images = []
        for inp in inputs:
            if isinstance(inp, (str, Path)):
                img = Image.open(inp).convert("RGB")
            elif isinstance(inp, np.ndarray):
                img = Image.fromarray(inp.astype(np.uint8)).convert("RGB")
            elif isinstance(inp, Image.Image):
                img = inp.convert("RGB")
            else:
                raise InputValidationError(f"Unsupported image type: {type(inp)}")
            pil_images.append(img)

        detection_results = []
        raw_predictions = []

        with torch.no_grad():
            for idx, img in enumerate(pil_images):
                w, h = img.size
                inputs_tensor = self._processor(
                    images=img,
                    text=text_prompt,
                    return_tensors="pt",
                ).to(dev)

                outputs = self._model(**inputs_tensor)
                target_sizes = [(h, w)]
                input_ids = inputs_tensor.input_ids
                results = self._processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]
                del inputs_tensor, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                labels = results["labels"]

                image_detections = []
                for box, score, label in zip(boxes, scores, labels):
                    clean_label = label.strip().rstrip(".")
                    target = self._prompter.resolve_prediction(clean_label)
                    canonical_id = target.path if target else clean_label

                    x1, y1, x2, y2 = box.tolist()
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)

                    det = {
                        "label": clean_label,
                        "canonical_path": canonical_id,
                        "score": float(score),
                        "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "box_xywh": [float(x1), float(y1), float(bw), float(bh)],
                        "normalized_box": [
                            float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)
                        ],
                        "is_small_object": bw * bh < (w * h * 0.05),
                    }
                    image_detections.append(det)

                detection_results.append({
                    "image_index": idx,
                    "width": w,
                    "height": h,
                    "detections": image_detections,
                    "detection_count": len(image_detections),
                })
                raw_predictions.append(results)

        diagnostics = {
            "checkpoint": self.checkpoint,
            "queries_evaluated": len(queries),
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "total_detections": sum(r["detection_count"] for r in detection_results),
        }

        return detection_results, raw_predictions, diagnostics
