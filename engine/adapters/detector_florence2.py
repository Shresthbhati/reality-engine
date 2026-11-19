"""Florence-2 Multi-Task Vision-Language Detector Adapter.

Uses Microsoft's Florence-2 architecture via HuggingFace transformers for
unified dense object detection, phrase grounding, and detailed image captioning.
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
from perception.detection.extensible_prompter import get_extensible_prompter


class Florence2DetectorAdapter(BaseEngineAdapter):
    """Adapter for Florence-2 multi-task vision detector."""

    DEFAULT_CHECKPOINT = "microsoft/Florence-2-base"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        super().__init__(
            engine_name="florence2_detector",
            task_category="open_vocabulary_detection",
            licence="MIT",
            device=device,
        )
        self.checkpoint = checkpoint
        self._processor = None
        self._prompter = get_extensible_prompter()

    def verify_dependencies(self) -> tuple[bool, str]:
        import importlib.util
        missing = [pkg for pkg in ("torch", "transformers") if importlib.util.find_spec(pkg) is None]
        if missing:
            return False, f"Missing dependencies for Florence-2: {', '.join(missing)}"
        return True, "torch and transformers available"

    def load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        dev = self._custom_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self._processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.checkpoint, trust_remote_code=True)
            self._model = model.to(dev).eval()
        except Exception as e:
            raise DependencyMissingError(f"Failed to load Florence-2 ({self.checkpoint}): {e}") from e

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
        task_prompt = options.get("task", "<OD>")  # Object Detection

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
                    text=task_prompt,
                    images=img,
                    return_tensors="pt",
                ).to(dev)

                generated_ids = self._model.generate(
                    input_ids=inputs_tensor["input_ids"],
                    pixel_values=inputs_tensor["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
                generated_text = self._processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]

                parsed_results = self._processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(w, h),
                )

                od_result = parsed_results.get(task_prompt, {})
                bboxes = od_result.get("bboxes", [])
                labels = od_result.get("labels", [])

                image_detections = []
                for box, label in zip(bboxes, labels):
                    clean_label = label.strip()
                    target = self._prompter.resolve_prediction(clean_label)
                    canonical_id = target.path if target else clean_label

                    x1, y1, x2, y2 = box
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)

                    det = {
                        "label": clean_label,
                        "canonical_path": canonical_id,
                        "score": 1.0,  # Florence-2 OD task does not emit scalar confidence scores
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
                raw_predictions.append(parsed_results)

        diagnostics = {
            "checkpoint": self.checkpoint,
            "task_prompt": task_prompt,
            "total_detections": sum(r["detection_count"] for r in detection_results),
        }

        return detection_results, raw_predictions, diagnostics
