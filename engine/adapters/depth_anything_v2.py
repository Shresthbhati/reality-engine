"""Depth Anything V2 Monocular Depth Adapter.

Uses Depth Anything V2 Small via HuggingFace transformers for fine geometric
depth boundaries and high-frequency edge preservation.
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


class DepthAnythingV2Adapter(BaseEngineAdapter):
    """Adapter for Depth Anything V2 Small monocular depth estimation."""

    DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        super().__init__(
            engine_name="depth_anything_v2",
            task_category="depth_estimation",
            licence="Apache-2.0",
            device=device,
        )
        self.checkpoint = checkpoint
        self._processor = None

    def verify_dependencies(self) -> tuple[bool, str]:
        import importlib.util
        missing = [pkg for pkg in ("torch", "transformers", "PIL") if importlib.util.find_spec(pkg) is None]
        if missing:
            return False, f"Missing dependencies: {', '.join(missing)}"
        return True, "torch, transformers, and PIL available"

    def load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        dev = self._custom_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint)
            self._model = model.to(dev).eval()
        except Exception as e:
            raise DependencyMissingError(f"Failed to load Depth Anything V2 ({self.checkpoint}): {e}") from e

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

        # Load images
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

        depth_results = []
        raw_depth_maps = []

        with torch.no_grad():
            for idx, img in enumerate(pil_images):
                w, h = img.size
                inputs_tensor = self._processor(images=img, return_tensors="pt").to(dev)
                outputs = self._model(**inputs_tensor)
                predicted_depth = outputs.predicted_depth

                # Interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth_np = prediction.cpu().numpy().astype(np.float32)

                # Depth statistics for calibration and boundary sharpness
                min_val = float(np.min(depth_np))
                max_val = float(np.max(depth_np))
                mean_val = float(np.mean(depth_np))
                std_val = float(np.std(depth_np))

                # Compute depth gradient magnitude for edge sharpness evaluation
                gy, gx = np.gradient(depth_np)
                grad_mag = np.sqrt(gx**2 + gy**2)
                mean_edge_grad = float(np.mean(grad_mag))

                depth_results.append({
                    "image_index": idx,
                    "width": w,
                    "height": h,
                    "min_depth": min_val,
                    "max_depth": max_val,
                    "mean_depth": mean_val,
                    "std_depth": std_val,
                    "edge_gradient_mean": mean_edge_grad,
                    "is_metric": False,  # Relative inverse depth
                    "depth_array": depth_np,
                })
                raw_depth_maps.append(depth_np)

        diagnostics = {
            "num_images": len(pil_images),
            "checkpoint": self.checkpoint,
            "mean_edge_sharpness": float(np.mean([r["edge_gradient_mean"] for r in depth_results])),
        }

        return depth_results, raw_depth_maps, diagnostics
