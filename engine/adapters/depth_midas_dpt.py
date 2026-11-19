"""MiDaS / DPT-Hybrid Monocular Depth Adapter (Baseline).

Provides the legacy/baseline depth estimation adapter wrapping DPT_Hybrid
via PyTorch Hub for comparative evaluation against Depth Anything V2.
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


class MidasDPTAdapter(BaseEngineAdapter):
    """Adapter for MiDaS DPT_Hybrid monocular depth estimation."""

    def __init__(self, device: str | None = None) -> None:
        super().__init__(
            engine_name="midas_dpt_hybrid",
            task_category="depth_estimation",
            licence="MIT",
            device=device,
        )
        self._transform = None

    def verify_dependencies(self) -> tuple[bool, str]:
        import importlib.util
        missing = [pkg for pkg in ("torch", "torchvision") if importlib.util.find_spec(pkg) is None]
        if missing:
            return False, f"Missing dependencies for MiDaS: {', '.join(missing)}"
        return True, "torch and torchvision available"

    def load_model(self) -> None:
        if self._model is not None:
            return

        import torch

        dev = self._custom_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
            self._model = model.to(dev).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = midas_transforms.dpt_transform
        except Exception as e:
            raise DependencyMissingError(f"Failed to load MiDaS DPT_Hybrid: {e}") from e

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
        import cv2
        import torch

        dev = self._custom_device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        np_images = []
        for inp in inputs:
            if isinstance(inp, (str, Path)):
                img = cv2.imread(str(inp))
                if img is None:
                    raise InputValidationError(f"Could not read image from {inp}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(inp, Image.Image):
                img = np.array(inp.convert("RGB"))
            elif isinstance(inp, np.ndarray):
                img = inp
            else:
                raise InputValidationError(f"Unsupported image type: {type(inp)}")
            np_images.append(img)

        depth_results = []
        raw_depth_maps = []

        with torch.no_grad():
            for idx, img in enumerate(np_images):
                h, w = img.shape[:2]
                input_batch = self._transform(img).to(dev)
                prediction = self._model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth_np = prediction.cpu().numpy().astype(np.float32)

                gy, gx = np.gradient(depth_np)
                grad_mag = np.sqrt(gx**2 + gy**2)

                depth_results.append({
                    "image_index": idx,
                    "width": w,
                    "height": h,
                    "min_depth": float(np.min(depth_np)),
                    "max_depth": float(np.max(depth_np)),
                    "mean_depth": float(np.mean(depth_np)),
                    "std_depth": float(np.std(depth_np)),
                    "edge_gradient_mean": float(np.mean(grad_mag)),
                    "is_metric": False,
                    "depth_array": depth_np,
                })
                raw_depth_maps.append(depth_np)

        diagnostics = {
            "num_images": len(np_images),
            "model_type": "DPT_Hybrid",
            "mean_edge_sharpness": float(np.mean([r["edge_gradient_mean"] for r in depth_results])),
        }

        return depth_results, raw_depth_maps, diagnostics
