"""Segment Anything (SAM) Box-Prompted Segmentation Adapter.

Takes 2D bounding boxes (from Grounding DINO, OWL-ViT, or Florence-2) and
generates fine pixel-level segmentation masks with interior hole topology,
extracting true polygonal boundaries via OpenCV RETR_CCOMP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from engine.adapters.base import (
    BaseEngineAdapter,
    DependencyMissingError,
    InputValidationError,
)
from perception.specialists.segment import extract_mask_topology


class SamPromptedSegmenterAdapter(BaseEngineAdapter):
    """Adapter for SAM 1 box-prompted instance segmentation."""

    DEFAULT_CHECKPOINT = "checkpoints/sam_vit_b.pth"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        model_type: str = "vit_b",
        device: str | None = None,
    ) -> None:
        super().__init__(
            engine_name="sam_prompted_segmenter",
            task_category="instance_segmentation",
            licence="Apache-2.0",
            device=device,
        )
        self.checkpoint = checkpoint
        self.model_type = model_type
        self._predictor = None

    def verify_dependencies(self) -> tuple[bool, str]:
        import importlib.util
        missing = [pkg for pkg in ("torch", "segment_anything") if importlib.util.find_spec(pkg) is None]
        if missing:
            return False, f"Missing dependencies: {', '.join(missing)}"
        return True, "segment_anything available"

    def load_model(self) -> None:
        if self._predictor is not None:
            return

        import torch
        from segment_anything import SamPredictor, sam_model_registry

        dev = self._custom_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        chk_path = Path(self.checkpoint)
        if not chk_path.exists():
            # Check default hub cache or local models directory
            alt_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "sam_vit_b_01ec64.pth"
            if alt_path.exists():
                chk_path = alt_path

        try:
            sam = sam_model_registry[self.model_type](checkpoint=str(chk_path) if chk_path.exists() else None)
            sam.to(dev).eval()
            self._model = sam
            self._predictor = SamPredictor(sam)
        except Exception as e:
            raise DependencyMissingError(f"Failed to load SAM ({self.model_type}): {e}") from e

    def validate_inputs(self, inputs: Any) -> None:
        if not isinstance(inputs, dict):
            raise InputValidationError("Inputs must be a dict containing 'image' and 'boxes'.")
        if "image" not in inputs:
            raise InputValidationError("Missing 'image' key in inputs dict.")
        if "boxes" not in inputs:
            raise InputValidationError("Missing 'boxes' key in inputs dict.")

    def _execute_inference(
        self,
        inputs: dict[str, Any],
        options: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
        raw_image = inputs["image"]
        boxes = inputs["boxes"]  # list of [x1, y1, x2, y2]

        if isinstance(raw_image, (str, Path)):
            img_bgr = cv2.imread(str(raw_image))
            if img_bgr is None:
                raise InputValidationError(f"Could not read image from {raw_image}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(raw_image, Image.Image):
            img_rgb = np.array(raw_image.convert("RGB"))
        elif isinstance(raw_image, np.ndarray):
            img_rgb = raw_image
        else:
            raise InputValidationError(f"Unsupported image type: {type(raw_image)}")

        h, w = img_rgb.shape[:2]
        self._predictor.set_image(img_rgb)

        segmented_regions = []
        raw_masks = []

        for idx, box in enumerate(boxes):
            box_arr = np.array(box, dtype=np.float32)
            masks, scores, logits = self._predictor.predict(
                box=box_arr[None, :],
                multimask_output=False,
            )
            mask_bool = masks[0].astype(bool)
            raw_masks.append(mask_bool)

            # Extract full polygon topology with interior holes
            _, _, topology = extract_mask_topology(mask_bool)

            segmented_regions.append({
                "region_index": idx,
                "input_box_xyxy": [float(c) for c in box],
                "score": float(scores[0]),
                "mask_area_px": float(np.sum(mask_bool)),
                "net_polygon_area_px": topology.total_area_px,
                "num_exterior_polygons": topology.num_components,
                "num_interior_holes": topology.num_holes,
                "has_holes": topology.num_holes > 0,
                "mask_shape": list(mask_bool.shape),
                "is_empty": not topology.is_valid or topology.total_area_px == 0.0,
            })

        diagnostics = {
            "checkpoint": self.checkpoint,
            "total_boxes_processed": len(boxes),
            "regions_with_holes": sum(1 for r in segmented_regions if r["has_holes"]),
            "mean_confidence": float(np.mean([r["score"] for r in segmented_regions])) if segmented_regions else 0.0,
        }

        import torch
        if hasattr(self._predictor, "reset_image"):
            self._predictor.reset_image()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return segmented_regions, raw_masks, diagnostics
