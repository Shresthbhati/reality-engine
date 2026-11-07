"""The two learned models the single-image layout stage needs, behind one small interface.

* Metric indoor depth: ``depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`` (Depth Anything V2 Small, Apache-2.0 per the
  upstream project; the Hugging Face model card carries no licence tag, so confirm before shipping a paid product).
* ADE20K surface segmentation: ``openmmlab/upernet-convnext-tiny`` (MIT per its model card). Gives wall / floor / ceiling /
  windowpane / door per pixel, which is what the solver needs.

Weights are loaded from the LOCAL Hugging Face cache only (nothing is downloaded at request time, matching the repo's other
backends). When they are absent a ``LayoutModelsUnavailable`` explains how to fetch them; the stage never fabricates a layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SEG_MODEL = "openmmlab/upernet-convnext-tiny"

#: ADE20K class names -> solver surface keys
SURFACE_CLASSES: Dict[str, tuple] = {
    "floor": ("floor", "rug"),
    "ceiling": ("ceiling",),
    "wall": ("wall",),
    "window": ("windowpane",),
    "door": ("door",),
}


class LayoutModelsUnavailable(RuntimeError):
    pass


@dataclass
class ImageAnalysis:
    depth_m: np.ndarray  # (H, W) metric z-depth
    masks: Dict[str, np.ndarray]  # solver surface masks
    label_map: np.ndarray  # (H, W) ADE20K class ids
    id2label: Dict[int, str]
    rgb: Optional[np.ndarray] = None  # (H, W, 3) uint8: lets the engine match features between photos


class LayoutModels:
    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device
        self._depth = None
        self._depth_proc = None
        self._seg = None
        self._seg_proc = None

    def _dev(self) -> str:
        if self._device:
            return self._device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._depth is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation, UperNetForSemanticSegmentation

            dev = self._dev()
            self._depth_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL, local_files_only=True)
            self._depth = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, local_files_only=True).to(dev).eval()
            self._seg_proc = AutoImageProcessor.from_pretrained(SEG_MODEL, local_files_only=True)
            self._seg = UperNetForSemanticSegmentation.from_pretrained(SEG_MODEL, local_files_only=True).to(dev).eval()
        except Exception as exc:  # noqa: BLE001
            self._depth = self._seg = None
            if "out of memory" in str(exc).lower() and self._device != "cpu":
                raise  # analyze() retries the whole thing on CPU
            raise LayoutModelsUnavailable(
                f"layout models are not available locally ({exc}). Fetch once with: "
                f"python -c \"from huggingface_hub import snapshot_download as d; d('{DEPTH_MODEL}'); d('{SEG_MODEL}')\""
            ) from exc

    def unload(self) -> None:
        self._depth = self._depth_proc = self._seg = self._seg_proc = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def analyze(self, image) -> ImageAnalysis:
        """``image``: a PIL RGB image (already EXIF-oriented). A full GPU (shared with COLMAP / other apps) retries on CPU."""
        try:
            return self._analyze_once(image)
        except Exception as exc:  # noqa: BLE001
            if "out of memory" not in str(exc).lower() or self._device == "cpu":
                raise
            self.unload()
            self._device = "cpu"
            return self._analyze_once(image)

    def _analyze_once(self, image) -> ImageAnalysis:
        import torch

        self._load()
        dev = self._dev()
        w, h = image.size
        with torch.no_grad():
            d_in = self._depth_proc(images=image, return_tensors="pt").to(dev)
            d_out = self._depth(**d_in)
            depth = self._depth_proc.post_process_depth_estimation(d_out, target_sizes=[(h, w)])[0]["predicted_depth"]
            s_in = self._seg_proc(images=image, return_tensors="pt").to(dev)
            s_out = self._seg(**s_in)
            label_map = self._seg_proc.post_process_semantic_segmentation(s_out, target_sizes=[(h, w)])[0]
        label_map = label_map.cpu().numpy().astype(np.int32)
        id2label = {int(k): str(v).strip() for k, v in self._seg.config.id2label.items()}
        masks = {}
        for key, names in SURFACE_CLASSES.items():
            ids = [i for i, n in id2label.items() if n in names]
            masks[key] = np.isin(label_map, ids)
        return ImageAnalysis(depth_m=depth.cpu().numpy().astype(np.float64), masks=masks, label_map=label_map, id2label=id2label,
                             rgb=np.asarray(image))


def analyze_path(models: LayoutModels, path: Path) -> ImageAnalysis:
    from evidence.image_check import open_oriented

    return models.analyze(open_oriented(Path(path)).convert("RGB"))
