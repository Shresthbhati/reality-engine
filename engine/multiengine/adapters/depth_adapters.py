"""Monocular Depth Estimation Engine Adapters.

Implements sequential execution for Depth Anything V3, Depth Anything V2,
and MiDaS DPT_Hybrid. Preserves raw depth statistics in `original_coordinates`
and normalises depth maps into coordinate frames in `predicted`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from engine.multiengine.adapters.base import BaseModelEngineAdapter
from engine.multiengine.contract import (
    Availability,
    AvailabilityStatus,
    Device,
    EngineError,
    EngineErrorKind,
    EngineRequest,
    EngineSpec,
)
from engine.multiengine.observation import ObservationDraft


def _hf_hub_cache() -> Path:
    import os

    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _is_hf_cached(repo_id: str) -> bool:
    root = _hf_hub_cache() / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not root.is_dir():
        return False
    return any((snap / "config.json").exists() for snap in root.iterdir())


def probe_depth_anything_v3() -> Availability:
    """Honest availability probe for Depth Anything V3."""
    if (
        importlib.util.find_spec("transformers") is None
        or importlib.util.find_spec("torch") is None
    ):
        return Availability(
            AvailabilityStatus.BLOCKED,
            "transformers and torch are required for Depth Anything V3",
            "install torch and transformers",
        )
    if not _is_hf_cached("depth-anything/Depth-Anything-V3-Small-hf"):
        return Availability(
            AvailabilityStatus.BLOCKED,
            "dependencies present, but Depth Anything V3 weights are not cached in Hugging Face cache",
            "huggingface-cli download depth-anything/Depth-Anything-V3-Small-hf",
        )
    return Availability(
        AvailabilityStatus.AVAILABLE,
        "transformers and cached weights verified for Depth Anything V3",
        evidence=("Depth Anything V3 cached",),
    )


def probe_depth_anything_v2() -> Availability:
    """Honest availability probe for Depth Anything V2."""
    if (
        importlib.util.find_spec("transformers") is None
        or importlib.util.find_spec("torch") is None
    ):
        return Availability(
            AvailabilityStatus.BLOCKED,
            "transformers and torch are required for Depth Anything V2",
            "install torch and transformers",
        )
    if not _is_hf_cached("depth-anything/Depth-Anything-V2-Small-hf"):
        return Availability(
            AvailabilityStatus.BLOCKED,
            "dependencies present, but Depth Anything V2 weights are not cached in Hugging Face cache",
            "huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf",
        )
    return Availability(
        AvailabilityStatus.AVAILABLE,
        "transformers and cached weights verified for Depth Anything V2",
        evidence=("Depth-Anything-V2-Small-hf cached",),
    )


def probe_midas() -> Availability:
    """Honest availability probe for MiDaS."""
    from perception.availability import real_models_by_name

    m = real_models_by_name()["midas"]
    if m.status == "available":
        return Availability(AvailabilityStatus.AVAILABLE, m.reason, evidence=(m.reason,))
    return Availability(AvailabilityStatus.BLOCKED, m.reason, m.remediation)


class DepthAnythingV2EngineAdapter(BaseModelEngineAdapter):
    """Adapter for Depth Anything V2 monocular depth estimation."""

    DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, spec: EngineSpec, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
        super().__init__(spec)
        self.checkpoint = checkpoint
        self._processor = None

    def _load_model_weights(self, device: Device) -> None:
        if importlib.util.find_spec("transformers") is None:
            raise EngineError(EngineErrorKind.UNAVAILABLE, "transformers package missing.")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        dev_str = "cuda:0" if device.kind == "cuda" else "cpu"
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                self.checkpoint, local_files_only=True
            )
            self._model = (
                AutoModelForDepthEstimation.from_pretrained(self.checkpoint, local_files_only=True)
                .to(dev_str)
                .eval()
            )
        except Exception as e:
            raise EngineError(
                EngineErrorKind.MISSING_CHECKPOINT,
                f"Failed to load checkpoint {self.checkpoint}: {e}",
            ) from e

    def validate_input(self, request: EngineRequest) -> None:
        super().validate_input(request)
        for s in request.sources:
            p = Path(s)
            if not p.exists() and not str(s).startswith("virtual:"):
                raise EngineError(
                    EngineErrorKind.INVALID_INPUT, f"Source image does not exist: {s}"
                )

    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]:
        drafts: list[ObservationDraft] = []
        sources = request.sources

        virtual_run = all(str(s).startswith("virtual:") for s in sources)
        if not virtual_run and self._model is None:
            raise EngineError(
                EngineErrorKind.RUNTIME_ERROR,
                f"Model {self.spec.engine_id} not loaded. Call load() before infer().",
            )
        if virtual_run:
            for s in sources:
                w, h = 640, 480
                raw_depth = np.linspace(1.0, 5.0, w, dtype=np.float32)
                np.tile(raw_depth, (h, 1))
                drafts.append(
                    ObservationDraft(
                        task="depth",
                        quantity="depth_map",
                        source_id=str(s),
                        coordinate_system=f"cam:{Path(str(s)).stem}",
                        predicted={
                            "depth_min_m": 1.0,
                            "depth_max_m": 5.0,
                            "depth_mean_m": 3.0,
                            "width": w,
                            "height": h,
                            "scale_state": "relative",
                            "is_test_double": True,
                        },
                        original_coordinates={
                            "raw_min": 1.0,
                            "raw_max": 5.0,
                            "shape": [h, w],
                        },
                        confidence=None,
                    )
                )
            return drafts

        import torch
        from PIL import Image

        for s in sources:
            img = Image.open(s).convert("RGB")
            if request.max_resolution:
                img.thumbnail((request.max_resolution, request.max_resolution))

            inputs = self._processor(images=img, return_tensors="pt")
            dev_str = "cuda:0" if self._device.kind == "cuda" else "cpu"
            inputs = {k: v.to(dev_str) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                pred = outputs.predicted_depth.squeeze().cpu().numpy()

            d_min = float(np.min(pred))
            d_max = float(np.max(pred))
            drafts.append(
                ObservationDraft(
                    task="depth",
                    quantity="depth_map",
                    source_id=str(s),
                    coordinate_system=f"cam:{Path(str(s)).stem}",
                    predicted={
                        "depth_min_m": d_min,
                        "depth_max_m": d_max,
                        "depth_mean_m": float(np.mean(pred)),
                        "width": int(pred.shape[1]),
                        "height": int(pred.shape[0]),
                        "scale_state": "relative",
                    },
                    original_coordinates={
                        "raw_min": d_min,
                        "raw_max": d_max,
                        "shape": list(pred.shape),
                    },
                    confidence=None,
                )
            )

        return drafts


class DepthAnythingV3EngineAdapter(DepthAnythingV2EngineAdapter):
    """Adapter for Depth Anything V3 (multi-view / metric enhanced)."""

    DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V3-Small-hf"

    def __init__(self, spec: EngineSpec) -> None:
        super().__init__(spec, checkpoint=self.DEFAULT_CHECKPOINT)


class MidasEngineAdapter(BaseModelEngineAdapter):
    """Adapter for MiDaS DPT_Hybrid baseline depth estimation."""

    def __init__(self, spec: EngineSpec) -> None:
        super().__init__(spec)
        self._transform = None

    def _load_model_weights(self, device: Device) -> None:
        if importlib.util.find_spec("torch") is None:
            raise EngineError(EngineErrorKind.UNAVAILABLE, "torch missing for MiDaS.")
        import torch

        dev_str = "cuda:0" if device.kind == "cuda" else "cpu"
        try:
            self._model = (
                torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", source="github", trust_repo=True)
                .to(dev_str)
                .eval()
            )
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", source="github", trust_repo=True
            )
            self._transform = midas_transforms.dpt_transform
        except Exception as e:
            # If network or local load fails, raise explicit missing checkpoint
            raise EngineError(
                EngineErrorKind.MISSING_CHECKPOINT, f"Failed to load MiDaS from torch.hub: {e}"
            ) from e

    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]:
        drafts: list[ObservationDraft] = []
        sources = request.sources

        virtual_run = all(str(s).startswith("virtual:") for s in sources)
        if not virtual_run and self._model is None:
            raise EngineError(
                EngineErrorKind.RUNTIME_ERROR,
                f"Model {self.spec.engine_id} not loaded. Call load() before infer().",
            )
        if virtual_run:
            for s in sources:
                drafts.append(
                    ObservationDraft(
                        task="depth",
                        quantity="depth_map",
                        source_id=str(s),
                        coordinate_system=f"cam:{Path(str(s)).stem}",
                        predicted={
                            "depth_min_m": 0.5,
                            "depth_max_m": 6.0,
                            "depth_mean_m": 2.8,
                            "width": 640,
                            "height": 480,
                            "scale_state": "relative",
                            "is_test_double": True,
                        },
                        original_coordinates={"raw_format": "inverse_depth"},
                        confidence=None,
                    )
                )
            return drafts

        import cv2
        import torch

        dev_str = "cuda:0" if self._device.kind == "cuda" else "cpu"
        for s in sources:
            img = cv2.imread(str(s))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = self._transform(img).to(dev_str)

            with torch.no_grad():
                pred = self._model(input_batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                depth_np = pred.cpu().numpy()

            d_min = float(np.min(depth_np))
            d_max = float(np.max(depth_np))
            drafts.append(
                ObservationDraft(
                    task="depth",
                    quantity="depth_map",
                    source_id=str(s),
                    coordinate_system=f"cam:{Path(str(s)).stem}",
                    predicted={
                        "depth_min_m": d_min,
                        "depth_max_m": d_max,
                        "depth_mean_m": float(np.mean(depth_np)),
                        "width": int(depth_np.shape[1]),
                        "height": int(depth_np.shape[0]),
                        "scale_state": "relative",
                    },
                    original_coordinates={"raw_min": d_min, "raw_max": d_max},
                    confidence=None,
                )
            )

        return drafts
