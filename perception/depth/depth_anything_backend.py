"""Depth Anything V2 monocular depth estimation backend (Apache 2.0).

Uses Depth Anything V2 (Small tier) via Hugging Face transformers.
Compared to MiDaS, Depth Anything V2 produces significantly sharper edge boundaries,
finer geometric structures, and handles complex indoor/outdoor lighting without
torchscript or deprecated legacy PyTorch APIs.

Weights: depth-anything/Depth-Anything-V2-Small-hf (Apache 2.0)
Device: CUDA with fp16 when available (via perception.device.select_device), CPU fallback.

Output convention (measured 2026-11-26 on the south_building photographs): the relative
checkpoint returns INVERSE depth (larger = nearer, range ~0-6.4), not depth and not
metres. The raw values are passed through unchanged -- a per-image min-max normalisation,
which this backend used to apply, makes two views of the same surface unrelated and makes
cross-view comparison and depth fusion impossible. ``DepthMap.quantity`` says which way the
numbers run so a consumer cannot invert the wrong one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from engine.core.device import drop_models, select_device
from provenance import Uncertainty

from .interface import DepthMap, IDepthBackend

logger = logging.getLogger(__name__)


class DepthBackendUnavailableError(RuntimeError):
    """Raised when torch, transformers, or the Depth Anything checkpoint isn't available."""

    pass


def _ensure_dependencies() -> None:
    """Verify torch, transformers, and PIL are importable."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import PIL
    except ImportError:
        missing.append("pillow")

    if missing:
        raise DepthBackendUnavailableError(
            f"Missing dependencies for Depth Anything V2: {', '.join(missing)}. "
            "Install with: pip install torch transformers pillow"
        )


def _masked_arrays(prediction) -> tuple:
    """``(values, valid_mask)`` where unmeasured pixels are NaN and False.

    The model returns a finite value for every pixel, but a non-positive or non-finite
    prediction is not a depth: it is a hole. It is recorded as one instead of being
    clamped into a plausible-looking number.
    """
    import numpy as np

    values = np.ascontiguousarray(prediction, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0.0)
    masked = np.where(valid, values, np.float32("nan")).astype(np.float32)
    return np.ascontiguousarray(masked), np.ascontiguousarray(valid)


class DepthAnythingV2Backend(IDepthBackend):
    """Depth Anything V2 relative depth backend.

    Defaults to 'depth-anything/Depth-Anything-V2-Small-hf' (Apache 2.0, ~25M params,
    fast inference on consumer GPUs and CPU). Output is relative inverse depth in the
    model's own units: comparable across views (no per-image normalisation), not metric.
    """

    DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"
    unit = "relative"
    quantity = "inverse_depth"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ):
        self._checkpoint = checkpoint
        self._custom_device = device

        self._processor = None
        self._model = None
        self._device = None
        #: One row per evidence item this backend could not process, with the reason.
        #: Reset at the start of every estimate_depth() call; never silently empty.
        self.skipped: list[dict] = []

    @property
    def checkpoint(self) -> str:
        return self._checkpoint

    def unload(self) -> None:
        drop_models(self, "_model", "_processor")

    def _skip(self, evidence_id: Any, reason: str) -> None:
        row = {"evidence_id": str(evidence_id), "reason": reason}
        self.skipped.append(row)
        logger.warning("depth backend skipped %s: %s", row["evidence_id"], reason)

    def _resolve_path(self, item: Any) -> Path | None:
        """On-disk path for an evidence item, or None with the reason recorded."""
        uri = getattr(item, "source_uri", None)
        if not uri:
            self._skip(getattr(item, "id", "?"), "no source_uri")
            return None
        try:
            # Safely parse URI (handles Windows file:// path conventions)
            parsed = urlparse(str(uri))
            if parsed.scheme == "file":
                raw_path = parsed.path
                nl = parsed.netloc
                if nl and len(nl) >= 2 and nl[1] == ":" and (len(nl) == 2 or nl[2] in "/\\"):
                    raw_path = nl + raw_path
                src_path = Path(url2pathname(raw_path))
            else:
                src_path = Path(str(uri))
        except Exception as exc:  # noqa: BLE001 -- a bad URI is a recorded skip, not a crash
            self._skip(getattr(item, "id", "?"), f"unparseable source_uri: {type(exc).__name__}: {exc}")
            return None
        if not src_path.exists():
            self._skip(getattr(item, "id", "?"), f"file does not exist: {src_path}")
            return None
        return src_path

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        _ensure_dependencies()
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self._custom_device:
            self._device = torch.device(self._custom_device)
        else:
            dev_info = select_device()
            self._device = torch.device(dev_info.device)

        try:
            self._processor = AutoImageProcessor.from_pretrained(self._checkpoint)
            self._model = (
                AutoModelForDepthEstimation.from_pretrained(self._checkpoint)
                .to(self._device)
                .eval()
            )
        except Exception as e:
            raise DepthBackendUnavailableError(
                f"Failed to load Depth Anything V2 checkpoint '{self._checkpoint}': {e}"
            ) from e

    def _predict(self, image, orig_w: int, orig_h: int):
        """Raw model prediction resized to the photo's own resolution."""
        import torch

        processor = self._processor
        model = self._model
        if processor is None or model is None:
            raise DepthBackendUnavailableError("Depth Anything model is not initialized")
        inputs = processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.detach().cpu().numpy()

    def _wants(self, item: Any) -> bool:
        return str(getattr(item, "kind", "")).lower() in (
            "photo", "video", "evidencekind.photo", "evidencekind.video")

    def estimate_depth(self, evidence: list[Any]) -> list[DepthMap]:
        """Produce one DepthMap per PHOTO/VIDEO evidence item this backend can process."""
        _ensure_dependencies()
        from engine.core.image_io import open_oriented

        self.skipped = []
        image_evidence = [e for e in evidence if self._wants(e)]
        if not image_evidence:
            return []

        self._ensure_model()
        results: list[DepthMap] = []

        for item in image_evidence:
            src_path = self._resolve_path(item)
            if src_path is None:
                continue
            try:
                image = open_oriented(src_path).convert("RGB")
                orig_w, orig_h = image.size
                raw = self._predict(image, orig_w, orig_h)
            except Exception as exc:  # noqa: BLE001 -- recorded with its reason, never fabricated
                self._skip(getattr(item, "id", "?"), f"{type(exc).__name__}: {exc}")
                continue

            values, valid = _masked_arrays(raw)
            results.append(
                DepthMap(
                    evidence_id=item.id,
                    width=orig_w,
                    height=orig_h,
                    values=values,
                    unit="relative",
                    quantity="inverse_depth",
                    valid_mask=valid,
                    uncertainty=Uncertainty(
                        confidence=0.92,
                        note="depth_anything_v2_small (relative inverse depth, uncalibrated)",
                    ),
                )
            )

        return results


class DepthAnythingV2MetricBackend(DepthAnythingV2Backend):
    """Metric indoor Depth Anything V2 Small (raw predicted metres, uncalibrated).

    ``scale_state`` says what this is: an in-domain estimate from a model trained on other
    indoor data. It is never promoted to a calibrated metric scale, and there is no ground
    truth here to check it against.
    """

    DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    scale_state = "metric_estimated"
    unit = "meters"
    quantity = "depth"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ):
        super().__init__(checkpoint=checkpoint, device=device)

    def estimate_depth(self, evidence: list[Any]) -> list[DepthMap]:
        from engine.core.image_io import open_oriented

        _ensure_dependencies()
        self.skipped = []
        image_evidence = [e for e in evidence if self._wants(e)]
        if not image_evidence:
            return []

        self._ensure_model()
        results: list[DepthMap] = []
        for item in image_evidence:
            src_path = self._resolve_path(item)
            if src_path is None:
                continue
            try:
                image = open_oriented(src_path).convert("RGB")
                orig_w, orig_h = image.size
                raw = self._predict(image, orig_w, orig_h)
            except Exception as exc:  # noqa: BLE001 -- recorded with its reason, never fabricated
                self._skip(getattr(item, "id", "?"), f"{type(exc).__name__}: {exc}")
                continue

            values, valid = _masked_arrays(raw)
            results.append(
                DepthMap(
                    evidence_id=item.id,
                    width=orig_w,
                    height=orig_h,
                    values=values,
                    unit="meters",
                    quantity="depth",
                    valid_mask=valid,
                    uncertainty=Uncertainty(
                        confidence=0.65,
                        note="depth_anything_v2_metric_indoor_small_estimated (metres, uncalibrated)",
                    ),
                )
            )
        return results
