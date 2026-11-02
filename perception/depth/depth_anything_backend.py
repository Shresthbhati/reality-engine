"""Depth Anything V2 monocular depth estimation backend (Apache 2.0).

Uses Depth Anything V2 (Small tier) via Hugging Face transformers.
Compared to MiDaS, Depth Anything V2 produces significantly sharper edge boundaries,
finer geometric structures, and handles complex indoor/outdoor lighting without
torchscript or deprecated legacy PyTorch APIs.

Weights: depth-anything/Depth-Anything-V2-Small-hf (Apache 2.0)
Device: CUDA with fp16 when available (via perception.device.select_device), CPU fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

from evidence.session import EvidenceItem, EvidenceKind
from perception.device import select_device
from provenance import Uncertainty

from .interface import DepthMap, IDepthBackend


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


class DepthAnythingV2Backend(IDepthBackend):
    """Depth Anything V2 depth estimation backend.

    Defaults to 'depth-anything/Depth-Anything-V2-Small-hf' (Apache 2.0, ~25M params,
    fast inference on consumer GPUs and CPU).
    """

    DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
    ):
        self._checkpoint = checkpoint
        self._custom_device = device

        self._processor = None
        self._model = None
        self._device = None

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

    def estimate_depth(self, evidence: List[EvidenceItem]) -> List[DepthMap]:
        """Produce one DepthMap per PHOTO/VIDEO evidence item."""
        _ensure_dependencies()
        import numpy as np
        import torch
        from PIL import Image

        image_evidence = [
            e for e in evidence if e.kind in (EvidenceKind.PHOTO, EvidenceKind.VIDEO)
        ]
        if not image_evidence:
            return []

        self._ensure_model()
        results: List[DepthMap] = []

        for item in image_evidence:
            try:
                # Safely parse URI (handles Windows file:// path conventions)
                parsed = urlparse(item.source_uri)
                if parsed.scheme == "file":
                    raw_path = parsed.path
                    nl = parsed.netloc
                    if nl and len(nl) >= 2 and nl[1] == ":" and (len(nl) == 2 or nl[2] in "/\\"):
                        raw_path = nl + raw_path
                    src_path = Path(url2pathname(raw_path))
                else:
                    src_path = Path(item.source_uri)

                if not src_path.exists():
                    continue

                image = Image.open(src_path).convert("RGB")
                orig_w, orig_h = image.size

                inputs = self._processor(images=image, return_tensors="pt").to(self._device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    predicted_depth = outputs.predicted_depth

                    # Resize to original resolution
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=(orig_h, orig_w),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                depth_np = prediction.cpu().numpy().astype(np.float32)

                # Normalize to [0, 1] relative depth
                d_min, d_max = float(depth_np.min()), float(depth_np.max())
                if d_max > d_min:
                    normalized = (depth_np - d_min) / (d_max - d_min)
                else:
                    normalized = np.zeros_like(depth_np)

                depth_list = [[float(v) for v in row] for row in normalized]

                results.append(
                    DepthMap(
                        evidence_id=item.id,
                        width=orig_w,
                        height=orig_h,
                        values=depth_list,
                        unit="relative",
                        uncertainty=Uncertainty(
                            confidence=0.92,
                            note="depth_anything_v2_small",
                        ),
                    )
                )
            except Exception:
                # Do not fabricate data on failure; skip unreadable items
                continue

        return results
