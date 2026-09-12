"""Real MiDaS-backed IDepthBackend (per docs/TECHNOLOGY_REGISTRY.md recommendation).

Loads a MiDaS model (Intel ISL) via torch.hub or local checkpoint and
produces per-pixel relative depth maps for PHOTO/VIDEO evidence items.

This backend requires the optional 'perception' dependency group:
    pip install -e .[perception]

If torch/torchvision/numpy are not available, reconstruct() raises
DepthBackendUnavailableError rather than silently no-op'ing -- this
repo doesn't fake perception output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from provenance import Uncertainty

from evidence.session import EvidenceItem, EvidenceKind
from .interface import IDepthBackend, DepthMap


class DepthBackendUnavailableError(RuntimeError):
    """Raised when torch/torchvision/numpy or a MiDaS checkpoint isn't available."""
    pass


def _ensure_perception_deps() -> None:
    """Verify torch, torchvision, and numpy are importable.

    Raises DepthBackendUnavailableError with a clear message if not.
    """
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import torchvision
    except ImportError:
        missing.append("torchvision")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        raise DepthBackendUnavailableError(
            f"Missing perception dependencies: {', '.join(missing)}. "
            "Install with: pip install -e .[perception]"
        )


class MiDaSDepthBackend(IDepthBackend):
    """MiDaS (Intel ISL) depth estimation backend.

    Supports MiDaS v2.1 (small, hybrid) and v3.1 (DPT_Large, DPT_Hybrid, DPT_Small)
    model types. Defaults to DPT_Hybrid for a balance of speed and quality.

    Model checkpoints are downloaded via torch.hub on first use (cached in
    ~/.cache/torch/hub/intel-isl_MiDaS_master). For offline/air-gapped
    environments, pre-download the checkpoint and pass `checkpoint_path`.
    """

    VALID_MODEL_TYPES = ("MiDaS_small", "DPT_Large", "DPT_Hybrid", "DPT_Small")

    def __init__(
        self,
        model_type: str = "DPT_Hybrid",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Valid: {self.VALID_MODEL_TYPES}"
            )
        self._model_type = model_type
        self._checkpoint_path = checkpoint_path
        self._device = device or ("cuda" if self._cuda_available() else "cpu")
        self._model = None
        self._transform = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load_model(self) -> None:
        """Load the MiDaS model and transform. Called lazily on first use."""
        _ensure_perception_deps()
        import torch
        import torchvision.transforms as T

        if self._model is not None:
            return

        # Load model via torch.hub (downloads on first use) or from local checkpoint
        if self._checkpoint_path and Path(self._checkpoint_path).exists():
            # Local checkpoint loading - MiDaS uses a specific state dict format
            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                self._model_type,
                source="local" if Path(self._checkpoint_path).parent.name == "MiDaS" else "github",
                pretrained=False,
            )
            state_dict = torch.load(self._checkpoint_path, map_location=self._device)
            self._model.load_state_dict(state_dict)
        else:
            # Download via torch.hub (requires internet on first run)
            self._model = torch.hub.load("intel-isl/MiDaS", self._model_type, pretrained=True)

        self._model.to(self._device)
        self._model.eval()

        # MiDaS preprocessing transform (varies by model type)
        if self._model_type == "MiDaS_small":
            self._transform = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # DPT models use 384x384
            self._transform = T.Compose([
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def estimate_depth(self, evidence: List[EvidenceItem]) -> List[DepthMap]:
        _ensure_perception_deps()
        import torch
        import numpy as np
        from PIL import Image

        self._load_model()

        image_evidence = [
            e for e in evidence if e.kind in (EvidenceKind.PHOTO, EvidenceKind.VIDEO)
        ]
        if not image_evidence:
            return []

        results = []
        for item in image_evidence:
            try:
                # Load image from source_uri
                src_path = Path(item.source_uri.replace("file://", "", 1))
                if not src_path.exists():
                    # Skip evidence items we can't read -- honest failure mode
                    continue

                image = Image.open(src_path).convert("RGB")
                orig_w, orig_h = image.size

                # Preprocess
                input_tensor = self._transform(image).unsqueeze(0).to(self._device)

                # Inference
                with torch.no_grad():
                    prediction = self._model(input_tensor)

                    # Resize to original resolution
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(orig_h, orig_w),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                # Convert to numpy list of lists (row-major)
                depth_np = prediction.cpu().numpy()

                # MiDaS outputs relative depth (inverse depth for DPT models).
                # Normalize to [0, 1] range for consistency.
                # Note: This is RELATIVE depth, not metric. The 'unit' field
                # reflects this honestly.
                depth_min, depth_max = depth_np.min(), depth_np.max()
                if depth_max > depth_min:
                    depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = np.zeros_like(depth_np)

                # Convert to list of lists (row-major)
                values = depth_normalized.tolist()

                # Estimate uncertainty from depth variance (heuristic)
                # Higher variance regions tend to be less reliable
                local_var = np.var(depth_normalized)
                confidence = float(max(0.01, 1.0 / (1.0 + local_var * 10)))

                results.append(DepthMap(
                    evidence_id=item.id,
                    width=orig_w,
                    height=orig_h,
                    values=values,
                    unit="relative",
                    uncertainty=Uncertainty(confidence=confidence),
                ))

            except Exception:
                # Skip items that fail processing -- don't fabricate output
                continue

        return results