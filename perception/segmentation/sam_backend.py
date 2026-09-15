"""Real SAM-backed ISegmentationBackend (per docs/TECHNOLOGY_REGISTRY.md recommendation).

Loads a Segment Anything Model (SAM, Meta AI) from the `segment-anything`
pip package plus a checkpoint file, and produces per-instance segmentation
masks for PHOTO/VIDEO evidence items.

This backend requires the optional 'perception' dependency group:
    pip install -e .[perception]

If torch/torchvision/numpy are not available, segment() raises
SegmentationBackendUnavailableError rather than silently no-op'ing -- this
repo doesn't fake perception output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from provenance import Uncertainty

from evidence.session import EvidenceItem, EvidenceKind
from .interface import ISegmentationBackend, SegmentedRegion, SegmentationResult


class SegmentationBackendUnavailableError(RuntimeError):
    """Raised when torch/torchvision/numpy or a SAM checkpoint isn't available."""
    pass


def _ensure_perception_deps() -> None:
    """Verify torch, torchvision, and numpy are importable.

    Raises SegmentationBackendUnavailableError with a clear message if not.
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
    try:
        import segment_anything
    except ImportError:
        missing.append("segment-anything")

    if missing:
        raise SegmentationBackendUnavailableError(
            f"Missing perception dependencies: {', '.join(missing)}. "
            "Install with: pip install -e .[perception]"
            + (
                " (and: pip install segment-anything)"
                if "segment-anything" in missing
                else ""
            )
        )


class SAMSegmentationBackend(ISegmentationBackend):
    """Segment Anything Model (Meta AI) segmentation backend.

    Supports SAM ViT-B, ViT-L, ViT-H model types. Defaults to ViT-H for quality.
    Also supports automatic mask generation (no prompts) via SAM's
    `SamAutomaticMaskGenerator`.

    Model code comes from the `segment-anything` pip package (the repo has no
    torch.hub hubconf.py, so `torch.hub.load("facebookresearch/segment-anything", ...)`
    cannot work -- verified upstream, 0 commits touching hubconf.py). The
    checkpoint is resolved in order: explicit `checkpoint_path` -> the shared
    torch-hub cache (`~/.cache/torch/hub/checkpoints/sam_<type>_<hash>.pth`, so
    pre-cached weights are picked up automatically) -> a direct download of the
    official URL. For offline/air-gapped environments, pre-download the
    checkpoint into that cache location or pass `checkpoint_path`.

    Note: SAM produces class-agnostic masks (no semantic labels). The `label`
    field in SegmentedRegion will be set to the backend's internal mask index
    (e.g., "mask_0", "mask_1") unless the caller provides prompts with labels.
    """

    VALID_MODEL_TYPES = ("vit_b", "vit_l", "vit_h")

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        # Automatic mask generator parameters (tunable for quality/speed)
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 0,
    ):
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Valid: {self.VALID_MODEL_TYPES}"
            )
        self._model_type = model_type
        self._checkpoint_path = checkpoint_path
        self._device = device or ("cuda" if self._cuda_available() else "cpu")
        self._model = None
        self._mask_generator = None

        # Store mask generator config
        self._mask_gen_config = {
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "crop_n_layers": crop_n_layers,
            "crop_n_points_downscale_factor": crop_n_points_downscale_factor,
            "min_mask_region_area": min_mask_region_area,
        }

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    #: Official SAM checkpoint URLs (keyed by model type), as published by
    #: the segment-anything project. Hash suffixes are the upstream names.
    _CHECKPOINT_URLS = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }

    def _resolve_checkpoint(self) -> str:
        """Resolve the checkpoint file for the configured model type.

        Order: explicit ``checkpoint_path`` -> the shared torch-hub
        checkpoint cache (so pre-cached weights are picked up without any
        code change) -> direct download of the official URL. Raises
        SegmentationBackendUnavailableError when nothing resolves; it never
        fabricates or substitutes a different checkpoint.
        """
        import torch

        if self._checkpoint_path:
            path = Path(self._checkpoint_path)
            if not path.exists():
                raise SegmentationBackendUnavailableError(
                    f"SAM checkpoint not found at checkpoint_path: {path}"
                )
            return str(path)

        cache = Path(torch.hub.get_dir()) / "checkpoints"
        url = self._CHECKPOINT_URLS[self._model_type]
        cached = cache / url.rsplit("/", 1)[-1]
        if cached.exists():
            return str(cached)

        try:
            return str(torch.hub.load_state_dict_from_url(url, progress=False))
        except Exception as exc:  # noqa: BLE001 - honest unavailability, any cause
            raise SegmentationBackendUnavailableError(
                f"SAM checkpoint for {self._model_type} could not be resolved "
                f"(no checkpoint_path given, not cached under {cache}, and "
                f"download failed: {exc}). Pre-download it, e.g.: "
                f"python -c \"import torch; torch.hub.load_state_dict_from_url('{url}')\""
            ) from exc

    def _load_model(self) -> None:
        """Load the SAM model and automatic mask generator. Called lazily on first use."""
        _ensure_perception_deps()
        import torch
        from segment_anything import sam_model_registry

        if self._model is not None:
            return

        # Canonical load path: the segment-anything package's model registry
        # plus a checkpoint file. torch.hub.load("facebookresearch/segment-anything")
        # is not a valid alternative -- that repo has no hubconf.py (verified
        # upstream, 0 commits touching it), so source="github" loads are
        # structurally broken for every consumer.
        ckpt = self._resolve_checkpoint()
        self._model = sam_model_registry[self._model_type](checkpoint=ckpt)
        self._model.to(self._device)
        self._model.eval()

        # Create automatic mask generator
        from segment_anything import SamAutomaticMaskGenerator
        self._mask_generator = SamAutomaticMaskGenerator(
            model=self._model,
            **self._mask_gen_config,
        )

    def segment(self, evidence: List[EvidenceItem]) -> List[SegmentationResult]:
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
                image_np = np.array(image)
                orig_h, orig_w = image_np.shape[:2]

                # SAM expects RGB uint8 [0, 255]
                if image_np.dtype != np.uint8:
                    image_np = (image_np * 255).astype(np.uint8)

                # Generate masks
                with torch.no_grad():
                    masks = self._mask_generator.generate(image_np)

                # Convert SAM output to SegmentedRegion list
                regions = []
                for i, mask_data in enumerate(masks):
                    # SAM returns masks as bool arrays (H, W)
                    mask_bool = mask_data["segmentation"]
                    if mask_bool.shape != (orig_h, orig_w):
                        # Should not happen with automatic generator, but handle anyway
                        continue

                    # Convert to list of lists (row-major)
                    mask_list = mask_bool.tolist()

                    # SAM provides 'predicted_iou' and 'stability_score' as quality metrics
                    # Use predicted_iou as confidence. It is nominally in [0, 1] but float
                    # round-off can push saturated values slightly above 1.0 (observed:
                    # 1.0005), which Uncertainty correctly rejects. Clamp at the conversion
                    # boundary -- skipping the region instead would silently discard valid
                    # detections (the per-item except below would swallow the error).
                    confidence = min(1.0, max(0.0, float(mask_data.get("predicted_iou", 0.5))))

                    # Create region with internal label (mask index)
                    # Downstream can map these to ontology labels
                    regions.append(SegmentedRegion(
                        region_id=f"{item.id}_mask_{i}",
                        evidence_id=item.id,
                        label=f"mask_{i}",  # Class-agnostic; downstream maps to ontology
                        mask=mask_list,
                        confidence=confidence,
                        uncertainty=Uncertainty(confidence=confidence),
                    ))

                if regions:
                    results.append(SegmentationResult(
                        evidence_id=item.id,
                        regions=regions,
                        model_name="sam",
                        model_version=self._model_type,
                    ))

            except Exception:
                # Skip items that fail processing -- don't fabricate output
                continue

        return results