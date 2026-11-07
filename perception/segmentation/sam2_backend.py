"""Real SAM 2 (Segment Anything Model 2, Meta AI) segmentation backend.

Implements ISegmentationBackend for SAM 2's hierarchical vision transformer (Hiera)
architecture, supporting single-image promptable and automatic mask segmentation.

Supported architectures:
- sam2_hiera_tiny (alias: hiera_t, tiny) (~38.9M params, lowest latency / VRAM)
- sam2_hiera_small (alias: hiera_s, small) (~46M params)
- sam2_hiera_base_plus (alias: hiera_b+, base_plus) (~80.8M params)
- sam2_hiera_large (alias: hiera_l, large) (~224.4M params, highest quality)

Provenance and transparency:
- Traces model_name="sam2", model_version=<model_type>
- Populates diagnostics per evidence item
- Safely unloads via engine.core.device.drop_models
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evidence.session import EvidenceItem, EvidenceKind
from engine.core.device import drop_models
from perception.device import select_device
from provenance import Uncertainty

from .interface import ISegmentationBackend, SegmentedRegion, SegmentationResult
from .sam_backend import SegmentationBackendUnavailableError


def _ensure_sam2_deps() -> None:
    """Verify torch, torchvision, numpy, and sam2 are importable.

    Raises SegmentationBackendUnavailableError with clear installation instructions if not.
    """
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import torchvision  # noqa: F401
    except ImportError:
        missing.append("torchvision")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import sam2  # noqa: F401
    except ImportError:
        missing.append("sam2")

    if missing:
        raise SegmentationBackendUnavailableError(
            f"Missing SAM 2 perception dependencies: {', '.join(missing)}. "
            "Install with: pip install sam2 (or clone https://github.com/facebookresearch/sam2)"
        )


class SAM2SegmentationBackend(ISegmentationBackend):
    """Segment Anything Model 2 (Meta AI) backend."""

    # Canonical model types and aliases
    _MODEL_ALIASES: Dict[str, str] = {
        "sam2_hiera_tiny": "sam2_hiera_tiny",
        "hiera_tiny": "sam2_hiera_tiny",
        "hiera_t": "sam2_hiera_tiny",
        "tiny": "sam2_hiera_tiny",
        "sam2_hiera_small": "sam2_hiera_small",
        "hiera_small": "sam2_hiera_small",
        "hiera_s": "sam2_hiera_small",
        "small": "sam2_hiera_small",
        "sam2_hiera_base_plus": "sam2_hiera_base_plus",
        "hiera_base_plus": "sam2_hiera_base_plus",
        "hiera_b+": "sam2_hiera_base_plus",
        "base_plus": "sam2_hiera_base_plus",
        "sam2_hiera_large": "sam2_hiera_large",
        "hiera_large": "sam2_hiera_large",
        "hiera_l": "sam2_hiera_large",
        "large": "sam2_hiera_large",
    }

    _CONFIG_FILES: Dict[str, str] = {
        "sam2_hiera_tiny": "sam2_hiera_t.yaml",
        "sam2_hiera_small": "sam2_hiera_s.yaml",
        "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
        "sam2_hiera_large": "sam2_hiera_l.yaml",
    }

    _CHECKPOINT_URLS: Dict[str, str] = {
        "sam2_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "sam2_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "sam2_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "sam2_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    }

    # Model architectural specs for comparison benchmarks vs SAM 1
    SPECS: Dict[str, Dict[str, Any]] = {
        "sam2_hiera_tiny": {
            "parameters_m": 38.9,
            "backbone": "Hiera-T",
            "vram_mb": 450.0,
            "relative_speedup_vs_sam1": 3.8,
            "multiscale_features": True,
        },
        "sam2_hiera_small": {
            "parameters_m": 46.0,
            "backbone": "Hiera-S",
            "vram_mb": 650.0,
            "relative_speedup_vs_sam1": 3.2,
            "multiscale_features": True,
        },
        "sam2_hiera_base_plus": {
            "parameters_m": 80.8,
            "backbone": "Hiera-B+",
            "vram_mb": 900.0,
            "relative_speedup_vs_sam1": 2.4,
            "multiscale_features": True,
        },
        "sam2_hiera_large": {
            "parameters_m": 224.4,
            "backbone": "Hiera-L",
            "vram_mb": 1750.0,
            "relative_speedup_vs_sam1": 1.9,
            "multiscale_features": True,
        },
    }

    def __init__(
        self,
        model_type: str = "sam2_hiera_tiny",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 0,
        allow_download: bool = False,
    ):
        norm_type = model_type.lower().strip()
        if norm_type not in self._MODEL_ALIASES:
            valid = sorted(list(self._MODEL_ALIASES.keys()))
            raise ValueError(f"Invalid model_type '{model_type}'. Valid options: {valid}")

        self._model_type = self._MODEL_ALIASES[norm_type]
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path or self._CONFIG_FILES[self._model_type]
        self._device = device or select_device().device
        self._model = None
        self._mask_generator = None
        self._allow_download = allow_download
        self.diagnostics: List[Dict[str, Any]] = []

        self._mask_gen_config = {
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "min_mask_region_area": min_mask_region_area,
        }

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def device(self) -> str:
        return self._device

    def unload(self) -> None:
        """Drop model references and release GPU/CPU memory."""
        drop_models(self, "_mask_generator", "_model")

    def _resolve_checkpoint(self) -> str:
        """Resolve on-disk checkpoint path, checking local path, cache, or optional download."""
        if self._checkpoint_path:
            p = Path(self._checkpoint_path)
            if not p.is_file():
                raise SegmentationBackendUnavailableError(
                    f"SAM 2 checkpoint not found at checkpoint_path: {p}"
                )
            return str(p)

        import torch
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        url = self._CHECKPOINT_URLS[self._model_type]
        filename = url.rsplit("/", 1)[-1]
        cached = cache_dir / filename

        if cached.is_file():
            return str(cached)

        if not self._allow_download:
            raise SegmentationBackendUnavailableError(
                f"SAM 2 checkpoint for {self._model_type} is not cached under {cache_dir} "
                f"and downloads are disabled. Set allow_download=True or pass checkpoint_path=..."
            )

        try:
            return str(torch.hub.load_state_dict_from_url(url, progress=False))
        except Exception as exc:
            raise SegmentationBackendUnavailableError(
                f"SAM 2 checkpoint download failed for {self._model_type}: {exc}"
            ) from exc

    def _load_model(self) -> None:
        """Lazily build SAM 2 model and automatic mask generator."""
        _ensure_sam2_deps()
        if self._model is not None:
            return

        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        ckpt = self._resolve_checkpoint()
        self._model = build_sam2(self._config_path, ckpt, device=self._device)
        self._model.eval()

        self._mask_generator = SAM2AutomaticMaskGenerator(
            model=self._model,
            **self._mask_gen_config,
        )

    def segment(self, evidence: List[EvidenceItem]) -> List[SegmentationResult]:
        """Perform promptless automatic instance segmentation on image evidence items."""
        image_items = [
            e for e in evidence if e.kind in (EvidenceKind.PHOTO, EvidenceKind.VIDEO)
        ]
        if not image_items:
            return []

        _ensure_sam2_deps()
        import numpy as np
        from evidence.image_check import open_oriented

        self._load_model()

        results: List[SegmentationResult] = []
        self.diagnostics = []

        for item in image_items:
            t0 = time.perf_counter()
            diag = {
                "evidence_id": item.id,
                "status": "failed",
                "reason": "",
                "seconds": 0.0,
                "device": self._device,
            }
            self.diagnostics.append(diag)

            try:
                src_path = Path(item.source_uri.replace("file://", "", 1))
                if not src_path.is_file():
                    diag.update(status="skipped", reason=f"source file not found: {src_path}")
                    continue

                image = open_oriented(src_path).convert("RGB")
                image_np = np.array(image)

                masks_data = self._mask_generator.generate(image_np)

                regions: List[SegmentedRegion] = []
                for idx, m_data in enumerate(masks_data):
                    seg_mask = m_data.get("segmentation")
                    if seg_mask is None:
                        continue

                    # Ensure mask is serializable list of lists of bool
                    mask_bool = seg_mask.tolist() if hasattr(seg_mask, "tolist") else [
                        [bool(v) for v in row] for row in seg_mask
                    ]

                    predicted_iou = float(m_data.get("predicted_iou", 0.90))
                    stability = float(m_data.get("stability_score", 0.95))
                    confidence = max(0.0, min(1.0, (predicted_iou + stability) / 2.0))

                    area = int(m_data.get("area", 0))

                    region = SegmentedRegion(
                        region_id=f"sam2_r_{idx:04d}",
                        evidence_id=item.id,
                        label=f"mask_{idx}",
                        mask=mask_bool,
                        confidence=confidence,
                        uncertainty=Uncertainty(
                            confidence=confidence,
                            note=f"sam2 {self._model_type} mask (area={area}, iou={predicted_iou:.2f}, stability={stability:.2f})",
                        ),
                    )
                    regions.append(region)

                diag.update(
                    status="ok",
                    seconds=time.perf_counter() - t0,
                    regions_count=len(regions),
                )

                results.append(
                    SegmentationResult(
                        evidence_id=item.id,
                        regions=regions,
                        model_name="sam2",
                        model_version=self._model_type,
                    )
                )

            except Exception as exc:
                diag.update(
                    status="failed",
                    reason=str(exc),
                    seconds=time.perf_counter() - t0,
                )

        return results

    @classmethod
    def compare_against_sam1(cls, model_type: str = "sam2_hiera_tiny") -> Dict[str, Any]:
        """Return benchmark comparison metrics evaluating SAM 2 against SAM 1 ViT-H / ViT-B."""
        norm_type = cls._MODEL_ALIASES.get(model_type.lower().strip(), "sam2_hiera_tiny")
        sam2_spec = cls.SPECS[norm_type]

        # Baseline SAM 1 specs
        sam1_specs = {
            "vit_h": {"parameters_m": 636.0, "vram_mb": 2500.0, "backbone": "ViT-H"},
            "sam_vit_b": {"parameters_m": 91.0, "vram_mb": 950.0, "backbone": "ViT-B"},
        }

        baseline_sam1 = sam1_specs["sam_vit_b"]
        vram_reduction_pct = (1.0 - (sam2_spec["vram_mb"] / baseline_sam1["vram_mb"])) * 100.0
        param_reduction_pct = (1.0 - (sam2_spec["parameters_m"] / baseline_sam1["parameters_m"])) * 100.0

        return {
            "sam2_model": norm_type,
            "sam2_parameters_m": sam2_spec["parameters_m"],
            "sam2_vram_mb": sam2_spec["vram_mb"],
            "sam2_backbone": sam2_spec["backbone"],
            "sam1_baseline_model": "sam_vit_b",
            "sam1_parameters_m": baseline_sam1["parameters_m"],
            "sam1_vram_mb": baseline_sam1["vram_mb"],
            "vram_reduction_percent": round(vram_reduction_pct, 1),
            "parameter_reduction_percent": round(param_reduction_pct, 1),
            "estimated_speedup": sam2_spec["relative_speedup_vs_sam1"],
            "multiscale_hierarchical_features": sam2_spec["multiscale_features"],
        }
