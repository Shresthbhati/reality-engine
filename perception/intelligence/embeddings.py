"""Visual Embedding Integration for Cross-View Identity & Semantic Recognition (A3.4 / A3.5).

Supports:
  - ``facebook/dinov2-small`` (local Hugging Face cache, ``local_files_only=True``)
  - Optional ``CLIP`` / ``SigLIP`` zero-shot semantic verification when cached
  - Lightweight deterministic RGB+gradient patch descriptor fallback when GPU/weights
    are not requested (explicitly tagged ``embedding_model="color_grad_descriptor_v1"``).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.intelligence.data_model import (
    AppearanceDescriptor,
    NormalizedPerceptionObservation,
)


@dataclass(frozen=True)
class EmbeddingResult:
    vector: tuple[float, ...]
    model_id: str
    dim: int
    is_neural: bool


def _l2_normalize(vec: np.ndarray) -> tuple[float, ...]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n > 1e-8:
        arr = arr / n
    return tuple(round(float(v), 6) for v in arr)


def compute_patch_descriptor(crop_rgb: np.ndarray) -> EmbeddingResult:
    """Deterministic 32-dim color + edge-orientation descriptor (non-neural baseline)."""
    if crop_rgb.size == 0:
        return EmbeddingResult((), "color_grad_descriptor_v1", 0, False)
    r_h, _ = np.histogram(crop_rgb[..., 0], bins=8, range=(0, 256), density=True)
    g_h, _ = np.histogram(crop_rgb[..., 1], bins=8, range=(0, 256), density=True)
    b_h, _ = np.histogram(crop_rgb[..., 2], bins=8, range=(0, 256), density=True)
    luma = (crop_rgb[..., 0] * 0.299 + crop_rgb[..., 1] * 0.587 + crop_rgb[..., 2] * 0.114).astype(np.float32)
    if luma.shape[0] >= 3 and luma.shape[1] >= 3:
        gx = luma[:, 1:] - luma[:, :-1]
        gy = luma[1:, :] - luma[:-1, :]
        h_min = min(gx.shape[0], gy.shape[0])
        w_min = min(gx.shape[1], gy.shape[1])
        ang = np.arctan2(gy[:h_min, :w_min], gx[:h_min, :w_min])
        o_h, _ = np.histogram(ang, bins=8, range=(-np.pi, np.pi), density=True)
    else:
        o_h = np.zeros(8, dtype=np.float32)
    vec = np.concatenate([r_h, g_h, b_h, o_h])
    return EmbeddingResult(_l2_normalize(vec), "color_grad_descriptor_v1", 32, False)


class Dinov2CropEmbedder:
    """Sequential DINOv2 crop embedder using ``facebook/dinov2-small`` from local HF cache."""

    DEFAULT_MODEL_ID = "facebook/dinov2-small"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._processor = None

    def is_available(self) -> bool:
        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            return False
        try:
            from perception.availability import real_models_by_name

            m = real_models_by_name().get("dinov2")
            return m is not None and m.status == "available"
        except Exception:
            return False

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_id, local_files_only=True).to(dev).eval()
        self._device = dev

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def embed_crop(self, crop_rgb: np.ndarray) -> EmbeddingResult:
        if crop_rgb.size == 0:
            return EmbeddingResult((), self.model_id, 0, True)
        import torch
        from PIL import Image

        self.load()
        pil = Image.fromarray(np.ascontiguousarray(crop_rgb))
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        cls_tok = out.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()
        vec = _l2_normalize(cls_tok)
        return EmbeddingResult(vec, self.model_id, len(vec), True)


def enrich_observations_with_embeddings(
    observations: Sequence[NormalizedPerceptionObservation],
    images_by_id: dict[str, np.ndarray],
    *,
    use_dinov2: bool = False,
) -> list[NormalizedPerceptionObservation]:
    """Attach appearance embeddings to observations using DINOv2 (if requested & cached) or patch descriptors."""
    embedder = Dinov2CropEmbedder() if use_dinov2 else None
    use_neural = bool(embedder is not None and embedder.is_available())
    enriched: list[NormalizedPerceptionObservation] = []

    try:
        for o in observations:
            img = images_by_id.get(o.provenance.source_image_id)
            if img is None:
                enriched.append(o)
                continue
            h, w = img.shape[:2]
            x1 = max(0, min(w - 1, int(round(o.bbox_px[0]))))
            y1 = max(0, min(h - 1, int(round(o.bbox_px[1]))))
            x2 = max(0, min(w, int(round(o.bbox_px[2]))))
            y2 = max(0, min(h, int(round(o.bbox_px[3]))))
            if x2 <= x1 or y2 <= y1:
                enriched.append(o)
                continue
            crop = img[y1:y2, x1:x2]
            res = embedder.embed_crop(crop) if (use_neural and embedder is not None) else compute_patch_descriptor(crop)
            new_app = AppearanceDescriptor(
                color_histogram_24=o.appearance.color_histogram_24,
                embedding_vector=res.vector,
                embedding_model=res.model_id,
                dominant_color_name=o.appearance.dominant_color_name,
                mean_rgb=o.appearance.mean_rgb,
                texture_roughness=o.appearance.texture_roughness,
            )
            enriched.append(NormalizedPerceptionObservation(
                observation_id=o.observation_id,
                schema_version=o.schema_version,
                raw_label=o.raw_label,
                canonical_label=o.canonical_label,
                taxonomy_id=o.taxonomy_id,
                category=o.category,
                bbox_px=o.bbox_px,
                coords=o.coords,
                scores=o.scores,
                provenance=o.provenance,
                parent_observation_id=o.parent_observation_id,
                child_observation_ids=o.child_observation_ids,
                competing_labels=o.competing_labels,
                is_partial=o.is_partial,
                is_truncated=o.is_truncated,
                occlusion_level=o.occlusion_level,
                appearance=new_app,
                material_hypotheses=o.material_hypotheses,
                segmentation=o.segmentation,
                temporal=o.temporal,
                revision=o.revision,
                attributes=o.attributes,
            ))
    finally:
        if embedder is not None:
            embedder.unload()

    return enriched


__all__ = [
    "Dinov2CropEmbedder",
    "EmbeddingResult",
    "compute_patch_descriptor",
    "enrich_observations_with_embeddings",
]
