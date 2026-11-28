"""DINOv2 Crop Appearance Embedding Extractor (A5.3 / Objective 4).

Extracts L2-normalized visual appearance embeddings for detected object crops
using locally cached ``facebook/dinov2-small`` or ``facebook/dinov2-base`` models
under sequential GPU lifecycle management.

Never fabricates embeddings when the model is unavailable — returns observations
unchanged with explicit status diagnostics.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from perception.intelligence.data_model import (
    AppearanceDescriptor,
    NormalizedPerceptionObservation,
)


@dataclass(frozen=True)
class Dinov2ExtractionReport:
    model_id: str
    status: str  # "succeeded" | "unavailable" | "failed" | "skipped"
    device_used: str
    observations_embedded: int
    embedding_dim: int
    wall_time_s: float
    peak_vram_mb: float | None
    reason: str | None = None


def _is_hf_cached(repo_id: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + repo_id.replace("/", "--")
    return (hub / slug).exists()


def probe_dinov2_checkpoint() -> str | None:
    """Return the first locally cached DINOv2 checkpoint, or None."""
    for repo in ("facebook/dinov2-small", "facebook/dinov2-base"):
        if _is_hf_cached(repo):
            return repo
    return None


def extract_dinov2_embeddings(
    observations: Sequence[NormalizedPerceptionObservation],
    images_rgb: Mapping[str, np.ndarray],
    *,
    model_id: str | None = None,
    batch_size: int = 8,
    local_files_only: bool = True,
) -> tuple[list[NormalizedPerceptionObservation], Dinov2ExtractionReport]:
    """Attach DINOv2 L2-normalized crop embeddings to observations when cached locally."""
    t0 = time.perf_counter()
    if not observations or not images_rgb:
        return list(observations), Dinov2ExtractionReport(
            model_id=model_id or "none",
            status="skipped",
            device_used="cpu",
            observations_embedded=0,
            embedding_dim=0,
            wall_time_s=0.0,
            peak_vram_mb=None,
            reason="no_observations_or_images",
        )

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return list(observations), Dinov2ExtractionReport(
            model_id=model_id or "none",
            status="unavailable",
            device_used="cpu",
            observations_embedded=0,
            embedding_dim=0,
            wall_time_s=0.0,
            peak_vram_mb=None,
            reason="torch_or_transformers_missing",
        )

    ckpt = model_id or probe_dinov2_checkpoint()
    if ckpt is None or (local_files_only and not _is_hf_cached(ckpt)):
        return list(observations), Dinov2ExtractionReport(
            model_id=ckpt or "facebook/dinov2-small",
            status="unavailable",
            device_used="cpu",
            observations_embedded=0,
            embedding_dim=0,
            wall_time_s=0.0,
            peak_vram_mb=None,
            reason="dinov2_not_in_local_hf_cache",
        )

    import PIL.Image as PILImage
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        processor = AutoImageProcessor.from_pretrained(ckpt, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(ckpt, local_files_only=local_files_only).to(device).eval()
    except Exception as exc:
        return list(observations), Dinov2ExtractionReport(
            model_id=ckpt,
            status="failed",
            device_used=device,
            observations_embedded=0,
            embedding_dim=0,
            wall_time_s=round(time.perf_counter() - t0, 4),
            peak_vram_mb=None,
            reason=f"load_failed:{type(exc).__name__}:{exc}",
        )

    crops: list[PILImage.Image] = []
    crop_indices: list[int] = []
    for idx, o in enumerate(observations):
        img = images_rgb.get(o.provenance.source_image_id)
        if img is None or img.ndim != 3:
            continue
        ih, iw = img.shape[:2]
        x1 = max(0, min(iw - 1, int(round(o.bbox_px[0]))))
        y1 = max(0, min(ih - 1, int(round(o.bbox_px[1]))))
        x2 = max(x1 + 2, min(iw, int(round(o.bbox_px[2]))))
        y2 = max(y1 + 2, min(ih, int(round(o.bbox_px[3]))))
        crops.append(PILImage.fromarray(np.asarray(img[y1:y2, x1:x2], dtype=np.uint8)))
        crop_indices.append(idx)

    emb_by_idx: dict[int, tuple[float, ...]] = {}
    emb_dim = 0
    try:
        for start in range(0, len(crops), max(1, batch_size)):
            batch_imgs = crops[start : start + batch_size]
            batch_idxs = crop_indices[start : start + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                cls_tok = outputs.last_hidden_state[:, 0, :]
                normed = torch.nn.functional.normalize(cls_tok, p=2, dim=-1)
            vecs = normed.detach().cpu().numpy()
            if vecs.ndim == 2 and vecs.shape[0] == len(batch_idxs):
                emb_dim = int(vecs.shape[1])
                for j, obs_idx in enumerate(batch_idxs):
                    emb_by_idx[obs_idx] = tuple(round(float(v), 5) for v in vecs[j].tolist())
    finally:
        peak_vram = (
            round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
            if device == "cuda"
            else None
        )
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    updated: list[NormalizedPerceptionObservation] = []
    for idx, o in enumerate(observations):
        vec = emb_by_idx.get(idx)
        if vec is None:
            updated.append(o)
            continue
        new_app = AppearanceDescriptor(
            color_histogram_24=o.appearance.color_histogram_24,
            embedding_vector=vec,
            embedding_model=ckpt,
            dominant_color_name=o.appearance.dominant_color_name,
            mean_rgb=o.appearance.mean_rgb,
            texture_roughness=o.appearance.texture_roughness,
        )
        updated.append(replace(o, appearance=new_app))

    return updated, Dinov2ExtractionReport(
        model_id=ckpt,
        status="succeeded" if emb_by_idx else "failed",
        device_used=device,
        observations_embedded=len(emb_by_idx),
        embedding_dim=emb_dim,
        wall_time_s=round(time.perf_counter() - t0, 4),
        peak_vram_mb=peak_vram,
    )


__all__ = [
    "Dinov2ExtractionReport",
    "extract_dinov2_embeddings",
    "probe_dinov2_checkpoint",
]
