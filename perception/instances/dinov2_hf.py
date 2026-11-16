"""DINOv2 appearance embeddings through Hugging Face ``transformers`` (``facebook/dinov2-small``, Apache-2.0).

Why this exists next to ``dinov2_reid.py``: that module falls back to a hand-made colour/gradient statistic dressed up as an
embedding when ``torch.hub`` cannot load the model. Feeding that into object association would pass off a fake descriptor as
learned appearance. This adapter loads the real checkpoint from the local Hugging Face cache or raises ``Dinov2Unavailable``;
the caller records the skip. It never substitutes anything.

    emb = Dinov2Embedder()
    vectors = emb.embed_masked_crops(rgb_uint8_image, [bool_mask_1, bool_mask_2])   # unit-norm float32, one per mask
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

MODEL_ID = "facebook/dinov2-small"
INPUT_SIZE = 224
_MIN_MASK_PIXELS = 64   # below this the crop is mostly background fill and says nothing about the object


class Dinov2Unavailable(RuntimeError):
    """The DINOv2 checkpoint or transformers support is missing; says how to fetch it."""


class Dinov2Embedder:
    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from transformers import AutoModel

            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModel.from_pretrained(self.model_id, local_files_only=True).to(dev).eval()
        except Exception as exc:  # noqa: BLE001 -- re-raised with the fix
            raise Dinov2Unavailable(
                f"DINOv2 is not available locally ({exc}); fetch once with "
                f"python -c \"from huggingface_hub import snapshot_download as d; d('{self.model_id}')\"") from exc
        self._device, self._model = dev, model
        return model

    def unload(self) -> None:
        self._model = None

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """The mask's bounding box with everything outside the mask painted neutral grey, resized to the model input."""
        import cv2

        if mask.shape[:2] != image.shape[:2] or int(mask.sum()) < _MIN_MASK_PIXELS:
            return None
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = image[y0:y1, x0:x1].copy()
        crop[~mask[y0:y1, x0:x1]] = 128
        return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

    def embed_masked_crops(self, image: np.ndarray, masks: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
        """One unit-norm vector per mask (the CLS token), or None for a mask too small to describe an object."""
        import torch

        crops = [self._crop(image, np.asarray(m, dtype=bool)) for m in masks]
        keep = [i for i, c in enumerate(crops) if c is not None]
        out: List[Optional[np.ndarray]] = [None] * len(crops)
        if not keep:
            return out
        model = self._load()
        batch = np.stack([crops[i] for i in keep]).astype(np.float32) / 255.0
        batch = (batch - self._mean) / self._std
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self._device)
        with torch.no_grad():
            cls = model(pixel_values=tensor).last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
        for row, i in zip(cls, keep):
            norm = float(np.linalg.norm(row))
            out[i] = row / norm if norm > 1e-12 else None
        return out
