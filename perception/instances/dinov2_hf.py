"""DINOv2 appearance embeddings through Hugging Face ``transformers`` (``facebook/dinov2-small``, Apache-2.0).

Why this exists next to ``dinov2_reid.py``: that module falls back to a hand-made colour/gradient statistic dressed up as an
embedding when ``torch.hub`` cannot load the model. Feeding that into object association would pass off a fake descriptor as
learned appearance. This adapter loads the real checkpoint from the local Hugging Face cache or raises ``Dinov2Unavailable``;
the caller records the skip. It never substitutes anything.

    emb = Dinov2Embedder()
    vectors = emb.embed_masked_crops(rgb_uint8_image, [bool_mask_1, bool_mask_2])   # unit-norm float32, one per mask
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import List

import numpy as np

MODEL_ID = "facebook/dinov2-small"
INPUT_SIZE = 224
_MIN_MASK_PIXELS = 64   # below this the crop is mostly background fill and says nothing about the object
PATCH = 14              # DINOv2 ViT-S/14 patch size
POOLED_GRID = 32        # embed_pooled resizes the photo to 32x32 patches (448 px)
MIN_POOLED_WEIGHT = 1.0  # total patch coverage (in patches) a mask needs; less is a speck, not an object


def pool_tokens(tokens: np.ndarray, weights: Sequence[np.ndarray | None]) -> List[np.ndarray | None]:
    """Unit-norm weighted mean of ``tokens`` (P, D) per weight vector (P,): the mask's coverage of each patch.
    None for a missing/mis-sized weight vector, total coverage under ``MIN_POOLED_WEIGHT`` patches, or a zero mean."""
    out: List[np.ndarray | None] = []
    for w in weights:
        if w is None or len(w) != len(tokens) or float(np.sum(w)) < MIN_POOLED_WEIGHT:
            out.append(None)
            continue
        v = (np.asarray(w, dtype=np.float32)[:, None] * tokens).sum(axis=0) / float(np.sum(w))
        n = float(np.linalg.norm(v))
        out.append(v / n if n > 1e-12 else None)
    return out


class Dinov2Unavailable(RuntimeError):
    """The DINOv2 checkpoint or transformers support is missing; says how to fetch it."""


class Dinov2Embedder:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
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

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """The mask's bounding box with everything outside the mask painted neutral grey, resized to the model input."""
        import cv2

        if mask.shape[:2] != image.shape[:2] or int(mask.sum()) < _MIN_MASK_PIXELS:
            return None
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = image[y0:y1, x0:x1].copy()
        crop[~mask[y0:y1, x0:x1]] = 128
        return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

    def embed_pooled(self, image: np.ndarray, masks: Sequence[np.ndarray]) -> List[np.ndarray | None]:
        """One forward pass for the WHOLE image; each mask's vector is the mean of the patch tokens under it, weighted by how much
        of each patch the mask covers (RE-PERF-000007). Cost is one forward per image however many objects it holds, and an
        object is described in the context it was seen in. NOT the same descriptor as ``embed_masked_crops`` (CLS of an
        isolated, grey-padded crop): the two are not comparable with each other. None for a mask smaller than ~one patch."""
        import cv2
        import torch

        h, w = image.shape[:2]
        side = PATCH * POOLED_GRID
        small = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        model = self._load()
        tensor = torch.from_numpy((small - self._mean) / self._std).permute(2, 0, 1)[None].to(self._device)
        with torch.no_grad():
            tokens = model(pixel_values=tensor).last_hidden_state[0, 1:1 + POOLED_GRID * POOLED_GRID].cpu().numpy().astype(np.float32)
        weights = []
        for m in masks:
            m = np.asarray(m, dtype=np.float32)
            weights.append(cv2.resize(m, (POOLED_GRID, POOLED_GRID), interpolation=cv2.INTER_AREA).ravel()
                           if m.shape[:2] == (h, w) else None)
        return pool_tokens(tokens, weights)

    def embed_masked_crops(self, image: np.ndarray, masks: Sequence[np.ndarray]) -> List[np.ndarray | None]:
        """One unit-norm vector per mask (the CLS token), or None for a mask too small to describe an object."""
        import torch

        crops = [self._crop(image, np.asarray(m, dtype=bool)) for m in masks]
        keep = [i for i, c in enumerate(crops) if c is not None]
        out: List[np.ndarray | None] = [None] * len(crops)
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
