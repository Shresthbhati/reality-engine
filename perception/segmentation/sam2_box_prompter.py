"""SAM 2 box-prompted instance masks through Hugging Face ``transformers`` (``Sam2Model``).

An isolated adapter, deliberately separate from ``sam2_backend.py``: that backend needs the ``sam2`` pip package and offers
automatic masks; this one needs nothing beyond ``transformers`` and does one thing, which is what the multi-view perception
stage needs -- one mask per detector box. Checkpoint: ``facebook/sam2.1-hiera-small`` (Apache-2.0 code and weights,
~46M parameters), loaded from the local Hugging Face cache only. A missing checkpoint raises; it never degrades into "no masks".

    prompter = Sam2BoxPrompter()
    masks = prompter.masks_for_boxes(pil_image, [(x0, y0, x1, y1), ...])   # list of bool arrays, image-sized

Plain fp32 on purpose: Grounding DINO's fp16 variant crashed inside the API job, and the model is small enough not to need it.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

MODEL_ID = "facebook/sam2.1-hiera-small"
#: revision verified on 2026-11-15 (Apache-2.0 on the model card)
MODEL_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"


class Sam2Unavailable(RuntimeError):
    """The SAM 2 checkpoint or transformers support is missing; says how to fetch it."""


class Sam2BoxPrompter:
    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor

            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            processor = Sam2Processor.from_pretrained(self.model_id, local_files_only=True)
            model = Sam2Model.from_pretrained(self.model_id, local_files_only=True).to(dev).eval()
        except Exception as exc:  # noqa: BLE001 -- re-raised with the fix
            raise Sam2Unavailable(
                f"SAM 2 is not available locally ({exc}); fetch once with "
                f"python -c \"from huggingface_hub import snapshot_download as d; d('{self.model_id}')\"") from exc
        self._device, self._model, self._processor = dev, model, processor
        return model, processor

    def masks_for_boxes(self, image, boxes_xyxy: Sequence[Tuple[float, float, float, float]]) -> List[np.ndarray]:
        """One boolean mask (image height x width) per box, in the order given. ``image`` is a PIL image (RGB)."""
        if not boxes_xyxy:
            return []
        import torch

        model, processor = self._load()
        inputs = processor(images=image, input_boxes=[[list(map(float, b)) for b in boxes_xyxy]], return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = model(**inputs, multimask_output=False)
        # (images, boxes, masks per box, H, W) at the model's resolution -> original size, binarised
        masks = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(), binarize=True)[0]
        return [np.asarray(masks[i, 0].numpy(), dtype=bool) for i in range(masks.shape[0])]

    def unload(self) -> None:
        self._model = self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
