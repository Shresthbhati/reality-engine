"""Grounding DINO Tiny open-vocabulary detector (Apache-2.0; build-plan task 8.8, spec M2).

Local files only: never downloads at inference. Fetch once with
``huggingface-cli download IDEA-Research/grounding-dino-tiny`` (~700 MB).
Queries are chunked because the text encoder caps the prompt at 256 tokens; a returned phrase that is not one of the
queries (the model can return a partial or merged phrase) is mapped to the query it contains, else dropped.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from perception.detection.interface import DetectorUnavailable, ImageDetection

_REPO = "IDEA-Research/grounding-dino-tiny"
_CHUNK = 12


class GroundingDinoDetector:
    name = "grounding_dino_tiny"
    install = f"huggingface-cli download {_REPO}  (~700 MB; needs user approval)"

    def __init__(self, repo: str = _REPO, box_threshold: float = 0.3, text_threshold: float = 0.25):
        self.repo, self.box_threshold, self.text_threshold = repo, box_threshold, text_threshold
        self._model = self._proc = None

    def available(self) -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from huggingface_hub import try_to_load_from_cache
        except Exception as e:  # noqa: BLE001
            return False, f"Grounding DINO needs torch + transformers ({e}); then run: {self.install}"
        have_cfg = isinstance(try_to_load_from_cache(self.repo, "config.json"), str)
        have_w = any(isinstance(try_to_load_from_cache(self.repo, f), str)
                     for f in ("model.safetensors", "pytorch_model.bin"))
        if not (have_cfg and have_w):
            return False, f"Grounding DINO weights not in local HF cache; run: {self.install}"
        return True, "cached"

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            self._proc = AutoProcessor.from_pretrained(self.repo, local_files_only=True)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.repo, local_files_only=True).eval()

    def detect(self, image_rgb: np.ndarray, queries: Sequence[str]):
        ok, reason = self.available()
        if not ok:
            raise DetectorUnavailable(reason)
        import torch
        from PIL import Image

        self._load()
        qs = [q.strip().lower() for q in queries if q.strip()]
        h, w = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        out = []
        for i in range(0, len(qs), _CHUNK):
            chunk = qs[i:i + _CHUNK]
            inputs = self._proc(images=pil, text=". ".join(chunk) + ".", return_tensors="pt")
            with torch.no_grad():
                raw = self._model(**inputs)
            post = self._proc.post_process_grounded_object_detection
            try:
                res = post(raw, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
                           target_sizes=[(h, w)])[0]
            except TypeError:       # older transformers name the box threshold differently
                res = post(raw, inputs.input_ids, box_threshold=self.box_threshold,
                           text_threshold=self.text_threshold, target_sizes=[(h, w)])[0]
            labels = res.get("text_labels", res.get("labels"))
            for s, lab, b in zip(res["scores"], labels, res["boxes"]):
                lab = str(lab).strip().lower()
                q = lab if lab in chunk else next((c for c in chunk if c in lab), None)
                if q is None:
                    continue
                out.append(ImageDetection(q, float(s), tuple(float(v) for v in b.tolist()), source=self.name))
        return out
