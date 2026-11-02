"""Detector backends. Import-safe with no weights; NEVER downloads anything."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple, Union

import numpy as np

from perception.detection.interface import DetectorUnavailable, ImageDetection, register_detector
from perception.detection.policy import canonical

_OWL_REPO = "google/owlvit-base-patch32"


class FakeDetector:
    """Scripted detector for tests: {label: [(box, score[, mask]), ...]} or callable(image, queries)."""

    name = "fake"

    def __init__(self, script: Union[dict, Callable]):
        self._script = script

    def available(self) -> Tuple[bool, str]:
        return True, "scripted"

    def detect(self, image_rgb, queries):
        if callable(self._script):
            return list(self._script(image_rgb, queries))
        out = []
        for label, items in self._script.items():
            for it in items:
                mask = it[2] if len(it) > 2 else None
                out.append(ImageDetection(label, float(it[1]), tuple(float(v) for v in it[0]), mask,
                                          source=self.name))
        return out


class OwlVitDetector:
    name = "owlvit"
    install = f"huggingface-cli download {_OWL_REPO}  (~600 MB; needs user approval)"

    def __init__(self, repo: str = _OWL_REPO, threshold: float = 0.15):
        self.repo, self.threshold = repo, threshold
        self._model = self._proc = None

    def available(self) -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from huggingface_hub import try_to_load_from_cache
        except Exception as e:  # noqa: BLE001
            return False, f"OWL-ViT needs torch + transformers ({e}); then run: {self.install}"
        have_cfg = isinstance(try_to_load_from_cache(self.repo, "config.json"), str)
        have_w = any(isinstance(try_to_load_from_cache(self.repo, f), str)
                     for f in ("model.safetensors", "pytorch_model.bin"))
        if not (have_cfg and have_w):
            return False, f"OWL-ViT weights not in local HF cache; run: {self.install}"
        return True, "cached"

    def detect(self, image_rgb: np.ndarray, queries: Sequence[str]):
        ok, reason = self.available()
        if not ok:
            raise DetectorUnavailable(reason)
        import torch
        from PIL import Image
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        if self._model is None:  # local_files_only: never triggers a download
            self._proc = OwlViTProcessor.from_pretrained(self.repo, local_files_only=True)
            self._model = OwlViTForObjectDetection.from_pretrained(self.repo, local_files_only=True).eval()
        qs = list(queries)
        h, w = image_rgb.shape[:2]
        inputs = self._proc(text=[qs], images=Image.fromarray(image_rgb), return_tensors="pt")
        with torch.no_grad():
            out = self._model(**inputs)
        # transformers 5 renamed the processor method
        post = getattr(self._proc, "post_process_object_detection", None) or self._proc.post_process_grounded_object_detection
        res = post(out, threshold=self.threshold, target_sizes=torch.tensor([[h, w]]))[0]
        return [ImageDetection(qs[int(i)], float(s), tuple(float(v) for v in b.tolist()), source=self.name)
                for s, i, b in zip(res["scores"], res["labels"], res["boxes"])]


class TorchvisionCocoDetector:
    name = "torchvision-coco"

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        self._model = None
        self._cats = None

    @staticmethod
    def _weights():
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as W
        return W.COCO_V1

    def _ckpt(self) -> Path:
        import torch
        return Path(torch.hub.get_dir()) / "checkpoints" / Path(self._weights().url).name

    def available(self) -> Tuple[bool, str]:
        try:
            ckpt = self._ckpt()
        except Exception as e:  # noqa: BLE001
            return False, f"torchvision detector needs torch + torchvision ({e})"
        if not ckpt.exists():
            return False, (f"COCO weights not found at {ckpt}; to fetch (~160 MB, needs user approval): "
                           "python -c \"from torchvision.models.detection import "
                           "fasterrcnn_mobilenet_v3_large_320_fpn as f; f(weights='DEFAULT')\"")
        return True, "cached"

    def detect(self, image_rgb: np.ndarray, queries: Sequence[str]):
        ok, reason = self.available()
        if not ok:
            raise DetectorUnavailable(reason)
        import torch
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
        if self._model is None:
            m = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
            m.load_state_dict(torch.load(self._ckpt(), map_location="cpu", weights_only=True))
            self._model, self._cats = m.eval(), self._weights().meta["categories"]
        t = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            r = self._model([t])[0]
        want = {canonical(q) or q for q in queries} if queries else None
        out = []
        for b, s, lab in zip(r["boxes"], r["scores"], r["labels"]):
            label = self._cats[int(lab)]
            if float(s) < self.threshold or (want is not None and (canonical(label) or label) not in want):
                continue
            out.append(ImageDetection(label, float(s), tuple(float(v) for v in b.tolist()), source=self.name))
        return out


register_detector("owlvit", OwlVitDetector)
register_detector("torchvision-coco", TorchvisionCocoDetector)
