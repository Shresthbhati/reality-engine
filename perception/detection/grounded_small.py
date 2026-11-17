"""Grounding DINO as the detector behind the tiled small-fittings pass (perception.entities.small_objects).

``inspect_small_objects`` needs any object with ``detect(image_rgb, queries)`` and ``available()``; none was bundled. This adapter
runs the project's already-loaded Grounding DINO (fp32, local cache only) on an RGB array, so a 640 px tile can be searched for
sockets, switches, handles. Detections are unverified 2D evidence -- on the 10-photo room the tiled pass returned ~55 boxes per
photograph (61 "thermostats" over 10 photos), so callers keep them as facts and never promote them to entities without a labelled set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

#: stricter than the whole-frame furniture threshold (0.30): a tile always contains *something* the model wants to name
DEFAULT_THRESHOLD = 0.45


@dataclass(frozen=True)
class RawDetection:
    label: str
    score: float
    box_xyxy: Tuple[float, float, float, float]


class GroundedSmallDetector:
    name = "grounding-dino-tiny"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def available(self) -> Tuple[bool, str]:
        try:
            from engine.pipeline.open_vocab_furniture import _load

            _load()
            return True, ""
        except Exception as exc:  # noqa: BLE001 -- unavailable is a reported state, not a crash
            return False, f"Grounding DINO is not available locally: {exc}"

    def detect(self, image_rgb: np.ndarray, queries: Sequence[str]) -> List[RawDetection]:
        import torch
        from PIL import Image

        from engine.pipeline.open_vocab_furniture import BATCH, _load

        proc, model, dev = _load()
        img = Image.fromarray(np.ascontiguousarray(image_rgb))
        out: List[RawDetection] = []
        for i in range(0, len(queries), BATCH):
            chunk = list(queries[i:i + BATCH])
            inp = proc(images=[img] * len(chunk), text=[f"{q}." for q in chunk], return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                res = model(**inp)
            post = proc.post_process_grounded_object_detection(
                res, inp.input_ids, threshold=self.threshold, text_threshold=self.threshold,
                target_sizes=[img.size[::-1]] * len(chunk))
            for q, r in zip(chunk, post):
                for score, box in zip(r["scores"].tolist(), r["boxes"].tolist()):
                    out.append(RawDetection(q, float(score), tuple(float(c) for c in box)))
        return out
