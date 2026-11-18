"""Reading text on nameplates and labels (RE-PERC-000016).

Florence-2's ``<OCR>`` task, loaded through the same local-cache loader as the second-opinion detector. Text is
evidence with basis ``"ocr"``; the task returns NO confidence, so none is invented (``p`` stays ``None``). A missing
checkpoint raises ``Florence2Unavailable`` with the fetch command, never an empty "no text found".
"""
from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from perception.detection.florence2_detector import Florence2Detector, Florence2Unavailable  # noqa: F401 (re-exported)

TASK = "<OCR>"
Box = Tuple[float, float, float, float]


def read_text(image_rgb: np.ndarray, box_px: Optional[Box] = None, detector: Optional[Florence2Detector] = None,
              pad: float = 0.10) -> str:
    """Text found in the whole image, or in ``box_px`` (x0, y0, x1, y1) padded by ``pad`` of its size. Empty string
    means the model read nothing on the crop."""
    import torch
    from PIL import Image

    h, w = image_rgb.shape[:2]
    if box_px is not None:
        x0, y0, x1, y1 = box_px
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        x0, y0, x1, y1 = max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return ""
        image_rgb = image_rgb[y0:y1, x0:x1]
    image = Image.fromarray(np.ascontiguousarray(image_rgb)).convert("RGB")
    det = detector or Florence2Detector()
    model, processor = det._load()
    inputs = processor(text=TASK, images=image, return_tensors="pt").to(det._device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=128, num_beams=3, do_sample=False)
    raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(raw, task=TASK, image_size=image.size)
    return str(parsed.get(TASK, parsed) if isinstance(parsed, dict) else parsed).strip()


def _norm(s: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


def match_attribute(text: str, allowed: Mapping[str, Sequence[str]]) -> dict:
    """Map read text onto an attribute's allowed values. ``allowed``: value -> phrases that name it (the value itself
    is always a phrase). Exactly one value matched -> that value; none or several -> ``value: None`` (unknown), with
    the reason and every match kept. Whole words/phrases only, so "exit" does not match "exits"."""
    hay = _norm(text)
    hits = [v for v, phrases in allowed.items()
            if v != "unknown" and any(_norm(p) in hay for p in (v, *phrases) if p.strip())]
    if len(hits) == 1:
        return {"value": hits[0], "p": None, "basis": "ocr", "text": text, "matches": hits}
    reason = "no allowed value named in the text" if not hits else "text names several values"
    return {"value": None, "p": None, "basis": "ocr", "text": text, "matches": hits, "reason": reason}
