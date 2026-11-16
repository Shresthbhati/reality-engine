"""Florence-2 (``florence-community/Florence-2-base``, native in ``transformers``) as a SECOND-OPINION object detector.

Grounding DINO + SAM 2 is the pipeline's detector. Florence-2 is a different model family with different failure modes, so where the
two disagree a human has something to look at. Its ``<OD>`` task returns boxes and free-text labels but NO confidence, so none is
invented here. Loaded from the local Hugging Face cache only; a missing checkpoint raises ``Florence2Unavailable`` with the fetch
command (never an empty "found nothing").
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

MODEL_ID = "florence-community/Florence-2-base"
TASK = "<OD>"
Box = Tuple[float, float, float, float]


class Florence2Unavailable(RuntimeError):
    """The checkpoint or transformers support is missing; says how to fetch it."""


class Florence2Detector:
    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None) -> None:
        self.model_id, self._device, self._model, self._processor = model_id, device, None, None

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration

            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
            model = Florence2ForConditionalGeneration.from_pretrained(self.model_id, local_files_only=True).to(dev).eval()
        except Exception as exc:  # noqa: BLE001 -- re-raised with the fix
            raise Florence2Unavailable(
                f"Florence-2 is not available locally ({exc}); fetch once with "
                f"python -c \"from huggingface_hub import snapshot_download as d; d('{self.model_id}')\"") from exc
        self._device, self._model, self._processor = dev, model, processor
        return model, processor

    def unload(self) -> None:
        self._model = self._processor = None

    def detect(self, image) -> List[Dict[str, object]]:
        """``[{label, box_xyxy}]`` in the image's pixels, for a PIL RGB image. No scores (the task has none)."""
        import torch

        model, processor = self._load()
        inputs = processor(text=TASK, images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=512, num_beams=3, do_sample=False)
        text = processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(text, task=TASK, image_size=image.size)
        node = parsed.get(TASK, parsed)
        return [{"label": str(lab), "box_xyxy": tuple(float(v) for v in box)}
                for lab, box in zip(node.get("labels", []), node.get("bboxes", []))]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_agreement(primary: Sequence[dict], second: Sequence[dict], iou_min: float = 0.5) -> dict:
    """Where two detectors' boxes overlap (IoU >= ``iou_min``, one-to-one, greedy): ``matched`` pairs with both labels,
    ``primary_only`` and ``second_only`` lists for a human. ``primary`` items carry ``bbox_px``, ``second`` items ``box_xyxy``.
    This is agreement BETWEEN DETECTORS, not accuracy: both can be wrong the same way."""
    used, matched, primary_only = set(), [], []
    for p in primary:
        best, best_i = 0.0, None
        for i, s in enumerate(second):
            v = _iou(p["bbox_px"], s["box_xyxy"])
            if i not in used and v >= iou_min and v > best:
                best, best_i = v, i
        if best_i is None:
            primary_only.append({"label": p["label"], "bbox_px": p["bbox_px"]})
        else:
            used.add(best_i)
            matched.append({"primary": p["label"], "second": second[best_i]["label"], "iou": round(best, 3)})
    second_only = [{"label": s["label"], "bbox_px": [round(v, 1) for v in s["box_xyxy"]]} for i, s in enumerate(second) if i not in used]
    return {"matched": matched, "primary_only": primary_only, "second_only": second_only, "iou_min": iou_min}
