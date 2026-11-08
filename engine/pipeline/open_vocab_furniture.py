"""Open-vocabulary furniture detection (Grounding DINO Tiny) for the single-image furniture placer.

COCO (the Mask R-CNN detector) has no coffee table, TV stand or bookshelf, and calls most room contents "potted plant" or
"vase". Grounding DINO takes the placer's own furniture vocabulary as text, so a living room's coffee table, TV stand and shelf
become placeable. Each class is queried SEPARATELY: a single combined prompt returns merged labels such as
"wardrobe bookshelf dresser tv stand", which cannot be assigned to one class without guessing.

Output is the placer's ``FurnitureDetection`` (boxes in the ``w`` x ``h`` grid it is given). The model is loaded from the local
Hugging Face cache only; absence raises (never an empty "nothing found"). ``combined_furniture_detector`` merges these with the
Mask R-CNN detections so a sofa Mask R-CNN is good at is never lost.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from perception.instances.furniture_from_detections import FurnitureDetection, _canon

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
#: text prompts, one query each (they map onto the placer's SIZE_PRIORS vocabulary through its aliases)
PROMPTS: Tuple[str, ...] = (
    "sofa", "armchair", "chair", "dining table", "coffee table", "desk", "bed", "wardrobe", "bookshelf", "dresser",
    "nightstand", "tv stand", "television", "floor lamp",
)
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
#: boxes covering more than this share of the frame are scene-level false positives, not objects
MAX_BOX_FRAME_SHARE = 0.80
MIN_BOX_FRAME_SHARE = 0.004
NMS_IOU = 0.5
BATCH = 7

_MODEL = None


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clean_detections(raw: Sequence[Tuple[str, float, Tuple[float, float, float, float]]], *, view_id: str,
                     src_size: Tuple[int, int], grid_size: Tuple[int, int]) -> List[FurnitureDetection]:
    """Filter, class-aware de-duplicate and rescale ``raw`` = [(label, score, box in source pixels)] to the placer's grid."""
    sw, sh = src_size
    gw, gh = grid_size
    kept: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    for label, score, box in sorted(raw, key=lambda r: -r[1]):
        if _canon(label) is None:
            continue
        share = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / float(sw * sh)
        if not (MIN_BOX_FRAME_SHARE <= share <= MAX_BOX_FRAME_SHARE):
            continue
        if any(_canon(k[0]) == _canon(label) and _iou(k[2], box) > NMS_IOU for k in kept):
            continue
        kept.append((label, score, box))
    sx, sy = gw / float(sw), gh / float(sh)
    return [FurnitureDetection(label=lab, score=float(sc), box_xyxy=(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), view_id=view_id)
            for lab, sc, b in kept]


def _load():
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            proc = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID, local_files_only=True).to(dev).eval()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Grounding DINO is not available locally ({exc}); fetch once with "
                f"python -c \"from huggingface_hub import snapshot_download as d; d('{MODEL_ID}')\"") from exc
        _MODEL = (proc, model, dev)
    return _MODEL


def unload_open_vocab_furniture() -> None:
    global _MODEL
    _MODEL = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _grounded_raw(path: Path):
    import torch
    from engine.core.image_io import open_oriented

    proc, model, dev = _load()
    img = open_oriented(path).convert("RGB")
    raw: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    for i in range(0, len(PROMPTS), BATCH):
        chunk = PROMPTS[i:i + BATCH]
        inp = proc(images=[img] * len(chunk), text=[f"{p}." for p in chunk], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[img.size[::-1]] * len(chunk))
        for prompt, r in zip(chunk, res):
            for score, box in zip(r["scores"].tolist(), r["boxes"].tolist()):
                raw.append((prompt, float(score), tuple(float(c) for c in box)))
    return raw, img.size


def grounded_furniture_detector(item, w: int, h: int) -> list:
    """Grounding DINO detections for ``item`` in the ``w`` x ``h`` grid. Raises when the model cannot run."""
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    u = urlparse(item.source_uri)
    path = Path(url2pathname(unquote(u.path)) if u.scheme == "file" else item.source_uri)
    raw, size = _grounded_raw(path)
    return clean_detections(raw, view_id=item.id, src_size=size, grid_size=(w, h))


def merge_detections(gd: Sequence[FurnitureDetection], rcnn: Sequence[FurnitureDetection]) -> List[FurnitureDetection]:
    """Union of two detectors' boxes: one label per physical object. Boxes that overlap are one object whatever each detector
    called it (a coffee table and a dining table cannot be the same spot), so the higher score wins across classes too."""
    merged: List[FurnitureDetection] = []
    for d in sorted(list(gd) + list(rcnn), key=lambda d: -d.score):
        if _canon(d.label) is None:
            merged.append(d)  # unknown labels stay so the placer records them as rejected, never silently dropped
            continue
        if any(_canon(m.label) is not None and _iou(m.box_xyxy, d.box_xyxy) > NMS_IOU for m in merged):
            continue
        merged.append(d)
    return merged


def combined_furniture_detector(item, w: int, h: int) -> list:
    """Grounding DINO + Mask R-CNN, merged class-aware. Grounding DINO unavailable -> Mask R-CNN alone; both unavailable -> raises."""
    from engine.pipeline.single_image_furniture import maskrcnn_furniture_detector

    rcnn = maskrcnn_furniture_detector(item, w, h)
    try:
        gd = grounded_furniture_detector(item, w, h)
    except Exception:  # noqa: BLE001 -- the open-vocabulary model is an enhancement; COCO detections still stand
        return rcnn
    return merge_detections(gd, rcnn)
