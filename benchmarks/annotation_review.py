"""Human-review loop for object detections: pipeline predictions -> CVAT / Label Studio / FiftyOne -> reviewed labels -> scores.

Every accuracy number for objects is blocked on human-checked labels. The pipeline records its 2D detections per photograph
(``perception.detections_2d``); this module turns them into review packages and reads the corrected labels back:

    coco = to_coco(detections_2d, {evidence_id: image_path})          # CVAT: Tasks > Upload annotations > COCO 1.0
    ls   = to_label_studio(detections_2d, {evidence_id: image_url})   # Label Studio: Import > JSON (with LABEL_STUDIO_CONFIG)
    ds   = to_fiftyone(coco_path, images_dir)                         # FiftyOne dataset, when the library is installed
    gt   = read_coco(reviewed_coco_path)                              # after a human corrected the boxes
    report = score(detections_2d, gt, iou_min=0.5)                    # per-label precision / recall, honest about empty sets

Nothing here invents labels: an image nobody reviewed is not scored, and an empty review is reported as unreviewed, not as zero
objects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]

LABEL_STUDIO_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
{labels}
  </RectangleLabels>
</View>"""


class UnreviewedTrainingDataError(ValueError):
    """Raised when unreviewed model predictions/pre-labels are supplied for training (RE-DATA-000005)."""


def validate_training_dataset(data: Sequence[dict] | Mapping) -> list[dict]:
    """Ensure every annotation record has review_status == 'reviewed' before any training use.

    (RE-DATA-000005: Model pre-labels could become truth -> reviewed flag required before any training use).
    Raises UnreviewedTrainingDataError if any unreviewed record or pre-label draft is found.
    """
    records: Sequence[dict]
    if isinstance(data, Mapping) and "records" in data:
        if str(data.get("status", "")).startswith("DRAFT"):
            raise UnreviewedTrainingDataError(f"Cannot use unreviewed draft data for training: {data.get('status')}")
        records = data["records"]
    elif isinstance(data, Mapping) and "annotations" in data:
        records = data["annotations"]
    elif isinstance(data, Sequence):
        records = data
    else:
        raise TypeError(f"Unsupported dataset format: {type(data)}")

    validated = []
    for r in records:
        status = r.get("review_status")
        source = (r.get("attributes") or {}).get("source") if isinstance(r.get("attributes"), dict) else r.get("source")
        if status == "unreviewed" or source in ("prediction", "second_opinion", "prelabel", "draft"):
            raise UnreviewedTrainingDataError(
                f"Record {r.get('entity_id') or r.get('id')} has unreviewed status {status!r} / source {source!r}; "
                "training requires human-reviewed truth."
            )
        if status != "reviewed" and not r.get("is_ground_truth") and source != "ground_truth":
            raise UnreviewedTrainingDataError(
                f"Record {r.get('entity_id') or r.get('id')} missing explicit review_status='reviewed'."
            )
        validated.append(r)
    return validated


def _size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _labels(detections: Mapping[str, Sequence[dict]]) -> List[str]:
    return sorted({d["label"] for rows in detections.values() for d in rows})


def to_coco(detections: Mapping[str, Sequence[dict]], images: Mapping[str, str | Path], sizes: Optional[Mapping[str, Tuple[int, int]]] = None,
            extra_boxes: Optional[Mapping[str, Sequence[dict]]] = None, extra_info: Optional[dict] = None) -> dict:
    """COCO 1.0 detection JSON. ``images`` maps evidence id -> file; ``sizes`` (evidence id -> (w, h)) avoids opening files.
    Predictions carry ``score`` and the pipeline's ``region_id`` as attributes so a reviewer's edits can be matched back.
    ``extra_boxes`` ({evidence id: [{label, bbox_px}]}) are boxes the pipeline did NOT produce (a second detector's disagreements): they
    are exported with ``attributes.source == "second_opinion"`` and no score, for the reviewer to keep or delete. ``extra_info`` is merged
    into the file's ``info`` block (COCO readers ignore unknown keys)."""
    extra_boxes = extra_boxes or {}
    cats = {name: i + 1 for i, name in enumerate(sorted(set(_labels(detections)) | {b["label"] for rows in extra_boxes.values() for b in rows}))}
    out_images, anns = [], []
    for img_id, (eid, path) in enumerate(sorted(images.items()), start=1):
        w, h = (sizes or {}).get(eid) or _size(Path(path))
        out_images.append({"id": img_id, "file_name": Path(path).name, "width": w, "height": h, "evidence_id": eid})
        for d in detections.get(eid, ()):
            x0, y0, x1, y1 = d["bbox_px"]
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cats[d["label"]],
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0,
                         "score": d.get("confidence"), "attributes": {"region_id": d.get("region_id"), "source": "prediction"}})
        for b in extra_boxes.get(eid, ()):
            x0, y0, x1, y1 = b["bbox_px"]
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cats[b["label"]],
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0,
                         "score": b.get("score"), "attributes": {"region_id": None, "source": b.get("source", "second_opinion")}})
    return {"info": {"description": "Reality Engine detection predictions for human review", **(extra_info or {})},
            "categories": [{"id": i, "name": n} for n, i in cats.items()], "images": out_images, "annotations": anns}


def coco_from_perception(perception: Mapping, images: Mapping[str, str | Path], sizes: Optional[Mapping[str, Tuple[int, int]]] = None,
                         version_id: Optional[str] = None) -> dict:
    """The one review package a reviewer needs, built from a version's recorded perception facts: the pipeline's detections, the
    second detector's disagreements as extra (marked) boxes, and in ``info.reality_engine`` what the plausibility filter dropped. Only
    photographs present in ``images`` are included. Raises ValueError when no detections were recorded."""
    detections = perception.get("detections_2d")
    if not detections:
        raise ValueError("no 2D detections were recorded (the perception stage did not run, or found nothing)")
    second = perception.get("second_opinion") or {}
    extra = {eid: [{"label": b["label"], "bbox_px": b["bbox_px"], "source": "second_opinion"} for b in (v.get("second_only") or [])]
             for eid, v in (second.get("per_view") or {}).items() if eid in images}
    small = perception.get("small_fittings") or {}
    for d in small.get("detections") or []:                      # unverified tiled detections: a reviewer keeps the real ones
        if d.get("view") in images:
            extra.setdefault(d["view"], []).append({"label": d["class"], "bbox_px": d["box_px"], "score": d.get("score"), "source": "small_fitting"})
    info = {"reality_engine": {
        "small_fittings": ({"status": small.get("status"), "detections_by_class": small.get("detections_by_class")} if small else None),
        "version_id": version_id,
        "dropped_by_plausibility_filter": (perception.get("resolution") or {}).get("dropped_detail", []),
        "second_opinion": ({"status": second.get("status"), "detector": second.get("detector"), "totals": second.get("totals")} if second else None),
        "note": "attributes.source == 'second_opinion' boxes were NOT produced by the pipeline: keep the real ones, delete the rest"}}
    return to_coco({k: v for k, v in detections.items() if k in images}, images, sizes, extra_boxes=extra, extra_info=info)


def to_label_studio(detections: Mapping[str, Sequence[dict]], image_urls: Mapping[str, str],
                    sizes: Mapping[str, Tuple[int, int]], model_version: str = "reality-engine") -> Tuple[List[dict], str]:
    """(tasks, labeling_config_xml): one task per photograph with the pipeline's boxes as editable predictions (percent units)."""
    tasks = []
    for eid, url in sorted(image_urls.items()):
        w, h = sizes[eid]
        results = [{"id": d.get("region_id") or f"{eid}-{i}", "type": "rectanglelabels", "from_name": "label", "to_name": "image",
                    "original_width": w, "original_height": h, "image_rotation": 0,
                    "value": {"x": 100 * d["bbox_px"][0] / w, "y": 100 * d["bbox_px"][1] / h,
                              "width": 100 * (d["bbox_px"][2] - d["bbox_px"][0]) / w,
                              "height": 100 * (d["bbox_px"][3] - d["bbox_px"][1]) / h, "rotation": 0,
                              "rectanglelabels": [d["label"]]}} for i, d in enumerate(detections.get(eid, ()))]
        scores = [d.get("confidence") or 0.0 for d in detections.get(eid, ())]
        tasks.append({"data": {"image": url, "evidence_id": eid},
                      "predictions": [{"model_version": model_version, "score": (sum(scores) / len(scores)) if scores else 0.0,
                                       "result": results}]})
    labels = "\n".join(f'    <Label value="{n}"/>' for n in _labels(detections))
    return tasks, LABEL_STUDIO_CONFIG.format(labels=labels)


def to_fiftyone(coco: dict, images_dir: str | Path, name: str = "reality-engine-review"):
    """A FiftyOne dataset of the predictions for browsing and correcting. Raises ImportError with the install hint when absent."""
    try:
        import fiftyone as fo
    except ImportError as exc:
        raise ImportError("fiftyone is not installed (pip install fiftyone); the COCO export works without it") from exc
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_img: Dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    samples = []
    for im in coco["images"]:
        dets = [fo.Detection(label=cats[a["category_id"]], confidence=a.get("score"),
                             bounding_box=[a["bbox"][0] / im["width"], a["bbox"][1] / im["height"],
                                           a["bbox"][2] / im["width"], a["bbox"][3] / im["height"]])
                for a in by_img.get(im["id"], ())]
        s = fo.Sample(filepath=str(Path(images_dir) / im["file_name"]), evidence_id=im.get("evidence_id"))
        s["predictions"] = fo.Detections(detections=dets)
        samples.append(s)
    ds = fo.Dataset(name, overwrite=True)
    ds.add_samples(samples)
    return ds


def read_coco(path_or_dict) -> Dict[str, List[dict]]:
    """Reviewed COCO -> ``{evidence_id: [{label, bbox_px}]}``. Images the export did not tag with an evidence id fall back to the
    file name, so a CVAT round-trip that drops custom fields still maps."""
    coco = path_or_dict if isinstance(path_or_dict, dict) else json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    key = {im["id"]: (im.get("evidence_id") or im["file_name"]) for im in coco["images"]}
    out: Dict[str, List[dict]] = {k: [] for k in key.values()}      # an image present with no boxes IS reviewed-empty
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        out[key[a["image_id"]]].append({"label": cats[a["category_id"]], "bbox_px": [x, y, x + w, y + h]})
    return out


def score_sources(exported_coco: dict, reviewed_coco: dict, iou_min: float = 0.5) -> dict:
    """How much of what each EXTRA source proposed did a human keep? For every box the export marked ``second_opinion`` or
    ``small_fitting``, it counts as kept when the reviewed file has a box with the same label at IoU >= ``iou_min`` on that image.
    Precision per source and per label, over REVIEWED images only; ``unreviewed`` when the review covers none of them. This measures
    how useful a source's proposals are to a reviewer, not detector accuracy in general."""
    reviewed = read_coco(reviewed_coco)
    cats = {c["id"]: c["name"] for c in exported_coco["categories"]}
    key = {im["id"]: (im.get("evidence_id") or im["file_name"]) for im in exported_coco["images"]}
    tally: Dict[str, Dict[str, Dict[str, int]]] = {}
    for a in exported_coco["annotations"]:
        src = (a.get("attributes") or {}).get("source")
        eid = key[a["image_id"]]
        if src in (None, "prediction") or eid not in reviewed:
            continue
        x, y, w, h = a["bbox"]
        label = cats[a["category_id"]]
        hit = any(g["label"] == label and _iou([x, y, x + w, y + h], g["bbox_px"]) >= iou_min for g in reviewed[eid])
        t = tally.setdefault(src, {}).setdefault(label, {"proposed": 0, "kept": 0})
        t["proposed"] += 1
        t["kept"] += int(hit)
    if not tally:
        return {"status": "unreviewed", "note": "no extra-source boxes on reviewed images", "sources": {}}
    out = {}
    for src, labels in tally.items():
        proposed, kept = sum(v["proposed"] for v in labels.values()), sum(v["kept"] for v in labels.values())
        out[src] = {"proposed": proposed, "kept": kept, "precision": kept / proposed if proposed else None,
                    "per_label": {k: {**v, "precision": v["kept"] / v["proposed"]} for k, v in sorted(labels.items())}}
    return {"status": "scored", "iou_min": iou_min, "sources": out,
            "note": "share of each source's proposals that a human kept; not detector accuracy in general"}


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score(predictions: Mapping[str, Sequence[dict]], truth: Mapping[str, Sequence[dict]], iou_min: float = 0.5) -> dict:
    """Per-label precision / recall of ``predictions`` against reviewed ``truth``, greedy one-to-one matching by IoU. Only
    reviewed images are scored (those present in ``truth``); with none, ``status`` is ``unreviewed`` and no number is produced."""
    reviewed = [eid for eid in truth]
    if not reviewed:
        return {"status": "unreviewed", "note": "no reviewed images: nothing to score", "per_label": {}}
    tally: Dict[str, Dict[str, int]] = {}
    for eid in reviewed:
        preds = sorted(predictions.get(eid, ()), key=lambda d: -(d.get("confidence") or 0.0))
        gts = list(truth[eid])
        used = set()
        for p in preds:
            t = tally.setdefault(p["label"], {"tp": 0, "fp": 0, "fn": 0})
            best, best_i = 0.0, None
            for i, g in enumerate(gts):
                if i in used or g["label"] != p["label"]:
                    continue
                v = _iou(p["bbox_px"], g["bbox_px"])
                if v >= iou_min and v > best:
                    best, best_i = v, i
            if best_i is None:
                t["fp"] += 1
            else:
                used.add(best_i)
                t["tp"] += 1
        for i, g in enumerate(gts):
            if i not in used:
                tally.setdefault(g["label"], {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
    per = {}
    for label, t in sorted(tally.items()):
        tp, fp, fn = t["tp"], t["fp"], t["fn"]
        per[label] = {**t, "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}
    return {"status": "scored", "images_reviewed": len(reviewed), "iou_min": iou_min, "per_label": per,
            "note": "2D box agreement with human-reviewed labels; says nothing about 3D position or size"}
