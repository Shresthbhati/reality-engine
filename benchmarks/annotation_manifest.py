"""Reviewable annotation manifest and multi-model comparison runner (Workstreams 2 & 3).

Strictly separates ``human_reviewed`` ground truth from ``model_pseudo_label``
entries so a model is never evaluated against its own predictions.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.dataset_adapters import compute_detection_metrics


class InvalidEvaluationGroundTruthError(ValueError):
    """Raised if model pseudo-labels are used as ground truth for evaluating that model."""


@dataclass(frozen=True)
class BoxAnnotation:
    label: str
    bbox_px: tuple[float, float, float, float]
    category: str = "object"  # "architecture" | "opening" | "object"
    ambiguous: bool = False
    unknown: bool = False


@dataclass(frozen=True)
class ImageAnnotation:
    image_id: str
    image_path: str
    width: int
    height: int
    source_kind: str  # "human_reviewed" | "model_pseudo_label"
    annotator_or_model: str
    boxes: tuple[BoxAnnotation, ...] = ()
    room_type: str | None = None
    relationships: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AnnotationManifest:
    manifest_id: str
    dataset_name: str
    split: str  # "train" | "val" | "test" | "unscored"
    images: tuple[ImageAnnotation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: Path) -> AnnotationManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        imgs = tuple(
            ImageAnnotation(
                image_id=im["image_id"],
                image_path=im["image_path"],
                width=int(im["width"]),
                height=int(im["height"]),
                source_kind=im["source_kind"],
                annotator_or_model=im["annotator_or_model"],
                boxes=tuple(
                    BoxAnnotation(
                        label=b["label"],
                        bbox_px=tuple(float(v) for v in b["bbox_px"]),  # type: ignore[arg-type]
                        category=b.get("category", "object"),
                        ambiguous=bool(b.get("ambiguous", False)),
                        unknown=bool(b.get("unknown", False)),
                    )
                    for b in im.get("boxes", ())
                ),
                room_type=im.get("room_type"),
                relationships=tuple(im.get("relationships", ())),
            )
            for im in raw.get("images", ())
        )
        return cls(
            manifest_id=raw["manifest_id"],
            dataset_name=raw["dataset_name"],
            split=raw["split"],
            images=imgs,
            metadata=dict(raw.get("metadata", {})),
        )


def evaluate_detector_on_manifest(
    manifest: AnnotationManifest,
    detector_name: str,
    predict_fn: Callable[[ImageAnnotation], Sequence[tuple[str, tuple[float, float, float, float]]]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a detector against ``human_reviewed`` annotations only."""
    reviewed = [im for im in manifest.images if im.source_kind == "human_reviewed"]
    for im in manifest.images:
        if im.source_kind == "model_pseudo_label" and im.annotator_or_model == detector_name and not reviewed:
            raise InvalidEvaluationGroundTruthError(
                f"Refusing to evaluate {detector_name!r} against its own pseudo-labels"
            )
    if not reviewed:
        return {
            "detector": detector_name,
            "status": "unscored_no_human_reviewed_gt",
            "images_evaluated": 0,
            "overall": None,
            "per_class": {},
        }

    all_pred: list[tuple[str, tuple[float, float, float, float]]] = []
    all_gt: list[tuple[str, tuple[float, float, float, float]]] = []
    per_class_pred: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    per_class_gt: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)

    for im in reviewed:
        preds = list(predict_fn(im))
        gts = [(b.label, b.bbox_px) for b in im.boxes if not b.unknown]
        all_pred.extend(preds)
        all_gt.extend(gts)
        for p in preds:
            per_class_pred[p[0].lower()].append(p)
        for g in gts:
            per_class_gt[g[0].lower()].append(g)

    overall = compute_detection_metrics(all_pred, all_gt, iou_threshold=iou_threshold)
    per_class = {
        cls_name: compute_detection_metrics(per_class_pred[cls_name], per_class_gt[cls_name], iou_threshold=iou_threshold)
        for cls_name in sorted(set(per_class_pred) | set(per_class_gt))
    }
    return {
        "detector": detector_name,
        "status": "scored",
        "images_evaluated": len(reviewed),
        "overall": overall,
        "per_class": per_class,
    }


def compare_detectors_on_manifest(
    manifest: AnnotationManifest,
    detectors: dict[str, Callable[[ImageAnnotation], Sequence[tuple[str, tuple[float, float, float, float]]]]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare multiple detectors (and optional ensembles) on the same manifest."""
    return {
        "manifest_id": manifest.manifest_id,
        "dataset_name": manifest.dataset_name,
        "split": manifest.split,
        "results": {
            name: evaluate_detector_on_manifest(manifest, name, fn, iou_threshold=iou_threshold)
            for name, fn in detectors.items()
        },
    }
