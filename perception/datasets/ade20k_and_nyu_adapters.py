"""Additional Indoor Dataset Ingestion Adapters: ADE20K, NYU Depth V2, Structured3D, Hypersim (A4.1).

Normalizes each dataset's native metadata and class taxonomy into
``NormalizedDatasetSample`` while preserving licensing and synthetic-vs-surveyed
provenance.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from perception.datasets.indoor_registry import (
    NormalizedDatasetSample,
    normalize_dataset_sample,
)

ADE20K_CLASS_MAP: dict[str, str] = {
    "wall": "wall",
    "floor, flooring": "floor",
    "ceiling": "ceiling",
    "door, double door": "door",
    "windowpane, window": "window",
    "chair": "chair",
    "table": "table",
    "desk": "desk",
    "sofa, couch, lounge": "sofa",
    "bed": "bed",
    "cabinet": "cabinet",
    "wardrobe, closet, press": "wardrobe",
    "shelf": "shelf",
    "lamp": "lamp",
    "sink": "sink",
    "toilet, can, commode, crapper, pot, potty, stool, throne": "toilet",
    "bathtub, bathing tub, bath, tub": "bathtub",
    "refrigerator, icebox": "refrigerator",
    "stairs, steps": "staircase",
    "column, pillar": "column",
}

NYU_CLASS_MAP: dict[str, str] = {
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "door": "door",
    "window": "window",
    "chair": "chair",
    "table": "table",
    "desk": "desk",
    "sofa": "sofa",
    "bed": "bed",
    "cabinet": "cabinet",
    "bookshelf": "shelf",
    "counter": "countertop",
    "lamp": "lamp",
    "sink": "sink",
    "toilet": "toilet",
    "bathtub": "bathtub",
    "refrigerator": "refrigerator",
}


def load_ade20k_samples(manifest_json: Path | str) -> list[NormalizedDatasetSample]:
    """Load ADE20K-style JSON records ``[{sample_id, image_path, width, height, objects: [{name, bbox_xyxy}]}]``."""
    records = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    out: list[NormalizedDatasetSample] = []
    for rec in records:
        boxes = []
        for obj in rec.get("objects", ()):
            raw_name = str(obj.get("name", "unknown")).strip().lower()
            mapped = ADE20K_CLASS_MAP.get(raw_name, raw_name.split(",")[0].strip())
            boxes.append({
                "label": mapped,
                "bbox_xyxy": obj.get("bbox_xyxy", [0, 0, 0, 0]),
                "is_occluded": bool(obj.get("occluded", False)),
            })
        out.append(normalize_dataset_sample({
            "sample_id": rec["sample_id"],
            "scene_id": rec.get("scene_id", rec["sample_id"]),
            "image_path": rec["image_path"],
            "image_width": rec["width"],
            "image_height": rec["height"],
            "boxes": boxes,
        }, dataset_id="ade20k"))
    return out


def load_nyu_depth_v2_samples(manifest_json: Path | str) -> list[NormalizedDatasetSample]:
    """Load NYU Depth V2 records with RGB + depth paths."""
    records = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    out: list[NormalizedDatasetSample] = []
    for rec in records:
        boxes = []
        for obj in rec.get("objects", ()):
            raw_name = str(obj.get("label", "unknown")).strip().lower()
            mapped = NYU_CLASS_MAP.get(raw_name, raw_name)
            boxes.append({"label": mapped, "bbox_xyxy": obj.get("bbox_xyxy", [0, 0, 0, 0])})
        out.append(normalize_dataset_sample({
            "sample_id": rec["sample_id"],
            "scene_id": rec.get("scene_id", rec["sample_id"]),
            "image_path": rec["image_path"],
            "depth_path": rec.get("depth_path"),
            "image_width": rec.get("width", 640),
            "image_height": rec.get("height", 480),
            "boxes": boxes,
        }, dataset_id="nyu_depth_v2"))
    return out


def load_synthetic_indoor_samples(
    manifest_json: Path | str,
    *,
    dataset_id: str = "structured3d",
) -> list[NormalizedDatasetSample]:
    """Load Structured3D or Hypersim records (explicitly marked ``is_synthetic=True``)."""
    if dataset_id not in ("structured3d", "hypersim"):
        raise ValueError(f"dataset_id must be 'structured3d' or 'hypersim', got {dataset_id!r}")
    records = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    out: list[NormalizedDatasetSample] = []
    for rec in records:
        out.append(normalize_dataset_sample({
            "sample_id": rec["sample_id"],
            "scene_id": rec.get("scene_id", rec["sample_id"]),
            "image_path": rec["image_path"],
            "depth_path": rec.get("depth_path"),
            "camera_intrinsics": rec.get("camera_intrinsics"),
            "image_width": rec["width"],
            "image_height": rec["height"],
            "boxes": rec.get("boxes", ()),
            "is_synthetic": True,
            "is_physically_surveyed": False,
        }, dataset_id=dataset_id))
    return out


__all__ = [
    "ADE20K_CLASS_MAP",
    "NYU_CLASS_MAP",
    "load_ade20k_samples",
    "load_nyu_depth_v2_samples",
    "load_synthetic_indoor_samples",
]
