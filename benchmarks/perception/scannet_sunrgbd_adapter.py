"""Benchmark Adapters for Licensed Indoor Datasets (SUN RGB-D & ScanNetv2) (A3.11).

Normalizes SUN RGB-D 2D/3D oriented bounding boxes and ScanNetv2 semantic/instance
annotations into Reality Engine's ``GroundTruthBox`` and canonical taxonomy classes
without bundling or downloading dataset weights/images.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.perception.indoor_dataset_adapter import (
    GroundTruthBox,
    IndoorDatasetManifest,
)
from engine.multiengine.spatial_intelligence.taxonomy import resolve_taxonomy_class

SUN_RGBD_CLASS_MAP: dict[str, str] = {
    "chair": "chair",
    "table": "table",
    "desk": "desk",
    "bed": "bed",
    "sofa": "sofa",
    "sofa_chair": "armchair",
    "bookshelf": "shelf",
    "cabinet": "cabinet",
    "counter": "countertop",
    "dresser": "cabinet",
    "night_stand": "cabinet",
    "lamp": "lamp",
    "sink": "sink",
    "toilet": "toilet",
    "bathtub": "bathtub",
    "door": "door",
    "window": "window",
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "refrigerator": "refrigerator",
    "oven": "oven",
    "microwave": "microwave",
}

SCANNET_NYU40_MAP: dict[str, str] = {
    "wall": "wall",
    "floor": "floor",
    "cabinet": "cabinet",
    "bed": "bed",
    "chair": "chair",
    "sofa": "sofa",
    "table": "table",
    "door": "door",
    "window": "window",
    "bookshelf": "shelf",
    "counter": "countertop",
    "desk": "desk",
    "shelves": "shelf",
    "refrigerator": "refrigerator",
    "shower curtain": "curtain",
    "toilet": "toilet",
    "sink": "sink",
    "bathtub": "bathtub",
    "ceiling": "ceiling",
}


def normalize_indoor_dataset_label(raw_label: str, *, dataset_kind: str = "sunrgbd") -> str:
    """Map SUN RGB-D or ScanNetv2 class names to canonical Reality Engine taxonomy names."""
    clean = str(raw_label or "").strip().lower()
    mapping = SUN_RGBD_CLASS_MAP if dataset_kind == "sunrgbd" else SCANNET_NYU40_MAP
    mapped = mapping.get(clean, clean)
    return resolve_taxonomy_class(mapped).name


def load_sunrgbd_2d_annotations(
    manifest_json_path: Path | str,
    *,
    license_name: str = "SUN RGB-D Research License",
) -> tuple[IndoorDatasetManifest, dict[str, list[GroundTruthBox]]]:
    """Load SUN RGB-D-style JSON records ``[{image_id, objects: [{label, bbox_xyxy}]}]``."""
    p = Path(manifest_json_path)
    records = json.loads(p.read_text(encoding="utf-8"))
    gt_by_image: dict[str, list[GroundTruthBox]] = defaultdict(list)

    for rec in records:
        img_id = str(rec["image_id"])
        for obj in rec.get("objects", ()):
            raw_lbl = str(obj.get("label", "unknown"))
            canon = normalize_indoor_dataset_label(raw_lbl, dataset_kind="sunrgbd")
            box = obj.get("bbox_xyxy")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 <= x1 or y2 <= y1:
                continue
            gt_by_image[img_id].append(GroundTruthBox(
                image_id=img_id,
                raw_label=raw_lbl,
                canonical_label=canon,
                box_2d=(x1, y1, x2, y2),
                is_occluded=bool(obj.get("is_occluded", False)),
                is_truncated=bool(obj.get("is_truncated", False)),
            ))

    manifest = IndoorDatasetManifest(
        dataset_name="sunrgbd",
        license=license_name,
        is_physically_surveyed=True,
        samples=tuple({"image_id": k, "gt_count": len(v)} for k, v in sorted(gt_by_image.items())),
    )
    return manifest, dict(gt_by_image)


def load_scannet_2d_annotations(
    manifest_json_path: Path | str,
    *,
    license_name: str = "ScanNet Terms of Use (Non-Commercial Research)",
) -> tuple[IndoorDatasetManifest, dict[str, list[GroundTruthBox]]]:
    """Load ScanNetv2-style 2D projection records ``[{image_id, instances: [{nyu40_label, bbox_xywh}]}]``."""
    p = Path(manifest_json_path)
    records = json.loads(p.read_text(encoding="utf-8"))
    gt_by_image: dict[str, list[GroundTruthBox]] = defaultdict(list)

    for rec in records:
        img_id = str(rec["image_id"])
        for inst in rec.get("instances", ()):
            raw_lbl = str(inst.get("nyu40_label") or inst.get("label") or "unknown")
            canon = normalize_indoor_dataset_label(raw_lbl, dataset_kind="scannet")
            box = inst.get("bbox_xywh")
            if not box or len(box) != 4:
                continue
            x, y, w, h = (float(v) for v in box)
            if w <= 0 or h <= 0:
                continue
            gt_by_image[img_id].append(GroundTruthBox(
                image_id=img_id,
                raw_label=raw_lbl,
                canonical_label=canon,
                box_2d=(x, y, x + w, y + h),
                is_occluded=bool(inst.get("is_occluded", False)),
                is_truncated=bool(inst.get("is_truncated", False)),
            ))

    manifest = IndoorDatasetManifest(
        dataset_name="scannetv2",
        license=license_name,
        is_physically_surveyed=True,
        samples=tuple({"image_id": k, "gt_count": len(v)} for k, v in sorted(gt_by_image.items())),
    )
    return manifest, dict(gt_by_image)


__all__ = [
    "SCANNET_NYU40_MAP",
    "SUN_RGBD_CLASS_MAP",
    "load_scannet_2d_annotations",
    "load_sunrgbd_2d_annotations",
    "normalize_indoor_dataset_label",
]
