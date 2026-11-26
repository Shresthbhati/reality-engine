"""ScanNet, ScanNet++, SUN RGB-D, and Replica Dataset Adapters (A4.1).

Provides structured ingestion, taxonomy mapping, camera intrinsics/pose
validation, depth-unit normalization, and terms-acceptance enforcement for:
  - ScanNetv2 (NYU40 labels, requires Terms of Use acceptance)
  - ScanNet++ (laser/DSLR/iPhone RGB-D, requires Terms of Use acceptance)
  - SUN RGB-D (oriented 2D/3D boxes & room layout)
  - Replica (synthetic photorealistic scans — explicitly flagged ``is_synthetic=True``)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from perception.datasets.indoor_registry import (
    INDOOR_DATASET_CATALOG,
    NormalizedDatasetSample,
    normalize_dataset_sample,
)

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
    "picture": "picture",
    "counter": "countertop",
    "blinds": "window_blinds",
    "desk": "desk",
    "shelves": "shelf",
    "curtain": "curtain",
    "dresser": "cabinet",
    "pillow": "pillow",
    "mirror": "mirror",
    "floor mat": "rug",
    "clothes": "clothing",
    "ceiling": "ceiling",
    "books": "book",
    "refrigerator": "refrigerator",
    "television": "tv",
    "paper": "paper",
    "towel": "towel",
    "shower curtain": "curtain",
    "box": "box",
    "whiteboard": "whiteboard",
    "person": "person",
    "night stand": "nightstand",
    "toilet": "toilet",
    "sink": "sink",
    "lamp": "lamp",
    "bathtub": "bathtub",
    "bag": "bag",
    "otherstructure": "architectural_structure",
    "otherfurniture": "furniture",
    "otherprop": "object",
}

SUNRGBD_CLASS_MAP: dict[str, str] = {
    "bed": "bed",
    "table": "table",
    "sofa": "sofa",
    "chair": "chair",
    "toilet": "toilet",
    "desk": "desk",
    "dresser": "cabinet",
    "night_stand": "nightstand",
    "bookshelf": "shelf",
    "bathtub": "bathtub",
    "sink": "sink",
    "counter": "countertop",
    "door": "door",
    "window": "window",
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "lamp": "lamp",
    "tv": "tv",
    "refrigerator": "refrigerator",
}

REPLICA_CLASS_MAP: dict[str, str] = {
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "chair": "chair",
    "sofa": "sofa",
    "table": "table",
    "desk": "desk",
    "door": "door",
    "window": "window",
    "cabinet": "cabinet",
    "bed": "bed",
    "tv-screen": "tv",
    "lamp": "lamp",
    "pillar": "column",
    "stair": "stairs",
}


class RestrictedDatasetTermsError(PermissionError):
    """Raised when attempting to load a terms-gated dataset without explicit acceptance."""


def _validate_camera_intrinsics(intrinsics: dict[str, Any] | None) -> dict[str, float] | None:
    if intrinsics is None:
        return None
    required = ("fx", "fy", "cx", "cy")
    parsed: dict[str, float] = {}
    for key in required:
        if key not in intrinsics:
            raise ValueError(f"Camera intrinsics missing required key {key!r}")
        val = float(intrinsics[key])
        if not math.isfinite(val):
            raise ValueError(f"Camera intrinsic {key!r} is non-finite: {val!r}")
        if key in ("fx", "fy") and val <= 0.0:
            raise ValueError(f"Focal length {key!r} must be positive, got {val!r}")
        parsed[key] = val
    return parsed


def _validate_pose_4x4(pose: Any) -> tuple[tuple[float, ...], ...] | None:
    if pose is None:
        return None
    if len(pose) != 4 or any(len(row) != 4 for row in pose):
        raise ValueError("camera_pose_4x4 must be a 4x4 matrix")
    rows: list[tuple[float, ...]] = []
    for row in pose:
        vals = tuple(float(v) for v in row)
        if any(not math.isfinite(v) for v in vals):
            raise ValueError("camera_pose_4x4 contains non-finite entries")
        rows.append(vals)
    return tuple(rows)


def _require_terms_if_needed(dataset_id: str, *, terms_accepted: bool) -> None:
    spec = INDOOR_DATASET_CATALOG[dataset_id]
    if spec.requires_terms_acceptance and not terms_accepted:
        raise RestrictedDatasetTermsError(
            f"Dataset {spec.name!r} ({spec.license}) requires explicit terms_accepted=True before ingestion."
        )


def load_scannet_samples(
    manifest_json: Path | str,
    *,
    dataset_id: str = "scannet",
    terms_accepted: bool = False,
) -> list[NormalizedDatasetSample]:
    """Load ScanNet or ScanNet++ local manifest after verifying Terms of Use acceptance."""
    if dataset_id not in ("scannet", "scannetpp"):
        raise ValueError(f"dataset_id must be 'scannet' or 'scannetpp', got {dataset_id!r}")
    _require_terms_if_needed(dataset_id, terms_accepted=terms_accepted)

    records = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    out: list[NormalizedDatasetSample] = []
    for rec in records:
        boxes = []
        for obj in rec.get("objects", ()):
            raw_lbl = str(obj.get("nyu40_label") or obj.get("label") or "unknown").strip().lower()
            mapped = SCANNET_NYU40_MAP.get(raw_lbl, raw_lbl)
            boxes.append({
                "label": mapped,
                "bbox_xyxy": obj.get("bbox_xyxy", [0, 0, 0, 0]),
                "is_occluded": bool(obj.get("is_occluded", False)),
                "is_truncated": bool(obj.get("is_truncated", False)),
            })
        out.append(normalize_dataset_sample({
            "sample_id": rec["sample_id"],
            "scene_id": rec.get("scene_id", rec["sample_id"]),
            "image_path": rec["image_path"],
            "depth_path": rec.get("depth_path"),
            "camera_intrinsics": _validate_camera_intrinsics(rec.get("camera_intrinsics")),
            "camera_pose_4x4": _validate_pose_4x4(rec.get("camera_pose_4x4")),
            "image_width": rec.get("width", 1296),
            "image_height": rec.get("height", 968),
            "boxes": boxes,
        }, dataset_id=dataset_id))
    return out


def load_sunrgbd_samples(manifest_json: Path | str) -> list[NormalizedDatasetSample]:
    """Load SUN RGB-D local manifest with oriented 2D boxes and RGB-D metadata."""
    records = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    out: list[NormalizedDatasetSample] = []
    for rec in records:
        boxes = []
        for obj in rec.get("objects", ()):
            raw_lbl = str(obj.get("label", "unknown")).strip().lower()
            mapped = SUNRGBD_CLASS_MAP.get(raw_lbl, raw_lbl)
            boxes.append({
                "label": mapped,
                "bbox_xyxy": obj.get("bbox_xyxy", [0, 0, 0, 0]),
                "is_occluded": bool(obj.get("occluded", False)),
            })
        out.append(normalize_dataset_sample({
            "sample_id": rec["sample_id"],
            "scene_id": rec.get("scene_id", rec["sample_id"]),
            "image_path": rec["image_path"],
            "depth_path": rec.get("depth_path"),
            "camera_intrinsics": _validate_camera_intrinsics(rec.get("camera_intrinsics")),
            "camera_pose_4x4": _validate_pose_4x4(rec.get("camera_pose_4x4")),
            "image_width": rec.get("width", 730),
            "image_height": rec.get("height", 530),
            "boxes": boxes,
        }, dataset_id="sunrgbd"))
    return out


def load_replica_samples(manifest_json: Path | str) -> list[NormalizedDatasetSample]:
    """Load Replica local manifest (explicitly flagged ``is_synthetic=True``)."""
    records = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    out: list[NormalizedDatasetSample] = []
    for rec in records:
        boxes = []
        for obj in rec.get("objects", ()):
            raw_lbl = str(obj.get("label", "unknown")).strip().lower()
            mapped = REPLICA_CLASS_MAP.get(raw_lbl, raw_lbl)
            boxes.append({"label": mapped, "bbox_xyxy": obj.get("bbox_xyxy", [0, 0, 0, 0])})
        out.append(normalize_dataset_sample({
            "sample_id": rec["sample_id"],
            "scene_id": rec.get("scene_id", rec["sample_id"]),
            "image_path": rec["image_path"],
            "depth_path": rec.get("depth_path"),
            "camera_intrinsics": _validate_camera_intrinsics(rec.get("camera_intrinsics")),
            "camera_pose_4x4": _validate_pose_4x4(rec.get("camera_pose_4x4")),
            "image_width": rec.get("width", 640),
            "image_height": rec.get("height", 480),
            "boxes": boxes,
            "is_synthetic": True,
            "is_physically_surveyed": False,
        }, dataset_id="replica"))
    return out


__all__ = [
    "REPLICA_CLASS_MAP",
    "RestrictedDatasetTermsError",
    "SCANNET_NYU40_MAP",
    "SUNRGBD_CLASS_MAP",
    "load_replica_samples",
    "load_scannet_samples",
    "load_sunrgbd_samples",
]
