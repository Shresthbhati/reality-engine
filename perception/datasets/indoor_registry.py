"""Real Indoor Dataset Infrastructure & Licensing Registry (A4.1).

Supports cataloguing, licensing verification, scene-disjoint train/val/test
splitting, integrity validation, and annotation normalization for:
  - ScanNet / ScanNet++
  - SUN RGB-D
  - NYU Depth V2
  - ADE20K
  - Structured3D
  - Hypersim
  - Replica

Never downloads restricted datasets without terms/credentials, and never
presents synthetic datasets (Structured3D, Hypersim, Replica) as physical-world
survey validation.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from engine.multiengine.spatial_intelligence.taxonomy import resolve_taxonomy_class


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    name: str
    license: str
    commercial_ok: bool
    requires_terms_acceptance: bool
    is_synthetic: bool
    is_physically_surveyed: bool
    annotations_available: tuple[str, ...]
    has_depth: bool
    has_camera_poses: bool
    depth_units: str  # "meters" | "millimeters" | "none"
    url: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


INDOOR_DATASET_CATALOG: dict[str, DatasetSpec] = {
    "scannet": DatasetSpec(
        dataset_id="scannet",
        name="ScanNetv2",
        license="ScanNet Terms of Use (Non-Commercial Research)",
        commercial_ok=False,
        requires_terms_acceptance=True,
        is_synthetic=False,
        is_physically_surveyed=True,
        annotations_available=("bbox_2d", "bbox_3d", "instance_mask", "semantic_mask", "mesh"),
        has_depth=True,
        has_camera_poses=True,
        depth_units="millimeters",
        url="http://www.scan-net.org/",
        notes="Requires signed Terms of Use; 1,513 indoor scans with NYU40 labels.",
    ),
    "scannetpp": DatasetSpec(
        dataset_id="scannetpp",
        name="ScanNet++",
        license="ScanNet++ Terms of Use (Non-Commercial Research)",
        commercial_ok=False,
        requires_terms_acceptance=True,
        is_synthetic=False,
        is_physically_surveyed=True,
        annotations_available=("bbox_3d", "instance_mask", "semantic_mask", "laser_scan"),
        has_depth=True,
        has_camera_poses=True,
        depth_units="millimeters",
        url="https://kaldir.vc.in.tum.de/scannetpp/",
        notes="Sub-millimeter Faro laser scans + DSLR + iPhone RGB-D.",
    ),
    "sunrgbd": DatasetSpec(
        dataset_id="sunrgbd",
        name="SUN RGB-D",
        license="SUN RGB-D Research License (MIT-style code, mixed data sources)",
        commercial_ok=False,
        requires_terms_acceptance=False,
        is_synthetic=False,
        is_physically_surveyed=True,
        annotations_available=("bbox_2d", "bbox_3d", "semantic_mask", "room_layout"),
        has_depth=True,
        has_camera_poses=True,
        depth_units="millimeters",
        url="https://rgbd.cs.princeton.edu/",
        notes="10,335 RGB-D images with 2D/3D oriented boxes and room layout.",
    ),
    "nyu_depth_v2": DatasetSpec(
        dataset_id="nyu_depth_v2",
        name="NYU Depth V2",
        license="BSD / Academic Research",
        commercial_ok=False,
        requires_terms_acceptance=False,
        is_synthetic=False,
        is_physically_surveyed=True,
        annotations_available=("semantic_mask", "instance_mask", "depth"),
        has_depth=True,
        has_camera_poses=False,
        depth_units="meters",
        url="https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html",
        notes="1,449 densely labelled indoor RGB-D frames.",
    ),
    "ade20k": DatasetSpec(
        dataset_id="ade20k",
        name="ADE20K",
        license="BSD-3-Clause (MIT CSAIL)",
        commercial_ok=True,
        requires_terms_acceptance=False,
        is_synthetic=False,
        is_physically_surveyed=False,
        annotations_available=("semantic_mask", "instance_mask", "part_mask"),
        has_depth=False,
        has_camera_poses=False,
        depth_units="none",
        url="https://groups.csail.mit.edu/vision/datasets/ADE20K/",
        notes="20K+ scene-centric images with object and part-level masks.",
    ),
    "structured3d": DatasetSpec(
        dataset_id="structured3d",
        name="Structured3D",
        license="Structured3D Terms of Use (Non-Commercial Research)",
        commercial_ok=False,
        requires_terms_acceptance=True,
        is_synthetic=True,
        is_physically_surveyed=False,
        annotations_available=("room_layout", "semantic_mask", "instance_mask", "depth", "wireframe"),
        has_depth=True,
        has_camera_poses=True,
        depth_units="millimeters",
        url="https://structured3d-dataset.org/",
        notes="Synthetic photorealistic indoor scenes; never report as physical accuracy.",
    ),
    "hypersim": DatasetSpec(
        dataset_id="hypersim",
        name="Hypersim (Apple)",
        license="CC-BY-4.0 / Apple Sample Code",
        commercial_ok=True,
        requires_terms_acceptance=False,
        is_synthetic=True,
        is_physically_surveyed=False,
        annotations_available=("semantic_mask", "instance_mask", "depth", "normals"),
        has_depth=True,
        has_camera_poses=True,
        depth_units="meters",
        url="https://github.com/apple/ml-hypersim",
        notes="Synthetic indoor scenes; commercial-friendly CC-BY-4.0.",
    ),
    "replica": DatasetSpec(
        dataset_id="replica",
        name="Replica (Meta)",
        license="Replica Research License (Non-Commercial)",
        commercial_ok=False,
        requires_terms_acceptance=True,
        is_synthetic=False,
        is_physically_surveyed=True,
        annotations_available=("semantic_mask", "instance_mask", "mesh"),
        has_depth=True,
        has_camera_poses=True,
        depth_units="meters",
        url="https://github.com/facebookresearch/Replica-Dataset",
        notes="High-fidelity reconstructed indoor spaces.",
    ),
}


@dataclass(frozen=True)
class NormalizedDatasetSample:
    sample_id: str
    dataset_id: str
    scene_id: str
    image_path: str
    image_width: int
    image_height: int
    depth_path: str | None = None
    camera_intrinsics: dict[str, float] | None = None
    camera_pose_4x4: tuple[tuple[float, ...], ...] | None = None
    boxes: tuple[dict[str, Any], ...] = ()
    is_synthetic: bool = False
    is_physically_surveyed: bool = False


@dataclass(frozen=True)
class DatasetIntegrityReport:
    dataset_id: str
    total_samples: int
    valid_samples: int
    corrupt_samples: tuple[dict[str, str], ...]
    scene_count: int
    leakage_detected: bool
    is_healthy: bool


def normalize_dataset_sample(
    raw: dict[str, Any],
    *,
    dataset_id: str,
) -> NormalizedDatasetSample:
    """Normalize a raw dataset record into ``NormalizedDatasetSample`` with canonical labels."""
    spec = INDOOR_DATASET_CATALOG.get(dataset_id)
    is_syn = spec.is_synthetic if spec else bool(raw.get("is_synthetic", False))
    is_phys = spec.is_physically_surveyed if spec else bool(raw.get("is_physically_surveyed", False))

    norm_boxes: list[dict[str, Any]] = []
    for b in raw.get("boxes", ()):
        raw_lbl = str(b.get("label", "unknown"))
        canon = resolve_taxonomy_class(raw_lbl).name
        xyxy = [float(v) for v in b.get("bbox_xyxy", (0, 0, 0, 0))]
        norm_boxes.append({
            "raw_label": raw_lbl,
            "canonical_label": canon,
            "bbox_xyxy": xyxy,
            "is_occluded": bool(b.get("is_occluded", False)),
            "is_truncated": bool(b.get("is_truncated", False)),
        })

    return NormalizedDatasetSample(
        sample_id=str(raw["sample_id"]),
        dataset_id=dataset_id,
        scene_id=str(raw.get("scene_id", raw["sample_id"])),
        image_path=str(raw["image_path"]),
        image_width=int(raw["image_width"]),
        image_height=int(raw["image_height"]),
        depth_path=raw.get("depth_path"),
        camera_intrinsics=raw.get("camera_intrinsics"),
        camera_pose_4x4=raw.get("camera_pose_4x4"),
        boxes=tuple(norm_boxes),
        is_synthetic=is_syn,
        is_physically_surveyed=is_phys,
    )


def create_scene_disjoint_splits(
    samples: Sequence[NormalizedDatasetSample],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: str = "reality_engine_a4",
) -> dict[str, list[NormalizedDatasetSample]]:
    """Deterministically partition samples by ``scene_id`` so no scene leaks across splits."""
    by_scene: dict[str, list[NormalizedDatasetSample]] = defaultdict(list)
    for s in samples:
        by_scene[s.scene_id].append(s)

    scenes = sorted(
        by_scene.keys(),
        key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode("utf-8")).hexdigest(),
    )
    n = len(scenes)
    n_train = max(1, int(round(n * train_ratio))) if n >= 3 else n
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0

    splits: dict[str, list[NormalizedDatasetSample]] = {"train": [], "val": [], "test": []}
    for idx, sid in enumerate(scenes):
        if idx < n_train:
            splits["train"].extend(by_scene[sid])
        elif idx < n_train + n_val:
            splits["val"].extend(by_scene[sid])
        else:
            splits["test"].extend(by_scene[sid])
    return splits


def validate_dataset_integrity(
    dataset_id: str,
    samples: Sequence[NormalizedDatasetSample],
    splits: dict[str, list[NormalizedDatasetSample]] | None = None,
    *,
    check_files_exist: bool = False,
) -> DatasetIntegrityReport:
    """Validate dimensions, bounding boxes, file paths (optional), and zero scene leakage."""
    corrupt: list[dict[str, str]] = []
    valid = 0
    scenes: set[str] = set()

    for s in samples:
        scenes.add(s.scene_id)
        if s.image_width <= 0 or s.image_height <= 0:
            corrupt.append({"sample_id": s.sample_id, "reason": "non_positive_image_dimensions"})
            continue
        if check_files_exist and not Path(s.image_path).is_file():
            corrupt.append({"sample_id": s.sample_id, "reason": f"missing_image_file:{s.image_path}"})
            continue
        bad_box = False
        for b in s.boxes:
            xy = b.get("bbox_xyxy", ())
            if len(xy) != 4 or xy[2] <= xy[0] or xy[3] <= xy[1]:
                corrupt.append({"sample_id": s.sample_id, "reason": "degenerate_bbox"})
                bad_box = True
                break
        if not bad_box:
            valid += 1

    leakage = False
    if splits is not None:
        tr_s = {x.scene_id for x in splits.get("train", ())}
        va_s = {x.scene_id for x in splits.get("val", ())}
        te_s = {x.scene_id for x in splits.get("test", ())}
        if (tr_s & va_s) or (tr_s & te_s) or (va_s & te_s):
            leakage = True

    return DatasetIntegrityReport(
        dataset_id=dataset_id,
        total_samples=len(samples),
        valid_samples=valid,
        corrupt_samples=tuple(corrupt),
        scene_count=len(scenes),
        leakage_detected=leakage,
        is_healthy=(len(corrupt) == 0 and not leakage),
    )


__all__ = [
    "DatasetIntegrityReport",
    "DatasetSpec",
    "INDOOR_DATASET_CATALOG",
    "NormalizedDatasetSample",
    "create_scene_disjoint_splits",
    "normalize_dataset_sample",
    "validate_dataset_integrity",
]
