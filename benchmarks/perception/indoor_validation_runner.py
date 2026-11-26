"""Real Indoor Dataset Validation & Six-Room Measured Capture Pathway (A5 Objective 2).

Supports reproducible validation across:
  - ScanNet / ScanNet++
  - SUN RGB-D
  - NYU Depth V2
  - Structured3D / Replica
  - Six-room measured capture pathway (never fabricates measurements or marks
    blocked/missing rooms as verified).

Strictly separates ``labelled_real``, ``unlabelled_real``, and ``synthetic`` metrics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.perception.a4_accuracy_suite import (
    A4ComprehensiveBenchmark,
    run_a4_comprehensive_benchmark,
)
from benchmarks.perception.evaluator import EvaluationMode
from benchmarks.perception.indoor_dataset_adapter import GroundTruthBox
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.datasets.indoor_registry import (
    INDOOR_DATASET_CATALOG,
    NormalizedDatasetSample,
    validate_dataset_integrity,
)


_ARCH_SET = frozenset({
    "wall", "wall_segment", "floor", "ceiling", "door", "door_frame",
    "window", "window_frame", "window_sill", "opening", "column", "beam", "stairs",
})


@dataclass(frozen=True)
class RoomCaptureValidationStatus:
    room_id: str
    status: str  # "verified" | "blocked_awaiting_human_capture" | "unlabelled_only"
    image_count: int
    has_ground_truth: bool
    has_measured_dimensions: bool
    blocker_reason: str | None = None


@dataclass(frozen=True)
class IndoorDatasetValidationReport:
    dataset_id: str
    evaluation_mode: EvaluationMode
    is_synthetic: bool
    is_physically_surveyed: bool
    samples_evaluated: int
    integrity_healthy: bool
    architectural_ap50: float | None
    furniture_ap50: float | None
    overall_ap50: float | None
    duplicate_rate: float
    mask_coverage_rate: float
    accuracy_claim_allowed: bool
    caveat: str
    benchmark_detail: dict[str, Any] = field(default_factory=dict)


def _mean_class_ap(per_class: Mapping[str, dict[str, Any]], class_set: frozenset[str], *, include: bool) -> float | None:
    vals = [
        float(v["ap"])
        for k, v in per_class.items()
        if (k in class_set) == include and "ap" in v
    ]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def run_indoor_dataset_validation(
    dataset_id: str,
    samples: Sequence[NormalizedDatasetSample],
    predictions_by_image: Mapping[str, Sequence[DetectionBox]],
    *,
    duplicate_count: int = 0,
    total_raw_detections: int = 0,
    masked_detections: int = 0,
    wall_time_s: float = 0.0,
    peak_vram_mb: float | None = None,
) -> IndoorDatasetValidationReport:
    """Validate perception predictions against a normalized indoor dataset with strict honesty."""
    spec = INDOOR_DATASET_CATALOG.get(dataset_id)
    is_syn = spec.is_synthetic if spec else False
    is_phys = spec.is_physically_surveyed if spec else False

    integrity = validate_dataset_integrity(dataset_id, samples, check_files_exist=False)
    gt_by_image: dict[str, list[GroundTruthBox]] = {}
    for s in samples:
        boxes = [
            GroundTruthBox(
                image_id=s.sample_id,
                raw_label=str(b["raw_label"]),
                canonical_label=str(b["canonical_label"]),
                box_2d=(
                    float(b["bbox_xyxy"][0]),
                    float(b["bbox_xyxy"][1]),
                    float(b["bbox_xyxy"][2]),
                    float(b["bbox_xyxy"][3]),
                ),
                is_occluded=bool(b.get("is_occluded", False)),
                is_truncated=bool(b.get("is_truncated", False)),
            )
            for b in s.boxes
        ]
        if boxes:
            gt_by_image[s.sample_id] = boxes

    if is_syn:
        mode = EvaluationMode.SYNTHETIC
    elif gt_by_image:
        mode = EvaluationMode.LABELLED_REAL
    else:
        mode = EvaluationMode.UNLABELLED_REAL

    bench: A4ComprehensiveBenchmark = run_a4_comprehensive_benchmark(
        mode=mode,
        dataset_name=dataset_id,
        predictions_by_image=predictions_by_image,
        gt_by_image=gt_by_image if gt_by_image else None,
        wall_time_s=wall_time_s,
        peak_vram_mb=peak_vram_mb,
    )

    from benchmarks.perception.evaluator import run_perception_benchmark

    base = run_perception_benchmark(
        mode=mode,
        dataset_name=dataset_id,
        predictions_by_image=predictions_by_image,
        gt_by_image=gt_by_image if gt_by_image else None,
    )
    arch_ap = _mean_class_ap(base.per_class, _ARCH_SET, include=True) if bench.accuracy_claim_allowed else None
    furn_ap = _mean_class_ap(base.per_class, _ARCH_SET, include=False) if bench.accuracy_claim_allowed else None
    dup_rate = round(duplicate_count / float(max(1, total_raw_detections)), 4)
    mask_cov = round(masked_detections / float(max(1, total_raw_detections)), 4)

    return IndoorDatasetValidationReport(
        dataset_id=dataset_id,
        evaluation_mode=mode,
        is_synthetic=is_syn,
        is_physically_surveyed=is_phys,
        samples_evaluated=len(samples),
        integrity_healthy=integrity.is_healthy,
        architectural_ap50=arch_ap,
        furniture_ap50=furn_ap,
        overall_ap50=bench.detection_map if bench.accuracy_claim_allowed else None,
        duplicate_rate=dup_rate,
        mask_coverage_rate=mask_cov,
        accuracy_claim_allowed=bench.accuracy_claim_allowed,
        caveat=bench.caveat,
        benchmark_detail=bench.to_dict(),
    )


def prepare_six_room_measured_capture_pathway(
    capture_root: Path | str,
    *,
    room_ids: Sequence[str] = ("room_01", "room_02", "room_03", "room_04", "room_05", "room_06"),
) -> dict[str, Any]:
    """Inspect the six-room measured capture directory without fabricating data for missing rooms."""
    root = Path(capture_root)
    room_statuses: list[RoomCaptureValidationStatus] = []

    for rid in room_ids:
        rdir = root / rid
        if not rdir.exists():
            room_statuses.append(RoomCaptureValidationStatus(
                room_id=rid,
                status="blocked_awaiting_human_capture",
                image_count=0,
                has_ground_truth=False,
                has_measured_dimensions=False,
                blocker_reason=f"Directory {rdir} does not exist.",
            ))
            continue

        imgs = [p for p in rdir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        gt_file = rdir / "ground_truth.json"
        meas_file = rdir / "measured_dimensions.json"
        has_gt = gt_file.exists()
        has_meas = meas_file.exists()

        if not imgs:
            room_statuses.append(RoomCaptureValidationStatus(
                room_id=rid,
                status="blocked_awaiting_human_capture",
                image_count=0,
                has_ground_truth=has_gt,
                has_measured_dimensions=has_meas,
                blocker_reason="No RGB photographs present in room directory.",
            ))
        elif not (has_gt and has_meas):
            room_statuses.append(RoomCaptureValidationStatus(
                room_id=rid,
                status="unlabelled_only",
                image_count=len(imgs),
                has_ground_truth=has_gt,
                has_measured_dimensions=has_meas,
                blocker_reason="Missing human-reviewed ground_truth.json or measured_dimensions.json.",
            ))
        else:
            gt_payload = json.loads(gt_file.read_text(encoding="utf-8"))
            is_human_rev = bool(gt_payload.get("is_human_reviewed", False))
            room_statuses.append(RoomCaptureValidationStatus(
                room_id=rid,
                status="verified" if is_human_rev else "unlabelled_only",
                image_count=len(imgs),
                has_ground_truth=True,
                has_measured_dimensions=True,
                blocker_reason=None if is_human_rev else "ground_truth.json not marked is_human_reviewed=True.",
            ))

    verified_cnt = sum(1 for r in room_statuses if r.status == "verified")
    blocked_cnt = sum(1 for r in room_statuses if r.status == "blocked_awaiting_human_capture")
    return {
        "capture_root": str(root),
        "total_rooms": len(room_statuses),
        "verified_rooms": verified_cnt,
        "blocked_rooms": blocked_cnt,
        "ready_for_full_six_room_benchmark": verified_cnt == len(room_statuses),
        "rooms": [asdict(r) for r in room_statuses],
    }


__all__ = [
    "IndoorDatasetValidationReport",
    "RoomCaptureValidationStatus",
    "prepare_six_room_measured_capture_pathway",
    "run_indoor_dataset_validation",
]
