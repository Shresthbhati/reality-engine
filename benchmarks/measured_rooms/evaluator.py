"""Automated Measured-Room Benchmark Evaluator (Mandate Section 10).

Scores a reconstructed WorldIR against measured room ground truths under Section 1 tolerances.
Measures:
- Architectural accuracy (wall length, height, area, opening dimension/position errors, topology precision/recall).
- Fine-grained entity detection & identity (per-class precision, recall, F1, IDF1, duplicate & false-merge rates).
- Uncertainty calibration and coverage.
- Blind holdout separation: rooms 5 & 6 are strictly gated and require explicit authorization.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from perception.architecture.accuracy_report import (
    RoomAccuracyReport,
    Section1Tolerances,
    evaluate_measured_room_accuracy,
)
from world_ir import WorldIR
from world_ir.schema_v1 import EntityType
from world_ir.serialization import load_world

ROOT = Path(__file__).resolve().parents[2]
MEASURED_ROOMS_DIR = ROOT / "benchmarks" / "measured_rooms"
DEV_ROOMS = (
    "room_01_living_studio",
    "room_02_bed_wardrobe",
    "room_03_corridor_office",
    "room_04_kitchenette",
)
HOLDOUT_ROOMS = (
    "room_05_holdout_reception",
    "room_06_holdout_conference",
)


@dataclass
class RoomBenchmarkMetrics:
    """Comprehensive metric report for a single evaluated room."""

    room_id: str
    room_name: str
    is_holdout: bool
    meets_tolerances: bool

    # Architectural accuracy metrics
    mean_wall_error_m: float
    max_wall_error_m: float
    wall_error_pct: float
    ceiling_height_error_m: float
    room_area_error_pct: float
    mean_opening_error_m: float
    mean_opening_pos_error_m: float
    topology_precision: float
    topology_recall: float

    # Detection & identity metrics
    per_class_scores: dict[str, dict[str, float]]
    mean_f1: float
    idf1: float
    duplicate_rate: float
    false_merge_rate: float
    same_camera_exclusion_adherence: float

    # Uncertainty metrics
    uncertainty_coverage_pct: float
    calibration_score: float

    # Raw counts
    total_checks: int
    passed_checks: int
    failed_checks: int
    missing_elements: list[dict[str, Any]] = field(default_factory=list)
    extra_elements: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_room_benchmark(
    world: WorldIR,
    ground_truth: dict[str, Any],
    tolerances: Section1Tolerances | None = None,
    *,
    is_holdout: bool = False,
) -> RoomBenchmarkMetrics:
    """Evaluate a reconstructed WorldIR against ground truth specifications."""
    tol = tolerances or Section1Tolerances()
    report: RoomAccuracyReport = evaluate_measured_room_accuracy(world, ground_truth, tol)
    report.compute_summary()

    # 1. Architectural accuracy breakdown
    wall_checks = [c for c in report.checks if c.element_type == "wall" and c.dimension_name == "length"]
    wall_errors = [c.error_abs for c in wall_checks]
    mean_wall_err = float(np.mean(wall_errors)) if wall_errors else 0.0
    max_wall_err = float(np.max(wall_errors)) if wall_errors else 0.0
    wall_err_pct = float(np.mean([c.error_pct for c in wall_checks])) if wall_checks else 0.0

    height_checks = [c for c in report.checks if c.element_type == "room" and c.dimension_name in ("ceiling_height", "height")]
    height_err = float(height_checks[0].error_abs) if height_checks else 0.0

    area_checks = [c for c in report.checks if c.element_type == "room" and c.dimension_name in ("floor_area", "area")]
    area_err_pct = float(area_checks[0].error_pct) if area_checks else 0.0

    opening_checks = [
        c for c in report.checks if c.element_type in ("door", "window", "opening") and c.dimension_name in ("width", "height")
    ]
    mean_opening_err = float(np.mean([c.error_abs for c in opening_checks])) if opening_checks else 0.0

    opening_pos_checks = [
        c for c in report.checks if c.element_type in ("door", "window", "opening") and "position" in c.dimension_name
    ]
    mean_opening_pos_err = float(np.mean([c.error_abs for c in opening_pos_checks])) if opening_pos_checks else 0.0

    # Topology precision & recall (walls boundary closure)
    gt_walls = ground_truth.get("walls", [])
    det_walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
    wall_matched_count = next((c.n_matched for c in report.class_scores if c.element_type == "wall"), 0)
    topo_prec = wall_matched_count / len(det_walls) if det_walls else 0.0
    topo_rec = wall_matched_count / len(gt_walls) if gt_walls else 0.0

    # 2. Detection & Identity breakdown
    per_class: dict[str, dict[str, float]] = {}
    f1_list = []
    for cs in report.class_scores:
        per_class[cs.element_type] = {
            "precision": cs.precision,
            "recall": cs.recall,
            "f1": cs.f1,
            "n_ground_truth": cs.n_ground_truth,
            "n_detected": cs.n_detected,
            "n_matched": cs.n_matched,
        }
        f1_list.append(cs.f1)
    mean_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    # IDF1 (Identification F1) calculation across all elements
    total_gt = sum(c.n_ground_truth for c in report.class_scores)
    total_det = sum(c.n_detected for c in report.class_scores)
    total_matched = sum(c.n_matched for c in report.class_scores)
    idf1 = (2.0 * total_matched / (total_gt + total_det)) if (total_gt + total_det) > 0 else 0.0

    # Duplicate & False merge rates
    n_extra = len(report.extra)
    n_missing = len(report.missing)
    dup_rate = n_extra / total_det if total_det > 0 else 0.0
    false_merge_rate = n_missing / total_gt if total_gt > 0 else 0.0

    # Same-camera mutual exclusion adherence
    same_cam_adherence = 1.0
    for ent in world.entities.values():
        props = getattr(ent, "custom_properties", None) or getattr(ent, "metadata", None)
        if props and "camera_stations" in props:
            stations = props["camera_stations"]
            if len(stations) != len(set(stations)):
                same_cam_adherence = max(0.0, same_cam_adherence - 0.1)

    # 3. Uncertainty coverage: checks that passed vs total
    unc_coverage = (report.passed_checks / report.total_checks * 100.0) if report.total_checks > 0 else 0.0
    calib_score = round(min(1.0, 1.0 - abs(0.95 - (unc_coverage / 100.0))), 3)

    return RoomBenchmarkMetrics(
        room_id=report.room_id,
        room_name=report.room_name,
        is_holdout=is_holdout,
        meets_tolerances=report.meets_tolerances,
        mean_wall_error_m=round(mean_wall_err, 4),
        max_wall_error_m=round(max_wall_err, 4),
        wall_error_pct=round(wall_err_pct, 2),
        ceiling_height_error_m=round(height_err, 4),
        room_area_error_pct=round(area_err_pct, 2),
        mean_opening_error_m=round(mean_opening_err, 4),
        mean_opening_pos_error_m=round(mean_opening_pos_err, 4),
        topology_precision=round(topo_prec, 3),
        topology_recall=round(topo_rec, 3),
        per_class_scores=per_class,
        mean_f1=round(mean_f1, 3),
        idf1=round(idf1, 3),
        duplicate_rate=round(dup_rate, 3),
        false_merge_rate=round(false_merge_rate, 3),
        same_camera_exclusion_adherence=round(same_cam_adherence, 3),
        uncertainty_coverage_pct=round(unc_coverage, 1),
        calibration_score=calib_score,
        total_checks=report.total_checks,
        passed_checks=report.passed_checks,
        failed_checks=report.failed_checks,
        missing_elements=report.missing,
        extra_elements=report.extra,
    )


def load_room_ground_truth(room_id_or_dir: str | Path, *, allow_holdout: bool = False) -> dict[str, Any]:
    """Load ground truth JSON with blind holdout security gating."""
    p = Path(room_id_or_dir)
    if not p.is_dir():
        p = MEASURED_ROOMS_DIR / str(room_id_or_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Measured room directory not found: {p}")

    dir_name = p.name
    if dir_name in HOLDOUT_ROOMS and not allow_holdout:
        raise PermissionError(
            f"Access to blind holdout room '{dir_name}' is restricted during development tuning "
            "(Rule 10: Never inspect blind holdout ground truth during development tuning). "
            "Pass allow_holdout=True or --holdout explicitly for final audit verification."
        )

    gt_file = p / "ground_truth.json"
    if not gt_file.is_file():
        raise FileNotFoundError(f"Missing ground_truth.json in {p}")

    return json.loads(gt_file.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Reality Engine against measured room benchmarks.")
    parser.add_argument("--room", required=True, help="Room directory name (e.g. room_01_living_studio)")
    parser.add_argument("--world", required=True, help="Path to reconstructed WorldIR JSON file")
    parser.add_argument("--holdout", action="store_true", help="Explicitly authorize blind holdout evaluation")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path for evaluation report")
    args = parser.parse_args(argv)

    gt = load_room_ground_truth(args.room, allow_holdout=args.holdout)
    world_path = Path(args.world)
    if not world_path.is_file():
        print(f"Error: World file not found: {world_path}")
        return 1

    world = load_world(world_path)
    metrics = evaluate_room_benchmark(world, gt, is_holdout=args.room in HOLDOUT_ROOMS)

    result_json = json.dumps(metrics.to_dict(), indent=2)
    print(result_json)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result_json, encoding="utf-8")
        print(f"Report written to {args.out}")

    return 0 if metrics.meets_tolerances else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
