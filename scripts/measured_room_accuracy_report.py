#!/usr/bin/env python3
"""Measured-room accuracy report CLI (Task 12.2).

Evaluates reconstructed WorldIR rooms against ground truth specifications under the
Master Turnaround Plan Section 1 tolerances:
- Wall length: +-2 cm (0.02 m)
- Room area: +-1.0 %
- Ceiling height: +-2 cm (0.02 m)
- Opening width: +-2 cm (0.02 m)
- Opening height: +-2 cm (0.02 m)
- Fixture position: +-5 cm (0.05 m)

Usage:
    python scripts/measured_room_accuracy_report.py --fixture synthetic
    python scripts/measured_room_accuracy_report.py --fixture synthetic --json out.json
    python scripts/measured_room_accuracy_report.py --fixture synthetic --fail-on-tolerance
    python scripts/measured_room_accuracy_report.py --world world.json --ground-truth gt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.architecture.accuracy_report import (  # noqa: E402
    Section1Tolerances,
    evaluate_measured_room_accuracy,
)
from world_ir.serialization import load_world  # noqa: E402


def evaluate_synthetic_fixture(tolerances: Section1Tolerances) -> tuple[dict[str, Any], str]:
    """Compile the synthetic fidelity fixture and evaluate against Section 1 tolerances."""
    from scripts.room_fidelity_report import build_and_compile

    gt_path = ROOT / "synthetic" / "fidelity_room" / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Synthetic ground truth not found at {gt_path}")
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

    _truth, _obs, world, _diag = build_and_compile(colors="albedo", voxel_m=0.08)
    report = evaluate_measured_room_accuracy(world, ground_truth, tolerances=tolerances)
    return report.to_dict(), report.format_table("Synthetic Fidelity Room Accuracy Report")


def evaluate_custom_world(
    world_path: Path, gt_path: Path, tolerances: Section1Tolerances
) -> tuple[dict[str, Any], str]:
    """Evaluate a user-supplied WorldIR file against ground truth JSON."""
    world = load_world(world_path)
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    report = evaluate_measured_room_accuracy(world, ground_truth, tolerances=tolerances)
    return report.to_dict(), report.format_table(f"Custom Room Accuracy Report ({world_path.name})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measured-Room Accuracy Report against Section 1 Tolerances (Task 12.2)"
    )
    parser.add_argument(
        "--fixture",
        choices=("synthetic", "real"),
        default="synthetic",
        help="Target fixture to evaluate (default: synthetic)",
    )
    parser.add_argument(
        "--world",
        type=Path,
        help="Path to reconstructed WorldIR JSON",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Path to ground truth JSON specification",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Output path for machine-readable JSON report",
    )
    parser.add_argument(
        "--fail-on-tolerance",
        action="store_true",
        help="Exit with code 1 if any Section 1 tolerance check fails",
    )
    parser.add_argument(
        "--wall-tol-m",
        type=float,
        default=0.02,
        help="Wall length tolerance in meters (default: 0.02 m / 2 cm)",
    )
    parser.add_argument(
        "--area-tol-pct",
        type=float,
        default=1.0,
        help="Room area tolerance in percent (default: 1.0 %%)",
    )
    parser.add_argument(
        "--height-tol-m",
        type=float,
        default=0.02,
        help="Ceiling height tolerance in meters (default: 0.02 m / 2 cm)",
    )
    parser.add_argument(
        "--opening-tol-m",
        type=float,
        default=0.02,
        help="Opening width/height tolerance in meters (default: 0.02 m / 2 cm)",
    )

    args = parser.parse_args(argv)

    tolerances = Section1Tolerances(
        wall_length_m=args.wall_tol_m,
        room_area_pct=args.area_tol_pct,
        ceiling_height_m=args.height_tol_m,
        opening_width_m=args.opening_tol_m,
        opening_height_m=args.opening_tol_m,
    )

    if args.world and args.ground_truth:
        report_dict, formatted_table = evaluate_custom_world(args.world, args.ground_truth, tolerances)
    elif args.fixture == "synthetic":
        report_dict, formatted_table = evaluate_synthetic_fixture(tolerances)
    else:
        print("Real room fixture requires --world and --ground-truth paths.", file=sys.stderr)
        return 2

    print(formatted_table)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report_dict, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote machine-readable JSON report to: {args.json}")

    if args.fail_on_tolerance and not report_dict.get("meets_tolerances"):
        print("ERROR: One or more Section 1 tolerances exceeded!", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
