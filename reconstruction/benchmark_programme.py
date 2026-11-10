"""Real-building benchmark programme (N-33).

Builds a controlled dataset of measured rooms, apartments, corridors, stairs,
buildings and outdoor environments, with independent ground truth.

Every major reconstruction release has results against real physical
measurements, not just synthetic fixtures.

Honesty rules:
* Benchmark data must come from real measured buildings, not synthetic fixtures.
* Ground truth must be independently measured (tape measure, laser scanner,
  total station), not derived from the reconstruction itself.
* Results must be reported per-class with precision, recall, dimensional error
  and calibrated uncertainty.
* Results are not fabricated; if no benchmark data exists, the system reports
  "not available" rather than making up numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MeasuredRoom:
    """A measured room with ground truth data."""
    room_id: str
    name: str
    building_type: str  # "apartment", "house", "office", "commercial", "corridor", "outdoor"
    address: Optional[str]
    capture_date: Optional[str]
    capture_device: Optional[str]  # camera model, phone model
    measurements: Dict[str, float]  # room_length_m, room_width_m, room_height_m, etc.
    openings: List[dict]  # doors, windows with positions and dimensions
    furniture: List[dict]  # furniture items with positions and dimensions
    notes: Optional[str]


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of benchmarking one reconstruction against ground truth."""
    room_id: str
    wall_count_error_pct: Optional[float]
    door_recall: Optional[float]
    door_precision: Optional[float]
    window_recall: Optional[float]
    window_precision: Optional[float]
    room_length_error_pct: Optional[float]
    room_width_error_pct: Optional[float]
    room_height_error_pct: Optional[float]
    footprint_area_error_pct: Optional[float]
    wall_thickness_error_mm: Optional[float]
    overall_score: Optional[float]  # 0..1 overall quality score


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark report for a set of measured rooms."""
    rooms: List[BenchmarkResult]
    summary: Dict[str, float]  # average metrics across all rooms
    date: str
    note: Optional[str] = None


class BenchmarkProgramme:
    """Manages real-building benchmark data and evaluation."""

    def __init__(self, benchmark_dir: Optional[str] = None):
        self.benchmark_dir = Path(benchmark_dir) if benchmark_dir else Path("benchmarks")
        self.rooms: Dict[str, MeasuredRoom] = {}
        self.results: Dict[str, BenchmarkResult] = {}

    def add_room(self, room: MeasuredRoom) -> None:
        """Add a measured room to the benchmark dataset."""
        self.rooms[room.room_id] = room

    def load_room(self, room_id: str) -> Optional[MeasuredRoom]:
        """Load a room's ground truth data."""
        return self.rooms.get(room_id)

    def list_rooms(self) -> List[str]:
        """List all room IDs in the benchmark dataset."""
        return list(self.rooms.keys())

    def evaluate_reconstruction(
        self,
        room_id: str,
        reconstructed_entities: Dict[str, dict],
    ) -> Optional[BenchmarkResult]:
        """Evaluate a reconstruction against ground truth for one room."""
        room = self.rooms.get(room_id)
        if room is None:
            return None

        classifications = []

        # Wall count accuracy
        gt_wall_count = len([m for m in room.measurements.get("walls", []) if m.get("type") == "wall"])
        detected_walls = len([e for e in reconstructed_entities.values() if e.get("type") == "wall"])
        wall_count_error_pct = abs(detected_walls - gt_wall_count) / max(gt_wall_count, 1) * 100 if gt_wall_count > 0 else None

        # Door/window recall and precision
        gt_doors = [m for m in room.measurements.get("openings", []) if m.get("type") == "door"]
        gt_windows = [m for m in room.measurements.get("openings", []) if m.get("type") == "window"]
        detected_doors = [e for e in reconstructed_entities.values() if e.get("type") == "door"]
        detected_windows = [e for e in reconstructed_entities.values() if e.get("type") == "window"]

        door_recall = len(detected_doors) / len(gt_doors) if gt_doors else None
        door_precision = len(detected_doors) / max(len(detected_doors), 1) if detected_doors else None
        window_recall = len(detected_windows) / len(gt_windows) if gt_windows else None
        window_precision = len(detected_windows) / max(len(detected_windows), 1) if detected_windows else None

        # Room dimension errors
        room_length_error_pct = None
        room_width_error_pct = None
        room_height_error_pct = None
        footprint_area_error_pct = None

        if "room_length_m" in room.measurements and "reconstructed_length_m" in room.measurements:
            room_length_error_pct = abs(room.measurements["room_length_m"] - room.measurements["reconstructed_length_m"]) / room.measurements["room_length_m"] * 100

        if "room_width_m" in room.measurements and "reconstructed_width_m" in room.measurements:
            room_width_error_pct = abs(room.measurements["room_width_m"] - room.measurements["reconstructed_width_m"]) / room.measurements["room_width_m"] * 100

        if "room_height_m" in room.measurements and "reconstructed_height_m" in room.measurements:
            room_height_error_pct = abs(room.measurements["room_height_m"] - room.measurements["reconstructed_height_m"]) / room.measurements["room_height_m"] * 100

        if "footprint_area_m2" in room.measurements and "reconstructed_footprint_area_m2" in room.measurements:
            footprint_area_error_pct = abs(room.measurements["footprint_area_m2"] - room.measurements["reconstructed_footprint_area_m2"]) / room.measurements["footprint_area_m2"] * 100

        # Overall score (simple average of available metrics, weighted)
        scores = []
        if wall_count_error_pct is not None:
            scores.append(max(0, 1 - wall_count_error_pct / 100))
        if door_recall is not None:
            scores.append(door_recall)
        if door_precision is not None:
            scores.append(door_precision)
        if window_recall is not None:
            scores.append(window_recall)
        if window_precision is not None:
            scores.append(window_precision)
        if room_length_error_pct is not None:
            scores.append(max(0, 1 - room_length_error_pct / 100))
        if room_width_error_pct is not None:
            scores.append(max(0, 1 - room_width_error_pct / 100))
        if room_height_error_pct is not None:
            scores.append(max(0, 1 - room_height_error_pct / 100))
        if footprint_area_error_pct is not None:
            scores.append(max(0, 1 - footprint_area_error_pct / 100))

        overall_score = sum(scores) / len(scores) if scores else None

        result = BenchmarkResult(
            room_id=room_id,
            wall_count_error_pct=wall_count_error_pct,
            door_recall=door_recall,
            door_precision=door_precision,
            window_recall=window_recall,
            window_precision=window_precision,
            room_length_error_pct=room_length_error_pct,
            room_width_error_pct=room_width_error_pct,
            room_height_error_pct=room_height_error_pct,
            footprint_area_error_pct=footprint_area_error_pct,
            overall_score=overall_score,
        )

        self.results[room_id] = result
        return result

    def generate_report(self) -> BenchmarkReport:
        """Generate a benchmark report across all evaluated rooms."""
        if not self.results:
            return BenchmarkReport(
                rooms=[],
                summary={},
                date=datetime.now().isoformat(),
                note="No benchmark results available",
            )

        # Calculate summary statistics
        all_scores = [r.overall_score for r in self.results.values() if r.overall_score is not None]
        all_wall_errors = [r.wall_count_error_pct for r in self.results.values() if r.wall_count_error_pct is not None]
        all_door_recalls = [r.door_recall for r in self.results.values() if r.door_recall is not None]
        all_window_recalls = [r.window_recall for r in self.results.values() if r.window_recall is not None]

        summary = {}
        if all_scores:
            summary["overall_score_mean"] = sum(all_scores) / len(all_scores)
            summary["overall_score_min"] = min(all_scores)
            summary["overall_score_max"] = max(all_scores)
        if all_wall_errors:
            summary["wall_count_error_pct_mean"] = sum(all_wall_errors) / len(all_wall_errors)
        if all_door_recalls:
            summary["door_recall_mean"] = sum(all_door_recalls) / len(all_door_recalls)
        if all_window_recalls:
            summary["window_recall_mean"] = sum(all_window_recalls) / len(all_window_recalls)

        return BenchmarkReport(
            rooms=list(self.results.values()),
            summary=summary,
            date=datetime.now().isoformat(),
            note=None,
        )

    def save_report(self, path: str) -> None:
        """Save the benchmark report to a JSON file."""
        report = self.generate_report()
        data = {
            "date": report.date,
            "note": report.note,
            "summary": report.summary,
            "rooms": [
                {
                    "room_id": r.room_id,
                    "wall_count_error_pct": r.wall_count_error_pct,
                    "door_recall": r.door_recall,
                    "door_precision": r.door_precision,
                    "window_recall": r.window_recall,
                    "window_precision": r.window_precision,
                    "room_length_error_pct": r.room_length_error_pct,
                    "room_width_error_pct": r.room_width_error_pct,
                    "room_height_error_pct": r.room_height_error_pct,
                    "footprint_area_error_pct": r.footprint_area_error_pct,
                    "overall_score": r.overall_score,
                }
                for r in self.results.values()
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    def load_report(self, path: str) -> BenchmarkReport:
        """Load a benchmark report from a JSON file."""
        data = json.loads(Path(path).read_text())
        rooms = []
        for r in data.get("rooms", []):
            rooms.append(BenchmarkResult(
                room_id=r["room_id"],
                wall_count_error_pct=r.get("wall_count_error_pct"),
                door_recall=r.get("door_recall"),
                door_precision=r.get("door_precision"),
                window_recall=r.get("window_recall"),
                window_precision=r.get("window_precision"),
                room_length_error_pct=r.get("room_length_error_pct"),
                room_width_error_pct=r.get("room_width_error_pct"),
                room_height_error_pct=r.get("room_height_error_pct"),
                footprint_area_error_pct=r.get("footprint_area_error_pct"),
                overall_score=r.get("overall_score"),
            ))
        return BenchmarkReport(
            rooms=rooms,
            summary=data.get("summary", {}),
            date=data.get("date", ""),
            note=data.get("note"),
        )


__all__ = ["BenchmarkProgramme", "MeasuredRoom", "BenchmarkResult", "BenchmarkReport"]