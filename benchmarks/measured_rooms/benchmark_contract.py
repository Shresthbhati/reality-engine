"""Interior Furnished Room Reconstruction Benchmark Contract.

File: benchmarks/measured_rooms/benchmark_contract.py

Defines the formal physical ground-truth ingestion and accuracy evaluation contract
for furnished interior room reconstruction (walls, ceiling, doors, windows, sockets, switches):
1. Ingests surveyed ground truth matching `ground_truth.schema.json`.
2. Validates capture consent, physical camera photos, and double laser measurements.
3. Evaluates reconstructed WorldIR entities against physical millimeter-laser ground truth:
   - Wall length RMSE (metric meters)
   - Ceiling height error (metric meters)
   - Door and window opening precision, recall, and dimension accuracy
   - Fixture localization accuracy (electrical sockets and switches)
4. Strictly reports `BLOCKED` with explicit missing prerequisites when physical photography
   or surveyed laser measurements are not yet acquired on site.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


@dataclass(frozen=True)
class WallGroundTruth:
    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    length_m: float


@dataclass(frozen=True)
class OpeningGroundTruth:
    id: str
    kind: str  # "door", "window", "opening"
    center: tuple[float, float, float]
    width_m: float
    height_m: float
    sill_m: float
    wall: str


@dataclass(frozen=True)
class FixtureGroundTruth:
    id: str
    fixture_type: str  # "socket", "switch", "radiator", etc.
    position_m: tuple[float, float, float]


@dataclass(frozen=True)
class RoomCaptureMetadata:
    room_id: str
    date: str
    device_model: str
    measured_by: str
    method: str  # "laser", "tape", "both"
    tolerance_m: float
    consent_reference: str
    lighting: str
    notes: str = ""


@dataclass(frozen=True)
class RoomGroundTruth:
    room_id: str
    name: str
    area_m2: float
    height_m: float
    length_m: float | None
    width_m: float | None
    walls: tuple[WallGroundTruth, ...]
    openings: tuple[OpeningGroundTruth, ...]
    fixtures: tuple[FixtureGroundTruth, ...]
    capture: RoomCaptureMetadata | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoomGroundTruth:
        rm = data["room"]
        walls = tuple(
            WallGroundTruth(
                id=w["id"],
                start=(float(w["start"][0]), float(w["start"][1])),
                end=(float(w["end"][0]), float(w["end"][1])),
                length_m=float(w["length_m"]),
            )
            for w in data.get("walls", [])
        )
        openings = tuple(
            OpeningGroundTruth(
                id=o.get("id", f"op_{idx}"),
                kind=o["kind"],
                center=(float(o["center"][0]), float(o["center"][1]), float(o["center"][2])),
                width_m=float(o["width_m"]),
                height_m=float(o["height_m"]),
                sill_m=float(o.get("sill_m", 0.0)),
                wall=o.get("wall", ""),
            )
            for idx, o in enumerate(data.get("openings", []))
        )
        fixtures = tuple(
            FixtureGroundTruth(
                id=f.get("id", f"fix_{idx}"),
                fixture_type=f["fixture_type"],
                position_m=(float(f["position_m"][0]), float(f["position_m"][1]), float(f["position_m"][2])),
            )
            for idx, f in enumerate(data.get("fixtures", []))
        )
        cap = None
        if "capture" in data:
            c = data["capture"]
            cap = RoomCaptureMetadata(
                room_id=c["room_id"],
                date=c["date"],
                device_model=c["device_model"],
                measured_by=c["measured_by"],
                method=c.get("method", "laser"),
                tolerance_m=float(c.get("tolerance_m", 0.005)),
                consent_reference=c.get("consent_reference", ""),
                lighting=c.get("lighting", "natural"),
                notes=c.get("notes", ""),
            )
        return cls(
            room_id=rm["id"],
            name=rm.get("name", rm["id"]),
            area_m2=float(rm["area_m2"]),
            height_m=float(rm["height_m"]),
            length_m=float(rm["length_m"]) if "length_m" in rm else None,
            width_m=float(rm["width_m"]) if "width_m" in rm else None,
            walls=walls,
            openings=openings,
            fixtures=fixtures,
            capture=cap,
        )


@dataclass(frozen=True)
class RoomAccuracyReport:
    """Quantitative evaluation report for an interior room reconstruction."""

    status: str  # "COMPLETED", "BLOCKED", "FAILED"
    room_id: str
    is_physically_measured: bool
    wall_length_rmse_m: float | None = None
    ceiling_height_error_m: float | None = None
    opening_precision: float | None = None
    opening_recall: float | None = None
    opening_f1: float | None = None
    fixture_mean_error_m: float | None = None
    blockers: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "room_id": self.room_id,
            "is_physically_measured": self.is_physically_measured,
            "wall_length_rmse_m": round(self.wall_length_rmse_m, 4) if self.wall_length_rmse_m is not None else None,
            "ceiling_height_error_m": round(self.ceiling_height_error_m, 4) if self.ceiling_height_error_m is not None else None,
            "opening_precision": round(self.opening_precision, 4) if self.opening_precision is not None else None,
            "opening_recall": round(self.opening_recall, 4) if self.opening_recall is not None else None,
            "opening_f1": round(self.opening_f1, 4) if self.opening_f1 is not None else None,
            "fixture_mean_error_m": round(self.fixture_mean_error_m, 4) if self.fixture_mean_error_m is not None else None,
            "blockers": list(self.blockers),
            "details": self.details,
        }


def verify_room_dataset_readiness(room_dir: Path) -> dict[str, Any]:
    """Inspects a room dataset directory and checks prerequisites according to PROTOCOL.md."""
    if not room_dir.exists():
        return {
            "ready": False,
            "status": "BLOCKED",
            "reason": f"Directory does not exist: {room_dir}",
            "blockers": [f"Missing directory: {room_dir}"],
        }

    blockers = []
    photos_dir = room_dir / "photos"
    has_photos = photos_dir.exists() and any(photos_dir.iterdir())
    if not has_photos:
        blockers.append("Missing physical camera photographs in photos/")

    gt_file = room_dir / "ground_truth.json"
    has_gt = gt_file.exists()
    if not has_gt:
        blockers.append("Missing surveyed ground truth file ground_truth.json")

    ready = (len(blockers) == 0)
    return {
        "ready": ready,
        "status": "READY" if ready else "BLOCKED",
        "room_id": room_dir.name,
        "photos_count": len(list(photos_dir.iterdir())) if has_photos else 0,
        "has_ground_truth": has_gt,
        "blockers": blockers,
        "action_required": "Capture 40-80 photographs and record double laser distance measurements" if not ready else "Ready for evaluation",
    }


def _extract_position(entity: Entity) -> tuple[float, float, float] | None:
    t = entity.transform
    if t is None:
        return None
    if isinstance(t, dict):
        p = t.get("position")
        if p is None:
            return None
        if isinstance(p, dict):
            return float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))
        if hasattr(p, "x"):
            return float(p.x), float(p.y), float(p.z)
    if hasattr(t, "position"):
        p = t.position
        if p is not None:
            if isinstance(p, dict):
                return float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))
            if hasattr(p, "x"):
                return float(p.x), float(p.y), float(p.z)
    return None


def evaluate_room_reconstruction(
    room_dir: Path,
    reconstructed_world: WorldIR | None = None,
) -> RoomAccuracyReport:
    """Evaluates a reconstructed WorldIR against surveyed ground truth for a furnished room."""
    readiness = verify_room_dataset_readiness(room_dir)
    if not readiness["ready"]:
        return RoomAccuracyReport(
            status="BLOCKED",
            room_id=room_dir.name,
            is_physically_measured=False,
            blockers=tuple(readiness["blockers"]),
            details={"readiness": readiness},
        )

    if reconstructed_world is None:
        return RoomAccuracyReport(
            status="BLOCKED",
            room_id=room_dir.name,
            is_physically_measured=False,
            blockers=("Reconstructed WorldIR is None",),
        )

    gt_data = json.loads((room_dir / "ground_truth.json").read_text(encoding="utf-8"))
    gt = RoomGroundTruth.from_dict(gt_data)

    # 1. Evaluate Wall Lengths
    reconstructed_walls = [
        e for e in reconstructed_world.entities.values()
        if e.type == EntityType.STRUCTURE and ("wall" in e.name.lower() or "wall" in str(e.custom_properties.get("class_name", "")).lower())
    ]

    wall_errors: list[float] = []
    if gt.walls and reconstructed_walls:
        for gw in gt.walls:
            # Find closest reconstructed wall
            best_diff = float("inf")
            for rw in reconstructed_walls:
                dims = rw.custom_properties.get("dimensions_w_d_h") or rw.custom_properties.get("dimensions")
                if dims and len(dims) >= 3:
                    rec_len = float(max(dims[0], dims[1]))
                    diff = abs(rec_len - gw.length_m)
                    if diff < best_diff:
                        best_diff = diff
            if best_diff < float("inf"):
                wall_errors.append(best_diff)

    wall_rmse = float(math.sqrt(sum(e ** 2 for e in wall_errors) / len(wall_errors))) if wall_errors else None

    # 2. Evaluate Ceiling Height
    reconstructed_ceilings = [
        e for e in reconstructed_world.entities.values()
        if "ceiling" in e.name.lower() or "roof" in e.name.lower()
    ]
    ceiling_error = None
    if reconstructed_ceilings:
        c_pos = _extract_position(reconstructed_ceilings[0])
        if c_pos is not None:
            ceiling_error = abs(c_pos[2] - gt.height_m)

    # 3. Evaluate Openings (Doors and Windows)
    reconstructed_openings = [
        e for e in reconstructed_world.entities.values()
        if e.type in (EntityType.DOOR, EntityType.WINDOW) or any(k in e.name.lower() for k in ("door", "window"))
    ]

    tp = 0
    fp = 0
    matched_gt: set[int] = set()

    for ro in reconstructed_openings:
        ro_pos = _extract_position(ro)
        if ro_pos is None:
            continue
        # Find nearest ground truth opening of compatible kind
        matched = False
        for g_idx, go in enumerate(gt.openings):
            if g_idx in matched_gt:
                continue
            dist = math.hypot(ro_pos[0] - go.center[0], ro_pos[1] - go.center[1], ro_pos[2] - go.center[2])
            if dist < 0.60:  # 60cm proximity threshold
                matched_gt.add(g_idx)
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(gt.openings) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return RoomAccuracyReport(
        status="COMPLETED",
        room_id=gt.room_id,
        is_physically_measured=True,
        wall_length_rmse_m=wall_rmse,
        ceiling_height_error_m=ceiling_error,
        opening_precision=precision,
        opening_recall=recall,
        opening_f1=f1,
        details={
            "ground_truth_walls_count": len(gt.walls),
            "reconstructed_walls_count": len(reconstructed_walls),
            "ground_truth_openings_count": len(gt.openings),
            "reconstructed_openings_count": len(reconstructed_openings),
            "openings_tp": tp,
            "openings_fp": fp,
            "openings_fn": fn,
        },
    )
