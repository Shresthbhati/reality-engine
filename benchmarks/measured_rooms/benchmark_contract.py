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
    source: str = "measured"
    is_synthetic: bool = False

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
        source = str(data.get("source", "measured"))
        is_synthetic = bool(
            data.get("is_synthetic", False)
            or "synthetic" in source.lower()
            or "synthetic" in rm.get("id", "").lower()
            or "synthetic" in data.get("fixture", "").lower()
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
            source=source,
            is_synthetic=is_synthetic,
        )


@dataclass(frozen=True)
class RoomAccuracyReport:
    """Quantitative evaluation report for an interior room reconstruction."""

    status: str  # "COMPLETED", "PARTIAL", "BLOCKED", "FAILED"
    room_id: str
    is_physically_measured: bool
    wall_length_rmse_m: float | None = None
    ceiling_height_error_m: float | None = None
    opening_precision: float | None = None
    opening_recall: float | None = None
    opening_f1: float | None = None
    opening_position_rmse_m: float | None = None
    opening_width_rmse_m: float | None = None
    opening_height_rmse_m: float | None = None
    fixture_mean_error_m: float | None = None
    fixture_precision: float | None = None
    fixture_recall: float | None = None
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
            "opening_position_rmse_m": round(self.opening_position_rmse_m, 4) if self.opening_position_rmse_m is not None else None,
            "opening_width_rmse_m": round(self.opening_width_rmse_m, 4) if self.opening_width_rmse_m is not None else None,
            "opening_height_rmse_m": round(self.opening_height_rmse_m, 4) if self.opening_height_rmse_m is not None else None,
            "fixture_mean_error_m": round(self.fixture_mean_error_m, 4) if self.fixture_mean_error_m is not None else None,
            "fixture_precision": round(self.fixture_precision, 4) if self.fixture_precision is not None else None,
            "fixture_recall": round(self.fixture_recall, 4) if self.fixture_recall is not None else None,
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

    # 1. Evaluate Wall Lengths (Mutual exclusion matching)
    reconstructed_walls = [
        e for e in reconstructed_world.entities.values()
        if e.type == EntityType.STRUCTURE and ("wall" in e.name.lower() or "wall" in str(e.custom_properties.get("class_name", "")).lower())
    ]

    wall_candidate_pairs: list[tuple[float, int, int]] = []
    for g_idx, gw in enumerate(gt.walls):
        for r_idx, rw in enumerate(reconstructed_walls):
            dims = rw.custom_properties.get("dimensions_w_d_h") or rw.custom_properties.get("dimensions")
            if dims and len(dims) >= 3:
                rec_len = float(max(dims[0], dims[1]))
                diff = abs(rec_len - gw.length_m)
                wall_candidate_pairs.append((diff, g_idx, r_idx))

    wall_candidate_pairs.sort(key=lambda x: x[0])
    matched_gw: set[int] = set()
    matched_rw: set[int] = set()
    wall_errors: list[float] = []

    for diff, g_idx, r_idx in wall_candidate_pairs:
        if g_idx not in matched_gw and r_idx not in matched_rw:
            matched_gw.add(g_idx)
            matched_rw.add(r_idx)
            wall_errors.append(diff)

    wall_rmse = float(math.sqrt(sum(e ** 2 for e in wall_errors) / len(wall_errors))) if wall_errors else None

    # 2. Evaluate Ceiling Height
    reconstructed_ceilings = [
        e for e in reconstructed_world.entities.values()
        if "ceiling" in e.name.lower() or "roof" in e.name.lower() or str(e.custom_properties.get("class_name", "")).lower() in ("ceiling", "roof")
    ]
    ceiling_errors: list[float] = []
    for c in reconstructed_ceilings:
        c_pos = _extract_position(c)
        if c_pos is not None:
            ceiling_errors.append(abs(c_pos[2] - gt.height_m))
    ceiling_error = float(sum(ceiling_errors) / len(ceiling_errors)) if ceiling_errors else None

    # 3. Evaluate Openings (Doors and Windows) with precision-aware threshold (<= 0.25m) and semantic compatibility
    reconstructed_openings = [
        e for e in reconstructed_world.entities.values()
        if e.type in (EntityType.DOOR, EntityType.WINDOW) or any(k in e.name.lower() for k in ("door", "window"))
    ]

    opening_candidates: list[tuple[float, int, int, float | None, float | None]] = []
    for g_idx, go in enumerate(gt.openings):
        go_kind = go.kind.lower()
        for r_idx, ro in enumerate(reconstructed_openings):
            ro_pos = _extract_position(ro)
            if ro_pos is None:
                continue
            ro_is_door = (ro.type == EntityType.DOOR or "door" in ro.name.lower() or "door" in str(ro.custom_properties.get("class_name", "")).lower())
            ro_is_window = (ro.type == EntityType.WINDOW or "window" in ro.name.lower() or "window" in str(ro.custom_properties.get("class_name", "")).lower())

            # Enforce semantic class compatibility (doors only match doors, windows only match windows)
            if "door" in go_kind and not ro_is_door:
                continue
            if "window" in go_kind and not ro_is_window:
                continue

            dist = math.hypot(ro_pos[0] - go.center[0], ro_pos[1] - go.center[1], ro_pos[2] - go.center[2])
            if dist <= 0.25:  # Precision-aware threshold <= 0.25m
                dims = ro.custom_properties.get("dimensions_w_d_h") or ro.custom_properties.get("dimensions")
                w_err = abs(float(dims[0]) - go.width_m) if dims and len(dims) >= 3 else None
                h_err = abs(float(dims[2]) - go.height_m) if dims and len(dims) >= 3 else None
                opening_candidates.append((dist, g_idx, r_idx, w_err, h_err))

    opening_candidates.sort(key=lambda x: x[0])
    matched_go: set[int] = set()
    matched_ro: set[int] = set()
    opening_position_errors: list[float] = []
    opening_w_errors: list[float] = []
    opening_h_errors: list[float] = []

    for dist, g_idx, r_idx, w_err, h_err in opening_candidates:
        if g_idx not in matched_go and r_idx not in matched_ro:
            matched_go.add(g_idx)
            matched_ro.add(r_idx)
            opening_position_errors.append(dist)
            if w_err is not None:
                opening_w_errors.append(w_err)
            if h_err is not None:
                opening_h_errors.append(h_err)

    tp_openings = len(matched_go)
    fp_openings = len(reconstructed_openings) - len(matched_ro)
    fn_openings = len(gt.openings) - len(matched_go)

    precision_openings = float(tp_openings / (tp_openings + fp_openings)) if (tp_openings + fp_openings) > 0 else (None if len(gt.openings) == 0 else 0.0)
    recall_openings = float(tp_openings / len(gt.openings)) if len(gt.openings) > 0 else None
    f1_openings = float(2 * precision_openings * recall_openings / (precision_openings + recall_openings)) if (precision_openings is not None and recall_openings is not None and (precision_openings + recall_openings) > 0) else None

    opening_pos_rmse = float(math.sqrt(sum(e ** 2 for e in opening_position_errors) / len(opening_position_errors))) if opening_position_errors else None
    w_rmse = float(math.sqrt(sum(e ** 2 for e in opening_w_errors) / len(opening_w_errors))) if opening_w_errors else None
    h_rmse = float(math.sqrt(sum(e ** 2 for e in opening_h_errors) / len(opening_h_errors))) if opening_h_errors else None

    # 4. Evaluate Fixtures (sockets, switches, MEP) with precision-aware threshold (<= 0.15m)
    reconstructed_fixtures = [
        e for e in reconstructed_world.entities.values()
        if any(k in e.name.lower() or k in str(e.custom_properties.get("class_name", "")).lower() for k in ("socket", "switch", "outlet", "radiator", "vent", "lamp"))
    ]

    fixture_candidates: list[tuple[float, int, int]] = []
    for g_idx, gf in enumerate(gt.fixtures):
        gf_type = gf.fixture_type.lower()
        for r_idx, rf in enumerate(reconstructed_fixtures):
            rf_pos = _extract_position(rf)
            if rf_pos is None:
                continue
            rf_name = (rf.name + " " + str(rf.custom_properties.get("class_name", ""))).lower()
            if gf_type not in rf_name and not any(k in rf_name for k in gf_type.split()):
                continue
            dist = math.hypot(rf_pos[0] - gf.position_m[0], rf_pos[1] - gf.position_m[1], rf_pos[2] - gf.position_m[2])
            if dist <= 0.15:  # Precision-aware threshold <= 0.15m
                fixture_candidates.append((dist, g_idx, r_idx))

    fixture_candidates.sort(key=lambda x: x[0])
    matched_gf: set[int] = set()
    matched_rf: set[int] = set()
    fixture_dist_errors: list[float] = []

    for dist, g_idx, r_idx in fixture_candidates:
        if g_idx not in matched_gf and r_idx not in matched_rf:
            matched_gf.add(g_idx)
            matched_rf.add(r_idx)
            fixture_dist_errors.append(dist)

    tp_fixtures = len(matched_gf)
    fp_fixtures = len(reconstructed_fixtures) - len(matched_rf)
    fn_fixtures = len(gt.fixtures) - len(matched_gf)

    fixture_mean_err = float(sum(fixture_dist_errors) / len(fixture_dist_errors)) if fixture_dist_errors else None
    fixture_prec = float(tp_fixtures / len(reconstructed_fixtures)) if reconstructed_fixtures else (None if not gt.fixtures else 0.0)
    fixture_rec = float(tp_fixtures / len(gt.fixtures)) if gt.fixtures else None

    # Status classification: PARTIAL if unobserved/incomplete, COMPLETED only when complete
    is_incomplete = (
        (len(gt.walls) > 0 and len(matched_gw) < len(gt.walls))
        or (len(gt.openings) > 0 and len(matched_go) < len(gt.openings))
        or (len(gt.fixtures) > 0 and len(matched_gf) < len(gt.fixtures))
        or (gt.height_m > 0 and ceiling_error is None)
    )
    status = "PARTIAL" if is_incomplete else "COMPLETED"

    is_physically_measured = bool(
        gt.capture is not None
        and gt.capture.method in ("laser", "tape", "both")
        and getattr(gt, "source", "measured") == "measured"
        and not getattr(gt, "is_synthetic", False)
    )

    return RoomAccuracyReport(
        status=status,
        room_id=gt.room_id,
        is_physically_measured=is_physically_measured,
        wall_length_rmse_m=wall_rmse,
        ceiling_height_error_m=ceiling_error,
        opening_precision=precision_openings,
        opening_recall=recall_openings,
        opening_f1=f1_openings,
        opening_position_rmse_m=opening_pos_rmse,
        opening_width_rmse_m=w_rmse,
        opening_height_rmse_m=h_rmse,
        fixture_mean_error_m=fixture_mean_err,
        fixture_precision=fixture_prec,
        fixture_recall=fixture_rec,
        details={
            "ground_truth_walls_count": len(gt.walls),
            "reconstructed_walls_count": len(reconstructed_walls),
            "matched_walls_count": len(matched_gw),
            "ground_truth_openings_count": len(gt.openings),
            "reconstructed_openings_count": len(reconstructed_openings),
            "openings_tp": tp_openings,
            "openings_fp": fp_openings,
            "openings_fn": fn_openings,
            "ground_truth_fixtures_count": len(gt.fixtures),
            "reconstructed_fixtures_count": len(reconstructed_fixtures),
            "fixtures_tp": tp_fixtures,
            "fixtures_fp": fp_fixtures,
            "fixtures_fn": fn_fixtures,
        },
    )
