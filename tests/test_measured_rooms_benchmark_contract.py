"""Tests for Interior Furnished Room Reconstruction Benchmark Contract.

Validates:
1. Strict BLOCKED status when physical photos or ground truth measurements are missing.
2. Correct ingestion of RoomGroundTruth matching ground_truth.schema.json.
3. Accurate evaluation of wall RMSE, ceiling height, and opening precision/recall/F1 against physical ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.measured_rooms.benchmark_contract import (
    RoomGroundTruth,
    evaluate_room_reconstruction,
    verify_room_dataset_readiness,
)
from world_ir.schema_v1 import Entity, EntityType, Provenance
from world_ir.world_v1 import WorldIR


def test_verify_room_dataset_readiness_blocked_on_missing_photos(tmp_path: Path) -> None:
    room_dir = tmp_path / "room_01"
    room_dir.mkdir(parents=True)

    readiness = verify_room_dataset_readiness(room_dir)
    assert not readiness["ready"]
    assert readiness["status"] == "BLOCKED"
    assert any("photos" in b for b in readiness["blockers"])
    assert any("ground_truth.json" in b for b in readiness["blockers"])


def test_room_ground_truth_parsing_and_evaluation(tmp_path: Path) -> None:
    room_dir = tmp_path / "room_test"
    room_dir.mkdir(parents=True)
    photos_dir = room_dir / "photos"
    photos_dir.mkdir(parents=True)
    (photos_dir / "test_photo_01.jpg").write_text("fake_photo")

    gt_dict = {
        "room": {
            "id": "room_test",
            "name": "Living Studio",
            "area_m2": 24.5,
            "height_m": 2.80,
            "length_m": 6.0,
            "width_m": 4.08,
        },
        "walls": [
            {"id": "wall_north", "start": [0.0, 0.0], "end": [6.0, 0.0], "length_m": 6.0},
            {"id": "wall_south", "start": [0.0, 4.08], "end": [6.0, 4.08], "length_m": 6.0},
        ],
        "openings": [
            {
                "id": "door_main",
                "kind": "door",
                "center": [3.0, 0.0, 1.05],
                "width_m": 0.90,
                "height_m": 2.10,
                "sill_m": 0.0,
                "wall": "wall_north",
            },
            {
                "id": "window_south",
                "kind": "window",
                "center": [2.5, 4.08, 1.40],
                "width_m": 1.20,
                "height_m": 1.50,
                "sill_m": 0.80,
                "wall": "wall_south",
            },
        ],
        "fixtures": [
            {"id": "socket_1", "fixture_type": "socket", "position_m": [1.0, 0.0, 0.30]},
        ],
        "capture": {
            "room_id": "room_test",
            "date": "2026-11-19",
            "device_model": "iPhone 15 Pro",
            "measured_by": "Surveyor A & B",
            "method": "laser",
            "tolerance_m": 0.003,
            "consent_reference": "consent-ref-001",
            "lighting": "natural_daylight",
            "notes": "Double laser measurement with Leica Disto",
        },
    }

    gt_file = room_dir / "ground_truth.json"
    gt_file.write_text(json.dumps(gt_dict, indent=2), encoding="utf-8")

    # Verify direct RoomGroundTruth parsing
    gt = RoomGroundTruth.from_dict(gt_dict)
    assert gt.room_id == "room_test"
    assert len(gt.walls) == 2
    assert len(gt.openings) == 2

    # Readiness should now be READY
    readiness = verify_room_dataset_readiness(room_dir)
    assert readiness["ready"]
    assert readiness["status"] == "READY"

    # Create a reconstructed WorldIR that matches walls and door with small error
    world = WorldIR()
    world.entities["ent-wall_north"] = Entity(
        id="ent-wall_north",
        type=EntityType.STRUCTURE,
        name="Wall",
        transform={"position": {"x": 3.0, "y": 0.0, "z": 1.4}, "frame_id": "world"},
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"dimensions_w_d_h": [6.02, 0.2, 2.8], "class_name": "wall"},
    )
    world.entities["ent-door_main"] = Entity(
        id="ent-door_main",
        type=EntityType.DOOR,
        name="Door",
        transform={"position": {"x": 3.02, "y": 0.01, "z": 1.04}, "frame_id": "world"},
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"dimensions_w_d_h": [0.91, 0.1, 2.11], "class_name": "door"},
    )

    report = evaluate_room_reconstruction(room_dir, world)
    assert report.status == "COMPLETED"
    assert report.is_physically_measured
    assert report.wall_length_rmse_m is not None
    assert abs(report.wall_length_rmse_m - 0.02) < 1e-3  # 6.02 vs 6.0 = 2cm error
    assert report.opening_precision == 1.0  # door was correctly matched
    assert report.opening_recall == 0.5  # 1 of 2 openings matched (window was not reconstructed)
    assert report.opening_f1 is not None and abs(report.opening_f1 - 0.6667) < 1e-3


def test_parse_interior_colmap_cameras(tmp_path: Path) -> None:
    """Verifies parsing of pinhole camera intrinsics and extrinsics from COLMAP format."""
    from benchmarks.measured_rooms.run_interior_benchmark import parse_interior_colmap_cameras

    room_dir = tmp_path / "room_colmap"
    room_dir.mkdir(parents=True)

    cameras_file = room_dir / "cameras.txt"
    cameras_file.write_text("1 PINHOLE 4032 3024 9874.28 9941.91 2016.0 1512.0\n", encoding="utf-8")

    images_file = room_dir / "images.txt"
    images_file.write_text(
        "1 0.7071 0.0 0.7071 0.0 1.0 2.0 3.0 1 test_img_01.jpg\n\n",
        encoding="utf-8",
    )

    cams = parse_interior_colmap_cameras(room_dir)
    assert "test_img_01.jpg" in cams
    cam = cams["test_img_01.jpg"]
    assert cam.intrinsics.width == 4032
    assert cam.intrinsics.height == 3024
    assert abs(cam.intrinsics.fx - 9874.28) < 1e-2
    assert abs(cam.intrinsics.fy - 9941.91) < 1e-2
    assert abs(cam.intrinsics.cx - 2016.0) < 1e-2
    assert abs(cam.intrinsics.cy - 1512.0) < 1e-2
