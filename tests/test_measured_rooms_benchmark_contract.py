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
    assert report.status == "PARTIAL"  # Incomplete reconstruction (window and 1 wall missing)
    assert report.is_physically_measured
    assert report.wall_length_rmse_m is not None
    assert abs(report.wall_length_rmse_m - 0.02) < 1e-3  # 6.02 vs 6.0 = 2cm error
    assert report.opening_precision == 1.0  # door was correctly matched
    assert report.opening_recall == 0.5  # 1 of 2 openings matched (window was not reconstructed)
    assert report.opening_f1 is not None and abs(report.opening_f1 - 0.6667) < 1e-3


def test_room_reconstruction_complete_and_fixtures(tmp_path: Path) -> None:
    """Verifies complete reconstruction with one-to-one matching, fixture error, and COMPLETED status."""
    room_dir = tmp_path / "room_full"
    room_dir.mkdir(parents=True)
    photos_dir = room_dir / "photos"
    photos_dir.mkdir(parents=True)
    (photos_dir / "p1.jpg").write_text("photo")

    gt_dict = {
        "room": {"id": "rf", "name": "Office", "area_m2": 20.0, "height_m": 2.5},
        "walls": [
            {"id": "w1", "start": [0, 0], "end": [5, 0], "length_m": 5.0},
        ],
        "openings": [
            {"id": "d1", "kind": "door", "center": [2.5, 0.0, 1.0], "width_m": 0.9, "height_m": 2.1, "sill_m": 0.0, "wall": "w1"},
        ],
        "fixtures": [
            {"id": "s1", "fixture_type": "socket", "position_m": [1.0, 0.0, 0.3]},
        ],
        "capture": {
            "room_id": "rf", "date": "2026-11-19", "device_model": "Phone", "measured_by": "Smith",
            "method": "laser", "tolerance_m": 0.005, "consent_reference": "ref", "lighting": "day",
        },
    }
    (room_dir / "ground_truth.json").write_text(json.dumps(gt_dict), encoding="utf-8")

    world = WorldIR()
    world.entities["w1"] = Entity(
        id="w1", type=EntityType.STRUCTURE, name="Wall",
        transform={"position": {"x": 2.5, "y": 0.0, "z": 1.25}, "frame_id": "world"},
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"dimensions_w_d_h": [5.01, 0.2, 2.5], "class_name": "wall"},
    )
    world.entities["d1"] = Entity(
        id="d1", type=EntityType.DOOR, name="Door",
        transform={"position": {"x": 2.52, "y": 0.0, "z": 1.01}, "frame_id": "world"},
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"dimensions_w_d_h": [0.91, 0.1, 2.12], "class_name": "door"},
    )
    world.entities["s1"] = Entity(
        id="s1", type=EntityType.INFRASTRUCTURE, name="Electrical Socket",
        transform={"position": {"x": 1.02, "y": 0.0, "z": 0.31}, "frame_id": "world"},
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"class_name": "socket"},
    )
    world.entities["c1"] = Entity(
        id="c1", type=EntityType.STRUCTURE, name="Ceiling",
        transform={"position": {"x": 2.5, "y": 0.0, "z": 2.51}, "frame_id": "world"},
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"class_name": "ceiling"},
    )

    report = evaluate_room_reconstruction(room_dir, world)
    assert report.status == "COMPLETED"
    assert report.is_physically_measured
    assert report.ceiling_height_error_m is not None
    assert abs(report.ceiling_height_error_m - 0.01) < 1e-4
    assert report.opening_precision == 1.0
    assert report.opening_recall == 1.0
    assert report.opening_f1 == 1.0
    assert report.opening_position_rmse_m is not None
    assert report.opening_position_rmse_m < 0.05
    assert report.fixture_precision == 1.0
    assert report.fixture_recall == 1.0
    assert report.fixture_mean_error_m is not None
    assert report.fixture_mean_error_m < 0.05  # ~2.2 cm error


def test_room_reconstruction_adversarial_missing_dimensions_omitted_from_rmse(tmp_path: Path) -> None:
    """Verifies that missing opening dimensions leave RMSE as None rather than 0.0."""
    room_dir = tmp_path / "room_dims"
    room_dir.mkdir(parents=True)
    (room_dir / "photos").mkdir(parents=True)
    (room_dir / "photos" / "p.jpg").write_text("photo")

    gt_dict = {
        "room": {"id": "rd", "name": "Room", "area_m2": 10.0, "height_m": 2.5},
        "walls": [],
        "openings": [
            {"id": "d1", "kind": "door", "center": [2.0, 0.0, 1.0], "width_m": 0.9, "height_m": 2.1, "sill_m": 0.0, "wall": "w"},
        ],
        "fixtures": [],
        "capture": {"room_id": "rd", "date": "2026-11-19", "device_model": "Phone", "measured_by": "A", "method": "laser", "tolerance_m": 0.01, "consent_reference": "ref", "lighting": "day"},
    }
    (room_dir / "ground_truth.json").write_text(json.dumps(gt_dict), encoding="utf-8")

    world = WorldIR()
    # Door matched by position but has NO dimensions
    world.entities["d1"] = Entity(
        id="d1", type=EntityType.DOOR, name="Door",
        transform={"position": {"x": 2.01, "y": 0.0, "z": 1.0}, "frame_id": "world"},
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"class_name": "door"},  # No dimensions_w_d_h
    )

    report = evaluate_room_reconstruction(room_dir, world)
    assert report.opening_precision == 1.0
    assert report.opening_recall == 1.0
    assert report.opening_position_rmse_m is not None
    # Missing dimensions MUST be None, not 0.0!
    assert report.opening_width_rmse_m is None
    assert report.opening_height_rmse_m is None


def test_room_reconstruction_adversarial_synthetic_provenance_flag(tmp_path: Path) -> None:
    """Verifies that synthetic ground truth is never marked as is_physically_measured=True."""
    room_dir = tmp_path / "room_synth"
    room_dir.mkdir(parents=True)
    (room_dir / "photos").mkdir(parents=True)
    (room_dir / "photos" / "p.jpg").write_text("photo")

    gt_dict = {
        "room": {"id": "synthetic_room_01", "name": "Synthetic Room", "area_m2": 10.0, "height_m": 2.5},
        "walls": [],
        "openings": [],
        "fixtures": [],
        "source": "synthetic",
        "is_synthetic": True,
        "capture": {"room_id": "synthetic_room_01", "date": "2026-11-19", "device_model": "Blender", "measured_by": "Sim", "method": "laser", "tolerance_m": 0.001, "consent_reference": "none", "lighting": "virtual"},
    }
    (room_dir / "ground_truth.json").write_text(json.dumps(gt_dict), encoding="utf-8")

    world = WorldIR()
    report = evaluate_room_reconstruction(room_dir, world)
    assert not report.is_physically_measured


def test_room_reconstruction_rejects_class_incompatible_opening_match(tmp_path: Path) -> None:
    """A window cannot match a ground-truth door even if nearby in 3D."""
    room_dir = tmp_path / "room_mismatch"
    room_dir.mkdir(parents=True)
    (room_dir / "photos").mkdir(parents=True)
    (room_dir / "photos" / "p.jpg").write_text("photo")

    gt_dict = {
        "room": {"id": "rm", "name": "Room", "area_m2": 10.0, "height_m": 2.5},
        "walls": [],
        "openings": [
            {"id": "d1", "kind": "door", "center": [2.0, 0.0, 1.0], "width_m": 0.9, "height_m": 2.1, "sill_m": 0.0, "wall": "w"},
        ],
        "fixtures": [],
        "capture": {"room_id": "rm", "date": "2026-11-19", "device_model": "Phone", "measured_by": "A", "method": "laser", "tolerance_m": 0.01, "consent_reference": "ref", "lighting": "day"},
    }
    (room_dir / "ground_truth.json").write_text(json.dumps(gt_dict), encoding="utf-8")

    world = WorldIR()
    # Reconstructed entity is a WINDOW right at the door's position
    world.entities["win1"] = Entity(
        id="win1", type=EntityType.WINDOW, name="Window",
        transform={"position": {"x": 2.01, "y": 0.0, "z": 1.0}, "frame_id": "world"},
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"class_name": "window"},
    )

    report = evaluate_room_reconstruction(room_dir, world)
    assert report.opening_precision == 0.0
    assert report.opening_recall == 0.0
    assert report.details["openings_tp"] == 0
    assert report.details["openings_fp"] == 1
    assert report.details["openings_fn"] == 1



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
