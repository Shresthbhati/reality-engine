"""Tests for Phase 6: Measured real-room dataset validation and accuracy evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator

from perception.architecture.accuracy_report import (
    Section1Tolerances,
    evaluate_measured_room_accuracy,
)
from world_ir import Entity, EntityType, Geometry, GeometryType, RelationshipKind, Vector3, WorldIR

ROOT = Path(__file__).resolve().parents[1]
MEASURED_ROOMS_DIR = ROOT / "benchmarks" / "measured_rooms"
SCHEMA_PATH = MEASURED_ROOMS_DIR / "ground_truth.schema.json"


@pytest.fixture(scope="module")
def schema_validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_six_measured_rooms_exist_and_validate_schema(schema_validator):
    room_dirs = sorted([d for d in MEASURED_ROOMS_DIR.iterdir() if d.is_dir() and d.name.startswith("room_")])
    assert len(room_dirs) == 6, f"Expected 6 measured rooms, found {len(room_dirs)}"

    for r_dir in room_dirs:
        gt_path = r_dir / "ground_truth.json"
        assert gt_path.exists(), f"Missing ground_truth.json in {r_dir.name}"
        doc = json.loads(gt_path.read_text(encoding="utf-8"))
        errors = list(schema_validator.iter_errors(doc))
        assert errors == [], f"Validation errors in {r_dir.name}: {[e.message for e in errors]}"


def test_capture_metadata_meets_protocol_spec():
    room_dirs = sorted([d for d in MEASURED_ROOMS_DIR.iterdir() if d.is_dir() and d.name.startswith("room_")])
    dev_rooms = []
    holdout_rooms = []

    for r_dir in room_dirs:
        cap_path = r_dir / "capture.json"
        assert cap_path.exists(), f"Missing capture.json in {r_dir.name}"
        cap = json.loads(cap_path.read_text(encoding="utf-8"))

        # Photo count must be 40-80 per room (per PROTOCOL.md)
        assert 40 <= cap["photo_count"] <= 80, f"Room {r_dir.name} photo_count {cap['photo_count']} not in 40..80"
        assert cap["survey"]["precision_m"] <= 0.005
        assert cap["consent_reference"]
        assert cap["camera"]["device"]

        if cap.get("split") == "blind_holdout":
            holdout_rooms.append(r_dir.name)
        else:
            dev_rooms.append(r_dir.name)

    assert len(dev_rooms) == 4, f"Expected 4 dev rooms, got {dev_rooms}"
    assert len(holdout_rooms) == 2, f"Expected 2 holdout rooms, got {holdout_rooms}"


def test_evaluate_measured_room_accuracy_on_real_ground_truth():
    gt_path = MEASURED_ROOMS_DIR / "room_01_living_studio" / "ground_truth.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    # Synthesize an accurate reconstructed WorldIR matching ground truth within 5mm
    world = WorldIR(id="wld-eval-test", name="Evaluated Living Studio")
    room_ent = Entity(
        id="room-01",
        type=EntityType.ROOM,
        custom_properties={
            "floor_area_m2": 21.30,  # GT is 21.32 (-0.09% error, passes 1.0% tolerance)
            "height_m": 2.645,  # GT is 2.65 (-5mm error, passes 2cm tolerance)
            "floor_dimension_u_m": 5.204,  # GT is 5.20 (+4mm, passes 2cm)
            "floor_dimension_v_m": 4.103,  # GT is 4.10 (+3mm, passes 2cm)
        },
    )
    world.entities[room_ent.id] = room_ent

    # 4 walls with midpoints matching GT wall positions
    for w in gt["walls"]:
        mid_x = (w["start"][0] + w["end"][0]) / 2.0
        mid_y = (w["start"][1] + w["end"][1]) / 2.0
        wall_ent = Entity(
            id=f"rec-{w['id']}",
            type=EntityType.WALL,
            custom_properties={
                "position": [mid_x, mid_y, 0.0],
                "length_m": w["length_m"] + 0.005,  # +5mm error, passes 2cm tolerance
            },
        )
        world.entities[wall_ent.id] = wall_ent

    # Openings (doors and windows mapped to EntityType.DOOR / EntityType.WINDOW)
    for op in gt.get("openings", []):
        kind = op["kind"]
        op_type = EntityType.DOOR if kind == "door" else EntityType.WINDOW
        op_ent = Entity(
            id=f"rec-{op['id']}",
            type=op_type,
            semantic_labels=[kind],
            custom_properties={
                "kind": kind,
                "position": list(op["center"]),
                "width_m": op["width_m"] - 0.004,  # -4mm error, passes 2cm
                "height_m": op["height_m"] + 0.003,  # +3mm error, passes 2cm
            },
        )
        world.entities[op_ent.id] = op_ent

    # Fixtures (sockets, switches)
    for fix in gt.get("fixtures", []):
        fix_pos = fix["position_m"]
        f_type = fix.get("fixture_type", "fixture")
        fix_ent = Entity(
            id=f"rec-{fix['id']}",
            type=EntityType.INFRASTRUCTURE,
            semantic_labels=[f_type],
            custom_properties={
                "position_m": [fix_pos[0] + 0.01, fix_pos[1], fix_pos[2]],  # +1cm error, passes 5cm
            },
        )
        world.entities[fix_ent.id] = fix_ent

    report = evaluate_measured_room_accuracy(world, gt, tolerances=Section1Tolerances())
    assert report.room_found is True
    assert report.failed_checks == 0
    assert report.passed_checks > 0
    assert report.meets_tolerances is True
    assert any(cs.element_type == "wall" and cs.precision == 1.0 for cs in report.class_scores)
    assert any(cs.element_type == "door" and cs.precision == 1.0 for cs in report.class_scores)
    assert any(cs.element_type == "window" and cs.precision == 1.0 for cs in report.class_scores)
    assert any(cs.element_type == "fixture" and cs.precision == 1.0 for cs in report.class_scores)
