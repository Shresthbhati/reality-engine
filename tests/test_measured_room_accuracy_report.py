"""Tests for Measured-Room Accuracy Report against Section 1 Tolerances (Task 12.2).

Verifies:
1. Section 1 tolerance defaults match 33_TURNAROUND_PLAN.md Section 1.
2. Room, wall, opening, and fixture accuracy checks compute accurate errors and uncertainties.
3. Formatted ASCII error table per room/element and JSON export.
4. Pass/fail verdicts per tolerance and overall gate logic.
5. CLI execution via scripts/measured_room_accuracy_report.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from perception.architecture.accuracy_report import (
    ElementAccuracyCheck,
    RoomAccuracyReport,
    Section1Tolerances,
    evaluate_measured_room_accuracy,
)
from scripts.measured_room_accuracy_report import main as cli_main
from world_ir import Entity, EntityType, WorldIR


def test_section1_tolerances_defaults():
    """Verify Section 1 tolerance values match Turnaround Plan specification."""
    tol = Section1Tolerances()
    assert tol.wall_length_m == 0.02  # +-2 cm
    assert tol.room_area_pct == 1.0  # +-1 %
    assert tol.ceiling_height_m == 0.02  # +-2 cm
    assert tol.opening_width_m == 0.02  # +-2 cm
    assert tol.opening_height_m == 0.02  # +-2 cm
    assert tol.fixture_pos_m == 0.05  # +-5 cm


def test_element_accuracy_check_serialization():
    """Verify individual element check dataclass and dictionary conversion."""
    check = ElementAccuracyCheck(
        element_id="wall-101",
        element_type="wall",
        dimension_name="length",
        measured=5.012,
        ground_truth=5.000,
        unit="m",
        error_abs=0.012,
        error_pct=0.24,
        uncertainty="+-0.010 m",
        tolerance_threshold=0.020,
        passed=True,
        notes="north wall",
    )
    d = check.to_dict()
    assert d["element_id"] == "wall-101"
    assert d["passed"] is True
    assert d["error_abs"] == 0.012
    assert d["tolerance_threshold"] == 0.020


def test_accuracy_report_formatting_and_summary():
    """Verify error table rendering with pass/fail per tolerance."""
    tol = Section1Tolerances()
    rep = RoomAccuracyReport(room_id="r-1", room_name="Living Room", tolerances=tol)

    rep.checks.append(
        ElementAccuracyCheck(
            element_id="r-1",
            element_type="room",
            dimension_name="floor_area",
            measured=24.15,
            ground_truth=24.00,
            unit="m2",
            error_abs=0.15,
            error_pct=0.625,
            uncertainty="+-0.5%",
            tolerance_threshold=1.0,
            passed=True,
        )
    )
    rep.checks.append(
        ElementAccuracyCheck(
            element_id="w-1",
            element_type="wall",
            dimension_name="length",
            measured=6.035,
            ground_truth=6.000,
            unit="m",
            error_abs=0.035,
            error_pct=0.583,
            uncertainty="+-0.012 m",
            tolerance_threshold=0.020,
            passed=False,  # Exceeds 2 cm
        )
    )

    rep.compute_summary()
    assert rep.total_checks == 2
    assert rep.passed_checks == 1
    assert rep.failed_checks == 1
    assert rep.meets_tolerances is False

    table = rep.format_table()
    assert "Living Room" in table
    assert "floor_area" in table
    assert "PASS" in table
    assert "FAIL" in table
    assert "Summary: 1/2 passed (TOLERANCES EXCEEDED)" in table

    d = rep.to_dict()
    assert d["meets_tolerances"] is False
    assert d["pass_rate_pct"] == 50.0
    assert len(d["checks"]) == 2


def test_evaluate_measured_room_accuracy_on_synthetic_fixture():
    """Verify accuracy evaluation against the synthetic fidelity fixture."""
    from scripts.room_fidelity_report import build_and_compile

    gt_path = Path("synthetic/fidelity_room/ground_truth.json")
    assert gt_path.exists()
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    _truth, _obs, world, _diag = build_and_compile(colors="albedo", voxel_m=0.08)
    rep = evaluate_measured_room_accuracy(world, gt)

    # Check that wall lengths and ceiling heights were evaluated
    check_dims = [c.dimension_name for c in rep.checks]
    assert "floor_area" in check_dims
    assert "ceiling_height" in check_dims
    assert "room_length" in check_dims
    assert "room_width" in check_dims
    assert "length" in check_dims

    # On synthetic fixture, ceiling height and room length/width match ground truth perfectly
    ch_check = next(c for c in rep.checks if c.dimension_name == "ceiling_height")
    assert ch_check.passed is True
    assert ch_check.error_abs <= 0.02

    len_check = next(c for c in rep.checks if c.dimension_name == "room_length")
    assert len_check.passed is True
    assert len_check.error_abs <= 0.02


def test_evaluate_measured_room_accuracy_custom_perfect_world():
    """Verify evaluation on a clean reconstructed world within all Section 1 tolerances."""
    world = WorldIR(id="w-perfect", name="Perfect Survey")

    room = Entity(id="room-1", type=EntityType.ROOM, name="Studio")
    room.custom_properties = {
        "floor_area_m2": 20.01,  # 0.05% error (well within 1.0%)
        "height_m": 2.605,  # 5 mm error (within 2 cm)
        "floor_dimension_u_m": 5.008,  # 8 mm error (within 2 cm)
        "floor_dimension_v_m": 3.995,  # 5 mm error (within 2 cm)
        "footprint_polygon_m": [[0, 0], [5, 0], [5, 4], [0, 4]],
    }
    world.entities[room.id] = room

    # Add 4 walls matching 5x4m
    coords = [
        ([0.0, 0.0, 0.0], [5.0, 0.0, 0.0], 5.005),
        ([5.0, 0.0, 0.0], [5.0, 4.0, 0.0], 3.998),
        ([5.0, 4.0, 0.0], [0.0, 4.0, 0.0], 5.002),
        ([0.0, 4.0, 0.0], [0.0, 0.0, 0.0], 4.004),
    ]
    for i, (s, e, l_m) in enumerate(coords):
        wid = f"w-{i+1}"
        w = Entity(id=wid, type=EntityType.WALL, name=f"Wall {i+1}")
        w.semantic_labels = ["wall_slab"]
        w.custom_properties = {
            "room_id": "room-1",
            "length_m": l_m,
            "wall_slab": {"start_xyz_m": s, "end_xyz_m": e, "length_m": l_m},
        }
        world.entities[wid] = w

    # Add door (0.905m x 2.055m) and window (1.205m x 1.105m)
    door = Entity(id="d-1", type=EntityType.DOOR, name="Door 1")
    door.custom_properties = {
        "position": [1.45, 0.0, 1.025],
        "width_m": 0.908,  # 8 mm error (within 2 cm)
        "height_m": 2.055,  # 5 mm error (within 2 cm)
    }
    world.entities[door.id] = door

    window = Entity(id="win-1", type=EntityType.WINDOW, name="Window 1")
    window.custom_properties = {
        "position": [2.9, 4.0, 1.45],
        "width_m": 1.205,  # 5 mm error (within 2 cm)
        "height_m": 1.104,  # 4 mm error (within 2 cm)
    }
    world.entities[window.id] = window

    # Add a socket fixture at [1.0, 0.0, 0.30]
    socket = Entity(id="sock-1", type=EntityType.STRUCTURE, name="Socket 1")
    socket.semantic_labels = ["socket"]
    socket.custom_properties = {
        "position_m": [1.01, 0.0, 0.31],  # ~1.4 cm error (within 5 cm)
    }
    world.entities[socket.id] = socket

    gt = {
        "room": {
            "id": "room-1",
            "name": "Studio",
            "area_m2": 20.0,
            "height_m": 2.60,
            "length_m": 5.0,
            "width_m": 4.0,
        },
        "walls": [
            {"id": "w-s", "start": [0, 0], "end": [5, 0], "length_m": 5.0},
            {"id": "w-e", "start": [5, 0], "end": [5, 4], "length_m": 4.0},
            {"id": "w-n", "start": [5, 4], "end": [0, 4], "length_m": 5.0},
            {"id": "w-w", "start": [0, 4], "end": [0, 0], "length_m": 4.0},
        ],
        "openings": [
            {"kind": "door", "center": [1.45, 0.0, 1.025], "width_m": 0.90, "height_m": 2.05},
            {"kind": "window", "center": [2.9, 4.0, 1.45], "width_m": 1.20, "height_m": 1.10},
        ],
        "fixtures": [
            {"fixture_type": "socket", "position_m": [1.0, 0.0, 0.30]},
        ],
    }

    rep = evaluate_measured_room_accuracy(world, gt)
    assert rep.meets_tolerances is True
    assert rep.failed_checks == 0
    assert rep.total_checks == 13
    assert rep.passed_checks == 13
    for c in rep.checks:
        assert c.passed is True


def test_cli_execution_synthetic_fixture(tmp_path: Path):
    """Verify CLI prints table and outputs valid JSON."""
    json_out = tmp_path / "accuracy_report.json"
    code = cli_main(["--fixture", "synthetic", "--json", str(json_out)])
    assert code == 0
    assert json_out.exists()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert "room_id" in data
    assert "checks" in data
    assert "tolerances" in data
    assert len(data["checks"]) >= 8
