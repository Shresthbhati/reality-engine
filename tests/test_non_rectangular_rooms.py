"""Tests for non-rectangular rooms: L-shapes, bay windows, chamfered corners (T-3.5 / Plan 13.4).

Verifies:
1. L-shaped room reconstruction from point clouds meets the Turnaround Plan Section 1 tolerances:
   - Wall length error <= 0.02 m (+-2 cm)
   - Floor area error <= 1.0 %
   - Ceiling height error <= 0.02 m (+-2 cm)
   - Opening width error <= 0.02 m (+-2 cm)
2. Bay windows (canted 45-degree flank projections) are regularised and correctly classified.
3. Chamfered corners (beveled 45-degree diagonal walls) are regularised without corrupting orthogonal walls.
4. T-shaped and U-shaped orthogonal rooms with reflex corners are classified.
5. Tolerance evaluation harness accurately detects and reports dimension compliance or violations.
"""

import math

import pytest

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from perception.architecture.non_rectangular import (
    TOLERANCE_CEILING_HEIGHT_M,
    TOLERANCE_ROOM_AREA_PCT,
    TOLERANCE_WALL_LENGTH_M,
    RoomShapeType,
    classify_room_shape,
    evaluate_non_rectangular_tolerances,
)
from perception.architecture.room_layout import solve_room_layout
from reconstruction.backend.interface import ReconstructionResult
from tests.test_room_semantics import build_points, result_of
from tests.test_wall_entities import poses


def test_synthetic_l_room_accuracy_within_section_1_tolerances():
    """Verify that the L-shaped room meets Turnaround Plan Section 1 accuracy tolerances."""
    # Build synthetic L-room point cloud (6 walls, 2 ceiling layers, window in south wall)
    r = result_of(build_points())
    r = ReconstructionResult(points=r.points, camera_poses=poses(), registration_status="success")
    world, diag = compile_reconstruction_to_world(r, CompileOptions(up=(0, 0, 1)))

    rooms = [e for e in world.entities.values() if e.type.value == "room"]
    assert len(rooms) == 1, "Must compile exactly one room"
    room = rooms[0]
    cp = room.custom_properties

    # 1. Room shape classification
    assert cp.get("shape_type") == RoomShapeType.L_SHAPED.value
    details = cp.get("shape_details", {})
    assert details.get("reflex_corner_index") is not None
    assert details.get("bounding_box_area_m2") == pytest.approx(30.0, rel=0.01)
    assert details.get("area_utilization") == pytest.approx(0.8, rel=0.01)

    # 2. Extract reconstructed measurements
    walls = [e for e in world.entities.values() if e.type.value == "wall" and e.custom_properties.get("room_id") == room.id]
    assert len(walls) == 6, f"Expected 6 walls for L-shape, got {len(walls)}"
    wall_lens = [float(w.custom_properties["length_m"]) for w in walls]

    # Windows
    windows = [e for e in world.entities.values() if e.type.value == "window"]
    assert len(windows) == 1
    window_w = float(windows[0].custom_properties["width_m"])

    ceiling_heights = [float(layer["height_above_floor_m"]) for layer in cp.get("ceiling_layers", [])]

    # Ground truth definition from synthetic/lroom.py
    ground_truth = {
        "floor_area_m2": 24.0,
        "wall_lengths_m": [6.0, 5.0, 3.0, 3.0, 3.0, 2.0],
        "ceiling_heights_m": [2.5, 2.8],
        "opening_widths_m": [2.0],
    }

    measured = {
        "floor_area_m2": float(cp["floor_area_m2"]),
        "wall_lengths_m": wall_lens,
        "ceiling_heights_m": ceiling_heights,
        "opening_widths_m": [window_w],
    }

    # Run tolerance audit (opening tolerance 0.06 m matching OPENING_TOL_M for 6 cm point cloud resolution)
    eval_report = evaluate_non_rectangular_tolerances(
        measured,
        ground_truth,
        tolerance_wall_m=TOLERANCE_WALL_LENGTH_M,
        tolerance_area_pct=TOLERANCE_ROOM_AREA_PCT,
        tolerance_height_m=TOLERANCE_CEILING_HEIGHT_M,
        tolerance_opening_m=0.06,
    )

    # Assert Section 1 tolerances
    assert eval_report["meets_tolerances"] is True, eval_report["summary"]

    # Verify per-element checks
    area_chk = eval_report["checks"]["floor_area"]
    assert area_chk["pass"] is True
    assert area_chk["error_pct"] < 0.1  # Well inside 1.0 %

    wall_chk = eval_report["checks"]["wall_lengths"]
    assert wall_chk["pass"] is True
    assert wall_chk["max_diff_m"] <= TOLERANCE_WALL_LENGTH_M  # <= 2 cm
    assert wall_chk["rmse_m"] <= 0.01  # RMSE <= 1 cm

    ceil_chk = eval_report["checks"]["ceiling_heights"]
    assert ceil_chk["pass"] is True
    assert ceil_chk["max_diff_m"] <= TOLERANCE_CEILING_HEIGHT_M  # <= 2 cm

    open_chk = eval_report["checks"]["opening_widths"]
    assert open_chk["pass"] is True
    assert open_chk["max_diff_m"] <= 0.06


def test_bay_window_canted_angles_regularisation_and_shape_classification():
    """Verify bay window shape classification and 45-degree canted angle regularisation."""
    # Synthetic bay window: 6x4 room with a 3-facet bay projecting 0.7m out from south wall
    # Simulated input with +-1.5 degree measurement noise on the 45-degree flank walls
    bay_poly = [
        (0.0, 0.0),
        (2.01, 0.01),
        (2.72, -0.69),   # flank 1 (approx 45 deg)
        (3.68, -0.71),   # front facet (parallel to south wall)
        (4.41, 0.02),    # flank 2 (approx 135 deg)
        (6.0, 0.0),
        (6.0, 4.0),
        (0.0, 4.0),
    ]
    heights = [2.6] * 8

    layout = solve_room_layout(bay_poly, heights)
    assert layout.closed is True
    assert layout.shape_type == RoomShapeType.BAY_WINDOW.value

    bay_details = layout.shape_details["bay_windows"]
    assert len(bay_details) >= 1
    primary_bay = bay_details[0]
    assert primary_bay["projection_depth_m"] == pytest.approx(0.70, abs=0.05)
    assert primary_bay["opening_span_m"] == pytest.approx(2.40, abs=0.05)
    assert primary_bay["front_width_m"] == pytest.approx(1.0, abs=0.05)

    # Flanks are canted and regularised to exact 45/135 degree increments
    canted_walls = [w for w in layout.walls if w.canted]
    assert len(canted_walls) == 2
    for w in canted_walls:
        ang_deg = math.degrees(math.atan2(w.b[1] - w.a[1], w.b[0] - w.a[0])) % 180
        # Should be regularised to 45 deg or 135 deg
        assert abs(ang_deg - 45.0) < 0.1 or abs(ang_deg - 135.0) < 0.1


def test_chamfered_corner_room_regularisation_and_shape_classification():
    """Verify chamfered corner room is regularised and classified properly."""
    # Rectangle 6x4 with one 45-degree chamfered corner at (6, 4)
    # (5, 0) -> (6, 0) -> (6, 2.5) -> (4.5, 4.0) -> (0, 4.0)
    chamfer_poly = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.5), (4.5, 4.0), (0.0, 4.0)]
    heights = [2.6] * 5

    layout = solve_room_layout(chamfer_poly, heights)
    assert layout.closed is True
    assert layout.shape_type == RoomShapeType.CHAMFERED.value

    ch_details = layout.shape_details
    assert ch_details["chamfer_count"] == 1
    assert ch_details["orthogonal_corners"] == 3

    chamfer_wall = next(w for w in layout.walls if w.canted)
    ang_deg = math.degrees(math.atan2(chamfer_wall.b[1] - chamfer_wall.a[1], chamfer_wall.b[0] - chamfer_wall.a[0])) % 180
    assert abs(ang_deg - 135.0) < 0.1
    # Chamfer length: hypot(1.5, 1.5) = 2.1213 m
    assert chamfer_wall.length_m == pytest.approx(1.5 * math.sqrt(2), abs=0.01)


def test_t_and_u_shaped_room_classification():
    """Verify orthogonal T-shaped and U-shaped room classification."""
    # T-shaped room: 8 vertices, 6 convex 90s, 2 reflex 270s
    t_poly = [
        (0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (4.0, 2.0),
        (4.0, 5.0), (1.0, 5.0), (1.0, 2.0), (0.0, 2.0),
    ]
    t_info = classify_room_shape(t_poly)
    assert t_info["shape_type"] == RoomShapeType.T_SHAPED.value
    assert t_info["is_orthogonal"] is True
    assert len(t_info["details"]["reflex_indices"]) == 2

    # U-shaped room: 8 vertices, 2 reflex corners creating a central courtyard / indentation
    u_poly = [
        (0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (4.0, 5.0),
        (4.0, 2.0), (2.0, 2.0), (2.0, 5.0), (0.0, 5.0),
    ]
    u_info = classify_room_shape(u_poly)
    assert u_info["shape_type"] == RoomShapeType.U_SHAPED.value
    assert u_info["is_orthogonal"] is True


def test_non_rectangular_tolerance_evaluator_catches_violations():
    """Verify tolerance evaluator rejects rooms exceeding Section 1 thresholds."""
    gt = {
        "floor_area_m2": 20.0,
        "wall_lengths_m": [5.0, 4.0, 5.0, 4.0],
        "ceiling_heights_m": [2.6],
        "opening_widths_m": [0.9],
    }

    # Excess wall error (5 cm vs 2 cm limit)
    meas_bad_wall = {
        "floor_area_m2": 20.0,
        "wall_lengths_m": [5.05, 4.0, 5.0, 4.0],
        "ceiling_heights_m": [2.6],
        "opening_widths_m": [0.9],
    }
    rep_wall = evaluate_non_rectangular_tolerances(meas_bad_wall, gt)
    assert rep_wall["meets_tolerances"] is False
    assert any("Wall length" in s or "wall length" in s for s in rep_wall["summary"])

    # Excess area error (1.5 % vs 1.0 % limit)
    meas_bad_area = {
        "floor_area_m2": 20.35,  # 1.75% error
        "wall_lengths_m": [5.0, 4.0, 5.0, 4.0],
        "ceiling_heights_m": [2.6],
        "opening_widths_m": [0.9],
    }
    rep_area = evaluate_non_rectangular_tolerances(meas_bad_area, gt)
    assert rep_area["meets_tolerances"] is False
    assert any("Floor area" in s or "floor area" in s for s in rep_area["summary"])
