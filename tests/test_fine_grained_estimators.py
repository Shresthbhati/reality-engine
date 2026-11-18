"""Tests for fine-grained fixture and component parameter estimators."""

from __future__ import annotations

import math

from perception.fine_grained.estimators import (
    BallValveEstimator,
    PipeRunEstimator,
    SupportSurfaceClassifier,
    SwitchboardEstimator,
)


def test_switchboard_estimator_gang_count_from_aspect():
    # 1-gang standard: 86mm x 86mm (aspect 1.0)
    attr1 = SwitchboardEstimator.estimate(width_m=0.086, height_m=0.086)
    assert attr1.gang_count == 1
    assert attr1.orientation_deg == 0.0
    assert attr1.is_grounded is True

    # 2-gang horizontal: 146mm x 86mm (aspect ~1.7)
    attr2 = SwitchboardEstimator.estimate(width_m=0.146, height_m=0.086)
    assert attr2.gang_count == 2
    assert attr2.orientation_deg == 0.0

    # 2-gang vertical: 86mm x 146mm (aspect ~1.7, vertical orientation)
    attr2_v = SwitchboardEstimator.estimate(width_m=0.086, height_m=0.146)
    assert attr2_v.gang_count == 2
    assert attr2_v.orientation_deg == 90.0

    # 3-gang horizontal: 206mm x 86mm (aspect ~2.4)
    attr3 = SwitchboardEstimator.estimate(width_m=0.206, height_m=0.086)
    assert attr3.gang_count == 3

    # Detections override
    attr_custom = SwitchboardEstimator.estimate(
        width_m=0.20, height_m=0.08, module_detections=["socket_3pin", "switch", "switch", "switch"]
    )
    assert attr_custom.gang_count == 4
    assert attr_custom.confidence >= 0.90


def test_pipe_run_estimator_snaps_nominal_schedules():
    # 15mm supply pipe (measured 15.2mm)
    p15 = PipeRunEstimator.estimate(measured_thickness_m=0.0152, length_m=2.5)
    assert p15.nominal_diameter_mm == 15.0
    assert p15.nominal_size_inch == '1/2"'
    assert p15.standard_role == "potable_supply"
    assert p15.confidence > 0.90

    # 22mm branch (measured 22.1mm)
    p22 = PipeRunEstimator.estimate(measured_thickness_m=0.0221, length_m=1.8)
    assert p22.nominal_diameter_mm == 22.0
    assert p22.nominal_size_inch == '3/4"'
    assert p22.standard_role == "supply_branch"

    # 50mm branch drain (measured 54.0mm against 56mm outer)
    p50 = PipeRunEstimator.estimate(measured_thickness_m=0.054, length_m=0.8)
    assert p50.nominal_diameter_mm == 50.0
    assert p50.nominal_size_inch == '2"'
    assert p50.standard_role == "branch_drain"

    # 110mm soil stack (measured 109.0mm)
    p110 = PipeRunEstimator.estimate(measured_thickness_m=0.109, length_m=3.0)
    assert p110.nominal_diameter_mm == 110.0
    assert p110.nominal_size_inch == '4"'
    assert p110.standard_role == "soil_stack"


def test_ball_valve_estimator_open_closed_partial():
    # Flow axis along X: (1, 0, 0)
    pipe_axis = (1.0, 0.0, 0.0)

    # Handle parallel: (1, 0, 0) -> OPEN
    v_open = BallValveEstimator.estimate(handle_vector=(1.0, 0.0, 0.0), pipe_axis=pipe_axis)
    assert v_open.state == "OPEN"
    assert v_open.handle_angle_deg <= 5.0
    assert v_open.confidence > 0.90

    # Handle perpendicular: (0, 1, 0) -> CLOSED
    v_closed = BallValveEstimator.estimate(handle_vector=(0.0, 1.0, 0.0), pipe_axis=pipe_axis)
    assert v_closed.state == "CLOSED"
    assert v_closed.handle_angle_deg >= 85.0
    assert v_closed.confidence > 0.90

    # Handle at 45 deg: (1, 1, 0) -> PARTIAL
    v_partial = BallValveEstimator.estimate(handle_vector=(1.0, 1.0, 0.0), pipe_axis=pipe_axis)
    assert v_partial.state == "PARTIAL"
    assert 40.0 <= v_partial.handle_angle_deg <= 50.0


def test_support_surface_classifier():
    # Table top: horizontal plane (0, 0, 1), height = 0.75m above floor, 1.2m x 0.8m
    surf_table = SupportSurfaceClassifier.classify(
        normal=(0.0, 0.0, 1.0),
        bounds_min=(0.0, 0.0, 0.70),
        bounds_max=(1.2, 0.8, 0.75),
        floor_elevation_m=0.0,
        up_axis=2,
    )
    assert surf_table.is_support_surface is True
    assert surf_table.role == "support_surface"
    assert math.isclose(surf_table.usable_area_m2, 0.96, rel_tol=1e-3)
    assert surf_table.elevation_above_floor_m == 0.75

    # Floor slab: elevation = 0.05m
    surf_floor = SupportSurfaceClassifier.classify(
        normal=(0.0, 0.0, 1.0),
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(4.0, 5.0, 0.05),
        floor_elevation_m=0.0,
        up_axis=2,
    )
    assert surf_floor.is_support_surface is False
    assert surf_floor.role == "floor"

    # Wall plane: vertical normal (1, 0, 0)
    surf_wall = SupportSurfaceClassifier.classify(
        normal=(1.0, 0.0, 0.0),
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(0.15, 4.0, 2.7),
        floor_elevation_m=0.0,
        up_axis=2,
    )
    assert surf_wall.is_support_surface is False
    assert surf_wall.role == "wall_or_slanted"
