"""The RELATIVE working unit: COLMAP's scale is arbitrary, so it is normalised before any metre-based threshold sees it."""

from __future__ import annotations

import pytest

from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult
from reconstruction.scene_scale import NOMINAL_VIEW_DISTANCE, view_distance, working_unit


def _result(scale: float) -> ReconstructionResult:
    """Two cameras 4 units apart looking at a wall of points 3 units away, all multiplied by an arbitrary solver scale."""
    cams = [ReconstructedCameraPose("a", (0.0, 0.0, 0.0), (1, 0, 0, 0)),
            ReconstructedCameraPose("b", (4.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5))]
    cams = [ReconstructedCameraPose(c.evidence_id, tuple(v * scale for v in c.position), c.rotation) for c in cams]
    pts = [ReconstructedPoint((x * scale, 0.0, 3.0 * scale), f"t{i}", ["a", "b"]) for i, x in enumerate(range(5))]
    return ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")


@pytest.mark.parametrize("scale", [0.003, 1.0, 900.0])
def test_unit_is_independent_of_the_solver_scale(scale):
    out, facts = working_unit(_result(scale))
    assert view_distance(out) == pytest.approx(NOMINAL_VIEW_DISTANCE)
    assert facts["unit_k"] == pytest.approx(NOMINAL_VIEW_DISTANCE / view_distance(_result(scale)))
    assert facts["reused_from_head"] is False and "NOT metres" in facts["basis"]


def test_rotations_ids_and_order_survive():
    src = _result(7.0)
    out, _ = working_unit(src)
    assert [p.evidence_id for p in out.camera_poses] == ["a", "b"]
    assert [p.rotation for p in out.camera_poses] == [p.rotation for p in src.camera_poses]
    assert [p.track_id for p in out.points] == [p.track_id for p in src.points]


def test_head_unit_is_kept_while_the_solve_agrees_and_dropped_when_it_does_not():
    src = _result(1.0)
    own = NOMINAL_VIEW_DISTANCE / view_distance(src)
    _, kept = working_unit(src, head_k=own * 1.1)           # same solver frame: one unit across versions
    assert kept["reused_from_head"] and kept["unit_k"] == pytest.approx(own * 1.1)
    _, dropped = working_unit(src, head_k=own * 2.0)        # a different solve: the head's unit does not apply
    assert not dropped["reused_from_head"] and dropped["unit_k"] == pytest.approx(own)


def _box_room(scale: float, step: float = 0.4) -> ReconstructionResult:
    """A 5 x 4 x 2.6 room (floor, ceiling, four walls; 1 cm noise) in an arbitrary solver frame and scale."""
    import numpy as np

    rng = np.random.default_rng(1)
    w, d, h = 5.0, 4.0, 2.6
    xs, ys, zs = (np.arange(0, v + 1e-9, step) for v in (w, h, d))
    pts = ([(x, 0, z) for x in xs for z in zs] + [(x, h, z) for x in xs for z in zs]
           + [(x, y, 0) for x in xs for y in ys] + [(x, y, d) for x in xs for y in ys]
           + [(0, y, z) for y in ys for z in zs] + [(w, y, z) for y in ys for z in zs])
    cams = [(1.5 + i * 0.6, 1.4, 1.2 + (i % 2) * 1.5) for i in range(5)]
    ids = [f"ev{i}" for i in range(5)]
    th = 0.4
    rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    P = (np.array(pts) + rng.normal(0, 0.01, (len(pts), 3))) @ rot.T * scale
    C = np.array(cams) @ rot.T * scale
    return ReconstructionResult(
        points=[ReconstructedPoint(tuple(p), f"t{i}", ids) for i, p in enumerate(P)],
        camera_poses=[ReconstructedCameraPose(i, tuple(c), (1, 0, 0, 0)) for i, c in zip(ids, C)],
        registration_status="success")


@pytest.mark.parametrize("scale", [0.003, 900.0])
def test_a_room_is_found_whatever_scale_the_solver_chose(scale):
    from engine.compiler.world_compiler import compile_reconstruction_to_world
    from reconstruction.frame import FrameCanonicalizationError, canonicalize_frame

    raw = _box_room(scale)
    # the defect: at these solver scales the metre-based thresholds see no plane (no 'up', no room) ...
    try:
        rc, _ = canonicalize_frame(raw, 42)
        raw_world, _ = compile_reconstruction_to_world(rc)
        raw_kinds = {e.type.value for e in raw_world.entities.values()}
    except FrameCanonicalizationError:
        raw_kinds = set()
    assert "room" not in raw_kinds
    # ... and in the working unit the same capture compiles to the same structure at every scale
    unit, _ = working_unit(raw)
    rc, _ = canonicalize_frame(unit, 42)
    world, _ = compile_reconstruction_to_world(rc)
    kinds = [e.type.value for e in world.entities.values()]
    assert kinds.count("floor") >= 1 and kinds.count("ceiling") >= 1 and kinds.count("wall") >= 4


def test_unmeasurable_scale_is_left_alone():
    empty = ReconstructionResult(points=[], camera_poses=[], registration_status="failed")
    out, facts = working_unit(empty)
    assert out is empty and facts is None


def test_room_prior_finds_one_floor_one_ceiling_and_four_walls_in_a_noisy_box():
    """RANSAC shreds a furnished room into dozens of fragments; the room prior must give the room's own planes."""
    from perception.geometry.room_box import detect_room_planes

    raw = _box_room(1.0, step=0.1)
    det = detect_room_planes(raw)
    assert det is not None
    horizontal = [p for p in det.planes if abs(p.normal[1]) > 0.9]
    vertical = [p for p in det.planes if abs(p.normal[1]) < 0.1]
    assert len(horizontal) == 2 and len(vertical) == 4


def test_room_prior_declines_a_cloud_that_is_not_a_room():
    from perception.geometry.room_box import detect_room_planes

    pts = [ReconstructedPoint((i * 0.01, (i * 7 % 13) * 0.3, (i * 5 % 11) * 0.2), f"t{i}", ["a"]) for i in range(400)]
    cams = [ReconstructedCameraPose(c, (float(i), 1.0, 0.0), (1, 0, 0, 0)) for i, c in enumerate("abc")]
    assert detect_room_planes(ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")) is None
