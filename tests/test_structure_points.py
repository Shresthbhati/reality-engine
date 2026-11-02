"""structure_points: the corners that define a room, from the compiled wall slabs only (scale-free, deduplicated)."""

import numpy as np
import pytest

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from reconstruction.scene_scale import rescale, working_unit
from reconstruction.structure_points import structure_points
from tests.test_room_semantics import POLY, WINDOW, build_points, result_of
from tests.test_wall_entities import poses
from reconstruction.backend.interface import ReconstructionResult
from world_ir.artifact_store import MemoryArtifactStore

SLAB, FALSE = 2.8, 2.5      # wall heights in the L-room fixture (the y<=3 wing has a false ceiling)


def _compiled(scale: float):
    r = result_of(build_points())
    r = ReconstructionResult(points=r.points, camera_poses=poses(), registration_status="success")
    r = rescale(r, scale)
    r, facts = working_unit(r)           # the production path: RELATIVE results are normalised before compiling
    world, _ = compile_reconstruction_to_world(r, CompileOptions(up=(0, 0, 1), artifact_store=MemoryArtifactStore()))
    return world, scale * facts["unit_k"]


@pytest.fixture(scope="module", params=[0.003, 1.0, 900.0])
def case(request):
    world, k = _compiled(request.param)
    return structure_points(world), k


def _has(points, xyz, tol):
    return any(np.linalg.norm(np.array(p["position"]) - xyz) <= tol for p in points)


def test_every_floor_corner_and_a_ceiling_corner_above_it_are_found_at_any_scale(case):
    pts, k = case
    tol = 0.15 * k
    corners = [p for p in pts if p["kind"] == "corner"]
    for x, y in POLY:
        assert _has(corners, np.array([x, y, 0.0]) * k, tol), (x, y)
        assert any(abs(p["position"][0] - x * k) <= tol and abs(p["position"][1] - y * k) <= tol
                   and min(abs(p["position"][2] - FALSE * k), abs(p["position"][2] - SLAB * k)) <= tol for p in corners)
    assert len(corners) == len({tuple(np.round(p["position"], 9)) for p in corners})      # no coincident duplicates


def test_window_gives_four_opening_corners_on_its_host_wall(case):
    pts, k = case
    x0, x1, z0, z1 = WINDOW
    op = [p for p in pts if p["kind"] == "opening_corner"]
    assert len(op) == 4 and all(p["label"] == "window" for p in op)
    for x in (x0, x1):
        for z in (z0, z1):
            assert _has(op, np.array([x, 0.0, z]) * k, 0.15 * k), (x, z)
    assert len({e for p in op for e in p["entity_ids"]}) == 2          # the window and its one host wall


def test_shared_corners_are_merged_not_repeated(case):
    pts, _ = case
    shared = [p for p in pts if p["kind"] == "corner" and len(p["entity_ids"]) > 2]
    assert shared, "adjacent walls must share their corners"
    assert len({p["id"] for p in pts}) == len(pts)


def test_no_structure_gives_no_points():
    assert structure_points([]) == []
    world = compile_reconstruction_to_world(result_of([(0, 0, 0), (1, 0, 0)]))[0]
    assert structure_points(world) == []
