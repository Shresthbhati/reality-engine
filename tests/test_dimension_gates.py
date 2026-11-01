"""D3 (dimension reconciliation, L-room false positive) and D4 (synthetic ground-truth gate).

SYNTHETIC. The gate compares the compiled fixture against synthetic/fidelity_room/ground_truth.json; the real-room variant
(needs COLMAP and the tape measurements in datasets/README.md) is opt-in and skipped unless its inputs exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from perception.architecture.dimension_consistency import reconcile_room_dimensions
from perception.architecture.wall_slabs import layout_consistency
from tests.test_room_fidelity_fixture import _compiled, _of
from world_ir import Entity, EntityType

ROOT = Path(__file__).resolve().parents[1]
GT = json.loads((ROOT / "synthetic" / "fidelity_room" / "ground_truth.json").read_text(encoding="utf-8"))
DIM_TOL_PCT = 2.0           # plan D3: L/W/H error <= 2 % on the synthetic fixture
OPENING_TOL_M = 0.06        # the fixture grid is 5 cm; a drift beyond one cell is a regression


def _wall(i, a, b, h=2.5):
    L = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    return SimpleNamespace(index=i, a=a, b=b, length_m=L, height_m=h)


def _consistency(points):
    walls = [_wall(i, points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    return layout_consistency(SimpleNamespace(walls=walls))


# ---- false positive on non-rectangular rooms -------------------------------------------------------------------------

def test_rectangle_with_unequal_opposite_walls_is_still_flagged():
    ok = _consistency([(0, 0), (5, 0), (5, 4), (0, 4)])
    assert ok["ok"] and not ok["mismatches"]
    bad = _consistency([(0, 0), (5, 0), (4.5, 4), (0, 4)])           # a 5.0 m wall faces a 4.5 m wall
    assert not bad["ok"] and bad["mismatches"]


def test_l_shaped_loop_is_consistent_because_a_long_wall_faces_several_collinear_ones():
    res = _consistency([(0, 0), (6, 0), (6, 2), (3, 2), (3, 5), (0, 5)])
    assert res["ok"], res["mismatches"]


def test_synthetic_l_room_compiles_without_a_dimensional_inconsistency_warning():
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from reconstruction.backend.interface import ReconstructionResult
    from tests.test_room_semantics import build_points, result_of
    from tests.test_wall_entities import poses

    r = result_of(build_points())
    r = ReconstructionResult(points=r.points, camera_poses=poses(), registration_status="success")
    world, diag = compile_reconstruction_to_world(r, CompileOptions(up=(0, 0, 1)))
    assert not [w for w in diag.interior_warnings if "dimensional inconsistency" in w], diag.interior_warnings
    assert world.metadata["room_layout"]["rooms"][0]["consistency"]["ok"] is True
    # the L is not a rectangle: no width x length identity is asserted against its bounding box
    rep = world.metadata["dimension_consistency"]["rooms"][0]
    assert [c for c in rep["checks"] if c["check"] == "area_vs_extent"][0]["ok"] is None
    assert not [w for w in diag.interior_warnings if "dimension consistency" in w], diag.interior_warnings


# ---- D3 reconciliation ----------------------------------------------------------------------------------------------

def test_fixture_world_reconciles_cleanly_and_records_net_wall_area():
    _obs, world, diag = _compiled("albedo", False)
    rep = world.metadata["dimension_consistency"]
    assert rep["warnings"] == [] and not [w for w in diag.interior_warnings if "dimension consistency" in w]
    names = {c["check"]: c for c in rep["rooms"][0]["checks"]}
    assert names["area_vs_extent"]["ok"] is True and names["net_wall_area"]["ok"] is True
    cp = _of(world, EntityType.ROOM)[0].custom_properties
    assert cp["net_wall_area_m2"] == pytest.approx(cp["gross_wall_area_m2"] - cp["opening_area_m2"])
    assert cp["opening_area_m2"] > 0                      # the door and the window are carved into their host walls


def _room_world(area=20.0, states=("RECONSTRUCTED",) * 3, opening=None):
    room = Entity(id="r", type=EntityType.ROOM, name="r")
    room.custom_properties.update({
        "footprint_polygon_m": [[0, 0], [5, 0], [5, 4], [0, 4]], "floor_area_m2": area,
        "dimensions": {k: {"state": s} for k, s in zip(("floor_area_m2", "floor_dimension_u_m", "floor_dimension_v_m"), states)}})
    wall = Entity(id="w", type=EntityType.WALL, name="w")
    wall.custom_properties.update({"room_id": "r", "wall_slab": {"length_m": 5.0, "height_m": 2.6,
                                                                  "openings": [opening] if opening else []}})
    return SimpleNamespace(entities={"r": room, "w": wall}, metadata={})


def test_inconsistent_area_is_a_warning_not_a_silent_fix():
    rep = reconcile_room_dimensions(_room_world(area=14.0))
    assert any("area_vs_extent" in w and "14.000" in w for w in rep["warnings"])


def test_area_may_not_claim_a_stronger_state_than_its_dimensions():
    rep = reconcile_room_dimensions(_room_world(states=("MEASURED", "ESTIMATED", "RECONSTRUCTED")))
    assert any("dimension_states" in w for w in rep["warnings"])
    assert not reconcile_room_dimensions(_room_world(states=("RECONSTRUCTED", "RECONSTRUCTED", "MEASURED")))["warnings"]


def test_opening_must_fit_its_host_wall():
    big = {"kind": "door", "entity_id": "d", "width_m": 1.0, "height_m": 2.0, "offset_m": 4.6, "sill_m": 0.0}
    assert any("opening_fits_host" in w for w in reconcile_room_dimensions(_room_world(opening=big))["warnings"])
    assert not reconcile_room_dimensions(_room_world(opening=dict(big, offset_m=1.0)))["warnings"]


# ---- D4 ground-truth gate -------------------------------------------------------------------------------------------

def test_D4_room_length_width_height_within_tolerance_of_ground_truth():
    _obs, world, _diag = _compiled("albedo", False)
    room = _of(world, EntityType.ROOM)[0].custom_properties
    r = GT["room"]
    u, v = room["floor_dimension_u_m"], room["floor_dimension_v_m"]
    cons = room["dimensional_consistency"]
    for name, got, ref in (("footprint length", max(u, v), r["length_m"]), ("footprint width", min(u, v), r["width_m"]),
                           ("layout length", cons["length_m"], r["length_m"]), ("layout width", cons["width_m"], r["width_m"]),
                           ("height", room["height_m"], r["height_m"]), ("layout height", cons["height_m"], r["height_m"])):
        assert abs(got - ref) / ref * 100 <= DIM_TOL_PCT, (name, got, ref)
    assert cons["ok"] is True


def test_D4_door_and_window_sizes_within_tolerance_of_ground_truth():
    _obs, world, _diag = _compiled("albedo", False)
    for kind in ("door", "window"):
        gt = [o for o in GT["openings"] if o["kind"] == kind]
        got = _of(world, EntityType(kind))
        assert len(got) == len(gt) == 1, kind
        e, o = got[0].custom_properties, gt[0]
        assert abs(e["width_m"] - o["width_m"]) <= OPENING_TOL_M, (kind, e["width_m"], o["width_m"])
        assert abs(e["height_m"] - o["height_m"]) <= OPENING_TOL_M, (kind, e["height_m"], o["height_m"])


@pytest.mark.skipif(not (ROOT / "datasets" / "real_room" / "ground_truth.json").exists(),
                    reason="opt-in: needs the real photos + tape measurements and COLMAP (datasets/README.md)")
def test_D4_real_room_variant_is_opt_in():
    pytest.skip("real-room reconstruction needs COLMAP; run scripts/room_fidelity_report.py --fixture real")
