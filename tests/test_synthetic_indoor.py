"""SYNTHETIC INDOOR VERIFIED -- REAL INDOOR DATASET VERIFICATION PENDING.

Controlled indoor scenes (rooms, corridor, doors, windows, stairs, ramp, storeys, clutter, occlusion) rendered through
the synthetic RGB-D sensor and pushed through the REAL architectural chain (plane detection -> orientation ->
promotion -> openings -> room graph -> storeys). Ground truth comes from the scene builders, so every assertion is a
measured error against a known answer.

Three kinds of test, deliberately separate:

  * guarantees        -- what the chain gets right inside its design envelope (clean sensor); a regression here is a
                         real regression.
  * known limitations -- behaviour measured to be WRONG, pinned as strict xfail with the reason: they flip to a
                         failure the day the engine improves, forcing the pin (and the ledger) to be updated. They are
                         never silently dropped and never weakened into passing assertions.
  * envelope          -- the same scene under realistic depth noise, asserting only what survives it.

Nothing here validates behaviour on real buildings; that needs a real indoor photo set (ledger: P7-03 BLOCKED).
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np
import pytest

from synthetic.indoor import (
    INDOOR_SENSOR_CLEAN, INDOOR_SENSOR_NOISY, corridor_with_rooms, observe, panorama, ramp_room, single_room, tower,
    two_storey_with_stairs,
)
from synthetic.scene import distance_to_scene, raycast

VOXEL = 0.12


def _geoms(world, entity):
    return [world.geometries[g] for g in entity.geometry_ids if g in world.geometries]


def _bounds(world, entity):
    g = next(x for x in _geoms(world, entity) if x.bounds_min is not None)
    return (np.array([g.bounds_min.x, g.bounds_min.y, g.bounds_min.z]),
            np.array([g.bounds_max.x, g.bounds_max.y, g.bounds_max.z]))


def _of(scene_result, kind):
    return [e for e in scene_result.world.entities.values() if e.type.value == kind]


@lru_cache(maxsize=None)
def _assembled(name: str, noisy: bool = False):
    from perception.architecture.scene import assemble_interior_scene

    ind = {"single": single_room, "corridor": corridor_with_rooms, "two": two_storey_with_stairs,
           "ramp": ramp_room, "tower": tower}[name]()
    obs = observe(ind, sensor=INDOOR_SENSOR_NOISY if noisy else INDOOR_SENSOR_CLEAN, voxel_m=VOXEL)
    return ind, obs, assemble_interior_scene(obs.result, up=ind.up, seed=42)


# ------------------------------------------------------------------------------------------- the generator itself

def test_scene_openings_are_real_holes_and_windows_are_glass():
    ind = single_room()
    door = next(o for o in ind.scene.truth["openings"] if o["kind"] == "door")
    win = next(o for o in ind.scene.truth["openings"] if o["kind"] == "window")
    # a ray through the door centre from inside passes the wall plane (y=0); the south wall elsewhere is hit
    eye = np.array([door["center"][0], 1.0, 1.0])
    t, qi = raycast(ind.scene, eye, np.array([[0.0, -1.0, 0.0]]))
    assert not np.isfinite(t[0]) or ind.scene.quads[qi[0]].ref != "room-a-wall-south"
    t, qi = raycast(ind.scene, np.array([0.6, 1.0, 1.0]), np.array([[0.0, -1.0, 0.0]]))
    assert ind.scene.quads[qi[0]].ref == "room-a-wall-south" and t[0] == pytest.approx(1.0)
    # a window ray hits GLASS at the wall plane
    eye = np.array([win["center"][0], 2.0, win["center"][2]])
    t, qi = raycast(ind.scene, eye, np.array([[0.0, 1.0, 0.0]]))
    assert ind.scene.quads[qi[0]].label == "glass" and t[0] == pytest.approx(2.0)


def test_rendered_points_lie_on_the_true_surfaces():
    ind = single_room()
    obs = observe(ind, voxel_m=0.08)
    pts = np.array([p.position for p in obs.result.points])
    err = distance_to_scene(ind.scene, pts)
    assert len(pts) > 5000
    assert np.percentile(err, 95) < 0.01 and err.max() < 0.05, (np.percentile(err, 95), err.max())


def test_a_table_hides_the_floor_beneath_it_and_glass_returns_no_depth():
    ind = single_room()
    obs = observe(ind, voxel_m=0.05)
    pts = np.array([p.position for p in obs.result.points])
    under = (pts[:, 0] > 2.05) & (pts[:, 0] < 3.15) & (pts[:, 1] > 1.25) & (pts[:, 1] < 1.95) & (pts[:, 2] < 0.1)
    assert not under.any(), "floor under the table was 'observed' through the table"
    win = next(o for o in ind.scene.truth["openings"] if o["kind"] == "window")
    in_window = (np.abs(pts[:, 0] - win["center"][0]) < win["width"] / 2 - 0.1) & (pts[:, 1] > 3.9) & \
        (pts[:, 2] > win["bottom"] + 0.1) & (pts[:, 2] < win["top"] - 0.1)
    assert not in_window.any(), "a depth sensor returned range through glass"


def test_observation_is_deterministic():
    a = observe(single_room(), voxel_m=0.2)
    b = observe(single_room(), voxel_m=0.2)
    assert [p.position for p in a.result.points] == [p.position for p in b.result.points]
    assert [p.source_evidence_ids for p in a.result.points] == [p.source_evidence_ids for p in b.result.points]


def test_each_point_names_a_camera_that_could_see_it():
    obs = observe(single_room(), voxel_m=0.2)
    cams = {c.evidence_id for c in obs.result.camera_poses}
    assert all(set(p.source_evidence_ids) <= cams for p in obs.result.points)


# ------------------------------------------------------------------------------------------------- guarantees

def test_single_room_structure_extent_and_openings_match_truth():
    ind, _obs, sc = _assembled("single")
    truth = ind.scene.truth
    assert Counter(e.type.value for e in sc.world.entities.values() if e.type.value in
                   ("floor", "ceiling", "wall")) == {"floor": 1, "ceiling": 1, "wall": 4}
    (floor,), (ceiling,) = _of(sc, "floor"), _of(sc, "ceiling")
    assert _bounds(sc.world, floor)[1][2] == pytest.approx(0.0, abs=0.03)
    assert _bounds(sc.world, ceiling)[0][2] == pytest.approx(2.6, abs=0.04)
    (room,) = sc.rooms
    t = truth["rooms"][0]
    assert room.floor_area_m2 == pytest.approx(t["area_m2"], rel=0.03)
    assert np.allclose(room.bounds_min[:2], t["min"], atol=0.05) and np.allclose(room.bounds_max[:2], t["max"], atol=0.05)
    ops = {o.kind: o for fits in sc.openings_by_plane.values() for o in fits}
    assert sorted(ops) == ["door", "window"], f"measured openings {sorted(ops)} vs truth door+window"
    assert ops["door"].width_m == pytest.approx(0.9, abs=0.12)
    assert ops["window"].width_m == pytest.approx(1.2, abs=0.1)
    assert ops["window"].sill_height_m == pytest.approx(0.9, abs=0.1)


def test_clutter_is_never_promoted_as_structure():
    ind, _obs, sc = _assembled("single")
    for wall in _of(sc, "wall"):
        lo, hi = _bounds(sc.world, wall)
        assert max(hi[0] - lo[0], hi[1] - lo[1]) > 3.0, f"{wall.id} is a pillar-scale fragment promoted as a wall"
    assert len(_of(sc, "floor")) == 1, "a table top / ramp was promoted as a second floor"


def test_corridor_scene_measures_every_door_and_window_exactly_once():
    ind, _obs, sc = _assembled("corridor")
    kinds = sorted(o.kind for fits in sc.openings_by_plane.values() for o in fits)
    assert kinds == ["door"] * 3 + ["window"] * 2, kinds
    for fits in sc.openings_by_plane.values():
        for o in fits:
            assert o.width_m == pytest.approx(0.9 if o.kind == "door" else 1.2, abs=0.12)
    plan = np.array([[r.bounds_min[0], r.bounds_min[1], r.bounds_max[0], r.bounds_max[1]] for r in sc.rooms])
    assert plan[:, 0].min() == pytest.approx(0.0, abs=0.05) and plan[:, 2].max() == pytest.approx(8.0, abs=0.05)
    assert plan[:, 1].min() == pytest.approx(0.0, abs=0.05) and plan[:, 3].max() == pytest.approx(5.0, abs=0.05)
    assert sum(r.floor_area_m2 for r in sc.rooms) == pytest.approx(40.0, rel=0.03)


def test_two_storeys_keep_their_own_floors_ceilings_and_rooms_with_a_stairwell_between():
    ind, _obs, sc = _assembled("two")
    assert len(sc.storey_entity_ids) == 2
    floors = sorted(_bounds(sc.world, e)[1][2] for e in _of(sc, "floor"))
    ceilings = sorted(_bounds(sc.world, e)[0][2] for e in _of(sc, "ceiling"))
    assert floors == pytest.approx([0.0, 2.9], abs=0.04), floors
    assert ceilings == pytest.approx([2.6, 5.5], abs=0.05), ceilings
    assert len(sc.rooms) == 2
    assert all(r.floor_area_m2 == pytest.approx(36.0, rel=0.05) for r in sc.rooms), [r.floor_area_m2 for r in sc.rooms]
    # the stairwell hole is not a phantom floor: exactly two floors, two ceilings
    assert len(_of(sc, "floor")) == 2 and len(_of(sc, "ceiling")) == 2


def _flight_points(ind, obs):
    from synthetic.scene import nearest_quad

    st = ind.scene.truth["stairs"][0]
    lo, hi = np.array(st["min"]), np.array(st["max"])
    pts = np.array([p.position for p in obs.result.points])
    box = ((pts[:, 0] >= lo[0] - 0.05) & (pts[:, 0] <= hi[0] + 0.05) & (pts[:, 1] >= lo[1] - 0.05)
           & (pts[:, 1] <= hi[1] + 0.05) & (pts[:, 2] > 0.05) & (pts[:, 2] < st["rise"] - 0.05))
    pts = pts[box]
    _d, qi = nearest_quad(ind.scene, pts)
    labels = np.array([ind.scene.quads[i].label for i in qi])
    return st, pts, labels


def _flight_cameras(heights):
    from synthetic.rgbd import look_at

    return [look_at((3.0, y, h), (4.7, y + 0.6, h - 0.4)) for h in heights for y in (1.0, 2.0, 3.0, 4.0)]


def test_stair_rhythm_is_measured_given_tread_segmentation():
    """The measurement stage, fed the points on the TREADS (ground-truth segmentation: this tests the rhythm fit, not
    segmentation). Cameras at two heights see every tread; the fit must match the true riser."""
    from perception.architecture.stairs import detect_stairs

    ind = two_storey_with_stairs()
    obs = observe(ind, cameras=_flight_cameras((1.3, 2.3)), voxel_m=0.02, stride=1)
    st, pts, labels = _flight_points(ind, obs)
    treads = [tuple(p) for p in pts[labels == "stair_tread"]]
    fit = detect_stairs(treads)
    assert fit.rise_m == pytest.approx(st["rise"] / 16, abs=0.03), (fit.rise_m, st["rise"] / 16)
    assert fit.n_steps >= 8


@pytest.mark.xfail(strict=True, reason="KNOWN LIMIT (P7-03): detect_stairs requires a regular rise between consecutive "
                   "observed bands, so a flight whose middle treads were never observed (measured gaps: 0.181 x4, "
                   "0.725, 0.181 x2 -- 0.725 is exactly 4 missed treads) is REFUSED instead of fitted with the gap "
                   "as an integer multiple of the rise.")
def test_stair_rhythm_tolerates_unobserved_treads():
    from perception.architecture.stairs import detect_stairs

    ind = two_storey_with_stairs()
    obs = observe(ind, cameras=_flight_cameras((1.3,)) + _flight_cameras((1.6,)), voxel_m=0.02, stride=1)
    st, pts, labels = _flight_points(ind, obs)
    fit = detect_stairs([tuple(p) for p in pts[labels == "stair_tread"]])
    assert fit.rise_m == pytest.approx(st["rise"] / 16, abs=0.03)


@pytest.mark.xfail(strict=True, reason="KNOWN LIMIT (P7-03): detect_stairs bins all z values into level bands, so the "
                   "riser faces of a real flight (continuous in z) merge the tread bands; it refuses on an unsegmented "
                   "scan of treads+risers (measured: bands merge to 3 spanning ~1 m). It needs horizontal-surface "
                   "segmentation first, which interior assembly does not hand it.")
def test_stair_rhythm_is_measured_from_the_unsegmented_flight():
    from perception.architecture.stairs import detect_stairs

    ind = two_storey_with_stairs()
    obs = observe(ind, cameras=_flight_cameras((1.3, 2.3)), voxel_m=0.02, stride=1)
    st, pts, _labels = _flight_points(ind, obs)
    fit = detect_stairs([tuple(p) for p in pts])
    assert fit.rise_m == pytest.approx(st["rise"] / 16, abs=0.03)


def test_ramp_is_not_a_floor_or_ceiling_and_creates_no_second_storey():
    ind, _obs, sc = _assembled("ramp")
    assert len(_of(sc, "floor")) == 1 and len(_of(sc, "ceiling")) == 1
    assert len(sc.storey_entity_ids) == 1
    (room,) = sc.rooms
    assert room.floor_area_m2 == pytest.approx(24.0, rel=0.03)
    for e in _of(sc, "floor") + _of(sc, "ceiling"):
        lo, hi = _bounds(sc.world, e)
        assert not (0.1 < (lo[2] + hi[2]) / 2 < 2.0), f"{e.id} sits at ramp height"


def test_three_disconnected_storeys_stay_three():
    ind, _obs, sc = _assembled("tower")
    assert len(sc.storey_entity_ids) == 3 and len(sc.rooms) == 3
    floors = sorted(_bounds(sc.world, e)[1][2] for e in _of(sc, "floor"))
    assert floors == pytest.approx([s["floor_z"] for s in ind.scene.truth["storeys"]], abs=0.04)
    assert all(r.floor_area_m2 == pytest.approx(16.0, rel=0.06) for r in sc.rooms)


def test_partial_observation_never_extends_the_room_beyond_what_was_seen():
    from perception.architecture.scene import SceneAssemblyError, assemble_interior_scene

    ind = single_room()
    obs = observe(ind, cameras=panorama((1.3, 1.0, 1.4), yaws=8), voxel_m=VOXEL)   # one standing position
    try:
        sc = assemble_interior_scene(obs.result, up=ind.up, seed=42)
    except SceneAssemblyError:
        return                                       # refusing is an honest answer to thin evidence
    t = ind.scene.truth["rooms"][0]
    for r in sc.rooms:
        assert r.floor_area_m2 <= t["area_m2"] * 1.03, "measured more floor than the room has"
        assert np.all(np.array(r.bounds_min[:2]) >= np.array(t["min"]) - 0.05)
        assert np.all(np.array(r.bounds_max[:2]) <= np.array(t["max"]) + 0.05)


def test_assembly_is_deterministic():
    from perception.architecture.scene import assemble_interior_scene

    ind = single_room()
    obs = observe(ind, voxel_m=0.2)
    a = assemble_interior_scene(obs.result, up=ind.up, seed=42)
    b = assemble_interior_scene(obs.result, up=ind.up, seed=42)
    assert sorted(a.world.entities) == sorted(b.world.entities)
    assert [round(r.floor_area_m2, 9) for r in a.rooms] == [round(r.floor_area_m2, 9) for r in b.rooms]


# ------------------------------------------------------------------------------------------ known limitations

@pytest.mark.xfail(strict=True, reason="KNOWN LIMIT (P7-03): rooms joined by doorways over one coplanar floor are "
                   "merged into ONE room (measured: corridor + 2 rooms -> 1 room of 40 m2). Room partition through "
                   "door-bearing shared walls is not implemented.")
def test_corridor_and_two_rooms_are_three_rooms():
    _ind, _obs, sc = _assembled("corridor")
    assert len(sc.rooms) == 3


@pytest.mark.xfail(strict=True, reason="KNOWN LIMIT (P7-03): stairs are not wired into interior assembly -- the flight "
                   "is measured by perception.architecture.stairs.detect_stairs but assemble_interior_scene creates no "
                   "stair entity and no storey link (measured: storey_links == []).")
def test_assembly_links_the_two_storeys_through_the_stairs():
    _ind, _obs, sc = _assembled("two")
    assert sc.storey_links, "no storey link produced from a measured flight of stairs"


# ---------------------------------------------------------------------------------------------------- envelope

def test_realistic_depth_noise_keeps_floor_ceiling_and_room_extent():
    ind, _obs, sc = _assembled("single", noisy=True)
    t = ind.scene.truth["rooms"][0]
    floors = [_bounds(sc.world, e)[1][2] for e in _of(sc, "floor")]
    ceilings = [_bounds(sc.world, e)[0][2] for e in _of(sc, "ceiling")]
    assert any(abs(z - 0.0) < 0.05 for z in floors) and any(abs(z - 2.6) < 0.06 for z in ceilings)
    assert len(sc.rooms) == 1 and sc.rooms[0].floor_area_m2 == pytest.approx(t["area_m2"], rel=0.10)


@pytest.mark.xfail(strict=True, reason="KNOWN LIMIT (P7-03): under realistic depth noise (sigma ~1.6 cm at 5 m, 2% "
                   "dropout) the opening scanner reports phantom openings from coverage gaps (measured: ramp room, "
                   "1 true door -> 11 openings; two-storey, 2 true -> 28). The clean-sensor guarantee above is the "
                   "supported envelope.")
def test_ramp_room_under_noise_reports_only_the_real_door():
    _ind, _obs, sc = _assembled("ramp", noisy=True)
    assert sum(len(f) for f in sc.openings_by_plane.values()) == 1
