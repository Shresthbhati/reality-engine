"""Several photos of one room -> ONE layout, with no SfM: the engine registers the photos itself and solves them together.

Ground truth: the textured 5 x 5 x 2.6 m room in tests/layout_synth.py seen from known poses.
"""
import numpy as np
import pytest

from perception.layout.registration import register_views
from perception.layout.surface_layout import ViewInput, solve_layout, solve_layout_views
from tests.layout_synth import PLANES, relative_pose, render, rotation_error_deg

WINDOW = (-0.7, 0.7, 0.9, 2.0)  # on the +z wall: 1.4 m wide, sill 0.9, head 2.0


def _views():
    # 80 degree lenses: in a 5 m room a narrower field of view cannot see floor AND ceiling at once
    kw = dict(window=WINDOW, hfov=80.0)
    return [render(0.0, 0.0, 0.0, 0.0, **kw), render(0.9, -0.2, 28.0, 0.0, **kw),
            render(1.6, -0.6, 55.0, 0.0, **kw), render(-0.8, 0.3, -30.0, 0.0, **kw)]


def test_registers_a_chain_of_photos_and_refuses_one_that_shares_nothing():
    views = _views() + [render(0.0, 0.0, 180.0, 6.0)]  # the last one looks at the opposite wall
    res = register_views([(f"v{i}", v.rgb, v.depth) for i, v in enumerate(views)], views[0].hfov, ref_index=0)
    assert set(res.poses) >= {"v0", "v1", "v3"}
    assert "v4" in res.refused and "no consistent" in res.refused["v4"]
    for i in (1, 2, 3):
        name = f"v{i}"
        if name not in res.poses:
            continue
        R, t = relative_pose(views[0], views[i])
        assert rotation_error_deg(res.poses[name].R, R) < 5.0 and np.linalg.norm(res.poses[name].t - t) < 0.3, name


def _wall_world_distance(layout_wall, view0):
    """Which true room wall (world frame) is this solved wall? Returns the matching plane name and the offset error in metres."""
    n_ref = np.asarray(layout_wall.plane.normal)
    n_world = view0.R_wc @ n_ref
    ax = int(np.argmax(np.abs(n_world)))
    # plane in the ref frame: n_ref . p + d = 0 -> in the world: n_world . (p_w - t0) + d = 0 -> n_world . p_w = n_world . t0 - d
    off = float(n_world @ view0.t_wc - layout_wall.plane.d)
    val = off / n_world[ax]
    name, err = min(((k, abs(val - v)) for k, (a, v) in PLANES.items() if a == ax and k.startswith("wall")), key=lambda x: x[1])
    return name, err


def _fused(views):
    reg = register_views([(f"v{i}", v.rgb, v.depth) for i, v in enumerate(views)], views[0].hfov, ref_index=0)
    inputs = [ViewInput(views[0].depth, views[0].masks, views[0].hfov)]
    for i in range(1, len(views)):
        if f"v{i}" in reg.poses:
            p = reg.poses[f"v{i}"]
            inputs.append(ViewInput(views[i].depth, views[i].masks, views[i].hfov, R=p.R, t=p.t, name=f"v{i}"))
    return inputs, solve_layout_views(inputs)


def test_the_fused_layout_sees_more_of_the_room_than_any_single_photo():
    views = _views()
    inputs, fused = _fused(views)
    single = solve_layout(views[0].depth, views[0].masks, views[0].hfov)
    assert len(inputs) >= 3
    assert not fused.refused, fused.refused
    fused_names = {_wall_world_distance(w, views[0])[0] for w in fused.walls}
    single_names = {_wall_world_distance(w, views[0])[0] for w in single.walls}
    assert len(fused_names) > len(single_names), (fused_names, single_names)
    for w in fused.walls:
        name, err = _wall_world_distance(w, views[0])
        assert err < 0.25, (name, err)  # every solved wall is a REAL wall of the room, at its real position
    assert fused.camera_height_m == pytest.approx(1.5, abs=0.12) and fused.room_height_m == pytest.approx(2.6, abs=0.15)


def test_a_window_seen_by_several_photos_is_one_window_not_several():
    _, fused = _fused(_views())
    wins = [o for o in fused.openings if o.kind == "window"]
    assert len(wins) == 1, wins
    assert wins[0].s_max - wins[0].s_min == pytest.approx(1.4, abs=0.25)
    assert wins[0].h_min == pytest.approx(0.9, abs=0.15) and wins[0].h_max == pytest.approx(2.0, abs=0.15)


def test_a_single_view_through_the_multiview_solver_matches_the_single_view_solver():
    v = render(0.0, 0.0, 0.0, 6.0, window=WINDOW)
    a = solve_layout(v.depth, v.masks, v.hfov)
    b = solve_layout_views([ViewInput(v.depth, v.masks, v.hfov)])
    assert a.camera_height_m == pytest.approx(b.camera_height_m) and len(a.walls) == len(b.walls) and len(a.openings) == len(b.openings)


def test_a_view_whose_vertical_disagrees_with_the_reference_is_rejected_not_blended():
    views = _views()
    reg = register_views([(f"v{i}", v.rgb, v.depth) for i, v in enumerate(views)], views[0].hfov, ref_index=0)
    p = reg.poses["v1"]
    tilt = np.array([[1, 0, 0], [0, np.cos(0.5), -np.sin(0.5)], [0, np.sin(0.5), np.cos(0.5)]])  # a 29 degree wrong pitch
    bad = ViewInput(views[1].depth, views[1].masks, views[1].hfov, R=p.R @ tilt, t=p.t, name="bad")
    fused = solve_layout_views([ViewInput(views[0].depth, views[0].masks, views[0].hfov), bad])
    assert any("bad" in n and "vertical" in n for n in fused.notes), fused.notes
