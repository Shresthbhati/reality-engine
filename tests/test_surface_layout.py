"""Single-image room layout from metric depth + surface labels (turnaround plan: replace noisy sparse-point planes).

Ground truth is an analytically ray-cast box room, so every assertion is against known geometry, not a model's opinion.
Room (camera frame, x right / y down / z forward, metres): camera 1.5 m above the floor, ceiling 2.6 m above the floor,
left wall x=-1.6, right wall x=+1.8, front wall z=3.5, a window 1.4 m wide on the front wall, sill 0.9 m, head 2.0 m.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.layout.surface_layout import backproject, solve_layout

W, H, HFOV = 320, 240, 65.0
CAM_H, ROOM_H = 1.5, 2.6


def _render(pitch_deg: float = 0.0, noise: float = 0.0, window=True, seed: int = 0):
    """Ray-cast the box room. Returns (depth metres HxW, masks dict)."""
    f = (W / 2) / math.tan(math.radians(HFOV) / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u)], -1)
    p = math.radians(pitch_deg)  # positive pitch looks DOWN (y is down)
    rot = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])  # forward (0,0,1) -> (0, sin p, cos p)
    # World frame is level; the camera is pitched down by p. A camera-frame ray r points along rot_cw @ r in world.
    world_rays = rays @ rot.T
    planes = {"floor": (1, CAM_H), "ceiling": (1, CAM_H - ROOM_H), "wall_l": (0, -1.6), "wall_r": (0, 1.8), "wall_f": (2, 3.5)}
    best_t = np.full((H, W), np.inf)
    best = np.full((H, W), "", dtype=object)
    for name, (ax, val) in planes.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            t = val / world_rays[..., ax]
        ok = (t > 0) & (t < best_t)
        best_t = np.where(ok, t, best_t)
        best = np.where(ok, name, best)
    world_pts = world_rays * best_t[..., None]
    cam_pts = world_pts @ rot  # back to the camera frame
    depth = cam_pts[..., 2]
    masks = {
        "floor": best == "floor",
        "ceiling": best == "ceiling",
        "wall": np.isin(best, ["wall_l", "wall_r", "wall_f"]),
        "window": np.zeros((H, W), bool),
        "door": np.zeros((H, W), bool),
    }
    if window:
        x, y = world_pts[..., 0], world_pts[..., 1]
        win = (best == "wall_f") & (x > -0.7) & (x < 0.7) & (y < CAM_H - 0.9) & (y > CAM_H - 2.0)
        masks["window"] = win
        masks["wall"] = masks["wall"] & ~win
        # glass: a depth model sees THROUGH a window (far away); the solver must not trust depth there
        depth = np.where(win, 12.0, depth)
    if noise:
        rng = np.random.default_rng(seed)
        depth = depth * (1.0 + rng.normal(0, noise, depth.shape))
    return depth, masks


def _up_error_deg(layout, pitch_deg):
    p = math.radians(pitch_deg)
    # world up is -y; in a camera pitched DOWN by p it reads (0, -cos p, -sin p)
    true_up = np.array([0.0, -math.cos(p), -math.sin(p)])
    c = float(np.clip(np.dot(layout.up, true_up), -1, 1))
    return math.degrees(math.acos(c))


def test_backproject_matches_pinhole_model():
    d = np.full((H, W), 2.0)
    pts = backproject(d, HFOV)
    assert pts.shape == (H, W, 3)
    assert np.allclose(pts[..., 2], 2.0)
    f = (W / 2) / math.tan(math.radians(HFOV) / 2)
    assert math.isclose(pts[0, 0, 0], (0.5 - W / 2) / f * 2.0, rel_tol=1e-9)


def test_recovers_floor_height_ceiling_and_up_vector_level():
    depth, masks = _render()
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert not lay.refused, lay.refused
    assert lay.camera_height_m == pytest.approx(CAM_H, abs=0.04)
    assert lay.room_height_m == pytest.approx(ROOM_H, abs=0.06)
    assert _up_error_deg(lay, 0.0) < 1.5


def test_recovers_up_vector_when_the_camera_is_pitched_down():
    depth, masks = _render(pitch_deg=10.0)  # at 10 degrees down the ceiling is genuinely out of frame
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert not lay.refused, lay.refused
    assert _up_error_deg(lay, 10.0) < 1.5
    assert lay.camera_height_m == pytest.approx(CAM_H, abs=0.05)
    assert lay.room_height_m is None and "ceiling not visible in this photo" in lay.notes


def test_pitched_slightly_the_ceiling_height_is_still_recovered_in_the_floor_frame():
    depth, masks = _render(pitch_deg=5.0)
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert _up_error_deg(lay, 5.0) < 1.5
    assert lay.room_height_m == pytest.approx(ROOM_H, abs=0.08)


def test_recovers_walls_and_attaches_the_window_to_the_right_wall():
    depth, masks = _render()
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    dists = sorted(round(w.plane.d, 1) for w in lay.walls)
    assert dists == [1.6, 1.8, 3.5], dists  # exactly the three visible walls, no phantoms
    wins = [o for o in lay.openings if o.kind == "window"]
    assert len(wins) == 1
    win = wins[0]
    assert lay.walls[win.wall_index].plane.d == pytest.approx(3.5, abs=0.06)  # the FRONT wall
    assert win.s_max - win.s_min == pytest.approx(1.4, abs=0.12)
    assert win.h_min == pytest.approx(0.9, abs=0.1) and win.h_max == pytest.approx(2.0, abs=0.1)


def test_window_geometry_ignores_depth_seen_through_the_glass():
    depth, masks = _render()
    assert depth[masks["window"]].min() > 10  # the fixture really does poison depth inside the window
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    win = next(o for o in lay.openings if o.kind == "window")
    assert 0.8 < win.h_min < 1.0 and 1.9 < win.h_max < 2.1


def test_tolerates_depth_noise():
    depth, masks = _render(pitch_deg=5.0, noise=0.02, seed=3)  # a level camera sees only a 0.5 m floor strip here
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert not lay.refused, lay.refused
    assert lay.camera_height_m == pytest.approx(CAM_H, abs=0.08)
    assert lay.room_height_m == pytest.approx(ROOM_H, abs=0.15)
    assert len(lay.walls) == 3


def test_refuses_instead_of_guessing_when_there_is_no_floor():
    depth, masks = _render()
    masks = dict(masks, floor=np.zeros((H, W), bool))
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert lay.refused and "floor" in lay.refused[0]
    assert lay.camera_height_m is None and lay.walls == ()


def test_a_ceiling_label_on_noise_is_not_accepted_as_a_ceiling():
    depth, masks = _render()
    rng = np.random.default_rng(1)
    bad = np.zeros((H, W), bool)
    bad[10:40, 10:60] = True
    depth = depth.copy()
    depth[bad] = rng.uniform(0.5, 9.0, int(bad.sum()))  # scattered depths: not a plane
    masks = dict(masks, ceiling=bad)
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert lay.room_height_m is None
    assert any("ceiling" in r for r in lay.refused)


def test_wall_label_on_clutter_does_not_create_extra_walls():
    depth, masks = _render()
    depth = depth.copy()
    masks = dict(masks)
    clutter = np.zeros((H, W), bool)
    clutter[150:170, 100:110] = True  # a thin sliver labelled "wall" at an unrelated depth
    depth[clutter] = 1.2
    masks["wall"] = masks["wall"] | clutter
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert len(lay.walls) == 3


def test_a_narrow_floor_strip_is_flagged_because_the_up_vector_is_then_weakly_constrained():
    depth, masks = _render(pitch_deg=0.0)  # level camera: only ~0.5 m of floor is in frame
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert any("floor spans only" in n for n in lay.notes), lay.notes
    depth, masks = _render(pitch_deg=10.0)
    assert not any("floor spans only" in n for n in solve_layout(depth, masks, hfov_deg=HFOV).notes)


def test_one_slightly_bent_wall_is_one_wall_not_two():
    """Monocular depth bends a flat wall by a few degrees; that must not become two walls."""
    depth, masks = _render()
    pts = backproject(depth, HFOV)
    front = masks["wall"] & (np.abs(depth - 3.5) < 0.02)
    left_half = front & (pts[..., 0] < 0)
    depth = depth.copy()
    depth[left_half] = depth[left_half] + 0.10 * pts[..., 0][left_half]  # slope 0.1 -> ~5.7 degrees of bend
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert len(lay.walls) == 3, [(round(w.plane.d, 2), w.support_px) for w in lay.walls]


def test_weak_and_implausible_walls_are_dropped_with_a_reason():
    depth, masks = _render()
    depth = depth.copy()
    masks = dict(masks)
    sliver = np.zeros((H, W), bool)
    sliver[60:200, 5:25] = True  # a block of "wall" pixels only 0.3 m from the camera
    depth[sliver] = 0.3
    masks["wall"] = masks["wall"] | sliver
    lay = solve_layout(depth, masks, hfov_deg=HFOV)
    assert all(w.plane.d >= 0.4 for w in lay.walls)  # the 0.3 m block is not a wall
    assert any(abs(w.plane.d - 3.5) < 0.06 for w in lay.walls)
    assert any('implausible' in n for n in lay.notes)
