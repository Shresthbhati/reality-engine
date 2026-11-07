"""T-3.4: sockets, switches, radiators and ceiling lights placed on the right wall with a metric height above the floor.

Ground truth: a fixture at a known world position on the textured synthetic room (tests/layout_synth.py) is projected into a view
as a small detection box; placement must recover its wall, its position along the wall and its height to a few centimetres.
"""
import math

import numpy as np
import pytest

from perception.layout.fixtures import place_fixtures
from perception.layout.surface_layout import solve_layout
from tests.layout_synth import render

HFOV = 80.0


def _project(view, world_pt):
    """World point -> pixel in ``view`` (pinhole, same convention as the renderer)."""
    h, w = view.depth.shape
    f = (w / 2) / math.tan(math.radians(HFOV) / 2)
    cam = (np.asarray(world_pt) - view.t_wc) @ view.R_wc
    return w / 2 + f * cam[0] / cam[2], h / 2 + f * cam[1] / cam[2]


def _box(view, world_pt, half=7):
    u, v = _project(view, world_pt)
    return (u - half, v - half, u + half, v + half)


@pytest.fixture(scope="module")
def scene():
    view = render(0.0, 0.0, 0.0, 0.0, hfov=HFOV)
    return view, solve_layout(view.depth, view.masks, HFOV)


def _wall_index(layout, distance):
    return min(range(len(layout.walls)), key=lambda i: abs(layout.walls[i].plane.d - distance))


def test_a_socket_is_placed_on_its_wall_at_its_real_height(scene):
    view, layout = scene
    # a socket on the +z wall (3.0 m away), 0.30 m above the floor (y is down, floor at +1.5), 0.5 m to the right
    det = [("socket", 0.8, _box(view, (0.5, 1.5 - 0.30, 3.0)))]
    placed, refused = place_fixtures(det, layout, hfov_deg=HFOV, image_size=view.depth.shape[::-1])
    assert not refused and len(placed) == 1
    f = placed[0]
    assert f.kind == "socket" and f.wall_index == _wall_index(layout, 3.0)
    assert f.h == pytest.approx(0.30, abs=0.04)
    s_true = float(np.dot((0.5, 0, 3.0), layout.walls[f.wall_index].tangent))  # position along the wall in the ref frame
    assert f.s == pytest.approx(s_true, abs=0.05)


def test_a_switch_is_hung_at_switch_height_not_socket_height(scene):
    view, layout = scene
    det = [("switch", 0.7, _box(view, (-0.8, 1.5 - 1.2, 3.0)))]
    placed, _ = place_fixtures(det, layout, hfov_deg=HFOV, image_size=view.depth.shape[::-1])
    assert placed[0].h == pytest.approx(1.2, abs=0.05)


def test_a_ceiling_light_is_placed_on_the_ceiling_plane(scene):
    view, layout = scene
    det = [("ceiling_light", 0.6, _box(view, (0.3, -1.1, 1.8)))]  # on the ceiling (y = -1.1), 1.8 m ahead
    placed, refused = place_fixtures(det, layout, hfov_deg=HFOV, image_size=view.depth.shape[::-1])
    assert not refused and placed[0].wall_index is None
    assert placed[0].h == pytest.approx(2.6, abs=0.06)


def test_a_socket_at_an_impossible_height_is_refused_with_the_reason(scene):
    view, layout = scene
    det = [("socket", 0.9, _box(view, (0.5, 1.5 - 2.3, 3.0)))]  # near the ceiling: a detection error, not a socket
    placed, refused = place_fixtures(det, layout, hfov_deg=HFOV, image_size=view.depth.shape[::-1])
    assert placed == [] and "height" in refused[0]["reason"]


def test_a_detection_whose_ray_misses_every_wall_is_refused_not_guessed(scene):
    view, layout = scene
    det = [("socket", 0.9, (150.0, 5.0, 170.0, 20.0))]  # a patch of ceiling: no wall behind it
    placed, refused = place_fixtures(det, layout, hfov_deg=HFOV, image_size=view.depth.shape[::-1])
    assert placed == [] and refused


def test_unknown_kinds_are_reported_not_dropped(scene):
    view, layout = scene
    placed, refused = place_fixtures([("teapot", 0.9, _box(view, (0.5, 1.2, 3.0)))], layout, hfov_deg=HFOV,
                                     image_size=view.depth.shape[::-1])
    assert placed == [] and "unknown" in refused[0]["reason"]
