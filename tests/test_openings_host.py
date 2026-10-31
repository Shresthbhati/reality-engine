"""S4: attach a detected opening to its host wall (plane distance + in-plane bounds) and express it in the wall's frame."""

from types import SimpleNamespace

import pytest

from perception.architecture.openings import host_opening_on_wall


def _wall(a, b, h=2.6):
    L = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    return SimpleNamespace(a=a, b=b, height_m=h, length_m=L)


def _box(a0, a1, b0, b1, h0, h1):
    return [(a, b, h) for a in (a0, a1) for b in (b0, b1) for h in (h0, h1)]


WALLS = [_wall((0, 0), (6, 0)), _wall((6, 0), (6, 4)), _wall((6, 4), (0, 4)), _wall((0, 4), (0, 0))]


def test_door_on_the_south_wall_gets_its_in_wall_rectangle():
    r = host_opening_on_wall(_box(1.0, 1.9, -0.02, 0.02, 0.0, 2.05), WALLS)
    assert r.wall_index == 0
    assert (r.offset_m, r.width_m, r.sill_m, r.height_m) == pytest.approx((1.0, 0.9, 0.0, 2.05))


def test_offset_is_measured_along_the_walls_own_direction():
    # the north wall runs from x=6 to x=0, so a window at x 1.5..3.5 starts 2.5 m along it
    r = host_opening_on_wall(_box(1.5, 3.5, 3.9, 4.1, 0.9, 2.1), WALLS)
    assert r.wall_index == 2
    assert r.offset_m == pytest.approx(2.5) and r.width_m == pytest.approx(2.0)
    assert r.sill_m == pytest.approx(0.9) and r.height_m == pytest.approx(1.2)


def test_recessed_window_still_belongs_to_the_wall_in_front_of_it():
    r = host_opening_on_wall(_box(1.5, 3.5, 0.0, 0.15, 0.9, 2.1), WALLS)
    assert r.wall_index == 0 and r.plane_distance_m == pytest.approx(0.075)


def test_far_from_every_wall_has_no_host():
    assert host_opening_on_wall(_box(2.0, 3.0, 1.9, 2.1, 0.9, 2.0), WALLS) is None


def test_opening_mostly_outside_the_walls_extent_has_no_host():
    assert host_opening_on_wall(_box(5.0, 7.5, -0.02, 0.02, 0.0, 2.0), WALLS) is None


def test_rectangle_is_clipped_to_the_wall_and_to_its_height():
    r = host_opening_on_wall(_box(5.0, 6.3, -0.02, 0.02, -0.1, 2.9), [_wall((0, 0), (6, 0), 2.5)])
    assert r.offset_m == pytest.approx(5.0) and r.width_m == pytest.approx(1.0)
    assert r.sill_m == 0.0 and r.height_m == pytest.approx(2.5)


def test_nearest_of_two_candidate_walls_wins():
    near = [_wall((0, 0), (6, 0)), _wall((0, 0.3), (6, 0.3))]
    assert host_opening_on_wall(_box(1, 2, 0.25, 0.35, 0.9, 2.0), near).wall_index == 1
