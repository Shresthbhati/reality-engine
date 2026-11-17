"""B09: a storey has one floor; furniture tops and steps are support surfaces, never floors.

Floor of a storey = the lowest horizontal cluster that reaches >= 2 wall bases and spans >= 40 % of the footprint those
walls enclose. Anything else horizontal is ``support_surface``. Without >= 2 walls the rule cannot be assessed and no
role changes (unknown stays unknown)."""

from __future__ import annotations

import numpy as np

from perception.architecture.fixtures import find_floor_elevation
from perception.geometry.orientation import OrientedPlane, consolidate_horizontal_roles
from perception.geometry.planes import DetectedPlane
from provenance import Uncertainty

UP = (0.0, 1.0, 0.0)
CAMS = [(1.0, 1.4, 1.0), (3.0, 1.5, 2.0), (5.0, 1.3, 4.0)]
STEP = 0.1


def _mk(pid, role, normal, d, pts, positions):
    ids = [f"{pid}-{k:05d}" for k in range(len(pts))]
    for tid, p in zip(ids, pts):
        positions[tid] = tuple(float(c) for c in p)
    dp = DetectedPlane(plane_id=pid, normal=normal, d=d, inlier_ids=ids, inlier_rms_distance_m=0.01,
                       uncertainty=Uncertainty(confidence=0.9, note="x"))
    return OrientedPlane(plane=dp, role=role, normal=normal, d=d, tilt_from_role_rad=0.0,
                         uncertainty=Uncertainty(confidence=0.9, note="x"))


def _slab(pid, y, x0, x1, z0, z1, positions):
    xs, zs = np.meshgrid(np.arange(x0, x1 + 1e-9, STEP), np.arange(z0, z1 + 1e-9, STEP))
    pts = [(x, y, z) for x, z in zip(xs.ravel(), zs.ravel())]
    return _mk(pid, "floor", (0.0, 1.0, 0.0), -y, pts, positions)


def _walls(positions, y0, y1, x0=0.0, x1=6.0, z0=0.0, z1=5.0):
    """Four walls of a rectangular room standing from y0 to y1."""
    ys = np.arange(y0, y1 + 1e-9, STEP)
    out = []
    for name, fixed, ax in (("w-x0", x0, "x"), ("w-x1", x1, "x"), ("w-z0", z0, "z"), ("w-z1", z1, "z")):
        run = np.arange(z0 if ax == "x" else x0, (z1 if ax == "x" else x1) + 1e-9, STEP)
        pts = [(fixed, y, r) if ax == "x" else (r, y, fixed) for y in ys for r in run]
        n = (1.0, 0.0, 0.0) if ax == "x" else (0.0, 0.0, 1.0)
        out.append(_mk(f"{name}-{y0:g}", "wall", n, -fixed, pts, positions))
    return out


def _roles(planes):
    return {p.plane.plane_id: p.role for p in planes}


def _run(planes, pos, cams=CAMS, **kw):
    return consolidate_horizontal_roles(planes, UP, cams, pos, classify_support_surfaces=True, **kw)


def test_floor_with_a_table_top_is_one_floor_and_one_support_surface():
    pos = {}
    floor = _slab("floor", 0.0, 0, 6, 0, 5, pos)
    table = _slab("table", 0.75, 1, 2.5, 1, 2.5, pos)          # 2.25 m2 of a 30 m2 room: 7.5 %
    out, stats = _run([floor, table, *_walls(pos, 0.0, 2.5)], pos)
    assert _roles(out)["floor"] == "floor" and _roles(out)["table"] == "support_surface"
    assert stats["support_surfaces"] == 1 and "support" in out[1].uncertainty.note


def test_floor_fragment_too_small_for_the_footprint_is_a_support_surface_even_at_wall_base_height():
    pos = {}
    floor = _slab("floor", 0.0, 0, 6, 0, 5, pos)
    patch = _slab("patch", 0.12, 1, 2.5, 1, 2.5, pos)          # 0.12 above: a separate cluster, but 7.5 % of the room
    out, _ = _run([floor, patch, *_walls(pos, 0.0, 2.5)], pos)
    assert _roles(out)["floor"] == "floor" and _roles(out)["patch"] == "support_surface"


def test_floor_of_a_mezzanine_with_its_own_walls_is_a_second_storey_floor():
    pos = {}
    ground = _slab("ground", 0.0, 0, 6, 0, 5, pos)
    upper = _slab("upper", 3.0, 0, 6, 0, 5, pos)
    walls = _walls(pos, 0.0, 2.9) + _walls(pos, 3.0, 5.5)
    cams = CAMS + [(1.0, 4.2, 1.0), (3.0, 4.4, 2.0)]
    obs = {"ground": CAMS, "upper": cams[3:]}
    out, stats = _run([ground, upper, *walls], pos, cams=cams, observers=obs)
    assert _roles(out)["ground"] == "floor" and _roles(out)["upper"] == "floor"
    assert stats["support_surfaces"] == 0


def test_floor_steps_of_a_staircase_are_support_surfaces_not_floors():
    pos = {}
    floor = _slab("floor", 0.0, 0, 6, 0, 5, pos)
    steps = [_slab(f"step-{k}", 0.18 * (k + 1), 5.0 - 0.3 * k, 6.0 - 0.3 * k, 1, 2, pos) for k in range(5)]
    out, stats = _run([floor, *steps, *_walls(pos, 0.0, 2.5)], pos)
    r = _roles(out)
    assert r["floor"] == "floor" and all(r[f"step-{k}"] == "support_surface" for k in range(5))
    assert stats["support_surfaces"] == 5


def test_floor_roles_are_untouched_when_fewer_than_two_walls_make_the_rule_unassessable():
    pos = {}
    floor = _slab("floor", 0.0, 0, 6, 0, 5, pos)
    table = _slab("table", 0.75, 1, 2.5, 1, 2.5, pos)
    out, stats = _run([floor, table, *_walls(pos, 0.0, 2.5)[:1]], pos)
    assert _roles(out)["floor"] == "floor" and _roles(out)["table"] == "floor"
    assert stats["support_surfaces"] == 0


def test_floor_roles_are_untouched_unless_support_classification_is_requested():
    pos = {}
    floor = _slab("floor", 0.0, 0, 6, 0, 5, pos)
    table = _slab("table", 0.75, 1, 2.5, 1, 2.5, pos)
    out, stats = consolidate_horizontal_roles([floor, table, *_walls(pos, 0.0, 2.5)], UP, CAMS, pos)
    assert _roles(out)["table"] == "floor" and "support_surfaces" not in stats


def test_find_floor_elevation_returns_the_storey_floor_not_a_support_surface():
    class _E:                                                  # the function only reads custom_properties / position
        def __init__(self, z, role=None):
            self.custom_properties = {"elevation_m": z, **({"role": role} if role else {})}
            self.transform = None

    real, table = _E(0.0), _E(0.75, "support_surface")
    assert find_floor_elevation((1.0, 1.0, 1.2), [real, table]) == 0.0
    assert find_floor_elevation((1.0, 1.0, 1.2), [real, _E(0.75)]) == 0.75     # a plain floor entity is still honoured
