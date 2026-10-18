"""C3: deterministic apartment layout solver (no ML)."""

from __future__ import annotations

import math
import time

import pytest

from procedural.layout_solver import (
    ApartmentSpec,
    adjacent_length,
    solve_layout,
    validate_plan,
)


def two_bed_spec(seed=1, **over) -> ApartmentSpec:
    rooms = [
        dict(id="living", kind="living", min_area=18, max_area=30),
        dict(id="kitchen", kind="kitchen", min_area=8, max_area=14),
        dict(id="bedroom_1", kind="bedroom", min_area=12, max_area=18),
        dict(id="bedroom_2", kind="bedroom", min_area=10, max_area=16),
        dict(id="bath_1", kind="bathroom", min_area=4, max_area=7),
        dict(id="bath_2", kind="bathroom", min_area=4, max_area=7),
        dict(id="hall", kind="hallway", min_area=6, max_area=12, aspect_max=3.5),
        dict(id="balcony", kind="balcony", min_area=4, max_area=8, aspect_max=3.0),
    ]
    adjacency = [
        ("living", "kitchen", "required"),
        ("living", "hall", "required"),
        ("hall", "bedroom_1", "required"),
        ("bedroom_1", "bath_1", "preferred"),
        ("hall", "bath_2", "preferred"),
        ("living", "balcony", "preferred"),
    ]
    kw = dict(name="flat", footprint=(10.0, 8.0), rooms=rooms,
              adjacency=adjacency, entrance="hall", seed=seed)
    kw.update(over)
    return ApartmentSpec(**kw)


def test_two_bedroom_example_solves_fast_and_is_valid():
    spec = two_bed_spec()
    t0 = time.perf_counter()
    res = solve_layout(spec)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, elapsed
    assert res.ok, res.violations
    assert 1 <= len(res.plans) <= 3
    for plan in res.plans:
        assert validate_plan(spec, plan.rects) == []
        assert plan.unsatisfied_required == []
        # required adjacencies really share >= 0.9 m of wall
        for a, b, lvl in spec.adjacency:
            if lvl == "required":
                assert adjacent_length(plan.rects[a], plan.rects[b]) >= 0.9 - 1e-9
        # 0.1 m grid
        for r in plan.rects.values():
            for v in r:
                assert abs(v * 10 - round(v * 10)) < 1e-6


def test_rooms_tile_the_footprint_without_overlap():
    spec = two_bed_spec()
    res = solve_layout(spec)
    for plan in res.plans:
        total = sum(w * d for (_x, _y, w, d) in plan.rects.values())
        assert total == pytest.approx(80.0, abs=1e-6)


def test_same_seed_identical_result():
    a = solve_layout(two_bed_spec(seed=5))
    b = solve_layout(two_bed_spec(seed=5))
    assert a.to_dict() == b.to_dict()


def test_alternatives_are_distinct_and_capped():
    for seed in (1, 2, 3):
        res = solve_layout(two_bed_spec(seed=seed), max_alternatives=3)
        assert res.ok
        sigs = {tuple(sorted((k, tuple(round(x) for x in v)) for k, v in p.rects.items()))
                for p in res.plans}
        assert len(res.plans) == 3 and len(sigs) == 3, (seed, len(res.plans))
        assert [p.score for p in res.plans] == sorted((p.score for p in res.plans), reverse=True)


def test_infeasible_three_big_bedrooms_in_small_footprint():
    spec = ApartmentSpec(
        name="tight", footprint=(5.0, 5.0),
        rooms=[dict(id=f"bedroom_{i}", kind="bedroom", min_area=20, max_area=30)
               for i in (1, 2, 3)],
        adjacency=[], entrance="bedroom_1", seed=0)
    res = solve_layout(spec)
    assert res.ok is False and res.plans == []
    text = " | ".join(res.violations)
    assert "60.0 m2" in text and "25.0 m2" in text, text


def test_single_room_does_not_fit_named():
    spec = ApartmentSpec(
        name="tiny", footprint=(4.0, 3.0),
        rooms=[dict(id="bedroom_2", kind="bedroom", min_area=14, max_area=20),
               dict(id="bath", kind="bathroom", min_area=2, max_area=4)],
        adjacency=[], entrance="bath", seed=0)
    res = solve_layout(spec)
    assert not res.ok
    assert any("bedroom_2 min_area 14 m2 cannot fit footprint 4x3" in v
               for v in res.violations), res.violations


def test_unmeetable_required_adjacency_names_the_pair_instead_of_distorting():
    # Four fixed-size 2x2 squares tile a 4x4 footprint only as a 2x2 grid,
    # where a corner room touches just two others: a-b, a-c AND a-d is impossible.
    rooms = [dict(id=i, kind="storage", min_area=4, max_area=4, aspect_max=1.0)
             for i in "abcd"]
    spec = ApartmentSpec(
        name="diag", footprint=(4.0, 4.0), rooms=rooms,
        adjacency=[("a", "b", "required"), ("a", "c", "required"), ("a", "d", "required")],
        entrance="a", seed=0)
    res = solve_layout(spec)
    assert res.ok is False and res.plans == []
    assert len(res.violations) >= 1 and all("required adjacency" in v for v in res.violations)
    assert any("'a'-'" in v for v in res.violations), res.violations


def test_footprint_derived_when_omitted():
    spec = two_bed_spec(footprint=None)
    res = solve_layout(spec)
    assert res.footprint is not None
    w, d = res.footprint
    mins = sum(r["min_area"] for r in spec.to_dict()["rooms"])
    assert mins <= w * d <= sum(r["max_area"] for r in spec.to_dict()["rooms"])
    assert res.ok, res.violations


def test_validate_plan_rejects_tampered_rects():
    spec = two_bed_spec()
    plan = solve_layout(spec).plans[0]
    bad = dict(plan.rects)
    x, y, w, d = bad["living"]
    bad["living"] = (x, y, w + 0.5, d)             # overlaps / breaks coverage
    assert validate_plan(spec, bad)
    missing = {k: v for k, v in plan.rects.items() if k != "kitchen"}
    assert any("kitchen" in m for m in validate_plan(spec, missing))
    off_grid = dict(plan.rects)
    x, y, w, d = off_grid["hall"]
    off_grid["hall"] = (x + 0.03, y, w, d)
    assert validate_plan(spec, off_grid)


@pytest.mark.parametrize("patch", [
    dict(rooms=[]),
    dict(entrance="nope"),
    dict(adjacency=[("living", "ghost", "required")]),
    dict(adjacency=[("living", "living", "required")]),
    dict(adjacency=[("living", "kitchen", "maybe")]),
    dict(footprint=(float("nan"), 5.0)),
    dict(footprint=(-1.0, 5.0)),
    dict(seed="x"),
])
def test_spec_validation_rejects_bad_input(patch):
    with pytest.raises(ValueError):
        two_bed_spec(**patch)


def test_spec_validation_numbers_and_counts():
    base = dict(id="r", kind="living")
    with pytest.raises(ValueError):
        ApartmentSpec(name="x", rooms=[dict(base, min_area=0, max_area=5)], entrance="r")
    with pytest.raises(ValueError):
        ApartmentSpec(name="x", rooms=[dict(base, min_area=float("nan"), max_area=5)], entrance="r")
    with pytest.raises(ValueError):
        ApartmentSpec(name="x", rooms=[dict(base, min_area=6, max_area=5)], entrance="r")
    with pytest.raises(ValueError):
        ApartmentSpec(name="x", rooms=[dict(base, kind="dungeon", min_area=5, max_area=6)], entrance="r")
    many = [dict(id=f"r{i}", kind="storage", min_area=1, max_area=2) for i in range(21)]
    with pytest.raises(ValueError):
        ApartmentSpec(name="x", rooms=many, entrance="r0")
    dup = [dict(id="r", kind="storage", min_area=1, max_area=2)] * 2
    with pytest.raises(ValueError):
        ApartmentSpec(name="x", rooms=dup, entrance="r")
    assert math.isfinite(solve_layout(two_bed_spec()).plans[0].score)
