"""C3: apartment grammar (rooms + shared walls + doors + corridor links)."""

from __future__ import annotations

import json

import pytest

from procedural import spec_store as ss
from procedural.apartment_grammar import generate_apartment
from procedural.layout_solver import ApartmentSpec, solve_layout
from world_ir.schema_v1 import EntityType
from world_ir.statement_state import StatementState
from provenance import Provenance

T = 0.2   # wall thickness used by the room grammar


def spec_with(extra_adjacency=(), seed=1, **over) -> ApartmentSpec:
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
        *extra_adjacency,
    ]
    kw = dict(name="flat", footprint=(10.0, 8.0), rooms=rooms,
              adjacency=adjacency, entrance="hall", seed=seed)
    kw.update(over)
    return ApartmentSpec(**kw)


def world_for(spec, index=0):
    plan = solve_layout(spec).plans[index]
    return generate_apartment(spec, plan), plan


def _aabb(world, eid):
    g = world.geometries[world.entities[eid].geometry_ids[0]]
    return (g.bounds_min.x, g.bounds_min.y, g.bounds_min.z,
            g.bounds_max.x, g.bounds_max.y, g.bounds_max.z)


def _of(world, etype):
    return [e for e in world.entities.values() if e.type == etype]


def _touch(door, floor, tol=1e-9):
    """Plan-view contact: positive shared length along one axis, touching or
    overlapping along the other (corner-only contact does not count)."""
    dx = min(door[3], floor[3]) - max(door[0], floor[0])
    dz = min(door[5], floor[5]) - max(door[2], floor[2])
    return (dx > tol and dz >= -tol) or (dz > tol and dx >= -tol)


def door_graph(world, spec):
    """Rooms joined by door GEOMETRY (a door box touches both rooms' floors),
    plus the rooms whose door touches only one floor (exterior entrance)."""
    floors = {r.id: _aabb(world, f"flat-{r.id}-floor") for r in spec.rooms}
    edges = {r.id: set() for r in spec.rooms}
    outside = set()
    for door in _of(world, EntityType.DOOR):
        box = _aabb(world, door.id)
        near = [rid for rid, f in floors.items() if _touch(box, f)]
        if len(near) == 2:
            edges[near[0]].add(near[1])
            edges[near[1]].add(near[0])
        elif len(near) == 1:
            outside.add(near[0])
        else:
            raise AssertionError((door.id, near))
    return edges, outside


def reachable(edges, start):
    seen, stack = {start}, [start]
    while stack:
        for nxt in edges[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def test_rooms_floors_and_provenance():
    spec = spec_with()
    world, _ = world_for(spec)
    for r in spec.rooms:
        assert f"flat-{r.id}-room" in world.entities
        assert f"flat-{r.id}-floor" in world.entities
        assert f"flat-{r.id}-ceiling" in world.entities
    assert "flat-apartment" in world.entities
    for e in world.entities.values():
        assert e.statement_state == StatementState.PROCEDURAL
        assert e.provenance == Provenance.GENERATED


def test_doors_for_satisfied_adjacency_and_entrance():
    spec = spec_with()
    world, plan = world_for(spec)
    door_ids = {e.id for e in _of(world, EntityType.DOOR)}
    assert "flat-door-entrance" in door_ids
    for a, b, _lvl in plan.satisfied:
        assert f"flat-door-{min(a, b)}-{max(a, b)}" in door_ids
    for door in _of(world, EntityType.DOOR):
        _x0, y0, _z0, _x1, y1, _z1 = _aabb(world, door.id)
        assert y0 == pytest.approx(0.0) and y1 == pytest.approx(2.1)   # bottom 0, top 2.1 (world Y is up)


def test_door_width_is_0_9_m():
    world, _ = world_for(spec_with())
    for door in _of(world, EntityType.DOOR):
        x0, _y0, z0, x1, _y1, z1 = _aabb(world, door.id)
        run = max(x1 - x0, z1 - z0)                  # along the wall; the other side is wall thickness
        assert run == pytest.approx(0.9, abs=1e-6), door.id


def test_every_room_reachable_from_entrance():
    for spec in (spec_with(), spec_with(extra_adjacency=[("hall", "bedroom_2", "preferred")])):
        for idx in range(3):
            world, _ = world_for(spec, idx)
            edges, outside = door_graph(world, spec)
            assert spec.entrance in outside, "entrance door must reach the outside"
            assert reachable(edges, spec.entrance) == {r.id for r in spec.rooms}, (idx, edges)


def test_no_corridor_when_every_room_already_has_a_door_path():
    spec = spec_with(adjacency=[
        ("living", "kitchen", "required"), ("living", "hall", "required"),
        ("hall", "bedroom_1", "required"), ("hall", "bedroom_2", "required"),
        ("bedroom_1", "bath_1", "required"), ("hall", "bath_2", "required"),
        ("living", "balcony", "required")], footprint=(11.0, 9.0))
    res = solve_layout(spec)
    assert res.ok, res.violations
    world = generate_apartment(spec, res.plans[0])
    assert _of(world, EntityType.CORRIDOR) == []


def test_corridor_links_are_added_only_for_rooms_without_a_door_path():
    spec = spec_with()    # bedroom_2 has no adjacency at all
    world, plan = world_for(spec)
    # components of the door graph made only of the plan's satisfied adjacencies
    ids = [r.id for r in spec.rooms]
    comp = {i: {i} for i in ids}
    for a, b, _lvl in plan.satisfied:
        merged = comp[a] | comp[b]
        for i in merged:
            comp[i] = merged
    entrance_comp = comp[spec.entrance]
    stray = {frozenset(c) for i, c in comp.items() if i not in entrance_comp}
    assert any("bedroom_2" in c for c in stray)
    corridors = _of(world, EntityType.CORRIDOR)
    assert len(corridors) == len(stray)
    assert "flat-link-bedroom_2-corridor" in {c.id for c in corridors} or         any("bedroom_2" in c and len(c) > 1 for c in stray)
    assert all(c.statement_state == StatementState.PROCEDURAL for c in corridors)


def test_one_wall_entity_per_boundary_segment():
    spec = spec_with()
    world, plan = world_for(spec)
    walls = _of(world, EntityType.WALL)
    boxes = [_aabb(world, w.id) for w in walls]
    assert len(set(boxes)) == len(boxes), "identical wall boxes found"
    rects = plan.rects
    ids = list(rects)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ax, ay, aw, ad = rects[a]
            bx, by, bw, bd = rects[b]
            # shared vertical boundary x = ax+aw (a left of b)
            for (lx, ly, lw, ld), (rx, ry, rw, rd) in (((ax, ay, aw, ad), (bx, by, bw, bd)),
                                                       ((bx, by, bw, bd), (ax, ay, aw, ad))):
                if abs(lx + lw - rx) < 1e-9:
                    lo, hi = max(ly, ry), min(ly + ld, ry + rd)
                    if hi - lo > 1e-9:
                        px, py = lx + lw + T / 2, (lo + hi) / 2     # local Z-up
                        hits = [bx_ for bx_ in boxes
                                if bx_[0] - 1e-9 <= px <= bx_[3] + 1e-9
                                and bx_[2] - 1e-9 <= -py <= bx_[5] + 1e-9]
                        assert len(hits) == 1, (a, b, hits)
                if abs(ly + ld - ry) < 1e-9:
                    lo, hi = max(lx, rx), min(lx + lw, rx + rw)
                    if hi - lo > 1e-9:
                        px, py = (lo + hi) / 2, ly + ld + T / 2
                        hits = [bx_ for bx_ in boxes
                                if bx_[0] - 1e-9 <= px <= bx_[3] + 1e-9
                                and bx_[2] - 1e-9 <= -py <= bx_[5] + 1e-9]
                        assert len(hits) == 1, (a, b, hits)


def test_geometry_is_y_up_and_inside_footprint():
    spec = spec_with()
    world, _ = world_for(spec)
    for r in spec.rooms:
        x0, y0, z0, x1, y1, z1 = _aabb(world, f"flat-{r.id}-ceiling")
        assert y0 == pytest.approx(2.8) and y1 == pytest.approx(3.0)   # height lands on world Y
        assert -0.01 <= x0 and x1 <= 10.0 + 0.01 and -8.0 - 0.01 <= z0 and z1 <= 0.01


def test_generation_is_deterministic():
    spec = spec_with()
    a, _ = world_for(spec)
    b, _ = world_for(spec)
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)


def test_origin_offsets_everything():
    spec = spec_with()
    plan = solve_layout(spec).plans[0]
    a = generate_apartment(spec, plan, origin=(0.0, 0.0, 0.0))
    b = generate_apartment(spec, plan, origin=(20.0, 0.0, 0.0))
    fa, fb = _aabb(a, "flat-hall-floor"), _aabb(b, "flat-hall-floor")
    assert fb[0] - fa[0] == pytest.approx(20.0)


def test_unsatisfied_required_plan_is_refused():
    spec = spec_with()
    plan = solve_layout(spec).plans[0]
    bad = dict(plan.rects)
    bad["kitchen"], bad["bath_2"] = bad["bath_2"], bad["kitchen"]  # swap geometry -> area/adjacency broken
    with pytest.raises(ValueError):
        generate_apartment(spec, bad)


# ---------------------------------------------------------------- C1 integration

def _stamped(spec, plan, **extra):
    params = {**spec.to_dict(), "rects": {k: list(v) for k, v in plan.rects.items()},
              "plan_index": 0, **extra}
    params.pop("name")
    return ss.GeneratedSpec("apartment", spec.name, params=params, seed=spec.seed)


def test_apartment_is_stamped_and_owns_all_its_entities():
    spec = spec_with()
    plan = solve_layout(spec).plans[0]
    world = ss.generate(None, _stamped(spec, plan))
    specs = ss.load_specs(world)
    assert set(specs) == {"flat-apartment"}
    assert specs["flat-apartment"].generator == "apartment_grammar@1"
    for eid, e in world.entities.items():
        assert e.custom_properties["generated_by"] == "flat-apartment", eid


def test_regenerate_changed_footprint_is_scoped_and_deterministic():
    spec = spec_with()
    plan = solve_layout(spec).plans[0]
    other = ss.generate(None, ss.GeneratedSpec(
        "room", "Study", params={"width": 4, "depth": 4, "height": 2.8, "openings": [], "seed": 0},
        origin=[40.0, 0.0, 0.0]))
    world = ss.generate(other, _stamped(spec, plan))
    study_before = {e: json.dumps(world.entities[e].to_dict(), sort_keys=True)
                    for e in ss.owned_ids(world, "Study-room")}

    new1, changed1, removed1, unchanged1 = ss.regenerate(world, "flat-apartment",
                                                         {"footprint": [11.0, 8.0]})
    new2, changed2, _r, _u = ss.regenerate(world, "flat-apartment", {"footprint": [11.0, 8.0]})
    assert json.dumps(new1.to_dict(), sort_keys=True) == json.dumps(new2.to_dict(), sort_keys=True)
    assert changed1 and sorted(changed1) == sorted(changed2)
    for eid, fp in study_before.items():
        assert eid in unchanged1 and json.dumps(new1.entities[eid].to_dict(), sort_keys=True) == fp
    # the new plan is valid for the new footprint (stale rects were discarded, re-solved)
    spec2 = ApartmentSpec.from_dict({"name": "flat", **ss.load_specs(new1)["flat-apartment"].params})
    floor = _aabb(new1, "flat-hall-floor")
    assert floor[3] <= 11.0 + 0.01
    assert spec2.footprint == (11.0, 8.0)


def test_regenerate_changed_room_list_removes_dropped_room():
    spec = spec_with()
    plan = solve_layout(spec).plans[0]
    world = ss.generate(None, _stamped(spec, plan))
    rooms = [r for r in spec.to_dict()["rooms"] if r["id"] != "bath_2"]
    adjacency = [a for a in spec.to_dict()["adjacency"] if "bath_2" not in a[:2]]
    new, changed, removed, _u = ss.regenerate(
        world, "flat-apartment", {"rooms": rooms, "adjacency": adjacency, "footprint": [9.0, 8.0]})
    assert "flat-bath_2-room" not in new.entities
    assert "flat-bath_2-room" in removed
    assert "flat-hall-room" in new.entities
