"""C4: building grammar (storeys, facade windows, stairs, roof) + scoped regeneration."""

from __future__ import annotations

import json
import math

import pytest

from procedural import spec_store as ss
from procedural.building_grammar import BuildingSpec, core_rect, generate_building
from world_ir.schema_v1 import EntityType, RelationshipKind

T = 0.2          # wall / slab thickness
W, D = 12.0, 10.0


def layout():
    return {
        "rooms": [
            dict(id="living", kind="living", min_area=25, max_area=40),
            dict(id="kitchen", kind="kitchen", min_area=10, max_area=16),
            dict(id="bedroom_1", kind="bedroom", min_area=14, max_area=22),
            dict(id="bedroom_2", kind="bedroom", min_area=12, max_area=18),
            dict(id="bath", kind="bathroom", min_area=4, max_area=8),
            dict(id="hall", kind="hallway", min_area=6, max_area=14, aspect_max=3.5),
        ],
        "adjacency": [["living", "kitchen", "required"], ["living", "hall", "required"],
                      ["hall", "bedroom_1", "required"], ["hall", "bedroom_2", "preferred"],
                      ["hall", "bath", "preferred"]],
        "entrance": "hall", "seed": 1,
    }


def spec(**over) -> BuildingSpec:
    kw = dict(name="tower", footprint=(W, D), storeys=3, floor_layout=layout(),
              core={"stairs": True, "side": "east"})
    kw.update(over)
    return BuildingSpec(**kw)


@pytest.fixture(scope="module")
def built():
    s = spec()
    return s, generate_building(s)


def of(world, etype):
    return [e for e in world.entities.values() if e.type == etype]


def aabb(world, eid):
    """Union AABB over the entity's geometries, world axes (x, y-up, z)."""
    gs = [world.geometries[g] for g in world.entities[eid].geometry_ids]
    return (min(g.bounds_min.x for g in gs), min(g.bounds_min.y for g in gs),
            min(g.bounds_min.z for g in gs), max(g.bounds_max.x for g in gs),
            max(g.bounds_max.y for g in gs), max(g.bounds_max.z for g in gs))


# ---------------------------------------------------------------- entity counts

def test_counts_three_storey_two_bedroom(built):
    s, w = built
    assert len(of(w, EntityType.STOREY)) == 3
    assert len(of(w, EntityType.STAIRS)) == 2
    assert len(of(w, EntityType.ROOF)) == 1
    assert len([e for e in of(w, EntityType.DOOR) if e.id == "tower-entrance"]) == 1
    # interior rooms exist on every storey from the stored layout
    for k in range(3):
        for rid in ("living", "kitchen", "bedroom_1", "bedroom_2", "bath", "hall"):
            assert f"tower-s{k}-{rid}-room" in w.entities
    # exterior walls: 4 per storey
    assert all(f"tower-s{k}-ext-{side}" in w.entities
               for k in range(3) for side in ("south", "north", "east", "west"))


def test_windows_on_all_four_sides_every_storey(built):
    _, w = built
    for k in range(3):
        sides = {e.id.split("-win-")[1].split("-")[0] for e in of(w, EntityType.WINDOW)
                 if e.id.startswith(f"tower-s{k}-win-")}
        assert sides == {"south", "north", "east", "west"}, (k, sides)


def test_everything_is_generated_and_namespaced(built):
    _, w = built
    for e in w.entities.values():
        assert e.id.startswith("tower-"), e.id
        assert e.statement_state.name == "PROCEDURAL" and e.provenance.name == "GENERATED"


def test_determinism(built):
    s, w = built
    a = json.dumps(w.to_dict(), sort_keys=True)
    b = json.dumps(generate_building(spec()).to_dict(), sort_keys=True)
    assert a == b


def test_open_plate_when_no_layout():
    w = generate_building(spec(floor_layout=None))
    for k in range(3):
        assert f"tower-s{k}-room" in w.entities
        assert w.entities[f"tower-s{k}"].relationships[0].target_id == f"tower-s{k}-room"
    assert len(of(w, EntityType.STOREY)) == 3


# ------------------------------------------------------------------- stairs

def test_stairs_rise_equals_storey_height_and_connect(built):
    s, w = built
    heights = [3.0, 3.0, 3.0]
    for k in range(2):
        e = w.entities[f"tower-stairs-{k}"]
        x0, y0, z0, x1, y1, z1 = aabb(w, e.id)
        assert math.isclose(y1 - y0, heights[k], abs_tol=1e-6)        # world Y = up
        assert math.isclose(y0, sum(heights[:k]), abs_tol=1e-6)
        assert len(e.geometry_ids) == math.ceil(heights[k] / 0.18)
        assert math.isclose(x1 - x0, 1.1, abs_tol=1e-6)
        targets = {r.target_id for r in e.relationships if r.kind == RelationshipKind.ADJACENT_TO}
        assert targets == {f"tower-s{k}", f"tower-s{k + 1}"}


def test_storeys_connected_through_stairs_graph(built):
    _, w = built
    adj = {f"tower-s{k}": set() for k in range(3)}
    for e in of(w, EntityType.STAIRS):
        t = [r.target_id for r in e.relationships if r.kind == RelationshipKind.ADJACENT_TO]
        assert len(t) == 2
        adj[t[0]].add(t[1])
        adj[t[1]].add(t[0])
    seen, stack = {"tower-s0"}, ["tower-s0"]
    while stack:
        for n in adj[stack.pop()]:
            if n not in seen:
                seen.add(n)
                stack.append(n)
    assert seen == set(adj)
    building = w.entities["tower-building"]
    contained = {r.target_id for r in building.relationships if r.kind == RelationshipKind.CONTAINS}
    assert {f"tower-s{k}" for k in range(3)} <= contained


def test_stairs_have_headroom(built):
    _, w = built
    # the upper floor slab is open over the stair run: no slab piece may cover a
    # point where stair top + 2.0 m would pierce the slab underside
    for k in (1, 2):
        steps = [w.geometries[g] for g in w.entities[f"tower-stairs-{k - 1}"].geometry_ids]
        underside = 3.0 * k - T            # world y of the slab underside above storey k-1
        for piece in [e for e in w.entities.values()
                      if e.id.startswith(f"tower-s{k}-floor-")]:
            px0, _, pz0, px1, _, pz1 = aabb(w, piece.id)
            for g in steps:
                overlap = (min(px1, g.bounds_max.x) - max(px0, g.bounds_min.x) > 1e-9 and
                           min(pz1, g.bounds_max.z) - max(pz0, g.bounds_min.z) > 1e-9)
                if overlap:
                    assert underside - g.bounds_max.y >= 2.0 - 1e-9


def test_windows_never_overlap_core_or_corner(built):
    s, w = built
    cx0, cy0, cx1, cy1 = core_rect(s)            # local plan rect (x, y)
    m = 0.3
    for e in of(w, EntityType.WINDOW):
        x0, _, z0, x1, _, z1 = aabb(w, e.id)
        y0, y1 = -z1, -z0                        # world z -> local y
        assert not (x0 < cx1 + m and x1 > cx0 - m and y0 < cy1 + m and y1 > cy0 - m), e.id
        side = e.id.split("-win-")[1].split("-")[0]
        lo, hi, length = ((x0, x1, W) if side in ("south", "north") else (y0, y1, D))
        assert lo >= m - 1e-9 and hi <= length - m + 1e-9, (e.id, lo, hi)


def test_infeasible_stairs_raise():
    with pytest.raises(ValueError, match="stair"):
        generate_building(spec(footprint=(5.0, 5.0), floor_layout=None))


@pytest.mark.parametrize("over", [
    dict(storeys=0), dict(storeys=31), dict(storey_heights=[3.0, 3.0]),
    dict(storey_heights=[3.0, 3.0, 7.0]), dict(storey_heights=[3.0, 2.0, 3.0]),
    dict(core={"side": "up"}), dict(footprint=(4.0, 12.0)), dict(footprint=(12.0, math.nan)),
    dict(facade={"bay": 9.0}), dict(facade={"bay": 1.0}), dict(facade={"sill": 1.5, "window_h": 1.5}),
    dict(facade={"bogus": 1}), dict(core={"nope": 1}),
])
def test_bad_specs_raise(over):
    with pytest.raises(ValueError):
        generate_building(spec(floor_layout=None, **over))


# --------------------------------------------------- spec_store + scoped regen

def stored(**params):
    base = dict(footprint=[W, D], storeys=3, storey_heights=None, floor_layout=None,
                plan_index=0, core={"stairs": True, "side": "east"}, facade={}, roof=True, seed=0)
    base.update(params)
    return ss.generate(None, ss.GeneratedSpec("building", "tower", params=base))


def test_building_is_stamped_and_regenerable():
    w = stored()
    assert "tower-building" in ss.load_specs(w)
    owned = set(ss.owned_ids(w, "tower-building"))
    assert owned == set(w.entities)
    w2, changed, removed, unchanged = ss.regenerate(w, "tower-building", {"roof": False})
    assert "tower-roof" in removed
    assert _storey_ids(w, (0, 1, 2)) <= set(unchanged)


def _storey_ids(world, ks):
    return {e for e in world.entities if any(e == f"tower-s{k}" or e.startswith(f"tower-s{k}-")
                                              for k in ks)}


def test_top_storey_height_change_keeps_lower_storeys_identical():
    w = stored(storey_heights=[3.0, 3.0, 3.0])
    w2, changed, removed, unchanged = ss.regenerate(w, "tower-building",
                                                    {"storey_heights": [3.0, 3.0, 4.2]})
    low = _storey_ids(w, (0, 1)) | {"tower-stairs-0", "tower-stairs-1"}
    assert low <= set(unchanged), sorted(low - set(unchanged))
    assert _storey_ids(w, (2,)) & set(changed)       # the top storey did change
    assert "tower-roof" in changed
    before = ss._fingerprints(w)
    after = ss._fingerprints(w2)
    assert all(before[e] == after[e] for e in low)   # byte-identical


def test_adding_a_storey_keeps_existing_storeys_identical():
    w = stored(storey_heights=[3.0, 3.0, 3.0])
    w2, changed, removed, unchanged = ss.regenerate(
        w, "tower-building", {"storeys": 4, "storey_heights": [3.0, 3.0, 3.0, 3.0]})
    keep = _storey_ids(w, (0, 1, 2)) | {"tower-stairs-0", "tower-stairs-1"}
    assert keep <= set(unchanged), sorted(keep - set(unchanged))
    assert "tower-s3" in w2.entities and "tower-stairs-2" in w2.entities
    before, after = ss._fingerprints(w), ss._fingerprints(w2)
    assert all(before[e] == after[e] for e in keep)


def test_middle_height_change_only_moves_that_storey_and_above():
    w = stored(storey_heights=[3.0, 3.0, 3.0])
    _, changed, _, unchanged = ss.regenerate(w, "tower-building",
                                             {"storey_heights": [3.0, 3.4, 3.0]})
    assert _storey_ids(w, (0,)) <= set(unchanged)
    assert _storey_ids(w, (1,)) & set(changed) and _storey_ids(w, (2,)) & set(changed)
