"""C5: district grammar (named streets, parcels, building families, detail policy)."""

from __future__ import annotations

import json

import pytest

from procedural import spec_store as ss
from procedural.district_grammar import DistrictSpec, generate_district, parcel_assignment
from world_ir.schema_v1 import EntityType


def dump(world) -> str:
    return json.dumps(world.to_dict(), sort_keys=True)


def dump_e(world, eid) -> str:
    return json.dumps(world.entities[eid].to_dict(), sort_keys=True)


def spec(**over) -> DistrictSpec:
    kw = dict(name="dt", blocks_x=3, blocks_y=2, seed=7, detail={"max_full": 2})
    kw.update(over)
    return DistrictSpec(**kw)


def gspec(**over) -> ss.GeneratedSpec:
    return ss.GeneratedSpec("district", "dt", params=spec(**over).to_params(), seed=7)


def of(world, etype):
    return [e for e in world.entities.values() if e.type == etype]


def test_same_seed_is_byte_identical():
    assert dump(generate_district(spec())) == dump(generate_district(spec()))
    assert dump(generate_district(spec())) != dump(generate_district(spec(seed=8)))


def test_grid_structure_and_provenance():
    w = generate_district(spec())
    streets = of(w, EntityType.STREET)
    assert len(streets) == (2 + 1) + (3 + 1)
    assert {s.name for s in streets} >= {"dt Street 1", "dt Avenue 4"}
    assert len(of(w, EntityType.PLOT)) == 6 and len(of(w, EntityType.BUILDING)) == 6
    assert len(of(w, EntityType.DISTRICT)) == 1
    assert all(e.statement_state.value == "PROCEDURAL" and e.provenance.value == "GENERATED"
               for e in w.entities.values())
    details = sorted(e.custom_properties["detail"] for e in of(w, EntityType.BUILDING))
    assert details == ["footprint_only"] * 4 + ["full"] * 2


def test_footprint_only_is_one_box_with_summed_height():
    sp = spec(detail={"max_full": 0})
    w = generate_district(sp)
    b = of(w, EntityType.BUILDING)[0]
    fam = sp.families[b.custom_properties["family"]]
    assert len(b.geometry_ids) == 1
    g = w.geometries[b.geometry_ids[0]]
    assert g.bounds_max.y - g.bounds_min.y == pytest.approx(fam["storeys"] * fam["storey_height"])
    assert g.bounds_max.x - g.bounds_min.x == pytest.approx(18.0)     # 24 - 2*3 setback
    assert not of(w, EntityType.WINDOW)


def test_spine_layout_has_main_street_and_cross_streets():
    sp = spec(layout="spine", blocks_x=5, cross_every=2, detail={"max_full": 0})
    w = generate_district(sp)
    names = {e.name for e in of(w, EntityType.STREET)}
    assert "dt Main Street" in names and "dt Cross Street 4" in names   # 3 groups -> 4 cross streets
    assert len(of(w, EntityType.BUILDING)) == 10 and sp.blocks_y == 2


def test_detail_policy_distance_and_cap():
    near = parcel_assignment(spec(blocks_x=5, blocks_y=5, detail={"max_full": None, "full_within": 20.0}))
    full = [k for k, (_, d) in near.items() if d == "full"]
    assert (2, 2) in full and 0 < len(full) < 25
    capped = parcel_assignment(spec(blocks_x=5, blocks_y=5, detail={"max_full": 3}))
    assert sum(d == "full" for _, d in capped.values()) == 3


def test_editing_one_parcel_family_leaves_others_unchanged():
    w = ss.generate(None, gspec())
    old_fam = parcel_assignment(spec())[(2, 1)][0]
    new_fam = (old_fam + 1) % 3
    new, changed, removed, _ = ss.regenerate(
        w, "dt-district", {"parcels": {"2,1": {"family": new_fam}}})
    touched = {e for e in changed + removed if e != "dt-district"}
    assert touched and all(e.startswith("dt-p2-1-") for e in touched), touched
    for eid in w.entities:
        if eid != "dt-district" and not eid.startswith("dt-p2-1-"):
            assert dump_e(w, eid) == dump_e(new, eid), eid
    assert new.entities["dt-p2-1-building"].custom_properties["family"] == new_fam
    _, ch, rm, _ = ss.regenerate(w, "dt-district", {})
    assert ch == [] and rm == []


def test_upgrade_footprint_to_full_in_place():
    w = ss.generate(None, gspec(detail={"max_full": 0}))
    new, *_ = ss.regenerate(w, "dt-district", {"parcels": {"0,0": {"detail": "full"}}})
    assert new.entities["dt-p0-0-building"].custom_properties["detail"] == "full"
    assert len(new.entities) > len(w.entities)
    for eid in w.entities:
        if eid != "dt-district" and not eid.startswith("dt-p0-0-"):
            assert dump_e(w, eid) == dump_e(new, eid)


def test_validation():
    for bad in (dict(layout="ring"), dict(blocks_x=0), dict(blocks_x=40, blocks_y=40),
                dict(setback=20.0), dict(parcels={"9,9": {"family": 0}}),
                dict(parcels={"0,0": {"family": 9}}), dict(parcels={"0,0": {"detail": "x"}}),
                dict(detail={"max_full": -1}), dict(families=[]),
                dict(families=[{"storeys": 0}]), dict(lot_w=float("nan"))):
        with pytest.raises(ValueError):
            spec(**bad)


def test_stored_params_roundtrip_and_resolve_families():
    p = spec().to_params()
    assert len(p["families"]) == 3 and json.loads(json.dumps(p)) == p
    assert DistrictSpec.from_params("dt", p).to_params() == p
