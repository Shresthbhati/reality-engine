"""geo.osm_context: roofs, landscape areas, hedges, tree rows, tagged trees and generated woodland (fixtures only)."""

from __future__ import annotations

import json
from pathlib import Path

from geo import osm_context as osm

FX = Path(__file__).parent / "fixtures" / "osm"
LAT, LON = 22.5757, 88.4312


def _world(elements=None):
    els = elements if elements is not None else json.loads((FX / "synthetic_gardens_roofs.json").read_text("utf-8"))["elements"]
    return osm.osm_to_world(els, LAT, LON, "T", radius_m=300, fetched_at="2026-10-31T00:00:00+00:00")


def _e(w, eid):
    return w.entities[eid].custom_properties


def _generated(w):
    return [e for e in w.entities.values() if e.custom_properties.get("generated_by")]


def test_query_asks_for_landscape_and_tree_rows():
    q = osm.build_query(LAT, LON, 300)
    for needle in ("tree_row", "garden", "flowerbed", "orchard", "wetland", "multipolygon", "shrub"):
        assert needle in q


def test_kinds_and_counts():
    c = osm.counts_by_kind(_world())
    imported = {k: v for k, v in c.items() if k != "tree"}
    assert imported == {"building": 5, "fence": 1, "flowerbed": 1, "garden": 1, "hedge": 1, "shrub": 1, "tree_row": 1,
                        "water": 1, "woodland": 1}
    w = _world()
    tagged = [e for e in w.entities.values() if e.custom_properties["kind"] == "tree" and e.custom_properties.get("imported")]
    assert len(tagged) == 3


def test_roof_tags_and_basis_wording():
    w = _world()
    hip, skil, odd, dflt, apt = (_e(w, f"osm-way-{i}")["roof"] for i in (7001, 7002, 7003, 7004, 7005))
    assert hip["shape"] == "hipped" and hip["height_m"] == 3.0 and "OSM roof:shape tag" in hip["basis"]
    assert skil["shape"] == "skillion" and "roof:angle" in skil["basis"]
    assert odd["shape"] == "flat" and "not recognised" in odd["basis"]
    assert dflt["shape"] == "gabled" and dflt["basis"].startswith("default for building=house (not mapped)")
    assert apt["shape"] == "flat" and apt["basis"] == "default for building=apartments (not mapped)"
    for i in (7001, 7002, 7004):  # roof heights are never measured
        d = _e(w, f"osm-way-{i}")["dimensions"]["roof_height"]
        assert d["state"] == "ESTIMATED" and d["value"] == _e(w, f"osm-way-{i}")["roof"]["height_m"]
    assert "roof_height" not in _e(w, "osm-way-7005")["dimensions"]


def test_tagged_height_includes_roof_and_untagged_sits_on_walls():
    els = [{"type": "way", "id": 1, "tags": {"building": "house", "height": "9", "roof:shape": "gabled", "roof:height": "3"},
            "geometry": _sq(22.5770, 88.4300)},
           {"type": "way", "id": 2, "tags": {"building": "house", "building:levels": "2", "roof:shape": "gabled", "roof:height": "3"},
            "geometry": _sq(22.5770, 88.4310)}]
    w = _world(els)
    assert _e(w, "osm-way-1")["roof"]["eave_height_m"] == 6.0 and _e(w, "osm-way-1")["bounds_max"][1] == 9.0
    assert _e(w, "osm-way-2")["roof"]["eave_height_m"] == 6.0 and _e(w, "osm-way-2")["bounds_max"][1] == 9.0


def _sq(la, lo, d=0.0001):
    return [{"lat": la, "lon": lo}, {"lat": la, "lon": lo + d}, {"lat": la + d, "lon": lo + d}, {"lat": la + d, "lon": lo},
            {"lat": la, "lon": lo}]


def test_centre_only_building_stays_flat():
    w = _world([{"type": "way", "id": 5, "center": {"lat": LAT, "lon": LON}, "tags": {"building": "house"}}])
    r = _e(w, "osm-way-5")["roof"]
    assert r["shape"] == "flat" and r["height_m"] == 0.0 and "centre point" in r["basis"]


def test_areas_have_polygon_and_estimated_state():
    w = _world()
    for eid, kind in (("osm-way-7010", "garden"), ("osm-way-7011", "flowerbed"), ("osm-way-7014", "woodland"),
                      ("osm-way-7015", "water")):
        p = _e(w, eid)
        assert p["kind"] == kind and p["imported"] is True and len(p["footprint_polygon_xyz_m"]) == 4
        assert p["dimensions"]["area"]["state"] == "ESTIMATED"
        assert 0 < w.entities[eid].confidence <= osm.MAX_CONFIDENCE
    assert w.entities["osm-way-7015"].type.value == "water"
    assert w.entities["osm-way-7010"].type.value == "vegetation"


def test_unclosed_area_and_relation_rings():
    open_way = {"type": "way", "id": 9, "tags": {"leisure": "garden"}, "geometry": _sq(22.577, 88.43)[:-1]}
    ring = _sq(22.577, 88.431)
    rel = {"type": "relation", "id": 10, "tags": {"type": "multipolygon", "landuse": "grass"},
           "members": [{"role": "outer", "geometry": ring}, {"role": "inner", "geometry": _sq(22.57702, 88.43102, 0.00003)}]}
    w = _world([open_way, rel])
    assert "osm-way-9" not in w.entities
    p = _e(w, "osm-relation-10")
    assert p["kind"] == "lawn" and p["holes_ignored"] == 1


def test_hedge_and_barrier_kind():
    w = _world()
    h = _e(w, "osm-way-7012")
    assert h["kind"] == "hedge" and h["barrier_kind"] == "hedge"
    assert h["dimensions"]["height"] == {"value": 1.5, "unit": "m", "state": "ESTIMATED", "basis": "default_hedge_height"}
    assert h["dimensions"]["width"]["value"] == 0.8 and h["dimensions"]["width"]["state"] == "ESTIMATED"
    assert _e(w, "osm-way-7016")["barrier_kind"] == "fence"


def test_tagged_trees_and_shrub():
    w = _world()
    broad, pine, bare, shrub = (_e(w, f"osm-node-{i}") for i in (7021, 7022, 7023, 7024))
    assert broad["canopy"]["style"] == "round" and broad["canopy"]["radius_m"] == 4.0
    assert broad["dimensions"]["footprint_side"]["state"] == "INFERRED"
    assert pine["canopy"]["style"] == "cone" and "genus" in pine["canopy"]["basis"]
    assert bare["canopy"]["style"] == "round" and "default" in bare["canopy"]["basis"]
    assert bare["dimensions"]["height"]["state"] == "ESTIMATED"
    assert shrub["kind"] == "shrub" and shrub["dimensions"]["height"]["value"] == 1.0


def test_generated_trees_flagged_and_never_imported():
    w = _world()
    gen = _generated(w)
    assert gen
    for e in gen:
        p = e.custom_properties
        assert p["label"] == "Generated (not captured)" and p["asserted"] is False and p["state"] == "ESTIMATED"
        assert p["imported"] is False and p.get("source") != "OpenStreetMap"
        assert e.confidence <= 0.3 and p["dimensions"]["height"]["state"] == "ESTIMATED"
        assert p["derived_from"] in w.entities
    imported_trees = [e for e in w.entities.values() if e.custom_properties["kind"] == "tree" and e.custom_properties.get("imported")]
    assert len(imported_trees) == 3  # the tagged ones only
    row = [e for e in gen if e.custom_properties["derived_from_kind"] == "tree_row"]
    assert len(row) == 12  # 7 m spacing along ~82 m: trees at 0, 7 ... 77 m
    assert all(e.id.startswith("gen-tree-way-7013-") for e in row)


def test_generation_is_deterministic():
    a, b = _world(), _world()
    ga, gb = sorted(e.id for e in _generated(a)), sorted(e.id for e in _generated(b))
    assert ga == gb and ga
    pa = {e.id: e.custom_properties["position_xyz_m"] for e in _generated(a)}
    assert pa == {e.id: e.custom_properties["position_xyz_m"] for e in _generated(b)}
    wood = [e for e in _generated(a) if e.custom_properties["derived_from_kind"] == "woodland"]
    assert 20 < len(wood) <= osm.MAX_FILL_PER_POLYGON
    ring = _e(a, "osm-way-7014")["footprint_polygon_xyz_m"]
    xs, zs = [p[0] for p in ring], [p[2] for p in ring]
    for e in wood:
        x, _, z = e.custom_properties["position_xyz_m"]
        assert min(xs) <= x <= max(xs) and min(zs) <= z <= max(zs)


def test_caps_per_polygon_per_world_and_per_tree_row():
    big = {"type": "way", "id": 100, "tags": {"natural": "wood"}, "geometry": _sq(22.57, 88.43, 0.003)}
    w = _world([big])
    assert sum(1 for e in _generated(w)) == osm.MAX_FILL_PER_POLYGON
    many = [{"type": "way", "id": 200 + i, "tags": {"natural": "wood"}, "geometry": _sq(22.57 + i * 0.004, 88.43, 0.003)}
            for i in range(10)]
    assert len(_generated(_world(many))) == osm.MAX_FILL_PER_WORLD
    row = {"type": "way", "id": 300, "tags": {"natural": "tree_row"},
           "geometry": [{"lat": 22.57, "lon": 88.43}, {"lat": 22.57, "lon": 88.44}]}
    assert len(_generated(_world([row]))) == osm.MAX_TREE_ROW_TREES


def test_scrub_and_meadow_are_not_filled_with_trees():
    w = _world([{"type": "way", "id": 1, "tags": {"natural": "scrub"}, "geometry": _sq(22.57, 88.43, 0.001)}])
    assert osm.counts_by_kind(w) == {"meadow": 1}


def test_world_roundtrips():
    from world_ir.world_v1 import WorldIR
    w = _world()
    again = WorldIR.from_dict(json.loads(json.dumps(w.to_dict())))
    assert len(again.entities) == len(w.entities)
