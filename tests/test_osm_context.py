"""geo.osm_context: Overpass query/fetch (mock transport, no network) and OSM -> WorldIR conversion."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from geo import osm_context as osm

FX = Path(__file__).parent / "fixtures" / "osm"
LAT, LON = 22.5757, 88.4312


def _load(name):
    return json.loads((FX / name).read_text(encoding="utf-8"))


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_query_contains_radius_and_classes():
    q = osm.build_query(LAT, LON, 300)
    assert "around:300,22.575700,88.431200" in q and "[out:json]" in q
    for needle in ("building", "highway", "barrier", "subway_entrance", "street_lamp", "natural", "fire_hydrant", "bridge"):
        assert needle in q


def test_fetch_sends_user_agent_and_caches(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=_load("synthetic_street_furniture.json"))

    data, src = osm.fetch_overpass(LAT, LON, 300, client=_client(handler), cache_dir=tmp_path)
    assert src == "overpass" and len(calls) == 1
    assert calls[0].headers["user-agent"] == "RealityEngine/0.1 (research prototype)"
    assert calls[0].headers["accept"] == "application/json"
    data2, src2 = osm.fetch_overpass(LAT, LON, 300, client=_client(lambda r: pytest.fail("network hit")), cache_dir=tmp_path)
    assert src2 == "cache" and data2 == data
    assert len(calls) == 1


@pytest.mark.parametrize("resp,exc", [
    (httpx.Response(429), osm.OverpassRateLimited),
    (httpx.Response(504), osm.OverpassUnavailable),
    (httpx.Response(200, text="<html>"), osm.OverpassUnavailable),
    (httpx.Response(200, json={"remark": "runtime error: out of memory", "elements": []}), osm.OverpassUnavailable),
])
def test_fetch_errors_are_typed_and_not_cached(tmp_path, resp, exc):
    with pytest.raises(exc):
        osm.fetch_overpass(LAT, LON, 300, client=_client(lambda r: resp), cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.json"))


def test_fetch_network_error_is_unavailable():
    def boom(req):
        raise httpx.ConnectError("down")

    with pytest.raises(osm.OverpassUnavailable):
        osm.fetch_overpass(LAT, LON, 300, client=_client(boom))


def _world(fx="synthetic_street_furniture.json"):
    return osm.osm_to_world(_load(fx)["elements"], LAT, LON, "T", radius_m=300, fetched_at="2026-10-31T00:00:00+00:00")


def test_synthetic_every_branch():
    w = _world()
    counts = osm.counts_by_kind(w)
    assert counts == {
        "bench": 1, "bin": 1, "bridge": 1, "building": 3, "fire_hydrant": 1, "fence": 1, "jersey_barrier": 1,
        "lamp_post": 1, "metro_line": 1, "post_box": 1, "road": 1, "sidewalk": 1, "station": 1,
        "station_entrance": 1, "steps": 1, "traffic_sign": 1, "tree": 2,
    }  # park and bicycle_parking are not imported
    assert w.metadata["compiled_from"]["up"] == [0, 1, 0]
    g = w.metadata["geo"]
    assert g["anchor_lat"] == LAT and g["source"] == "OpenStreetMap" and g["licence"] == "ODbL"
    assert g["attribution"] == "(c) OpenStreetMap contributors" and g["fetched_at"]
    for e in w.entities.values():
        p = e.custom_properties
        assert p["source"] == "OpenStreetMap" and p["imported"] is True and p["osm_id"]
        assert 0 < e.confidence < 1


def test_building_height_rules_and_confidence():
    w = _world()
    b = {e.custom_properties["osm_id"]: e for e in w.entities.values() if e.custom_properties["kind"] == "building"}
    both, levels, bare = b[8006], b[8007], b[8008]
    assert both.custom_properties["dimensions"]["height"] == {"value": 20.0, "unit": "m", "state": "INFERRED", "basis": "osm_tag:height"}
    assert both.confidence == 0.7
    assert levels.custom_properties["dimensions"]["height"]["value"] == 9.0
    assert levels.custom_properties["dimensions"]["height"]["state"] == "ESTIMATED"
    assert bare.custom_properties["dimensions"]["height"]["value"] == 6.0
    assert bare.custom_properties["dimensions"]["height"]["state"] == "ESTIMATED"
    assert bare.confidence == 0.4
    assert osm.estimated_count(w) > 0
    assert both.custom_properties["bounds_max"][1] == 20.0  # height on the Y (up) axis


def test_frame_east_x_north_negative_z():
    w = _world()
    lamp = next(e for e in w.entities.values() if e.custom_properties["kind"] == "lamp_post")
    x, y, z = lamp.custom_properties["position_xyz_m"]
    assert x > 0 and z < 0 and y == 0  # lamp is north-east of the anchor
    assert 5 < x < 15 and 1 < -z < 15


def test_bridge_tunnel_and_barrier_details():
    w = _world()
    by = {e.custom_properties["osm_id"]: e.custom_properties for e in w.entities.values()}
    assert by[8002]["dimensions"]["deck_elevation"]["value"] == 5.0
    assert by[8002]["dimensions"]["deck_elevation"]["state"] == "ESTIMATED"
    assert by[8001]["tunnel"] is True and by[8001]["layer"] == -2.0
    assert by[8004]["dimensions"]["width"]["state"] == "INFERRED" and by[8004]["dimensions"]["width"]["value"] == 7.0
    assert by[9004]["barrier_kind"] == "jersey_barrier"
    assert by[9005]["dimensions"]["height"]["value"] == 9.0 and by[9006]["dimensions"]["height"]["state"] == "ESTIMATED"


def test_real_city_centre_buildings_fixture():
    d = _load("city_centre_kolkata_buildings.json")
    w = osm.osm_to_world(d["elements"], LAT, LON, "Real", radius_m=300)
    assert osm.counts_by_kind(w) == {"building": len(d["elements"])}
    assert all(e.custom_properties["footprint_estimated"] for e in w.entities.values())  # fixture is centre-only
    assert all(e.confidence <= 0.7 for e in w.entities.values())
    # all centres are within the ~300 m query radius
    for e in w.entities.values():
        bx = e.custom_properties["bounds_min"]
        assert abs(bx[0]) < 400 and abs(bx[2]) < 400


def test_world_roundtrips_through_to_dict():
    from world_ir.world_v1 import WorldIR
    w = _world()
    again = WorldIR.from_dict(json.loads(json.dumps(w.to_dict())))
    assert len(again.entities) == len(w.entities)


def test_map_world_declares_metric_scale_with_estimated_dimensions():
    from world_ir.measure import effective_state, scale_state_of
    w = _world()
    assert scale_state_of(w.metadata) == "metric"
    # sizes come from tags and defaults, so no dimension of a map world can be called MEASURED
    assert effective_state("MEASURED", "metric", w.metadata["scale"]["dimension_state"]) == "ESTIMATED"
