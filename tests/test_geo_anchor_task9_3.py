"""Unit tests for Task 9.3: Geo anchor and coordinate transforms (Blueprint 16 & 30 Task 9.3)."""

import math
from pathlib import Path
import pytest

from world_ir.geo import GeoAnchor
from world_ir.world_v1 import WorldIR
from world_ir.schema_v1 import Geometry, GeometryType, Vector3
from evidence.geo_anchor import (
    extract_geo_anchor_from_evidence,
    attach_geo_anchor_to_world,
)


def test_geo_anchor_validation_and_roundtrip():
    # Valid anchor in San Francisco
    anchor = GeoAnchor(
        latitude=37.774929,
        longitude=-122.419416,
        altitude_m=15.5,
        heading_deg=45.0,
        crs="EPSG:4326",
        accuracy_m=3.2,
        source="exif_gps",
    )

    d = anchor.to_dict()
    assert d["latitude"] == pytest.approx(37.774929, abs=1e-6)
    assert d["longitude"] == pytest.approx(-122.419416, abs=1e-6)
    assert d["altitude_m"] == 15.5
    assert d["heading_deg"] == 45.0
    assert d["accuracy_m"] == 3.2

    restored = GeoAnchor.from_dict(d)
    assert restored.latitude == anchor.latitude
    assert restored.longitude == anchor.longitude
    assert restored.altitude_m == anchor.altitude_m
    assert restored.heading_deg == anchor.heading_deg
    assert restored.accuracy_m == anchor.accuracy_m

    # Invalid latitude/longitude bounds
    with pytest.raises(ValueError):
        GeoAnchor(latitude=95.0, longitude=0.0)
    with pytest.raises(ValueError):
        GeoAnchor(latitude=0.0, longitude=185.0)


def test_pyproj_local_geographic_transform():
    # Anchor at Greenwich
    anchor = GeoAnchor(
        latitude=51.4769,
        longitude=0.0,
        altitude_m=20.0,
        heading_deg=0.0,
    )

    # (0, 0, 0) local must map to exact anchor lat/lon/alt
    lat, lon, alt = anchor.local_to_geographic(0.0, 0.0, 0.0)
    assert lat == pytest.approx(51.4769, abs=1e-5)
    assert lon == pytest.approx(0.0, abs=1e-5)
    assert alt == pytest.approx(20.0, abs=1e-3)

    # Inverse of center
    lx, ly, lz = anchor.geographic_to_local(lat, lon, alt)
    assert lx == pytest.approx(0.0, abs=1e-3)
    assert ly == pytest.approx(0.0, abs=1e-3)
    assert lz == pytest.approx(0.0, abs=1e-3)

    # 100m East, 200m North
    lat_off, lon_off, alt_off = anchor.local_to_geographic(100.0, 200.0, 5.0)
    rec_x, rec_y, rec_z = anchor.geographic_to_local(lat_off, lon_off, alt_off)
    assert rec_x == pytest.approx(100.0, abs=1e-3)
    assert rec_y == pytest.approx(200.0, abs=1e-3)
    assert rec_z == pytest.approx(5.0, abs=1e-3)


def test_heading_rotation():
    # Heading = 90 degrees clockwise (local +Y is East, local +X is South)
    anchor = GeoAnchor(
        latitude=40.7128,
        longitude=-74.0060,
        altitude_m=10.0,
        heading_deg=90.0,
    )

    # Moving 50m forward in heading direction (+Y) should add ~50m East
    lat_fwd, lon_fwd, _ = anchor.local_to_geographic(0.0, 50.0, 0.0)
    lx, ly, _ = anchor.geographic_to_local(lat_fwd, lon_fwd, 10.0)
    assert lx == pytest.approx(0.0, abs=1e-3)
    assert ly == pytest.approx(50.0, abs=1e-3)

    # Verify East offset is positive
    assert lon_fwd > anchor.longitude


def test_bounding_box_to_geojson_polygon():
    anchor = GeoAnchor(latitude=37.7749, longitude=-122.4194)
    geojson = anchor.bounding_box_to_geojson_polygon((-10.0, -20.0, 0.0), (10.0, 20.0, 3.0))

    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "Polygon"
    coords = geojson["geometry"]["coordinates"][0]
    assert len(coords) == 5  # 4 corners + closed loop
    assert coords[0] == coords[-1]  # Closed polygon


def test_worldir_geo_anchor_integration_and_local_invariance():
    anchor = GeoAnchor(
        latitude=34.0522,
        longitude=-118.2437,
        altitude_m=71.0,
        heading_deg=0.0,
        accuracy_m=4.5,
    )

    # World with local metric geometry
    g1 = Geometry(
        id="geom-room-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(6.0, 8.0, 2.8),
    )
    world = WorldIR(id="w-geo-test", geometries={"geom-room-1": g1}, geo_anchor=anchor)

    # Invariant: local geometries remain in metric meters!
    assert world.geometries["geom-room-1"].bounds_max.x == 6.0
    assert world.geometries["geom-room-1"].bounds_max.y == 8.0
    assert world.geometries["geom-room-1"].bounds_max.z == 2.8

    # Serialization roundtrip preserves geo_anchor
    d = world.to_dict()
    assert "geo_anchor" in d
    assert d["geo_anchor"]["latitude"] == pytest.approx(34.0522, abs=1e-5)

    restored = WorldIR.from_dict(d)
    assert restored.geo_anchor is not None
    assert restored.geo_anchor.latitude == pytest.approx(34.0522, abs=1e-5)
    assert restored.geometries["geom-room-1"].bounds_max.x == 6.0

    # Bounds and GeoJSON computation
    bmin, bmax = restored.compute_bounds()
    assert bmin == (0.0, 0.0, 0.0)
    assert bmax == (6.0, 8.0, 2.8)

    poly = restored.geographic_bounds_geojson()
    assert poly is not None
    assert poly["geometry"]["type"] == "Polygon"


def test_extract_geo_anchor_from_evidence_items():
    evidence_items = [
        {"latitude_deg": 37.77490, "longitude_deg": -122.41940, "altitude_m": 10.0},
        {"latitude_deg": 37.77500, "longitude_deg": -122.41950, "altitude_m": 12.0},
        {"latitude_deg": 37.77510, "longitude_deg": -122.41930, "altitude_m": 11.0},
    ]

    anchor = extract_geo_anchor_from_evidence(evidence_items)
    assert anchor is not None
    assert anchor.latitude == pytest.approx(37.77500, abs=1e-4)
    assert anchor.longitude == pytest.approx(-122.41940, abs=1e-4)
    assert anchor.altitude_m == pytest.approx(11.0, abs=0.5)
    assert anchor.accuracy_m is not None

    # Empty or GPS-free evidence returns None
    assert extract_geo_anchor_from_evidence([]) is None
    assert extract_geo_anchor_from_evidence([{"width": 1920, "height": 1080}]) is None


def test_attach_geo_anchor_to_world():
    world = WorldIR(id="w-attach-test")
    assert world.geo_anchor is None

    anchor = GeoAnchor(latitude=52.5200, longitude=13.4050)
    attach_geo_anchor_to_world(world, anchor)
    assert world.geo_anchor is not None
    assert world.geo_anchor.latitude == 52.5200
