"""GeoJSON evidence -> CityFeature -> WorldIR bridge (P14-01: the
--geojson half of reality city-compile). Mirrors
tests/test_osm_city_features.py's structure over the GeoJSON path."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from engine.compiler.city_compiler import compile_city_world  # noqa: E402
from engine.compiler.geojson_features import (  # noqa: E402
    GeoJsonCompileReport,
    geojson_record_to_city_feature,
    geojson_records_to_city_features,
)
from evidence.city_import import import_geojson  # noqa: E402
from evidence.packages import DeterministicPackageBuilder, EvidenceSource  # noqa: E402
from world_ir.schema_v1 import EntityType  # noqa: E402

_GEOJSON_DOC = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature", "id": "bldg-1",
            "geometry": {"type": "Polygon", "coordinates": [[
                [-79.05, 35.90], [-79.05, 35.9001],
                [-79.0499, 35.9001], [-79.0499, 35.90], [-79.05, 35.90],
            ]]},
            "properties": {"building": "yes", "height": "9.0", "name": "Lab Building"},
        },
        {
            "type": "Feature", "id": "road-1",
            "geometry": {"type": "LineString", "coordinates": [
                [-79.06, 35.91], [-79.0599, 35.9101],
            ]},
            "properties": {"highway": "residential"},
        },
        {
            "type": "Feature", "id": "pt-1",
            "geometry": {"type": "Point", "coordinates": [-79.05, 35.90]},
            "properties": {"amenity": "bench"},
        },
        {
            "type": "Feature", "id": "unknown-1",
            "geometry": {"type": "Polygon", "coordinates": [[
                [0, 0], [0, 1], [1, 1], [1, 0], [0, 0],
            ]]},
            "properties": {"leisure": "picnic_table"},
        },
    ],
}


def _import_synthetic_geojson():
    builder = DeterministicPackageBuilder()
    source = EvidenceSource(source_id="src-test", platform="external_map_data",
                             device="GIS/GeoJSON")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".geojson", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(_GEOJSON_DOC, fh)
        path = fh.name
    import_geojson(builder, source, path)
    return builder.build("pkg-geojson-test")


def _gis_records(package):
    return [
        asset.sensor_metadata for asset in package.all_assets()
        if "gis" in asset.sensor_metadata
    ]


class TestGeojsonRecordToCityFeature:
    def test_polygon_building_with_height_and_name(self):
        record = {"gis": {"feature": "bldg-1",
                           "geometry": {"type": "Polygon", "coordinates": [[
                               [0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
                           "properties": {"building": "yes", "height": "9.0",
                                          "name": "Lab Building"}}}
        report = GeoJsonCompileReport()
        feature = geojson_record_to_city_feature(record, report)
        assert feature is not None
        assert feature.kind == "building"
        assert feature.name == "Lab Building"
        assert feature.height == 9.0
        assert feature.evidence_note == "GeoJSON feature bldg-1"

    def test_levels_height_estimate(self):
        record = {"gis": {"feature": "b2",
                           "geometry": {"type": "Polygon", "coordinates": [[
                               [0, 0], [0, 1], [1, 1], [0, 0]]]},
                           "properties": {"building": "yes", "levels": "3"}}}
        feature = geojson_record_to_city_feature(record, GeoJsonCompileReport())
        assert feature.height == 9.0  # 3 * 3.0 m/level

    def test_point_geometry_is_unsupported_and_recorded(self):
        record = {"gis": {"feature": "pt-1",
                           "geometry": {"type": "Point", "coordinates": [0, 0]},
                           "properties": {"amenity": "bench"}}}
        report = GeoJsonCompileReport()
        feature = geojson_record_to_city_feature(record, report)
        assert feature is None
        assert report.unsupported_geometry == ["pt-1"]

    def test_linestring_geometry_is_unsupported_and_recorded(self):
        record = {"gis": {"feature": "road-1",
                           "geometry": {"type": "LineString",
                                        "coordinates": [[0, 0], [1, 1]]},
                           "properties": {"highway": "residential"}}}
        report = GeoJsonCompileReport()
        feature = geojson_record_to_city_feature(record, report)
        assert feature is None
        assert report.unsupported_geometry == ["road-1"]

    def test_unmapped_properties_return_none_and_recorded(self):
        record = {"gis": {"feature": "u1",
                           "geometry": {"type": "Polygon", "coordinates": [[
                               [0, 0], [0, 1], [1, 1], [0, 0]]]},
                           "properties": {"leisure": "picnic_table"}}}
        report = GeoJsonCompileReport()
        feature = geojson_record_to_city_feature(record, report)
        assert feature is None
        assert report.unmapped_kind == ["u1"]

    def test_non_gis_record_returns_none_and_recorded(self):
        record = {"osm": {"element_type": "way"}}
        report = GeoJsonCompileReport()
        feature = geojson_record_to_city_feature(record, report)
        assert feature is None
        assert report.not_gis_feature == 1


class TestGeojsonBatchAndPipeline:
    def test_batch_report_counts_match_synthetic_doc(self):
        package = _import_synthetic_geojson()
        records = _gis_records(package)
        features, report = geojson_records_to_city_features(records)
        assert report.compiled == 1  # only bldg-1 is a mappable Polygon
        assert report.unsupported_geometry == ["road-1", "pt-1"]
        assert report.unmapped_kind == ["unknown-1"]

    def test_features_compile_into_worldir(self):
        package = _import_synthetic_geojson()
        records = _gis_records(package)
        features, _ = geojson_records_to_city_features(records)
        world = compile_city_world(features, name="synthetic-gis")
        assert len(world.entities) == 1
        entity = next(iter(world.entities.values()))
        assert entity.type == EntityType.BUILDING
        assert entity.name == "Lab Building"

    def test_full_pipeline_is_deterministic(self):
        package = _import_synthetic_geojson()
        records = _gis_records(package)
        f1, r1 = geojson_records_to_city_features(records)
        f2, r2 = geojson_records_to_city_features(records)
        assert [f.name for f in f1] == [f.name for f in f2]
        assert r1.to_dict() == r2.to_dict()
