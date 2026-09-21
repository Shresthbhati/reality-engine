"""OSM evidence -> CityFeature -> WorldIR bridge (P14-01 remainder).

Uses a small synthetic OSM XML doc (not the large real fixture used by
test_city_import.py) so this module's mapping/geometry rules are
verified deterministically without depending on a gitignored dataset.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from engine.compiler.city_compiler import compile_city_world  # noqa: E402
from engine.compiler.osm_features import (  # noqa: E402
    classify_osm_tags,
    osm_record_to_city_feature,
    osm_records_to_city_features,
    OsmCompileReport,
)
from evidence.city_import import import_osm_xml  # noqa: E402
from evidence.packages import DeterministicPackageBuilder, EvidenceSource  # noqa: E402
from world_ir.schema_v1 import EntityType  # noqa: E402

_OSM_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="35.9000" lon="-79.0500"/>
  <node id="2" lat="35.9001" lon="-79.0500"/>
  <node id="3" lat="35.9001" lon="-79.0499"/>
  <node id="4" lat="35.9000" lon="-79.0499"/>
  <node id="5" lat="35.9100" lon="-79.0600"/>
  <node id="6" lat="35.9101" lon="-79.0600"/>
  <node id="7" lat="35.9101" lon="-79.0599"/>
  <way id="100">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="yes"/>
    <tag k="height" v="12.5"/>
    <tag k="name" v="South Building"/>
  </way>
  <way id="101">
    <nd ref="5"/><nd ref="6"/><nd ref="7"/>
    <tag k="highway" v="residential"/>
  </way>
  <way id="102">
    <nd ref="1"/><nd ref="2"/>
    <tag k="building" v="yes"/>
  </way>
  <way id="103">
    <nd ref="1"/><nd ref="2"/><nd ref="99"/>
    <tag k="leisure" v="picnic_table"/>
  </way>
</osm>
"""


def _import_synthetic_osm():
    builder = DeterministicPackageBuilder()
    source = EvidenceSource(source_id="src-test", platform="external_map_data",
                             device="OpenStreetMap/Overpass")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".osm", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(_OSM_DOC)
        path = fh.name
    import_osm_xml(builder, source, path)
    package = builder.build("pkg-osm-test")
    return package


def _osm_records(package):
    return [
        asset.sensor_metadata for asset in package.all_assets()
        if "osm" in asset.sensor_metadata
        and asset.sensor_metadata["osm"].get("element_type") == "way"
    ]


class TestClassifyOsmTags:
    def test_building_tag_maps_to_building(self):
        assert classify_osm_tags({"building": "yes"}) == "building"

    def test_highway_tag_maps_to_road(self):
        assert classify_osm_tags({"highway": "residential"}) == "road"

    def test_natural_water_maps_to_water(self):
        assert classify_osm_tags({"natural": "water"}) == "water"

    def test_landuse_forest_maps_to_vegetation(self):
        assert classify_osm_tags({"landuse": "forest"}) == "vegetation"

    def test_unmapped_tags_return_none(self):
        assert classify_osm_tags({"leisure": "picnic_table"}) is None

    def test_first_matching_rule_wins(self):
        # building takes priority over highway when (implausibly) both present
        assert classify_osm_tags({"building": "yes", "highway": "service"}) == "building"


class TestOsmRecordToCityFeature:
    def test_building_way_with_height_and_name(self):
        record = {"osm": {"element_type": "way", "osm_id": "100",
                           "tags": {"building": "yes", "height": "12.5",
                                    "name": "South Building"},
                           "geometry": [[-79.05, 35.90], [-79.05, 35.9001],
                                        [-79.0499, 35.9001], [-79.0499, 35.90]]}}
        report = OsmCompileReport()
        feature = osm_record_to_city_feature(record, report)
        assert feature is not None
        assert feature.kind == "building"
        assert feature.name == "South Building"
        assert feature.height == 12.5
        assert feature.evidence_note == "OSM way 100"
        assert report.compiled == 0  # caller increments in batch form only

    def test_building_levels_height_estimate(self):
        record = {"osm": {"element_type": "way", "osm_id": "200",
                           "tags": {"building": "yes", "building:levels": "4"},
                           "geometry": [[0, 0], [0, 1], [1, 1]]}}
        feature = osm_record_to_city_feature(record, OsmCompileReport())
        assert feature.height == 12.0  # 4 * 3.0 m/level

    def test_untagged_height_stays_flat(self):
        record = {"osm": {"element_type": "way", "osm_id": "300",
                           "tags": {"highway": "residential"},
                           "geometry": [[0, 0], [0, 1], [1, 1]]}}
        feature = osm_record_to_city_feature(record, OsmCompileReport())
        assert feature.height is None

    def test_unmapped_kind_returns_none_and_is_recorded(self):
        record = {"osm": {"element_type": "way", "osm_id": "400",
                           "tags": {"leisure": "picnic_table"},
                           "geometry": [[0, 0], [0, 1], [1, 1]]}}
        report = OsmCompileReport()
        feature = osm_record_to_city_feature(record, report)
        assert feature is None
        assert report.unmapped_kind == ["400"]

    def test_too_few_points_returns_none_and_is_recorded(self):
        record = {"osm": {"element_type": "way", "osm_id": "500",
                           "tags": {"building": "yes"},
                           "geometry": [[0, 0], [0, 1]]}}
        report = OsmCompileReport()
        feature = osm_record_to_city_feature(record, report)
        assert feature is None
        assert report.too_few_points == ["500"]

    def test_non_way_record_returns_none_and_is_recorded(self):
        record = {"osm": {"element_type": "node_collection", "osm_id": "nodes-7"}}
        report = OsmCompileReport()
        feature = osm_record_to_city_feature(record, report)
        assert feature is None
        assert report.not_osm_way == 1

    def test_malformed_height_ignored_not_raised(self):
        record = {"osm": {"element_type": "way", "osm_id": "600",
                           "tags": {"building": "yes", "height": "not-a-number"},
                           "geometry": [[0, 0], [0, 1], [1, 1]]}}
        feature = osm_record_to_city_feature(record, OsmCompileReport())
        assert feature.height is None


class TestBatchAndPipeline:
    def test_batch_report_counts_match_synthetic_doc(self):
        package = _import_synthetic_osm()
        records = _osm_records(package)
        features, report = osm_records_to_city_features(records)
        # way 100 -> building (compiled), 101 -> road (compiled),
        # 102 -> too few points, 103 -> unmapped kind (leisure)
        assert report.compiled == 2
        assert len(features) == 2
        assert report.too_few_points == ["102"]
        assert report.unmapped_kind == ["103"]

    def test_features_compile_into_worldir_with_correct_entity_types(self):
        package = _import_synthetic_osm()
        records = _osm_records(package)
        features, _ = osm_records_to_city_features(records)
        world = compile_city_world(features, name="synthetic")
        kinds = {e.type for e in world.entities.values()}
        assert EntityType.BUILDING in kinds
        assert EntityType.ROAD in kinds
        assert len(world.entities) == 2

    def test_full_pipeline_is_deterministic(self):
        package = _import_synthetic_osm()
        records = _osm_records(package)
        f1, r1 = osm_records_to_city_features(records)
        f2, r2 = osm_records_to_city_features(records)
        assert [f.name for f in f1] == [f.name for f in f2]
        assert r1.to_dict() == r2.to_dict()
