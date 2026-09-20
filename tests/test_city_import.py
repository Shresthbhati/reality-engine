"""City evidence ingestion boundary (REAL_RECONSTRUCTION_PERCEPTION_
CITY_READY increment B): OSM XML and GeoJSON become canonical
EvidenceAssets in the EXISTING evidence model — no separate city-data
schema, no parallel provenance path.

Real fixture: datasets/city_osm/south_building_campus.osm — a REAL
OpenStreetMap extract (ODbL) around UNC Chapel Hill's South Building,
the same real structure the datasets/south_building photographs cover.
Attributes + sha256 recorded in datasets/city_osm/MANIFEST.json.

Rules under test:
  - One EvidenceAsset per OSM way (kind=OTHER): measured geometry
    (resolved node lat/lon), verbatim tags, OSM element id — the
    asset's sensor_metadata answers "where did this come from".
  - Provenance is OBSERVED (externally recorded data, not our
    inference) with confidence 1.0 for the verbatim facts.
  - Deterministic: document order, stable ids, same input -> same
    package bytes.
  - Dedup rides the builder's content-hash contract: re-importing the
    same file is duplicates_skipped, never duplicated assets.
  - Malformed XML raises CorruptEvidenceError (same honest gate as
    photos/LAS) — never a silent empty import.
  - GeoJSON features import by the same rules (geometry verbatim,
    properties recorded).
  - Unresolvable node references are recorded per way as
    unresolved_node_refs (a measured fact), never silently dropped
    vertices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evidence.packages import DeterministicPackageBuilder, EvidenceSource  # noqa: E402

OSM_FIXTURE = REPO / "datasets" / "city_osm" / "south_building_campus.osm"


def _source_of(path: Path) -> EvidenceSource:
    return EvidenceSource(
        source_id=f"src-{path.name}",
        platform="external_map_data",
        device="OpenStreetMap/Overpass",
        notes="REAL external map evidence (ODbL) - city ingestion boundary",
    )


def _import_osm(builder, path=OSM_FIXTURE):
    from evidence.city_import import import_osm_xml

    return import_osm_xml(builder, _source_of(path), str(path))


@pytest.mark.skipif(not OSM_FIXTURE.is_file(), reason="real OSM fixture missing")
class TestOsmImport:
    def test_real_fixture_imports_ways_with_geometry_and_tags(self):
        builder = DeterministicPackageBuilder(seed="city-test")
        report = _import_osm(builder)
        package = builder.build()
        assets = list(package.all_assets())
        assert report.imported_features == 137
        assert len(assets) >= 137  # ways + at least one collection asset

        way_assets = [a for a in assets if a.sensor_metadata.get("osm", {}).get("element_type") == "way"]
        assert len(way_assets) == 137
        sample = next(
            a for a in way_assets
            if "building" in a.sensor_metadata["osm"]["tags"]
        )
        osm_meta = sample.sensor_metadata["osm"]
        assert osm_meta["osm_id"]  # real OSM element id recorded
        geom = osm_meta["geometry"]
        assert len(geom) >= 4
        lon = [pt[0] for pt in geom]
        lat = [pt[1] for pt in geom]
        # Measured coordinates sit in a plausible band around the campus
        # (ways may extend slightly past the query bbox; Overpass returns
        # any way touching the area). Padded by ~0.002 deg (~200 m).
        PAD = 0.002
        assert -79.0550 - PAD <= min(lon) and max(lon) <= -79.0520 + PAD
        assert 35.9110 - PAD <= min(lat) and max(lat) <= 35.9130 + PAD

    def test_provenance_is_observed_with_verbatim_facts(self):
        builder = DeterministicPackageBuilder(seed="city-test")
        _import_osm(builder)
        package = builder.build()
        for asset in package.all_assets():
            assert asset.provenance.value == "OBSERVED"
            assert asset.uncertainty.confidence == 1.0
            osm_meta = asset.sensor_metadata.get("osm", {})
            assert osm_meta.get("source_note") == "OpenStreetMap (ODbL) (c) OpenStreetMap contributors"

    def test_reimport_is_deduped_by_content_hash(self):
        builder = DeterministicPackageBuilder(seed="city-test")
        r1 = _import_osm(builder)
        assert r1.imported_features == 137
        n_assets_before = len(builder.build().all_assets())
        r2 = _import_osm(builder)  # same extract, same builder: all deduped
        assert r2.imported_features == 0
        # 137 ways + the node-collection asset are all duplicates.
        assert len(r2.duplicates_skipped) == 138
        assert len(builder.build().all_assets()) == n_assets_before

    def test_deterministic_ids_across_runs(self):
        b1 = DeterministicPackageBuilder(seed="city-test")
        _import_osm(b1)
        b2 = DeterministicPackageBuilder(seed="city-test")
        _import_osm(b2)
        ids1 = sorted(a.id for a in b1.build().all_assets())
        ids2 = sorted(a.id for a in b2.build().all_assets())
        assert ids1 == ids2

    def test_malformed_xml_raises_corrupt_evidence_error(self, tmp_path):
        from evidence.packages import CorruptEvidenceError

        bad = tmp_path / "broken.osm"
        bad.write_text("<osm version='0.6'><node id='1' lat='1.0'")
        builder = DeterministicPackageBuilder(seed="city-test")
        with pytest.raises(CorruptEvidenceError):
            _import_osm(builder, bad)

    def test_unresolved_node_refs_recorded_not_silently_dropped(self, tmp_path):
        builder = DeterministicPackageBuilder(seed="city-test")
        incomplete = tmp_path / "incomplete.osm"
        incomplete.write_text(
            "<?xml version='1.0' encoding='UTF-8'?>\n"
            "<osm version='0.6'>\n"
            "  <way id='99' version='1'>\n"
            "    <nd ref='111'/><nd ref='222'/><nd ref='111'/>\n"
            "    <tag k='building' v='yes'/>\n"
            "  </way>\n"
            "</osm>\n"
        )
        report = _import_osm(builder, incomplete)
        assert report.imported_features == 1
        package = builder.build()
        way_assets = [
            a for a in package.all_assets()
            if a.sensor_metadata.get("osm", {}).get("element_type") == "way"
        ]
        assert len(way_assets) == 1
        osm_meta = way_assets[0].sensor_metadata["osm"]
        assert osm_meta["unresolved_node_refs"] == ["111", "222"]
        assert osm_meta["geometry"] == []  # nothing fabricated from missing nodes

    def test_report_is_json_serializable(self):
        builder = DeterministicPackageBuilder(seed="city-test")
        report = _import_osm(builder)
        blob = json.dumps(report.to_dict())
        assert "imported_features" in blob


class TestGeoJsonImport:
    def test_feature_collection_imports_geometry_and_properties(self, tmp_path):
        from evidence.city_import import import_geojson

        gj = tmp_path / "parcels.geojson"
        gj.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "parcel-1",
                    "geometry": {"type": "Polygon", "coordinates": [
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
                    ]},
                    "properties": {"name": "lot A", "height_m": 12.5},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
                    "properties": {"class": "road"},
                },
            ],
        }))
        builder = DeterministicPackageBuilder(seed="city-test")
        report = import_geojson(builder, _source_of(gj), str(gj))
        assert report.imported_features == 2
        package = builder.build()
        feats = [
            a for a in package.all_assets()
            if a.sensor_metadata.get("gis", {}).get("feature")
        ]
        assert len(feats) == 2
        named = next(
            f for f in feats
            if f.sensor_metadata["gis"]["properties"].get("name") == "lot A"
        )
        assert named.sensor_metadata["gis"]["geometry"]["type"] == "Polygon"
        assert named.provenance.value == "OBSERVED"

    def test_malformed_geojson_raises(self, tmp_path):
        from evidence.city_import import import_geojson
        from evidence.packages import CorruptEvidenceError

        bad = tmp_path / "bad.geojson"
        bad.write_text('{"type": "FeatureCollection", "features": [')
        builder = DeterministicPackageBuilder(seed="city-test")
        with pytest.raises(CorruptEvidenceError):
            import_geojson(builder, _source_of(bad), str(bad))
