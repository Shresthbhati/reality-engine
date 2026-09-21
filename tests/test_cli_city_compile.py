"""`reality city-compile` CLI: OSM/GeoJSON files -> WorldIR, no photo
capture required. Exercises the full apps/cli/main.py wiring over the
same evidence.city_import / engine.compiler.osm_features /
engine.compiler.city_compiler stack proven at the library level in
tests/test_osm_city_features.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from apps.cli.main import main  # noqa: E402
from world_ir.world_v1 import WorldIR  # noqa: E402

_OSM_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="35.9000" lon="-79.0500"/>
  <node id="2" lat="35.9001" lon="-79.0500"/>
  <node id="3" lat="35.9001" lon="-79.0499"/>
  <node id="4" lat="35.9000" lon="-79.0499"/>
  <way id="100">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="yes"/>
    <tag k="height" v="12.5"/>
    <tag k="name" v="South Building"/>
  </way>
  <way id="101">
    <nd ref="1"/><nd ref="2"/>
    <tag k="building" v="yes"/>
  </way>
</osm>
"""

_GEOJSON_DOC = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": "f1",
         "geometry": {"type": "Point", "coordinates": [-79.05, 35.90]},
         "properties": {"name": "test point"}},
    ],
}

_GEOJSON_POLYGON_DOC = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": "bldg-1",
         "geometry": {"type": "Polygon", "coordinates": [[
             [-79.05, 35.90], [-79.05, 35.9001],
             [-79.0499, 35.9001], [-79.0499, 35.90], [-79.05, 35.90],
         ]]},
         "properties": {"building": "yes", "name": "GIS Building"}},
    ],
}


def test_city_compile_requires_at_least_one_input(tmp_path: Path, capsys) -> None:
    rc = main(["city-compile", "-o", str(tmp_path / "world.json")])
    assert rc == 1
    assert "at least one --osm or --geojson" in capsys.readouterr().err


def test_city_compile_rejects_malformed_osm(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.osm"
    bad.write_text("<not-xml", encoding="utf-8")
    rc = main(["city-compile", "--osm", str(bad), "-o", str(tmp_path / "world.json")])
    assert rc == 1
    assert "malformed OSM XML" in capsys.readouterr().err


def test_city_compile_osm_produces_worldir_and_report(tmp_path: Path, capsys) -> None:
    osm_path = tmp_path / "campus.osm"
    osm_path.write_text(_OSM_DOC, encoding="utf-8")
    out = tmp_path / "world.json"

    rc = main(["city-compile", "--osm", str(osm_path), "-o", str(out), "--name", "test-city"])
    assert rc == 0

    stdout = capsys.readouterr().out
    assert "compiled 1 entities" in stdout
    assert "0 unmapped" in stdout

    world = WorldIR.from_dict(json.loads(out.read_text(encoding="utf-8")))
    assert len(world.entities) == 1
    entity = next(iter(world.entities.values()))
    assert entity.name == "South Building"

    report_path = Path(str(out) + ".report.json")
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["entities_compiled"] == 1
    assert report["osm_compile"]["compiled"] == 1
    assert report["osm_compile"]["too_few_points"] == ["101"]
    assert len(report["imports"]["osm"]) == 1


def test_city_compile_geojson_point_is_imported_but_not_a_compilable_footprint(tmp_path: Path) -> None:
    """A GeoJSON Point has no polygon footprint to compile -- honest:
    0 entities, not a crash, not fabricated geometry."""
    geojson_path = tmp_path / "extra.geojson"
    geojson_path.write_text(json.dumps(_GEOJSON_DOC), encoding="utf-8")
    out = tmp_path / "world.json"

    rc = main(["city-compile", "--geojson", str(geojson_path), "-o", str(out)])
    assert rc == 0

    world = WorldIR.from_dict(json.loads(out.read_text(encoding="utf-8")))
    assert len(world.entities) == 0

    report = json.loads(Path(str(out) + ".report.json").read_text(encoding="utf-8"))
    assert len(report["imports"]["geojson"]) == 1
    assert report["imports"]["geojson"][0]["imported_features"] == 1
    # untagged point: classification runs before geometry-shape check,
    # so an unmapped property set is recorded as unmapped_kind here
    assert report["geojson_compile"]["unmapped_kind"] == ["f1"]


def test_city_compile_geojson_polygon_compiles_into_worldir(tmp_path: Path, capsys) -> None:
    geojson_path = tmp_path / "buildings.geojson"
    geojson_path.write_text(json.dumps(_GEOJSON_POLYGON_DOC), encoding="utf-8")
    out = tmp_path / "world.json"

    rc = main(["city-compile", "--geojson", str(geojson_path), "-o", str(out)])
    assert rc == 0
    assert "compiled 1 entities" in capsys.readouterr().out

    world = WorldIR.from_dict(json.loads(out.read_text(encoding="utf-8")))
    assert len(world.entities) == 1
    entity = next(iter(world.entities.values()))
    assert entity.name == "GIS Building"

    report = json.loads(Path(str(out) + ".report.json").read_text(encoding="utf-8"))
    assert report["geojson_compile"]["compiled"] == 1


def test_city_compile_combines_osm_and_geojson_sources(tmp_path: Path) -> None:
    osm_path = tmp_path / "campus.osm"
    osm_path.write_text(_OSM_DOC, encoding="utf-8")
    geojson_path = tmp_path / "buildings.geojson"
    geojson_path.write_text(json.dumps(_GEOJSON_POLYGON_DOC), encoding="utf-8")
    out = tmp_path / "world.json"

    rc = main(["city-compile", "--osm", str(osm_path), "--geojson", str(geojson_path),
               "-o", str(out)])
    assert rc == 0

    world = WorldIR.from_dict(json.loads(out.read_text(encoding="utf-8")))
    # 1 from the OSM building (way 100) + 1 from the GeoJSON polygon
    assert len(world.entities) == 2
    names = {e.name for e in world.entities.values()}
    assert names == {"South Building", "GIS Building"}


def test_city_compile_custom_report_path(tmp_path: Path) -> None:
    osm_path = tmp_path / "campus.osm"
    osm_path.write_text(_OSM_DOC, encoding="utf-8")
    out = tmp_path / "world.json"
    report_path = tmp_path / "custom_report.json"

    rc = main(["city-compile", "--osm", str(osm_path), "-o", str(out),
               "--report", str(report_path)])
    assert rc == 0
    assert report_path.exists()
    assert not Path(str(out) + ".report.json").exists()
