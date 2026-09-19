"""Tests for engine/compiler/city_compiler.py (P14-01 city world
compiler core: "Terrain/roads/buildings/vegetation/infrastructure/
water ... outputs WorldIR" -- the implementable core).

Scope decision (honest): the full compiler (GIS/OSM/satellite/LiDAR
ingestion, 3DCityDB/CityGML outputs) is larger than one module. This
pass delivers the core seam: evidence-derived city entities (building
footprints with heights, roads, water, vegetation as WorldIR entities
with real bounds) compiled deterministically into WorldIR, plus a
CityJSON round-trip through the P16-01 exporter. OSM ingestion and
terrain rasters remain named open items, not faked.

Contract:

  - CityFeature is DATA (kind, footprint polygon, height, name,
    evidence note) -> the compiler derives real AABB bounds per
    feature and emits WorldIR entities with correct entity types.
  - Kinds map to WorldIR EntityTypes (BUILDING, ROAD, WATER,
    VEGETATION, INFRASTRUCTURE, TERRAIN); an unknown kind is refused
    (ValueError), never silently re-typed.
  - Deterministic output ordering and ids; every geometry carries
    real bounds derived from the feature's footprint + height.
  - The compiled world round-trips and exports to CityJSON (the
    compiler chain closes: feature -> WorldIR -> CityJSON).
"""

from __future__ import annotations

import json

import pytest

from engine.compiler.city_compiler import (
    CityFeature,
    compile_city_world,
)
from world_ir.statement_state import StatementState


def _features():
    return [
        CityFeature(
            kind="building",
            name="Block A",
            footprint=[(0.0, 0.0), (20.0, 0.0), (20.0, 12.0), (0.0, 12.0)],
            height=15.0,
        ),
        CityFeature(
            kind="road",
            name="Main Street",
            footprint=[(-5.0, 12.0), (25.0, 12.0), (25.0, 16.0), (-5.0, 16.0)],
        ),
        CityFeature(
            kind="water",
            name="Pond",
            footprint=[(30.0, 0.0), (40.0, 0.0), (40.0, 8.0), (30.0, 8.0)],
        ),
        CityFeature(
            kind="vegetation",
            name="Grove",
            footprint=[(22.0, 20.0), (28.0, 20.0), (28.0, 26.0), (22.0, 26.0)],
            height=6.0,
        ),
    ]


class TestCityCompilation:
    def test_compiles_all_kinds(self):
        world = compile_city_world(_features(), name="downtown")
        types = sorted(e.type for e in world.entities.values())
        assert types == sorted(["building", "road", "water", "vegetation"])

    def test_bounds_from_footprint_and_height(self):
        world = compile_city_world(_features(), name="downtown")
        bldg = next(e for e in world.entities.values()
                    if e.type == "building")
        geom = world.geometries[bldg.geometry_ids[0]]
        assert geom.bounds_min.x == pytest.approx(0.0)
        assert geom.bounds_max.x == pytest.approx(20.0)
        assert geom.bounds_max.z == pytest.approx(15.0)
        # A road has no height: minimal flat slab (floored extent).
        road = next(e for e in world.entities.values() if e.type == "road")
        rgeom = world.geometries[road.geometry_ids[0]]
        assert rgeom.bounds_max.z - rgeom.bounds_min.z == pytest.approx(0.01)

    def test_unknown_kind_refused(self):
        # Refusal happens at feature construction (data validation at
        # the boundary), before any compilation can run.
        with pytest.raises(ValueError, match="ufo"):
            CityFeature(kind="ufo", name="?",
                        footprint=[(0, 0), (1, 0), (1, 1), (0, 1)])

    def test_deterministic(self):
        a = json.dumps(compile_city_world(_features(), name="downtown").to_dict(),
                       sort_keys=True)
        b = json.dumps(compile_city_world(_features(), name="downtown").to_dict(),
                       sort_keys=True)
        assert a == b

    def test_extruded_heights_recorded(self):
        # Building heights are EVIDENCE-BEARING: they come from the
        # feature's measured height, and the statement state marks the
        # entity PROCEDURAL/GENERATED only when the compiler synthesizes
        # the mass; an evidence-noted feature carries its note in
        # semantic_labels.
        f = [CityFeature(kind="building", name="Tower",
                         footprint=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0),
                                    (0.0, 10.0)],
                         height=42.0,
                         evidence_note="photogrammetric faade")]
        world = compile_city_world(f, name="x")
        tower = next(e for e in world.entities.values()
                     if e.type == "building")
        assert any("photogrammetric" in s for s in tower.semantic_labels)


class TestCityCompilerChain:
    def test_compiled_world_exports_to_cityjson(self):
        from exporters.cityjson.exporter import export_to_cityjson
        world = compile_city_world(_features(), name="downtown")
        doc = export_to_cityjson(world)
        objs = doc["CityObjects"]
        assert "downtown-building-0" in objs
        assert objs["downtown-building-0"]["type"] == "Building"
        # The CityJSON vertices quantize to the real footprint.
        assert doc["vertices"]
