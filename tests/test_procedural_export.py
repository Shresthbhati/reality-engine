"""Procedural (GENERATED, transform-less, world-space-bounds) worlds must export from every format that can represent
them. Before sdk.reality applied exporters.boxes.with_explicit_placement, 11 of 14 exporters exported 0 of 35 entities
of a generated building."""

from __future__ import annotations

import numpy as np
import pytest

from exporters.boxes import with_explicit_placement
from procedural.building_grammar import BuildingSpec, generate_building
from provenance import Provenance
from sdk import reality
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3

FORMATS = sorted(set(reality._EXPORTERS) - {"sumo", "ifc"})      # sumo = road network only; ifc needs IfcOpenShell


def building():
    return generate_building(BuildingSpec(name="tower", footprint=(10.0, 8.0), storeys=2))


def boxed_ids(world):
    return {e.id for e in world.entities.values()
            if any(world.geometries[g].type in (GeometryType.BOX, GeometryType.PLANE) for g in e.geometry_ids)}


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_boxed_entity_of_a_procedural_building_is_exported(fmt):
    world = building()
    _, report = reality.export(world, fmt)
    assert set(report.entities_exported) == boxed_ids(world) and len(report.entities_exported) > 0


@pytest.mark.parametrize("profile", sorted(reality.EXPORT_PROFILES))
def test_profiles_agree_with_the_formats(profile):
    world = building()
    _, report = reality.export_profile(world, profile)
    assert set(report.entities_exported) == boxed_ids(world)


def test_placement_is_faithful_cityjson_extent_equals_geometry_bounds():
    world = building()
    doc, _ = reality.export(world, "cityjson")
    scale = doc["transform"]["scale"][0]
    verts = np.array(doc["vertices"]) * scale
    boxes = [world.geometries[g] for e in world.entities.values() for g in e.geometry_ids
             if world.geometries[g].type in (GeometryType.BOX, GeometryType.PLANE)]
    lo = [min(getattr(g.bounds_min, a) for g in boxes) for a in "xyz"]
    hi = [max(getattr(g.bounds_max, a) for g in boxes) for a in "xyz"]
    assert verts.min(0) == pytest.approx(lo, abs=1e-3) and verts.max(0) == pytest.approx(hi, abs=1e-3)


def test_input_world_is_not_mutated():
    world = building()
    assert all(e.transform is None for e in world.entities.values())
    reality.export(world, "gltf")
    assert all(e.transform is None for e in world.entities.values())


def test_only_generated_entities_are_placed_reconstructed_ones_stay_skipped():
    world = building()
    world.geometries["g-rec"] = Geometry(id="g-rec", type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0),
                                         bounds_max=Vector3(1, 1, 1), provenance=Provenance.OBSERVED)
    world.entities["rec"] = Entity(id="rec", name="rec", type=EntityType.WALL, geometry_ids=["g-rec"],
                                   provenance=Provenance.OBSERVED)          # no transform; frame of bounds unknown
    _, report = reality.export(world, "cityjson")
    assert "rec" in report.entities_skipped and "rec" not in report.entities_exported
    assert with_explicit_placement(world).entities["rec"].transform in (None, {})


def test_sumo_stays_road_only_with_its_own_reason():
    _, report = reality.export(building(), "sumo")
    assert report.entities_exported == () and "not a road" in " ".join(report.skip_reasons)


def test_world_that_needs_no_change_is_returned_as_is():
    world = building()
    placed = with_explicit_placement(world)
    assert with_explicit_placement(placed) is placed
