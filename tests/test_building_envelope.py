"""Tests for Whole-Building Multi-Storey Envelope and Vertical Level Intelligence (Wave 2, Blocks 10 & 11)."""

from __future__ import annotations

import pytest

from engine.compiler.building_envelope import (
    BuildingEnvelopeReport,
    BuildingStoreyLevel,
    LevelClassification,
    reconstruct_building_envelope,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, RelationshipKind, Vector3
from world_ir.world_v1 import WorldIR


def test_multi_storey_building_clustering_and_envelope():
    world = WorldIR()

    # --- Storey -1: Basement at Y = -3.0 ---
    g_b1 = Geometry(id="g-b1", type=GeometryType.BOX, bounds_min=Vector3(x=0, y=-3.0, z=0), bounds_max=Vector3(x=8, y=-0.2, z=6))
    rm_b1 = Entity(id="rm-basement", name="Basement Storage", type=EntityType.ROOM, geometry_ids=["g-b1"],
                   custom_properties={"area_m2": 48.0})
    world.geometries["g-b1"] = g_b1
    world.entities["rm-basement"] = rm_b1

    # --- Storey 0: Ground Floor at Y = 0.0 ---
    g_g1 = Geometry(id="g-g1", type=GeometryType.BOX, bounds_min=Vector3(x=0, y=0.0, z=0), bounds_max=Vector3(x=6, y=2.8, z=6))
    rm_g1 = Entity(id="rm-living", name="Living Room", type=EntityType.ROOM, geometry_ids=["g-g1"],
                   custom_properties={"area_m2": 36.0})
    g_g2 = Geometry(id="g-g2", type=GeometryType.BOX, bounds_min=Vector3(x=6, y=0.0, z=0), bounds_max=Vector3(x=10, y=2.8, z=6))
    rm_g2 = Entity(id="rm-kitchen", name="Kitchen", type=EntityType.ROOM, geometry_ids=["g-g2"],
                   custom_properties={"area_m2": 24.0})
    world.geometries["g-g1"] = g_g1
    world.geometries["g-g2"] = g_g2
    world.entities["rm-living"] = rm_g1
    world.entities["rm-kitchen"] = rm_g2

    # --- Storey 1: Level 1 at Y = 3.0 ---
    g_l1 = Geometry(id="g-l1", type=GeometryType.BOX, bounds_min=Vector3(x=0, y=3.0, z=0), bounds_max=Vector3(x=10, y=5.8, z=6))
    rm_l1 = Entity(id="rm-bed-1", name="Master Bedroom", type=EntityType.ROOM, geometry_ids=["g-l1"],
                   custom_properties={"area_m2": 60.0})
    world.geometries["g-l1"] = g_l1
    world.entities["rm-bed-1"] = rm_l1

    # Perimeter Facade Wall (0.25m thick)
    g_facade = Geometry(id="g-fac", type=GeometryType.BOX, bounds_min=Vector3(x=0, y=0, z=-0.25), bounds_max=Vector3(x=10, y=6.0, z=0.0))
    w_facade = Entity(id="wall-ext-north", name="North Exterior Facade", type=EntityType.WALL, geometry_ids=["g-fac"],
                      custom_properties={"thickness_m": 0.25})
    world.geometries["g-facade"] = g_facade
    world.entities["wall-ext-north"] = w_facade

    report = reconstruct_building_envelope(world)

    assert isinstance(report, BuildingEnvelopeReport)
    assert report.total_storeys == 3
    assert report.footprint_area_m2 == pytest.approx(60.0)  # Ground floor: 36 + 24
    assert report.gross_floor_area_m2 == pytest.approx(168.0)  # 48 + 36 + 24 + 60
    assert report.total_height_m >= 8.5  # From -3.0 to +5.8

    # Verify storey classification
    classifications = {lvl.level_index: lvl.classification for lvl in report.levels}
    assert classifications[-1] == LevelClassification.BASEMENT
    assert classifications[0] == LevelClassification.GROUND
    assert classifications[1] == LevelClassification.INTERMEDIATE

    # Verify building hierarchy entities in WorldIR
    bld_id = report.building_id
    assert bld_id in world.entities
    bld_ent = world.entities[bld_id]
    assert bld_ent.type == EntityType.BUILDING

    # Verify storeys created
    storey_ents = [e for e in world.entities.values() if e.type == EntityType.STOREY]
    assert len(storey_ents) == 3

    # Check building CONTAINS storeys
    bld_contains = [r.target_id for r in bld_ent.relationships if r.kind == RelationshipKind.CONTAINS]
    assert "storey-lvl--1" in bld_contains
    assert "storey-lvl-+0" in bld_contains
    assert "storey-lvl-+1" in bld_contains

    # Check room CONTAINS relationship to storey
    ground_storey = world.entities["storey-lvl-+0"]
    ground_rooms = [r.target_id for r in ground_storey.relationships if r.kind == RelationshipKind.CONTAINS]
    assert "rm-living" in ground_rooms
    assert "rm-kitchen" in ground_rooms

    # Check facade wall detection
    assert "wall-ext-north" in report.facade_wall_ids
