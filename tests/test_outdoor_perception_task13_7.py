"""Unit tests for Outdoor Perception: Vegetation, Roads, Terrain, Facades, Roofs (Task 13.7)."""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np
import pytest

from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import EntityType, GeometryType, RelationshipKind
from perception.environment.outdoor import (
    FacadeFit,
    FacadeRefused,
    FootpathFit,
    LamppostFit,
    OutdoorSceneReconstruction,
    RoadSurfaceFit,
    RoofFit,
    RoofRefused,
    ShrubFit,
    ShrubRefused,
    StreetFurnitureFit,
    TerrainFit,
    TerrainRefused,
    TreeFit,
    fit_facade,
    fit_footpath,
    fit_lamppost,
    fit_road_surface,
    fit_roof,
    fit_shrub,
    fit_street_furniture,
    fit_terrain,
    fit_tree,
    parse_and_reconstruct_outdoor_scene,
    promote_outdoor_entity,
)

Point3 = Tuple[float, float, float]


def _gen_shrub_points(
    radius: float = 0.9,
    height: float = 1.4,
    base_z: float = 0.5,
    n_pts: int = 40,
    seed: int = 101,
) -> List[Point3]:
    """Generate 3D points representing a compact vegetative bush / shrub."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_pts):
        u = rng.uniform(0, 1)
        theta = rng.uniform(0, 2 * math.pi)
        r = radius * math.sqrt(u)
        x = r * math.cos(theta) + rng.normal(0, 0.02)
        y = r * math.sin(theta) + rng.normal(0, 0.02)
        z = base_z + rng.uniform(0, height)
        pts.append((float(x), float(y), float(z)))
    return pts


def _gen_terrain_points(
    slope_deg: float = 15.0,
    aspect_deg: float = 90.0,
    base_elev: float = 2.0,
    span_x: float = 25.0,
    span_y: float = 25.0,
    n_pts: int = 60,
    noise: float = 0.03,
    seed: int = 102,
) -> List[Point3]:
    """Generate 3D points representing outdoor terrain with slope and elevation."""
    rng = np.random.default_rng(seed)
    slope_rad = math.radians(slope_deg)
    tan_slope = math.tan(slope_rad)
    pts = []
    for _ in range(n_pts):
        x = rng.uniform(-span_x / 2.0, span_x / 2.0)
        y = rng.uniform(-span_y / 2.0, span_y / 2.0)
        z = base_elev + x * tan_slope + rng.normal(0, noise)
        pts.append((float(x), float(y), float(z)))
    return pts


def _gen_facade_points(
    width: float = 12.0,
    height: float = 8.0,
    base_z: float = 0.0,
    facing: str = "south",
    n_pts: int = 50,
    noise: float = 0.015,
    seed: int = 103,
) -> List[Point3]:
    """Generate 3D points representing a vertical building facade."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_pts):
        u = rng.uniform(-width / 2.0, width / 2.0)
        v = rng.uniform(0, height)
        z = base_z + v
        if facing == "south":
            x = u
            y = 10.0 + rng.normal(0, noise)
        elif facing == "east":
            x = 10.0 + rng.normal(0, noise)
            y = u
        elif facing == "north":
            x = u
            y = -10.0 + rng.normal(0, noise)
        else:  # west
            x = -10.0 + rng.normal(0, noise)
            y = u
        pts.append((float(x), float(y), float(z)))
    return pts


def _gen_roof_points(
    width: float = 10.0,
    length: float = 14.0,
    pitch_deg: float = 30.0,
    ridge_z: float = 12.0,
    n_pts: int = 40,
    noise: float = 0.02,
    seed: int = 104,
) -> List[Point3]:
    """Generate 3D points representing a pitched roof plane sloping down in +x."""
    rng = np.random.default_rng(seed)
    tan_p = math.tan(math.radians(pitch_deg))
    pts = []
    for _ in range(n_pts):
        x = rng.uniform(0, width)
        y = rng.uniform(0, length)
        z = ridge_z - x * tan_p + rng.normal(0, noise)
        pts.append((float(x), float(y), float(z)))
    return pts


# =============================================================================
# 1. Shrub / Bush Perception Tests
# =============================================================================

def test_fit_shrub_valid():
    pts = _gen_shrub_points(radius=1.0, height=1.6, base_z=0.2, n_pts=45)
    fit = fit_shrub(pts)

    assert isinstance(fit, ShrubFit)
    assert fit.height_m == pytest.approx(1.6, abs=0.25)
    assert fit.spread_radius_m == pytest.approx(1.0, abs=0.25)
    assert fit.ground_contact_z == pytest.approx(0.2, abs=0.1)
    assert fit.confidence >= 0.5
    assert fit.point_count == 45

    d = fit.to_dict()
    assert d["entity_class"] == "vegetation.shrub"
    assert "spread_radius_m" in d
    assert "ground_contact_z" in d


def test_fit_shrub_refusals():
    # 1. Too few points
    with pytest.raises(ShrubRefused, match="Insufficient points"):
        fit_shrub([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

    # 2. Oversized cluster (> 6m)
    big_pts = [(float(x), float(y), float(z)) for x in range(10) for y in (0, 1) for z in (0, 1)]
    with pytest.raises(ShrubRefused, match="outside shrub plausibility"):
        fit_shrub(big_pts)

    # 3. Slender tall tree-like cluster (dz=4.2m within [0.3, 5.0]*1.3, spread=0.1m, aspect > 3.0)
    tall_pts = [(0.1 * math.cos(i), 0.1 * math.sin(i), 4.2 * i / 11) for i in range(12)]
    with pytest.raises(ShrubRefused, match="tall slender aspect ratio"):
        fit_shrub(tall_pts)


# =============================================================================
# 2. Terrain Perception Tests
# =============================================================================

def test_fit_terrain_sloped():
    pts = _gen_terrain_points(slope_deg=18.0, base_elev=5.0, span_x=30.0, span_y=20.0, n_pts=60)
    fit = fit_terrain(pts)

    assert isinstance(fit, TerrainFit)
    assert fit.mean_slope_deg == pytest.approx(18.0, abs=3.0)
    assert fit.elevation_range_m > 5.0
    assert fit.roughness_rms_m < 0.10
    assert fit.area_m2 >= 400.0
    assert fit.confidence >= 0.70

    d = fit.to_dict()
    assert d["entity_class"] == "exterior.terrain"
    assert "mean_slope_deg" in d
    assert "roughness_rms_m" in d
    assert "elevation_range_m" in d


def test_fit_terrain_flat():
    rng = np.random.default_rng(201)
    pts = [(float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10)), float(rng.normal(1.0, 0.01))) for _ in range(40)]
    fit = fit_terrain(pts)

    assert fit.mean_slope_deg < 3.0
    assert fit.elevation_range_m < 0.2
    assert fit.confidence >= 0.75


def test_fit_terrain_refusals():
    # 1. Too few points (< 12)
    with pytest.raises(TerrainRefused, match="Insufficient points"):
        fit_terrain([(0.0, 0.0, 0.0)] * 5)

    # 2. Vertical wall / cliff (> 70 deg)
    wall_pts = [(0.0, float(y), float(z)) for y in range(5) for z in range(5)]
    with pytest.raises(TerrainRefused, match="too steep for terrain"):
        fit_terrain(wall_pts)


# =============================================================================
# 3. Building Facade Perception Tests
# =============================================================================

def test_fit_facade_south_facing():
    pts = _gen_facade_points(width=14.0, height=9.0, base_z=1.0, facing="south", n_pts=50)
    fit = fit_facade(pts, interior_point=(0.0, 15.0, 5.0))  # building center is North of facade

    assert isinstance(fit, FacadeFit)
    assert fit.width_m == pytest.approx(14.0, abs=0.5)
    assert fit.height_m == pytest.approx(9.0, abs=0.5)
    assert fit.base_elevation_m == pytest.approx(1.0, abs=0.35)
    assert fit.top_elevation_m == pytest.approx(10.0, abs=0.35)
    assert fit.area_m2 == pytest.approx(126.0, abs=15.0)
    assert fit.rms_residual_m < 0.05
    assert fit.confidence >= 0.75

    # South facing normal in horizontal plane
    assert fit.normal[1] < -0.8
    assert fit.azimuth_deg == pytest.approx(180.0, abs=15.0)

    d = fit.to_dict()
    assert d["entity_class"] == "exterior.facade"
    assert "azimuth_deg" in d
    assert "base_elevation_m" in d


def test_fit_facade_east_facing():
    pts = _gen_facade_points(width=10.0, height=6.0, base_z=0.0, facing="east", n_pts=45)
    fit = fit_facade(pts, interior_point=(0.0, 0.0, 3.0))  # building center at origin

    assert fit.normal[0] > 0.8
    assert fit.azimuth_deg == pytest.approx(90.0, abs=15.0)
    assert fit.width_m == pytest.approx(10.0, abs=0.5)
    assert fit.height_m == pytest.approx(6.0, abs=0.5)


def test_fit_facade_refusals():
    # 1. Too few points
    with pytest.raises(FacadeRefused, match="Insufficient points"):
        fit_facade([(0.0, 0.0, 0.0)] * 4)

    # 2. Horizontal plane (floor/ceiling) -> normal is along z, not vertical
    floor_pts = [(float(x), float(y), 0.0) for x in range(5) for y in range(5)]
    with pytest.raises(FacadeRefused, match="not vertical"):
        fit_facade(floor_pts)

    # 3. Tiny dimensions (height 1.0m < 1.8m)
    tiny_pts = [(float(x) * 0.1, 0.0, float(z) * 0.1) for x in range(5) for z in range(5)]
    with pytest.raises(FacadeRefused, match="Facade height .* is too low"):
        fit_facade(tiny_pts)


# =============================================================================
# 4. Roof Perception Tests
# =============================================================================

def test_fit_roof_pitched():
    pts = _gen_roof_points(width=8.0, length=12.0, pitch_deg=35.0, ridge_z=15.0, n_pts=45)
    fit = fit_roof(pts)

    assert isinstance(fit, RoofFit)
    assert fit.roof_type == "pitched"
    assert fit.pitch_deg == pytest.approx(35.0, abs=3.0)
    assert fit.ridge_elevation_m == pytest.approx(15.0, abs=0.5)
    assert fit.surface_area_m2 > fit.projected_area_m2
    assert fit.confidence >= 0.70

    d = fit.to_dict()
    assert d["entity_class"] == "exterior.roof"
    assert d["roof_type"] == "pitched"
    assert "pitch_deg" in d
    assert "surface_area_m2" in d


def test_fit_roof_flat():
    rng = np.random.default_rng(301)
    pts = [(float(rng.uniform(0, 10)), float(rng.uniform(0, 10)), float(12.0 + rng.normal(0, 0.01))) for _ in range(35)]
    fit = fit_roof(pts)

    assert fit.roof_type == "flat"
    assert fit.pitch_deg < 5.0
    assert fit.projected_area_m2 >= 50.0


def test_fit_roof_refusals():
    # 1. Too few points
    with pytest.raises(RoofRefused, match="Insufficient points"):
        fit_roof([(0.0, 0.0, 10.0)] * 3)

    # 2. Downward facing normal
    vert_pts = [(0.0, float(y), float(z)) for y in range(5) for z in range(5)]
    with pytest.raises(RoofRefused, match="not upward-facing"):
        fit_roof(vert_pts)

    # 3. Tiny area (< 1.5 m2)
    tiny_roof = [(float(x) * 0.1, float(y) * 0.1, 10.0) for x in range(4) for y in range(4)]
    with pytest.raises(RoofRefused, match="projected area .* is too small"):
        fit_roof(tiny_roof)


# =============================================================================
# 5. WorldIR Promotion for All Outdoor Types
# =============================================================================

def test_promote_all_outdoor_entities_to_world():
    world = WorldIR(id="outdoor-world-all-classes")

    shrub_pts = _gen_shrub_points()
    terrain_pts = _gen_terrain_points()
    facade_pts = _gen_facade_points()
    roof_pts = _gen_roof_points()

    fit_sh = fit_shrub(shrub_pts)
    fit_tr = fit_terrain(terrain_pts)
    fit_fa = fit_facade(facade_pts)
    fit_ro = fit_roof(roof_pts)

    ent_sh = promote_outdoor_entity(fit_sh, world, evidence_ids=["img-shrub-1"])
    ent_tr = promote_outdoor_entity(fit_tr, world, evidence_ids=["img-terrain-1"])
    ent_fa = promote_outdoor_entity(fit_fa, world, evidence_ids=["img-facade-1"])
    ent_ro = promote_outdoor_entity(fit_ro, world, evidence_ids=["img-roof-1"])

    # 1. Shrub verification
    assert ent_sh.type == EntityType.VEGETATION
    assert ent_sh.custom_properties["entity_class"] == "vegetation.shrub"
    assert "height_m" in ent_sh.custom_properties["measurements"]
    assert ent_sh.id in world.entities

    # 2. Terrain verification
    assert ent_tr.type == EntityType.TERRAIN
    assert ent_tr.custom_properties["entity_class"] == "exterior.terrain"
    assert "mean_slope_deg" in ent_tr.custom_properties["measurements"]
    assert ent_tr.id in world.entities

    # 3. Facade verification
    assert ent_fa.type == EntityType.WALL
    assert ent_fa.custom_properties["entity_class"] == "exterior.facade"
    assert "azimuth_deg" in ent_fa.custom_properties["measurements"]
    assert ent_fa.id in world.entities

    # 4. Roof verification
    assert ent_ro.type == EntityType.ROOF
    assert ent_ro.custom_properties["entity_class"] == "exterior.roof"
    assert "pitch_deg" in ent_ro.custom_properties["measurements"]
    assert ent_ro.id in world.entities

    # Verify WorldIR can serialize cleanly
    world_dict = world.to_dict()
    assert len(world_dict["entities"]) == 4
    assert len(world_dict["geometries"]) == 4


# =============================================================================
# 6. Outdoor Scene Reconstruction & Topological Integration
# =============================================================================

def test_parse_and_reconstruct_outdoor_scene_with_topology():
    world = WorldIR(id="outdoor-full-scene")

    clusters = [
        ("terrain", _gen_terrain_points(base_elev=0.0)),
        ("facade", _gen_facade_points(base_z=0.0, height=8.0)),
        ("roof", _gen_roof_points(ridge_z=12.0)),
        ("shrub", _gen_shrub_points(base_z=0.2)),
    ]

    scene = parse_and_reconstruct_outdoor_scene(clusters, world, evidence_ids=["outdoor-cam-1"])

    assert isinstance(scene, OutdoorSceneReconstruction)
    assert scene.terrain is not None
    assert len(scene.facades) == 1
    assert len(scene.roofs) == 1
    assert len(scene.vegetation) == 1

    # Verify topological relationships
    facade_entity = scene.facades[0]
    roof_entity = scene.roofs[0]
    shrub_entity = scene.vegetation[0]

    # 1. Facade rests on terrain
    assert any(
        r.kind == RelationshipKind.RESTS_ON and r.target_id == scene.terrain.id
        for r in facade_entity.relationships
    )

    # 2. Roof supported by facade
    assert any(
        r.kind == RelationshipKind.SUPPORTED_BY and r.target_id == facade_entity.id
        for r in roof_entity.relationships
    )

    # 3. Shrub rests on terrain
    assert any(
        r.kind == RelationshipKind.RESTS_ON and r.target_id == scene.terrain.id
        for r in shrub_entity.relationships
    )

    # Verify world contains all entities
    assert len(world.entities) == 4
    for ent in scene.all_entities:
        assert ent.id in world.entities
