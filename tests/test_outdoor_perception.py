"""Unit tests for outdoor and environmental perception (Master Development Plan Task 6.7)."""

import math
import numpy as np
import pytest

from world_ir import WorldIR
from world_ir.schema_v1 import EntityType, GeometryType
from perception.environment.outdoor import (
    FootpathFit,
    FootpathRefused,
    LamppostFit,
    LamppostRefused,
    RoadRefused,
    RoadSurfaceFit,
    StreetFurnitureFit,
    StreetFurnitureRefused,
    TreeFit,
    TreeRefused,
    fit_footpath,
    fit_lamppost,
    fit_road_surface,
    fit_street_furniture,
    fit_tree,
    promote_outdoor_entity,
)


def _generate_synthetic_tree(
    trunk_radius: float = 0.2,
    trunk_height: float = 3.0,
    crown_radius: float = 1.8,
    crown_height: float = 4.0,
    n_trunk: int = 30,
    n_crown: int = 60,
    seed: int = 42,
) -> list:
    """Generate 3D points representing a tree with vertical trunk and spherical canopy."""
    rng = np.random.default_rng(seed)
    pts = []

    # Trunk points: cylinder along z
    for _ in range(n_trunk):
        theta = rng.uniform(0, 2 * math.pi)
        z = rng.uniform(0, trunk_height)
        r = trunk_radius + rng.normal(0, 0.01)
        pts.append((r * math.cos(theta), r * math.sin(theta), z))

    # Crown points: ellipsoid above trunk
    crown_center_z = trunk_height + crown_height / 2.0
    for _ in range(n_crown):
        u = rng.uniform(0, 1)
        phi = rng.uniform(0, 2 * math.pi)
        theta = rng.uniform(0, math.pi)
        r = (crown_radius * (u ** (1 / 3))) + rng.normal(0, 0.02)
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = crown_center_z + (crown_height / 2.0) * math.cos(theta) * (u ** (1 / 3))
        pts.append((x, y, z))

    return pts


def _generate_synthetic_lamppost(
    shaft_radius: float = 0.08,
    total_height: float = 6.0,
    arm_reach: float = 1.0,
    n_shaft: int = 35,
    n_head: int = 10,
    seed: int = 42,
) -> list:
    """Generate 3D points representing a slender street lamppost."""
    rng = np.random.default_rng(seed)
    pts = []

    # Slender vertical shaft
    for _ in range(n_shaft):
        theta = rng.uniform(0, 2 * math.pi)
        z = rng.uniform(0, total_height * 0.85)
        r = shaft_radius + rng.normal(0, 0.005)
        pts.append((r * math.cos(theta), r * math.sin(theta), z))

    # Top luminaire fixture offset along +x
    for _ in range(n_head):
        z = rng.uniform(total_height * 0.85, total_height)
        frac = (z - total_height * 0.85) / (total_height * 0.15)
        x = frac * arm_reach + rng.normal(0, 0.02)
        y = rng.normal(0, 0.02)
        pts.append((x, y, z))

    return pts


def _generate_synthetic_road(
    width: float = 7.0,
    length: float = 20.0,
    elevation: float = 0.5,
    n_points: int = 50,
    seed: int = 42,
) -> list:
    """Generate 3D points representing a horizontal road carriageway."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_points):
        x = rng.uniform(-width / 2.0, width / 2.0)
        y = rng.uniform(0, length)
        z = elevation + rng.normal(0, 0.01)
        pts.append((x, y, z))
    return pts


def _generate_synthetic_footpath(
    width: float = 1.8,
    length: float = 15.0,
    elevation: float = 0.65,
    n_points: int = 30,
    seed: int = 42,
) -> list:
    """Generate 3D points representing a pedestrian sidewalk."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_points):
        x = rng.uniform(0, width)
        y = rng.uniform(0, length)
        z = elevation + rng.normal(0, 0.005)
        pts.append((x, y, z))
    return pts


# -----------------------------------------------------------------------------
# Tree Tests
# -----------------------------------------------------------------------------

def test_fit_tree_success():
    pts = _generate_synthetic_tree(
        trunk_radius=0.25,
        trunk_height=3.0,
        crown_radius=2.0,
        crown_height=4.5,
    )
    fit = fit_tree(pts)
    assert isinstance(fit, TreeFit)
    assert 6.0 <= fit.total_height_m <= 8.5
    assert 0.15 <= fit.trunk_radius_m <= 0.35
    assert fit.crown_radius_m > fit.trunk_radius_m * 2.0
    assert fit.confidence > 0.5
    assert fit.point_count == len(pts)


def test_fit_tree_refusals():
    # Insufficient points
    with pytest.raises(TreeRefused, match="Insufficient points"):
        fit_tree([(0, 0, z) for z in range(5)])

    # Too short to be a tree (e.g. 0.5m)
    pts_short = [(0, 0, z * 0.05) for z in range(20)]
    with pytest.raises(TreeRefused, match="outside tree plausibility"):
        fit_tree(pts_short)

    # Pure cylinder without canopy expansion
    pts_pole = []
    for z in np.linspace(0, 4.0, 30):
        pts_pole.append((0.1, 0.0, float(z)))
        pts_pole.append((-0.1, 0.0, float(z)))
    with pytest.raises(TreeRefused, match="does not expand beyond trunk"):
        fit_tree(pts_pole)


# -----------------------------------------------------------------------------
# Lamppost Tests
# -----------------------------------------------------------------------------

def test_fit_lamppost_success():
    pts = _generate_synthetic_lamppost(
        shaft_radius=0.10,
        total_height=6.5,
        arm_reach=1.2,
    )
    fit = fit_lamppost(pts)
    assert isinstance(fit, LamppostFit)
    assert 6.0 <= fit.total_height_m <= 7.0
    assert 0.06 <= fit.shaft_radius_m <= 0.18
    assert fit.luminaire_offset_m >= 0.4
    assert fit.confidence >= 0.5


def test_fit_lamppost_refusals():
    # Too short
    pts_short = [(0.05, 0.0, float(z)) for z in np.linspace(0, 1.0, 15)]
    with pytest.raises(LamppostRefused, match="outside lamppost bounds"):
        fit_lamppost(pts_short)

    # Too stocky (e.g., radius = 1.0m, column/silo)
    pts_silo = []
    for theta in np.linspace(0, 2 * math.pi, 20):
        for z in np.linspace(0, 5.0, 5):
            pts_silo.append((1.0 * math.cos(theta), 1.0 * math.sin(theta), float(z)))
    with pytest.raises(LamppostRefused, match="too wide"):
        fit_lamppost(pts_silo)


# -----------------------------------------------------------------------------
# Road and Footpath Tests
# -----------------------------------------------------------------------------

def test_fit_road_surface_success():
    pts = _generate_synthetic_road(width=7.5, length=25.0, elevation=0.2)
    fit = fit_road_surface(pts)
    assert isinstance(fit, RoadSurfaceFit)
    assert 6.5 <= fit.width_m <= 8.5
    assert 20.0 <= fit.length_m <= 28.0
    assert abs(fit.elevation_m - 0.2) < 0.1
    assert fit.confidence >= 0.6


def test_fit_road_surface_refusals():
    # Too narrow (1.2m is not a carriageway)
    pts_narrow = _generate_synthetic_road(width=1.2, length=10.0)
    with pytest.raises(RoadRefused, match="too narrow"):
        fit_road_surface(pts_narrow)

    # Vertical wall points (not horizontal)
    pts_wall = [(0.0, float(y), float(z)) for y in range(10) for z in range(5)]
    with pytest.raises(RoadRefused, match="not horizontal"):
        fit_road_surface(pts_wall)


def test_fit_footpath_success():
    pts = _generate_synthetic_footpath(width=2.0, length=12.0, elevation=0.35)
    fit = fit_footpath(pts)
    assert isinstance(fit, FootpathFit)
    assert 1.5 <= fit.width_m <= 2.5
    assert 10.0 <= fit.length_m <= 14.0
    assert abs(fit.elevation_m - 0.35) < 0.05


def test_fit_footpath_refusals():
    # Width exceeds footpath limit
    pts_wide = _generate_synthetic_footpath(width=10.0, length=20.0)
    with pytest.raises(FootpathRefused, match="exceeds typical sidewalk limits"):
        fit_footpath(pts_wide)


# -----------------------------------------------------------------------------
# Street Furniture Tests
# -----------------------------------------------------------------------------

def test_fit_street_furniture():
    # Bench: 1.8m length, 0.6m depth, 0.8m height
    pts_bench = [
        (-0.9, -0.3, 0.0), (0.9, -0.3, 0.0), (-0.9, 0.3, 0.0), (0.9, 0.3, 0.0),
        (-0.9, -0.3, 0.8), (0.9, -0.3, 0.8), (-0.9, 0.3, 0.8), (0.9, 0.3, 0.8),
    ]
    fit = fit_street_furniture(pts_bench, kind="bench")
    assert isinstance(fit, StreetFurnitureFit)
    assert fit.furniture_kind == "bench"
    assert abs(fit.height_m - 0.8) < 0.01


# -----------------------------------------------------------------------------
# WorldIR Promotion Tests
# -----------------------------------------------------------------------------

def test_promote_outdoor_entities_to_world():
    world = WorldIR(id="test-outdoor-world")

    tree_pts = _generate_synthetic_tree()
    tree_fit = fit_tree(tree_pts)
    tree_ent = promote_outdoor_entity(tree_fit, world, evidence_ids=["img-tree-1"])

    assert tree_ent.id in world.entities
    assert tree_ent.type == EntityType.VEGETATION
    assert tree_ent.custom_properties["entity_class"] == "vegetation.tree"
    assert "total_height_m" in tree_ent.custom_properties["measurements"]
    assert "trunk_radius_m" in tree_ent.custom_properties["measurements"]
    geom_tree = world.geometries[tree_ent.geometry_ids[0]]
    assert geom_tree.type == GeometryType.CYLINDER

    lamp_pts = _generate_synthetic_lamppost()
    lamp_fit = fit_lamppost(lamp_pts)
    lamp_ent = promote_outdoor_entity(lamp_fit, world, evidence_ids=["img-lamp-1"])

    assert lamp_ent.id in world.entities
    assert lamp_ent.type == EntityType.STRUCTURE
    assert lamp_ent.custom_properties["entity_class"] == "street.lamppost"
    assert "shaft_radius_m" in lamp_ent.custom_properties["measurements"]

    road_pts = _generate_synthetic_road()
    road_fit = fit_road_surface(road_pts)
    road_ent = promote_outdoor_entity(road_fit, world, evidence_ids=["img-road-1"])

    assert road_ent.id in world.entities
    assert road_ent.type == EntityType.ROAD
    assert road_ent.custom_properties["entity_class"] == "road.carriageway"
    assert "width_m" in road_ent.custom_properties["measurements"]
    assert "area_m2" in road_ent.custom_properties["measurements"]

    path_pts = _generate_synthetic_footpath()
    path_fit = fit_footpath(path_pts)
    path_ent = promote_outdoor_entity(path_fit, world, evidence_ids=["img-path-1"])

    assert path_ent.id in world.entities
    assert path_ent.type == EntityType.SIDEWALK
    assert path_ent.custom_properties["entity_class"] == "road.footpath"

    # Verify serialization integrity
    world_dict = world.to_dict()
    assert len(world_dict["entities"]) == 4
    assert len(world_dict["geometries"]) == 4


def test_outdoor_spatial_reasoning_integration():
    from perception.spatial_reasoning import infer_spatial_relationships
    from world_ir.schema_v1 import RelationshipKind

    world = WorldIR(id="test-outdoor-spatial")

    # Footpath at ground level z = 0.0
    path_pts = _generate_synthetic_footpath(width=2.0, length=10.0, elevation=0.0)
    path_fit = fit_footpath(path_pts)
    path_ent = promote_outdoor_entity(path_fit, world)

    # Lamppost placed on the footpath (base at z = 0.0)
    lamp_pts = _generate_synthetic_lamppost(shaft_radius=0.08, total_height=5.0)
    lamp_fit = fit_lamppost(lamp_pts)
    lamp_ent = promote_outdoor_entity(lamp_fit, world)

    res = infer_spatial_relationships(world)
    assert res.resting_created >= 1

    # Check bilateral relationships
    lamp_rels = [r for r in lamp_ent.relationships if r.target_id == path_ent.id]
    assert any(r.kind == RelationshipKind.RESTS_ON for r in lamp_rels)

    path_rels = [r for r in path_ent.relationships if r.target_id == lamp_ent.id]
    assert any(r.kind == RelationshipKind.SUPPORTS for r in path_rels)
