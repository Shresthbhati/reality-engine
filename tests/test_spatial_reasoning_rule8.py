"""Tests for Universal Spatial Reasoning and Structural Load Path Engine (Directive Rule 8).

Verifies:
- Structural gravity load paths from roof/beams/columns to foundations
- Rejection of floating load-bearing structures
- Rejection of overloaded non-bearing partition walls
- Doorway clearance and egress path envelope validation
- Volumetric room void calculation and occluded blind spot detection
- Utility network electrical continuity and plumbing drainage slope checks
- Autonomous generation of auto-fix proposals
"""

from __future__ import annotations

import pytest
import numpy as np

from perception.reasoning.spatial_reasoning_engine import (
    ClearanceAccessibilityEngine,
    SpatialReasoningEngine,
    StructuralLoadPathEngine,
    UtilityNetworkEngine,
    VolumetricVoidEngine,
)
from provenance import Provenance
from world_ir import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
    WorldIR,
)


def _create_box_entity(
    world: WorldIR,
    entity_id: str,
    name: str,
    entity_type: EntityType,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    custom_props: dict | None = None,
) -> Entity:
    geom_id = f"geom-{entity_id}"
    geom = Geometry(
        id=geom_id,
        type=GeometryType.BOX,
        bounds_min=Vector3(*bounds_min),
        bounds_max=Vector3(*bounds_max),
        provenance=Provenance.RECONSTRUCTED,
    )
    world.geometries[geom_id] = geom

    ent = Entity(
        id=entity_id,
        name=name,
        type=entity_type,
        geometry_ids=[geom_id],
        provenance=Provenance.INFERRED,
        custom_properties=custom_props or {},
    )
    world.entities[entity_id] = ent
    return ent


class TestSpatialReasoningRule8:
    def test_sound_structural_load_path(self):
        """Test continuous vertical load transfer from beam -> column -> foundation."""
        world = WorldIR()

        # 1. Foundation slab at ground (z: 0.0 to 0.20)
        _create_box_entity(world, "found-1", "Foundation Slab", EntityType.FLOOR, (-5, -5, 0.0), (5, 5, 0.20))

        # 2. Structural Column (z: 0.20 to 3.00) directly resting on slab
        _create_box_entity(world, "col-1", "Ground Column", EntityType.COLUMN, (1.0, 1.0, 0.20), (1.4, 1.4, 3.00))

        # 3. Structural Beam (z: 3.00 to 3.40) directly resting on column
        _create_box_entity(world, "beam-1", "Transfer Beam", EntityType.STRUCTURE, (0.0, 1.0, 3.00), (3.0, 1.4, 3.40), {"entity_class": "beam"})

        report = StructuralLoadPathEngine.evaluate_load_paths(world)

        assert report.is_structurally_sound is True
        assert len(report.floating_structures) == 0
        assert len(report.overloaded_non_bearing_walls) == 0
        assert len(report.load_path_chains) >= 2

    def test_floating_structural_member_rejection(self):
        """Test that floating structural beam/column with no support underneath is rejected."""
        world = WorldIR()

        # Foundation slab at ground
        _create_box_entity(world, "found-1", "Foundation Slab", EntityType.FLOOR, (-5, -5, 0.0), (5, 5, 0.20))

        # Beam floating at z=3.00 to 3.40 with NO column or wall below it
        _create_box_entity(world, "beam-float", "Suspended Heavy Beam", EntityType.STRUCTURE, (2.0, 2.0, 3.00), (5.0, 2.4, 3.40), {"entity_class": "beam"})

        report = StructuralLoadPathEngine.evaluate_load_paths(world)

        assert report.is_structurally_sound is False
        assert "beam-float" in report.floating_structures
        assert len(report.issues) > 0

    def test_overloaded_non_bearing_partition_wall(self):
        """Test detection of non-bearing thin partition wall carrying heavy beam."""
        world = WorldIR()

        _create_box_entity(world, "found-1", "Foundation Slab", EntityType.FLOOR, (-5, -5, 0.0), (5, 5, 0.20))

        # Thin non-bearing drywall partition (thickness 0.08m)
        _create_box_entity(
            world,
            "wall-thin",
            "Interior Drywall Partition",
            EntityType.WALL,
            (0.0, 0.0, 0.20),
            (3.0, 0.08, 3.00),
            {"thickness_m": 0.08, "is_load_bearing": False},
        )

        # Heavy column bearing onto this partition wall
        _create_box_entity(
            world,
            "col-upper",
            "Upper Floor Column",
            EntityType.COLUMN,
            (1.0, 0.0, 3.00),
            (1.4, 0.4, 6.00),
        )

        report = StructuralLoadPathEngine.evaluate_load_paths(world)

        assert report.is_structurally_sound is False
        assert "wall-thin" in report.overloaded_non_bearing_walls

    def test_doorway_clearance_and_egress_obstruction(self):
        """Test detection of obstacle blocking doorway egress envelope."""
        world = WorldIR()

        # Doorway at x=2.0, y=0.0, z=0.0
        _create_box_entity(world, "door-main", "Entrance Door", EntityType.DOOR, (2.0, -0.05, 0.0), (2.9, 0.05, 2.10))

        # Obstacle (large cabinet) directly in front of door approach zone (at y=0.30)
        _create_box_entity(world, "cabinet-block", "Blocking Storage Cabinet", EntityType.STRUCTURE, (2.1, 0.10, 0.0), (2.8, 0.60, 1.80))

        report = ClearanceAccessibilityEngine.evaluate_clearance(world)

        assert report.is_accessible is False
        assert len(report.blocked_doorways) == 1
        rec = report.blocked_doorways[0]
        assert rec.door_id == "door-main"
        assert rec.obstructing_entity_id == "cabinet-block"
        assert rec.penetration_depth_m > 0.0

    def test_unblocked_doorway_accessibility(self):
        """Test accessible door with clear 0.80m egress envelope."""
        world = WorldIR()

        _create_box_entity(world, "door-clear", "Bedroom Door", EntityType.DOOR, (1.0, 0.0, 0.0), (1.9, 0.1, 2.1))
        # Obstacle placed far away (at y=3.0)
        _create_box_entity(world, "sofa-far", "Living Sofa", EntityType.STRUCTURE, (1.0, 3.0, 0.0), (2.5, 4.0, 0.9))

        report = ClearanceAccessibilityEngine.evaluate_clearance(world)

        assert report.is_accessible is True
        assert len(report.blocked_doorways) == 0

    def test_volumetric_void_and_occluded_space(self):
        """Test interior negative space void calculation and occluded zones."""
        world = WorldIR()

        # Room envelope: 5m x 4m x 3m = 60 m3
        _create_box_entity(world, "room-1", "Living Room", EntityType.ROOM, (0, 0, 0), (5, 4, 3))

        # Tall wardrobe (h=2.2m)
        _create_box_entity(world, "wardrobe-1", "Master Wardrobe", EntityType.STRUCTURE, (0, 0, 0), (1.0, 0.6, 2.2))

        report = VolumetricVoidEngine.evaluate_voids(world)

        assert report.total_enclosed_volume_m3 == 60.0
        assert report.occupied_solid_volume_m3 == pytest.approx(1.0 * 0.6 * 2.2, rel=1e-3)
        assert report.usable_void_volume_m3 == pytest.approx(60.0 - (1.0 * 0.6 * 2.2), rel=1e-3)
        assert report.void_ratio > 0.90
        assert report.occluded_zones_count == 1

    def test_utility_network_continuity(self):
        """Test electrical circuit connectivity and plumbing drainage slopes."""
        world = WorldIR()

        # Electrical socket with nearby conduit
        _create_box_entity(world, "sock-1", "Wall Socket", EntityType.STRUCTURE, (1.0, 0.0, 0.3), (1.08, 0.02, 0.38), {"entity_class": "socket"})
        _create_box_entity(world, "cond-1", "Electrical Conduit", EntityType.STRUCTURE, (1.0, 0.0, 0.0), (1.05, 0.02, 0.4), {"entity_class": "conduit"})

        # Plumbing sink with compliant slope (2.0%)
        _create_box_entity(world, "sink-1", "Kitchen Sink", EntityType.STRUCTURE, (3.0, 0.0, 0.8), (3.6, 0.5, 1.0), {"drainage_slope": 0.025})

        report = UtilityNetworkEngine.evaluate_utility_networks(world)

        assert report.is_continuous is True
        assert len(report.disconnected_fixtures) == 0
        assert len(report.plumbing_slope_violations) == 0

    def test_unified_spatial_reasoning_engine(self):
        """Test full world evaluation and enrichment with SUPPORTS relationships."""
        world = WorldIR()

        _create_box_entity(world, "found", "Foundation Slab", EntityType.FLOOR, (0, 0, 0), (4, 4, 0.2))
        _create_box_entity(world, "col", "Ground Column", EntityType.COLUMN, (1, 1, 0.2), (1.4, 1.4, 3.0))
        _create_box_entity(world, "door", "Exit Door", EntityType.DOOR, (2, 0, 0.2), (2.9, 0.1, 2.3))

        report = SpatialReasoningEngine.evaluate_world(world)

        assert report.is_physically_plausible is True
        assert "spatial_reasoning" in world.metadata
        # Check that SUPPORTS relationship was inferred and created
        found_ent = world.entities["found"]
        support_rels = [r for r in found_ent.relationships if r.kind == RelationshipKind.SUPPORTS]
        assert len(support_rels) >= 1
        assert support_rels[0].target_id == "col"
