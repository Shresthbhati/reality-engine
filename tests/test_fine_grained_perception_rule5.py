"""Unit and integration tests for Universal Spatial Intelligence Directive Rule 5 & 6:
Fine-Grained Perception, Small-Object Detection, Occlusion Reasoning, and Multi-View Identity Association.
"""

from __future__ import annotations

import pytest

from perception.entities import taxonomy
from perception.fine_grained import (
    FineGrainedPerceptionEngine,
    FineGrainedPerceptionOptions,
    MultiViewIdentityAssociator,
    OcclusionEngine,
    VisualObservation3D,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, RelationshipKind, Vector3
from world_ir.world_v1 import WorldIR


def test_hierarchical_taxonomy_coverage():
    """Verify taxonomy covers electrical, plumbing, HVAC, architectural, and urban objects."""
    all_classes = {c.id: c for c in taxonomy.all_classes()}

    # Electrical
    assert "electrical.switchboard.module.socket" in all_classes
    assert "electrical.switchboard.module.switch" in all_classes
    assert "electrical.distribution_board" in all_classes
    assert "electrical.conduit" in all_classes

    # Plumbing
    assert "plumbing.pipe" in all_classes
    assert "plumbing.tap" in all_classes
    assert "plumbing.drain" in all_classes

    # HVAC
    assert "hvac.vent" in all_classes
    assert "hvac.thermostat" in all_classes
    assert "hvac.radiator" in all_classes

    # Outdoor & Urban
    assert "road.carriageway" in all_classes
    assert "street.lamppost" in all_classes
    assert "vegetation.tree" in all_classes


def test_occlusion_engine_line_of_sight():
    """Verify occlusion engine handles unobstructed, obstructed, and grazing angle viewpoints."""
    engine = OcclusionEngine(max_view_angle_deg=75.0)

    target_center = (2.0, 1.2, 0.0)
    target_normal = (0.0, 0.0, 1.0)  # Facing +Z

    # 1. Frontal clear camera at (2.0, 1.2, 2.0)
    res_clear = engine.check_visibility(
        camera_position=(2.0, 1.2, 2.0),
        target_center=target_center,
        target_normal=target_normal,
    )
    assert res_clear.is_visible is True
    assert res_clear.occlusion_ratio == 0.0
    assert res_clear.view_angle_deg < 5.0

    # 2. Back-facing camera at (2.0, 1.2, -2.0)
    res_back = engine.check_visibility(
        camera_position=(2.0, 1.2, -2.0),
        target_center=target_center,
        target_normal=target_normal,
    )
    assert res_back.is_visible is False
    assert res_back.occlusion_ratio == 1.0

    # 3. Obstructed by column between target and camera
    column_aabb = ("column-1", (1.8, 0.0, 0.5), (2.2, 2.5, 1.0))
    res_obstructed = engine.check_visibility(
        camera_position=(2.0, 1.2, 2.0),
        target_center=target_center,
        target_normal=target_normal,
        occluders=[column_aabb],
    )
    assert res_obstructed.is_visible is False
    assert res_obstructed.obstruction_entity_id == "column-1"


def test_multiview_identity_association_and_deduplication():
    """Verify observations of the same entity from different cameras merge, while distinct entities stay separate."""
    associator = MultiViewIdentityAssociator(small_object_dist_thresh_m=0.15)

    # Two observations of Socket A (from camera 1 and camera 2 with slight 2cm discrepancy)
    obs_a1 = VisualObservation3D(
        observation_id="obs_a1",
        class_id="electrical.switchboard.module.socket",
        camera_id="cam_01",
        center_world=(1.20, 0.30, 0.0),
        normal_world=(0.0, 0.0, 1.0),
        dimensions_m=(0.086, 0.086, 0.02),
        confidence=0.88,
        host_surface_id="wall_north",
    )
    obs_a2 = VisualObservation3D(
        observation_id="obs_a2",
        class_id="electrical.switchboard.module.socket",
        camera_id="cam_02",
        center_world=(1.22, 0.31, 0.0),
        normal_world=(0.0, 0.0, 1.0),
        dimensions_m=(0.086, 0.086, 0.02),
        confidence=0.92,
        host_surface_id="wall_north",
    )

    # Distinct Socket B located 1.5m away
    obs_b1 = VisualObservation3D(
        observation_id="obs_b1",
        class_id="electrical.switchboard.module.socket",
        camera_id="cam_01",
        center_world=(2.70, 0.30, 0.0),
        normal_world=(0.0, 0.0, 1.0),
        dimensions_m=(0.086, 0.086, 0.02),
        confidence=0.90,
        host_surface_id="wall_north",
    )

    clusters = associator.associate_observations([obs_a1, obs_a2, obs_b1])

    # Exactly 2 distinct physical entities must be formed
    assert len(clusters) == 2

    # Find cluster for Socket A
    cluster_a = [c for c in clusters if c.fused_center[0] < 2.0][0]
    assert cluster_a.observation_count == 2
    assert "cam_01" in cluster_a.source_camera_ids
    assert "cam_02" in cluster_a.source_camera_ids
    assert cluster_a.fused_confidence > 0.92  # Confidence elevated by multi-view confirmation
    assert cluster_a.spatial_variance_m > 0.0  # Measured variance between views

    # Find cluster for Socket B
    cluster_b = [c for c in clusters if c.fused_center[0] > 2.0][0]
    assert cluster_b.observation_count == 1
    assert cluster_b.source_camera_ids == ["cam_01"]


def test_fine_grained_perception_pipeline_world_promotion():
    """Verify fine-grained pipeline promotes small components with exact geometry and host relationships."""
    world = WorldIR(id="test-fine-grained-world")

    # Host wall
    g_wall = Geometry(
        id="geom-wall-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, -0.1),
        bounds_max=Vector3(4.0, 2.8, 0.1),
    )
    e_wall = Entity(
        id="wall-1",
        name="North Bedroom Wall",
        type=EntityType.WALL,
        geometry_ids=["geom-wall-1"],
    )
    world.geometries[g_wall.id] = g_wall
    world.entities[e_wall.id] = e_wall

    observations = [
        # Electrical socket on wall-1
        VisualObservation3D(
            observation_id="obs-s1",
            class_id="electrical.switchboard.module.socket",
            camera_id="cam-01",
            center_world=(1.5, 0.30, 0.1),
            normal_world=(0.0, 0.0, 1.0),
            dimensions_m=(0.086, 0.086, 0.025),
            confidence=0.94,
            host_surface_id="wall-1",
        ),
        # Rocker light switch on wall-1
        VisualObservation3D(
            observation_id="obs-sw1",
            class_id="electrical.switchboard.module.switch",
            camera_id="cam-01",
            center_world=(0.9, 1.20, 0.1),
            normal_world=(0.0, 0.0, 1.0),
            dimensions_m=(0.086, 0.086, 0.025),
            confidence=0.96,
            host_surface_id="wall-1",
        ),
        # Copper plumbing pipe
        VisualObservation3D(
            observation_id="obs-p1",
            class_id="plumbing.pipe",
            camera_id="cam-01",
            center_world=(3.0, 0.45, 0.12),
            normal_world=(0.0, 0.0, 1.0),
            dimensions_m=(1.2, 0.022, 0.022),
            confidence=0.91,
            host_surface_id="wall-1",
        ),
        # Absurd size candidate (e.g. 5-meter socket) that must be rejected by taxonomy priors
        VisualObservation3D(
            observation_id="obs-absurd",
            class_id="electrical.switchboard.module.socket",
            camera_id="cam-01",
            center_world=(2.0, 1.5, 0.1),
            normal_world=(0.0, 0.0, 1.0),
            dimensions_m=(5.0, 5.0, 1.0),
            confidence=0.99,
            host_surface_id="wall-1",
        ),
    ]

    pipeline = FineGrainedPerceptionEngine()
    updated_world, summary = pipeline.process_observations_and_promote(
        raw_observations=observations,
        world=world,
        camera_positions={"cam-01": (1.5, 1.0, 2.5)},
    )

    assert summary.total_entities_detected == 3  # The absurd 5m socket was rejected!
    assert summary.electrical_components_count == 2
    assert summary.plumbing_components_count == 1

    # Verify promoted entities exist in WorldIR
    promoted_entities = [e for e in updated_world.entities.values() if e.id != "wall-1"]
    assert len(promoted_entities) == 3

    for ent in promoted_entities:
        # Must have attached_to relationship pointing to host wall
        assert len(ent.relationships) == 1
        rel = ent.relationships[0]
        assert rel.kind == RelationshipKind.ATTACHED_TO
        assert rel.target_id == "wall-1"

        # Must have valid 3D geometry in world
        assert len(ent.geometry_ids) == 1
        gid = ent.geometry_ids[0]
        assert gid in updated_world.geometries
        geom = updated_world.geometries[gid]
        assert geom.bounds_min is not None
        assert geom.bounds_max is not None
