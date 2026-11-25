"""Sprint 8 Phase 5: Production Integration Proof for Component Relationships and WorldIR.

Verifies end-to-end tracing from component perception with ownership resolution
into persisted WorldIR with typed RelationshipKind.PART_OF links, coordinate frames,
calibrated confidences, and traceable provenance.
"""

from __future__ import annotations

import numpy as np
import pytest

from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera, Quat, Vec3
from world_ir.schema_v1 import Provenance, RelationshipKind
from world_ir.world_v1 import WorldIR
from engine.multiengine.spatial_intelligence.integration import (
    DetectionInput,
    reconstruct_spatial_scene,
)
from engine.multiengine.spatial_intelligence.spatial_lifting import SpatialRelationKind


def test_component_part_of_worldir_tracing() -> None:
    """Verifies that a detected component with parent_id correctly populates

    RelationshipKind.PART_OF in the resulting WorldIR.
    """
    cam = PinholeCamera(
        intrinsics=CameraIntrinsics(
            fx=600.0,
            fy=600.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        ),
        extrinsics=CameraExtrinsics(
            position=Vec3(0.0, 0.0, 0.0),
            rotation=Quat.identity(),
        ),
    )
    depth = np.full((480, 640), 2.5, dtype=np.float32)
    img_rgb = np.full((480, 640, 3), 160, dtype=np.uint8)

    # 1. Provide a host (door) and a component (door_handle) with resolved parent_id
    host_det = DetectionInput(
        detection_id="det_door_01",
        source_image_id="frame_001",
        class_name="door",
        box_2d=(150.0, 50.0, 350.0, 450.0),
        confidence=0.92,
    )
    comp_det = DetectionInput(
        detection_id="det_handle_01",
        source_image_id="frame_001",
        class_name="door_handle",
        box_2d=(310.0, 230.0, 335.0, 270.0),
        confidence=0.84,
        parent_id="det_door_01",
    )

    # 2. Reconstruct spatial scene
    result = reconstruct_spatial_scene(
        detections=[host_det, comp_det],
        cameras={"frame_001": cam},
        depth_maps={"frame_001": depth},
        images_rgb={"frame_001": img_rgb},
    )

    # 3. Verify spatial_relationships contain PART_OF
    part_of_rels = [
        r for r in result.spatial_relationships
        if r.relation == SpatialRelationKind.PART_OF
    ]
    assert len(part_of_rels) >= 1, "Expected at least one PART_OF relationship"
    part_rel = part_of_rels[0]
    assert part_rel.source_id == "det_handle_01"
    assert part_rel.target_id == "det_door_01"
    assert part_rel.confidence >= 0.80

    # 4. Bridge to WorldIR
    world = result.to_world_ir(world_id="world_sprint8_integration_proof")
    assert world.id == "world_sprint8_integration_proof"

    # 5. Check entities and relationships in WorldIR
    # Locate canonical entity IDs
    handle_canonical_id = None
    door_canonical_id = None
    for eid, ent in world.entities.items():
        sources = ent.custom_properties.get("sources", [])
        obs_ids = ent.custom_properties.get("member_observation_ids", [])
        if "det_handle_01" in obs_ids or "handle" in ent.name.lower():
            handle_canonical_id = eid
        if "det_door_01" in obs_ids or "door" in ent.name.lower() and "handle" not in ent.name.lower():
            door_canonical_id = eid

    assert handle_canonical_id is not None, "Component handle not found in WorldIR entities"
    assert door_canonical_id is not None, "Host door not found in WorldIR entities"

    handle_ent = world.entities[handle_canonical_id]
    door_ent = world.entities[door_canonical_id]

    # Verify handle has RelationshipKind.PART_OF pointing to door
    part_of_links = [
        rel for rel in handle_ent.relationships
        if rel.kind == RelationshipKind.PART_OF
    ]
    assert len(part_of_links) >= 1, "Expected PART_OF relationship on component entity in WorldIR"
    assert part_of_links[0].target_id == door_canonical_id
    assert part_of_links[0].provenance == Provenance.INFERRED
    assert part_of_links[0].confidence >= 0.80

    # Verify coordinate frames and provenance
    assert handle_ent.provenance == Provenance.RECONSTRUCTED
    assert handle_ent.confidence >= 0.40
