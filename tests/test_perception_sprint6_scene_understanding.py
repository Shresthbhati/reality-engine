"""Tests for Gemini Perception Sprint 6: Scene Understanding and Reconstruction Intelligence."""

from __future__ import annotations

import numpy as np
import pytest

from engine.multiengine.spatial_intelligence import (
    CaptureIssueType,
    CapturePriority,
    CaptureRequest,
    DetectionInput,
    EdgeValidationStatus,
    EntityVisualAttributes,
    EvidencePresenceState,
    EvidenceRelationship,
    GeometryValidationResult,
    MaterialType,
    ModelCapabilityRegistry,
    ModelDescriptor,
    ModelTaskType,
    NodePresenceState,
    ObjectCondition,
    ObjectEvidenceAssessment,
    ObjectGeometryEvidence,
    ObjectState,
    RelationshipEngine,
    RelationshipType,
    RoomCompletenessEngine,
    RoomCompletenessReport,
    SceneCategory,
    SceneGraphEdge,
    SceneGraphNode,
    SceneObservationGraph,
    SequentialModelExecutionManager,
    SpatialEntity3D,
    SpatialRelationKind,
    SpatialRelationship,
    SurfaceFinish,
    build_scene_observation_graph,
    classify_entity_category,
    extract_visual_attributes,
    fuse_multiview_attributes,
    reconstruct_spatial_scene,
    validate_chair_geometry,
    validate_door_geometry,
    validate_object_geometry,
    validate_table_geometry,
    validate_window_geometry,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera, Quat, Vec3


def test_scene_observation_graph_construction_and_contradiction_detection() -> None:
    graph = SceneObservationGraph(scene_id="test_scene_001")

    door_node = SceneGraphNode(
        node_id="door_01",
        class_name="door",
        taxonomy_id="interior.opening.door",
        entity_category="opening",
        presence_state=NodePresenceState.VERIFIED_PRESENT,
        confidence=0.92,
        uncertainty=0.08,
        is_multiview_confirmed=True,
        subpart_ids=("handle_01",),
    )
    handle_node = SceneGraphNode(
        node_id="handle_01",
        class_name="door_handle",
        taxonomy_id="interior.opening.door_handle",
        entity_category="component",
        presence_state=NodePresenceState.VERIFIED_PRESENT,
        confidence=0.88,
        uncertainty=0.12,
        parent_id="door_01",
    )
    graph.add_node(door_node)
    graph.add_node(handle_node)

    assert len(graph.nodes) == 2
    subparts = graph.get_subparts("door_01")
    assert len(subparts) == 1
    assert subparts[0].node_id == "handle_01"

    # Add valid edge: handle part_of door
    graph.add_edge(SceneGraphEdge(
        source_id="handle_01",
        target_id="door_01",
        relation_type="part_of",
        confidence=0.95,
        validation_status=EdgeValidationStatus.VALIDATED,
    ))

    # Add contradictory edge: handle also contains door
    graph.add_edge(SceneGraphEdge(
        source_id="handle_01",
        target_id="door_01",
        relation_type="contains",
        confidence=0.80,
    ))

    contradictions = graph.detect_contradictions()
    assert len(contradictions) >= 1
    assert any(c["type"] == "direct_relation_conflict" for c in contradictions)

    # Serialization
    g_dict = graph.to_dict()
    assert g_dict["scene_id"] == "test_scene_001"
    assert g_dict["node_count"] == 2
    assert len(g_dict["contradictions"]) >= 1


def test_geometry_validators_for_supported_classes() -> None:
    # 1. Door validation
    # Plausible door: vertical aperture (aspect ratio > 1.2), horizontal normal
    valid_door = ObjectGeometryEvidence(
        box_2d=(100.0, 50.0, 200.0, 300.0),  # w=100, h=250 -> ar = 2.5
        class_name="door",
        surface_normal=(0.0, 1.0, 0.0),  # normal along Y (wall), nz = 0
    )
    res_valid = validate_door_geometry(valid_door)
    assert res_valid.is_valid is True
    assert len(res_valid.anomalies) == 0

    # Anomalous door: wide aspect ratio (e.g. h/w = 0.8) and tilted normal
    bad_door = ObjectGeometryEvidence(
        box_2d=(50.0, 50.0, 300.0, 150.0),  # w=250, h=100 -> ar = 0.4
        class_name="door",
        surface_normal=(0.0, 0.0, 0.95),  # vertical normal
    )
    res_bad = validate_door_geometry(bad_door)
    assert res_bad.is_valid is False
    assert len(res_bad.anomalies) >= 1

    # 2. Table validation
    table_top = ObjectGeometryEvidence(
        box_2d=(100.0, 100.0, 300.0, 140.0),
        class_name="table_top",
        surface_normal=(0.0, 0.0, 0.98),  # horizontal top plane
    )
    leg = ObjectGeometryEvidence(
        box_2d=(110.0, 145.0, 130.0, 280.0),  # leg below table top
        class_name="table_leg",
    )
    table_whole = ObjectGeometryEvidence(
        box_2d=(100.0, 100.0, 300.0, 280.0),
        class_name="table",
    )
    res_table = validate_table_geometry(table_whole, table_top_evidence=table_top, legs_evidence=[leg])
    assert res_table.is_valid is True

    # 3. Chair validation
    seat = ObjectGeometryEvidence(
        box_2d=(100.0, 180.0, 200.0, 220.0),
        class_name="chair_seat",
    )
    backrest = ObjectGeometryEvidence(
        box_2d=(100.0, 80.0, 200.0, 180.0),  # backrest above seat (lower y)
        class_name="chair_backrest",
    )
    chair_whole = ObjectGeometryEvidence(
        box_2d=(100.0, 80.0, 200.0, 280.0),
        class_name="chair",
    )
    res_chair = validate_chair_geometry(chair_whole, seat_evidence=seat, backrest_evidence=backrest)
    assert res_chair.is_valid is True

    # Inverted chair: backrest below seat
    bad_chair = validate_chair_geometry(chair_whole, seat_evidence=backrest, backrest_evidence=seat)
    assert len(bad_chair.anomalies) >= 1


def test_relationship_engine_and_attachment_semantics() -> None:
    engine = RelationshipEngine()

    wall = SpatialEntity3D(
        entity_id="wall_01",
        class_name="wall",
        position_m=(0.0, 2.0, 1.5),
        dimensions_m=(4.0, 0.2, 3.0),
        orientation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance_xyz=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        surface_normal=(0.0, 1.0, 0.0),
        is_physically_metric=True,
        scale_state="metric_anchored",
    )

    socket = SpatialEntity3D(
        entity_id="socket_01",
        class_name="socket",
        position_m=(0.2, 2.05, 0.4),  # on wall surface (horiz dist = 0.05m <= 0.15m)
        dimensions_m=(0.08, 0.04, 0.08),
        orientation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance_xyz=((0.001, 0.0, 0.0), (0.0, 0.001, 0.0), (0.0, 0.0, 0.001)),
        is_physically_metric=True,
        scale_state="metric_anchored",
    )

    table = SpatialEntity3D(
        entity_id="table_01",
        class_name="table",
        position_m=(0.0, 1.0, 0.4),
        dimensions_m=(1.2, 0.8, 0.75),
        orientation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance_xyz=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        is_physically_metric=True,
        scale_state="metric_anchored",
    )

    rels = engine.evaluate_relationships([wall, socket, table])

    # Socket on wall should be embedded_in and mounted_on
    socket_rels = [r for r in rels if r.source_id == "socket_01" and r.target_id == "wall_01"]
    assert any(r.relation_type == RelationshipType.EMBEDDED_IN for r in socket_rels)
    assert any(r.relation_type == RelationshipType.MOUNTED_ON for r in socket_rels)

    # Table is near socket (distance ~ 1.0m) -> adjacent_to, but NOT attached_to because table is not a wall-hosted fixture!
    table_to_wall = [r for r in rels if r.source_id == "table_01" and r.target_id == "wall_01"]
    assert not any(r.relation_type == RelationshipType.ATTACHED_TO for r in table_to_wall)

    # Test contradiction detector
    incompatible_rels = [
        EvidenceRelationship(
            source_id="box_a",
            target_id="box_b",
            relation_type=RelationshipType.ABOVE,
            confidence=0.9,
        ),
        EvidenceRelationship(
            source_id="box_a",
            target_id="box_b",
            relation_type=RelationshipType.BELOW,
            confidence=0.8,
        ),
    ]
    contradictions = engine.detect_contradictions(incompatible_rels)
    assert len(contradictions) == 1
    assert contradictions[0]["type"] == "incompatible_relation_pair"


def test_room_completeness_engine_profiles_and_evidence_states() -> None:
    completeness_engine = RoomCompletenessEngine()

    # Test category inference
    assert completeness_engine.infer_scene_category(["bed", "wardrobe", "lamp"]) == SceneCategory.BEDROOM
    assert completeness_engine.infer_scene_category(["toilet", "sink", "mirror"]) == SceneCategory.BATHROOM
    assert completeness_engine.infer_scene_category(["refrigerator", "oven", "cabinet"]) == SceneCategory.KITCHEN
    assert completeness_engine.infer_scene_category(["desk", "chair"]) == SceneCategory.OFFICE

    # Assess mock bedroom entities
    class MockEntity:
        def __init__(self, eid: str, cname: str, conf: float, is_mv: bool = False, missing: tuple[str, ...] = ()):
            self.canonical_entity_id = eid
            self.primary_class = cname
            self.confidence = conf
            self.is_multiview = is_mv
            self.member_observation_ids = ("obs1", "obs2") if is_mv else ("obs1",)
            self.attributes = {"missing_expected_parts": missing} if missing else {}

    entities = [
        MockEntity("wall_01", "wall", 0.95, is_mv=True),
        MockEntity("floor_01", "floor", 0.90, is_mv=True),
        MockEntity("door_01", "door", 0.85, is_mv=True, missing=("door_handle",)),
        MockEntity("bed_01", "bed", 0.88, is_mv=True),
        MockEntity("chair_01", "chair", 0.40, is_mv=False),  # single view, unresolved
    ]

    report = completeness_engine.assess_completeness(entities, forced_category=SceneCategory.BEDROOM, camera_view_count=2)
    assert report.scene_category == SceneCategory.BEDROOM
    assert report.coverage_score == 1.0  # wall, floor, door all present
    assert len(report.verified_entities) == 4
    assert len(report.unresolved_entities) == 1
    assert report.unresolved_entities[0].class_name == "chair"
    assert "door_01" in report.missing_expected_components
    assert report.missing_expected_components["door_01"] == ("door_handle",)

    # Test serialization
    r_dict = report.to_dict()
    assert r_dict["scene_category"] == "bedroom"
    assert r_dict["coverage_score"] == 1.0
    assert r_dict["verified_count"] == 4


def test_surface_material_probabilistic_inference_and_fusion() -> None:
    img = np.full((100, 100, 3), (139, 90, 43), dtype=np.uint8)  # wood brown

    attrs = extract_visual_attributes(img, box_2d=(10, 10, 90, 90), class_hint="table")
    assert attrs.dominant_color_name == "wood_brown"
    assert "wood" in attrs.material.value
    # Verify probabilistic distribution
    assert len(attrs.material_probabilities) >= 2
    assert attrs.material_probabilities[attrs.material.value] <= 0.95  # Never 100% certainty from RGB alone

    # Test multi-view fusion
    attrs_v1 = attrs
    attrs_v2 = EntityVisualAttributes(
        material=MaterialType.WOOD_POLISHED,
        material_confidence=0.75,
        material_probabilities={"wood_polished": 0.75, "plastic": 0.25},
        dominant_color_name="wood_brown",
        dominant_rgb=(0.55, 0.35, 0.17),
    )
    fused = fuse_multiview_attributes([attrs_v1, attrs_v2])
    assert "wood" in fused.material.value
    assert fused.confidence_basis == "multiview_fused_attribute_consensus"


def test_active_capture_and_model_capability_registry() -> None:
    # 1. Model Registry
    dino = ModelCapabilityRegistry.get_descriptor("grounding_dino_tiny")
    assert dino is not None
    assert dino.task_type == ModelTaskType.DETECTION
    assert dino.gpu_vram_budget_mb <= 2000.0

    # Sequential manager
    mgr = SequentialModelExecutionManager(max_vram_budget_mb=4000.0)
    mgr.register_active_model("dino", object())
    mgr.release_model("dino")
    assert "dino" not in mgr._active_models

    # 2. Active Capture Request
    req = CaptureRequest(
        request_id="cap_001",
        issue_type=CaptureIssueType.OCCLUDED_OPENING,
        priority=CapturePriority.HIGH,
        target_entity_ids=["door_01"],
        target_class="door",
        guidance_text="Door panel partially occluded by foreground object. Step left to reveal frame.",
        expected_evidence="Unoccluded door frame boundary",
        expected_uncertainty_reduction=0.35,
    )
    r_dict = req.to_dict()
    assert r_dict["issue_type"] == "occluded_opening"
    assert r_dict["expected_uncertainty_reduction"] == 0.35


def test_reconstruct_spatial_scene_integration_and_world_ir_bridge() -> None:
    cam = PinholeCamera(
        intrinsics=CameraIntrinsics(
            fx=800.0,
            fy=800.0,
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
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    img_rgb = np.full((480, 640, 3), 128, dtype=np.uint8)

    dets = [
        DetectionInput(
            detection_id="det_door",
            source_image_id="img_01",
            class_name="door",
            box_2d=(200.0, 100.0, 300.0, 380.0),
            confidence=0.90,
        ),
        DetectionInput(
            detection_id="det_table",
            source_image_id="img_01",
            class_name="table",
            box_2d=(320.0, 240.0, 500.0, 420.0),
            confidence=0.85,
        ),
    ]

    result = reconstruct_spatial_scene(
        detections=dets,
        cameras={"img_01": cam},
        depth_maps={"img_01": depth},
        images_rgb={"img_01": img_rgb},
    )

    assert result.scene_graph is not None
    assert len(result.scene_graph.nodes) >= 2
    assert result.completeness_report is not None
    assert len(result.geometry_validations) >= 2

    # Verify WorldIR export
    world = result.to_world_ir(world_id="test_world_sprint6")
    assert "scene_observation_graph" in world.metadata
    assert "room_completeness" in world.metadata
    assert world.metadata["scene_observation_graph"]["node_count"] >= 2
