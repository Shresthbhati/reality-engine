"""Tests for the Universal 14-Stage Fusion Architecture and Pipeline."""

import tempfile
from pathlib import Path

from engine.multiengine.evidence_matrix import (
    EvidenceMatrixEntry,
    EvidenceMatrixStore,
)
from engine.multiengine.universal_fusion import (
    UniversalFusionPipeline,
    UniversalFusionStage,
    project_3d_to_pixel,
)
from world.hierarchy.spatial_hierarchy import SpatialHierarchyLevel


def test_14_stage_universal_fusion_execution() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = EvidenceMatrixStore(storage_root=Path(tmp_dir))

        art_1 = store.save_artifact("ev1", "pred.json", '{"confidence": 0.95}')
        entry_1 = EvidenceMatrixEntry(
            evidence_id="ev1",
            model_id="grounding_dino",
            architecture_id="dino",
            checkpoint_id="swint",
            run_id="run_01",
            source_ids=["img1.jpg"],
            input_parameters={"target_class": "chair", "label": "chair"},
            prediction_type="bounding_box_3d",
            raw_output_reference=art_1,
            coordinate_frame="canonical_world",
            entity_association="chair_main",
            confidence=0.95,
        )
        store.add_entry(entry_1)

        art_2 = store.save_artifact("ev2", "pred.json", '{"confidence": 0.89}')
        entry_2 = EvidenceMatrixEntry(
            evidence_id="ev2",
            model_id="sam2",
            architecture_id="sam",
            checkpoint_id="hiera_l",
            run_id="run_01",
            source_ids=["img1.jpg"],
            input_parameters={"target_class": "seat", "label": "seat"},
            prediction_type="segmentation_mask",
            raw_output_reference=art_2,
            coordinate_frame="canonical_world",
            entity_association="chair_main_seat",
            confidence=0.89,
        )
        store.add_entry(entry_2)

        pipeline = UniversalFusionPipeline(evidence_store=store)

        result = pipeline.run(
            session_id="test_sess_01",
            evidence_entries=[entry_1, entry_2],
            camera_metadata={"cam0": {"intrinsics": [500, 500, 320, 240]}},
            target_hierarchy_level=SpatialHierarchyLevel.ROOM,
        )

        assert result.session_id == "test_sess_01"
        assert result.success is True
        assert len(result.stage_results) == 14

        # Verify all 14 stages ran in order
        expected_stages = [
            UniversalFusionStage.STAGE_01_INPUT_CAMERA_VALIDATION,
            UniversalFusionStage.STAGE_02_COORDINATE_FRAME_ALIGNMENT,
            UniversalFusionStage.STAGE_03_CROSS_VIEW_CORRESPONDENCE,
            UniversalFusionStage.STAGE_04_ENTITY_IDENTITY_ASSOCIATION,
            UniversalFusionStage.STAGE_05_GEOMETRIC_CONSISTENCY,
            UniversalFusionStage.STAGE_06_REPROJECTION_CHECKING,
            UniversalFusionStage.STAGE_07_SEMANTIC_ONTOLOGY_MAPPING,
            UniversalFusionStage.STAGE_08_UNCERTAINTY_CALIBRATION,
            UniversalFusionStage.STAGE_09_CONFLICT_DETECTION,
            UniversalFusionStage.STAGE_10_CORRELATION_AWARE_FUSION,
            UniversalFusionStage.STAGE_11_CANONICAL_HIERARCHY_GENERATION,
            UniversalFusionStage.STAGE_12_WORLDIR_INVARIANT_VALIDATION,
            UniversalFusionStage.STAGE_13_QUALITY_GATES,
            UniversalFusionStage.STAGE_14_ACTIVE_CAPTURE_REQUESTS,
        ]
        actual_stages = [sr.stage for sr in result.stage_results]
        assert actual_stages == expected_stages
        assert all(sr.success for sr in result.stage_results)

        # Verify spatial hierarchy generation & part association
        tree = result.hierarchy_tree
        valid, errors = tree.validate_hierarchy()
        assert valid is True
        assert errors == []

        # seat should be recognized as part of chair
        seat_node = tree.get_node("chair_main_seat")
        assert seat_node is not None
        assert seat_node.level == SpatialHierarchyLevel.OBJECT_PART
        assert seat_node.parent_id == "chair_main"


def test_no_fabricated_geometry_when_unobserved() -> None:
    """Stage 5 must NOT fabricate default geometry when no 3D geometry is supplied."""
    entry = EvidenceMatrixEntry(
        evidence_id="ev_no_geom",
        model_id="grounding_dino",
        architecture_id="dino",
        checkpoint_id="swint",
        run_id="run_no_geom",
        source_ids=["img1.jpg"],
        input_parameters={"label": "lamp"},  # No box_3d
        prediction_type="bounding_box_2d",
        raw_output_reference="raw_ref",
        coordinate_frame="canonical_world",
        entity_association="lamp_01",
        confidence=0.88,
    )
    pipeline = UniversalFusionPipeline()
    result = pipeline.run(
        session_id="sess_no_geom",
        evidence_entries=[entry],
        camera_metadata={},
    )
    fused = result.fused_entities["lamp_01"]
    assert fused.bounding_box_3d is None  # Must remain explicitly None, NOT [0,0,0], [1,1,1]

    # Active capture planner must request 3D view for unobserved geometry
    unobserved_reqs = [r for r in result.capture_requests if "lacks 3D metric geometry" in r.rationale]
    assert len(unobserved_reqs) == 1
    assert unobserved_reqs[0].target_region == "lamp_01"


def test_actual_reprojection_pinhole_projection() -> None:
    """Stage 6 must calculate exact reprojection using pinhole camera equations."""
    # Camera with f=1000, cx=640, cy=360 at origin
    cam_meta = {
        "intrinsics": [1000.0, 1000.0, 640.0, 360.0],
        "extrinsics": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    # 3D point at (0.2, 0.1, 2.0)
    # Expected projection: u = 1000 * (0.2 / 2.0) + 640 = 740, v = 1000 * (0.1 / 2.0) + 360 = 410
    proj = project_3d_to_pixel([0.2, 0.1, 2.0], cam_meta)
    assert proj is not None
    assert abs(proj[0] - 740.0) < 1e-4
    assert abs(proj[1] - 410.0) < 1e-4

    # Point behind camera (Z = -1.0) must return None
    assert project_3d_to_pixel([0.0, 0.0, -1.0], cam_meta) is None

    # Test full stage reprojection evaluation
    entry = EvidenceMatrixEntry(
        evidence_id="ev_proj",
        model_id="colmap",
        architecture_id="colmap",
        checkpoint_id="v3.11",
        run_id="run_proj",
        source_ids=["cam0"],
        input_parameters={
            "label": "table",
            "box_3d": {"center": [0.2, 0.1, 2.0], "size": [1.0, 1.0, 0.8]},
            "center_2d": [741.0, 410.0],  # 1 px error in u
        },
        prediction_type="bounding_box_3d",
        raw_output_reference="raw_ref",
        coordinate_frame="canonical_world",
        entity_association="table_01",
        confidence=0.92,
    )
    pipeline = UniversalFusionPipeline()
    res = pipeline.run(
        session_id="sess_proj",
        evidence_entries=[entry],
        camera_metadata={"cam0": cam_meta},
    )
    stage6 = [s for s in res.stage_results if s.stage == UniversalFusionStage.STAGE_06_REPROJECTION_CHECKING][0]
    # Reprojection error should be exactly 1.0 px (from [740, 410] to [741, 410])
    mean_err = stage6.metrics["mean_reprojection_px"]
    assert mean_err is not None
    assert abs(mean_err - 1.0) < 1e-3


def test_unidentified_entity_never_defaults_to_chair() -> None:
    """Stage 7 must NOT default unknown/unidentified entities to chair."""
    entry = EvidenceMatrixEntry(
        evidence_id="ev_unknown",
        model_id="sensor",
        architecture_id="generic",
        checkpoint_id="v1",
        run_id="run_unk",
        source_ids=["img1.jpg"],
        input_parameters={},  # No label
        prediction_type="raw_points",
        raw_output_reference="raw_ref",
        coordinate_frame="canonical_world",
        entity_association="entity_unlabeled_01",
        confidence=0.5,
    )
    pipeline = UniversalFusionPipeline()
    result = pipeline.run(
        session_id="sess_unk",
        evidence_entries=[entry],
        camera_metadata={},
    )
    fused = result.fused_entities["entity_unlabeled_01"]
    assert fused.canonical_class == "unidentified"
    assert fused.canonical_class != "chair"


def test_correlation_discounting_weights_and_confidence_pooling() -> None:
    """Stage 8 and 10 must apply 1/sqrt(N) correlation discounting across model families."""
    # Create 4 entries from the same family 'dino' (each with conf 0.90)
    # and 1 entry from independent family 'florence' (with conf 0.80)
    entries = []
    for i in range(4):
        entries.append(
            EvidenceMatrixEntry(
                evidence_id=f"ev_dino_{i}",
                model_id=f"dino_v{i}",
                architecture_id="dino_family",
                checkpoint_id=f"ckpt_{i}",
                run_id="run_corr",
                source_ids=["img1.jpg"],
                input_parameters={"label": "refrigerator"},
                prediction_type="detection",
                raw_output_reference=f"raw_{i}",
                coordinate_frame="canonical_world",
                entity_association="fridge_01",
                confidence=0.90,
            )
        )
    entries.append(
        EvidenceMatrixEntry(
            evidence_id="ev_florence_0",
            model_id="florence2",
            architecture_id="florence_family",
            checkpoint_id="florence2_large",
            run_id="run_corr",
            source_ids=["img1.jpg"],
            input_parameters={"label": "refrigerator"},
            prediction_type="detection",
            raw_output_reference="raw_flo",
            coordinate_frame="canonical_world",
            entity_association="fridge_01",
            confidence=0.80,
        )
    )

    pipeline = UniversalFusionPipeline()
    result = pipeline.run(
        session_id="sess_corr",
        evidence_entries=entries,
        camera_metadata={},
    )
    stage8 = [s for s in result.stage_results if s.stage == UniversalFusionStage.STAGE_08_UNCERTAINTY_CALIBRATION][0]
    weights = stage8.metrics["weights"]
    # 4 dino entries -> weight = 1 / sqrt(4) = 0.5
    assert abs(weights["dino_family"] - 0.5) < 1e-4
    # 1 florence entry -> weight = 1 / sqrt(1) = 1.0
    assert abs(weights["florence_family"] - 1.0) < 1e-4

    fused = result.fused_entities["fridge_01"]
    # Pooled confidence: (4 * 0.5 * 0.90 + 1 * 1.0 * 0.80) / (4 * 0.5 + 1.0) = (1.8 + 0.8) / 3.0 = 2.6 / 3.0 = 0.8667
    assert fused.confidence is not None
    assert abs(fused.confidence - 0.8667) < 1e-3


def test_quality_gates_identifies_underlying_measurements() -> None:
    """Stage 13 must report actual underlying measurements and honest unmeasured status."""
    entry = EvidenceMatrixEntry(
        evidence_id="ev_gate",
        model_id="m1",
        architecture_id="fam1",
        checkpoint_id="c1",
        run_id="run_gate",
        source_ids=["img1.jpg"],
        input_parameters={"label": "sofa"},
        prediction_type="detection",
        raw_output_reference="raw_ref",
        coordinate_frame="canonical_world",
        entity_association="sofa_01",
        confidence=0.85,
    )
    pipeline = UniversalFusionPipeline()
    result = pipeline.run(
        session_id="sess_gate",
        evidence_entries=[entry],
        camera_metadata={},  # No cameras -> reprojection unmeasured
    )
    stage13 = [s for s in result.stage_results if s.stage == UniversalFusionStage.STAGE_13_QUALITY_GATES][0]
    metrics = stage13.metrics
    assert metrics["fidelity_score"] == 0.85
    assert metrics["measured_mean_reprojection_px"] is None
    assert metrics["reprojection_gate_passed"] is False
    assert metrics["quality_gate_passed"] is False  # Cannot pass when reprojection is unmeasured
