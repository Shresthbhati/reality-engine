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
