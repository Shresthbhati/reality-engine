"""A4.11, A4.12, A4.13 & End-to-End tests: benchmarks, Claude handoff contract, diagnostics 2.0, and pipeline_v2."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.perception.a4_accuracy_suite import (
    compare_benchmark_regression,
    compute_identity_idf1,
    compute_mask_iou_and_boundary_f1,
    compute_relationship_f1,
    run_a4_comprehensive_benchmark,
)
from benchmarks.perception.evaluator import EvaluationMode
from benchmarks.perception.indoor_dataset_adapter import GroundTruthBox
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.contracts.claude_handoff_v1 import (
    HANDOFF_SCHEMA_VERSION,
    HandoffBundle,
    HandoffValidationError,
)
from perception.datasets import DatasetIntegrityReport
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    CrossViewIdentityHypothesis,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ScoreSemantics,
    SpatialRelationshipObservation,
)
from perception.intelligence.depth_aware import DepthMapContract, associate_observations_with_depth
from perception.intelligence.diagnostics_v2 import run_production_diagnostics_v2
from perception.intelligence.identity import CameraPoseConstraint
from perception.intelligence.pipeline_v2 import A4PerceptionPipelineResult, run_a4_perception_pipeline


def _obs(
    obs_id: str,
    image_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.85,
    category: str = "furniture",
    engine_id: str = "owlv2",
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=obs_id,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"re:{label}",
        category=category,
        bbox_px=bbox,
        coords=CoordinateSystemSpec(
            frame_id="image_px",
            units="pixels",
            image_width=800,
            image_height=600,
        ),
        scores=ScoreSemantics(raw_score=confidence, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance(
                engine_id=engine_id,
                architecture_family="vlm",
                checkpoint_id=f"{engine_id}_v1",
            ),
        ),
    )


def test_mask_iou_boundary_f1_idf1_and_relationship_f1() -> None:
    pred_mask = np.zeros((100, 100), dtype=bool)
    pred_mask[20:60, 20:60] = True
    gt_mask = np.zeros((100, 100), dtype=bool)
    gt_mask[20:60, 22:62] = True
    iou, bf1 = compute_mask_iou_and_boundary_f1(pred_mask, gt_mask)
    assert 0.85 <= iou <= 1.0
    assert 0.40 <= bf1 <= 1.0

    identities = [
        CrossViewIdentityHypothesis(
            identity_id="track_1",
            member_observation_ids=("o1", "o2"),
            canonical_label="chair",
            status="confirmed_match",
            identity_score=0.9,
        ),
        CrossViewIdentityHypothesis(
            identity_id="track_2",
            member_observation_ids=("o3",),
            canonical_label="table",
            status="singleton",
            identity_score=0.88,
        ),
    ]
    idf1 = compute_identity_idf1(
        identities,
        {"o1": "gt_a", "o2": "gt_a", "o3": "gt_b"},
    )
    assert idf1 == pytest.approx(1.0, abs=1e-6)

    pred_rels = [
        SpatialRelationshipObservation(
            relationship_id="r1",
            subject_id="door_1",
            predicate="door_contained_in_wall",
            object_id="wall_1",
            coordinate_frame="image_space",
            support_state="visually_observed",
            confidence_score=0.9,
        ),
        SpatialRelationshipObservation(
            relationship_id="r2",
            subject_id="chair_1",
            predicate="next_to",
            object_id="table_1",
            coordinate_frame="image_space",
            support_state="visually_observed",
            confidence_score=0.75,
        ),
    ]
    rel_f1 = compute_relationship_f1(
        pred_rels,
        [("door_1", "door_contained_in_wall", "wall_1"), ("win_1", "window_contained_in_wall", "wall_1")],
    )
    assert rel_f1 == pytest.approx(0.5, abs=1e-6)


def test_comprehensive_benchmark_enforces_honesty_for_unlabelled_and_synthetic() -> None:
    preds = [
        DetectionBox(box_id="p1", class_name="chair", confidence=0.90, box_2d=(100, 200, 220, 400), source_engine="owlv2"),
        DetectionBox(box_id="p2", class_name="wall", confidence=0.88, box_2d=(0, 20, 800, 480), source_engine="owlv2"),
    ]
    gts = [
        GroundTruthBox(image_id="img_1", raw_label="chair", canonical_label="chair", box_2d=(100, 200, 220, 400)),
        GroundTruthBox(image_id="img_1", raw_label="wall", canonical_label="wall", box_2d=(0, 20, 800, 480)),
    ]
    unlabelled = run_a4_comprehensive_benchmark(
        mode=EvaluationMode.UNLABELLED_REAL,
        dataset_name="south_building_unlabelled",
        predictions_by_image={"img_1": preds},
        gt_by_image=None,
    )
    assert unlabelled.accuracy_claim_allowed is False
    assert unlabelled.mode is EvaluationMode.UNLABELLED_REAL

    synthetic = run_a4_comprehensive_benchmark(
        mode=EvaluationMode.SYNTHETIC,
        dataset_name="structured3d_subset",
        predictions_by_image={"img_1": preds},
        gt_by_image={"img_1": gts},
    )
    assert synthetic.accuracy_claim_allowed is False
    assert synthetic.mode is EvaluationMode.SYNTHETIC

    labelled = run_a4_comprehensive_benchmark(
        mode=EvaluationMode.LABELLED_REAL,
        dataset_name="human_reviewed_indoor_15",
        predictions_by_image={"img_1": preds},
        gt_by_image={"img_1": gts},
    )
    assert labelled.accuracy_claim_allowed is True
    assert labelled.mode is EvaluationMode.LABELLED_REAL
    assert labelled.detection_map is not None and labelled.detection_map > 0.8

    reg = compare_benchmark_regression(labelled, labelled, max_allowed_drop=0.02)
    assert reg["passed"] is True


def test_claude_handoff_bundle_export_roundtrip_and_validation() -> None:
    obs = [
        _obs("w1", "img_1", "wall", (0, 20, 800, 480), category="architecture"),
        _obs("c1", "img_1", "chair", (150, 260, 280, 460)),
    ]
    pipeline_res = run_a4_perception_pipeline("run_handoff_test", obs)
    bundle = pipeline_res.handoff
    assert isinstance(bundle, HandoffBundle)
    assert bundle.schema_version == HANDOFF_SCHEMA_VERSION
    assert len(bundle.observations) == 2

    raw_json = bundle.to_json()
    reloaded = HandoffBundle.from_json(raw_json)
    assert reloaded.run_id == "run_handoff_test"
    assert len(reloaded.observations) == 2

    with pytest.raises(HandoffValidationError):
        HandoffBundle.from_json('{"schema_version": "wrong.v0", "run_id": "x"}')


def test_diagnostics_v2_surfaces_timeouts_vram_depth_corruption_and_fp_fn() -> None:
    obs = [
        _obs("o1", "img_1", "chair", (100, 200, 220, 400)),
        _obs("o_fp", "img_1", "sofa", (500, 300, 620, 450)),
    ]
    pipeline_res = run_a4_perception_pipeline("run_diag", obs)
    bad_integrity = DatasetIntegrityReport(
        dataset_id="scannet",
        total_samples=10,
        valid_samples=8,
        corrupt_samples=({"sample_id": "bad_1", "reason": "zero_dim"},),
        scene_count=2,
        leakage_detected=True,
        is_healthy=False,
    )
    empty_depth = associate_observations_with_depth(
        obs,
        DepthMapContract(
            image_id="img_1",
            depth_array=np.full((600, 800), np.nan, dtype=np.float32),
            scale_state="relative",
            units="relative",
            model_id="depth_anything_v2",
        ),
    )
    diag = run_production_diagnostics_v2(
        pipeline_res.bundle,
        depth_results=[empty_depth],
        dataset_integrity=bad_integrity,
        gt_by_image={"img_1": [GroundTruthBox(image_id="img_1", raw_label="chair", canonical_label="chair", box_2d=(100, 200, 220, 400))]},
        failed_engines={"sam2": "oom"},
        timed_out_engines=["grounding_dino"],
        peak_vram_mb=5950.0,
        vram_limit_mb=5800.0,
    )
    issue_codes = {f.code for f in diag.findings}
    assert "model_timeout" in issue_codes
    assert "gpu_memory_pressure" in issue_codes
    assert "missing_depth_association" in issue_codes
    assert "dataset_corruption_or_leakage" in issue_codes
    assert "ground_truth_fp_fn_summary" in issue_codes


def test_end_to_end_a4_perception_pipeline_integrates_all_subsystems() -> None:
    img = np.full((600, 800, 3), 210, dtype=np.uint8)
    img[60:480, 100:104] = 20
    img[60:480, 700:704] = 20
    img[480:484, 100:704] = 20

    depth_arr = np.full((600, 800), 3.0, dtype=np.float32)
    depth_arr[260:460, 180:300] = 1.4

    observations = [
        _obs("w_a", "cam_1", "wall", (0, 20, 800, 480), category="architecture"),
        _obs("d_a", "cam_1", "door", (120, 120, 220, 470), category="architecture"),
        _obs("c_a", "cam_1", "chair", (180, 260, 300, 460)),
        _obs("t_a", "cam_1", "table", (320, 280, 560, 470)),
        _obs("c_b", "cam_2", "chair", (185, 262, 305, 462)),
        _obs("w_b", "cam_2", "wall", (0, 22, 800, 482), category="architecture"),
    ]
    cameras = {
        "cam_1": CameraPoseConstraint(image_id="cam_1", position_m=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)),
        "cam_2": CameraPoseConstraint(image_id="cam_2", position_m=(0.2, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    }
    depth_contract = DepthMapContract(
        image_id="cam_1",
        depth_array=depth_arr,
        scale_state="relative",
        units="relative",
        model_id="depth_anything_v2",
    )

    result = run_a4_perception_pipeline(
        "e2e_run_01",
        observations,
        images_rgb={"cam_1": img},
        cameras=cameras,
        depth_contracts=[depth_contract],
        camera_ids_by_image={"cam_1": "cam_1", "cam_2": "cam_2"},
        depth_refs_by_image={"cam_1": "depth/cam_1.npy"},
    )
    assert isinstance(result, A4PerceptionPipelineResult)
    assert len(result.arch_geometry) == 1
    assert len(result.depth_results) == 1
    assert result.handoff.schema_version == HANDOFF_SCHEMA_VERSION
    assert result.active_refinement.recommended_next_step
    assert len(result.fusion_decisions) >= 4


def test_a4_accuracy_suite_edge_cases_and_healthy_diagnostics() -> None:
    empty_m = np.zeros((32, 32), dtype=bool)
    iou, bf1 = compute_mask_iou_and_boundary_f1(empty_m, empty_m)
    assert iou == 0.0 and bf1 == 0.0

    mismatched_a = np.ones((16, 16), dtype=bool)
    mismatched_b = np.ones((32, 32), dtype=bool)
    assert compute_mask_iou_and_boundary_f1(mismatched_a, mismatched_b) == (0.0, 0.0)

    assert compute_identity_idf1([], {"o1": "g1"}) is None
    assert compute_relationship_f1([], []) is None

    # Regression check catches a drop exceeding max_allowed_drop
    preds_hi = [
        DetectionBox(box_id="p1", class_name="chair", confidence=0.92, box_2d=(100, 200, 220, 400), source_engine="owlv2"),
    ]
    preds_lo = [
        DetectionBox(box_id="p1", class_name="sofa", confidence=0.92, box_2d=(100, 200, 220, 400), source_engine="owlv2"),
    ]
    gts = [
        GroundTruthBox(image_id="img_1", raw_label="chair", canonical_label="chair", box_2d=(100, 200, 220, 400)),
    ]
    base_b = run_a4_comprehensive_benchmark(
        mode=EvaluationMode.LABELLED_REAL,
        dataset_name="gt_base",
        predictions_by_image={"img_1": preds_hi},
        gt_by_image={"img_1": gts},
    )
    cand_b = run_a4_comprehensive_benchmark(
        mode=EvaluationMode.LABELLED_REAL,
        dataset_name="gt_cand",
        predictions_by_image={"img_1": preds_lo},
        gt_by_image={"img_1": gts},
    )
    reg = compare_benchmark_regression(base_b, cand_b, max_allowed_drop=0.05)
    assert reg["passed"] is False
    assert len(reg["regressions"]) >= 1

    # Healthy diagnostics check when no timeouts, VRAM pressure, or failures occur
    obs = [
        _obs("w1", "cam_1", "wall", (0, 20, 800, 480), category="architecture"),
        _obs("c1", "cam_1", "chair", (180, 260, 300, 460)),
    ]
    depth_arr = np.full((600, 800), 2.5, dtype=np.float32)
    dc = DepthMapContract(
        image_id="cam_1",
        depth_array=depth_arr,
        scale_state="relative",
        units="relative",
        model_id="depth_anything_v2",
    )
    res = run_a4_perception_pipeline("healthy_run", obs, depth_contracts=[dc])
    healthy_diag = run_production_diagnostics_v2(
        res.bundle,
        depth_results=res.depth_results,
        peak_vram_mb=2100.0,
        vram_limit_mb=5800.0,
    )
    codes = {f.code for f in healthy_diag.findings}
    assert "model_timeout" not in codes
    assert "gpu_memory_pressure" not in codes
    assert "missing_depth_association" not in codes


def test_pipeline_v2_and_scene_audit_integration_with_elevation() -> None:
    from perception.intelligence.architectural_elevation import analyze_architectural_elevations
    from perception.intelligence.scene_audit_v2 import run_scene_perception_audit

    obs = [
        _obs("w1", "v1", "wall", (0, 40, 400, 480), category="architecture"),
        _obs("w2", "v1", "wall", (400, 40, 800, 480), category="architecture"),
        _obs("f1", "v1", "floor", (0, 470, 800, 600), category="architecture"),
        _obs("c1", "v1", "ceiling", (0, 0, 800, 45), category="architecture"),
        _obs("d1", "v1", "door", (120, 120, 220, 470), category="architecture"),
        _obs("t1", "v1", "table", (320, 280, 560, 470)),
        _obs("ch1", "v1", "chair", (240, 290, 340, 470)),
        _obs("w1_v2", "v2", "wall", (0, 42, 400, 482), category="architecture"),
        _obs("w2_v2", "v2", "wall", (400, 42, 800, 482), category="architecture"),
        _obs("f1_v2", "v2", "floor", (0, 472, 800, 600), category="architecture"),
        _obs("c1_v2", "v2", "ceiling", (0, 0, 800, 46), category="architecture"),
        _obs("ch2", "v2", "chair", (244, 292, 344, 472)),
    ]
    res = run_a4_perception_pipeline("audit_integ_run", obs)
    elev = analyze_architectural_elevations(obs)
    audit = run_scene_perception_audit(
        "audit_integ_run",
        obs,
        elevation_report=elev,
        furniture_report=res.furniture_v2,
        multiview_report=res.multiview,
    )
    assert audit.is_survey_ready is True
    assert audit.overall_readiness_score >= 0.65
    assert elev.summary["wall_elevation_profiles"] == 4
