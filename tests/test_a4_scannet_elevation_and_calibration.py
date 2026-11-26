"""A4.1, A4.3, A4.4 & A4.11 extended tests: ScanNet/SUN RGB-D/Replica adapters, elevation profiles, and calibration."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from perception.datasets.scannet_and_sunrgbd_adapters import (
    REPLICA_CLASS_MAP,
    SCANNET_NYU40_MAP,
    SUNRGBD_CLASS_MAP,
    RestrictedDatasetTermsError,
    load_replica_samples,
    load_scannet_samples,
    load_sunrgbd_samples,
)
from perception.intelligence.architectural_elevation import (
    ArchitecturalElevationReport,
    analyze_architectural_elevations,
)
from perception.intelligence.calibration_v2 import (
    CalibrationReport,
    apply_temperature,
    compute_calibration_report,
    fit_temperature_scaling,
)
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ScoreSemantics,
)


def _obs(
    obs_id: str,
    image_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.85,
    category: str = "architecture",
    width: int = 1000,
    height: int = 800,
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
            image_width=width,
            image_height=height,
        ),
        scores=ScoreSemantics(raw_score=confidence, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance(
                engine_id="test_engine",
                architecture_family="vlm",
                checkpoint_id="test_ckpt",
            ),
        ),
    )


def _valid_intrinsics() -> dict[str, float]:
    return {"fx": 577.5, "fy": 577.5, "cx": 319.5, "cy": 239.5}


def _identity_pose() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------------
# A4.1 ScanNet / ScanNet++ / SUN RGB-D / Replica ingestion & terms guards
# ---------------------------------------------------------------------------


def test_scannet_and_scannetpp_require_explicit_terms_acceptance(tmp_path: Path) -> None:
    manifest = tmp_path / "scannet.json"
    manifest.write_text(
        json.dumps([
            {
                "sample_id": "scene0000_00_0",
                "scene_id": "scene0000_00",
                "image_path": "scannet/scene0000_00/color/0.jpg",
                "depth_path": "scannet/scene0000_00/depth/0.png",
                "camera_intrinsics": _valid_intrinsics(),
                "camera_pose_4x4": _identity_pose(),
                "width": 1296,
                "height": 968,
                "objects": [
                    {"nyu40_label": "bookshelf", "bbox_xyxy": [100, 120, 340, 620]},
                    {"nyu40_label": "night stand", "bbox_xyxy": [420, 380, 580, 610]},
                ],
            }
        ]),
        encoding="utf-8",
    )

    with pytest.raises(RestrictedDatasetTermsError):
        load_scannet_samples(manifest, dataset_id="scannet", terms_accepted=False)

    with pytest.raises(RestrictedDatasetTermsError):
        load_scannet_samples(manifest, dataset_id="scannetpp", terms_accepted=False)

    samples = load_scannet_samples(manifest, dataset_id="scannet", terms_accepted=True)
    assert len(samples) == 1
    s = samples[0]
    assert s.dataset_id == "scannet"
    assert s.is_synthetic is False
    assert s.is_physically_surveyed is True
    assert s.camera_intrinsics == _valid_intrinsics()
    assert s.camera_pose_4x4 is not None
    assert len(s.boxes) == 2
    assert s.boxes[0]["canonical_label"] == "shelf"
    assert s.boxes[1]["raw_label"] == "nightstand"


def test_scannet_adapter_rejects_invalid_intrinsics_and_poses(tmp_path: Path) -> None:
    bad_intrinsics = tmp_path / "bad_intrinsics.json"
    bad_intrinsics.write_text(
        json.dumps([
            {
                "sample_id": "s1",
                "image_path": "s1.jpg",
                "camera_intrinsics": {"fx": -10.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
                "objects": [],
            }
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Focal length"):
        load_scannet_samples(bad_intrinsics, terms_accepted=True)

    bad_pose = tmp_path / "bad_pose.json"
    bad_pose.write_text(
        json.dumps([
            {
                "sample_id": "s2",
                "image_path": "s2.jpg",
                "camera_pose_4x4": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "objects": [],
            }
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="4x4"):
        load_scannet_samples(bad_pose, terms_accepted=True)


def test_sunrgbd_and_replica_adapters_preserve_real_vs_synthetic_provenance(tmp_path: Path) -> None:
    sun_path = tmp_path / "sunrgbd.json"
    sun_path.write_text(
        json.dumps([
            {
                "sample_id": "sun_001",
                "scene_id": "kv1_nyu_01",
                "image_path": "sunrgbd/sun_001.jpg",
                "depth_path": "sunrgbd/sun_001_depth.png",
                "width": 730,
                "height": 530,
                "objects": [
                    {"label": "dresser", "bbox_xyxy": [40, 140, 260, 420], "occluded": True},
                    {"label": "chair", "bbox_xyxy": [320, 220, 480, 470]},
                ],
            }
        ]),
        encoding="utf-8",
    )
    sun_samples = load_sunrgbd_samples(sun_path)
    assert len(sun_samples) == 1
    assert sun_samples[0].is_synthetic is False
    assert sun_samples[0].is_physically_surveyed is True
    assert sun_samples[0].boxes[0]["canonical_label"] == "cabinet"
    assert sun_samples[0].boxes[0]["is_occluded"] is True
    assert "toilet" in SUNRGBD_CLASS_MAP
    assert "wall" in SCANNET_NYU40_MAP

    rep_path = tmp_path / "replica.json"
    rep_path.write_text(
        json.dumps([
            {
                "sample_id": "rep_001",
                "scene_id": "office_0",
                "image_path": "replica/rep_001.png",
                "width": 640,
                "height": 480,
                "objects": [
                    {"label": "pillar", "bbox_xyxy": [120, 20, 180, 460]},
                    {"label": "tv-screen", "bbox_xyxy": [300, 140, 520, 280]},
                ],
            }
        ]),
        encoding="utf-8",
    )
    rep_samples = load_replica_samples(rep_path)
    assert len(rep_samples) == 1
    assert rep_samples[0].dataset_id == "replica"
    assert rep_samples[0].boxes[0]["canonical_label"] == "column"
    assert "stair" in REPLICA_CLASS_MAP


# ---------------------------------------------------------------------------
# A4.3 / A4.4 Architectural elevation profiles & room-envelope completeness
# ---------------------------------------------------------------------------


def test_architectural_elevations_order_walls_and_compute_opening_ratios() -> None:
    obs = [
        _obs("w_right", "cam_1", "wall", (500, 60, 1000, 660)),
        _obs("w_left", "cam_1", "wall", (0, 60, 500, 660)),
        _obs("door_1", "cam_1", "door", (120, 180, 260, 640)),
        _obs("win_1", "cam_1", "window", (620, 140, 860, 380)),
        _obs("floor_1", "cam_1", "floor", (0, 650, 1000, 800)),
        _obs("ceil_1", "cam_1", "ceiling", (0, 0, 1000, 65)),
        _obs("col_1", "cam_1", "column", (485, 60, 525, 660)),
    ]
    report = analyze_architectural_elevations(obs)
    assert isinstance(report, ArchitecturalElevationReport)
    assert report.ordered_wall_ids_by_image["cam_1"] == ("w_left", "w_right")
    assert report.summary["is_metric"] is False
    assert report.summary["wall_elevation_profiles"] == 2

    by_wall = {p.wall_id: p for p in report.profiles}
    assert "door_1" in by_wall["w_left"].contained_opening_ids
    assert "win_1" in by_wall["w_right"].contained_opening_ids
    assert by_wall["w_left"].opening_to_wall_area_ratio > 0.10
    assert by_wall["w_right"].sill_height_norm is not None
    assert by_wall["w_right"].lintel_height_norm is not None
    assert all(p.is_metric is False for p in report.profiles)

    env = report.envelopes[0]
    assert env.has_floor is True
    assert env.has_ceiling is True
    assert env.wall_count == 2
    assert env.opening_count == 2
    assert env.column_or_beam_count == 1
    assert env.completeness_score == pytest.approx(1.0, abs=1e-4)
    assert env.missing_elements == ()


def test_architectural_elevations_flags_missing_envelope_components() -> None:
    obs = [
        _obs("w_only", "cam_partial", "wall", (0, 80, 700, 600)),
    ]
    report = analyze_architectural_elevations(obs)
    env = report.envelopes[0]
    assert env.has_floor is False
    assert env.has_ceiling is False
    assert env.wall_count == 1
    assert env.completeness_score < 0.30
    assert set(env.missing_elements) == {"floor", "ceiling", "multi_wall_boundary", "openings"}


def test_architectural_elevations_handles_empty_observations() -> None:
    report = analyze_architectural_elevations([])
    assert report.profiles == ()
    assert report.envelopes == ()
    assert report.summary["images_analyzed"] == 0
    assert report.summary["mean_envelope_completeness"] == 0.0


# ---------------------------------------------------------------------------
# A4.11 Confidence calibration, ECE, MCE, Brier & strict honesty guard
# ---------------------------------------------------------------------------


def test_calibration_refuses_to_mark_calibrated_on_synthetic_or_unreviewed_data() -> None:
    scores = [0.90, 0.85, 0.70, 0.40, 0.30, 0.20]
    outcomes = [1, 1, 1, 0, 0, 0]

    assert fit_temperature_scaling(scores, outcomes, is_human_reviewed_real=False) is None

    rep_syn = compute_calibration_report(
        scores,
        outcomes,
        dataset_name="structured3d",
        is_human_reviewed_real=False,
    )
    assert isinstance(rep_syn, CalibrationReport)
    assert rep_syn.is_calibrated is False
    assert rep_syn.fitted_temperature is None
    assert "uncalibrated" in rep_syn.note
    assert 0.0 <= rep_syn.ece <= 1.0
    assert 0.0 <= rep_syn.mce <= 1.0
    assert 0.0 <= rep_syn.brier_score <= 1.0


def test_calibration_fits_temperature_and_improves_or_preserves_calibration_on_real_gt() -> None:
    # Overconfident raw scores
    raw_scores = [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.78, 0.75, 0.65, 0.55]
    outcomes = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]

    rep = compute_calibration_report(
        raw_scores,
        outcomes,
        dataset_name="human_reviewed_indoor_15",
        is_human_reviewed_real=True,
        n_bins=5,
        fit_temperature=True,
    )
    assert rep.is_calibrated is True
    assert rep.fitted_temperature is not None
    assert rep.fitted_temperature > 1.0
    assert rep.sample_count == 10
    assert len(rep.bins) == 5

    scaled = [apply_temperature(s, rep.fitted_temperature) for s in raw_scores]
    assert all(0.0 <= s <= 1.0 for s in scaled)
    # Temperature > 1 softens overconfident predictions toward 0.5
    assert scaled[0] < raw_scores[0]


def test_calibration_handles_empty_and_mismatched_inputs() -> None:
    empty_rep = compute_calibration_report(
        [],
        [],
        dataset_name="empty",
        is_human_reviewed_real=True,
    )
    assert empty_rep.sample_count == 0
    assert empty_rep.is_calibrated is False

    with pytest.raises(ValueError, match="equal length"):
        compute_calibration_report(
            [0.5, 0.8],
            [1],
            dataset_name="bad",
            is_human_reviewed_real=True,
        )


# ---------------------------------------------------------------------------
# A4.10 / A4.13 Scene Perception Audit Engine & Extended Subsystem Tests
# ---------------------------------------------------------------------------


def test_scene_perception_audit_complete_vs_incomplete_survey() -> None:
    from perception.intelligence.identity import CameraPoseConstraint
    from perception.intelligence.multiview import run_multiview_perception
    from perception.intelligence.scene_audit_v2 import (
        ScenePerceptionAuditReport,
        run_scene_perception_audit,
    )

    obs_complete = [
        _obs("w1", "cam_1", "wall", (0, 60, 500, 660), category="architecture"),
        _obs("w2", "cam_1", "wall", (500, 60, 1000, 660), category="architecture"),
        _obs("f1", "cam_1", "floor", (0, 650, 1000, 800), category="architecture"),
        _obs("c1", "cam_1", "ceiling", (0, 0, 1000, 65), category="architecture"),
        _obs("d1", "cam_1", "door", (120, 180, 260, 640), category="architecture"),
        _obs("t1", "cam_1", "table", (300, 400, 650, 640), category="furniture"),
        _obs("ch1", "cam_1", "chair", (240, 420, 340, 640), category="furniture"),
        _obs("w1_b", "cam_2", "wall", (0, 62, 500, 662), category="architecture"),
        _obs("w2_b", "cam_2", "wall", (500, 62, 1000, 662), category="architecture"),
        _obs("f1_b", "cam_2", "floor", (0, 652, 1000, 800), category="architecture"),
        _obs("c1_b", "cam_2", "ceiling", (0, 0, 1000, 68), category="architecture"),
        _obs("d1_b", "cam_2", "door", (125, 182, 265, 642), category="architecture"),
        _obs("ch1_b", "cam_2", "chair", (244, 422, 344, 642), category="furniture"),
    ]
    cameras = {
        "cam_1": CameraPoseConstraint(image_id="cam_1", position_m=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)),
        "cam_2": CameraPoseConstraint(image_id="cam_2", position_m=(0.15, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    }
    mv = run_multiview_perception(obs_complete, cameras=cameras, match_threshold=0.55)
    audit = run_scene_perception_audit(
        "audit_complete",
        obs_complete,
        multiview_report=mv,
        has_depth=True,
        has_human_reviewed_calibration=False,
    )
    assert isinstance(audit, ScenePerceptionAuditReport)
    assert audit.is_survey_ready is True
    assert audit.overall_readiness_score >= 0.80
    assert audit.image_count == 2
    assert audit.observation_count == len(obs_complete)
    assert audit.remediation_actions == ()
    serialized = audit.to_dict()
    assert serialized["is_survey_ready"] is True
    assert len(serialized["checklist"]) == 6

    # Incomplete single-image survey without furniture or openings
    obs_incomplete = [
        _obs("w_only", "cam_solo", "wall", (0, 60, 700, 640), category="architecture"),
    ]
    audit_inc = run_scene_perception_audit("audit_inc", obs_incomplete)
    assert audit_inc.is_survey_ready is False
    assert audit_inc.overall_readiness_score < 0.55
    assert len(audit_inc.remediation_actions) >= 3


def test_depth_aware_handles_partial_valid_mask_and_discontinuity() -> None:
    from perception.intelligence.depth_aware import DepthMapContract, associate_observations_with_depth

    depth_arr = np.full((800, 1000), 4.0, dtype=np.float32)
    depth_arr[300:600, 200:450] = 1.6
    valid_mask = np.ones((800, 1000), dtype=bool)
    valid_mask[300:450, 200:325] = False  # Partial missing depth inside box

    contract = DepthMapContract(
        image_id="cam_d",
        depth_array=depth_arr,
        scale_state="sfm_gauge",
        units="sfm_units",
        model_id="colmap_dense",
        valid_mask=valid_mask,
    )
    obs = [
        _obs("fg", "cam_d", "chair", (200, 300, 450, 600), category="furniture"),
        _obs("bg", "cam_d", "wall", (50, 50, 950, 650), category="architecture"),
    ]
    res = associate_observations_with_depth(obs, contract)
    assert res.scale_state == "sfm_gauge"
    assert res.units == "sfm_units"
    by_id = {o.observation_id: o for o in res.object_depths}
    assert 0.5 < by_id["fg"].valid_fraction < 1.0
    assert by_id["fg"].median_depth == pytest.approx(1.6, abs=0.05)
    assert by_id["fg"].boundary_discontinuity_score >= 0.0
    assert res.front_to_back_order[0] == "fg"


def test_fusion_v2_quality_degradation_scales_weight_and_preserves_explanation() -> None:
    from perception.intelligence.fusion_v2 import fuse_observations_v2

    obs = [
        _obs("o1", "img_q", "door", (100, 100, 260, 560), confidence=0.88),
        _obs("o2", "img_q", "door", (104, 104, 258, 558), confidence=0.84),
    ]
    good = fuse_observations_v2(obs, image_quality_grade="good")
    poor = fuse_observations_v2(obs, image_quality_grade="unusable_blank")
    assert len(good) == 1 and len(poor) == 1
    assert good[0].quality_weight > poor[0].quality_weight
    assert poor[0].propagated_uncertainty >= good[0].propagated_uncertainty
    d = good[0].to_dict()
    assert d["selected_label"] == "door"
    assert len(d["explanation"]) >= 1


def test_handoff_bundle_rejects_non_finite_and_degenerate_boxes() -> None:
    from perception.contracts.claude_handoff_v1 import HandoffObservation, HandoffValidationError

    with pytest.raises(HandoffValidationError, match="degenerate"):
        HandoffObservation(
            observation_id="bad_1",
            source_image_id="img_1",
            camera_id=None,
            image_width=800,
            image_height=600,
            coordinate_frame="image_px",
            bbox_px=(200.0, 200.0, 100.0, 400.0),
            mask_ref=None,
            semantic_class="chair",
            taxonomy_id="re:chair",
            component_hierarchy=(),
            parent_observation_id=None,
            relationship_hypotheses=(),
            depth_ref=None,
            cross_view_identity_id=None,
            raw_scores={"owlv2": 0.8},
            calibrated_probability=None,
            is_calibrated=False,
            uncertainty=0.2,
            provenance={},
            revision_number=1,
            lifecycle_status="active",
        )

    with pytest.raises(HandoffValidationError, match="calibrated_probability"):
        HandoffObservation(
            observation_id="bad_2",
            source_image_id="img_1",
            camera_id=None,
            image_width=800,
            image_height=600,
            coordinate_frame="image_px",
            bbox_px=(100.0, 100.0, 200.0, 400.0),
            mask_ref=None,
            semantic_class="chair",
            taxonomy_id="re:chair",
            component_hierarchy=(),
            parent_observation_id=None,
            relationship_hypotheses=(),
            depth_ref=None,
            cross_view_identity_id=None,
            raw_scores={"owlv2": 0.8},
            calibrated_probability=0.75,
            is_calibrated=False,
            uncertainty=0.2,
            provenance={},
            revision_number=1,
            lifecycle_status="active",
        )


def test_architectural_v2_empty_and_small_image_safety() -> None:
    from perception.intelligence.architectural_v2 import (
        analyze_architectural_geometry_v2,
        extract_image_line_segments,
    )

    tiny = np.zeros((8, 8, 3), dtype=np.uint8)
    assert extract_image_line_segments(tiny) == []

    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert extract_image_line_segments(blank) == []
    res = analyze_architectural_geometry_v2("blank_1", blank, [])
    assert res.line_segments == ()
    assert res.plane_hypotheses == ()
    assert res.builtin_cabinet_ids == ()


def test_calibration_and_audit_additional_edge_cases() -> None:
    from perception.intelligence.scene_audit_v2 import run_scene_perception_audit

    # Perfectly calibrated predictions should have near-zero ECE and Brier
    scores = [0.99, 0.99, 0.99, 0.99, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01]
    outcomes = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    rep = compute_calibration_report(
        scores,
        outcomes,
        dataset_name="ideal_real_gt",
        is_human_reviewed_real=True,
        n_bins=5,
        fit_temperature=True,
    )
    assert rep.is_calibrated is True
    assert rep.ece < 0.02
    assert rep.brier_score < 0.01

    # Audit with multi-image observations but missing depth & openings
    obs = [
        _obs("w1", "v1", "wall", (0, 50, 1000, 700), category="architecture"),
        _obs("w2", "v2", "wall", (0, 50, 1000, 700), category="architecture"),
        _obs("c1", "v1", "chair", (200, 350, 380, 680), category="furniture"),
        _obs("c2", "v2", "chair", (210, 355, 390, 685), category="furniture"),
    ]
    audit = run_scene_perception_audit(
        "audit_no_depth",
        obs,
        has_depth=False,
        has_human_reviewed_calibration=False,
    )
    assert audit.class_histogram == {"chair": 2, "wall": 2}
    depth_item = next(c for c in audit.checklist if c.check_id == "depth_association")
    assert depth_item.passed is False
    cal_item = next(c for c in audit.checklist if c.check_id == "calibration_honesty")
    assert cal_item.passed is True


def test_elevation_sorts_multiple_walls_and_ignores_non_contained_openings() -> None:
    obs = [
        _obs("w_c", "img_e", "wall", (650, 50, 980, 650)),
        _obs("w_a", "img_e", "wall", (20, 50, 320, 650)),
        _obs("w_b", "img_e", "wall", (330, 50, 640, 650)),
        # Window inside w_b only
        _obs("win_b", "img_e", "window", (400, 150, 560, 350)),
        # Opening outside all walls
        _obs("door_out", "img_e", "door", (0, 660, 100, 790)),
    ]
    rep = analyze_architectural_elevations(obs)
    assert rep.ordered_wall_ids_by_image["img_e"] == ("w_a", "w_b", "w_c")
    by_wall = {p.wall_id: p for p in rep.profiles}
    assert by_wall["w_a"].contained_opening_ids == ()
    assert by_wall["w_b"].contained_opening_ids == ("win_b",)
    assert by_wall["w_c"].contained_opening_ids == ()


def test_open_vocab_confusion_summary_and_unknown_fallback() -> None:
    from perception.intelligence.open_vocab import classify_open_vocabulary_label, track_class_confusions

    c_wall = classify_open_vocabulary_label("drywall partition wall")
    assert c_wall.canonical_label in ("wall", "drywall_partition_wall")

    c_unk = classify_open_vocabulary_label("   ")
    assert c_unk.is_known_class is False
    assert c_unk.canonical_label == "unknown_object"

    summary = track_class_confusions([
        ("chair", "stool"),
        ("stool", "chair"),
        ("door", "window"),
    ])
    assert summary["total_confusion_events"] == 3
    assert summary["top_confusions"][0]["pair"] == ["chair", "stool"]
    assert summary["top_confusions"][0]["count"] == 2


def test_ground_truth_empty_dataset_quality_and_disputed_penalty() -> None:
    from perception.datasets.ground_truth import (
        AnnotationProvenance,
        GroundTruthAnnotation,
        ReviewStatus,
        score_ground_truth_dataset_quality,
    )

    empty_q = score_ground_truth_dataset_quality([])
    assert empty_q["total_annotations"] == 0
    assert empty_q["quality_score"] == 0.0

    disp = GroundTruthAnnotation(
        annotation_id="d1",
        image_id="img_d",
        raw_label="chair",
        canonical_label="chair",
        bbox_px=(10.0, 10.0, 100.0, 100.0),
        image_width=640,
        image_height=480,
        provenance=AnnotationProvenance(
            annotator_id="ann_1",
            review_status=ReviewStatus.DISPUTED,
        ),
    )
    q = score_ground_truth_dataset_quality([disp])
    assert q["disputed_fraction"] == 1.0
    assert q["quality_score"] == 0.0
