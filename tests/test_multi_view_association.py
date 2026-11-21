"""Tests for Multi-View Physical Identity Association and Geometric Reconciliation.

Validates:
1. Strict refusal of same_image_conflict (two detections from the same photo cannot merge).
2. Proper cross-view merging of identical physical entities across different camera views.
3. Strict refusal of label_conflict for incompatible semantic classes.
4. Correct decision ledger recording and summary telemetry.
5. Geometric reconciliation: weighted centroid, median dimensions, and evidence-calibrated confidence.
"""

from __future__ import annotations

from engine.math import Quat, Vec3
from engine.multiengine.association import MERGED, REFUSED
from engine.multiengine.multi_view_association import (
    MultiViewAssociator,
    MultiViewEntityObservation,
    compute_cross_view_pair_score,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


def _make_camera(pos: tuple[float, float, float], rot: Quat = Quat.identity()) -> PinholeCamera:
    return PinholeCamera(
        intrinsics=CameraIntrinsics(fx=800.0, fy=800.0, cx=512.0, cy=384.0, width=1024, height=768),
        extrinsics=CameraExtrinsics(position=Vec3(pos[0], pos[1], pos[2]), rotation=rot),
    )


def test_same_image_detections_never_merge() -> None:
    """Two detections in the same photograph MUST NEVER merge into the same physical entity."""
    cam = _make_camera((0.0, 0.0, 0.0))
    # Two windows detected in photo_01 at slightly different coordinates
    obs1 = MultiViewEntityObservation(
        observation_id="obs_photo_01_win_01",
        source_id="photo_01.jpg",
        class_name="window",
        position=(1.0, 0.5, 5.0),
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.85,
        box_2d=[100.0, 100.0, 200.0, 250.0],
        camera=cam,
    )
    obs2 = MultiViewEntityObservation(
        observation_id="obs_photo_01_win_02",
        source_id="photo_01.jpg",
        class_name="window",
        position=(1.1, 0.5, 5.0),  # Very close in 3D!
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.88,
        box_2d=[220.0, 100.0, 320.0, 250.0],
        camera=cam,
    )

    associator = MultiViewAssociator()
    entities, summary, decisions = associator.associate_observations([obs1, obs2])

    # Must produce 2 distinct physical entities, NOT 1
    assert len(entities) == 2
    assert summary.unique_physical_entities_count == 2
    assert summary.multiview_merged_entities_count == 0
    assert summary.single_view_entities_count == 2

    # Check pair score returns 0 with same_image_conflict
    ps = compute_cross_view_pair_score(obs1, obs2)
    assert ps is not None
    assert ps.score == 0.0
    assert ps.basis.get("reason") == "same_image_conflict"


def test_cross_view_matching_merges_same_physical_object() -> None:
    """Observations of the same physical window in photo_01 and photo_02 must merge."""
    cam1 = _make_camera((0.0, 0.0, 0.0))
    cam2 = _make_camera((1.0, 0.0, 0.0))  # Camera moved 1m along X

    win_world_pos = (0.5, 0.2, 6.0)

    # Project into cam1 and cam2 to get realistic 2D bounding boxes
    uv1 = cam1.project(Vec3(*win_world_pos))
    uv2 = cam2.project(Vec3(*win_world_pos))
    assert uv1 is not None and uv2 is not None

    obs1 = MultiViewEntityObservation(
        observation_id="obs_cam1_win",
        source_id="cam1.jpg",
        class_name="window",
        position=win_world_pos,
        dimensions=(1.2, 0.2, 1.6),
        confidence=0.80,
        box_2d=[uv1[0] - 30, uv1[1] - 40, uv1[0] + 30, uv1[1] + 40],
        camera=cam1,
    )
    obs2 = MultiViewEntityObservation(
        observation_id="obs_cam2_win",
        source_id="cam2.jpg",
        class_name="window",
        position=(win_world_pos[0] + 0.05, win_world_pos[1] - 0.02, win_world_pos[2] + 0.03),  # slight noise
        dimensions=(1.18, 0.22, 1.58),
        confidence=0.85,
        box_2d=[uv2[0] - 28, uv2[1] - 38, uv2[0] + 28, uv2[1] + 38],
        camera=cam2,
    )

    associator = MultiViewAssociator(max_distance_3d=2.0)
    entities, summary, decisions = associator.associate_observations([obs1, obs2])

    assert len(entities) == 1
    assert summary.unique_physical_entities_count == 1
    assert summary.multiview_merged_entities_count == 1
    assert summary.single_view_entities_count == 0

    merged_entity = entities[0]
    assert merged_entity.is_multiview
    assert len(merged_entity.sources) == 2
    assert "cam1.jpg" in merged_entity.sources
    assert "cam2.jpg" in merged_entity.sources
    assert merged_entity.observation_count == 2

    # Check evidence-calibrated confidence reflects multiple views
    assert merged_entity.calibrated_confidence > 0.5
    assert merged_entity.reprojection_residual_px is not None
    assert merged_entity.reprojection_residual_px < 5.0  # sub-pixel or tiny residual

    # Check decision ledger
    merged_decisions = [d for d in decisions if d.outcome == MERGED]
    assert len(merged_decisions) == 1
    assert merged_decisions[0].reason == "accepted"


def test_label_conflict_prevents_merge() -> None:
    """Observations with incompatible labels (e.g. tree vs window) must refuse to merge even if geometrically aligned."""
    cam1 = _make_camera((0.0, 0.0, 0.0))
    cam2 = _make_camera((0.5, 0.0, 0.0))

    pos_win = (1.0, 0.0, 5.0)
    uv1 = cam1.project(Vec3(*pos_win))
    uv2 = cam2.project(Vec3(*pos_win))
    assert uv1 is not None and uv2 is not None

    obs_win = MultiViewEntityObservation(
        observation_id="obs_win",
        source_id="photo_A.jpg",
        class_name="window",
        position=pos_win,
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.90,
        box_2d=[uv1[0] - 25, uv1[1] - 30, uv1[0] + 25, uv1[1] + 30],
        camera=cam1,
    )
    obs_tree = MultiViewEntityObservation(
        observation_id="obs_tree",
        source_id="photo_B.jpg",
        class_name="tree",
        position=pos_win,  # Same 3D location
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.85,
        box_2d=[uv2[0] - 25, uv2[1] - 30, uv2[0] + 25, uv2[1] + 30],
        camera=cam2,
    )

    associator = MultiViewAssociator()
    entities, summary, decisions = associator.associate_observations([obs_win, obs_tree])

    assert len(entities) == 2
    assert summary.unique_physical_entities_count == 2
    assert summary.multiview_merged_entities_count == 0

    refused_decisions = [d for d in decisions if d.outcome == REFUSED and d.reason == "label_conflict"]
    assert len(refused_decisions) >= 1


def test_excessive_3d_distance_prevents_merge() -> None:
    """Two observations of windows far apart in 3D (e.g. different windows on a facade) must not merge."""
    cam1 = _make_camera((0.0, 0.0, 0.0))
    cam2 = _make_camera((1.0, 0.0, 0.0))

    obs_win1 = MultiViewEntityObservation(
        observation_id="obs_win1",
        source_id="photo_A.jpg",
        class_name="window",
        position=(0.0, 0.0, 5.0),
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.90,
        camera=cam1,
    )
    obs_win2 = MultiViewEntityObservation(
        observation_id="obs_win2",
        source_id="photo_B.jpg",
        class_name="window",
        position=(10.0, 0.0, 5.0),  # 10m away!
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.90,
        camera=cam2,
    )

    associator = MultiViewAssociator(max_distance_3d=3.0)
    entities, summary, decisions = associator.associate_observations([obs_win1, obs_win2])

    assert len(entities) == 2
    assert summary.unique_physical_entities_count == 2
    assert summary.multiview_merged_entities_count == 0


def test_three_view_chain_merging() -> None:
    """An object seen across three camera views (cam1, cam2, cam3) merges into 1 entity with 3 views."""
    cam1 = _make_camera((0.0, 0.0, 0.0))
    cam2 = _make_camera((0.5, 0.0, 0.0))
    cam3 = _make_camera((1.0, 0.0, 0.0))

    door_pos = (0.2, 0.0, 4.0)
    uv1 = cam1.project(Vec3(*door_pos))
    uv2 = cam2.project(Vec3(*door_pos))
    uv3 = cam3.project(Vec3(*door_pos))
    assert uv1 and uv2 and uv3

    obs1 = MultiViewEntityObservation(
        observation_id="door_v1", source_id="p1.jpg", class_name="door",
        position=door_pos, dimensions=(1.0, 0.1, 2.1), confidence=0.75,
        box_2d=[uv1[0] - 20, uv1[1] - 40, uv1[0] + 20, uv1[1] + 40], camera=cam1,
    )
    obs2 = MultiViewEntityObservation(
        observation_id="door_v2", source_id="p2.jpg", class_name="door",
        position=(door_pos[0] + 0.02, door_pos[1], door_pos[2] + 0.01), dimensions=(1.02, 0.11, 2.12), confidence=0.80,
        box_2d=[uv2[0] - 20, uv2[1] - 40, uv2[0] + 20, uv2[1] + 40], camera=cam2,
    )
    obs3 = MultiViewEntityObservation(
        observation_id="door_v3", source_id="p3.jpg", class_name="door",
        position=(door_pos[0] - 0.01, door_pos[1] + 0.01, door_pos[2] - 0.01), dimensions=(0.98, 0.09, 2.08), confidence=0.85,
        box_2d=[uv3[0] - 20, uv3[1] - 40, uv3[0] + 20, uv3[1] + 40], camera=cam3,
    )

    associator = MultiViewAssociator()
    entities, summary, decisions = associator.associate_observations([obs1, obs2, obs3])

    assert len(entities) == 1
    assert summary.unique_physical_entities_count == 1
    assert summary.multiview_merged_entities_count == 1
    assert summary.max_views_per_entity == 3
    assert entities[0].observation_count == 3
    assert len(entities[0].sources) == 3


def test_empty_observations_returns_clean_empty_summary() -> None:
    associator = MultiViewAssociator()
    entities, summary, decisions = associator.associate_observations([])

    assert len(entities) == 0
    assert summary.total_input_observations == 0
    assert summary.unique_physical_entities_count == 0
    assert len(decisions) == 0


def test_missing_camera_does_not_invent_reprojection_score() -> None:
    """When camera calibration is missing, reprojection score is NOT fabricated as 0.5."""
    obs1 = MultiViewEntityObservation(
        observation_id="obs_no_cam_1",
        source_id="photo1.jpg",
        class_name="window",
        position=(1.0, 0.0, 5.0),
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.85,
        camera=None,  # Missing camera!
    )
    obs2 = MultiViewEntityObservation(
        observation_id="obs_no_cam_2",
        source_id="photo2.jpg",
        class_name="window",
        position=(1.05, 0.0, 5.02),
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.85,
        camera=None,  # Missing camera!
    )

    ps = compute_cross_view_pair_score(obs1, obs2)
    assert ps is not None
    assert not ps.basis["reprojection_available"]
    assert ps.basis["score_reproj"] is None
    assert ps.basis["mean_reproj_px"] is None
    # Score is purely based on 3D geometric distance without fabricated 0.5 reprojection
    assert ps.score > 0.8


def test_appearance_dissimilarity_prevents_merge() -> None:
    """Observations with incompatible appearance features refuse to merge even if close in 3D."""
    obs_red = MultiViewEntityObservation(
        observation_id="obs_red_door",
        source_id="photoA.jpg",
        class_name="door",
        position=(0.0, 0.0, 4.0),
        dimensions=(1.0, 0.1, 2.0),
        confidence=0.9,
        appearance_features=[1.0, 0.0, 0.0],  # Red color vector
    )
    obs_blue = MultiViewEntityObservation(
        observation_id="obs_blue_door",
        source_id="photoB.jpg",
        class_name="door",
        position=(0.05, 0.0, 4.02),
        dimensions=(1.0, 0.1, 2.0),
        confidence=0.9,
        appearance_features=[0.0, 0.0, 1.0],  # Blue color vector (orthogonal)
    )

    ps = compute_cross_view_pair_score(obs_red, obs_blue, min_appearance_sim=0.35)
    assert ps is not None
    assert ps.score == 0.0
    assert ps.basis.get("reason") == "appearance_dissimilarity"


def test_uncertainty_incompatible_prevents_merge() -> None:
    """Observations whose separation exceeds 3.5 sigma Mahalanobis distance refuse to merge."""
    obs1 = MultiViewEntityObservation(
        observation_id="obs_u1",
        source_id="photoA.jpg",
        class_name="window",
        position=(0.0, 0.0, 5.0),
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.9,
        position_uncertainty=(0.01, 0.01, 0.01),  # Highly precise (1 cm)
    )
    obs2 = MultiViewEntityObservation(
        observation_id="obs_u2",
        source_id="photoB.jpg",
        class_name="window",
        position=(0.15, 0.0, 5.0),  # 15 cm separation = 10 sigma away!
        dimensions=(1.0, 0.2, 1.5),
        confidence=0.9,
        position_uncertainty=(0.01, 0.01, 0.01),
    )

    ps = compute_cross_view_pair_score(obs1, obs2)
    assert ps is not None
    assert ps.score == 0.0
    assert ps.basis.get("reason") == "uncertainty_incompatible"


def test_unmeasured_confidence_preserved_without_fallback() -> None:
    """Missing detector confidence (None) remains None in ReconciledPhysicalEntity."""
    obs1 = MultiViewEntityObservation(
        observation_id="obs_none_conf_1",
        source_id="photo1.jpg",
        class_name="window",
        position=(1.0, 0.0, 5.0),
        dimensions=(1.0, 0.2, 1.5),
        confidence=None,  # Missing confidence!
    )
    obs2 = MultiViewEntityObservation(
        observation_id="obs_none_conf_2",
        source_id="photo2.jpg",
        class_name="window",
        position=(1.02, 0.0, 5.01),
        dimensions=(1.0, 0.2, 1.5),
        confidence=None,  # Missing confidence!
    )

    associator = MultiViewAssociator()
    entities, summary, decisions = associator.associate_observations([obs1, obs2])

    assert len(entities) == 1
    reconciled = entities[0]
    assert reconciled.mean_detector_confidence is None
    assert reconciled.calibrated_confidence is None
    assert reconciled.confidence_basis is not None and "UNMEASURED" in reconciled.confidence_basis


def test_evaluate_association_benchmark_metrics() -> None:
    """Evaluates multi-view identity benchmark metrics with precision, recall, F1, false merge and split rates."""
    from engine.multiengine.multi_view_association import evaluate_association_benchmark

    # Ground truth: obs1 and obs2 are the same window (win_01); obs3 is a different window (win_02)
    obs1 = MultiViewEntityObservation(
        observation_id="obs1", source_id="p1.jpg", class_name="window",
        position=(0.0, 0.0, 5.0), dimensions=(1.0, 0.2, 1.5), confidence=0.85,
    )
    obs2 = MultiViewEntityObservation(
        observation_id="obs2", source_id="p2.jpg", class_name="window",
        position=(0.02, 0.0, 5.01), dimensions=(1.0, 0.2, 1.5), confidence=0.88,
    )
    obs3 = MultiViewEntityObservation(
        observation_id="obs3", source_id="p2.jpg", class_name="window",
        position=(5.0, 0.0, 5.0), dimensions=(1.0, 0.2, 1.5), confidence=0.90,
    )

    associator = MultiViewAssociator()
    entities, summary, decisions = associator.associate_observations([obs1, obs2, obs3])

    gt_identities = {
        "obs1": "physical_win_01",
        "obs2": "physical_win_01",
        "obs3": "physical_win_02",
    }

    report = evaluate_association_benchmark(
        ground_truth_entities=gt_identities,
        reconciled_entities=entities,
        observations=[obs1, obs2, obs3],
    )

    assert report.total_pairs_evaluated > 0
    assert report.true_positives == 1  # (obs1, obs2) correctly merged
    assert report.false_positives == 0  # no false merges
    assert report.false_negatives == 0  # no false splits
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1_score == 1.0
    assert report.false_merge_rate == 0.0
    assert report.false_split_rate == 0.0
    assert "window" in report.per_class_metrics
    assert report.per_class_metrics["window"]["f1"] == 1.0


def test_evaluate_association_benchmark_empty_ground_truth_returns_not_evaluated() -> None:
    """When independent ground truth mapping is empty or unavailable, returns NOT_EVALUATED."""
    from engine.multiengine.multi_view_association import evaluate_association_benchmark

    obs1 = MultiViewEntityObservation(
        observation_id="o1", source_id="p1.jpg", class_name="door",
        position=(0.0, 0.0, 3.0), dimensions=(0.9, 0.1, 2.1), confidence=0.9,
    )
    associator = MultiViewAssociator()
    entities, _, _ = associator.associate_observations([obs1])

    report = evaluate_association_benchmark(
        ground_truth_entities={},
        reconciled_entities=entities,
        observations=[obs1],
    )
    assert report.evaluation_status == "NOT_EVALUATED"
    assert report.precision is None
    assert report.recall is None
    assert report.f1_score is None
    assert report.unlabelled_observations_count == 1


def test_evaluate_association_benchmark_empty_positive_set_returns_none() -> None:
    """When GT contains only negative pairs (distinct physical objects), precision and recall are not 1.0."""
    from engine.multiengine.multi_view_association import evaluate_association_benchmark

    obs1 = MultiViewEntityObservation(
        observation_id="o1", source_id="p1.jpg", class_name="door",
        position=(0.0, 0.0, 3.0), dimensions=(0.9, 0.1, 2.1), confidence=0.9,
    )
    obs2 = MultiViewEntityObservation(
        observation_id="o2", source_id="p2.jpg", class_name="door",
        position=(10.0, 0.0, 3.0), dimensions=(0.9, 0.1, 2.1), confidence=0.9,
    )
    associator = MultiViewAssociator()
    entities, _, _ = associator.associate_observations([obs1, obs2])

    gt_identities = {
        "o1": "phys_door_A",
        "o2": "phys_door_B",
    }
    report = evaluate_association_benchmark(
        ground_truth_entities=gt_identities,
        reconciled_entities=entities,
        observations=[obs1, obs2],
    )
    assert report.evaluation_status == "COMPLETED"
    assert report.true_positives == 0
    assert report.false_positives == 0
    assert report.false_negatives == 0
    assert report.precision is None
    assert report.recall is None


def test_multi_view_association_across_2_3_4_views() -> None:
    """Tests an object observed and associated across 2, 3, and 4 distinct camera views."""
    obs_views = [
        MultiViewEntityObservation(
            observation_id=f"obs_view_{i}",
            source_id=f"camera_{i}.jpg",
            class_name="table",
            position=(2.5 + 0.01 * i, 1.0, 2.0 - 0.01 * i),
            dimensions=(1.5, 0.8, 0.75),
            confidence=0.85 + 0.02 * i,
        )
        for i in range(4)
    ]

    associator = MultiViewAssociator()

    # 2 views
    ent_2, sum_2, _ = associator.associate_observations(obs_views[:2])
    assert len(ent_2) == 1
    assert ent_2[0].is_multiview
    assert len(ent_2[0].sources) == 2

    # 3 views
    ent_3, sum_3, _ = associator.associate_observations(obs_views[:3])
    assert len(ent_3) == 1
    assert ent_3[0].is_multiview
    assert len(ent_3[0].sources) == 3

    # 4 views
    ent_4, sum_4, decisions = associator.associate_observations(obs_views)
    assert len(ent_4) == 1
    assert ent_4[0].is_multiview
    assert len(ent_4[0].sources) == 4
    assert sum_4.max_views_per_entity == 4


def test_cross_view_ledger_records_all_refusal_reasons() -> None:
    """Verifies that non-matching pairs are recorded in the ledger with explicit refusal reasons."""
    obs_close_door = MultiViewEntityObservation(
        observation_id="door_1", source_id="p1.jpg", class_name="door",
        position=(0.0, 0.0, 3.0), dimensions=(0.9, 0.1, 2.1), confidence=0.9,
    )
    obs_far_door = MultiViewEntityObservation(
        observation_id="door_2", source_id="p2.jpg", class_name="door",
        position=(50.0, 0.0, 3.0), dimensions=(0.9, 0.1, 2.1), confidence=0.9,
    )
    obs_same_img = MultiViewEntityObservation(
        observation_id="door_3", source_id="p1.jpg", class_name="door",
        position=(0.1, 0.0, 3.0), dimensions=(0.9, 0.1, 2.1), confidence=0.9,
    )

    associator = MultiViewAssociator()
    _, summary, decisions = associator.associate_observations([obs_close_door, obs_far_door, obs_same_img])

    reasons = [d.reason for d in decisions if d.outcome == "refused"]
    assert "excessive_3d_distance" in reasons
    assert "same_image_conflict" in reasons
    assert summary.refused_pairs_count >= 2


def test_ray_cpa_triangulation_merges_cross_view_observations_with_scale_discrepancy() -> None:
    """Verifies that viewing-ray CPA triangulation successfully merges cross-view observations
    even when relative monocular depth caused unprojected 3D distance > 2m."""
    intr1 = CameraIntrinsics(fx=800.0, fy=800.0, cx=400.0, cy=300.0, width=800, height=600)
    intr2 = CameraIntrinsics(fx=800.0, fy=800.0, cx=400.0, cy=300.0, width=800, height=600)
    # Cam 1 at (-1, 0, 0), Cam 2 at (1, 0, 0)
    cam1 = PinholeCamera(intrinsics=intr1, extrinsics=CameraExtrinsics(Vec3(-1.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    cam2 = PinholeCamera(intrinsics=intr2, extrinsics=CameraExtrinsics(Vec3(1.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

    # Real physical object at (0, 0, 4)
    # Projection in Cam 1: X_cam = 0 - (-1) = 1, Z = 4 -> u = 800 * (1/4) + 400 = 600, v = 300
    # Projection in Cam 2: X_cam = 0 - 1 = -1, Z = 4 -> u = 800 * (-1/4) + 400 = 200, v = 300
    box1 = (550.0, 250.0, 650.0, 350.0)
    box2 = (150.0, 250.0, 250.0, 350.0)

    # Observations with unscaled monocular depth causing distance > 2.5m
    obs1 = MultiViewEntityObservation(
        observation_id="table_v1",
        source_id="v1.jpg",
        class_name="table",
        position=(0.0, 0.0, 2.5),  # lifted at 2.5m
        dimensions=(1.2, 0.8, 0.75),
        confidence=0.88,
        camera=cam1,
        box_2d=box1,
    )
    obs2 = MultiViewEntityObservation(
        observation_id="table_v2",
        source_id="v2.jpg",
        class_name="table",
        position=(0.0, 0.0, 5.5),  # lifted at 5.5m (3.0m difference in Z!)
        dimensions=(1.2, 0.8, 0.75),
        confidence=0.91,
        camera=cam2,
        box_2d=box2,
    )

    associator = MultiViewAssociator()
    entities, summary, decisions = associator.associate_observations([obs1, obs2])

    assert len(entities) == 1
    assert entities[0].is_multiview
    assert set(entities[0].sources) == {"v1.jpg", "v2.jpg"}
    assert summary.accepted_merges_count == 1
    # Check that consensus position is close to the true triangulated intersection at (0, 0, 4)
    pos = entities[0].position
    assert abs(pos[0] - 0.0) < 0.2
    assert abs(pos[1] - 0.0) < 0.2
    assert abs(pos[2] - 4.0) < 0.3


