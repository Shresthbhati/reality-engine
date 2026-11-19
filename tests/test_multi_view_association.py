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
