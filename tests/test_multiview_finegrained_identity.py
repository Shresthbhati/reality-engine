"""Tests for Fine-Grained Multi-View Identity and Ground Truth Verification (Programmes 1 & 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from perception.fine_grained.multiview_associator import (
    MultiViewIdentityAssociator,
    VisualObservation3D,
)

ROOT = Path(__file__).resolve().parents[1]
GT_PATH = ROOT / "benchmarks" / "measured_rooms" / "room_01_living_studio" / "ground_truth.json"


def test_same_camera_exclusion_prevents_merging_adjacent_sockets():
    """Universal Spatial Intelligence Directive Rule:

    One physical object cannot appear twice in the same photo.
    Two duplex sockets separated by only 8cm on the same wall photographed together
    in cam_01 must NEVER merge into one entity, even if distance <= 15cm threshold.
    """
    associator = MultiViewIdentityAssociator(small_object_dist_thresh_m=0.15)

    # Duplex socket on the south wall: Socket 1 at 3.80m, Socket 2 at 3.88m (8cm apart)
    # Both observed in cam_01
    sock1 = VisualObservation3D(
        observation_id="obs_duplex_1",
        class_id="electrical.switchboard.module.socket",
        camera_id="cam_01",
        center_world=(3.80, 0.0, 0.30),
        normal_world=(0.0, 1.0, 0.0),
        dimensions_m=(0.086, 0.086, 0.02),
        confidence=0.85,
        host_surface_id="wall_south",
    )
    sock2 = VisualObservation3D(
        observation_id="obs_duplex_2",
        class_id="electrical.switchboard.module.socket",
        camera_id="cam_01",
        center_world=(3.88, 0.0, 0.30),  # 8cm apart, well within 15cm threshold!
        normal_world=(0.0, 1.0, 0.0),
        dimensions_m=(0.086, 0.086, 0.02),
        confidence=0.88,
        host_surface_id="wall_south",
    )

    clusters = associator.associate_observations([sock1, sock2])

    # MUST form 2 distinct entities because they share the same camera_id!
    assert len(clusters) == 2
    assert clusters[0].observation_count == 1
    assert clusters[1].observation_count == 1
    assert clusters[0].persistent_id != clusters[1].persistent_id


def test_four_views_of_single_socket_fuse_into_one_entity_with_confidence_boost():
    """Four photographs taken from different camera stations observing the same physical socket

    must fuse into exactly one persistent entity with multi-view confidence elevation.
    """
    associator = MultiViewIdentityAssociator(small_object_dist_thresh_m=0.15)

    # Socket on west wall at (0.0, 2.0, 0.3) observed from 4 different angles
    obs_list = [
        VisualObservation3D(
            observation_id=f"obs_west_sock_cam{i}",
            class_id="electrical.switchboard.module.socket",
            camera_id=f"cam_station_{i}",
            center_world=(0.005 * (i - 1.5), 2.0 + 0.004 * (i - 1), 0.30 + 0.002 * i),
            normal_world=(1.0, 0.0, 0.0),
            dimensions_m=(0.086, 0.086, 0.02),
            confidence=0.80,
            host_surface_id="wall_west",
        )
        for i in range(1, 5)
    ]

    clusters = associator.associate_observations(obs_list)

    assert len(clusters) == 1
    cl = clusters[0]
    assert cl.observation_count == 4
    assert len(cl.source_camera_ids) == 4
    # Multi-view confidence boost: 0.80 + 0.05 * 3 = 0.95
    assert cl.fused_confidence >= 0.95
    assert cl.spatial_variance_m > 0.0
    assert abs(cl.fused_center[1] - 2.0) < 0.02


def test_appearance_embeddings_gate_dissimilar_entities():
    """Even if two entities are within spatial distance, distinct visual embeddings prevent merging."""
    associator = MultiViewIdentityAssociator(small_object_dist_thresh_m=0.20, min_appearance_sim=0.30)

    # Synthetic DINOv2 style 4-dim unit embeddings:
    # White socket faceplate vs Dark metallic switch
    emb_white = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb_dark = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # orthogonal, sim = 0.0

    obs_white = VisualObservation3D(
        observation_id="obs_white",
        class_id="electrical.switchboard.module.socket",
        camera_id="cam_01",
        center_world=(1.0, 0.0, 0.3),
        normal_world=(0.0, 1.0, 0.0),
        dimensions_m=(0.086, 0.086, 0.02),
        confidence=0.9,
        feature_embedding=emb_white,
    )
    obs_dark = VisualObservation3D(
        observation_id="obs_dark",
        class_id="electrical.switchboard.module.socket",
        camera_id="cam_02",
        center_world=(1.05, 0.0, 0.3),  # 5cm apart, but completely different visual features!
        normal_world=(0.0, 1.0, 0.0),
        dimensions_m=(0.086, 0.086, 0.02),
        confidence=0.9,
        feature_embedding=emb_dark,
    )

    clusters = associator.associate_observations([obs_white, obs_dark])
    assert len(clusters) == 2


def test_real_room_ground_truth_fixtures_association_and_accuracy():
    """Verify multi-view association on simulated multi-camera observations of the ground truth fixtures

    recorded in room_01_living_studio (Bosch GLM 50-27 C ground truth).
    """
    assert GT_PATH.exists(), f"Ground truth fixture file missing: {GT_PATH}"
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_fixtures = gt.get("fixtures", [])
    assert len(gt_fixtures) == 5  # 1 switch near door, 4 sockets

    associator = MultiViewIdentityAssociator(small_object_dist_thresh_m=0.15)

    # Simulate 3 observations from 3 camera stations for each ground truth fixture
    simulated_obs = []
    for f in gt_fixtures:
        pos = f["position_m"]
        f_type = f.get("fixture_type", "socket")
        cls_id = f"electrical.switchboard.module.{f_type}"
        for cam_idx in range(1, 4):
            # Add small random noise +-5mm (well within Section 1 5cm tolerance)
            noisy_pos = (
                pos[0] + 0.003 * (cam_idx - 2),
                pos[1] + 0.002 * (cam_idx - 1),
                pos[2] + 0.001 * cam_idx,
            )
            simulated_obs.append(
                VisualObservation3D(
                    observation_id=f"{f['id']}_cam{cam_idx}",
                    class_id=cls_id,
                    camera_id=f"cam_station_{cam_idx}",
                    center_world=noisy_pos,
                    normal_world=(0.0, 1.0, 0.0),
                    dimensions_m=(0.086, 0.086, 0.02),
                    confidence=0.88,
                )
            )

    clusters = associator.associate_observations(simulated_obs)

    # Exactly 5 persistent clusters must be formed (one per ground truth fixture)
    assert len(clusters) == 5

    # Each cluster should have exactly 3 observations from the 3 cameras
    for cl in clusters:
        assert cl.observation_count == 3
        assert len(cl.source_camera_ids) == 3

    # Every ground truth fixture should match a cluster within 2cm
    for f in gt_fixtures:
        gt_p = np.array(f["position_m"])
        best_dist = min(np.linalg.norm(np.array(cl.fused_center) - gt_p) for cl in clusters)
        assert best_dist < 0.02, f"Fixture {f['id']} error {best_dist:.4f}m exceeds 2cm!"
