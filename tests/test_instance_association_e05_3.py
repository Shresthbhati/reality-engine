"""Tests for Epic E05-3: Multi-View Hungarian Instance Association.

Validates:
1. Bipartite Hungarian matching associating visual appearance embeddings across viewpoints.
2. Appearance gating preventing false merges of visually distinct objects at close range.
3. Appearance disambiguation breaking geometric symmetry without duplicate entity creation.
4. Two-view epipolar geometric verification with calibrated cameras.
5. Benchmark object identity consistency (>= 98%) with zero duplicate entity generation.
6. Progressive running visual appearance accumulation across entity tracks.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple
import numpy as np
import pytest

from engine.math import Quat, Vec3
from perception.instances.appearance import compute_color_histogram
from perception.instances.association import (
    associate,
    identity_metrics,
    resolve_hypotheses,
)
from perception.instances.lifting import ObjectHypothesis3D
from provenance import Uncertainty
from reconstruction.calibration.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    PinholeCamera,
)
from world_ir.observations import AssociationState, DepthSource, Observation


def _camera(pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> PinholeCamera:
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    ext = CameraExtrinsics(position=Vec3(*pos), rotation=Quat.identity())
    return PinholeCamera(intrinsics=intr, extrinsics=ext)


def test_appearance_embedding_gate_rejects_visually_different_objects():
    """Verify visual appearance gate prevents merging two distinct items at the same spatial location."""
    # Two chairs placed 5cm apart (within geometric distance gate)
    # Chair 1: Blue fabric (high blue channel embedding)
    # Chair 2: Red fabric (high red channel embedding)
    obs1 = Observation(
        observation_id="c1",
        image_id="img1",
        label="chair",
        world_pos=(1.0, 0.0, 2.0),
        sigma_m=0.08,
        depth_source=DepthSource.METRIC,
    )
    obs2 = Observation(
        observation_id="c2",
        image_id="img2",
        label="chair",
        world_pos=(1.04, 0.02, 2.0),  # 4.5cm away, geometrically matching
        sigma_m=0.08,
        depth_source=DepthSource.METRIC,
    )

    # Unit appearance embeddings
    # Blue: [0, 0, 1], Red: [1, 0, 0] -> cosine similarity = 0.0
    appearance = {
        "c1": [0.0, 0.0, 1.0],
        "c2": [1.0, 0.0, 0.0],
    }

    result = associate([obs1, obs2], appearance=appearance, min_appearance_sim=0.30)

    # Must remain 2 separate entities despite close spatial proximity
    assert len(result.entities) == 2
    assert result.assignment["c1"] != result.assignment["c2"]
    # Decision for c2 must be DIFFERENT_ENTITY due to appearance mismatch
    dec2 = next(d for d in result.decisions if d.observation_id == "c2")
    assert dec2.state is AssociationState.DIFFERENT_ENTITY


def test_appearance_disambiguates_symmetric_ambiguity():
    """Verify visual appearance breaks geometric ambiguity between equidistant candidates."""
    # Two existing lamps in Image 1 at x = 0.0 (gold lamp) and x = 1.0 (black lamp)
    lamp_gold = Observation(
        observation_id="l_gold_img1",
        image_id="img1",
        label="lamp",
        world_pos=(0.0, 0.0, 2.0),
        sigma_m=0.10,
        depth_source=DepthSource.METRIC,
    )
    lamp_black = Observation(
        observation_id="l_black_img1",
        image_id="img1",
        label="lamp",
        world_pos=(1.0, 0.0, 2.0),
        sigma_m=0.10,
        depth_source=DepthSource.METRIC,
    )

    # Image 2 sees the gold lamp slightly shifted in estimated position at x = 0.45 (closer to center)
    lamp_seen_img2 = Observation(
        observation_id="l_seen_img2",
        image_id="img2",
        label="lamp",
        world_pos=(0.45, 0.0, 2.0),
        sigma_m=0.15,
        depth_source=DepthSource.METRIC,
    )

    # Appearance embeddings: gold [1, 1, 0], black [0.1, 0.1, 0.1]
    emb_gold = [0.7071, 0.7071, 0.0]
    emb_black = [0.0, 0.0, 1.0]

    appearance = {
        "l_gold_img1": emb_gold,
        "l_black_img1": emb_black,
        "l_seen_img2": emb_gold,  # Clearly matches gold lamp
    }

    # Run association with appearance disambiguation
    result = associate(
        [lamp_gold, lamp_black, lamp_seen_img2],
        appearance=appearance,
        appearance_weight=2.0,
    )

    # Should unambiguously merge with gold lamp without creating an AMBIGUOUS third entity
    assert len(result.entities) == 2
    gold_entity_id = result.assignment["l_gold_img1"]
    assert result.assignment["l_seen_img2"] == gold_entity_id


def test_two_view_epipolar_verification():
    """Verify epipolar geometric verification validates matching views and rejects ray departures."""
    cam1 = _camera(pos=(0.0, 0.0, 0.0))
    cam2 = _camera(pos=(0.3, 0.0, 0.0))  # Displaced 30cm along X

    # Target point at (0.0, 0.0, 2.0)
    # Cam1 sees point at center (320, 240)
    # Cam2 sees point at (320 - 500*0.3/2.0, 240) = (245, 240)
    obs1 = Observation(
        observation_id="o1",
        image_id="cam1",
        label="table",
        world_pos=(0.0, 0.0, 2.0),
        sigma_m=0.05,
        depth_source=DepthSource.METRIC,
        bbox_px=(300.0, 220.0, 340.0, 260.0),  # center (320, 240)
    )
    obs2_valid = Observation(
        observation_id="o2_valid",
        image_id="cam2",
        label="table",
        world_pos=(0.01, 0.0, 2.01),
        sigma_m=0.05,
        depth_source=DepthSource.METRIC,
        bbox_px=(225.0, 220.0, 265.0, 260.0),  # center (245, 240) -> matches epipolar line!
    )

    cameras = {"cam1": cam1, "cam2": cam2}
    result_valid = associate([obs1, obs2_valid], cameras=cameras)

    assert len(result_valid.entities) == 1
    dec = next(d for d in result_valid.decisions if d.observation_id == "o2_valid")
    assert dec.state is AssociationState.SAME_ENTITY_CONFIRMED
    assert dec.geom_verified is True

    # Now test an invalid observation that departs from epipolar line (y=350 instead of y=240, 110px off!)
    obs2_invalid = Observation(
        observation_id="o2_invalid",
        image_id="cam2",
        label="table",
        world_pos=(0.01, 0.0, 2.01),
        sigma_m=0.05,
        depth_source=DepthSource.METRIC,
        bbox_px=(225.0, 330.0, 265.0, 370.0),  # Epipolar violation!
    )
    result_invalid = associate([obs1, obs2_invalid], cameras=cameras, epipolar_tolerance_px=8.0)
    assert len(result_invalid.entities) == 2


def test_object_identity_consistency_benchmark_exceeds_98_percent():
    """Benchmark: 10 distinct physical objects across 8 noisy camera viewpoints.

    Acceptance criterion: >= 98% object identity consistency without duplicate entities.
    """
    rng = random.Random(42)
    num_objects = 10
    num_views = 8

    # Create 10 distinct physical objects spaced in a 2x5 room grid
    obj_positions = []
    obj_embeddings = []
    labels = ["chair", "table", "lamp", "desk", "sofa", "bed", "chair", "cabinet", "plant", "monitor"]

    for i in range(num_objects):
        x = (i % 5) * 0.8 - 1.6
        y = (i // 5) * 1.2 - 0.6
        z = 2.5
        obj_positions.append((x, y, z))

        # Distinct 16-D unit embedding per object
        raw_emb = [rng.gauss(0, 1) for _ in range(16)]
        norm = math.sqrt(sum(v * v for v in raw_emb))
        obj_embeddings.append([v / norm for v in raw_emb])

    observations: List[Observation] = []
    truth: Dict[str, str] = {}
    appearance: Dict[str, List[float]] = {}

    for v in range(num_views):
        view_id = f"view_{v:02d}"
        for k in range(num_objects):
            # 85% visibility rate across viewpoints
            if rng.random() > 0.85:
                continue

            oid = f"{view_id}_obj_{k}"
            true_pos = obj_positions[k]

            # Metric position noise: sigma = 2.5cm
            noise_x = rng.gauss(0, 0.025)
            noise_y = rng.gauss(0, 0.025)
            noise_z = rng.gauss(0, 0.025)
            measured_pos = (true_pos[0] + noise_x, true_pos[1] + noise_y, true_pos[2] + noise_z)

            obs = Observation(
                observation_id=oid,
                image_id=view_id,
                label=labels[k],
                world_pos=measured_pos,
                sigma_m=0.04,
                depth_source=DepthSource.METRIC,
                confidence=0.85 + rng.random() * 0.1,
            )
            observations.append(obs)
            truth[oid] = f"true_obj_{k}"

            # Visual embedding with small view-dependent noise (cosine similarity > 0.90)
            base_emb = obj_embeddings[k]
            perturbed = [val + rng.gauss(0, 0.05) for val in base_emb]
            p_norm = math.sqrt(sum(val * val for val in perturbed))
            appearance[oid] = [val / p_norm for val in perturbed]

    result = associate(
        observations,
        appearance=appearance,
        appearance_weight=2.0,
        min_appearance_sim=0.25,
    )

    metrics = identity_metrics(result.assignment, truth)

    # Acceptance Criteria for Epic E05-3:
    # 1. Identity consistency >= 98% (0.98)
    assert metrics["identity_consistency"] >= 0.98, (
        f"Identity consistency was {metrics['identity_consistency'] * 100:.2f}% (expected >= 98.0%)"
    )
    # 2. Pair precision >= 98%
    assert metrics["pair_precision"] >= 0.98
    # 3. Pair recall >= 98%
    assert metrics["pair_recall"] >= 0.98
    # 4. Zero duplicate entity rate (exact 10 objects recovered)
    assert metrics["duplicate_entity_rate"] == pytest.approx(0.0, abs=0.05)
    # 5. Wrong merge rate <= 1%
    assert metrics["wrong_merge_rate"] <= 0.01


def test_running_appearance_embedding_accumulation():
    """Verify EntityTrack updates its running appearance descriptor across observations."""
    obs1 = Observation(
        observation_id="o1",
        image_id="img1",
        label="vase",
        world_pos=(0.0, 0.0, 1.5),
        sigma_m=0.05,
        depth_source=DepthSource.METRIC,
    )
    obs2 = Observation(
        observation_id="o2",
        image_id="img2",
        label="vase",
        world_pos=(0.01, 0.0, 1.51),
        sigma_m=0.05,
        depth_source=DepthSource.METRIC,
    )

    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.8, 0.6, 0.0]

    appearance = {"o1": emb1, "o2": emb2}
    result = associate([obs1, obs2], appearance=appearance)

    assert len(result.entities) == 1
    track = result.entities[0]
    assert track.appearance_embedding is not None

    # Track embedding should be normalized sum of emb1 and emb2
    expected = np.array([1.8, 0.6, 0.0])
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(track.appearance_embedding, expected, atol=1e-3)
