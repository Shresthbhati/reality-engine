"""B07 / RE-PERC-000007: DINOv2 appearance embeddings into multi-view association."""
from __future__ import annotations

import numpy as np

from perception.instances.association import associate
from perception.instances.dinov2_embeddings import _extract_mock_embedding
from world_ir.observations import DepthSource, Observation


def test_two_crops_of_same_texture_have_high_cosine_unrelated_lower():
    """B07: two crops of the same textured patch -> cosine > 0.8; unrelated -> lower."""
    # Textured patch: horizontal gradient + edge
    patch1 = np.zeros((100, 100, 3), dtype=np.uint8)
    patch1[:, :, 0] = np.tile(np.linspace(20, 220, 100, dtype=np.uint8), (100, 1))
    patch1[40:60, :, 1] = 180

    # Nearly identical patch (slight perturbation)
    patch2 = patch1.copy()
    patch2[41:59, :, 1] = 175

    # Completely different texture (vertical stripes, blue)
    unrelated = np.zeros((100, 100, 3), dtype=np.uint8)
    unrelated[:, :, 2] = np.tile(np.linspace(10, 200, 100, dtype=np.uint8).reshape(-1, 1), (1, 100))

    emb1 = _extract_mock_embedding(patch1)
    emb2 = _extract_mock_embedding(patch2)
    emb_unrel = _extract_mock_embedding(unrelated)

    sim_same = float(np.dot(emb1, emb2))
    sim_diff = float(np.dot(emb1, emb_unrel))

    assert sim_same > 0.80, f"Expected high cosine similarity for similar patches, got {sim_same}"
    assert sim_diff < sim_same, f"Expected lower similarity for unrelated patch, got {sim_diff} vs {sim_same}"


def test_associate_uses_appearance_to_separate_different_objects():
    """RE-PERC-000007: Appearance embeddings passed to associate(appearance=...) gate incompatible items."""
    # Two observations close in space (0.05m apart) in different images
    o1 = Observation(
        "img1#0", "img1", "furniture/chair", (1.0, 1.0, 0.5), 0.1, DepthSource.METRIC
    )
    o2 = Observation(
        "img2#0", "img2", "furniture/chair", (1.05, 1.02, 0.5), 0.1, DepthSource.METRIC
    )

    # 1. Without appearance (geometry only): close positions merge into 1 entity
    res_geom = associate([o1, o2])
    assert len(res_geom.entities) == 1

    # 2. With incompatible appearance (cosine similarity = 0.0, below MIN_APPEARANCE_SIM_GATE 0.20):
    # Two distinct vectors (orthogonal unit vectors)
    v1 = np.zeros(384, dtype=float); v1[0] = 1.0
    v2 = np.zeros(384, dtype=float); v2[1] = 1.0

    app_diff = {"img1#0": v1, "img2#0": v2}
    res_diff = associate([o1, o2], appearance=app_diff)
    # The appearance gate rejects matching them, splitting into 2 entities!
    assert len(res_diff.entities) == 2

    # 3. With matching appearance (cosine similarity = 1.0):
    app_same = {"img1#0": v1, "img2#0": v1}
    res_same = associate([o1, o2], appearance=app_same)
    assert len(res_same.entities) == 1
    # Check that running appearance embedding on entity was updated
    assert res_same.entities[0].appearance_embedding is not None


def test_resolve_hypotheses_wires_appearance_into_associate():
    """Verify resolve_hypotheses passes appearance dictionary directly into associate."""
    from engine.math import Vec3
    from perception.instances.association import resolve_hypotheses
    from perception.instances.lifting import ObjectHypothesis3D
    from world_ir.observations import DepthSource

    h1 = ObjectHypothesis3D(
        region_id="r1", evidence_id="img1", label="furniture/chair",
        position=Vec3(1.0, 1.0, 0.5), bounds_min=Vec3(0.8, 0.8, 0.0), bounds_max=Vec3(1.2, 1.2, 1.0),
        point_count=100, mask_pixel_count=100, confidence=0.9,
    )
    h2 = ObjectHypothesis3D(
        region_id="r1", evidence_id="img2", label="furniture/chair",
        position=Vec3(1.05, 1.02, 0.5), bounds_min=Vec3(0.8, 0.8, 0.0), bounds_max=Vec3(1.2, 1.2, 1.0),
        point_count=100, mask_pixel_count=100, confidence=0.9,
    )

    v1 = np.zeros(384, dtype=float); v1[0] = 1.0
    v2 = np.zeros(384, dtype=float); v2[1] = 1.0
    appearance = {"img1|r1": v1, "img2|r1": v2}

    candidates, assoc = resolve_hypotheses(
        [h1, h2],
        DepthSource.MVS,
        camera_positions={"img1": (0, 0, 0), "img2": (1, 0, 0)},
        appearance=appearance,
    )
    # With distinct appearances, the gate prevents merge
    assert len(assoc.entities) == 2
    assert len(candidates) == 2

