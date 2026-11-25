"""Open-vocabulary architectural perception, real photographic, and multi-res RTX 4050 benchmarks."""

from __future__ import annotations

from pathlib import Path
import pytest

from perception.availability import real_models_by_name
from perception.scene_intelligence import (
    RelationshipSupport,
    RelationshipType,
    bridge_scene_to_world,
    extract_candidate_relationships,
    run_sequential_perception_pipeline,
)
from perception.world_bridge import build_visual_observations_from_detections
from world_ir.observations import DepthSource
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


def test_cross_model_disagreement_part_of_and_blocking_relationships():
    obs_a = build_visual_observations_from_detections(
        source_evidence_id="ev1",
        source_image_id="img1",
        model_id="maskrcnn",
        model_version="1.0",
        depth_source=DepthSource.RELATIVE,
        detections=[
            {"label": "chair", "bbox_px": [100, 100, 250, 260], "score": 0.85, "world_pos": (0.0, 0.0, 0.0)},
        ],
    )
    obs_b = build_visual_observations_from_detections(
        source_evidence_id="ev1",
        source_image_id="img1",
        model_id="owlvit",
        model_version="1.0",
        depth_source=DepthSource.RELATIVE,
        detections=[
            {"label": "armchair", "bbox_px": [105, 105, 245, 255], "score": 0.42},
            {"label": "wall", "bbox_px": [0, 0, 600, 400], "score": 0.55},
            {"label": "door", "bbox_px": [200, 40, 320, 360], "score": 0.48},
        ],
    )
    rels = extract_candidate_relationships(obs_a + obs_b)
    assert any(r.rel_type is RelationshipType.OVERLAPPING and r.support_state is RelationshipSupport.CONTRADICTORY for r in rels)
    assert any(r.rel_type is RelationshipType.PART_OF and r.coordinate_frame == "image_space" for r in rels)


def test_real_colmap_photos_multi_resolution_rtx4050_and_worldstore(tmp_path):
    avail = real_models_by_name()
    if (
        avail["depth_anything"].status != "available"
        or avail["mask_rcnn"].status != "available"
        or avail["owlvit"].status != "available"
    ):
        pytest.skip("Required real cached models unavailable")

    session_dir = Path("data/worldstore/colmap-sessions/wld_1a3fc79fe10f/current/images")
    if not session_dir.is_dir():
        pytest.skip("Real photographic COLMAP session fixture not present")
    photos = sorted(str(p) for p in session_dir.glob("*.jpg"))[:3]
    if len(photos) < 2:
        pytest.skip("Fewer than 2 real photos in fixture")

    res_640 = run_sequential_perception_pipeline(photos, max_edge=640, use_owlvit=True, extract_materials=True)
    res_1024 = run_sequential_perception_pipeline(photos, max_edge=1024, use_owlvit=True, extract_materials=False)

    print(f"\n[Real Photos 640px]  {res_640['telemetry']}")
    print(f"[Real Photos 1024px] {res_1024['telemetry']}")

    assert res_640["telemetry"]["image_count"] == len(photos)
    assert res_1024["telemetry"]["image_count"] == len(photos)
    assert res_640["scene_graph"].observability.distinct_views == len(photos)

    # Persist real photographic scene graph to WorldIR + WorldStore
    world = WorldIR("real_photo_world")
    sg, _ = bridge_scene_to_world(world, res_640["scene_graph"].observations)
    assert "scene_intelligence" in world.metadata
    store = WorldStore(str(tmp_path / "ws"))
    sv = store.save_version(world, parent=None)
    reloaded = store.load_version(sv.version_id)
    assert reloaded.metadata["scene_intelligence"]["observation_count"] == len(sg.observations)
