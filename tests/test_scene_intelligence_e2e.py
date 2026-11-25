"""End-to-end Scene Intelligence & RTX 4050 Sequential GPU Benchmark suite."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from perception.availability import real_models_by_name
from perception.scene_intelligence import (
    RelationshipSupport,
    RelationshipType,
    assess_scene_observability,
    bridge_scene_to_world,
    build_visual_scene_graph,
    classify_room_hypotheses,
    extract_appearance_and_materials,
    extract_candidate_relationships,
    resolve_hierarchical_label,
    run_sequential_perception_pipeline,
)
from perception.world_bridge import (
    assess_image_quality,
    build_visual_observations_from_detections,
    extract_wall_slab_from_points,
)
from provenance import Provenance
from world_ir.observations import DepthSource, ledger_of
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


def test_hierarchical_object_classification_and_unknown_guard():
    lbl, path = resolve_hierarchical_label("dining chair", 0.85)
    assert lbl == "dining chair"
    assert path == ("Furniture", "Seating", "Chair", "Dining chair")

    unc_lbl, unc_path = resolve_hierarchical_label("sofa", 0.10, min_confidence=0.25)
    assert unc_lbl == "unknown"
    assert unc_path == ("Unknown",)


def test_room_classification_requires_multiple_cues_and_preserves_ambiguity():
    # Single isolated object -> UNKNOWN
    single = build_visual_observations_from_detections(
        source_evidence_id="ev1",
        source_image_id="img1",
        model_id="m",
        model_version="1",
        detections=[{"label": "sofa", "bbox_px": [10, 10, 100, 100], "score": 0.9}],
    )
    hyps_single = classify_room_hypotheses(single)
    assert hyps_single[0].room_type == "unknown"
    assert hyps_single[0].status == "unknown"

    # Multi-cue living room
    living = build_visual_observations_from_detections(
        source_evidence_id="ev2",
        source_image_id="img2",
        model_id="m",
        model_version="1",
        detections=[
            {"label": "sofa", "bbox_px": [10, 10, 100, 100], "score": 0.9},
            {"label": "tv", "bbox_px": [200, 10, 300, 100], "score": 0.88},
            {"label": "armchair", "bbox_px": [120, 50, 180, 140], "score": 0.82},
        ],
    )
    hyps_living = classify_room_hypotheses(living)
    assert hyps_living[0].room_type == "living_room"
    assert hyps_living[0].status == "plausible"


def test_image_space_vs_world_space_relationships_and_observability():
    obs = build_visual_observations_from_detections(
        source_evidence_id="ev1",
        source_image_id="img1",
        model_id="m",
        model_version="1",
        depth_source=DepthSource.METRIC,
        detections=[
            {"label": "floor", "bbox_px": [0, 300, 640, 480], "score": 0.9, "world_pos": (1.0, 1.0, 0.0)},
            {"label": "chair", "bbox_px": [100, 200, 240, 380], "score": 0.85, "world_pos": (1.0, 1.0, 0.45)},
            {"label": "table", "bbox_px": [160, 200, 340, 380], "score": 0.88, "world_pos": (1.6, 1.0, 0.45)},
        ],
    )
    rels = extract_candidate_relationships(obs)
    assert any(r.rel_type is RelationshipType.OVERLAPPING and r.coordinate_frame == "image_space" for r in rels)
    assert any(r.rel_type is RelationshipType.SUPPORTED_BY and r.support_state is RelationshipSupport.GEOMETRICALLY_SUPPORTED for r in rels)
    assert any(r.rel_type is RelationshipType.ADJACENT_TO and r.coordinate_frame == "world_metric" for r in rels)

    report = assess_scene_observability(obs, rels)
    assert report.distinct_views == 1
    assert any("additional viewpoints" in s.action.lower() for s in report.suggestions)


def test_material_appearance_and_scene_graph_worldstore_roundtrip(tmp_path):
    img = np.full((240, 320, 3), (140, 100, 70), dtype=np.uint8)
    mat = extract_appearance_and_materials(img, (20, 20, 120, 120), "furniture")
    assert mat["material_verified"] is False
    assert mat["mean_color_rgb"] is not None

    world = WorldIR("sg_world")
    wall = Entity(
        id="wall_1",
        type=EntityType.WALL,
        name="Wall 1",
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.9,
        semantic_labels=["architecture/wall"],
    )
    wg = Geometry(
        id="wall_1_geom",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, -0.05, 0.0),
        bounds_max=Vector3(4.0, 0.05, 2.8),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.9,
    )
    wall.geometry_ids = [wg.id]
    world.entities = {wall.id: wall}
    world.geometries = {wg.id: wg}

    slab = extract_wall_slab_from_points(
        np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 2.5]]),
        source_evidence="ev_1",
        source_camera="cam_1",
        depth_basis="metric",
    )
    obs = build_visual_observations_from_detections(
        source_evidence_id="ev_1",
        source_image_id="vid_1#0",
        source_video_id="vid_1",
        frame_index=0,
        timestamp_s=0.0,
        model_id="maskrcnn",
        model_version="1.0",
        depth_source=DepthSource.METRIC,
        detections=[
            {
                "label": "wall",
                "bbox_px": [0, 0, 300, 220],
                "score": 0.9,
                "world_pos": (2.0, 0.0, 1.4),
                "wall_slab": slab,
                "entity_hint_id": "wall_1",
            },
            {"label": "desk", "bbox_px": [40, 100, 160, 210], "score": 0.85, "world_pos": (1.5, 0.8, 0.4)},
            {"label": "chair", "bbox_px": [90, 120, 150, 220], "score": 0.82, "world_pos": (1.5, 1.3, 0.4)},
        ],
    )
    sg, bridge_rep = bridge_scene_to_world(world, obs)
    assert sg.room_hypotheses[0].room_type == "office"
    assert "scene_intelligence" in world.metadata
    assert bridge_rep.wall_slabs_populated == 1

    store = WorldStore(str(tmp_path / "ws"))
    sv = store.save_version(world, parent=None)
    reloaded = store.load_version(sv.version_id)
    assert "scene_intelligence" in reloaded.metadata
    r_obs, _ = ledger_of(reloaded)
    assert len(r_obs) == 3


def test_rtx4050_sequential_pipeline_telemetry(tmp_path):
    avail = real_models_by_name()
    if avail["depth_anything"].status != "available" or avail["mask_rcnn"].status != "available":
        pytest.skip("Real cached models unavailable")

    paths = []
    for i in range(2):
        p = tmp_path / f"view_{i}.png"
        img = np.full((480, 640, 3), 160 + i * 15, dtype=np.uint8)
        cv2.rectangle(img, (50 + i * 20, 50), (580, 430), (70, 70, 70), 4)
        cv2.imwrite(str(p), img)
        paths.append(str(p))

    out = run_sequential_perception_pipeline(paths, score_threshold=0.3)
    tel = out["telemetry"]
    print(f"\n[RTX 4050 Sequential] {tel}")
    assert tel["total_time_s"] > 0.0
    assert out["scene_graph"].observability.distinct_views == 2
