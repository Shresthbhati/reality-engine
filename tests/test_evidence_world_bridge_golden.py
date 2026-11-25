"""End-to-end golden test and benchmark for perception/world_bridge.py."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from perception.availability import real_models_by_name
from perception.world_bridge import (
    assess_image_quality,
    bridge_observations_to_world,
    build_visual_observations_from_detections,
    extract_wall_slab_from_points,
)
from provenance import Provenance
from world_ir.observations import AssociationState, DepthSource, ledger_of
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.uncertainty_model import EpistemicState
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


def _make_test_room_image(path, width: int = 640, height: int = 480) -> np.ndarray:
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    # Draw wall boundary, door, and furniture-like contrast features for sharpness
    cv2.rectangle(img, (40, 40), (600, 440), (90, 90, 90), 4)
    cv2.rectangle(img, (260, 140), (380, 438), (60, 40, 20), -1)  # door-like rectangle
    cv2.rectangle(img, (80, 260), (220, 420), (40, 100, 140), -1)  # chair/table-like rectangle
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def test_wall_slab_refuses_relative_depth_and_fits_metric_points():
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.01, 0.0], [4.0, -0.01, 2.5], [1.0, 0.0, 1.2]])

    rel_slab = extract_wall_slab_from_points(
        pts, source_evidence="ev_1", source_camera="cam_1", depth_basis="relative"
    )
    assert rel_slab.start_xyz_m is None
    assert rel_slab.end_xyz_m is None

    metric_slab = extract_wall_slab_from_points(
        pts, source_evidence="ev_1", source_camera="cam_1", depth_basis="metric"
    )
    assert metric_slab.start_xyz_m is not None
    assert metric_slab.end_xyz_m is not None
    assert metric_slab.length_m == pytest.approx(4.0, abs=0.05)


def test_end_to_end_golden_bridge_to_worldstore_and_idempotent_reprocessing(tmp_path):
    img_path = tmp_path / "room.png"
    img_rgb = _make_test_room_image(img_path)
    quality = assess_image_quality(img_rgb)
    assert quality.width == 640 and quality.height == 480
    assert not quality.is_blurry

    # Build a synthetic WorldIR with a wall + chair entity waiting for observation linkage
    world = WorldIR("golden_world")
    wall = Entity(
        id="wall_north",
        type=EntityType.WALL,
        name="North Wall",
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.85,
        semantic_labels=["architecture/wall"],
    )
    wall_geom = Geometry(
        id="wall_north_geom",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, -0.05, 0.0),
        bounds_max=Vector3(4.0, 0.05, 2.8),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.85,
    )
    wall.geometry_ids = [wall_geom.id]

    chair = Entity(
        id="chair_1",
        type=EntityType.FURNITURE,
        name="Chair",
        provenance=Provenance.OBSERVED,
        confidence=0.8,
        semantic_labels=["furniture/chair"],
    )
    world.entities = {wall.id: wall, chair.id: chair}
    world.geometries = {wall_geom.id: wall_geom}

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [4.0, 0.0, 2.5]])
    wall_slab = extract_wall_slab_from_points(
        pts, source_evidence="ev_img_1", source_camera="cam_1", depth_basis="metric"
    )

    # Detections from two viewpoints (one conflicting label on chair_1 to verify conflict preservation)
    obs_v1 = build_visual_observations_from_detections(
        source_evidence_id="ev_img_1",
        source_image_id="img_1",
        camera_id="cam_1",
        model_id="maskrcnn_resnet50_fpn",
        model_version="coco_v1",
        depth_source=DepthSource.METRIC,
        quality=quality,
        detections=[
            {
                "label": "wall",
                "bbox_px": [40, 40, 600, 440],
                "score": 0.91,
                "world_pos": (2.0, 0.0, 1.4),
                "wall_slab": wall_slab,
                "entity_hint_id": "wall_north",
            },
            {
                "label": "chair",
                "bbox_px": [80, 260, 220, 420],
                "score": 0.87,
                "world_pos": (1.2, 1.5, 0.4),
                "entity_hint_id": "chair_1",
            },
        ],
    )
    obs_v2_video = build_visual_observations_from_detections(
        source_evidence_id="ev_vid_1",
        source_image_id="vid_1#frame-000010",
        source_video_id="vid_1",
        frame_index=10,
        timestamp_s=1.5,
        camera_id="cam_2",
        model_id="owlvit-base-patch32",
        model_version="1.0",
        depth_source=DepthSource.METRIC,
        detections=[
            {
                "label": "armchair",
                "bbox_px": [85, 255, 225, 415],
                "score": 0.68,
                "world_pos": (1.25, 1.52, 0.42),
                "entity_hint_id": "chair_1",
            }
        ],
    )

    all_obs = obs_v1 + obs_v2_video
    report = bridge_observations_to_world(world, all_obs)

    assert report.added_ledger_observations == 3
    assert report.skipped_duplicate_observations == 0
    assert report.entity_observations_attached == 3
    assert report.wall_slabs_populated == 1
    assert "wall_slab" in world.entities["wall_north"].custom_properties
    assert world.entities["wall_north"].custom_properties["wall_slab"]["start_xyz_m"] is not None
    assert world.entities["wall_north"].custom_properties["wall_slab"]["end_xyz_m"] is not None

    # Conflict between "chair" and "armchair" at 0.06m must be preserved as AMBIGUOUS
    _, ledger_decs = ledger_of(world)
    assert any(d.state is AssociationState.AMBIGUOUS for d in ledger_decs)
    assert report.uncertainty_profiles["chair_1"].existence.state is EpistemicState.CONFLICTING

    # Idempotent reprocessing: re-running same observations adds 0 new ledger entries
    report2 = bridge_observations_to_world(world, all_obs)
    assert report2.added_ledger_observations == 0
    assert report2.skipped_duplicate_observations == 3

    # WorldStore persistence & reload
    store = WorldStore(str(tmp_path / "ws"))
    sv = store.save_version(world, parent=None)
    reloaded = store.load_version(sv.version_id)
    r_obs, r_decs = ledger_of(reloaded)
    assert len(r_obs) == 3
    assert len(reloaded.entities["wall_north"].observations) == 1
    assert len(reloaded.entities["chair_1"].observations) == 2
    assert reloaded.entities["wall_north"].custom_properties["wall_slab"]["start_xyz_m"] is not None


def test_real_rtx4050_models_execute_when_available(tmp_path):
    import time
    import torch
    from evidence.session import EvidenceItem, EvidenceKind
    from perception.depth.depth_anything_backend import DepthAnythingV2Backend
    from perception.detection.maskrcnn_backend import MaskRCNNDetector

    models = real_models_by_name()
    if models["depth_anything"].status != "available" or models["mask_rcnn"].status != "available":
        pytest.skip("Real cached models unavailable")

    img_path = tmp_path / "real_test.png"
    _make_test_room_image(img_path)
    ev = EvidenceItem(
        id="ev_rtx_1",
        kind=EvidenceKind.PHOTO,
        source_uri=str(img_path),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    depth_backend = DepthAnythingV2Backend()
    dmaps = depth_backend.estimate_depth([ev])
    t_depth = time.perf_counter() - t0
    assert len(dmaps) == 1
    depth_arr = np.asarray(dmaps[0].values, dtype=np.float32)
    assert depth_arr.shape == (480, 640)

    t1 = time.perf_counter()
    detector = MaskRCNNDetector(score_threshold=0.3)
    det_results = detector.detect([ev])
    t_det = time.perf_counter() - t1
    assert len(det_results) == 1
    assert det_results[0].evidence_id == ev.id

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    print(
        f"\n[RTX 4050] cuda={torch.cuda.is_available()} "
        f"depth_s={t_depth:.3f} det_s={t_det:.3f} peak_vram_mb={peak_mb:.1f}"
    )

    # Bridge real model outputs into WorldIR
    obs = build_visual_observations_from_detections(
        source_evidence_id=ev.id,
        source_image_id=ev.id,
        model_id="maskrcnn_resnet50_fpn",
        model_version="coco_v1",
        depth_map=depth_arr,
        depth_source=DepthSource.RELATIVE,
        detections=[
            {"label": d.label, "bbox_px": list(d.box), "score": d.score}
            for d in det_results[0].detections
        ],
    )
    assert isinstance(obs, list)
