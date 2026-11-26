"""A5 Tests (15, 16, 17, 18 + Experiments A/B/C): serialization, World handoff, persistence, and real-photo run."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perception.contracts.claude_handoff_v1 import HANDOFF_SCHEMA_VERSION, HandoffBundle
from perception.contracts.claude_integration_v2 import (
    ClaudeReconstructionFeedback,
    WorldEntityIdentity,
    parse_claude_camera_poses_from_json,
)
from perception.intelligence.a5_experiments import (
    run_experiment_a_south_building,
    run_experiment_b_indoor,
    run_experiment_c_multiview_indoor,
)
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    PerceptionBundle,
    ScoreSemantics,
)
from perception.intelligence.depth_aware import DepthMapContract
from perception.intelligence.production_pipeline_a5 import run_a5_production_pipeline
from perception.intelligence.resumable_runner import run_resumable_a5_job


def _obs(
    obs_id: str,
    image_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.88,
    category: str = "furniture",
    engine_id: str = "grounding_dino",
    width: int = 800,
    height: int = 600,
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=obs_id,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"re:{label}",
        category=category,
        bbox_px=bbox,
        coords=CoordinateSystemSpec("image_px", "pixels", width, height),
        scores=ScoreSemantics(raw_score=confidence, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance(engine_id, engine_id, f"{engine_id}_v1"),
        ),
    )


def _textured_room(width: int = 800, height: int = 600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(140, 210, size=(height, width, 3), dtype=np.uint8)
    img[60:480, 100:104] = 20
    img[60:480, 700:704] = 20
    img[480:484, 100:704] = 20
    # Distinct red chair on left, blue table on right
    img[280:460, 180:300] = [200, 40, 40]
    img[280:460, 180:300:3] = [120, 30, 30]
    img[290:470, 380:600] = [40, 80, 200]
    img[290:470, 380:600:3] = [30, 50, 140]
    return img


def test_15_16_17_serialization_claude_handoff_and_fresh_persistence_roundtrip(tmp_path: Path) -> None:
    img_a = _textured_room(seed=1)
    img_b = _textured_room(seed=2)
    mask_chair_a = np.zeros((600, 800), dtype=bool)
    mask_chair_a[285:455, 185:295] = True

    obs = [
        _obs("w1_a", "cam_1", "wall", (0, 20, 800, 480), category="architecture"),
        _obs("d1_a", "cam_1", "door", (110, 120, 200, 470), category="architecture"),
        _obs("ch_a", "cam_1", "chair", (180, 280, 300, 460)),
        _obs("ch_a_dup", "cam_1", "chair", (182, 282, 298, 458), confidence=0.72, engine_id="owlv2"),
        _obs("tb_a", "cam_1", "table", (380, 290, 600, 470)),
        _obs("ch_b", "cam_2", "chair", (184, 282, 304, 462)),
        _obs("w1_b", "cam_2", "wall", (0, 22, 800, 482), category="architecture"),
    ]
    cameras = parse_claude_camera_poses_from_json([
        {"image_id": "cam_1", "position_m": [0.0, 0.0, 0.0], "position_sigma_m": 0.02, "is_metric": True},
        {"image_id": "cam_2", "position_m": [0.15, 0.0, 0.0], "position_sigma_m": 0.02, "is_metric": True},
    ])
    feedback = ClaudeReconstructionFeedback(
        cameras=cameras,
        world_entities=(
            WorldEntityIdentity("world_chair_1", "chair", ("ch_a", "ch_b"), (1.2, 0.5, 0.0), True),
        ),
    )
    depth_arr = np.full((600, 800), 3.0, dtype=np.float32)
    depth_arr[280:460, 180:300] = 1.4
    depth_c = DepthMapContract("cam_1", depth_arr, "metric", "meters", "metric3d_v2")

    res = run_a5_production_pipeline(
        "a5_persist_test",
        images_rgb={"cam_1": img_a, "cam_2": img_b},
        observations=obs,
        precomputed_masks={"ch_a": mask_chair_a},
        run_live_detector=False,
        run_live_segmenter=True,
        claude_feedback=feedback,
        depth_contracts=[depth_c],
        depth_refs_by_image={"cam_1": "depth/cam_1.npy"},
        work_dir=tmp_path / "run_ws",
        persist_to_world=True,
    )
    # Duplicate chair suppressed before persistence
    assert res.dedup_result.summary["suppressed_count"] == 1
    assert res.segmentation_report.masks_produced == 1

    # Observation serialization roundtrip
    import json

    reloaded_bundle = PerceptionBundle.from_dict(json.loads(res.bundle.to_json()))
    assert len(reloaded_bundle.observations) == len(res.bundle.observations)

    # Handoff bundle roundtrip
    assert res.handoff.schema_version == HANDOFF_SCHEMA_VERSION
    reloaded_handoff = HandoffBundle.from_json(res.handoff.to_json())
    assert len(reloaded_handoff.observations) == len(res.bundle.observations)

    # Fresh-process EvidenceMatrix persistence verified
    assert res.claude_verification is not None
    assert res.claude_verification.evidence_matrix_persisted is True
    assert res.claude_verification.fresh_reload_observation_count == len(res.bundle.observations)
    assert res.claude_verification.world_entities_linked == 1

    # Resumable job runner reloads from cache on second invocation
    job_dir = tmp_path / "resumable_ws"
    st1, _, b1 = run_resumable_a5_job(
        "resumable_01",
        job_dir,
        observations=obs,
        run_live_detector=False,
        run_live_segmenter=False,
        claude_feedback=feedback,
    )
    assert st1.status == "completed"
    st2, res2, b2 = run_resumable_a5_job(
        "resumable_01",
        job_dir,
        observations=obs,
        run_live_detector=False,
        run_live_segmenter=False,
        claude_feedback=feedback,
    )
    assert st2.status == "resumed_from_cache"
    assert res2 is None
    assert len(b2.observations) == len(b1.observations)


def test_experiment_b_and_c_indoor_honesty_and_multiview_scene(tmp_path: Path) -> None:
    # Experiment B with no local indoor files reports explicit blocker
    exp_b = run_experiment_b_indoor([tmp_path / "missing_indoor_room.jpg"])
    assert exp_b["status"] == "blocked_no_local_indoor_photos"
    assert exp_b["accuracy_claim_allowed"] is False

    # Experiment C multi-view indoor scene
    img_1 = _textured_room(seed=10)
    img_2 = _textured_room(seed=11)
    m_ch1 = np.zeros((600, 800), dtype=bool)
    m_ch1[285:455, 185:295] = True
    m_ch2 = np.zeros((600, 800), dtype=bool)
    m_ch2[287:457, 189:299] = True
    obs = [
        _obs("w1", "v1", "wall", (0, 20, 800, 480), category="architecture"),
        _obs("f1", "v1", "floor", (0, 470, 800, 600), category="architecture"),
        _obs("ch1", "v1", "chair", (180, 280, 300, 460)),
        _obs("w2", "v2", "wall", (0, 22, 800, 482), category="architecture"),
        _obs("f2", "v2", "floor", (0, 472, 800, 600), category="architecture"),
        _obs("ch2", "v2", "chair", (184, 282, 304, 462)),
    ]
    cameras = parse_claude_camera_poses_from_json([
        {"image_id": "v1", "position_m": [0.0, 0.0, 0.0], "position_sigma_m": 0.03},
        {"image_id": "v2", "position_m": [0.12, 0.0, 0.0], "position_sigma_m": 0.03},
    ])
    exp_c = run_experiment_c_multiview_indoor(
        obs,
        images_rgb={"v1": img_1, "v2": img_2},
        precomputed_masks={"ch1": m_ch1, "ch2": m_ch2},
        claude_feedback=ClaudeReconstructionFeedback(cameras=cameras),
        work_dir=tmp_path / "exp_c",
    )
    assert exp_c["summary"]["masks_produced"] == 2
    assert exp_c["summary"]["evidence_matrix_persisted"] is True
    assert any(p["outcome"] == "confirmed_match" for p in exp_c["pairwise_outcomes"])


def test_18_real_photo_experiment_a_south_building_pipeline(tmp_path: Path) -> None:
    imgs = [
        Path("datasets/south_building/images/P1180141.JPG"),
        Path("datasets/south_building/images/P1180145.JPG"),
    ]
    if not all(p.exists() for p in imgs):
        pytest.skip("south_building images not present")

    out = run_experiment_a_south_building(imgs, work_dir=tmp_path / "exp_a")
    s = out["summary"]
    assert out["accuracy_claim_allowed"] is False
    assert s["images_ingested"] == 2
    assert s["raw_observations"] >= 1
    assert s["segmentation_status"] in ("succeeded", "partial")
    assert s["masks_produced"] >= 1
    assert s["evidence_matrix_persisted"] is True
